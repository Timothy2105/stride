"""Train the latent-space editor g_ψ with a DPO preference objective.

Instead of cosine alignment with corrective directions, we use Direct
Preference Optimization (DPO).  For each sample i, we extract a "winner"
action a_w (highest-influence KNN neighbour) and a "loser" action a_l
(lowest-influence KNN neighbour).  The editor is trained so that its
output is preferred:

    L_DPO = -log σ(β · (r_w − r_l))

where r_w = -‖a' − a_w‖²  and  r_l = -‖a' − a_l‖²,  so the loss
pushes a' closer to a_w and farther from a_l.

An auxiliary δz regularisation term ‖δz‖² keeps edits small.

The DPO formulation is strictly unique to STRIDE: it requires
influence scores (for winner/loser labeling) AND the VAE latent
editor (for generating a').  No baseline can replicate this.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from stride.data import load_pen_human, make_datasets, make_dataloaders
from stride.models.vae import ConditionalVAE
from stride.models.editor import LatentEditor
from stride.influence import (
    normalise_influence_scores,
    compute_corrective_directions,
    compute_preference_pairs,
)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")






class DPOEditorDataset(torch.utils.data.Dataset):
    """(obs, act, a_winner, a_loser, valid, direction) tuples."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        winners: np.ndarray,
        losers: np.ndarray,
        valid: np.ndarray,
        directions: np.ndarray,
    ):
        self.obs = torch.from_numpy(observations).float()
        self.act = torch.from_numpy(actions).float()
        self.win = torch.from_numpy(winners).float()
        self.los = torch.from_numpy(losers).float()
        self.valid = torch.from_numpy(valid.astype(np.float32))
        self.dirs = torch.from_numpy(directions).float()

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.act[idx],
            self.win[idx],
            self.los[idx],
            self.valid[idx],
            self.dirs[idx],
        )






def dpo_editor_loss(
    a_prime: torch.Tensor,
    a_orig: torch.Tensor,
    a_winner: torch.Tensor,
    a_loser: torch.Tensor,
    valid_mask: torch.Tensor,
    target_dir: torch.Tensor,
    delta_z: torch.Tensor,
    beta: float = 1.0,
    lambda_reg: float = 0.3,
    lambda_cos: float = 0.3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Combined DPO + cosine alignment + δz regularisation loss."""
    
    
    dist_to_winner = ((a_prime - a_winner) ** 2).sum(dim=-1)  
    dist_to_loser = ((a_prime - a_loser) ** 2).sum(dim=-1)   
    logit = beta * (dist_to_loser - dist_to_winner)
    dpo_per_sample = -F.logsigmoid(logit)  
    n_valid = valid_mask.sum().clamp(min=1.0)
    dpo_loss = (dpo_per_sample * valid_mask).sum() / n_valid

    
    correction = a_prime - a_orig
    dir_norm = target_dir.norm(dim=-1, keepdim=True)
    dir_valid = (dir_norm.squeeze(-1) > eps)

    if dir_valid.sum() > 0:
        cos_sim = F.cosine_similarity(
            correction[dir_valid], target_dir[dir_valid], dim=-1)
        cos_loss = (1.0 - cos_sim).mean()
    else:
        cos_loss = torch.tensor(0.0, device=a_prime.device)

    
    reg = (delta_z ** 2).mean()

    total = dpo_loss + (lambda_cos * cos_loss) + (lambda_reg * reg)

    info = {
        "dpo": dpo_loss.item(),
        "cos": cos_loss.item(),
        "reg": reg.item(),
        "total": total.item(),
        "pref_acc": (logit > 0).float().mean().item(),
    }
    return total, info






def train_editor_dpo(
    data: dict | None = None,
    vae: ConditionalVAE | None = None,
    influence_scores_raw: np.ndarray | None = None,
    vae_ckpt: str = "checkpoints/vae.pt",
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 256,
    k_neighbors: int = 10,
    proj_dim: int = 512,
    beta: float = 1.0,
    lambda_reg: float = 0.3,
    lambda_cos: float = 0.3,
    hidden: tuple[int, ...] = (256, 256),
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    train_frac: float = 0.8,
    num_workers: int = 0,
    device_str: str = "cpu",
    out_path: str = "checkpoints/editor.pt",
    seed: int = 42,
    verbose: bool = True,
    wandb_run=None,
) -> tuple[LatentEditor, np.ndarray]:
    """Train the latent editor g_ψ with DPO preference objective.

    Returns
    -------
    Trained LatentEditor (CPU), raw influence scores (N_train,).
    """
    torch.manual_seed(seed)
    device = _resolve_device(device_str)

    
    if data is None:
        data = load_pen_human()

    train_ds, val_ds, train_idx, _ = make_datasets(
        data,
        train_frac=train_frac,
        seed=seed,
    )

    obs_dim = data["observations"].shape[1]
    act_dim = data["actions"].shape[1]

    
    obs_train = data["observations"][train_idx]
    obs_norm = {
        "mean": obs_train.mean(axis=0),
        "std": obs_train.std(axis=0),
    }
    if verbose:
        print(f"[DPO-Editor] Obs norm: mean range [{obs_norm['mean'].min():.2f}, {obs_norm['mean'].max():.2f}], "
              f"std range [{obs_norm['std'].min():.2f}, {obs_norm['std'].max():.2f}]")

    
    if vae is None:
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        vae = ConditionalVAE(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"],
                              latent_dim=ckpt["latent_dim"])
        vae.load_state_dict(ckpt["state_dict"])
    vae = vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    latent_dim = vae.latent_dim

    
    if influence_scores_raw is None:
        raise ValueError("train_editor_dpo requires externally provided influence_scores_raw.")

    influence_raw = np.asarray(influence_scores_raw, dtype=np.float32)
    
    if influence_raw.shape[0] == len(data["observations"]):
        
        influence_raw = influence_raw[train_idx]
    elif influence_raw.shape[0] != train_idx.shape[0]:
        raise ValueError(
            f"influence_scores_raw length {influence_raw.shape[0]} does not match "
            f"full dataset ({len(data['observations'])}) or train split ({train_idx.shape[0]})"
        )
    if verbose:
        print("[DPO-Editor] Using externally provided influence scores …")

    
    influence_corrected = -influence_raw

    
    if verbose:
        print("[DPO-Editor] Computing VAE latents for KNN …")
    train_obs_np = data["observations"][train_idx]
    train_act_np = data["actions"][train_idx]

    with torch.no_grad():
        obs_t = torch.from_numpy(train_obs_np).float().to(device)
        act_t = torch.from_numpy(train_act_np).float().to(device)
        mu, _ = vae.encode(obs_t, act_t)
        latent_means = mu.cpu().numpy()

    
    if verbose:
        print("[DPO-Editor] Computing preference pairs …")
    winners, losers, valid = compute_preference_pairs(
        actions=train_act_np,
        influence_scores_corrected=influence_corrected,
        embeddings=latent_means,
        k=k_neighbors,
    )
    
    if verbose:
        print(f"  Valid preference pairs: {valid.sum()}/{len(valid)} "
              f"({valid.mean() * 100:.1f}%)")

    
    _, influence_weights = normalise_influence_scores(-influence_raw)
    directions = compute_corrective_directions(
        observations=train_obs_np,
        actions=train_act_np,
        influence_weights=influence_weights,
        embeddings=latent_means,
        k=k_neighbors,
    )

    
    editor_ds = DPOEditorDataset(train_obs_np, train_act_np, winners, losers, valid, directions)
    editor_loader = DataLoader(editor_ds, batch_size=batch_size,
                                shuffle=True, drop_last=False)

    
    editor = LatentEditor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        hidden=hidden,
    ).to(device)
    optimizer = optim.Adam(editor.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        editor.train()
        t0 = time.time()
        epoch_info = {"dpo": 0, "cos": 0, "reg": 0, "total": 0, "pref_acc": 0}
        n = 0

        for batch in editor_loader:
            obs_b, act_b, win_b, los_b, valid_b, dir_b = [x.to(device) for x in batch]

            a_prime, dz = editor.edit(obs_b, act_b, vae)

            loss, info = dpo_editor_loss(
                a_prime, act_b, win_b, los_b, valid_b, dir_b, dz,
                beta=beta, lambda_reg=lambda_reg, lambda_cos=lambda_cos,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(editor.parameters(), max_norm=grad_clip)
            optimizer.step()

            bs = obs_b.shape[0]
            for key in epoch_info:
                epoch_info[key] += info[key] * bs
            n += bs

        for key in epoch_info:
            epoch_info[key] /= max(n, 1)

        if epoch_info["total"] < best_loss:
            best_loss = epoch_info["total"]
            best_state = {k: v.cpu().clone() for k, v in editor.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"[DPO-Editor] epoch {epoch:4d}/{epochs}  "
                  f"total={epoch_info['total']:.4f}  "
                  f"dpo={epoch_info['dpo']:.4f}  "
                  f"cos={epoch_info['cos']:.4f}  "
                  f"reg={epoch_info['reg']:.4f}  "
                  f"pref_acc={epoch_info['pref_acc']:.1%}  "
                  f"({time.time()-t0:.1f}s)")

        
        if wandb_run is not None:
            wandb_run.log({
                "editor/total_loss": epoch_info["total"],
                "editor/dpo_loss": epoch_info["dpo"],
                "editor/cos_loss": epoch_info["cos"],
                "editor/reg_loss": epoch_info["reg"],
                "editor/pref_acc": epoch_info["pref_acc"],
                "editor/epoch": epoch,
            })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "latent_dim": latent_dim,
        "obs_mean": torch.from_numpy(obs_norm["mean"]),
        "obs_std": torch.from_numpy(obs_norm["std"]),
        "best_loss": best_loss,
    }
    torch.save(checkpoint, out_path)
    if verbose:
        print(f"[DPO-Editor] Best loss {best_loss:.4f} → saved to {out_path}")

    editor.load_state_dict(best_state)
    return editor.to("cpu"), influence_raw
