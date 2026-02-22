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

from stride.data.loader import load_pen_human, make_datasets, make_dataloaders
from stride.models.vae import ConditionalVAE
from stride.models.editor import LatentEditor
from stride.models.policy import BCPolicy
from stride.influence.trak import (
    compute_influence_scores_batched,
    compute_influence_scores,
)
from stride.influence.selection import (
    normalise_influence_scores,
    compute_corrective_directions,
    compute_preference_pairs,
)


# ---------------------------------------------------------------------------
# DPO preference dataset
# ---------------------------------------------------------------------------

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
        return (self.obs[idx], self.act[idx], self.win[idx],
                self.los[idx], self.valid[idx], self.dirs[idx])


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

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
    """Combined DPO + cosine alignment + δz regularisation loss.

    L = L_dpo + λ_cos · L_cosine + λ_reg · ‖δz‖²

    The DPO term pushes edited actions toward winner and away from loser.
    The cosine term maintains directional alignment (proven to work).
    The regularisation keeps edits conservative.

    Parameters
    ----------
    a_prime    : (B, act_dim) edited actions from editor
    a_orig     : (B, act_dim) original actions
    a_winner   : (B, act_dim) best-influence neighbour actions
    a_loser    : (B, act_dim) worst-influence neighbour actions
    valid_mask : (B,) 1.0 where preference pair is meaningful
    target_dir : (B, act_dim) corrective direction (for cosine term)
    delta_z    : (B, latent_dim) editor's latent residual
    beta       : DPO temperature (higher = sharper preference)
    lambda_reg : δz L2 weight
    lambda_cos : cosine alignment weight

    Returns
    -------
    loss       : scalar total loss
    info       : dict with component losses for logging
    """
    # ---- DPO preference term ----
    # r_w = -||a' - a_w||², r_l = -||a' - a_l||²
    # L_dpo = -log σ(β · (r_w - r_l))
    #       = -log σ(β · (||a' - a_l||² - ||a' - a_w||²))
    dist_to_winner = ((a_prime - a_winner) ** 2).sum(dim=-1)  # (B,)
    dist_to_loser = ((a_prime - a_loser) ** 2).sum(dim=-1)   # (B,)

    logit = beta * (dist_to_loser - dist_to_winner)
    dpo_per_sample = -F.logsigmoid(logit)  # (B,)

    # Only count valid preference pairs
    n_valid = valid_mask.sum().clamp(min=1.0)
    dpo_loss = (dpo_per_sample * valid_mask).sum() / n_valid

    # ---- Cosine alignment term (auxiliary) ----
    correction = a_prime - a_orig
    dir_norm = target_dir.norm(dim=-1, keepdim=True)
    dir_valid = (dir_norm.squeeze(-1) > eps)

    if dir_valid.sum() > 0:
        cos_sim = F.cosine_similarity(
            correction[dir_valid], target_dir[dir_valid], dim=-1)
        cos_loss = (1.0 - cos_sim).mean()
    else:
        cos_loss = torch.tensor(0.0, device=a_prime.device)

    # ---- δz regularisation ----
    reg = (delta_z ** 2).mean()

    total = dpo_loss + lambda_cos * cos_loss + lambda_reg * reg

    info = {
        "dpo": dpo_loss.item(),
        "cos": cos_loss.item(),
        "reg": reg.item(),
        "total": total.item(),
        "pref_acc": (logit > 0).float().mean().item(),  # accuracy of preference
    }
    return total, info


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train_editor_dpo(
    data: dict | None = None,
    vae: ConditionalVAE | None = None,
    bc_policy: BCPolicy | None = None,
    vae_ckpt: str = "checkpoints/vae.pt",
    bc_ckpt: str = "checkpoints/bc_policy.pt",
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 256,
    k_neighbors: int = 10,
    proj_dim: int = 512,
    beta: float = 1.0,
    lambda_reg: float = 0.3,
    lambda_cos: float = 0.3,
    device_str: str = "cpu",
    out_path: str = "checkpoints/editor.pt",
    seed: int = 42,
    verbose: bool = True,
) -> tuple[LatentEditor, np.ndarray]:
    """Train the latent editor g_ψ with DPO preference objective.

    Returns
    -------
    Trained LatentEditor (CPU), raw influence scores (N_train,).
    """
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu"
                          else "cpu")

    # ---- Load data -------------------------------------------------------
    if data is None:
        data = load_pen_human()

    train_ds, val_ds, train_idx, val_idx = make_datasets(data, seed=seed)
    train_loader, val_loader = make_dataloaders(train_ds, val_ds, batch_size=batch_size)

    obs_dim = data["observations"].shape[1]
    act_dim = data["actions"].shape[1]

    # ---- Load VAE --------------------------------------------------------
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

    # ---- Load BC policy (for influence computation) ----------------------
    if bc_policy is None:
        ckpt = torch.load(bc_ckpt, map_location="cpu")
        bc_policy = BCPolicy(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"])
        bc_policy.load_state_dict(ckpt["state_dict"])
    bc_policy = bc_policy.to(device)
    bc_policy.eval()

    # ---- Compute influence scores ----------------------------------------
    if verbose:
        print("[DPO-Editor] Computing TRAK influence scores …")
    try:
        influence_raw = compute_influence_scores_batched(
            bc_policy, train_loader, val_loader, proj_dim=proj_dim,
            seed=seed, device=device)
    except Exception:
        influence_raw = compute_influence_scores(
            bc_policy, train_loader, val_loader, proj_dim=proj_dim,
            seed=seed, device=device)

    # Corrected sign: negate so positive = helpful
    influence_corrected = -influence_raw

    # ---- Compute VAE latent means for KNN --------------------------------
    if verbose:
        print("[DPO-Editor] Computing VAE latents for KNN …")
    train_obs_np = data["observations"][train_idx]
    train_act_np = data["actions"][train_idx]

    with torch.no_grad():
        obs_t = torch.from_numpy(train_obs_np).float().to(device)
        act_t = torch.from_numpy(train_act_np).float().to(device)
        mu, _ = vae.encode(obs_t, act_t)
        latent_means = mu.cpu().numpy()

    # ---- Compute preference pairs ----------------------------------------
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

    # ---- Also compute corrective directions (auxiliary cosine loss) -------
    _, influence_weights = normalise_influence_scores(-influence_raw)
    directions = compute_corrective_directions(
        observations=train_obs_np,
        actions=train_act_np,
        influence_weights=influence_weights,
        embeddings=latent_means,
        k=k_neighbors,
    )

    # ---- Build editor dataset --------------------------------------------
    editor_ds = DPOEditorDataset(
        train_obs_np, train_act_np, winners, losers, valid, directions)
    editor_loader = DataLoader(editor_ds, batch_size=batch_size,
                                shuffle=True, drop_last=False)

    # ---- Initialise editor -----------------------------------------------
    editor = LatentEditor(obs_dim=obs_dim, act_dim=act_dim,
                           latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(editor.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        editor.train()
        t0 = time.time()
        epoch_info = {"dpo": 0, "cos": 0, "reg": 0, "total": 0, "pref_acc": 0}
        n = 0

        for obs_b, act_b, win_b, los_b, valid_b, dir_b in editor_loader:
            obs_b   = obs_b.to(device)
            act_b   = act_b.to(device)
            win_b   = win_b.to(device)
            los_b   = los_b.to(device)
            valid_b = valid_b.to(device)
            dir_b   = dir_b.to(device)

            a_prime, dz = editor.edit(obs_b, act_b, vae)

            loss, info = dpo_editor_loss(
                a_prime, act_b, win_b, los_b, valid_b, dir_b, dz,
                beta=beta, lambda_reg=lambda_reg, lambda_cos=lambda_cos,
            )

            optimizer.zero_grad()
            loss.backward()
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

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "latent_dim": latent_dim,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, out_path)
    if verbose:
        print(f"[DPO-Editor] Best loss {best_loss:.4f} → saved to {out_path}")

    editor.load_state_dict(best_state)
    return editor.to("cpu"), influence_raw
