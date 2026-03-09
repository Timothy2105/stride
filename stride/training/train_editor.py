"""Train the latent-space residual editor g_ψ.

The editor takes (s, a, ξ) and outputs δz such that decoding z + δz yields
an action that points toward the corrective direction Δa_i^target.

Editor loss:
    L_edit = 1 − cosine_similarity(a' − a, Δa^target)

where a' = D_φ(z + δz, s)  and  Δa^target is the pre-computed normalised
corrective direction from the influence-guided KNN analysis.

The VAE is frozen during editor training.

Usage
-----
    python -m stride.training.train_editor [--epochs N] [--lr LR]
                                            [--vae-ckpt checkpoints/vae.pt]
                                            [--bc-ckpt  checkpoints/bc_policy.pt]
                                            [--out      checkpoints/editor.pt]
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from stride.data.loader import load_pen_human, make_datasets, make_dataloaders
from stride.models.vae import ConditionalVAE
from stride.models.editor import LatentEditor
from stride.models.policy import BCPolicy
from stride.influence.trak import compute_influence_scores_batched, compute_influence_scores
from stride.influence.selection import normalise_influence_scores, compute_corrective_directions


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Editor dataset:  (obs, act, delta_a_target) triples
# ---------------------------------------------------------------------------

class EditorDataset(torch.utils.data.Dataset):
    """Pairs (obs, act, direction) for editor training."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        directions: np.ndarray,
    ):
        self.obs = torch.from_numpy(observations).float()
        self.act = torch.from_numpy(actions).float()
        self.dirs = torch.from_numpy(directions).float()

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.dirs[idx]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def editor_loss(
    a_prime: torch.Tensor,
    a_orig: torch.Tensor,
    target_dir: torch.Tensor,
    delta_z: torch.Tensor | None = None,
    lambda_reg: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine alignment loss between (a' − a) and Δa^target, with δz regularisation.

    Only penalises samples where ‖Δa^target‖ > eps (non-zero corrective dir).

    L_edit = (1 − cos_sim(a' − a, Δa^target)) + λ · ‖δz‖²
    """
    correction = a_prime - a_orig                          # (B, act_dim)
    target_norm = target_dir.norm(dim=-1, keepdim=True)    # (B, 1)
    valid_mask = (target_norm.squeeze(-1) > eps)           # (B,)

    if valid_mask.sum() == 0:
        cos_part = torch.tensor(0.0, device=a_prime.device, requires_grad=True)
    else:
        corr_v  = correction[valid_mask]
        tgt_v   = target_dir[valid_mask]
        # cosine similarity per sample → (B_valid,)
        cos_sim = F.cosine_similarity(corr_v, tgt_v, dim=-1)
        cos_part = (1.0 - cos_sim).mean()

    # L2 regularisation on latent residual magnitude to prevent large perturbations
    if delta_z is not None and lambda_reg > 0:
        reg = lambda_reg * (delta_z ** 2).mean()
        return cos_part + reg

    return cos_part


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train_editor(
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
    device_str: str = "cpu",
    out_path: str = "checkpoints/editor.pt",
    seed: int = 42,
    verbose: bool = True,
) -> tuple[LatentEditor, np.ndarray]:
    """Train the latent editor g_ψ guided by influence-corrective directions.

    Returns
    -------
    Trained LatentEditor (CPU), raw influence scores (N_train,).
    """
    torch.manual_seed(seed)
    device = _resolve_device(device_str)

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
        print("[Editor] Computing TRAK influence scores …")
    try:
        influence_raw = compute_influence_scores_batched(
            bc_policy, train_loader, val_loader, proj_dim=proj_dim,
            seed=seed, device=device)
    except Exception:
        influence_raw = compute_influence_scores(
            bc_policy, train_loader, val_loader, proj_dim=proj_dim,
            seed=seed, device=device)

    # SIGN CONVENTION FIX: our TRAK computes score = -(G_val·G_i),
    # so positive = harmful, negative = helpful.  For corrective directions
    # we want max(0, Ĩ) to select HELPFUL neighbours → negate before normalising.
    _, influence_weights = normalise_influence_scores(-influence_raw)

    # ---- Compute VAE latent means for KNN --------------------------------
    if verbose:
        print("[Editor] Computing VAE latents for KNN …")
    train_obs_np = data["observations"][train_idx]
    train_act_np = data["actions"][train_idx]

    with torch.no_grad():
        obs_t = torch.from_numpy(train_obs_np).float().to(device)
        act_t = torch.from_numpy(train_act_np).float().to(device)
        mu, _ = vae.encode(obs_t, act_t)
        latent_means = mu.cpu().numpy()

    # ---- Compute corrective directions -----------------------------------
    if verbose:
        print("[Editor] Computing corrective directions …")
    directions = compute_corrective_directions(
        observations=train_obs_np,
        actions=train_act_np,
        influence_weights=influence_weights,
        embeddings=latent_means,
        k=k_neighbors,
    )

    # ---- Build editor dataset --------------------------------------------
    editor_ds = EditorDataset(train_obs_np, train_act_np, directions)
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
        epoch_loss = 0.0
        n = 0

        for obs_b, act_b, dir_b in editor_loader:
            obs_b  = obs_b.to(device)
            act_b  = act_b.to(device)
            dir_b  = dir_b.to(device)

            a_prime, dz = editor.edit(obs_b, act_b, vae)
            loss = editor_loss(a_prime, act_b, dir_b, delta_z=dz, lambda_reg=0.3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * obs_b.shape[0]
            n += obs_b.shape[0]

        epoch_loss /= max(n, 1)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in editor.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"[Editor] epoch {epoch:4d}/{epochs}  "
                  f"loss={epoch_loss:.4f}  ({time.time()-t0:.1f}s)")

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
        print(f"[Editor] Best loss {best_loss:.4f} → saved to {out_path}")

    editor.load_state_dict(best_state)
    return editor.to("cpu"), influence_raw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train latent editor")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--vae-ckpt", default="checkpoints/vae.pt")
    p.add_argument("--bc-ckpt", default="checkpoints/bc_policy.pt")
    p.add_argument("--out", default="checkpoints/editor.pt")
    p.add_argument("--proj-dim", type=int, default=512)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    epochs = 2 if args.smoke_test else args.epochs
    data = load_pen_human()
    train_editor(
        data=data,
        vae_ckpt=args.vae_ckpt,
        bc_ckpt=args.bc_ckpt,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        k_neighbors=args.k,
        proj_dim=args.proj_dim,
        device_str=args.device,
        out_path=args.out,
    )
