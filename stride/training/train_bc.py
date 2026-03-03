"""Train the Behavior Cloning (BC) policy on pen-human-v2 demonstrations.

Usage
-----
    python -m stride.training.train_bc [--epochs N] [--lr LR] [--batch-size B]
                                        [--device cuda] [--out checkpoints/bc_policy.pt]
                                        [--smoke-test]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from stride.data.loader import load_pen_human, make_datasets, make_dataloaders
from stride.models.policy import BCPolicy, bc_loss
from stride.models.vae import ConditionalVAE


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: BCPolicy,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    weights: bool = False,
    obs_noise_std: float = 0.0,
    vae: ConditionalVAE | None = None,
    latent_stats: dict[str, torch.Tensor] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for obs_b, act_b, w_b in loader:
        obs_b, act_b = obs_b.to(device), act_b.to(device)

        # LSPO: Target is the VAE latent z instead of raw action a
        if vae is not None:
            with torch.no_grad():
                mu, _ = vae.encode(obs_b, act_b)
                if latent_stats is not None:
                    mu = (mu - latent_stats["mean"]) / (latent_stats["std"] + 1e-6)
                targets = mu
        else:
            targets = act_b

        # Observation noise augmentation for robustness
        if obs_noise_std > 0:
            obs_b = obs_b + torch.randn_like(obs_b) * obs_noise_std
        w = w_b.to(device) if weights else None
        pred = model(obs_b)
        loss = bc_loss(pred, targets, w)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * obs_b.shape[0]
        n += obs_b.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model: BCPolicy,
    loader: DataLoader,
    device: torch.device,
    vae: ConditionalVAE | None = None,
    latent_stats: dict[str, torch.Tensor] | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for obs_b, act_b, _ in loader:
        obs_b, act_b = obs_b.to(device), act_b.to(device)

        if vae is not None:
            mu, _ = vae.encode(obs_b, act_b)
            if latent_stats is not None:
                mu = (mu - latent_stats["mean"]) / (latent_stats["std"] + 1e-6)
            targets = mu
        else:
            targets = act_b

        pred = model(obs_b)
        loss = bc_loss(pred, targets)
        total_loss += loss.item() * obs_b.shape[0]
        n += obs_b.shape[0]
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_bc(
    data: dict | None = None,
    val_data: dict | None = None,
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 256,
    device_str: str = "cpu",
    out_path: str = "checkpoints/bc_policy.pt",
    use_weights: bool = False,
    influence_weights: np.ndarray | None = None,
    hidden: tuple[int, ...] = (256, 256),
    use_cosine_lr: bool = False,
    obs_noise_std: float = 0.0,
    use_lspo: bool = False,
    use_lspo_norm: bool = True,
    vae: ConditionalVAE | None = None,
    vae_ckpt: str | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> BCPolicy:
    """Train a BC policy and save checkpoint.

    Parameters
    ----------
    data              : dict from load_pen_human(); loaded if None
    epochs            : number of training epochs
    lr                : Adam learning rate
    batch_size        : mini-batch size
    device_str        : 'cpu' or 'cuda'
    out_path          : where to save the checkpoint
    use_weights       : whether to apply per-sample weights
    influence_weights : (N_train,) influence weights; only used if use_weights
    seed              : reproducibility seed
    use_lspo          : whether to use Latent-Space Policy Optimization
    vae               : provided VAE model for LSPO
    vae_ckpt          : path to VAE checkpoint for LSPO (if vae is None)
    verbose           : print epoch logs

    Returns
    -------
    Trained BCPolicy (moved to CPU).
    """
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu"
                          else "cpu")

    if data is None:
        data = load_pen_human()

    # Load VAE if LSPO is enabled but VAE is not provided
    if use_lspo and vae is None and vae_ckpt is not None:
        if verbose:
            print(f"[BC] LSPO enabled: loading VAE from {vae_ckpt}")
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        vae = ConditionalVAE(
            obs_dim=ckpt["obs_dim"],
            act_dim=ckpt["act_dim"],
            latent_dim=ckpt["latent_dim"]
        )
        vae.load_state_dict(ckpt["state_dict"])
        vae = vae.to(device).eval()

    # Identify training indices to compute normalization stats (avoid leakage)
    N = len(data["observations"])
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(N)[:int(N * 0.8)]
    
    # Calculate observation normalization stats from training data
    obs_train = data["observations"][train_idx]
    obs_norm = {
        "mean": obs_train.mean(axis=0),
        "std": obs_train.std(axis=0),
    }
    if verbose:
        print(f"[BC] Obs norm: mean range [{obs_norm['mean'].min():.2f}, {obs_norm['mean'].max():.2f}], "
              f"std range [{obs_norm['std'].min():.2f}, {obs_norm['std'].max():.2f}]")

    train_ds, val_ds, train_idx, _ = make_datasets(data, seed=seed,
                                                    weights=influence_weights
                                                    if use_weights else None,
                                                    obs_norm=obs_norm)

    # If separate validation data is provided (e.g. original unedited data),
    # use it for validation instead of the split from the training data.
    if val_data is not None:
        _, val_ds_orig, _, _ = make_datasets(val_data, seed=seed, obs_norm=obs_norm)
        val_ds = val_ds_orig

    train_loader, val_loader = make_dataloaders(train_ds, val_ds, batch_size=batch_size)

    obs_dim = data["observations"].shape[1]
    
    latent_stats = None
    if use_lspo and use_lspo_norm and vae is not None:
        target_dim = vae.latent_dim
        # Pre-compute latent statistics for normalization
        if verbose:
            print("[BC] LSPO: computing latent normalization stats...")
        with torch.no_grad():
            all_obs = torch.from_numpy(data["observations"][train_idx]).float().to(device)
            all_act = torch.from_numpy(data["actions"][train_idx]).float().to(device)
            
            # Process in batches to avoid OOM if dataset is large
            mus = []
            for i in range(0, len(all_obs), 1024):
                mu_b, _ = vae.encode(all_obs[i:i+1024], all_act[i:i+1024])
                mus.append(mu_b)
            all_mu = torch.cat(mus, dim=0)
            
            latent_stats = {
                "mean": all_mu.mean(dim=0, keepdim=True),
                "std": all_mu.std(dim=0, keepdim=True)
            }
            if verbose:
                print(f"  [LSPO] Latent mean range: [{latent_stats['mean'].min():.2f}, {latent_stats['mean'].max():.2f}]")
                print(f"  [LSPO] Latent std range:  [{latent_stats['std'].min():.2f}, {latent_stats['std'].max():.2f}]")
    else:
        target_dim = data["actions"].shape[1]

    model = BCPolicy(obs_dim=obs_dim, act_dim=target_dim, hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = None
    if use_cosine_lr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                  weights=use_weights,
                                  obs_noise_std=obs_noise_std,
                                  vae=vae,
                                  latent_stats=latent_stats)
        val_loss = eval_epoch(model, val_loader, device, vae=vae, latent_stats=latent_stats)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            cur_lr = scheduler.get_last_lr()[0] if scheduler else lr
            print(f"[BC] epoch {epoch:4d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={cur_lr:.1e}  "
                  f"({time.time()-t0:.1f}s)")

        if scheduler:
            scheduler.step()

    # Save best checkpoint
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "obs_dim": obs_dim,
        "act_dim": target_dim,
        "hidden": list(hidden),
        "best_val_loss": best_val,
        "use_lspo": use_lspo,
        "use_lspo_norm": use_lspo_norm,
        "vae_ckpt": vae_ckpt,
        "obs_mean": torch.from_numpy(obs_norm["mean"]),
        "obs_std": torch.from_numpy(obs_norm["std"]),
        "latent_mean": latent_stats["mean"].cpu() if latent_stats is not None else None,
        "latent_std": latent_stats["std"].cpu() if latent_stats is not None else None,
    }
    torch.save(checkpoint, out_path)
    if verbose:
        print(f"[BC] Best val loss {best_val:.4f} → saved to {out_path}")

    model.load_state_dict(best_state)
    return model.to("cpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train BC policy")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="checkpoints/bc_policy.pt")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run just 2 epochs for a quick sanity check")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    epochs = 2 if args.smoke_test else args.epochs
    data = load_pen_human()
    train_bc(
        data=data,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device_str=args.device,
        out_path=args.out,
    )
