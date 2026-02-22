"""Train the Conditional VAE on pen-human-v2 demonstrations.

The VAE encodes (s, a) into a latent space z and reconstructs a from (z, s).
Training objective: L_VAE = ‖a − â‖² + β·KL(q(z|s,a) ‖ N(0,I))

β is linearly annealed from 0 → target_beta over the first anneal_epochs epochs
(warmup prevents posterior collapse early in training).

Usage
-----
    python -m stride.training.train_vae [--epochs N] [--lr LR] [--beta B]
                                         [--device cuda] [--out checkpoints/vae.pt]
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from stride.data.loader import load_pen_human, make_datasets, make_dataloaders
from stride.models.vae import ConditionalVAE


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _beta_schedule(epoch: int, total_epochs: int, target_beta: float,
                    anneal_epochs: int) -> float:
    """Linear β annealing from 0 to target_beta over anneal_epochs."""
    if anneal_epochs <= 0:
        return target_beta
    frac = min(epoch / anneal_epochs, 1.0)
    return target_beta * frac


def train_epoch(
    vae: ConditionalVAE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    beta: float,
) -> tuple[float, float, float]:
    vae.train()
    total, recon_acc, kl_acc = 0.0, 0.0, 0.0
    n = 0
    for obs_b, act_b, _ in loader:
        obs_b, act_b = obs_b.to(device), act_b.to(device)
        loss, recon, kl = vae.loss(obs_b, act_b, beta=beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        B = obs_b.shape[0]
        total    += loss.item()  * B
        recon_acc += recon.item() * B
        kl_acc    += kl.item()   * B
        n += B
    return total / max(n, 1), recon_acc / max(n, 1), kl_acc / max(n, 1)


@torch.no_grad()
def eval_epoch(
    vae: ConditionalVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> tuple[float, float, float]:
    vae.eval()
    total, recon_acc, kl_acc = 0.0, 0.0, 0.0
    n = 0
    for obs_b, act_b, _ in loader:
        obs_b, act_b = obs_b.to(device), act_b.to(device)
        loss, recon, kl = vae.loss(obs_b, act_b, beta=beta)
        B = obs_b.shape[0]
        total    += loss.item()  * B
        recon_acc += recon.item() * B
        kl_acc    += kl.item()   * B
        n += B
    return total / max(n, 1), recon_acc / max(n, 1), kl_acc / max(n, 1)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_vae(
    data: dict | None = None,
    epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 256,
    latent_dim: int = 16,
    target_beta: float = 0.5,
    anneal_epochs: int = 50,
    device_str: str = "cpu",
    out_path: str = "checkpoints/vae.pt",
    seed: int = 42,
    verbose: bool = True,
) -> ConditionalVAE:
    """Train conditional VAE and save checkpoint.

    Returns
    -------
    Trained ConditionalVAE (moved to CPU).
    """
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu"
                          else "cpu")

    if data is None:
        data = load_pen_human()

    train_ds, val_ds, _, _ = make_datasets(data, seed=seed)
    train_loader, val_loader = make_dataloaders(train_ds, val_ds, batch_size=batch_size)

    obs_dim = data["observations"].shape[1]
    act_dim = data["actions"].shape[1]
    vae = ConditionalVAE(obs_dim=obs_dim, act_dim=act_dim,
                          latent_dim=latent_dim, beta=target_beta).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    best_val_recon = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        beta = _beta_schedule(epoch, epochs, target_beta, anneal_epochs)
        t0 = time.time()
        tr_tot, tr_re, tr_kl = train_epoch(vae, train_loader, optimizer, device, beta)
        va_tot, va_re, va_kl = eval_epoch(vae, val_loader, device, beta)

        if va_re < best_val_recon:
            best_val_recon = va_re
            best_state = {k: v.cpu().clone() for k, v in vae.state_dict().items()}

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"[VAE] epoch {epoch:4d}/{epochs}  β={beta:.3f}  "
                  f"train: total={tr_tot:.4f} recon={tr_re:.4f} kl={tr_kl:.4f}  "
                  f"val: recon={va_re:.4f} kl={va_kl:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "latent_dim": latent_dim,
        "best_val_recon": best_val_recon,
    }
    torch.save(checkpoint, out_path)
    if verbose:
        print(f"[VAE] Best val recon {best_val_recon:.4f} → saved to {out_path}")

    vae.load_state_dict(best_state)
    return vae.to("cpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train Conditional VAE")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--anneal-epochs", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="checkpoints/vae.pt")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    epochs = 2 if args.smoke_test else args.epochs
    data = load_pen_human()
    train_vae(
        data=data,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        target_beta=args.beta,
        anneal_epochs=args.anneal_epochs,
        device_str=args.device,
        out_path=args.out,
    )
