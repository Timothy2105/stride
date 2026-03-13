"""Train MLP Behavioral Cloning policy on Adroit demonstrations.

Training objective
------------------
L_BC = (1/N) Σ_i  w_i · ‖π_θ(s_i) − a_i‖²

where w_i are optional per-sample importance weights (default 1).

Features
--------
- Cosine-annealing learning rate schedule
- Gradient clipping
- Validation-based best-checkpoint selection
- Optional wandb logging of all training metrics
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from stride.data import DemoDataset, make_datasets
from stride.models.policy import MLPPolicy

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")






def _train_epoch(
    policy: MLPPolicy,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    policy.train()
    loss_sum, n = 0.0, 0
    for obs_b, act_b, w_b in loader:
        obs_b, act_b, w_b = obs_b.to(device), act_b.to(device), w_b.to(device)
        pred = policy(obs_b)
        per_sample = ((pred - act_b) ** 2).mean(dim=-1)          
        loss = (per_sample * w_b).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
        optimizer.step()

        B = obs_b.shape[0]
        loss_sum += loss.item() * B
        n += B
    return loss_sum / max(n, 1)


@torch.no_grad()
def _val_epoch(
    policy: MLPPolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    policy.eval()
    loss_sum, n = 0.0, 0
    for obs_b, act_b, _ in loader:
        obs_b, act_b = obs_b.to(device), act_b.to(device)
        pred = policy(obs_b)
        loss = F.mse_loss(pred, act_b)
        B = obs_b.shape[0]
        loss_sum += loss.item() * B
        n += B
    return loss_sum / max(n, 1)






def train_bc(
    data: dict,
    *,
    epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 256,
    hidden: tuple[int, ...] = (256, 256),
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    train_frac: float = 0.8,
    device_str: str = "cpu",
    out_dir: str = "results/checkpoints",
    run_name: str = "bc",
    seed: int = 42,
    weights: np.ndarray | None = None,
    wandb_run=None,
    verbose: bool = True,
    eval_callback=None,
    eval_every: int = 50,
) -> tuple[MLPPolicy, dict]:
    """Train MLP BC policy on demonstration data.

    Parameters
    ----------
    data       : dict with 'observations', 'actions', optionally 'episode_ends'.
    epochs     : number of training epochs.
    weights    : (N,) optional per-sample importance weights for the loss.
    wandb_run  : active wandb run for logging (or None to skip).
    run_name   : prefix used for logging keys and checkpoint filename.

    Returns
    -------
    policy  : trained MLPPolicy (on CPU, best val checkpoint).
    info    : dict with training history, checkpoint path, and obs norm stats.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _resolve_device(device_str)

    
    train_ds, val_ds, train_idx, val_idx = make_datasets(
        data, train_frac=train_frac, seed=seed, weights=weights,
    )

    obs_train = data["observations"][train_idx]
    obs_norm = {
        "mean": obs_train.mean(axis=0).astype(np.float32),
        "std": obs_train.std(axis=0).astype(np.float32),
    }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    obs_dim = data["observations"].shape[1]
    act_dim = data["actions"].shape[1]

    
    policy = MLPPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden).to(device)
    policy.set_obs_norm(obs_norm["mean"], obs_norm["std"])

    optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    info: dict = {
        "train_losses": [],
        "val_losses": [],
        "best_epoch": 0,
        "obs_norm": obs_norm,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_epoch(policy, train_loader, optimizer, device, grad_clip)
        val_loss = _val_epoch(policy, val_loader, device)
        scheduler.step()

        info["train_losses"].append(train_loss)
        info["val_losses"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
            info["best_epoch"] = epoch

        
        log_dict = {
            f"{run_name}/train_loss": train_loss,
            f"{run_name}/val_loss": val_loss,
            f"{run_name}/lr": scheduler.get_last_lr()[0],
            f"{run_name}/epoch": epoch,
        }
        if wandb_run is not None:
            wandb_run.log(log_dict)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            dt = time.time() - t0
            logger.info(
                f"[{run_name}] epoch {epoch:4d}/{epochs}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s)"
            )

        
        if eval_callback is not None and (epoch % eval_every == 0 or epoch == epochs):
            eval_callback(policy, epoch)

    
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{run_name}_policy.pt")
    checkpoint = {
        "state_dict": best_state,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden": list(hidden),
        "obs_mean": torch.from_numpy(obs_norm["mean"]),
        "obs_std": torch.from_numpy(obs_norm["std"]),
        "best_val_loss": best_val_loss,
        "best_epoch": info["best_epoch"],
    }
    torch.save(checkpoint, ckpt_path)
    info["best_val_loss"] = best_val_loss
    info["ckpt_path"] = ckpt_path

    policy.load_state_dict(best_state)
    if verbose:
        logger.info(
            f"[{run_name}] Best val loss {best_val_loss:.5f} "
            f"(epoch {info['best_epoch']}) → {ckpt_path}"
        )

    return policy.cpu(), info
