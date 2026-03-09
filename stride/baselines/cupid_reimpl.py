"""Baseline: CUPID-style reimplementation (influence-based data curation + BC).

This baseline approximates Cupid's core idea for low-dim BC:
1. Train (or load) a base BC policy.
2. Compute per-sample TRAK influence on a validation split.
3. Keep high-influence samples and drop low-influence samples.
4. Train BC on the curated subset.
"""

from __future__ import annotations

import numpy as np

from stride.data.loader import load_pen_human, make_datasets, make_dataloaders
from stride.models.policy import BCPolicy
from stride.training.train_bc import train_bc
from stride.influence.trak import compute_influence_scores_batched, compute_influence_scores


def run_cupid_reimpl_bc(
    data: dict | None = None,
    bc_policy: BCPolicy | None = None,
    bc_ckpt: str = "checkpoints/bc_policy.pt",
    proj_dim: int = 512,
    keep_ratio: float = 0.7,
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 256,
    device_str: str = "cpu",
    out_path: str = "checkpoints/cupid_reimpl_bc.pt",
    seed: int = 42,
    verbose: bool = True,
) -> BCPolicy:
    """Train a Cupid-style curated BC baseline."""
    import torch
    from stride.models.policy import BCPolicy as _BC

    if data is None:
        data = load_pen_human()

    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1].")

    train_ds, val_ds, train_idx, _ = make_datasets(data, seed=seed)
    train_loader, val_loader = make_dataloaders(train_ds, val_ds, batch_size=batch_size)

    if bc_policy is None:
        ckpt = torch.load(bc_ckpt, map_location="cpu")
        bc_policy = _BC(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"])
        bc_policy.load_state_dict(ckpt["state_dict"])

    if verbose:
        print("[CupidReimpl] Computing influence scores ...")

    try:
        influence_raw = compute_influence_scores_batched(
            bc_policy, train_loader, val_loader,
            proj_dim=proj_dim, seed=seed, device=device_str
        )
    except Exception:
        influence_raw = compute_influence_scores(
            bc_policy, train_loader, val_loader,
            proj_dim=proj_dim, seed=seed, device=device_str
        )

    n_keep = max(1, int(len(influence_raw) * keep_ratio))
    keep_local = np.argsort(influence_raw)[-n_keep:]
    keep_idx = train_idx[keep_local]

    curated_data = {
        "observations": data["observations"][keep_idx],
        "actions": data["actions"][keep_idx],
        # Mark as one episode to keep loader assumptions simple for this baseline.
        "episode_ends": [len(keep_idx)],
    }

    if verbose:
        kept_pct = 100.0 * n_keep / len(influence_raw)
        print(f"[CupidReimpl] Keeping {n_keep}/{len(influence_raw)} samples ({kept_pct:.1f}%).")

    return train_bc(
        data=curated_data,
        val_data=data,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device_str=device_str,
        out_path=out_path,
        seed=seed,
        verbose=verbose,
    )
