"""Baseline 3: Vanilla BC + Influence Weighting.

Adjusts the BC training loss using per-sample influence scores without
modifying the data itself.  The MSE loss for each sample is scaled by its
(non-negative, normalised) influence weight:

    L_weighted = Σ_i  w_i · ‖π_θ(s_i) − a_i‖²  /  Σ_i w_i

where  w_i = max(0, Ĩ_i) + soft_min
and    Ĩ_i = (I_i − μ) / σ   (standardised influence score).

A small soft_min (default 0.1) ensures every sample still contributes to
training, preventing extreme down-weighting of negatively-influential samples.
"""

from __future__ import annotations

import numpy as np

from stride.data.loader import load_pen_human, make_datasets, make_dataloaders
from stride.models.policy import BCPolicy
from stride.models.policy import bc_loss
from stride.training.train_bc import train_bc
from stride.influence.trak import compute_influence_scores_batched, compute_influence_scores
from stride.influence.selection import normalise_influence_scores


def run_influence_weighted_bc(
    data: dict | None = None,
    bc_policy: BCPolicy | None = None,
    bc_ckpt: str = "checkpoints/bc_policy.pt",
    proj_dim: int = 512,
    soft_min: float = 0.1,
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 256,
    device_str: str = "cpu",
    out_path: str = "checkpoints/influence_weighted_bc.pt",
    seed: int = 42,
    verbose: bool = True,
) -> BCPolicy:
    """Compute influence scores → train weighted BC.

    Parameters
    ----------
    data       : raw dataset dict; loaded if None
    bc_policy  : pre-trained BC policy for influence computation; loaded from
                 bc_ckpt if None
    proj_dim   : TRAK projection dimensionality
    soft_min   : minimum weight added to all samples (prevents zero weights)

    Returns
    -------
    Trained BCPolicy with influence-weighted loss.
    """
    import torch
    from stride.models.policy import BCPolicy as _BC

    device = device_str if str(device_str) != "cpu" else "cpu"

    if data is None:
        data = load_pen_human()

    # ---- Train/val split (same split used for influence) ------------------
    train_ds, val_ds, train_idx, val_idx = make_datasets(data, seed=seed)
    train_loader, val_loader = make_dataloaders(train_ds, val_ds, batch_size=batch_size)

    obs_dim = data["observations"].shape[1]
    act_dim = data["actions"].shape[1]

    # ---- Load base BC policy ---------------------------------------------
    if bc_policy is None:
        ckpt = torch.load(bc_ckpt, map_location="cpu")
        bc_policy = _BC(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"])
        bc_policy.load_state_dict(ckpt["state_dict"])

    if verbose:
        print("[InfluenceWeighted] Computing TRAK influence scores …")

    try:
        influence_raw = compute_influence_scores_batched(
            bc_policy, train_loader, val_loader,
            proj_dim=proj_dim, seed=seed, device=device)
    except Exception:
        influence_raw = compute_influence_scores(
            bc_policy, train_loader, val_loader,
            proj_dim=proj_dim, seed=seed, device=device)

    _, weights = normalise_influence_scores(influence_raw)
    # Add soft minimum so every sample has at least soft_min weight
    weights = weights + soft_min

    # ---- Expand weights to full dataset size (train_idx only) ------------
    # train_bc with use_weights=True expects weights aligned with train split
    # which is exactly what we have from the influence computation.
    # We create a full-dataset weight array and pass it to make_datasets.
    full_weights = np.ones(len(data["observations"]), dtype=np.float32)
    full_weights[train_idx] = weights  # only train samples have influence weights

    return train_bc(
        data=data,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device_str=device_str,
        out_path=out_path,
        use_weights=True,
        influence_weights=full_weights,
        seed=seed,
        verbose=verbose,
    )
