"""Baseline 1: Vanilla Behavior Cloning.

Standard BC with no dataset augmentation – a direct pass-through to
train_bc with the unmodified demonstration data.
"""

from __future__ import annotations

import numpy as np

from stride.data.loader import load_pen_human, make_datasets, make_dataloaders
from stride.models.policy import BCPolicy
from stride.training.train_bc import train_bc


def run_vanilla_bc(
    data: dict | None = None,
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 256,
    device_str: str = "cpu",
    out_path: str = "checkpoints/vanilla_bc.pt",
    seed: int = 42,
    verbose: bool = True,
) -> BCPolicy:
    """Train vanilla BC on the raw dataset (no modifications).

    Returns
    -------
    Trained BCPolicy.
    """
    if data is None:
        data = load_pen_human()

    return train_bc(
        data=data,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device_str=device_str,
        out_path=out_path,
        use_weights=False,
        influence_weights=None,
        seed=seed,
        verbose=verbose,
    )
