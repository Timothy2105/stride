"""Baseline 2: Gaussian-filtered BC.

Applies temporal Gaussian smoothing to action trajectories on a per-episode
basis prior to behavior cloning training.  This tests whether simple temporal
denoising improves performance without any influence-based supervision.

The smoothing is applied with scipy.ndimage.gaussian_filter1d along the time
axis independently for each action dimension.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from stride.data.loader import load_pen_human
from stride.models.policy import BCPolicy
from stride.training.train_bc import train_bc


def smooth_actions_per_episode(
    actions: np.ndarray,
    episode_ends: list[int],
    sigma: float = 2.0,
) -> np.ndarray:
    """Apply Gaussian smoothing to each episode's action trajectory independently.

    Parameters
    ----------
    actions       : (N, act_dim) concatenated action array
    episode_ends  : list of cumulative step counts marking episode boundaries
    sigma         : Gaussian kernel standard deviation (in time steps)

    Returns
    -------
    smoothed : (N, act_dim) smoothed action array
    """
    smoothed = actions.copy()
    start = 0
    for end in episode_ends:
        ep_acts = actions[start:end]           # (T, act_dim)
        # Filter along the time axis (axis=0) for each dimension
        ep_smoothed = gaussian_filter1d(ep_acts, sigma=sigma, axis=0)
        smoothed[start:end] = ep_smoothed
        start = end
    return smoothed.astype(np.float32)


def run_gaussian_filter_bc(
    data: dict | None = None,
    sigma: float = 2.0,
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 256,
    device_str: str = "cpu",
    out_path: str = "checkpoints/gaussian_filter_bc.pt",
    seed: int = 42,
    verbose: bool = True,
) -> BCPolicy:
    """Apply Gaussian filtering to action trajectories and train BC.

    Parameters
    ----------
    data   : raw dataset dict; loaded if None
    sigma  : smoothing bandwidth in time steps

    Returns
    -------
    Trained BCPolicy on the smoothed dataset.
    """
    if data is None:
        data = load_pen_human()

    smoothed_actions = smooth_actions_per_episode(
        data["actions"],
        data["episode_ends"],
        sigma=sigma,
    )

    smoothed_data = {
        "observations": data["observations"],
        "actions": smoothed_actions,
        "episode_ends": data["episode_ends"],
    }

    return train_bc(
        data=smoothed_data,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device_str=device_str,
        out_path=out_path,
        use_weights=False,
        seed=seed,
        verbose=verbose,
    )
