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

from stride.data import load_pen_human


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


def build_gaussian_filtered_data(
    data: dict | None = None,
    sigma: float = 2.0,
) -> dict:
    """Return Gaussian-filtered dataset without training a policy."""
    if data is None:
        data = load_pen_human()

    smoothed_actions = smooth_actions_per_episode(
        data["actions"],
        data["episode_ends"],
        sigma=sigma,
    )

    return {
        "observations": data["observations"],
        "actions": smoothed_actions,
        "rewards": data.get("rewards"),
        "terminals": data.get("terminals"),
        "episode_ends": data["episode_ends"],
    }
