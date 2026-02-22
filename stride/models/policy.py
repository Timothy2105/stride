"""Behavior Cloning (BC) policy for continuous control.

The policy is a deterministic MLP: π_θ(s) → a.
Training minimises MSE loss  ℓ(π_θ(s_i), a_i),
optionally weighted by per-sample influence scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCPolicy(nn.Module):
    """Deterministic MLP policy trained via behaviour cloning.

    Parameters
    ----------
    obs_dim  : dimension of the observation / state vector (45 for pen-v2)
    act_dim  : dimension of the action vector (24 for pen-v2)
    hidden   : sequence of hidden layer widths
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict action from observation.

        Parameters
        ----------
        obs : (B, obs_dim) float tensor

        Returns
        -------
        action : (B, act_dim) float tensor
        """
        return self.net(obs)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def bc_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted or unweighted mean-squared-error behaviour-cloning loss.

    Parameters
    ----------
    pred_actions   : (B, act_dim)  – policy predictions
    target_actions : (B, act_dim)  – demonstration actions
    weights        : (B,) optional per-sample non-negative weights
                     (e.g. influence weights).  If None, uniform weighting.

    Returns
    -------
    scalar loss tensor
    """
    # per_sample_loss : (B,)
    per_sample = F.mse_loss(pred_actions, target_actions, reduction="none").mean(dim=-1)

    if weights is not None:
        # Normalise so that the weighted average is comparable to the unweighted mean.
        w = weights.clamp(min=0.0)
        w = w / (w.sum() + 1e-8) * len(w)   # keep scale similar to uniform
        return (w * per_sample).mean()

    return per_sample.mean()
