"""MLP Behavioral Cloning policy for Adroit manipulation tasks.

Architecture
------------
π_θ(s) → a

Multi-layer perceptron with LayerNorm and ReLU activations.
Observation normalization is stored as model buffers so the policy
is self-contained at inference time.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """Deterministic MLP policy for behavioral cloning.

    Parameters
    ----------
    obs_dim : observation dimensionality
    act_dim : action dimensionality
    hidden  : hidden layer widths
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Observation normalization buffers (identity by default).
        self.register_buffer("_obs_mean", torch.zeros(obs_dim))
        self.register_buffer("_obs_std", torch.ones(obs_dim))

        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()])
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Obs normalisation helpers
    # ------------------------------------------------------------------

    def set_obs_norm(self, mean, std) -> None:
        """Store observation normalisation statistics."""
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean.astype(np.float32))
            std = torch.from_numpy(std.astype(np.float32))
        self._obs_mean.copy_(mean)
        self._obs_std.copy_(std)

    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self._obs_mean) / (self._obs_std + 1e-6)

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map observations to actions.

        Parameters
        ----------
        obs : (B, obs_dim) raw (unnormalised) observations

        Returns
        -------
        actions : (B, act_dim)
        """
        return self.net(self._norm_obs(obs))

    @torch.no_grad()
    def get_action(self, obs_np: np.ndarray) -> np.ndarray:
        """Single-step inference from a numpy observation.

        Parameters
        ----------
        obs_np : (obs_dim,) numpy array

        Returns
        -------
        action : (act_dim,) numpy array
        """
        device = self._obs_mean.device
        obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
        action = self.forward(obs).squeeze(0).cpu().numpy()
        return action


def load_policy_from_checkpoint(ckpt_path: str, device: str = "cpu") -> MLPPolicy:
    """Restore an MLPPolicy from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden = ckpt["hidden"]
    if isinstance(hidden, list):
        hidden = tuple(hidden)
    policy = MLPPolicy(
        obs_dim=ckpt["obs_dim"],
        act_dim=ckpt["act_dim"],
        hidden=hidden,
    )
    policy.load_state_dict(ckpt["state_dict"])
    return policy.to(device)
