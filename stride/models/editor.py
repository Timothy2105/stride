"""Latent-space residual editor  g_ψ(s, a, ξ) → δz.

The editor predicts a residual δz in the VAE latent space:

    z_i' = z_i + δz_i,    δz_i = g_ψ(s_i, a_i, ξ)

where ξ ~ N(0, I) is injected noise that allows the editor to be
stochastic and explore the latent neighbourhood.

The edited action is then obtained by decoding:
    a_i' = D_φ(z_i', s_i)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class LatentEditor(nn.Module):
    """MLP that maps (s, a, ξ) to a latent residual δz.

    Parameters
    ----------
    obs_dim    : state dimensionality
    act_dim    : action dimensionality
    latent_dim : VAE latent dimensionality (equals dim(δz))
    hidden     : hidden layer widths
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int = 16,
        hidden: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        # Input: concat(s, a, ξ) where ξ has the same dim as z
        in_dim = obs_dim + act_dim + latent_dim
        self.net = _mlp(in_dim, hidden, latent_dim)

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        xi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict latent residual δz.

        Parameters
        ----------
        obs : (B, obs_dim)
        act : (B, act_dim)
        xi  : (B, latent_dim) noise; sampled from N(0, I) if None

        Returns
        -------
        delta_z : (B, latent_dim)
        """
        if xi is None:
            xi = torch.randn(obs.shape[0], self.latent_dim, device=obs.device)
        x = torch.cat([obs, act, xi], dim=-1)
        return self.net(x)

    def edit(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        vae,
        xi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the editor through the frozen VAE to get edited actions.

        Parameters
        ----------
        obs, act : (B, obs_dim), (B, act_dim) – original state/action pairs
        vae      : ConditionalVAE (frozen during editor training)
        xi       : optional explicit noise

        Returns
        -------
        a_prime  : (B, act_dim) edited actions
        delta_z  : (B, latent_dim) predicted latent residual
        """
        with torch.no_grad():
            mu, _ = vae.encode(obs, act)

        delta_z = self.forward(obs, act, xi)
        z_prime = mu + delta_z
        a_prime = vae.decode(z_prime, obs)
        return a_prime, delta_z
