"""Conditional Variational Autoencoder (CVAE) for action representations.

Architecture
------------
Encoder  q(z | s, a)  →  (μ, log σ²)   inputs: concat(s, a)
Decoder  D_φ(z, s)   →  â              inputs: concat(z, s)

ELBO training objective:
  L_VAE = ‖a - â‖²  +  β · KL(q(z|s,a) ‖ N(0, I))

The reparameterisation trick is used during training.
At inference, z_i = μ(s_i, a_i)  (deterministic mean).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ConditionalVAE(nn.Module):
    """Conditional VAE operating on (state, action) pairs.

    Parameters
    ----------
    obs_dim    : state dimensionality (45 for pen-v2)
    act_dim    : action dimensionality (24 for pen-v2)
    latent_dim : dimensionality of latent space z (default 16)
    hidden     : hidden layer widths used in encoder and decoder
    beta       : KL divergence weight in the ELBO
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int = 16,
        hidden: tuple[int, ...] = (256, 256),
        beta: float = 0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.beta = beta

        
        
        self.register_buffer('_obs_mean', torch.zeros(obs_dim))
        self.register_buffer('_obs_std', torch.ones(obs_dim))

        
        enc_in = obs_dim + act_dim
        self.encoder_shared = _mlp(enc_in, hidden, hidden[-1])
        
        self.fc_mu = nn.Linear(hidden[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden[-1], latent_dim)

        
        dec_in = latent_dim + obs_dim
        self.decoder = _mlp(dec_in, hidden, act_dim)

    
    
    

    def set_obs_norm(self, mean, std) -> None:
        """Store observation normalisation statistics (mean/std arrays or tensors)."""
        import numpy as np
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean.astype(np.float32))
            std = torch.from_numpy(std.astype(np.float32))
        self._obs_mean.copy_(mean)
        self._obs_std.copy_(std)

    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise a batch of observations using stored statistics."""
        return (obs - self._obs_mean.to(obs.device)) / (self._obs_std.to(obs.device) + 1e-6)

    def encode(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (μ, log σ²) given (s, a).

        Parameters
        ----------
        obs : (B, obs_dim)  raw (unnormalised) observations
        act : (B, act_dim)

        Returns
        -------
        mu, logvar : each (B, latent_dim)
        """
        x = torch.cat([self._norm_obs(obs), act], dim=-1)
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + ε·σ  with  ε ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  

    def decode(self, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Decode latent z conditioned on state s → reconstructed action â.

        Parameters
        ----------
        z   : (B, latent_dim)
        obs : (B, obs_dim)

        Returns
        -------
        a_hat : (B, act_dim)
        """
        x = torch.cat([z, self._norm_obs(obs)], dim=-1)
        return self.decoder(x)

    
    
    

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward.

        Returns
        -------
        a_hat   : (B, act_dim) reconstructed action
        mu      : (B, latent_dim)
        logvar  : (B, latent_dim)
        """
        mu, logvar = self.encode(obs, act)
        z = self.reparameterise(mu, logvar)
        a_hat = self.decode(z, obs)
        return a_hat, mu, logvar

    
    
    

    def loss(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        beta: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE ELBO loss.

        L_VAE = recon_loss + β · kl_loss

        Parameters
        ----------
        obs, act : (B, *) tensors of states and actions
        beta     : override instance beta if given

        Returns
        -------
        total_loss, recon_loss, kl_loss  (all scalars)
        """
        beta = beta if beta is not None else self.beta
        a_hat, mu, logvar = self.forward(obs, act)

        recon_loss = F.mse_loss(a_hat, act, reduction="mean")
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total = recon_loss + beta * kl_loss
        return total, recon_loss, kl_loss
