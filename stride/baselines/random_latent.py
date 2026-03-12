"""Ablation baseline: random latent edits.

Instead of using the trained DPO editor to produce latent residuals,
we add isotropic Gaussian noise directly in the VAE latent space:

    z_i  = μ(s_i, a_i)          (encode via VAE)
    z'_i = z_i + ε,  ε ~ N(0, σ²I)   (random perturbation)
    a'_i = D_φ(z'_i, s_i)       (decode back to action space)

This ablation tests whether the learned editor direction matters or
whether any perturbation in latent space is sufficient.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stride.models.vae import ConditionalVAE

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


@torch.no_grad()
def random_latent_edit(
    observations: np.ndarray,
    actions: np.ndarray,
    vae: ConditionalVAE,
    noise_std: float = 0.1,
    blend_alpha: float = 0.3,
    batch_size: int = 256,
    device_str: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """Apply random latent-space perturbations through a trained VAE.

    Parameters
    ----------
    observations : (N, obs_dim)
    actions      : (N, act_dim)
    vae          : trained ConditionalVAE (frozen).
    noise_std    : standard deviation of isotropic Gaussian noise in latent space.
    blend_alpha  : interpolation weight (0 = original, 1 = fully edited).
    batch_size   : processing batch size.
    device_str   : compute device.
    seed         : random seed for reproducibility.

    Returns
    -------
    edited_actions : (N, act_dim) blended actions.
    """
    torch.manual_seed(seed)
    device = _resolve_device(device_str)
    vae = vae.to(device).eval()

    obs_t = torch.from_numpy(observations).float()
    act_t = torch.from_numpy(actions).float()
    ds = TensorDataset(obs_t, act_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    edited_list: list[torch.Tensor] = []

    for obs_b, act_b in loader:
        obs_b, act_b = obs_b.to(device), act_b.to(device)
        mu, _ = vae.encode(obs_b, act_b)

        epsilon = torch.randn_like(mu) * noise_std
        z_prime = mu + epsilon
        a_prime = vae.decode(z_prime, obs_b)
        edited_list.append(a_prime.cpu())

    edited_actions = torch.cat(edited_list, dim=0).numpy().astype(np.float32)

    # Blend original and edited
    blended = (1.0 - blend_alpha) * actions + blend_alpha * edited_actions
    blended = blended.astype(np.float32)

    delta_norm = np.linalg.norm(blended - actions, axis=1).mean()
    logger.info(
        f"[RandomLatent] Edited {len(actions)} samples  "
        f"noise_std={noise_std}  blend={blend_alpha:.2f}  "
        f"mean Δa={delta_norm:.4f}"
    )
    return blended
