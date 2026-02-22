"""Baseline 4: Random Latent Editing.

Applies isotropic Gaussian noise to the VAE latent space without any
influence-guided direction:

    z_i' = μ(s_i, a_i) + ε,   ε ~ N(0, σ²·I)

Then decodes  a_i' = D_φ(z_i', s_i)  and trains BC on these random
perturbations.  This tests whether random perturbations in the latent space
improve performance independent of the influence-guided direction in STRIDE.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stride.data.loader import load_pen_human, make_datasets
from stride.models.vae import ConditionalVAE
from stride.models.policy import BCPolicy
from stride.training.train_bc import train_bc


@torch.no_grad()
def random_latent_edit(
    observations: np.ndarray,
    actions: np.ndarray,
    vae: ConditionalVAE,
    noise_std: float = 0.1,
    batch_size: int = 256,
    device_str: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """Apply random latent perturbations and decode to action space.

    Parameters
    ----------
    observations : (N, obs_dim)
    actions      : (N, act_dim)
    vae          : trained ConditionalVAE (frozen)
    noise_std    : standard deviation of isotropic Gaussian noise in latent space
    batch_size   : processing batch size
    device_str   : torch device

    Returns
    -------
    edited_actions : (N, act_dim)
    """
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu"
                          else "cpu")
    vae = vae.to(device).eval()

    obs_t  = torch.from_numpy(observations).float()
    act_t  = torch.from_numpy(actions).float()
    ds     = TensorDataset(obs_t, act_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    edited: list[torch.Tensor] = []
    for obs_b, act_b in loader:
        obs_b = obs_b.to(device)
        act_b = act_b.to(device)

        mu, _ = vae.encode(obs_b, act_b)
        epsilon = torch.randn_like(mu) * noise_std
        z_prime = mu + epsilon
        a_prime = vae.decode(z_prime, obs_b)
        edited.append(a_prime.cpu())

    return torch.cat(edited, dim=0).numpy().astype(np.float32)


def run_random_latent_bc(
    data: dict | None = None,
    vae: ConditionalVAE | None = None,
    vae_ckpt: str = "checkpoints/vae.pt",
    noise_std: float = 0.1,
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 256,
    device_str: str = "cpu",
    out_path: str = "checkpoints/random_latent_bc.pt",
    seed: int = 42,
    verbose: bool = True,
) -> BCPolicy:
    """Apply random latent noise to training data and train BC.

    Parameters
    ----------
    data      : raw dataset; loaded if None
    vae       : trained ConditionalVAE; loaded from vae_ckpt if None
    noise_std : latent space noise level

    Returns
    -------
    Trained BCPolicy on randomly-perturbed dataset.
    """
    if data is None:
        data = load_pen_human()

    # ---- Load VAE -------------------------------------------------------
    if vae is None:
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        vae = ConditionalVAE(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"],
                              latent_dim=ckpt["latent_dim"])
        vae.load_state_dict(ckpt["state_dict"])

    # ---- Get same train split as other methods ---------------------------
    _, _, train_idx, _ = make_datasets(data, seed=seed)

    obs_train = data["observations"][train_idx]
    act_train = data["actions"][train_idx]

    if verbose:
        print(f"[RandomLatent] Editing {len(obs_train)} samples with σ={noise_std} …")

    edited_actions = random_latent_edit(
        obs_train, act_train, vae,
        noise_std=noise_std, batch_size=batch_size,
        device_str=device_str, seed=seed,
    )

    edited_data = {
        "observations": obs_train,
        "actions": edited_actions,
    }

    return train_bc(
        data=edited_data,
        val_data=data,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device_str=device_str,
        out_path=out_path,
        use_weights=False,
        seed=seed,
        verbose=verbose,
    )
