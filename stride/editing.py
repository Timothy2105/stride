"""Apply the trained STRIDE editor to produce an edited dataset D'.

Two-stage editing pipeline:
  Stage 1 — Influence-guided correction:
    For each (s_i, a_i):
      z_i' = μ(s_i, a_i) + δz_i · edit_scale
      a_edited_i = D_φ(z_i', s_i)
      a_corrected_i = (1 - blend_alpha) · a_i + blend_alpha · a_edited_i

  Stage 2 — Latent augmentation (optional):
    Generate n_aug copies of each corrected sample by adding small
    isotropic noise in latent space, similar to Random Latent BC.
    This expands the dataset and improves policy robustness.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stride.models.vae import ConditionalVAE
from stride.models.editor import LatentEditor


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def edit_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    vae: ConditionalVAE,
    editor: LatentEditor,
    batch_size: int = 256,
    device_str: str = "cpu",
    seed: int = 42,
    edit_scale: float = 1.0,
) -> np.ndarray:
    """Apply STRIDE editing to produce edited actions a'.

    Returns
    -------
    edited_actions : (N, act_dim)
    """
    torch.manual_seed(seed)
    device = _resolve_device(device_str)

    vae = vae.to(device).eval()
    editor = editor.to(device).eval()

    obs_t = torch.from_numpy(observations).float()
    act_t = torch.from_numpy(actions).float()

    ds = TensorDataset(obs_t, act_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    edited_actions_list: list[torch.Tensor] = []

    for obs_b, act_b in loader:
        obs_b = obs_b.to(device)
        act_b = act_b.to(device)

        mu, _ = vae.encode(obs_b, act_b)
        xi = torch.randn_like(mu)
        delta_z = editor(obs_b, act_b, xi) * edit_scale

        z_prime = mu + delta_z
        a_prime = vae.decode(z_prime, obs_b)
        edited_actions_list.append(a_prime.cpu())

    return torch.cat(edited_actions_list, dim=0).numpy().astype(np.float32)


@torch.no_grad()
def augment_in_latent_space(
    observations: np.ndarray,
    actions: np.ndarray,
    vae: ConditionalVAE,
    n_aug: int = 2,
    noise_std: float = 0.05,
    batch_size: int = 256,
    device_str: str = "cpu",
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Create augmented copies by adding noise in VAE latent space.

    For each (s_i, a_i), generates n_aug copies where:
      z_i = μ(s_i, a_i)
      z_aug = z_i + ε,  ε ~ N(0, noise_std² · I)
      a_aug = D_φ(z_aug, s_i)

    Returns
    -------
    aug_obs : (N * n_aug, obs_dim)  augmented observations (duplicated)
    aug_act : (N * n_aug, act_dim)  augmented actions
    """
    torch.manual_seed(seed)
    device = _resolve_device(device_str)
    vae = vae.to(device).eval()

    obs_t = torch.from_numpy(observations).float()
    act_t = torch.from_numpy(actions).float()
    ds = TensorDataset(obs_t, act_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_obs: list[torch.Tensor] = []
    all_act: list[torch.Tensor] = []

    for obs_b, act_b in loader:
        obs_b = obs_b.to(device)
        act_b = act_b.to(device)

        mu, _ = vae.encode(obs_b, act_b)

        for _ in range(n_aug):
            epsilon = torch.randn_like(mu) * noise_std
            z_aug = mu + epsilon
            a_aug = vae.decode(z_aug, obs_b)
            all_obs.append(obs_b.cpu())
            all_act.append(a_aug.cpu())

    aug_obs = torch.cat(all_obs, dim=0).numpy().astype(np.float32)
    aug_act = torch.cat(all_act, dim=0).numpy().astype(np.float32)
    return aug_obs, aug_act


def apply_stride(
    data: dict,
    train_idx: np.ndarray,
    vae: ConditionalVAE,
    editor: LatentEditor,
    edit_scale: float = 1.0,
    blend_alpha: float = 0.3,
    n_aug: int = 0,
    aug_noise_std: float = 0.05,
    batch_size: int = 256,
    device_str: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Produce an edited + augmented version of the training dataset.

    Parameters
    ----------
    data        : raw dataset dict
    train_idx   : training sample indices
    vae, editor : trained models (frozen)
    edit_scale  : δz scaling factor
    blend_alpha : action interpolation (0=original, 1=fully edited)
    n_aug       : number of latent augmentation copies per sample (0=disabled)
    aug_noise_std : std of latent noise for augmentation

    Returns
    -------
    dict with 'observations' and 'actions'
    """
    obs_train = data["observations"][train_idx]
    act_train = data["actions"][train_idx]

    # Stage 1: Influence-guided correction
    edited_actions = edit_dataset(
        obs_train, act_train, vae, editor,
        batch_size=batch_size, device_str=device_str, seed=seed,
        edit_scale=edit_scale,
    )

    # Blend original + edited
    corrected_actions = (
        (1.0 - blend_alpha) * act_train + blend_alpha * edited_actions
    ).astype(np.float32)

    if verbose:
        delta_norm = np.linalg.norm(corrected_actions - act_train, axis=1).mean()
        orig_norm = np.linalg.norm(act_train, axis=1).mean()
        print(f"  [STRIDE] Stage 1: Edited {len(act_train)} samples  "
              f"blend={blend_alpha:.2f}  scale={edit_scale:.2f}  "
              f"Δa={delta_norm:.4f} ({delta_norm / orig_norm * 100:.1f}%)")

    # Stage 2: Latent augmentation (if enabled)
    if n_aug > 0:
        aug_obs, aug_act = augment_in_latent_space(
            obs_train, corrected_actions, vae,
            n_aug=n_aug, noise_std=aug_noise_std,
            batch_size=batch_size, device_str=device_str, seed=seed + 1,
        )
        # Combine original corrected data with augmented data
        final_obs = np.concatenate([obs_train, aug_obs], axis=0)
        final_act = np.concatenate([corrected_actions, aug_act], axis=0)

        if verbose:
            print(f"  [STRIDE] Stage 2: Augmented with {n_aug} copies/sample "
                  f"(σ={aug_noise_std})  total={len(final_obs)} samples")
    else:
        final_obs = obs_train
        final_act = corrected_actions

    return {
        "observations": final_obs,
        "actions": final_act,
    }
