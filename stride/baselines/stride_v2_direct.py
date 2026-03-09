"""Baseline: STRIDE v2 direct pipeline wrapper.

Trains/uses VAE and DPO editor, edits the train split with STRIDE,
and then trains BC directly on the edited dataset.
"""

from __future__ import annotations

from stride.data.loader import load_pen_human, make_datasets
from stride.models.policy import BCPolicy
from stride.models.vae import ConditionalVAE
from stride.models.editor import LatentEditor
from stride.training.train_bc import train_bc
from stride.training.train_vae import train_vae
from stride.training.train_editor_dpo import train_editor_dpo
from stride.editing.edit import apply_stride


def run_stride_v2_direct_bc(
    data: dict | None = None,
    bc_policy: BCPolicy | None = None,
    vae: ConditionalVAE | None = None,
    editor: LatentEditor | None = None,
    epochs_bc: int = 100,
    epochs_vae: int = 200,
    epochs_editor: int = 100,
    train_frac: float = 0.8,
    lr_bc: float = 3e-4,
    lr_vae: float = 3e-4,
    lr_editor: float = 3e-4,
    batch_size: int = 256,
    num_workers: int = 0,
    edit_scale: float = 0.6,
    blend_alpha: float = 0.35,
    n_aug: int = 4,
    aug_noise_std: float = 0.07,
    # DPO editor loss / influence settings
    beta_dpo: float = 2.0,
    lambda_reg: float = 0.3,
    lambda_cos: float = 0.2,
    k_neighbors: int = 10,
    proj_dim: int = 512,
    # VAE settings
    latent_dim: int = 16,
    vae_hidden: tuple[int, ...] = (256, 256),
    vae_beta: float = 0.5,
    anneal_epochs: int = 50,
    vae_weight_decay: float = 1e-4,
    vae_grad_clip: float = 1.0,
    # Editor settings
    editor_hidden: tuple[int, ...] = (256, 256),
    editor_weight_decay: float = 1e-4,
    editor_grad_clip: float = 1.0,
    # Final BC settings
    bc_hidden: tuple[int, ...] = (256, 256),
    use_cosine_lr: bool = False,
    cosine_eta_min: float = 1e-5,
    bc_weight_decay: float = 1e-4,
    bc_grad_clip: float = 1.0,
    obs_noise_std: float = 0.0,
    device_str: str = "cpu",
    out_path: str = "checkpoints/stride_v2_direct_bc.pt",
    seed: int = 42,
    verbose: bool = True,
) -> BCPolicy:
    """Run the end-to-end STRIDE v2 direct baseline and train BC on edited data."""
    if data is None:
        data = load_pen_human()

    _, _, train_idx, _ = make_datasets(data, train_frac=train_frac, seed=seed)

    if vae is None:
        vae = train_vae(
            data=data,
            epochs=epochs_vae,
            lr=lr_vae,
            latent_dim=latent_dim,
            hidden=vae_hidden,
            target_beta=vae_beta,
            anneal_epochs=anneal_epochs,
            weight_decay=vae_weight_decay,
            grad_clip=vae_grad_clip,
            train_frac=train_frac,
            batch_size=batch_size,
            num_workers=num_workers,
            device_str=device_str,
            out_path="checkpoints/stride_v2_direct_vae.pt",
            seed=seed,
            verbose=verbose,
        )

    if editor is None:
        if bc_policy is None:
            bc_policy = train_bc(
                data=data,
                epochs=epochs_bc,
                lr=lr_bc,
                batch_size=batch_size,
                hidden=bc_hidden,
                use_cosine_lr=use_cosine_lr,
                cosine_eta_min=cosine_eta_min,
                weight_decay=bc_weight_decay,
                grad_clip=bc_grad_clip,
                obs_noise_std=obs_noise_std,
                train_frac=train_frac,
                num_workers=num_workers,
                device_str=device_str,
                out_path="checkpoints/stride_v2_direct_init_bc.pt",
                seed=seed,
                verbose=verbose,
            )

        editor, _ = train_editor_dpo(
            data=data,
            vae=vae,
            bc_policy=bc_policy,
            epochs=epochs_editor,
            lr=lr_editor,
            beta=beta_dpo,
            lambda_reg=lambda_reg,
            lambda_cos=lambda_cos,
            k_neighbors=k_neighbors,
            proj_dim=proj_dim,
            batch_size=batch_size,
            hidden=editor_hidden,
            weight_decay=editor_weight_decay,
            grad_clip=editor_grad_clip,
            train_frac=train_frac,
            num_workers=num_workers,
            device_str=device_str,
            out_path="checkpoints/stride_v2_direct_editor.pt",
            seed=seed,
            verbose=verbose,
        )

    edited_data = apply_stride(
        data=data,
        train_idx=train_idx,
        vae=vae,
        editor=editor,
        edit_scale=edit_scale,
        blend_alpha=blend_alpha,
        n_aug=n_aug,
        aug_noise_std=aug_noise_std,
        batch_size=batch_size,
        device_str=device_str,
        seed=seed,
        verbose=verbose,
    )

    return train_bc(
        data=edited_data,
        val_data=data,
        epochs=epochs_bc,
        lr=lr_bc,
        batch_size=batch_size,
        hidden=bc_hidden,
        use_cosine_lr=use_cosine_lr,
        cosine_eta_min=cosine_eta_min,
        weight_decay=bc_weight_decay,
        grad_clip=bc_grad_clip,
        obs_noise_std=obs_noise_std,
        train_frac=train_frac,
        num_workers=num_workers,
        device_str=device_str,
        out_path=out_path,
        seed=seed,
        verbose=verbose,
    )
