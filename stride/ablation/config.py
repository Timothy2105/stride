"""Centralized ablation configuration for the entire STRIDE pipeline.

Every tunable parameter in STRIDE is captured here so that ablation
experiments only need to override the fields they care about.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AblationConfig:
    """Complete configuration for a single STRIDE ablation run."""

    # -- Experiment metadata --------------------------------------------------
    name: str = "baseline"
    description: str = ""
    group: str = "default"  # Group name for aggregating results

    # -- Device / reproducibility ---------------------------------------------
    device: str = "cpu"
    seed: int = 42
    smoke_test: bool = False  # 2 epochs everywhere if True

    # -- Data -----------------------------------------------------------------
    task: str = "D4RL/pen/human-v2"

    # -- BC Policy architecture -----------------------------------------------
    policy_hidden: tuple[int, ...] = (256, 256)

    # -- BC training ----------------------------------------------------------
    epochs_bc: int = 100
    lr_bc: float = 3e-4
    batch_size: int = 256
    use_cosine_lr: bool = False
    obs_noise_std: float = 0.0

    # -- VAE architecture -----------------------------------------------------
    vae_latent_dim: int = 16
    vae_hidden: tuple[int, ...] = (256, 256)

    # -- VAE training ---------------------------------------------------------
    epochs_vae: int = 200
    lr_vae: float = 3e-4
    vae_beta: float = 0.5
    vae_anneal_epochs: int = 50

    # -- Editor architecture --------------------------------------------------
    editor_hidden: tuple[int, ...] = (256, 256)

    # -- Editor (DPO) training ------------------------------------------------
    epochs_editor: int = 100
    lr_editor: float = 3e-4
    dpo_beta: float = 2.0

    # -- Loss component weights -----------------------------------------------
    lambda_cos: Optional[float] = 0.2
    lambda_reg: Optional[float] = 0.3

    # -- Loss component toggles (for removal ablations) -----------------------
    use_dpo_loss: bool = True
    use_cosine_loss: bool = True
    use_reg_loss: bool = True

    # -- Influence computation ------------------------------------------------
    proj_dim: int = 512
    k_neighbors: int = 10

    # -- Editing parameters ---------------------------------------------------
    edit_scale: float = 0.6
    blend_alpha: float = 0.35
    n_aug: int = 4
    aug_noise_std: float = 0.07

    # -- Component-level toggles (full removal) --------------------------------
    use_influence: bool = True       # If False, skip TRAK → random directions
    use_editor: bool = True          # If False, skip editor → vanilla BC
    use_vae_editing: bool = True     # If False, skip VAE decode step
    use_augmentation: bool = True    # If False, n_aug = 0
    use_lspo: bool = False           # If True, train BC policy in latent space
    use_lspo_norm: bool = True       # If True, use latent normalization for LSPO
    use_ranking_dpo: bool = False    # If True, use ranking-based DPO loss

    # -- Baseline-specific flags -----------------------------------------------
    use_gaussian_filter: bool = False     # Baseline: Gaussian smooth actions
    gaussian_sigma: float = 2.0
    
    use_influence_weighting: bool = False # Baseline: Weight BC loss by influence
    soft_min_weight: float = 0.1
    
    use_random_latent: bool = False      # Baseline: Random noise in VAE latents
    latent_noise_std: float = 0.1

    # -- Evaluation -----------------------------------------------------------
    n_eval_episodes: int = 20
    skip_eval: bool = False

    # -- Helpers ---------------------------------------------------------------

    def apply_smoke_test(self) -> "AblationConfig":
        """Return a copy with all epoch counts set to 2."""
        c = self.clone()
        c.epochs_bc = 2
        c.epochs_vae = 2
        c.epochs_editor = 2
        c.vae_anneal_epochs = 1
        c.n_eval_episodes = 2
        c.smoke_test = True
        return c

    def clone(self, **overrides) -> "AblationConfig":
        """Return a shallow copy, optionally overriding fields."""
        d = asdict(self)
        # Convert lists back to tuples for hidden dims
        for k in ("policy_hidden", "vae_hidden", "editor_hidden"):
            if isinstance(d.get(k), list):
                d[k] = tuple(d[k])
        d.update(overrides)
        return AblationConfig(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        for k in ("policy_hidden", "vae_hidden", "editor_hidden"):
            if isinstance(d.get(k), tuple):
                d[k] = list(d[k])
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "AblationConfig":
        # Convert lists back to tuples for hidden dims
        for k in ("policy_hidden", "vae_hidden", "editor_hidden"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**d)

    def __repr__(self) -> str:
        return f"AblationConfig(name={self.name!r}, group={self.group!r}, descr={self.description!r})"
