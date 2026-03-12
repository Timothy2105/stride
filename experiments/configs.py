"""Experiment configurations for all STRIDE methods and baselines.

Each configuration specifies:
  - Method name and description
  - Data preprocessing (Gaussian σ, CUPID keep ratio, etc.)
  - STRIDE pipeline hyperparameters (VAE, editor, editing)
  - BC training hyperparameters
  - Evaluation settings

All methods for a given task share the same BC architecture and training
schedule so that differences are attributable to data processing only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ExperimentConfig:
    """Complete configuration for a single experiment run."""

    # -- Identity -----------------------------------------------------------
    method: str = "vanilla_bc"
    description: str = ""
    task: str = "pen"

    # -- Device / reproducibility -------------------------------------------
    device: str = "cuda"
    seed: int = 42

    # -- BC policy ----------------------------------------------------------
    bc_hidden: tuple[int, ...] = (256, 256)
    bc_epochs: int = 200
    bc_lr: float = 3e-4
    bc_batch_size: int = 256
    bc_weight_decay: float = 1e-4
    bc_grad_clip: float = 1.0

    # -- Gaussian baseline --------------------------------------------------
    gaussian_sigma: Optional[float] = None  # None = no Gaussian smoothing

    # -- CUPID / CUPID-Quality baseline -------------------------------------
    cupid_keep_ratio: Optional[float] = None  # None = no CUPID filtering

    # -- TRAK scoring (shared by CUPID, CUPID-Q, STRIDE) -------------------
    trak_proj_dim: int = 512
    trak_lambda_reg: float = 1e-3
    trak_n_rollouts: int = 100
    trak_rollout_seed: int = 0

    # -- VAE (STRIDE / ablations) -------------------------------------------
    vae_latent_dim: int = 16
    vae_hidden: tuple[int, ...] = (256, 256)
    vae_epochs: int = 200
    vae_lr: float = 3e-4
    vae_beta: float = 0.5
    vae_anneal_epochs: int = 50

    # -- DPO Editor (STRIDE) ------------------------------------------------
    editor_hidden: tuple[int, ...] = (256, 256)
    editor_epochs: int = 100
    editor_lr: float = 3e-4
    editor_beta_dpo: float = 2.0
    editor_lambda_reg: float = 0.3
    editor_lambda_cos: float = 0.2
    editor_k_neighbors: int = 10

    # -- STRIDE editing -----------------------------------------------------
    edit_scale: float = 0.6
    blend_alpha: float = 0.35
    n_aug: int = 4
    aug_noise_std: float = 0.07

    # -- Ablation flags -----------------------------------------------------
    use_influence: bool = True   # False → random scores for editor training
    use_editor: bool = True      # False → random latent edits instead
    random_latent_std: float = 0.1  # noise σ for random latent ablation

    # -- Evaluation ---------------------------------------------------------
    n_eval_episodes: int = 20
    render_videos: bool = True
    max_episode_steps: int = 400

    # -- Helpers ------------------------------------------------------------

    @property
    def run_name(self) -> str:
        return f"{self.task}_{self.method}_seed{self.seed}"

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("bc_hidden", "vae_hidden", "editor_hidden"):
            if isinstance(d.get(k), tuple):
                d[k] = list(d[k])
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ===================================================================
# Pre-built configurations for every method
# ===================================================================

def _base(task: str, seed: int = 42, device: str = "cuda") -> dict:
    """Shared defaults for a task."""
    return dict(task=task, seed=seed, device=device)


# 1) Vanilla MLP BC — no data processing
def vanilla_bc(task: str, seed: int = 42, **kw) -> ExperimentConfig:
    return ExperimentConfig(
        method="vanilla_bc",
        description="MLP BC trained on unmodified demonstrations.",
        **_base(task, seed, **kw),
    )


# 2) Gaussian filtering at sigma levels
_GAUSSIAN_SIGMAS = {"25": 2.5, "50": 5.0, "75": 7.5}

def gaussian(task: str, level: str, seed: int = 42, **kw) -> ExperimentConfig:
    sigma = _GAUSSIAN_SIGMAS[level]
    return ExperimentConfig(
        method=f"gaussian_{level}",
        description=f"Gaussian temporal smoothing (σ={sigma}).",
        gaussian_sigma=sigma,
        **_base(task, seed, **kw),
    )


# 3) CUPID filtering at keep ratios
def cupid(task: str, pct: int, seed: int = 42, **kw) -> ExperimentConfig:
    return ExperimentConfig(
        method=f"cupid_{pct}",
        description=f"CUPID curation — keep top {pct}% demos by TRAK influence.",
        cupid_keep_ratio=pct / 100.0,
        **_base(task, seed, **kw),
    )


# 4) CUPID-Quality filtering at keep ratios
def cupid_quality(task: str, pct: int, seed: int = 42, **kw) -> ExperimentConfig:
    return ExperimentConfig(
        method=f"cupid_quality_{pct}",
        description=(
            f"CUPID-Quality curation — keep top {pct}% demos using "
            "weighted ensemble of sum_of_sum, min_of_max, max_of_min."
        ),
        cupid_keep_ratio=pct / 100.0,
        **_base(task, seed, **kw),
    )


# 5) Full STRIDE
def stride_full(task: str, seed: int = 42, **kw) -> ExperimentConfig:
    return ExperimentConfig(
        method="stride",
        description=(
            "STRIDE: VAE + DPO editor trained with TRAK influence scores → "
            "edit data → train BC on edited data."
        ),
        use_influence=True,
        use_editor=True,
        **_base(task, seed, **kw),
    )


# 6) STRIDE without influence (ablation)
def stride_no_influence(task: str, seed: int = 42, **kw) -> ExperimentConfig:
    return ExperimentConfig(
        method="stride_no_influence",
        description=(
            "Ablation: STRIDE pipeline with random (uninformative) influence "
            "scores instead of TRAK. Tests whether influence guidance matters."
        ),
        use_influence=False,
        use_editor=True,
        **_base(task, seed, **kw),
    )


# 7) STRIDE with random latent edits (ablation)
def stride_random_edits(task: str, seed: int = 42, **kw) -> ExperimentConfig:
    return ExperimentConfig(
        method="stride_random_edits",
        description=(
            "Ablation: replace the trained DPO editor with random "
            "Gaussian noise in VAE latent space. Tests whether learned "
            "edit directions matter."
        ),
        use_influence=True,
        use_editor=False,
        **_base(task, seed, **kw),
    )


# 8) BC + Influence Reweighting
def influence_reweight(task: str, seed: int = 42, **kw) -> ExperimentConfig:
    return ExperimentConfig(
        method="influence_reweight",
        description=(
            "BC with per-sample loss weighting by TRAK influence score. "
            "Demonstrations with higher net positive influence on test "
            "successes receive higher weight in the BC loss."
        ),
        **_base(task, seed, **kw),
    )


# ===================================================================
# Build full experiment grid
# ===================================================================

TASKS = ("pen", "hammer", "door", "relocate")
GAUSSIAN_LEVELS = ("25", "50", "75")
CUPID_PCTS = (25, 50, 75)

def build_all_configs(
    tasks: tuple[str, ...] = TASKS,
    seeds: tuple[int, ...] = (42,),
    device: str = "cuda",
) -> list[ExperimentConfig]:
    """Generate the full experiment grid.

    For each task × seed, produces:
      1  × vanilla_bc
      3  × gaussian  (25%, 50%, 75%)
      3  × cupid     (25%, 50%, 75%)
      3  × cupid_quality (25%, 50%, 75%)
      1  × stride
      1  × stride_no_influence
      1  × stride_random_edits
      1  × influence_reweight
     ---
     14 configs per (task, seed).
    """
    configs: list[ExperimentConfig] = []
    for task in tasks:
        for seed in seeds:
            kw = dict(device=device)
            configs.append(vanilla_bc(task, seed, **kw))
            for lvl in GAUSSIAN_LEVELS:
                configs.append(gaussian(task, lvl, seed, **kw))
            for pct in CUPID_PCTS:
                configs.append(cupid(task, pct, seed, **kw))
            for pct in CUPID_PCTS:
                configs.append(cupid_quality(task, pct, seed, **kw))
            configs.append(stride_full(task, seed, **kw))
            configs.append(stride_no_influence(task, seed, **kw))
            configs.append(stride_random_edits(task, seed, **kw))
            configs.append(influence_reweight(task, seed, **kw))
    return configs
