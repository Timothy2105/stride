"""Run all STRIDE experiments end-to-end.

For each task the runner:
  1. Loads Adroit human-v2 demonstrations via Minari.
  2. Trains a reference MLP BC on the full data.
  3. Rolls out the reference BC to collect test episodes with success labels.
  4. Computes TRAK influence scores (CUPID and CUPID-Quality).
  5. For each method, processes data and trains a fresh BC policy.
  6. Evaluates every policy with rollouts + video capture.
  7. Logs everything to wandb and saves JSON results.

Usage
-----
    python -m experiments.run_experiments --task all --device cuda --seed 42
    python -m experiments.run_experiments --task pen --method stride
    python -m experiments.run_experiments --task door --method cupid_50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os

# Use EGL for headless MuJoCo rendering (SLURM / no display).
# Must be set before any mujoco or gymnasium import.
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stride.data import load_task_human, get_task_spec, make_datasets
from stride.training.train_bc import train_bc
from stride.training.train_vae import train_vae
from stride.training.train_editor_dpo import train_editor_dpo
from stride.models.policy import MLPPolicy
from stride.models.vae import ConditionalVAE
from stride.models.editor import LatentEditor
from stride.eval.evaluate import evaluate_policy, rollout_for_scoring
from stride.scoring import TRAKScorer, demo_scores_to_transition
from stride.editing import apply_stride
from stride.baselines.gaussian_filter import smooth_actions_per_episode
from stride.baselines.cupid_filter import filter_by_cupid
from stride.baselines.cupid_quality import filter_by_cupid_quality
from stride.baselines.random_latent import random_latent_edit
from stride.influence import normalise_influence_scores
from experiments.configs import (
    ExperimentConfig, build_all_configs, TASKS,
    GAUSSIAN_LEVELS, CUPID_PCTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stride.runner")


# ===================================================================
# Shared resource cache (per task)
# ===================================================================

class TaskResources:
    """Lazily-computed shared resources for a single task.

    These are computed once and reused across all methods that require them
    (CUPID, CUPID-Q, STRIDE, and ablations).
    """

    def __init__(self, task: str, cfg_template: ExperimentConfig):
        self.task = task
        self.cfg = cfg_template
        self.data: dict | None = None
        self.ref_policy: MLPPolicy | None = None
        self.ref_info: dict | None = None
        self.rollout_data: dict | None = None
        self.trak_results: dict | None = None
        self.vae: ConditionalVAE | None = None

    def get_data(self) -> dict:
        if self.data is None:
            logger.info(f"[{self.task}] Loading Adroit demonstrations …")
            self.data = load_task_human(self.task)
            n = len(self.data["observations"])
            n_ep = len(self.data["episode_ends"])
            logger.info(f"[{self.task}] Loaded {n} transitions in {n_ep} demos")
        return self.data

    def get_ref_policy(self, wandb_run=None) -> tuple[MLPPolicy, dict]:
        if self.ref_policy is None:
            data = self.get_data()
            out_dir = str(ROOT / "results" / "checkpoints" / self.task)
            logger.info(f"[{self.task}] Training reference BC policy …")
            self.ref_policy, self.ref_info = train_bc(
                data,
                epochs=self.cfg.bc_epochs,
                lr=self.cfg.bc_lr,
                batch_size=self.cfg.bc_batch_size,
                hidden=self.cfg.bc_hidden,
                weight_decay=self.cfg.bc_weight_decay,
                grad_clip=self.cfg.bc_grad_clip,
                device_str=self.cfg.device,
                out_dir=out_dir,
                run_name="ref_bc",
                seed=self.cfg.seed,
                wandb_run=wandb_run,
            )
        return self.ref_policy, self.ref_info

    def get_rollout_data(self, wandb_run=None) -> dict:
        if self.rollout_data is None:
            policy, _ = self.get_ref_policy(wandb_run)
            policy = policy.to(self.cfg.device)
            spec = get_task_spec(self.task)
            logger.info(
                f"[{self.task}] Rolling out reference BC for TRAK scoring "
                f"({self.cfg.trak_n_rollouts} episodes) …"
            )
            self.rollout_data = rollout_for_scoring(
                policy,
                env_name=spec["env_name"],
                n_episodes=self.cfg.trak_n_rollouts,
                seed=self.cfg.trak_rollout_seed,
            )
            policy.cpu()
            n_succ = self.rollout_data["successes"].sum()
            logger.info(
                f"[{self.task}] Rollout: {n_succ}/{self.cfg.trak_n_rollouts} "
                f"successes ({n_succ / self.cfg.trak_n_rollouts:.0%})"
            )
        return self.rollout_data

    def get_trak_results(self, wandb_run=None) -> dict:
        if self.trak_results is None:
            data = self.get_data()
            rollout = self.get_rollout_data(wandb_run)
            policy, _ = self.get_ref_policy(wandb_run)
            policy = policy.to(self.cfg.device)

            logger.info(f"[{self.task}] Computing TRAK influence scores …")
            scorer = TRAKScorer(
                policy,
                proj_dim=self.cfg.trak_proj_dim,
                lambda_reg=self.cfg.trak_lambda_reg,
                seed=self.cfg.seed,
            )
            self.trak_results = scorer.compute_demo_scores(
                train_obs=data["observations"],
                train_act=data["actions"],
                train_episode_ends=data["episode_ends"],
                test_obs=rollout["observations"],
                test_act=rollout["actions"],
                test_episode_ends=rollout["episode_ends"],
                test_successes=rollout["successes"],
                device=self.cfg.device,
            )
            policy.cpu()

            # Save scores
            score_dir = ROOT / "results" / "scores" / self.task
            score_dir.mkdir(parents=True, exist_ok=True)
            np.save(score_dir / "cupid_scores.npy", self.trak_results["cupid"])
            np.save(score_dir / "cupid_quality_scores.npy",
                     self.trak_results["cupid_quality"])
            logger.info(f"[{self.task}] TRAK scores saved to {score_dir}")
        return self.trak_results

    def get_vae(self, wandb_run=None) -> ConditionalVAE:
        if self.vae is None:
            data = self.get_data()
            out_path = str(ROOT / "results" / "checkpoints" / self.task / "vae.pt")
            logger.info(f"[{self.task}] Training VAE …")
            self.vae = train_vae(
                data=data,
                epochs=self.cfg.vae_epochs,
                lr=self.cfg.vae_lr,
                latent_dim=self.cfg.vae_latent_dim,
                hidden=self.cfg.vae_hidden,
                target_beta=self.cfg.vae_beta,
                anneal_epochs=self.cfg.vae_anneal_epochs,
                device_str=self.cfg.device,
                out_path=out_path,
                seed=self.cfg.seed,
                wandb_run=wandb_run,
            )
        return self.vae


# ===================================================================
# Per-method data processing
# ===================================================================

def _needs_trak(method: str) -> bool:
    return method.startswith("cupid") or method.startswith("stride") or method == "influence_reweight"


def _needs_vae(method: str) -> bool:
    return method.startswith("stride")


def process_data(
    cfg: ExperimentConfig,
    resources: TaskResources,
    wandb_run=None,
) -> dict:
    """Apply the method-specific data processing and return the processed dataset.

    Returns a dict with at least 'observations' and 'actions'.
    """
    data = resources.get_data()
    method = cfg.method

    # ---- Vanilla BC: no processing ----------------------------------------
    if method == "vanilla_bc":
        return data

    # ---- Gaussian filtering -----------------------------------------------
    if method.startswith("gaussian"):
        sigma = cfg.gaussian_sigma
        logger.info(f"[{cfg.run_name}] Gaussian smoothing σ={sigma}")
        smoothed = smooth_actions_per_episode(
            data["actions"], data["episode_ends"], sigma=sigma,
        )
        return {**data, "actions": smoothed}

    # ---- CUPID filtering --------------------------------------------------
    if method.startswith("cupid_quality"):
        trak = resources.get_trak_results(wandb_run)
        return filter_by_cupid_quality(
            data, trak["cupid_quality"], cfg.cupid_keep_ratio,
        )

    if method.startswith("cupid"):
        trak = resources.get_trak_results(wandb_run)
        return filter_by_cupid(data, trak["cupid"], cfg.cupid_keep_ratio)

    # ---- BC + Influence Reweighting ---------------------------------------
    if method == "influence_reweight":
        trak = resources.get_trak_results(wandb_run)
        demo_scores = trak["cupid"]  # (n_demos,)
        # Broadcast demo-level scores to per-transition weights
        n_trans = len(data["observations"])
        transition_scores = demo_scores_to_transition(
            demo_scores, data["episode_ends"], n_trans,
        )
        # Shift so min weight > 0, then normalise to mean 1
        shifted = transition_scores - transition_scores.min() + 1e-6
        weights = shifted / shifted.mean()
        return {**data, "_weights": weights.astype(np.float32)}

    # ---- STRIDE (full) ----------------------------------------------------
    if method == "stride":
        return _process_stride(cfg, resources, wandb_run,
                               use_influence=True, use_editor=True)

    # ---- STRIDE w/o influence (ablation) ----------------------------------
    if method == "stride_no_influence":
        return _process_stride(cfg, resources, wandb_run,
                               use_influence=False, use_editor=True)

    # ---- STRIDE with random edits (ablation) ------------------------------
    if method == "stride_random_edits":
        return _process_stride(cfg, resources, wandb_run,
                               use_influence=True, use_editor=False)

    raise ValueError(f"Unknown method: {method}")


def _process_stride(
    cfg: ExperimentConfig,
    resources: TaskResources,
    wandb_run,
    use_influence: bool,
    use_editor: bool,
) -> dict:
    """Run the STRIDE editing pipeline (or its ablations)."""
    data = resources.get_data()
    vae = resources.get_vae(wandb_run)

    _, _, train_idx, _ = make_datasets(data, seed=cfg.seed)

    # -- Determine influence scores -----------------------------------------
    if use_influence:
        trak = resources.get_trak_results(wandb_run)
        # Broadcast demo-level scores to per-transition
        influence_full = demo_scores_to_transition(
            trak["cupid"], data["episode_ends"], len(data["observations"]),
        )
    else:
        # Random influence (ablation)
        logger.info(f"[{cfg.run_name}] Using random influence scores (ablation)")
        rng = np.random.default_rng(cfg.seed + 999)
        influence_full = rng.standard_normal(
            len(data["observations"]),
        ).astype(np.float32)

    influence_train = influence_full[train_idx]

    # -- Random latent edits (ablation) -------------------------------------
    if not use_editor:
        logger.info(f"[{cfg.run_name}] Random latent edits (no trained editor)")
        obs_train = data["observations"][train_idx]
        act_train = data["actions"][train_idx]
        edited_actions = random_latent_edit(
            obs_train, act_train, vae,
            noise_std=cfg.random_latent_std,
            blend_alpha=cfg.blend_alpha,
            device_str=cfg.device,
            seed=cfg.seed,
        )
        return {
            "observations": obs_train,
            "actions": edited_actions,
            "episode_ends": _compute_train_episode_ends(
                data["episode_ends"], train_idx,
            ),
        }

    # -- Train DPO editor ---------------------------------------------------
    editor_out = str(
        ROOT / "results" / "checkpoints" / cfg.task / f"{cfg.method}_editor.pt"
    )
    logger.info(f"[{cfg.run_name}] Training DPO editor …")
    editor, _ = train_editor_dpo(
        data=data,
        vae=vae,
        influence_scores_raw=influence_train,
        epochs=cfg.editor_epochs,
        lr=cfg.editor_lr,
        beta=cfg.editor_beta_dpo,
        lambda_reg=cfg.editor_lambda_reg,
        lambda_cos=cfg.editor_lambda_cos,
        k_neighbors=cfg.editor_k_neighbors,
        hidden=cfg.editor_hidden,
        device_str=cfg.device,
        out_path=editor_out,
        seed=cfg.seed,
        wandb_run=wandb_run,
    )

    # -- Apply STRIDE edits -------------------------------------------------
    logger.info(f"[{cfg.run_name}] Applying STRIDE edits …")
    edited_data = apply_stride(
        data=data,
        train_idx=train_idx,
        vae=vae,
        editor=editor,
        edit_scale=cfg.edit_scale,
        blend_alpha=cfg.blend_alpha,
        n_aug=cfg.n_aug,
        aug_noise_std=cfg.aug_noise_std,
        device_str=cfg.device,
        seed=cfg.seed,
    )
    return edited_data


def _compute_train_episode_ends(
    full_episode_ends: list[int],
    train_idx: np.ndarray,
) -> list[int]:
    """Compute episode_ends for the train-split subset.

    Since the train split is a random permutation of transitions, the
    episode structure is lost. We produce a single-episode wrapper
    containing all transitions (used only for downstream compatibility).
    """
    return [len(train_idx)]


# ===================================================================
# Single experiment run
# ===================================================================

def run_single(
    cfg: ExperimentConfig,
    resources: TaskResources,
    wandb_run=None,
) -> dict:
    """Execute a single (task, method, seed) experiment.

    Returns
    -------
    dict with keys: config, train_info, eval_results, timing.
    """
    result: dict = {
        "config": cfg.to_dict(),
        "train_info": {},
        "eval_results": {},
        "timing": {},
    }
    logger.info(f"\n{'='*60}\n  Running: {cfg.run_name}\n  {cfg.description}\n{'='*60}")

    # ---- Process data -----------------------------------------------------
    t0 = time.time()
    processed_data = process_data(cfg, resources, wandb_run)
    result["timing"]["data_processing"] = time.time() - t0

    # ---- Train final BC ---------------------------------------------------
    t0 = time.time()
    out_dir = str(ROOT / "results" / "checkpoints" / cfg.task)
    # Extract per-sample weights if provided by the data processing step
    sample_weights = processed_data.pop("_weights", None)

    # Success-rate-over-training callback: periodically evaluate in env
    spec = get_task_spec(cfg.task)

    def _eval_callback(policy_in_training, epoch):
        """Quick eval (3 episodes) to track success rate during training."""
        policy_in_training.eval()
        try:
            quick_results = evaluate_policy(
                policy_in_training,
                env_name=spec["env_name"],
                n_episodes=3,
                seed=cfg.seed + 20000 + epoch,
                render=False,
                max_episode_steps=cfg.max_episode_steps,
                wandb_run=None,
                log_prefix=f"train_eval/{cfg.method}",
                verbose=False,
            )
            sr = quick_results["success_rate"]
            mr = quick_results["mean_reward"]
            logger.info(
                f"[{cfg.run_name}] epoch {epoch} eval: "
                f"reward={mr:.1f}  success={sr:.1%}"
            )
            if wandb_run is not None:
                wandb_run.log({
                    f"{cfg.method}/train_eval_success_rate": sr,
                    f"{cfg.method}/train_eval_reward": mr,
                    f"{cfg.method}/train_eval_epoch": epoch,
                })
        except Exception as exc:
            logger.warning(f"[{cfg.run_name}] eval callback failed: {exc}")

    policy, train_info = train_bc(
        processed_data,
        epochs=cfg.bc_epochs,
        lr=cfg.bc_lr,
        batch_size=cfg.bc_batch_size,
        hidden=cfg.bc_hidden,
        weight_decay=cfg.bc_weight_decay,
        grad_clip=cfg.bc_grad_clip,
        device_str=cfg.device,
        out_dir=out_dir,
        run_name=cfg.method,
        seed=cfg.seed,
        weights=sample_weights,
        wandb_run=wandb_run,
        eval_callback=_eval_callback,
        eval_every=50,
    )
    result["timing"]["bc_training"] = time.time() - t0
    result["train_info"] = {
        "best_val_loss": train_info["best_val_loss"],
        "best_epoch": train_info["best_epoch"],
        "ckpt_path": train_info["ckpt_path"],
    }

    # ---- Evaluate ---------------------------------------------------------
    t0 = time.time()
    video_dir = str(
        ROOT / "results" / "videos" / cfg.task / cfg.method / f"seed{cfg.seed}"
    )
    policy = policy.to(cfg.device)
    eval_results = evaluate_policy(
        policy,
        env_name=spec["env_name"],
        n_episodes=cfg.n_eval_episodes,
        seed=cfg.seed + 10000,
        render=cfg.render_videos,
        video_dir=video_dir,
        max_episode_steps=cfg.max_episode_steps,
        wandb_run=wandb_run,
        log_prefix=f"eval/{cfg.method}",
    )
    policy.cpu()
    result["timing"]["evaluation"] = time.time() - t0

    # Remove per-episode video paths from JSON (not serialisable paths)
    result["eval_results"] = {
        k: v for k, v in eval_results.items() if k != "per_episode"
    }
    result["eval_results"]["per_episode"] = [
        {k: v for k, v in ep.items() if k != "video_path"}
        for ep in eval_results["per_episode"]
    ]

    result["timing"]["total"] = sum(result["timing"].values())

    logger.info(
        f"[{cfg.run_name}] Done.  "
        f"reward={eval_results['mean_reward']:.2f}±{eval_results['std_reward']:.2f}  "
        f"success={eval_results['success_rate']:.1%}  "
        f"total_time={result['timing']['total']:.0f}s"
    )
    return result


# ===================================================================
# Task-level orchestration
# ===================================================================

def run_task(
    task: str,
    configs: list[ExperimentConfig],
    use_wandb: bool = True,
) -> list[dict]:
    """Run all experiments for a single task."""
    if not configs:
        return []

    # Use the first config as a template for shared resources
    template = configs[0]
    resources = TaskResources(task, template)

    results: list[dict] = []

    for cfg in configs:
        wandb_run = None
        if use_wandb:
            try:
                import wandb

                wandb_run = wandb.init(
                    entity="stride-cs229",
                    project="stride",
                    group=task,
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                    reinit=True,
                    tags=[task, cfg.method, f"seed{cfg.seed}"],
                )
            except Exception as e:
                logger.warning(f"wandb init failed: {e}; proceeding without wandb")
                wandb_run = None

        try:
            result = run_single(cfg, resources, wandb_run)
            results.append(result)

            # Save incremental result
            out_path = (
                ROOT / "results" / "experiment_results"
                / f"{cfg.run_name}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        except Exception as e:
            logger.error(f"[{cfg.run_name}] FAILED: {e}", exc_info=True)
            results.append({
                "config": cfg.to_dict(),
                "error": str(e),
            })
        finally:
            if wandb_run is not None:
                try:
                    wandb_run.finish()
                except Exception:
                    pass

    return results


# ===================================================================
# CLI
# ===================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run STRIDE experiments on Adroit manipulation tasks.",
    )
    p.add_argument(
        "--task", choices=[*TASKS, "all"], default="all",
        help="Which task to run (default: all).",
    )
    p.add_argument(
        "--method", default="all",
        help=(
            "Which method(s) to run.  Accepts: all, vanilla_bc, gaussian_25, "
            "gaussian_50, gaussian_75, cupid_25, cupid_50, cupid_75, "
            "cupid_quality_25, cupid_quality_50, cupid_quality_75, "
            "stride, stride_no_influence, stride_random_edits, "
            "influence_reweight.  "
            "Comma-separated for multiple."
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="Base random seed.")
    p.add_argument(
        "--n-trials", type=int, default=10,
        help="Number of independent trials (seeds). Seeds will be base..base+n-1.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    p.add_argument("--no-video", action="store_true", help="Disable video rendering.")
    p.add_argument(
        "--n-eval-episodes", type=int, default=20,
        help="Number of evaluation episodes.",
    )
    p.add_argument(
        "--trak-n-rollouts", type=int, default=100,
        help="Number of rollout episodes for TRAK scoring.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    tasks = list(TASKS) if args.task == "all" else [args.task]
    seeds = tuple(args.seed + i for i in range(args.n_trials))

    # Build configs
    requested_methods = (
        args.method.split(",") if args.method != "all"
        else None
    )

    all_configs = build_all_configs(
        tasks=tuple(tasks),
        seeds=seeds,
        device=args.device,
    )

    # Apply CLI overrides
    for cfg in all_configs:
        cfg.n_eval_episodes = args.n_eval_episodes
        cfg.trak_n_rollouts = args.trak_n_rollouts
        if args.no_video:
            cfg.render_videos = False

    # Filter by method if specified
    if requested_methods is not None:
        all_configs = [c for c in all_configs if c.method in requested_methods]

    if not all_configs:
        logger.error("No matching experiments found. Check --method and --task.")
        sys.exit(1)

    logger.info(
        f"Running {len(all_configs)} experiments across "
        f"{len(tasks)} task(s): {', '.join(tasks)}  "
        f"({len(seeds)} trial(s) per method, seeds={seeds})"
    )

    all_results: list[dict] = []
    for task in tasks:
        task_configs = [c for c in all_configs if c.task == task]
        if task_configs:
            results = run_task(task, task_configs, use_wandb=not args.no_wandb)
            all_results.extend(results)

    # Save aggregate results
    agg_path = ROOT / "results" / "all_results.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    logger.info(f"All results saved to {agg_path}")


if __name__ == "__main__":
    main()
