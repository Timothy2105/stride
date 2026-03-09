"""Optuna hyperparameter search for STRIDE.

Tunes full practical STRIDE knobs: model architectures, optimisation settings,
influence settings, editing parameters, split settings, and stage epochs.

Best params are saved to results/tune_<task>_best_params.json and can be
loaded into benchmark.py via --stride-params.

Usage
-----
    python experiments/tune_stride.py --task pen --trials 30 --device mps
    python experiments/tune_stride.py --task pen --trials 30 --no-wandb
    python experiments/tune_stride.py --task pen --trials 50 --storage sqlite:///tune.db
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
import optuna.logging as optuna_logging
from stride.data.loader import load_task_human, get_task_spec, make_datasets
from stride.training.train_bc import train_bc
from stride.training.train_vae import train_vae
from stride.training.train_editor_dpo import train_editor_dpo
from stride.editing.edit import apply_stride
from stride.eval.evaluate import evaluate_from_checkpoint


def _parse_args():
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for STRIDE")
    p.add_argument("--task", required=True, choices=["pen", "hammer", "relocate", "door"])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--epochs-bc", type=int, default=100)
    p.add_argument("--epochs-vae", type=int, default=200)
    p.add_argument("--epochs-editor", type=int, default=100)
    p.add_argument("--lr-bc", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", default=None)
    p.add_argument("--storage", default=None, help="Optuna DB URI, e.g. sqlite:///tune.db")
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", default="stride-v3-tune")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def _sample_params(trial: optuna.Trial) -> dict:
    """Sample STRIDE hyperparameters across all major pipeline components."""
    def _hidden(name: str) -> tuple[int, ...]:
        depth = trial.suggest_int(f"{name}_depth", 1, 3)
        width = trial.suggest_categorical(f"{name}_width", [128, 256, 384, 512])
        return tuple([width] * depth)

    return {
        # Stage epochs (BC is fixed via args)
        "epochs_vae": trial.suggest_int("epochs_vae", 80, 300, step=20),
        "epochs_editor": trial.suggest_int("epochs_editor", 60, 200, step=20),
        # VAE architecture / training
        "latent_dim": trial.suggest_categorical("latent_dim", [8, 16, 24, 32]),
        "vae_hidden": _hidden("vae_hidden"),
        "vae_beta": trial.suggest_float("vae_beta", 0.05, 1.0),
        "lr_vae": trial.suggest_float("lr_vae", 1e-4, 1e-3, log=True),
        "anneal_epochs": trial.suggest_int("anneal_epochs", 10, 80, step=10),
        "vae_weight_decay": trial.suggest_float("vae_weight_decay", 1e-6, 1e-3, log=True),
        "vae_grad_clip": trial.suggest_float("vae_grad_clip", 0.5, 2.0),
        # DPO editor
        "editor_hidden": _hidden("editor_hidden"),
        "beta_dpo": trial.suggest_float("beta_dpo", 0.5, 5.0),
        "lambda_reg": trial.suggest_float("lambda_reg", 0.0, 0.5),
        "lambda_cos": trial.suggest_float("lambda_cos", 0.0, 0.5),
        "k_neighbors": trial.suggest_int("k_neighbors", 5, 20),
        "proj_dim": trial.suggest_categorical("proj_dim", [256, 512, 1024]),
        "lr_editor": trial.suggest_float("lr_editor", 1e-4, 1e-3, log=True),
        "editor_weight_decay": trial.suggest_float("editor_weight_decay", 1e-6, 1e-3, log=True),
        "editor_grad_clip": trial.suggest_float("editor_grad_clip", 0.5, 2.0),
        # Editing step
        "edit_scale": trial.suggest_float("edit_scale", 0.1, 1.5),
        "blend_alpha": trial.suggest_float("blend_alpha", 0.1, 0.9),
        "n_aug": trial.suggest_int("n_aug", 0, 8),
        "aug_noise_std": trial.suggest_float("aug_noise_std", 0.01,  0.2, log=True),
    }


def make_objective(args, data, spec, init_bc, wandb_active: bool):
    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = _sample_params(trial)
        trial_seed = args.seed

        try:
            with tempfile.TemporaryDirectory() as tmp:
                # 1. Train VAE with tuned architecture + lr
                vae = train_vae(
                    data=data,
                    epochs=params["epochs_vae"],
                    lr=params["lr_vae"],
                    batch_size=args.batch_size,
                    latent_dim=params["latent_dim"],
                    hidden=params["vae_hidden"],
                    target_beta=params["vae_beta"],
                    anneal_epochs=min(params["anneal_epochs"], params["epochs_vae"]),
                    weight_decay=params["vae_weight_decay"],
                    grad_clip=params["vae_grad_clip"],
                    train_frac=args.train_frac,
                    num_workers=args.num_workers,
                    device_str=args.device,
                    out_path=str(Path(tmp) / "vae.pt"),
                    seed=trial_seed,
                    verbose=False,
                )

                # 2. Train DPO editor with tuned loss params
                editor, _ = train_editor_dpo(
                    data=data,
                    vae=vae,
                    bc_policy=init_bc,
                    epochs=params["epochs_editor"],
                    lr=params["lr_editor"],
                    batch_size=args.batch_size,
                    beta=params["beta_dpo"],
                    lambda_reg=params["lambda_reg"],
                    lambda_cos=params["lambda_cos"],
                    k_neighbors=params["k_neighbors"],
                    proj_dim=params["proj_dim"],
                    hidden=params["editor_hidden"],
                    weight_decay=params["editor_weight_decay"],
                    grad_clip=params["editor_grad_clip"],
                    train_frac=args.train_frac,
                    num_workers=args.num_workers,
                    device_str=args.device,
                    out_path=str(Path(tmp) / "editor.pt"),
                    seed=trial_seed,
                    verbose=False,
                )

                # 3. Apply STRIDE editing
                _, _, train_idx, _ = make_datasets(
                    data,
                    train_frac=args.train_frac,
                    seed=trial_seed,
                )
                edited_data = apply_stride(
                    data=data,
                    train_idx=train_idx,
                    vae=vae,
                    editor=editor,
                    edit_scale=params["edit_scale"],
                    blend_alpha=params["blend_alpha"],
                    n_aug=params["n_aug"],
                    aug_noise_std=params["aug_noise_std"],
                    batch_size=args.batch_size,
                    device_str=args.device,
                    seed=trial_seed,
                    verbose=False,
                )

                # 4. Train final BC on edited dataset
                bc_out = str(Path(tmp) / "stride_bc.pt")
                train_bc(
                    data=edited_data,
                    val_data=data,
                    epochs=args.epochs_bc,
                    lr=args.lr_bc,
                    batch_size=args.batch_size,
                    train_frac=args.train_frac,
                    num_workers=args.num_workers,
                    device_str=args.device,
                    out_path=bc_out,
                    seed=trial_seed,
                    verbose=False,
                )

                # 5. Evaluate
                stats = evaluate_from_checkpoint(
                    bc_out,
                    n_episodes=args.n_eval_episodes,
                    env_name=spec["env_name"],
                    seed=trial_seed,
                    device_str=args.device,
                )

        except Exception as e:
            print(f"  [Trial {trial.number:3d}] FAILED: {e}")
            return 0.0

        success = stats["success_rate"]
        reward  = stats["mean_reward"]
        elapsed = time.time() - t0

        param_str = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in params.items()
        )
        print(
            f"  [Trial {trial.number:3d}]  success={success:.2%}  "
            f"reward={reward:.1f}  ({elapsed:.0f}s)\n"
            f"    {param_str}"
        )

        if wandb_active:
            try:
                import wandb
                wandb.log(
                    {"success_rate": success, "mean_reward": reward,
                     "elapsed_s": elapsed, **params},
                    step=trial.number,
                )
            except Exception:
                pass

        return success

    return objective


def main():
    args = _parse_args()
    optuna_logging.set_verbosity(optuna_logging.WARNING)

    spec = get_task_spec(args.task)
    data = load_task_human(args.task)

    print(f"\n[tune] Task: {spec['task']}  env: {spec['env_name']}")
    print("[tune] Pre-training shared initial BC once (fixed across all trials) …")

    init_bc = train_bc(
        data=data,
        epochs=args.epochs_bc,
        lr=args.lr_bc,
        batch_size=args.batch_size,
        train_frac=args.train_frac,
        num_workers=args.num_workers,
        device_str=args.device,
        out_path="checkpoints/tune_init_bc.pt",
        seed=args.seed,
        verbose=True,
    )

    study_name = args.study_name or f"stride-{args.task}"

    # W&B setup — one run for the whole study, trials logged as steps
    wandb_active = False
    if args.wandb:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=study_name,
                config={
                    "task": args.task,
                    "env_name": spec["env_name"],
                    "n_trials": args.trials,
                    "fixed/epochs_bc": args.epochs_bc,
                    "fixed/lr_bc": args.lr_bc,
                    "fixed/batch_size": args.batch_size,
                    "fixed/train_frac": args.train_frac,
                    "fixed/num_workers": args.num_workers,
                    "fixed/n_eval_episodes": args.n_eval_episodes,
                    "seed": args.seed,
                },
            )
            wandb_active = True
            print(f"[tune] W&B: {run.get_url()}")
        except Exception as e:
            print(f"[tune] W&B init failed ({e}) — continuing without logging")

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    print(f"[tune] Starting {args.trials} trials …\n")
    study.optimize(
        make_objective(args, data, spec, init_bc, wandb_active),
        n_trials=args.trials,
    )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best trial #{best.number}  success_rate={best.value:.3f}")
    print(f"{'='*60}")
    for k, v in best.params.items():
        print(f"  {k:<20} {v}")

    # Save best params as JSON for use with benchmark --stride-params
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / f"tune_{args.task}_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"\nBest params saved → {params_path}")

    if wandb_active:
        try:
            import wandb
            wandb.log({
                "best_trial": best.number,
                "best_success_rate": best.value,
                **{f"best/{k}": v for k, v in best.params.items()},
            })
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
