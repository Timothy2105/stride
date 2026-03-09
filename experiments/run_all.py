"""Experiment orchestrator: run STRIDE vs requested baselines on Adroit tasks.

Pipeline
--------
1.  Load selected D4RL/*/human-v2 data
2.  Train initial BC policy (used for TRAK influence computation)
3.  Train Gaussian Filter baseline
4.  Train Cupid (re-implemented) baseline
5.  Train STRIDE v2 direct baseline
10. Evaluate all 4 methods in the matching Adroit env
11. Print results table and save to results/results.json

Usage
-----
    python experiments/run_all.py                  # full experiment
    python experiments/run_all.py --task hammer    # run a specific task
    python experiments/run_all.py --smoke-test      # 2 epochs per model
    python experiments/run_all.py --device cuda
    python experiments/run_all.py --skip-eval       # skip env rollouts (CI)
"""

from __future__ import annotations

import argparse
import json
import sys
import inspect
from pathlib import Path

# Allow running from project root without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stride.data.loader import load_task_human, get_task_spec
from stride.training.train_bc import train_bc
from stride.baselines.gaussian_filter import run_gaussian_filter_bc
from stride.baselines.cupid_reimpl import run_cupid_reimpl_bc
from stride.baselines.stride_v2_direct import run_stride_v2_direct_bc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Run full STRIDE experiment")
    p.add_argument(
        "--task",
        default="pen",
        choices=["pen", "hammer", "relocate", "door"],
        help="Adroit task key.",
    )
    p.add_argument("--device", default="cpu",
                   help="Torch device: 'cpu' or 'cuda'")
    p.add_argument("--epochs-bc", type=int, default=100,
                   help="BC training epochs")
    p.add_argument("--epochs-vae", type=int, default=200,
                   help="VAE training epochs")
    p.add_argument("--epochs-editor", type=int, default=100,
                   help="Editor training epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-eval-episodes", type=int, default=20,
                   help="Rollout episodes for evaluation")
    p.add_argument("--num-trials", type=int, default=1,
                   help="Number of multi-seed trials to average over")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", default="checkpoints",
                   help="Directory for model checkpoints")
    p.add_argument("--results-dir", default="results",
                   help="Directory for result files")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 2 epochs per model for a quick sanity check")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip environment rollout evaluation (for unit testing)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_results_table(results: dict[str, dict]):
    header = f"{'Method':<35} {'Mean Reward':>18} {'Success%':>12}"
    print("\n" + "=" * len(header))
    meta = results.get("_metadata", {})
    env_name = meta.get("env_name", "Adroit task")
    dataset_id = meta.get("dataset_id", "D4RL/*/human-v2")
    print(f"STRIDE Experiment Results — {env_name} ({dataset_id})")
    print(f"Aggregated over {results.get('_metadata', {}).get('num_trials', 1)} trials")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, stats in results.items():
        if name.startswith("_"): continue
        if "mean_reward" in stats:
            m_rew = stats['mean_reward']
            s_rew = stats.get('std_cross_trial_reward', 0.0)
            m_suc = stats['success_rate'] * 100
            s_suc = stats.get('std_cross_trial_success', 0.0) * 100
            print(
                f"{name:<35} {m_rew:>8.2f} ± {s_rew:>6.2f} "
                f"{m_suc:>7.1f} ± {s_suc:>4.1f}%"
            )
        else:
            print(f"{name:<35} {'N/A (eval skipped)':>32}")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_trial(trial_idx: int, args: argparse.Namespace, stride_params: dict | None = None):
    """Run a single experiment trial with a specific seed."""
    seed = args.seed + trial_idx
    ckpt = Path(args.ckpt_dir) / f"trial_{trial_idx}"
    ckpt.mkdir(parents=True, exist_ok=True)

    # Shorten epochs in smoke-test mode
    epochs_bc  = 2 if args.smoke_test else args.epochs_bc
    epochs_vae = 2 if args.smoke_test else args.epochs_vae
    epochs_ed  = 2 if args.smoke_test else args.epochs_editor
    n_eval     = 2 if args.smoke_test else args.n_eval_episodes

    kw_common = dict(
        device_str=args.device,
        batch_size=args.batch_size,
        seed=seed,
        verbose=True,
    )

    print(f"\n[Trial {trial_idx+1}/{args.num_trials}] Starting with seed={seed} …")
    spec = get_task_spec(args.task)
    data = load_task_human(args.task)

    # 1. Train initial BC policy
    print(f"\n[Trial {trial_idx+1}] Training initial BC policy …")
    bc_ckpt = str(ckpt / "bc_policy.pt")
    bc_policy = train_bc(data=data, epochs=epochs_bc, lr=3e-4, out_path=bc_ckpt, **kw_common)

    # 2. Requested baselines + STRIDE v2 direct
    print(f"\n[Trial {trial_idx+1}] Training baselines …")
    gf_ckpt = str(ckpt / "gaussian_filter_bc.pt")
    run_gaussian_filter_bc(data=data, sigma=2.0, epochs=epochs_bc, out_path=gf_ckpt, **kw_common)

    cupid_ckpt = str(ckpt / "cupid_reimpl_bc.pt")
    run_cupid_reimpl_bc(
        data=data,
        bc_policy=bc_policy,
        keep_ratio=0.7,
        epochs=epochs_bc,
        out_path=cupid_ckpt,
        **kw_common,
    )

    stride_bc_ckpt = str(ckpt / "stride_v2_direct_bc.pt")
    stride_cfg = {
        "epochs_bc": epochs_bc,
        "epochs_vae": epochs_vae,
        "epochs_editor": epochs_ed,
        "batch_size": args.batch_size,
        "seed": seed,
        "device_str": args.device,
        "verbose": True,
        "out_path": stride_bc_ckpt,
    }
    if stride_params:
        valid_stride_keys = set(inspect.signature(run_stride_v2_direct_bc).parameters.keys())
        stride_cfg.update({k: v for k, v in stride_params.items() if k in valid_stride_keys})

    run_stride_v2_direct_bc(
        data=data,
        bc_policy=bc_policy,
        **stride_cfg,
    )

    # 7. Evaluate
    trial_results = {}
    if not args.skip_eval:
        from stride.eval.evaluate import evaluate_from_checkpoint
        method_ckpts = {
            "Vanilla BC": bc_ckpt,
            "Gaussian Filter BC": gf_ckpt,
            "Cupid (Re-implemented)": cupid_ckpt,
            "STRIDE v2 Direct": stride_bc_ckpt,
        }
        for name, ckpt_p in method_ckpts.items():
            print(f"  Evaluating {name} (Trial {trial_idx+1}) …")
            stats = evaluate_from_checkpoint(
                ckpt_p,
                n_episodes=n_eval,
                env_name=spec["env_name"],
                seed=seed,
                device_str=args.device,
            )
            trial_results[name] = stats
    return trial_results


def main():
    args = _parse_args()
    res  = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    all_trials_data = []
    for t in range(args.num_trials):
        trial_res = run_trial(t, args)
        all_trials_data.append(trial_res)

    if args.skip_eval:
        print("\nAll trials completed (eval skipped).")
        return

    # Aggregate results
    methods = list(all_trials_data[0].keys())
    spec = get_task_spec(args.task)
    aggregated: dict[str, dict] = {
        "_metadata": {
            "num_trials": args.num_trials,
            "task": spec["task"],
            "dataset_id": spec["dataset_id"],
            "env_name": spec["env_name"],
        }
    }

    for m in methods:
        rewards = [t[m]["mean_reward"] for t in all_trials_data]
        successes = [t[m]["success_rate"] for t in all_trials_data]

        aggregated[m] = {
            "mean_reward": float(np.mean(rewards)),
            "std_cross_trial_reward": float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "std_cross_trial_success": float(np.std(successes)),
            "raw_trials": all_trials_data
        }

    # Save both task-specific and default filenames for compatibility.
    task_results_path = res / f"{spec['task']}_results.json"
    with open(task_results_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    results_path = res / "results.json"
    with open(results_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to {task_results_path}")
    print(f"Default alias updated at {results_path}")

    _print_results_table(aggregated)


if __name__ == "__main__":
    main()
