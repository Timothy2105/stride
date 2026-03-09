"""Focused benchmark: STRIDE vs Vanilla BC, Gaussian, and Cupid reimplementation.

Usage
-----
python experiments/benchmark.py --task pen --smoke-test
python experiments/benchmark.py --task hammer --device cuda --num-trials 3

Supported tasks: pen, hammer, relocate, door
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_all import run_trial, _print_results_table
from stride.data.loader import get_task_spec


def _parse_args():
    p = argparse.ArgumentParser(description="Run focused 4-way STRIDE benchmark")
    p.add_argument(
        "--task",
        default="pen",
        choices=["pen", "hammer", "relocate", "door"],
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs-bc", type=int, default=100)
    p.add_argument("--epochs-vae", type=int, default=200)
    p.add_argument("--epochs-editor", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--stride-params", default=None,
                   help="Path to JSON of tuned STRIDE params (from tune_stride.py)")
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", default="stride-v3")
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


def _log_wandb(aggregated: dict, args, stride_params: dict | None = None) -> None:
    """Log benchmark results to W&B project stride-v3."""
    try:
        import wandb
    except ImportError:
        print("[benchmark] wandb not installed — skipping W&B logging")
        return

    meta = aggregated.get("_metadata", {})
    run_name = args.wandb_run_name or f"benchmark-{meta.get('task', args.task)}"
    config = {
        "task": meta.get("task"),
        "env_name": meta.get("env_name"),
        "dataset_id": meta.get("dataset_id"),
        "num_trials": meta.get("num_trials"),
        "epochs_bc": args.epochs_bc,
        "epochs_vae": args.epochs_vae,
        "epochs_editor": args.epochs_editor,
    }
    if stride_params:
        config["stride_params"] = stride_params

    wandb.init(project=args.wandb_project, name=run_name, config=config)

    methods = [m for m in aggregated if not m.startswith("_")]
    metrics: dict = {}
    for m in methods:
        s = aggregated[m]
        key = m.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        metrics[f"{key}/mean_reward"]  = s["mean_reward"]
        metrics[f"{key}/success_rate"] = s["success_rate"]
        metrics[f"{key}/std_reward"]   = s.get("std_cross_trial_reward", 0.0)
    wandb.log(metrics)

    table = wandb.Table(
        columns=["Method", "Mean Reward", "Std Reward", "Success Rate"],
        data=[
            [m, aggregated[m]["mean_reward"],
             aggregated[m].get("std_cross_trial_reward", 0.0),
             aggregated[m]["success_rate"]]
            for m in methods
        ],
    )
    wandb.log({"results": table})
    wandb.finish()
    print(f"[benchmark] Logged to W&B project '{args.wandb_project}' run '{run_name}'")


def main():
    import numpy as np

    args = _parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    stride_params = None
    if args.stride_params:
        with open(args.stride_params) as f:
            stride_params = json.load(f)
        print(f"[benchmark] Loaded STRIDE params from {args.stride_params}")
        for k, v in stride_params.items():
            print(f"  {k:<20} {v}")

    all_trials_data = []
    for t in range(args.num_trials):
        all_trials_data.append(run_trial(t, args, stride_params=stride_params))

    if args.skip_eval:
        print("All trials completed (eval skipped).")
        return

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
            "raw_trials": all_trials_data,
        }

    out_path = results_dir / f"benchmark_{spec['task']}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    alias_path = results_dir / "benchmark_results.json"
    with open(alias_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Saved benchmark results to {out_path}")
    _print_results_table(aggregated)
    if args.wandb:
        _log_wandb(aggregated, args, stride_params)


if __name__ == "__main__":
    main()
