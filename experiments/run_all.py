"""Experiment orchestrator: run all five methods on pen-human-v2 and compare.

Pipeline
--------
1.  Load D4RL/pen/human-v2 data
2.  Train initial BC policy (used for TRAK influence computation)
3.  Train conditional VAE
4.  Train latent editor (STRIDE)
5.  Produce edited dataset D'
6.  Train STRIDE BC policy on D'
7.  Train Gaussian Filter baseline
8.  Train Influence-Weighted BC baseline
9.  Train Random Latent editing baseline
10. Evaluate all 5 methods in AdroitHandPen-v1
11. Print results table and save to results/results.json

Usage
-----
    python experiments/run_all.py                  # full experiment
    python experiments/run_all.py --smoke-test      # 2 epochs per model
    python experiments/run_all.py --device cuda
    python experiments/run_all.py --skip-eval       # skip env rollouts (CI)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from project root without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from stride.data.loader import load_pen_human, make_datasets
from stride.models.policy import BCPolicy
from stride.models.vae import ConditionalVAE
from stride.models.editor import LatentEditor
from stride.training.train_bc import train_bc
from stride.training.train_vae import train_vae
from stride.training.train_editor_dpo import train_editor_dpo
from stride.editing.edit import apply_stride
from stride.baselines.vanilla_bc import run_vanilla_bc
from stride.baselines.gaussian_filter import run_gaussian_filter_bc
from stride.baselines.influence_weighted_bc import run_influence_weighted_bc
from stride.baselines.random_latent import run_random_latent_bc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Run full STRIDE experiment")
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
    print("STRIDE Experiment Results — AdroitHandPen-v1 (pen-human-v2)")
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

def run_trial(trial_idx: int, args: argparse.Namespace):
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
    data = load_pen_human()
    train_ds, _, train_idx, _ = make_datasets(data, seed=seed)

    # 1. Train initial BC policy
    print(f"\n[Trial {trial_idx+1}] Training initial BC policy …")
    bc_ckpt = str(ckpt / "bc_policy.pt")
    bc_policy = train_bc(data=data, epochs=epochs_bc, lr=3e-4, out_path=bc_ckpt, **kw_common)

    # 2. Train VAE
    print(f"\n[Trial {trial_idx+1}] Training VAE …")
    vae_ckpt = str(ckpt / "vae.pt")
    vae = train_vae(data=data, epochs=epochs_vae, lr=3e-4, latent_dim=16,
                    target_beta=0.5, anneal_epochs=50 if not args.smoke_test else 1,
                    out_path=vae_ckpt, **kw_common)

    # 3. Train DPO latent editor
    print(f"\n[Trial {trial_idx+1}] Training DPO latent editor (STRIDE) …")
    editor_ckpt = str(ckpt / "editor.pt")
    editor, influence_raw = train_editor_dpo(data=data, vae=vae, bc_policy=bc_policy,
                                              epochs=epochs_ed, lr=3e-4, beta=2.0,
                                              lambda_reg=0.3, lambda_cos=0.2,
                                              out_path=editor_ckpt, **kw_common)

    # 4. Apply STRIDE edits
    print(f"\n[Trial {trial_idx+1}] Applying STRIDE edits + augmentation …")
    edited_data = apply_stride(data=data, train_idx=train_idx, vae=vae, editor=editor,
                               edit_scale=0.6, blend_alpha=0.35, n_aug=4,
                               aug_noise_std=0.07, device_str=args.device, seed=seed)

    # 5. Train BC on STRIDE data
    print(f"\n[Trial {trial_idx+1}] Training BC on STRIDE-edited dataset …")
    stride_bc_ckpt = str(ckpt / "stride_bc.pt")
    train_bc(data=edited_data, val_data=data, epochs=epochs_bc, lr=3e-4,
             out_path=stride_bc_ckpt, **kw_common)

    # 6. Baselines
    print(f"\n[Trial {trial_idx+1}] Training baselines …")
    gf_ckpt = str(ckpt / "gaussian_filter_bc.pt")
    run_gaussian_filter_bc(data=data, sigma=2.0, epochs=epochs_bc, out_path=gf_ckpt, **kw_common)

    iw_ckpt = str(ckpt / "influence_weighted_bc.pt")
    run_influence_weighted_bc(data=data, bc_policy=bc_policy, epochs=epochs_bc, out_path=iw_ckpt, **kw_common)

    rl_ckpt = str(ckpt / "random_latent_bc.pt")
    run_random_latent_bc(data=data, vae=vae, noise_std=0.1, epochs=epochs_bc, out_path=rl_ckpt, **kw_common)

    # 7. Evaluate
    trial_results = {}
    if not args.skip_eval:
        from stride.eval.evaluate import evaluate_from_checkpoint
        method_ckpts = {
            "Vanilla BC": bc_ckpt,
            "Gaussian Filter BC": gf_ckpt,
            "Influence-Weighted BC": iw_ckpt,
            "Random Latent BC": rl_ckpt,
            "STRIDE": stride_bc_ckpt,
        }
        for name, ckpt_p in method_ckpts.items():
            print(f"  Evaluating {name} (Trial {trial_idx+1}) …")
            stats = evaluate_from_checkpoint(ckpt_p, n_episodes=n_eval, seed=seed, device_str=args.device)
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
    aggregated: dict[str, dict] = {"_metadata": {"num_trials": args.num_trials}}

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

    results_path = res / "results.json"
    with open(results_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to {results_path}")

    _print_results_table(aggregated)


if __name__ == "__main__":
    main()
