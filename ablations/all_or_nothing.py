#!/usr/bin/env python3
"""All-or-nothing ablation script for STRIDE.

Toggles various components and loss terms on/off to measure their 
individual contribution to the overall performance.
"""

import argparse
import sys
import os
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stride.ablation.config import AblationConfig
from stride.ablation.orchestrator import run_sweep
from stride.ablation.plotting import plot_bar_ablation
from stride.ablation.runner import load_all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Quick run with 2 epochs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="D4RL/pen/human-v2")
    parser.add_argument("--num-trials", type=int, default=10, help="Number of trials to average over")
    args = parser.parse_args()

    # Base config starts with vanilla STRIDE (no LSPO/Ranking DPO by default)
    base_cfg = AblationConfig(
        device=args.device,
        task=args.task,
        smoke_test=args.smoke_test,
    )

    results_dir = "ablations/results_aon"
    plots_dir = "ablations/plots"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Define the "all or nothing" set
    print("\n=== Running All-or-Nothing Ablations ===")
    
    sweep = [
        # --- Baselines ---
        base_cfg.clone(name="Vanilla BC", use_editor=False, group="aon"),
        base_cfg.clone(name="Gaussian Filter", use_gaussian_filter=True, use_editor=False, group="aon"),
        base_cfg.clone(name="Influence Weighted", use_influence_weighting=True, use_editor=False, group="aon"),
        base_cfg.clone(name="Random Latent", use_random_latent=True, use_editor=False, group="aon"),
        base_cfg.clone(name="STRIDE (v1)", group="aon"), # LSPO/R-DPO are False by default
        
        # --- Additive Improvements (Single Features) ---
        base_cfg.clone(name="+ LSPO (Norm)", use_lspo=True, use_lspo_norm=True, group="aon"),
        base_cfg.clone(name="+ LSPO (No Norm)", use_lspo=True, use_lspo_norm=False, group="aon"),
        base_cfg.clone(name="+ Ranking DPO", use_ranking_dpo=True, group="aon"),
        
        # --- Full Pipeline (v2) ---
        base_cfg.clone(name="Full STRIDE (v2)", use_lspo=True, use_ranking_dpo=True, group="aon"),
        
        # --- Component Removals (Subtractive from v2) ---
        base_cfg.clone(name="- Editor", use_editor=False, use_lspo=True, use_ranking_dpo=True, group="aon"),
        base_cfg.clone(name="- Influence", use_influence=False, use_lspo=True, use_ranking_dpo=True, group="aon"),
        base_cfg.clone(name="- VAE Editing", use_vae_editing=False, use_lspo=True, use_ranking_dpo=True, group="aon"),
        base_cfg.clone(name="- Augmentation", use_augmentation=False, use_lspo=True, use_ranking_dpo=True, group="aon"),
        
        # --- Loss component removals (Subtractive from v2) ---
        base_cfg.clone(name="- DPO Loss", use_dpo_loss=False, use_lspo=True, use_ranking_dpo=True, group="aon"),
        base_cfg.clone(name="- Cosine Loss", use_cosine_loss=False, use_lspo=True, use_ranking_dpo=True, group="aon"),
        base_cfg.clone(name="- Reg Loss", use_reg_loss=False, use_lspo=True, use_ranking_dpo=True, group="aon"),
    ]

    # Run the sweep and plot incrementally
    for i, cfg in enumerate(sweep):
        # Run specific config (all trials)
        run_sweep([cfg], num_trials=args.num_trials, results_dir=results_dir, verbose=True)

        # Regenerate plot after each config finishes
        print(f"\n=== Updating All-or-Nothing Plot ({i+1}/{len(sweep)}) ===")
        all_res = load_all_results(results_dir)
        
        plot_bar_ablation(
            all_res, 
            group_name="aon", 
            out_path=f"{plots_dir}/all_or_nothing.png"
        )
        print(f"Plot updated: {plots_dir}/all_or_nothing.png")

    print(f"\nAll sweeps complete! Final plot saved to: {plots_dir}/all_or_nothing.png")


if __name__ == "__main__":
    main()
