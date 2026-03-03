"""Orchestrator for managing multi-config ablation sweeps.

This module provides tools to generate grids of configurations and execute
them sequentially or in parallel, aggregating results for visualization.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Dict, List

import numpy as np

from stride.ablation.config import AblationConfig
from stride.ablation.runner import run_ablation, save_ablation_result


def generate_sweep(
    base_cfg: AblationConfig,
    sweep_params: dict[str, Iterable[Any]],
    group_name: str | None = None,
) -> list[AblationConfig]:
    """Generate a grid of configurations for a parameter sweep.

    Parameters
    ----------
    base_cfg     : The starting configuration to clone from.
    sweep_params : A dict mapping config field names to lists of values.
    group_name   : Optional name for the sweep group.

    Returns
    -------
    List of AblationConfig objects.
    """
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    configs = []

    for combo in itertools.product(*values):
        overrides = dict(zip(keys, combo))
        
        # Build a descriptive name
        name_parts = [base_cfg.name]
        for k, v in overrides.items():
            name_parts.append(f"{k}={v}")
        
        cfg = base_cfg.clone(**overrides)
        cfg.name = "-".join(name_parts)
        if group_name:
            cfg.group = group_name
        
        configs.append(cfg)
    
    return configs


def run_sweep(
    configs: list[AblationConfig],
    num_trials: int = 1,
    ckpt_dir: str = "ablations/checkpoints",
    results_dir: str = "ablations/results",
    verbose: bool = True,
) -> list[dict]:
    """Execute a list of ablation configurations, potentially with multiple trials.

    Parameters
    ----------
    configs     : List of AblationConfig to run.
    num_trials  : Number of trials (different seeds) per configuration.
    ckpt_dir    : Checkpoint directory.
    results_dir : Directory to save result JSONs.
    verbose     : Print progress.

    Returns
    -------
    List of aggregated result dictionaries.
    """
    results = []
    n_configs = len(configs)
    
    for idx, cfg in enumerate(configs):
        trial_results = []
        for trial in range(num_trials):
            trial_seed = cfg.seed + trial
            trial_cfg = cfg.clone(seed=trial_seed)
            if num_trials > 1:
                trial_cfg.name = f"{cfg.name}_trial{trial}"
            
            if verbose:
                msg = f"\n>>> Running config {idx+1}/{n_configs}"
                if num_trials > 1:
                    msg += f", Trial {trial+1}/{num_trials}"
                print(f"{msg}: {cfg.name} (seed={trial_seed})")
            
            res = run_ablation(trial_cfg, ckpt_dir=ckpt_dir, verbose=verbose)
            trial_results.append(res)
        
        # Aggregate trial results
        if num_trials > 1:
            aggregated: Dict[str, Any] = {
                "config": cfg.to_dict(),
                "metrics": {},
                "timing": {},
                "training": {},
                "num_trials": num_trials,
                "trial_results": trial_results
            }
            
            # Aggregate metrics
            metric_keys = trial_results[0]["metrics"].keys()
            for key in metric_keys:
                vals = [tr["metrics"].get(key, 0.0) for tr in trial_results if "metrics" in tr]
                if vals and isinstance(vals[0], (int, float)):
                    aggregated["metrics"][key] = float(np.mean(vals))
                    aggregated["metrics"][f"std_{key}"] = float(np.std(vals))
            
            # Aggregate timing
            timing_keys = trial_results[0]["timing"].keys()
            for key in timing_keys:
                vals = [tr["timing"].get(key, 0.0) for tr in trial_results]
                aggregated["timing"][key] = float(np.mean(vals))
            
            res_to_save = aggregated
        else:
            res_to_save = trial_results[0]
            
        save_ablation_result(res_to_save, results_dir=results_dir)
        results.append(res_to_save)
    
    return results
