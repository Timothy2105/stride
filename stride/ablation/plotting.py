"""Plotting utilities for ablation sweeps.

Generates visualizations for ablation results, including line plots for
parameter ranges and bar charts for component-level ablations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_parameter_sweep(
    results: list[dict],
    param_name: str,
    metric_name: str = "success_rate",
    group_name: str | None = None,
    out_path: str | None = None,
):
    """Plot a metric vs a single parameter value.

    Assumes all results in the list vary only by `param_name`.
    """
    if group_name:
        results = [r for r in results if r["config"].get("group") == group_name]
    
    if not results:
        print(f"No results found for group {group_name}")
        return

    # Extract data
    x_vals = []
    y_vals = []
    y_errs = []

    for r in results:
        cfg = r["config"]
        metrics = r["metrics"]
        
        if param_name in cfg:
            x_vals.append(cfg[param_name])
            
            val = metrics.get(metric_name, 0.0)
            err = metrics.get(f"std_{metric_name}", 0.0)
            
            # If success_rate, convert to %
            if metric_name == "success_rate" and val <= 1.0:
                val *= 100
                err *= 100
                
            y_vals.append(val)
            y_errs.append(err)

    # Sort by x
    indices = np.argsort(x_vals)
    x_vals = np.array(x_vals)[indices]
    y_vals = np.array(y_vals)[indices]
    y_errs = np.array(y_errs)[indices]

    plt.figure(figsize=(8, 5))
    plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='-o', capsize=5, color="#4C72B0", linewidth=2)
    plt.xlabel(param_name.replace("_", " ").title())
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"Ablation: {param_name} vs {metric_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Add labels
    for i, v in enumerate(y_vals):
        plt.text(x_vals[i], v + y_errs[i] + 0.5, f"{v:.1f}", ha='center', va='bottom', fontsize=9)
    
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    
    plt.close()


def plot_bar_ablation(
    results: list[dict],
    group_name: str,
    metric_name: str = "success_rate",
    out_path: str | None = None,
):
    """Plot a comparison bar chart for all runs in a group."""
    results = [r for r in results if r["config"].get("group") == group_name]
    if not results:
        print(f"No results found for group {group_name}")
        return

    # Sort results by value (descending) and then by error (ascending for ties)
    # Higher value is better; for ties, lower std is better.
    results.sort(
        key=lambda r: (
            r["metrics"].get(metric_name, 0.0), 
            -r["metrics"].get(f"std_{metric_name}", 0.0)
        ), 
        reverse=True
    )

    names = [r["config"]["name"].replace(f"{r['config']['group']}-", "") for r in results]
    vals = [r["metrics"].get(metric_name, 0.0) for r in results]
    errs = [r["metrics"].get(f"std_{metric_name}", 0.0) for r in results]
    
    if metric_name == "success_rate" and all(v <= 1.0 for v in vals):
        vals = [v * 100 for v in vals]
        errs = [e * 100 for e in errs]
        ylabel = "Success Rate (%)"
    else:
        ylabel = metric_name.replace("_", " ").title()

    plt.figure(figsize=(10, 6))
    x = np.arange(len(names))
    plt.bar(x, vals, yerr=errs, capsize=5, color="#4C72B0", alpha=0.8, edgecolor="white")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"Ablation Group: {group_name}")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.ylim(bottom=40)

    # Add labels above bars
    for i, v in enumerate(vals):
        plt.text(i, v + errs[i] + 0.5, f"{v:.1f}", ha='center', va='bottom', fontweight='bold')

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved bar chart to {out_path}")
    
    plt.close()
