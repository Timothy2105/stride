"""Plot experiment results from JSON output.

Generates grouped bar charts showing mean reward and success rate
across all methods for each Adroit task.

Usage
-----
    python -m experiments.plot_results [--results results/all_results.json]
                                       [--output results/plots]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Method display names and colours
METHOD_META = {
    "vanilla_bc":           ("Vanilla BC",           "#4c72b0"),
    "gaussian_25":          ("Gaussian σ=2.5",       "#55a868"),
    "gaussian_50":          ("Gaussian σ=5.0",       "#64b5f6"),
    "gaussian_75":          ("Gaussian σ=7.5",       "#8da0cb"),
    "cupid_25":             ("CUPID 25%",            "#c44e52"),
    "cupid_50":             ("CUPID 50%",            "#dd8452"),
    "cupid_75":             ("CUPID 75%",            "#da8bc3"),
    "cupid_quality_25":     ("CUPID-Q 25%",          "#e377c2"),
    "cupid_quality_50":     ("CUPID-Q 50%",          "#f7b6d2"),
    "cupid_quality_75":     ("CUPID-Q 75%",          "#ccb974"),
    "stride":               ("STRIDE",               "#ff7f0e"),
    "stride_no_influence":  ("STRIDE w/o Influence", "#bcbd22"),
    "stride_random_edits":  ("STRIDE Random Edits",  "#7f7f7f"),
    "influence_reweight":   ("BC + Influence Reweight", "#17becf"),
}


def load_results(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_task_results(
    results: list[dict],
    task: str,
    output_dir: Path,
) -> None:
    """Generate reward and success rate bar plots for a single task.

    When multiple seeds are present for the same method, results are averaged
    across seeds and error bars show ±1 std across trials.
    """
    task_results = [
        r for r in results
        if r.get("config", {}).get("task") == task and "eval_results" in r
    ]
    if not task_results:
        return

    # Aggregate across seeds: group by method
    from collections import defaultdict
    method_data: dict[str, list[dict]] = defaultdict(list)
    for r in task_results:
        m = r["config"]["method"]
        if m in METHOD_META:
            method_data[m].append(r)

    if not method_data:
        return

    # Canonical method ordering (same as METHOD_META insertion order)
    canonical_order = list(METHOD_META.keys())
    ordered_methods = [m for m in canonical_order if m in method_data]

    methods = []
    rewards = []
    reward_stds = []
    success_rates = []
    success_stds = []
    colours = []
    n_trials_list = []

    for m in ordered_methods:
        runs = method_data[m]
        per_seed_rewards = [r["eval_results"]["mean_reward"] for r in runs]
        per_seed_success = [r["eval_results"]["success_rate"] for r in runs]

        methods.append(METHOD_META[m][0])
        colours.append(METHOD_META[m][1])
        rewards.append(float(np.mean(per_seed_rewards)))
        reward_stds.append(float(np.std(per_seed_rewards)))
        success_rates.append(float(np.mean(per_seed_success)) * 100)
        success_stds.append(float(np.std(per_seed_success)) * 100)
        n_trials_list.append(len(runs))

    n_trials_str = (
        f" (n={n_trials_list[0]} trials)"
        if len(set(n_trials_list)) == 1 and n_trials_list[0] > 1
        else ""
    )

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        f"Adroit {task.capitalize()}{n_trials_str}",
        fontsize=16, fontweight="bold",
    )

    # Reward
    axes[0].bar(x, rewards, yerr=reward_stds, color=colours,
                edgecolor="black", linewidth=0.5, capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Episode Reward (±std across trials)")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(rewards):
        axes[0].text(i, v + reward_stds[i] + 0.5, f"{v:.1f}",
                     ha="center", fontsize=7)

    # Success Rate
    axes[1].bar(x, success_rates, yerr=success_stds, color=colours,
                edgecolor="black", linewidth=0.5, capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_title("Success Rate (±std across trials)")
    axes[1].set_ylim(0, 105)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(success_rates):
        axes[1].text(i, v + success_stds[i] + 1, f"{v:.1f}%",
                     ha="center", fontsize=7)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"results_{task}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"results_{task}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots for {task} to {output_dir}")


def plot_all(results_path: Path, output_dir: Path) -> None:
    results = load_results(results_path)
    tasks = sorted({r.get("config", {}).get("task", "") for r in results} - {""})
    for task in tasks:
        plot_task_results(results, task, output_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot STRIDE experiment results.")
    p.add_argument("--results", default="results/all_results.json")
    p.add_argument("--output", default="results/plots")
    args = p.parse_args()
    plot_all(Path(args.results), Path(args.output))


if __name__ == "__main__":
    main()
