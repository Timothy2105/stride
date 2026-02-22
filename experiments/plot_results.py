"""Plot comparison bar chart from experiment results.

Reads results/results.json and produces a bar chart comparing all 5 methods
by mean reward (with ± std error bars) and a separate success-rate panel.

Usage
-----
    python experiments/plot_results.py [--results results/results.json]
                                        [--out results/comparison.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description="Plot STRIDE results comparison")
    p.add_argument("--results", default="results/results.json",
                   help="Path to results JSON file")
    p.add_argument("--out", default="results/comparison.png",
                   help="Output image path")
    args = p.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    with open(args.results) as f:
        results = json.load(f)

    # Filter to methods that have numeric results
    methods = []
    means = []
    stds = []
    success_rates = []

    for name, stats in results.items():
        if "mean_reward" in stats:
            methods.append(name)
            means.append(stats["mean_reward"])
            stds.append(stats["std_reward"])
            success_rates.append(stats["success_rate"] * 100)

    if not methods:
        print("No evaluation results found in results.json. "
              "Run experiments/run_all.py first.")
        return

    x = np.arange(len(methods))
    width = 0.55

    # Colour scheme: highlight STRIDE
    colors = ["#4C72B0"] * len(methods)
    stride_idx = next((i for i, m in enumerate(methods) if "STRIDE" in m), None)
    if stride_idx is not None:
        colors[stride_idx] = "#DD8452"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("STRIDE vs Baselines — AdroitHandPen-v1 (pen-human-v2)",
                 fontsize=14, fontweight="bold")

    # --- Mean reward ---
    ax = axes[0]
    bars = ax.bar(x, means, width, yerr=stds, color=colors,
                  capsize=5, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title("Mean Reward (± std)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(stds) * 0.05,
                f"{mean_val:.1f}",
                ha="center", va="bottom", fontsize=8)

    # --- Success rate ---
    ax2 = axes[1]
    bars2 = ax2.bar(x, success_rates, width, color=colors,
                    edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Success Rate (%)", fontsize=11)
    ax2.set_title("Success Rate", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, sr in zip(bars2, success_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + 1.5,
                 f"{sr:.1f}%",
                 ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
