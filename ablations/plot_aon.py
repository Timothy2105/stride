#!/usr/bin/env python3
"""Plot results from the all-or-nothing (aon) ablation experiment.

This script loads existing JSON results from ablations/results_aon/ and
generates a bar chart in ablations/plots/all_or_nothing.png.
"""

import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stride.ablation.plotting import plot_bar_ablation
from stride.ablation.runner import load_all_results

def main():
    results_dir = "ablations/results_aon"
    plots_dir = "ablations/plots"
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory {results_dir} not found.")
        print("Run 'python ablations/all_or_nothing.py' first to generate results.")
        return

    print(f"Loading results from {results_dir}...")
    all_res = load_all_results(results_dir)
    
    if not all_res:
        print("No results found in the directory.")
        return

    print(f"Generating All-or-Nothing Plot (group='aon')...")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_bar_ablation(
        all_res, 
        group_name="aon", 
        out_path=f"{plots_dir}/all_or_nothing.png"
    )

    print(f"\nDone! Result plot saved to: {plots_dir}/all_or_nothing.png")

if __name__ == "__main__":
    main()
