"""
Compute last 10 eval summaries for gaussian filtering baseline
"""

import json
import pathlib
import argparse

CUPID_TRAIN_DIR = pathlib.Path(__file__).parent.parent / "stride/third_party/cupid/data/outputs/train"
N = 10


def get_last_n_mean(train_dir: pathlib.Path, n: int = N):
    logs = train_dir / "logs.json.txt"
    if not logs.exists():
        return None, None
    scores, epochs = [], []
    with open(logs) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if "test/mean_score" in entry:
                    scores.append(entry["test/mean_score"])
                    epochs.append(entry["epoch"])
            except json.JSONDecodeError:
                continue
    if not scores:
        return None, None
    return scores[-n:], epochs[-n:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Filter by train date (e.g. 2026.02.18)")
    parser.add_argument("--save", action="store_true", help="Save eval_summary.json into each run dir")
    args = parser.parse_args()

    pattern = f"{args.date}*gaussian_filter*" if args.date else "*gaussian_filter*"
    runs = sorted(CUPID_TRAIN_DIR.glob(f"*/{pattern}"))

    if not runs:
        print(f"No gaussian filter runs found under {CUPID_TRAIN_DIR}")
        return

    print(f"\n{'Run':<80} {'Evals':>5} {'Last-epoch':>10} {'Mean (last {N})':>15}")
    print("-" * 115)
    for run in runs:
        scores, epochs = get_last_n_mean(run)
        if scores is None:
            print(f"{run.name:<80} {'(no logs)':>5}")
            continue
        mean = sum(scores) / len(scores)
        print(f"{run.name:<80} {len(scores):>5} {epochs[-1]:>10} {mean:>15.4f}")

        if args.save:
            summary = {
                "last_n": N,
                "num_logged_evals": len(scores),
                "last_n_epochs": epochs,
                "last_n_scores": scores,
                "mean_score": mean,
            }
            out = run / "eval_summary.json"
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  -> saved {out}")

    print()


if __name__ == "__main__":
    main()
