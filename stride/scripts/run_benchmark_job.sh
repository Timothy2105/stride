#!/bin/bash
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --exclude=<node_list>
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=results/sbatch_logs/benchmark-%j.out
#SBATCH --mail-user=<you@example.com>
#SBATCH --mail-type=END,FAIL

# Benchmark job runner.
# Usage: sbatch run_benchmark_job.sh <task> [extra_args...]

set -euo pipefail

TASK="${1:?Usage: sbatch run_benchmark_job.sh <task>}"
shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p results/sbatch_logs

export WANDB_INIT_TIMEOUT=300

echo "=== Benchmark STRIDE — ${TASK} ==="

python experiments/benchmark.py \
  --task "${TASK}" \
  --device cuda \
  --num-trials 50 \
  --stride-params "results/tune_${TASK}_best_params.json" \
  "${EXTRA_ARGS[@]}"
