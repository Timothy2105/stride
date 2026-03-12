#!/bin/bash
#SBATCH --account=juno
#SBATCH --partition=juno
#SBATCH --exclude=juno1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/juno/u/tyu2105/projects/stride/sbatch_logs/benchmark-%j.out
#SBATCH --mail-user=tyu2105@stanford.edu
#SBATCH --mail-type=END,FAIL

# Benchmark job runner.
# Usage: sbatch run_benchmark_job.sh <task> [extra_args...]

set -euo pipefail

TASK="${1:?Usage: sbatch run_benchmark_job.sh <task>}"
shift
EXTRA_ARGS=("$@")

REPO_ROOT="/juno/u/tyu2105/projects/stride"
cd "${REPO_ROOT}"

export WANDB_ENTITY="stride-cs229"
export WANDB_INIT_TIMEOUT=300

echo "=== Benchmark STRIDE — ${TASK} ==="

python -u experiments/benchmark.py \
  --task "${TASK}" \
  --device cuda \
  --method all \
  --score-dir results/cupid_scores \
  "${EXTRA_ARGS[@]}"
