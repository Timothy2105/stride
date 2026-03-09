#!/bin/bash
#SBATCH --account=juno
#SBATCH --partition=juno
#SBATCH --exclude=juno1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/juno/u/tyu2105/projects/stride/sbatch_logs/stride-edit-%j.out
#SBATCH --mail-user=tyu2105@stanford.edu
#SBATCH --mail-type=END,FAIL

# Tuning job runner.
# Usage: sbatch run_tuning_job.sh <task> [extra_args...]

set -euo pipefail

TASK="${1:?Usage: sbatch run_tuning_job.sh <task>}"
shift
EXTRA_ARGS=("$@")

# REPO_ROOT="/juno/u/tyu2105/projects/stride"

# cd "${REPO_ROOT}"
# mkdir -p results/sbatch_logs

export WANDB_INIT_TIMEOUT=300

echo "=== Tuning STRIDE — ${TASK} ==="

python experiments/tune_stride.py \
  --task "${TASK}" \
  --trials 30 \
  --device cuda \
  "${EXTRA_ARGS[@]}"
