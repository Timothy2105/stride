#!/bin/bash
#SBATCH --account=juno
#SBATCH --partition=juno-lo
#SBATCH --exclude=juno1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/juno/u/tyu2105/projects/stride/sbatch_logs/timothy-%j.out
#SBATCH --mail-user=tyu2105@stanford.edu
#SBATCH --mail-type=END,FAIL

# Timothy's experiment job runner.
# Usage: sbatch run_timothy_job.sh <task>

set -euo pipefail

TASK="${1:?Usage: sbatch run_timothy_job.sh <task>}"
METHODS="gaussian_50,gaussian_75,cupid_50,cupid_75,influence_reweight"

REPO_ROOT="/juno/u/tyu2105/projects/stride"
cd "${REPO_ROOT}"

export MUJOCO_GL=egl
export WANDB_ENTITY="stride-cs229"
export WANDB_INIT_TIMEOUT=300

echo "=== Timothy experiments — ${TASK} ==="

python -u -m experiments.run_experiments \
    --task "${TASK}" \
    --method "${METHODS}" \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 50
