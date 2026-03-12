#!/bin/bash
# Submit Timothy's experiments: 5 methods × 4 tasks × 10 trials.
# Launches one SLURM job per task (4 jobs total).
# Usage: bash experiments/submit_timothy.sh

set -euo pipefail

REPO_ROOT="/juno/u/tyu2105/projects/stride"
TASKS=(pen hammer door relocate)
METHODS="gaussian_50,gaussian_75,cupid_50,cupid_75,influence_reweight"

for task in "${TASKS[@]}"; do
  echo "Submitting: ${task}"
  sbatch --job-name="${task}" \
         --account=juno \
         --partition=juno \
         --exclude=juno1 \
         --cpus-per-task=12 \
         --mem=90G \
         --time=72:00:00 \
         --gres=gpu:1 \
         --output="${REPO_ROOT}/sbatch_logs/${task}-%j.out" \
         --mail-user=tyu2105@stanford.edu \
         --mail-type=END,FAIL \
         --wrap="
set -euo pipefail
cd ${REPO_ROOT}

export MUJOCO_GL=egl
export WANDB_ENTITY=stride-cs229
export WANDB_INIT_TIMEOUT=300

echo \"=== Timothy experiments — ${task} ===\"

python -u -m experiments.run_experiments \
    --task ${task} \
    --method ${METHODS} \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 50
"
done

echo "All ${#TASKS[@]} jobs submitted."
