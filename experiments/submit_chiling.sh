#!/bin/bash
# Submit Chiling's experiments: 5 methods × 4 tasks × 10 trials.
# Launches one SLURM job per task (4 jobs total).
# Usage: bash experiments/submit_chiling.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS=(pen hammer door relocate)

for task in "${TASKS[@]}"; do
  echo "Submitting: ${task}"
  sbatch --job-name="${task}" "${SCRIPT_DIR}/run_chiling_job.sh" "${task}"
done

echo "All ${#TASKS[@]} jobs submitted."
