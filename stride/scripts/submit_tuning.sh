#!/bin/bash
# Submit tuning for all STRIDE tasks.
# Usage: bash stride/scripts/submit_tuning.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# TASKS=(pen hammer relocate door)
TASKS=(pen)


for task in "${TASKS[@]}"; do
  echo "Submitting tuning: ${task}"
  sbatch --job-name="tune-${task}" "${SCRIPT_DIR}/run_tuning_job.sh" "${task}" "$@"
done

echo "All ${#TASKS[@]} tuning jobs submitted."
