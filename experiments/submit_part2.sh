#!/bin/bash




set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS=(pen hammer door relocate)

for task in "${TASKS[@]}"; do
  echo "Submitting: ${task}"
  sbatch --job-name="${task}" "${SCRIPT_DIR}/run_part2_job.sh" "${task}"
done

echo "All ${#TASKS[@]} jobs submitted."
