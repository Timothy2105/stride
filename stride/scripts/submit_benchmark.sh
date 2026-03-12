#!/bin/bash
# Submit benchmark for all STRIDE tasks.
# Usage: bash stride/scripts/submit_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS=(pen hammer relocate door)

for task in "${TASKS[@]}"; do
  echo "Submitting benchmark: ${task}"
  sbatch --job-name="benchmark-${task}" "${SCRIPT_DIR}/run_benchmark_job.sh" "${task}" "$@"
done

echo "All ${#TASKS[@]} benchmark jobs submitted."
