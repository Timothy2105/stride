#!/usr/bin/env bash
set -euo pipefail

# Submit one tuning job per Adroit task.
TASKS=(pen hammer relocate door)

submit_local() {
  local task="$1"
  local cmd="python experiments/tune_stride.py --task ${task} --trials 100 --device cuda"
  local log="results/tune_${task}.log"
  mkdir -p results
  echo "[local] ${cmd}"
  nohup bash -lc "${cmd}" > "${log}" 2>&1 &
  echo "[local] task=${task} pid=$! log=${log}"
}

submit_slurm() {
  local task="$1"
  local cmd="python experiments/tune_stride.py --task ${task} --trials 100 --device cuda"
  local job_name="stride-tune-${task}"
  local out="results/slurm-${job_name}-%j.out"
  mkdir -p results
  local job_id
  job_id=$(sbatch --parsable --job-name "${job_name}" --output "${out}" --wrap "cd $(pwd) && ${cmd}")
  echo "[slurm] task=${task} job_id=${job_id} out=${out}"
}

for task in "${TASKS[@]}"; do
  if command -v sbatch >/dev/null 2>&1; then
    submit_slurm "${task}"
  else
    submit_local "${task}"
  fi
done
