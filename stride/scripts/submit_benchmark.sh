#!/usr/bin/env bash
set -euo pipefail

# Submit one benchmark job per Adroit task using tuned STRIDE params.
TASKS=(pen hammer relocate door)

submit_local() {
  local task="$1"
  local cmd="python experiments/benchmark.py --task ${task} --device cuda --num-trials 50 --stride-params results/tune_${task}_best_params.json"
  local log="results/benchmark_${task}.log"
  mkdir -p results
  echo "[local] ${cmd}"
  nohup bash -lc "${cmd}" > "${log}" 2>&1 &
  echo "[local] task=${task} pid=$! log=${log}"
}

submit_slurm() {
  local task="$1"
  local cmd="python experiments/benchmark.py --task ${task} --device cuda --num-trials 50 --stride-params results/tune_${task}_best_params.json"
  local job_name="stride-benchmark-${task}"
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
