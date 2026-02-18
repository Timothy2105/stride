#!/bin/bash
set -e

STRIDE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILTER_PIPELINE="${STRIDE_ROOT}/scripts/run_filtering_pipeline.sh"
SLURM_HOSTNAME="sc.stanford.edu"
SLURM_SBATCH_FILE="${STRIDE_ROOT}/scripts/submit_filtering.sh"

TASKS="lift can square transport tool_hang"
FILTER_RATIOS="0.10 0.25 0.50"
HDF5_TYPE="low_dim"
SEED=0
WANDB_ENTITY="stride-cs229"
WANDB_PROJECT="complete-gaussian-filtered-baselines"

declare -A DATASET_TYPES=(
    ["lift"]="mh"
    ["can"]="mh"
    ["square"]="mh"
    ["transport"]="mh"
    ["tool_hang"]="ph"
)

while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks) TASKS="$2"; shift 2 ;;
        --filter_ratios) FILTER_RATIOS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${STRIDE_ROOT}/sbatch_logs"

for task in ${TASKS}; do
    dataset_type="${DATASET_TYPES[${task}]}"
    for ratio in ${FILTER_RATIOS}; do
        CMD="bash ${FILTER_PIPELINE} --task ${task} --dataset_type ${dataset_type} --hdf5_type ${HDF5_TYPE} --filter_ratio ${ratio} --seed ${SEED} --wandb_entity ${WANDB_ENTITY} --wandb_project ${WANDB_PROJECT} --use-system-python"
        echo "${CMD}"
        if [[ $(hostname) == "${SLURM_HOSTNAME}" ]]; then
            sbatch "${SLURM_SBATCH_FILE}" "${CMD}"
        else
            eval ${CMD}
        fi
    done
done
