#!/bin/bash
set -e

# Script to filter robomimic datasets and train diffusion policy on filtered data
# All operations happen under the cupid folder

# Default values
TASK="transport"
DATASET_TYPE="mh"
HDF5_TYPE="low_dim"
FILTER_RATIO=0.5
FILTER_METHOD="lowest_likelihood"
SEED=0
TRAIN_POLICY=1
DEBUG=0
USE_SYSTEM_PYTHON=0
# WandB defaults (can be overridden via CLI)
WANDB_ENTITY="stride-cs229"
WANDB_PROJECT="robomimic-baselines"

# Epochs table (mirrors cupid's train_policies.sh)
declare -A NUM_EPOCHS=(
    ## Standard (low_dim)
    ["pusht_low_dim"]=1001
    ["lift_mh_low_dim"]=1001
    ["can_mh_low_dim"]=1001
    ["square_mh_low_dim"]=1751
    ["transport_mh_low_dim"]=1001
    ["tool_hang_ph_low_dim"]=601
    ## Standard (image)
    ["pusht_image"]=301
    ["lift_mh_image"]=301
    ["can_mh_image"]=301
    ["square_mh_image"]=301
    ["transport_mh_image"]=301
    ["tool_hang_ph_image"]=301
)

# Paths (relative to stride root)
STRIDE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUPID_DIR="${STRIDE_ROOT}/stride/third_party/cupid"
FILTER_SCRIPT="${STRIDE_ROOT}/stride/src/filtering/filter.py"

# Parse command line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --task TASK              Task name (default: lift)"
    echo "                          Options: lift, can, square, transport, tool_hang"
    echo "  --dataset_type TYPE      Dataset type (default: mh)"
    echo "                          Options: mh, ph"
    echo "  --hdf5_type TYPE         HDF5 type (default: low_dim)"
    echo "                          Options: low_dim, image"
    echo "  --filter_ratio RATIO     Fraction to filter out (default: 0.5)"
    echo "  --filter_method METHOD   Filtering method (default: lowest_likelihood)"
    echo "                          Options: lowest_likelihood, threshold"
    echo "  --seed SEED              Random seed (default: 42)"
    echo "  --no_train               Skip training step"
    echo "  --debug                  Print commands without executing"
    echo "  --wandb_entity ENTITY    WandB entity (overrides default)"
    echo "  --wandb_project PROJECT  WandB project (default: ${WANDB_PROJECT})"
    echo "  --help                   Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --task transport --dataset_type mh --filter_ratio 0.3"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --dataset_type)
            DATASET_TYPE="$2"
            shift 2
            ;;
        --hdf5_type)
            HDF5_TYPE="$2"
            shift 2
            ;;
        --filter_ratio)
            FILTER_RATIO="$2"
            shift 2
            ;;
        --filter_method)
            FILTER_METHOD="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no_train)
            TRAIN_POLICY=0
            shift
            ;;
        --wandb_entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --debug)
            DEBUG=1
            shift
            ;;
        --use-system-python)
            USE_SYSTEM_PYTHON=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! "$TASK" =~ ^(lift|can|square|transport|tool_hang)$ ]]; then
    echo "Error: Invalid task: $TASK"
    exit 1
fi

if [[ ! "$DATASET_TYPE" =~ ^(mh|ph)$ ]]; then
    echo "Error: Invalid dataset type: $DATASET_TYPE"
    exit 1
fi

if [[ ! "$HDF5_TYPE" =~ ^(low_dim|image)$ ]]; then
    echo "Error: Invalid hdf5 type: $HDF5_TYPE"
    exit 1
fi

# Determine dataset file naming convention
# Look for datasets in cupid's dataset directory
CUPID_DATASETS_DIR="${CUPID_DIR}/data/robomimic/datasets"
DATASET_DIR="${CUPID_DATASETS_DIR}/${TASK}/${DATASET_TYPE}"

# Try to find the original dataset in cupid's dataset directory
ORIGINAL_DATASET=""
ORIGINAL_BASENAME=""
if [[ "$HDF5_TYPE" == "low_dim" ]]; then
    # Try different naming conventions (prefer abs_action version)
    for pattern in "low_dim_abs.hdf5" "low_dim.hdf5" "low_dim_v*.hdf5"; do
        found=$(find "${DATASET_DIR}" -maxdepth 1 -name "$pattern" -not -name "*_filtered.hdf5" 2>/dev/null | head -1)
        if [[ -n "$found" ]]; then
            ORIGINAL_DATASET="$found"
            ORIGINAL_BASENAME=$(basename "$found")
            break
        fi
    done
elif [[ "$HDF5_TYPE" == "image" ]]; then
    # Try different naming conventions (prefer abs_action version)
    for pattern in "image_abs.hdf5" "image.hdf5"; do
        found=$(find "${DATASET_DIR}" -maxdepth 1 -name "$pattern" -not -name "*_filtered.hdf5" 2>/dev/null | head -1)
        if [[ -n "$found" ]]; then
            ORIGINAL_DATASET="$found"
            ORIGINAL_BASENAME=$(basename "$found")
            break
        fi
    done
fi

if [[ -z "$ORIGINAL_DATASET" ]]; then
    echo "Error: Could not find original dataset for task=$TASK, dataset_type=$DATASET_TYPE, hdf5_type=$HDF5_TYPE"
    echo "Please ensure the dataset exists in: ${DATASET_DIR}/"
    echo "Looking for files matching:"
    if [[ "$HDF5_TYPE" == "low_dim" ]]; then
        echo "  - low_dim_abs.hdf5"
        echo "  - low_dim.hdf5"
    else
        echo "  - image_abs.hdf5"
        echo "  - image.hdf5"
    fi
    exit 1
fi

echo "Found original dataset: $ORIGINAL_DATASET"

# Determine output dataset name: preserve original naming but add "_filtered" before .hdf5
# e.g., low_dim_abs.hdf5 -> low_dim_abs_filtered.hdf5
#       low_dim.hdf5 -> low_dim_filtered.hdf5
if [[ "$ORIGINAL_BASENAME" == *.hdf5 ]]; then
    OUTPUT_FILENAME="${ORIGINAL_BASENAME%.hdf5}_filtered_${FILTER_RATIO}.hdf5"
else
    OUTPUT_FILENAME="${ORIGINAL_BASENAME}_filtered_${FILTER_RATIO}.hdf5"
fi

# Store filtered dataset in the same directory as the original
OUTPUT_DATASET="${DATASET_DIR}/${OUTPUT_FILENAME}"

echo ""
echo "=========================================="
echo "Filtering Pipeline Configuration"
echo "=========================================="
echo "Task: $TASK"
echo "Dataset Type: $DATASET_TYPE"
echo "HDF5 Type: $HDF5_TYPE"
echo "Filter Ratio: $FILTER_RATIO"
echo "Filter Method: $FILTER_METHOD"
echo "Seed: $SEED"
echo "Original Dataset: $ORIGINAL_DATASET"
echo "Output Dataset: $OUTPUT_DATASET"
echo "Train Policy: $TRAIN_POLICY"
if [[ $TRAIN_POLICY -eq 1 ]]; then
    epoch_key="${TASK}_${DATASET_TYPE}_${HDF5_TYPE}"
    if [[ -v NUM_EPOCHS["${epoch_key}"] ]]; then
        TRAIN_NUM_EPOCHS="${NUM_EPOCHS[${epoch_key}]}"
    else
        # Fallback if task is not in the table
        TRAIN_NUM_EPOCHS=1001
    fi
    echo "Training Epochs: $TRAIN_NUM_EPOCHS (key: ${epoch_key})"
fi
echo "=========================================="
echo ""

# Step 1: Filter the dataset
echo "Step 1: Filtering dataset..."
FILTER_CMD="python ${FILTER_SCRIPT} --dataset \"${ORIGINAL_DATASET}\" --output \"${OUTPUT_DATASET}\" --filter_ratio ${FILTER_RATIO} --method ${FILTER_METHOD} --seed ${SEED}"

if [[ $DEBUG -eq 1 ]]; then
    echo "[DEBUG] Would run: $FILTER_CMD"
else
    eval $FILTER_CMD
    if [[ $? -ne 0 ]]; then
        echo "Error: Filtering failed"
        exit 1
    fi
fi

echo "Filtering complete!"
echo ""

# Step 2: Train diffusion policy on filtered dataset
if [[ $TRAIN_POLICY -eq 1 ]]; then
    echo "Step 2: Training diffusion policy on filtered dataset..."
    
    # Change to cupid directory
    cd "${CUPID_DIR}"

    # If a WandB entity is provided, export it so cupid / hydra will use it.
    # WandB respects the WANDB_ENTITY environment variable.
    if [[ -n "${WANDB_ENTITY}" ]]; then
        export WANDB_ENTITY="${WANDB_ENTITY}"
        echo "Using WandB entity: ${WANDB_ENTITY}"
    fi
    
    # Determine the config path based on task and dataset type
    CONFIG_DIR="configs/${HDF5_TYPE}/${TASK}_${DATASET_TYPE}/diffusion_policy_cnn"
    
    if [[ ! -d "$CONFIG_DIR" ]]; then
        echo "Error: Config directory not found: $CONFIG_DIR"
        echo "Available configs:"
        ls -d configs/${HDF5_TYPE}/*/diffusion_policy_cnn 2>/dev/null || echo "None found"
        exit 1
    fi
    
    # Update the dataset path in the config or pass it via command line
    # Cupid uses hydra configs, so we can override the dataset path
    EXP_NAME="train_diffusion_unet_${HDF5_TYPE}_${TASK}_${DATASET_TYPE}_gaussian_filter_${FILTER_RATIO}"
    TRAIN_DATE=$(date +"%Y.%m.%d")
    TRAIN_NAME="${TRAIN_DATE}_${EXP_NAME}_${SEED}"
    
    # Build training command
    # Note: We need to override the dataset path to point to our filtered dataset
    # The path should be relative to cupid directory
    # Use the same naming convention as the original (e.g., low_dim_abs_filtered.hdf5)
    RELATIVE_DATASET_PATH="data/robomimic/datasets/${TASK}/${DATASET_TYPE}/${OUTPUT_FILENAME}"
    
    # Use the Python executable from the cupid conda environment
    # This ensures we use the correct Python (3.9) with hydra-core, not the base environment
    if [[ ${USE_SYSTEM_PYTHON} -eq 1 ]]; then
        CUPID_PYTHON="python"
    else
        CUPID_PYTHON="${HOME}/anaconda3/envs/cupid/bin/python"
        if [[ ! -f "${CUPID_PYTHON}" ]]; then
            echo "Error: Could not find Python in cupid environment at ${CUPID_PYTHON}"
            echo "Please ensure the cupid conda environment is installed."
            exit 1
        fi
    fi
    
    TRAIN_CMD="${CUPID_PYTHON} train.py --config-dir=${CONFIG_DIR} --config-name=config.yaml"
    TRAIN_CMD="${TRAIN_CMD} name=${EXP_NAME}"
    TRAIN_CMD="${TRAIN_CMD} hydra.run.dir=data/outputs/train/${TRAIN_DATE}/${TRAIN_NAME}"
    TRAIN_CMD="${TRAIN_CMD} training.seed=${SEED}"
    # Use epochs from NUM_EPOCHS table
    if [[ -n "${TRAIN_NUM_EPOCHS:-}" ]]; then
        TRAIN_CMD="${TRAIN_CMD} training.num_epochs=${TRAIN_NUM_EPOCHS}"
    fi
    TRAIN_CMD="${TRAIN_CMD} task.dataset.dataset_path=${RELATIVE_DATASET_PATH}"
    TRAIN_CMD="${TRAIN_CMD} task.dataset_path=${RELATIVE_DATASET_PATH}"
    TRAIN_CMD="${TRAIN_CMD} task.env_runner.dataset_path=${RELATIVE_DATASET_PATH}"
    TRAIN_CMD="${TRAIN_CMD} task.dataset.val_ratio=0.0"
    TRAIN_CMD="${TRAIN_CMD} logging.name=${TRAIN_NAME}"
    TRAIN_CMD="${TRAIN_CMD} logging.group=${TRAIN_DATE}_${EXP_NAME}_${TASK}_${DATASET_TYPE}"
    TRAIN_CMD="${TRAIN_CMD} logging.project=${WANDB_PROJECT}"
    
    if [[ $DEBUG -eq 1 ]]; then
        echo "[DEBUG] Would run: $TRAIN_CMD"
    else
        echo "Running training command..."
        eval $TRAIN_CMD
        if [[ $? -ne 0 ]]; then
            echo "Error: Training failed"
            exit 1
        fi
    fi
    
    echo "Training complete!"
    echo ""
    echo "Model checkpoints saved to: ${CUPID_DIR}/data/outputs/train/${TRAIN_DATE}/${TRAIN_NAME}"

fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Filtered dataset: ${OUTPUT_DATASET}"
if [[ $TRAIN_POLICY -eq 1 ]]; then
    echo "Training outputs: ${CUPID_DIR}/data/outputs/train/${TRAIN_DATE}/${TRAIN_NAME}"

fi
echo "=========================================="
