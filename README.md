# STRIDE - CS229 Final Project

## Repository Setup

Clone the repository:

```bash
git clone --recurse-submodules git@github.com:USERNAME/stride.git
cd stride
```

Create the environment and install dependencies:

```bash
conda create -n stride python=3.9 -y
conda activate stride

pip install poetry
poetry install
```

---

## Verify Installation

Download a dataset:

```bash
python -m robomimic.scripts.download_datasets \
    --tasks lift \
    --dataset_types mh
```

Inspect dataset structure:

```bash
python -m robomimic.scripts.get_dataset_info \
    --dataset third_party/robomimic/datasets/lift/mh/low_dim_v15.hdf5
```

---

## Set Up WandB

```bash
wandb login
python -m robomimic.scripts.setup_macros
```

Edit `stride/third_party/robomimic/robomimic/macros_private.py` to set `WANDB_ENTITY`.

---

## Conduct Simple Baselines

### Vanilla Behavior Cloning (BC)

Train the policy:

```bash
python -m robomimic.scripts.train \
  --config configs/bc_lift_mh.json \
  --dataset third_party/robomimic/datasets/lift/mh/low_dim_v15.hdf5
```
> Note: if working over SSH, a display may need to be configured.

Evaluate the policy:

```bash
python -m robomimic.scripts.run_trained_agent \
      --agent third_party/robomimic/bc_trained_models/bc_lift_mh/vanilla-bc-baseline/models/model_epoch_2000.pth \
      --n_rollouts 50 \
      --seed 0 \
      --video_path rollouts/vanilla_bc_baseline/eval_output.mp4 \
      --camera_names agentview
```

---

## Setup for CUPID

> Please refer to the official CUPID documentation for more detailed instructions.

Go to the root directory of the CUPID repo:

```bash
cd stride/third_party/cupid
```

Create a new conda environment:
```bash
mamba env create -f conda_environment.yaml
conda activate cupid

pip install patchelf
```

> Remember to configure the full path to CUPID in `scripts/submit.sh`.

1. Train policies on uncurated data:
```bash
bash scripts/train/train_policies.sh
```

2. Evaluate policies for rollouts on uncurated data:
```bash
bash scripts/eval/eval_save_episodes.sh
```

3. Estimate action influences:
```bash
bash scripts/eval/train_trak.sh
```

### Notes
- For all the scripts, make sure to set the `SLURM_HOSTNAME`, `SLURM_SBATCH_FILE`, `date` variables accordingly (refer to CUPID documentation for more details).
- For `eval_save_epsiodes.sh`, `train_date` should correspond to the `date` set in `train_policies.sh`.
- For `train_trak.sh`, `train_date` should correspond to the `date` set in `train_policies.sh`, and `eval_date` should correspond to the `date` set in `eval_save_episodes.sh`.