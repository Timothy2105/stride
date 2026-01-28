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
    --dataset external/robomimic/datasets/lift/mh/low_dim_v15.hdf5
```

## Set Up WandB

```bash
wandb login
python external/robomimic/robomimic/scripts/setup_macros.py
```

Edit `external/robomimic/robomimic/macros_private.py` to set `WANDB_ENTITY`.

## Conduct Simple Baselines

### Vanilla Behavior Cloning (BC)

> Note: if working over SSH, a display may need to be configured.

```bash
python -m robomimic.scripts.train \
  --config configs/bc_lift_mh.json \
  --dataset external/robomimic/datasets/lift/mh/low_dim_v15.hdf5
```
