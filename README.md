# STRIDE - CS229 Final Project

## Repository Setup
Clone the repo

```bash
git clone --recurse-submodules git@github.com:USERNAME/stride.git
cd stride
```

Create environment and install dependencies
```bash
conda create -n stride python=3.9 -y
conda activate stride
pip install --upgrade pip

conda install pytorch torchvision -c pytorch

cd external/robomimic
pip install -e .
cd ../..
cd external/robosuite
pip install -r requirements.txt

pip install --no-deps robosuite
pip install --only-binary=:all: mujoco==3.3.7
pip install mink==0.0.5 numba pynput pytest "qpsolvers[quadprog]>=4.3.1" scipy
pip install wandb
```

## Verify Installation
First download a dataset

```python
python external/robomimic/robomimic/scripts/download_datasets.py \
    --tasks lift \
    --dataset_types mh
```

Inspect dataset structure

```python
python external/robomimic/robomimic/scripts/get_dataset_info.py \
    --dataset external/robomimic/datasets/lift/mh/low_dim_v15.hdf5
```

## Set up WandB
```python
wandb login
python external/robomimic/robomimic/scripts/setup_macros.py
```
Edit `external/robomimic/robomimic/macros_private.py` to set up `WANDB_ENTITY` and `WANDB_PROJECT`

## Conduct Simple Baselines
Vanilla Behavior Cloning (BC)
> note: if working on ssh, you will need to configure a display
```python
python external/robomimic/robomimic/scripts/train.py \
  --config configs/bc_lift_mh.json \
  --dataset external/robomimic/datasets/lift/mh/low_dim_v15.hdf5
```