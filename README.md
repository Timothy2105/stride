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

pip install -e external/robomimic

pip install numpy h5py tqdm matplotlib gym
pip install torch torchvision
pip install mujoco==2.3.7
```

## Verify Installation
First download a dataset

```bash
python external/robomimic/robomimic/scripts/download_datasets.py \
    --tasks lift \
    --dataset_types ph
```

Inspect dataset structure

```bash
python external/robomimic/robomimic/scripts/get_dataset_info.py \
    --dataset external/robomimic/datasets/lift/ph/low_dim_v15.hdf5
```