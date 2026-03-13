# STRIDE: Strategic Trajectory Refinement via Influence-guided Data Editing

End-to-end framework for improving imitation learning on D4RL Adroit
manipulation tasks (pen, hammer, door, relocate) through strategic data
editing using influence scores, a conditional VAE, and a DPO-trained
latent editor.

**Policy**: Simple MLP Behavioral Cloning (no diffusion policy).

---

## Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | **Vanilla BC** | MLP BC trained on raw demonstrations |
| 2 | **Gaussian 25%** | Temporal Gaussian smoothing (σ=2.5) of actions before BC |
| 3 | **Gaussian 50%** | Temporal Gaussian smoothing (σ=5.0) |
| 4 | **Gaussian 75%** | Temporal Gaussian smoothing (σ=7.5) |
| 5 | **CUPID 25%** | Keep top 25% of demos by TRAK influence score, retrain BC |
| 6 | **CUPID 50%** | Keep top 50% |
| 7 | **CUPID 75%** | Keep top 75% |
| 8 | **CUPID-Quality 25%** | Keep top 25% by CUPID-Quality ensemble score |
| 9 | **CUPID-Quality 50%** | Keep top 50% |
| 10 | **CUPID-Quality 75%** | Keep top 75% |
| 11 | **STRIDE** | Full pipeline: VAE → DPO editor → edit data → BC |
| 12 | **STRIDE w/o Influence** | Ablation: editor trained with random influence scores |
| 13 | **STRIDE Random Edits** | Ablation: random latent noise instead of trained editor |
| 14 | **BC + Influence Reweighting** | BC loss weighted by TRAK influence scores per demo |

Total: **14 methods × 4 tasks × 10 seeds = 560 experiments**.

---

## Repository Structure

```
stride/
  data.py                # Minari data loading for Adroit human-v2 tasks
  influence.py           # KNN-based corrective directions and preference pairs
  editing.py             # STRIDE two-stage editing pipeline
  scoring.py             # TRAK influence scoring (CUPID reimplementation)
  models/
    policy.py            # MLP Behavioral Cloning policy
    vae.py               # Conditional VAE for action representations
    editor.py            # Latent-space residual editor
  training/
    train_bc.py          # BC training with wandb logging
    train_vae.py         # VAE training with β-annealing
    train_editor_dpo.py  # DPO editor training
  eval/
    evaluate.py          # Evaluation with video rendering
  baselines/
    gaussian_filter.py   # Gaussian temporal smoothing
    cupid_filter.py      # CUPID demo curation
    cupid_quality.py     # CUPID-Quality demo curation
    random_latent.py     # Random latent edits (ablation)
experiments/
  configs.py             # All experiment configurations
  run_experiments.py     # Main experiment runner
  plot_results.py        # Result visualization
results/                 # Output directory (auto-created)
```

---

## 1. Environment Setup

```bash
conda create -n stride_new python=3.10 -y
conda activate stride_new

# Install MuJoCo (if not already installed)
pip install mujoco

# Install all Python dependencies
pip install -r requirements.txt
```

**Dependencies** (see `requirements.txt`):
- `torch>=2.0` — deep learning (requires `torch.func` for TRAK)
- `gymnasium>=0.29` + `gymnasium-robotics>=1.2` — Adroit environments
- `minari>=0.4` — D4RL dataset loading
- `mujoco>=3.0` — physics simulation
- `wandb` — experiment logging
- `imageio` + `imageio-ffmpeg` — video saving
- `scipy`, `scikit-learn`, `matplotlib`, `numpy`, `h5py`

Verify the environment:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import gymnasium; import gymnasium_robotics; gymnasium.make('AdroitHandPen-v1')"
python -c "import minari; minari.load_dataset('D4RL/pen/human-v2', download=True)"
```

---

## 2. Running Experiments

### Smoke Test (run this first to verify everything works)

Runs all 14 methods on a single task with 1 trial, minimal eval episodes,
minimal TRAK rollouts, no wandb, and no video — finishes fast:

```bash
conda activate stride_new
python -m experiments.run_experiments \
    --task pen \
    --method all \
    --device cuda \
    --seed 42 \
    --n-trials 1 \
    --n-eval-episodes 2 \
    --trak-n-rollouts 5 \
    --no-wandb \
    --no-video
```

If you want to smoke-test **all 4 tasks** at once:

```bash
python -m experiments.run_experiments \
    --task all \
    --method all \
    --device cuda \
    --seed 42 \
    --n-trials 1 \
    --n-eval-episodes 2 \
    --trak-n-rollouts 5 \
    --no-wandb \
    --no-video
```

Smoke-test individual method groups:

```bash
# Baselines only
python -m experiments.run_experiments --task pen \
    --method vanilla_bc,gaussian_25,gaussian_50,gaussian_75 \
    --n-trials 1 --n-eval-episodes 2 --trak-n-rollouts 5 --no-wandb --no-video

# CUPID variants only
python -m experiments.run_experiments --task pen \
    --method cupid_25,cupid_50,cupid_75,cupid_quality_25,cupid_quality_50,cupid_quality_75 \
    --n-trials 1 --n-eval-episodes 2 --trak-n-rollouts 5 --no-wandb --no-video

# STRIDE and ablations only
python -m experiments.run_experiments --task pen \
    --method stride,stride_no_influence,stride_random_edits,influence_reweight \
    --n-trials 1 --n-eval-episodes 2 --trak-n-rollouts 5 --no-wandb --no-video
```

### Full Run (all 4 tasks × 14 methods × 10 trials)

Once the smoke test passes, run the full experiment suite:

```bash
conda activate stride_new
python -m experiments.run_experiments \
    --task all \
    --method all \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 20 \
    --trak-n-rollouts 100
```

### Run a single task (10 trials)

```bash
python -m experiments.run_experiments --task pen --method all --device cuda
```

### Run specific methods

```bash
# Just baselines
python -m experiments.run_experiments --task pen \
    --method vanilla_bc,gaussian_25,gaussian_50,gaussian_75 --n-trials 10

# Just CUPID variants
python -m experiments.run_experiments --task pen \
    --method cupid_25,cupid_50,cupid_75,cupid_quality_25,cupid_quality_50,cupid_quality_75 --n-trials 10

# Just STRIDE and ablations
python -m experiments.run_experiments --task pen \
    --method stride,stride_no_influence,stride_random_edits,influence_reweight --n-trials 10
```

### Disable wandb (offline mode)

```bash
python -m experiments.run_experiments --task pen --method all --no-wandb
```

### Disable video rendering (faster evaluation)

```bash
python -m experiments.run_experiments --task pen --method all --no-video
```