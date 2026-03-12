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

---

## 3. What Happens Under the Hood

For each task, the runner executes this pipeline:

### Shared Pre-computation (once per task)
1. **Load data**: Downloads Adroit human-v2 demos via Minari
2. **Train reference BC**: MLP BC on full dataset (200 epochs)
3. **Rollout reference**: 100 episodes in env → success/failure labels
4. **TRAK scoring**: Per-sample gradients → random projection → ridge solve → influence matrix
5. **Compute CUPID scores**: Net influence on successes vs failures
6. **Compute CUPID-Quality scores**: Weighted ensemble (sum\_of\_sum × 0.5 + min\_of\_max × 0.25 + max\_of\_min × 0.25)

### Per-Method Execution
7. **Process data** according to the method:
   - *Vanilla BC*: raw data
   - *Gaussian*: temporal smoothing with chosen σ
   - *CUPID / CUPID-Q*: keep top-K% demos by score
   - *STRIDE*: train VAE → train DPO editor → edit data with latent residuals
   - *STRIDE w/o Influence*: same but with random influence scores
   - *STRIDE Random Edits*: same VAE but random noise instead of editor
8. **Train final BC** on processed data (200 epochs, cosine LR)
9. **Evaluate**: 20 episodes with video rendering
10. **Log**: wandb (training curves, eval metrics, videos) + JSON files

---

## 4. Algorithm Details

### TRAK Influence Scoring

For a policy π_θ with MSE loss L:

$$\tau(z_\text{train}, z_\text{test}) = \phi(z_\text{train})^\top (X^\top X + \lambda I)^{-1} \phi(z_\text{test})$$

where $\phi(z) = P \cdot \nabla_\theta L(z; \theta) / \sqrt{k}$ and $P$ is a Rademacher random projection matrix.

### CUPID Score

$$\text{CUPID}(d) = \sum_{j \in \text{success}} S[d, j] - \sum_{j \in \text{failure}} S[d, j]$$

### CUPID-Quality Score

$$Q(d) = 0.5 \cdot \text{sum\_of\_sum}(d) + 0.25 \cdot \text{min\_of\_max}(d) + 0.25 \cdot \text{max\_of\_min}(d)$$

### STRIDE Pipeline

1. **VAE Training**: $L_\text{VAE} = \|a - \hat{a}\|^2 + \beta \cdot \text{KL}(q(z|s,a) \| \mathcal{N}(0,I))$
2. **DPO Editor Training**: $L_\text{DPO} = -\log\sigma(\beta(r_w - r_l))$ where $r = -\|a' - a_\text{target}\|^2$
3. **Editing**: $z' = \mu(s,a) + \text{scale} \cdot \delta z$, $a' = (1-\alpha) \cdot a + \alpha \cdot D_\phi(z', s)$

---

## 5. Output Structure

```
results/
  all_results.json              # Aggregate results for all experiments
  experiment_results/
    pen_vanilla_bc_seed42.json       # Per-experiment detailed results
    pen_stride_seed42.json
    ...
  checkpoints/
    pen/
      ref_bc_policy.pt          # Reference BC checkpoint
      vae.pt                    # Trained VAE
      stride_editor.pt          # Trained DPO editor
      vanilla_bc_policy.pt      # Final vanilla BC
      stride_policy.pt          # Final STRIDE BC
      ...
  videos/
    pen/
      vanilla_bc/seed42/
        ep_000.mp4              # Rollout videos
        ep_001.mp4
        ...
      stride/seed42/
        ...
  scores/
    pen/
      cupid_scores.npy          # Per-demo CUPID scores
      cupid_quality_scores.npy  # Per-demo CUPID-Quality scores
  plots/
    results_pen.png             # Bar charts
    results_pen.pdf
    ...
```

---

## 6. Plotting Results

After experiments complete:

```bash
python -m experiments.plot_results --results results/all_results.json --output results/plots
```

Generates per-task bar charts of mean reward and success rate across all 14 methods.

---

## 7. Logging

All metrics are logged to:
- **wandb**: training curves, eval metrics, rollout videos, configs
  - Project: `stride`
  - Group: task name
  - Run name: `{task}_{method}_seed{seed}`
- **JSON files**: `results/experiment_results/{run_name}.json`
- **Console**: progress logging via Python `logging` module

### Key Logged Metrics

| Stage | Metrics |
|-------|---------|
| BC Training | `bc/train_loss`, `bc/val_loss`, `bc/lr` |
| VAE Training | `vae/train_recon`, `vae/train_kl`, `vae/val_recon`, `vae/beta` |
| Editor Training | `editor/dpo_loss`, `editor/cos_loss`, `editor/reg_loss`, `editor/pref_acc` |
| Evaluation | `eval/mean_reward`, `eval/std_reward`, `eval/success_rate`, `eval/mean_length` |
| Videos | `eval/video_ep000`, `eval/video_ep001`, … |

---

## 8. Hyperparameters

All hyperparameters are defined in `experiments/configs.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bc_epochs` | 200 | BC training epochs |
| `bc_lr` | 3e-4 | BC learning rate |
| `bc_hidden` | (256, 256) | BC MLP hidden layers |
| `vae_epochs` | 200 | VAE training epochs |
| `vae_latent_dim` | 16 | VAE latent dimension |
| `vae_beta` | 0.5 | VAE KL weight |
| `editor_epochs` | 100 | DPO editor training epochs |
| `editor_beta_dpo` | 2.0 | DPO β temperature |
| `editor_k_neighbors` | 10 | KNN for preference pairs |
| `edit_scale` | 0.6 | Latent edit scaling factor |
| `blend_alpha` | 0.35 | Edit blending weight |
| `n_aug` | 4 | Latent augmentation copies |
| `trak_proj_dim` | 512 | TRAK projection dimension |
| `trak_n_rollouts` | 100 | Rollout episodes for scoring |
| `n_eval_episodes` | 20 | Evaluation episodes |

