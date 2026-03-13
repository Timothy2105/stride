# STRIDE: Strategic Trajectory Refinement via Influence-guided Data Editing

**Stanford CS229 Project** | Chiling Han, Timothy Yu, Yash Ranjith

Behavior Cloning assumes optimal demonstrations, yet real-world robotic datasets are frequently noisy and suboptimal. STRIDE refines existing demonstrations in a learned latent space using influence-guided editing, rather than naively filtering or discarding bad trajectories. Evaluated on Adroit Hand dexterous manipulation benchmarks, STRIDE consistently outperforms Vanilla BC, Gaussian filtering, and CUPID.

---

## Architecture

<p align="center">
  <img src="figures/stride-architecture.png" width="800"/>
</p>

STRIDE is a two-phase framework:
- **Stage 1 (Influence Estimation):** A naive BC policy is trained, then TRAK-based influence scores identify which training samples help or hurt validation performance. A conditional VAE encodes state-action pairs into a latent space, and nearest neighbors in that space form preference pairs.
- **Stage 2 (Action Editing):** A DPO-trained latent residual editor predicts corrective perturbations, guided by influence-derived preference pairs. Edited actions are decoded back to action space and used to train the final BC policy.

---

## Method

### Behavioral Cloning Objective

Given a demonstration dataset $\mathcal{D} = \{(s_i, a_i)\}_{i=1}^{N}$, BC learns a policy $\pi_\theta$ by minimizing:

$$\Large \mathcal{L}_{\text{BC}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \|\pi_\theta(s_i) - a_i\|^2$$

### Conditional VAE

We learn a latent mapping via a Conditional VAE. The encoder maps $(s, a)$ to a latent vector $z \in \mathbb{R}^{d_z}$, trained with the $\beta$-VAE loss:

$$\Large \mathcal{L}_{\text{VAE}} = \|a - \hat{a}\|^2 - \frac{\beta(t)}{2} \sum_{j=1}^{d_z} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

where $\beta(t)$ is linearly annealed from 0 to $\beta_{\max}$ over $T_{\text{anneal}}$ epochs.

### DPO-Trained Latent Residual Editor

A residual editor $g_\psi$ predicts a corrective perturbation $\delta z_i$ in latent space. The edited action is decoded from the shifted latent:

$$\Large z_i' = \mu(s_i, a_i) + g_\psi(s_i, a_i, \xi), \quad a_i' = D_\phi(z_i', s_i)$$

**Preference pairs** are constructed from $k$-nearest neighbors in the VAE latent space. The *winner* $a^w$ has the highest influence score, and the *loser* $a^l$ has the lowest:

$$\Large a_i^w = a_{j^*}, \quad a_i^l = a_{j_*}, \quad \text{where } j^* = \arg\max_{j \in \mathcal{N}_k(i)} I_j, \;\; j_* = \arg\min_{j \in \mathcal{N}_k(i)} I_j$$

The editor is trained with a DPO objective that encourages edits toward the winner and away from the loser:

$$\Large \mathcal{L}_{\text{DPO}} = -\frac{1}{\sum_i v_i} \sum_{i=1}^{N} v_i \cdot \log \sigma\!\Big(\beta_{\text{DPO}}\big(\|a_i' - a_i^l\|^2 - \|a_i' - a_i^w\|^2\big)\Big)$$

A cosine alignment loss steers edits toward influence-weighted neighbors:

$$\Large \Delta a_i^{\text{target}} = \frac{\sum_{j \in \mathcal{N}_k(i)} w_j(a_j - a_i)}{\|\sum_{j \in \mathcal{N}_k(i)} w_j(a_j - a_i)\| + \epsilon}, \quad \mathcal{L}_{\text{cos}} = \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}}\left(1 - \frac{(a_i' - a_i) \cdot \Delta a_i^{\text{target}}}{\|a_i' - a_i\| \cdot \|\Delta a_i^{\text{target}}\|}\right)$$

The total editor objective is $\mathcal{L}_{\text{editor}} = \mathcal{L}_{\text{DPO}} + \lambda_{\text{cos}}\mathcal{L}_{\text{cos}} + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}$, where $\mathcal{L}_{\text{reg}} = \|\delta z_i\|^2$ discourages excessively large edits.

---

## Results

### Quantitative Results

Performance (task success rate %) on Adroit Hand benchmarks, averaged over 10 seeds (50 rollouts each):

| Method | Hand-Pen | Hand-Door | Hand-Hammer | Hand-Relocate |
|:-------|:--------:|:---------:|:-----------:|:-------------:|
| Vanilla BC | 71.5 | 8.0 | 0.5 | 0.5 |
| BC + Influence Reweighting | 69.3 | 15.8 | 4.2 | 3.8 |
| Gaussian Filtering (25%) | 64.0 | 21.0 | 0.0 | 0.5 |
| Gaussian Filtering (50%) | 34.5 | 22.4 | 0.8 | 8.4 |
| Gaussian Filtering (75%) | 17.6 | 20.6 | 0.0 | 8.0 |
| CUPID (25%) | 33.5 | 11.5 | 0.5 | 0.0 |
| CUPID (50%) | 66.7 | 9.2 | 2.8 | 1.0 |
| CUPID (75%) | 64.4 | 10.0 | 4.8 | 2.4 |
| CUPID-Quality (25%) | 32.0 | 0.4 | 2.6 | 0.2 |
| CUPID-Quality (50%) | 68.0 | 9.2 | 2.8 | 1.0 |
| CUPID-Quality (75%) | 64.2 | 10.0 | 4.8 | 2.4 |
| **STRIDE (Ours)** | **83.0** | **49.8** | **20.2** | **13.8** |
| *STRIDE w/o Influence Obj.* | *64.6* | *7.4* | *0.4* | *11.6* |
| *Random Latent Edit Control* | *68.0* | *24.2* | *3.2* | *6.2* |

STRIDE outperforms all baselines on every task, achieving **83.0%** on Pen, **49.8%** on Door, **20.2%** on Hammer, and **13.8%** on Relocate.

### Edited Trajectories

Original vs. STRIDE-edited trajectories across all four Adroit tasks:

<p align="center">
  <img src="figures/pen_traj.png" width="220"/>
  <img src="figures/door_traj.png" width="220"/>
  <img src="figures/hammer_traj.png" width="220"/>
  <img src="figures/relocate_traj.png" width="220"/>
</p>
<p align="center">
  <b>(a)</b> Pen Rotation &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
  <b>(b)</b> Door Open &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
  <b>(c)</b> Hammer Nail &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
  <b>(d)</b> Relocate Ball
</p>

STRIDE edits localized suboptimal segments while preserving the surrounding trajectory structure. On Door, the editor discovers an elevated approach arc not seen in original demos (+41.8% over Vanilla BC). On Hammer, noisy approach paths are replaced with straighter strike trajectories. On Relocate, high-frequency jitter in the grasp-and-transport phase is suppressed.

---

## Setup

### Environment

```bash
conda create -n stride_new python=3.10 -y
conda activate stride_new

# Install MuJoCo (if not already installed)
pip install mujoco

# Install all Python dependencies
pip install -r requirements.txt
```

**Dependencies** (see `requirements.txt`):
- `torch>=2.0` -- deep learning (requires `torch.func` for TRAK)
- `gymnasium>=0.29` + `gymnasium-robotics>=1.2` -- Adroit environments
- `minari>=0.4` -- D4RL dataset loading
- `mujoco>=3.0` -- physics simulation
- `wandb` -- experiment logging
- `imageio` + `imageio-ffmpeg` -- video saving
- `scipy`, `scikit-learn`, `matplotlib`, `numpy`, `h5py`

Verify the environment:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import gymnasium; import gymnasium_robotics; gymnasium.make('AdroitHandPen-v1')"
python -c "import minari; minari.load_dataset('D4RL/pen/human-v2', download=True)"
```

### Running Experiments

**Smoke Test** (run first to verify everything works):

```bash
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

**Full Run** (all 4 tasks x 14 methods x 10 trials):

```bash
python -m experiments.run_experiments \
    --task all \
    --method all \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 20 \
    --trak-n-rollouts 100
```

**Run specific methods:**

```bash
# Baselines only
python -m experiments.run_experiments --task pen \
    --method vanilla_bc,gaussian_25,gaussian_50,gaussian_75 --n-trials 10

# CUPID variants only
python -m experiments.run_experiments --task pen \
    --method cupid_25,cupid_50,cupid_75,cupid_quality_25,cupid_quality_50,cupid_quality_75 --n-trials 10

# STRIDE and ablations only
python -m experiments.run_experiments --task pen \
    --method stride,stride_no_influence,stride_random_edits,influence_reweight --n-trials 10
```

**Options:**
- `--no-wandb` -- disable wandb logging
- `--no-video` -- disable video rendering (faster evaluation)

### Repository Structure

```
stride/
  data.py                # Minari data loading for Adroit human-v2 tasks
  influence.py           # KNN-based corrective directions and preference pairs
  editing.py             # STRIDE two-stage editing pipeline
  scoring.py             # TRAK influence scoring
  models/
    policy.py            # MLP Behavioral Cloning policy
    vae.py               # Conditional VAE for action representations
    editor.py            # Latent-space residual editor
  training/
    train_bc.py          # BC training with wandb logging
    train_vae.py         # VAE training with beta-annealing
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
