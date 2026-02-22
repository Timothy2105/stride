# STRIDE: Strategic Trajectory Refinement via Influence-guided Data Editing

STRIDE converts suboptimal human demonstrations into high-utility training data for imitation learning by combining **TRAK influence estimation** with **DPO-based latent-space editing**.

## Mathematical Formulation

### 1. Influence Estimation
We approximate the influence of sample $z_i$ on validation loss $\mathcal{L}(z_{val})$ using TRAK-style random gradient projections $P \in \mathbb{R}^{d \times p}$:
$$I_i = - \left( P \nabla_\theta \mathcal{L}(\theta, z_{val}) \right) \cdot \left( P \nabla_\theta \mathcal{L}(\theta, z_i) \right)$$
Samples with $I_i > 0$ are considered harmful, while $I_i < 0$ are helpful.

### 2. Preference Pair Extraction
For each training sample $a_i$, we find $k$-nearest neighbours $N(i)$ in VAE latent space. We define preference pairs $(a_{winner}, a_{loser})$ as:
$$a_{winner} = a_{\text{argmin } I_j, j \in N(i)}, \quad a_{loser} = a_{\text{argmax } I_j, j \in N(i)}$$

### 3. DPO Latent Editor
The editor $g_\psi(s, a, \xi) \to \delta z$ is trained via **Direct Preference Optimization** to push edited actions $a' = D_\phi(\mu(s, a) + \delta z, s)$ toward winners:
$$\mathcal{L}_{DPO} = -\log \sigma \left( \beta \left( \|a' - a_{loser}\|^2 - \|a' - a_{winner}\|^2 \right) \right)$$
The final objective combines DPO with directional alignment and regularization:
$$\mathcal{L}_{total} = \mathcal{L}_{DPO} + \lambda_{cos} (1 - \text{cos\_sim}(a' - a_{orig}, \Delta a_{target})) + \lambda_{reg} \|\delta z\|^2$$

### 4. Two-Stage Dataset Synthesis
First, we apply convex blending with factor $\alpha$:
$$a_{corrected} = (1 - \alpha) a_{orig} + \alpha a'$$
Then, we perform latent-space augmentation by sampling $n_{aug}$ copies with noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$:
$$\mathcal{D}' = \mathcal{D}_{corrected} \cup \{ (s, D_\phi(\mu + \epsilon, s)) \}_{n_{aug}}$$

## Experiment Results (AdroitHandPen-v1)

Evaluation on the `pen-human-v2` dataset (5000 steps). STRIDE is compared against baselines under equal compute constraints (100 epochs, no custom LR schedules).

| Method | Mean Reward | Std Dev | Success % |
| :--- | :---: | :---: | :---: |
| Vanilla BC | 5737.00 | 4758.99 | 55.0% |
| Gaussian Filter BC | 5839.42 | 4847.40 | 55.0% |
| Influence-Weighted BC | 4856.30 | 4551.14 | 50.0% |
| Random Latent BC | 5683.68 | 4521.58 | 65.0% |
| **STRIDE (DPO + Aug)** | **6658.40** | **4713.31** | **70.0%** |

## Project Structure

```bash
stride/
├── influence/   # TRAK scores & KNN preference extraction
├── models/      # BCPolicy, CVAE, and LatentEditor g_ψ
├── training/    # DPO-based editor training & BC loops
└── editing/     # Two-stage dataset refinement pipeline
```

## Quick Start
```bash
python experiments/run_all.py --device cpu
```
