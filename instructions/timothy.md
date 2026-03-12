# Timothy's Experiment Runbook

Assigned methods:

| # | Method | Config name | Description |
|---|--------|-------------|-------------|
| 3 | Gaussian 50% | `gaussian_50` | Temporal Gaussian smoothing (σ=5.0) |
| 4 | Gaussian 75% | `gaussian_75` | Temporal Gaussian smoothing (σ=7.5) |
| 6 | CUPID 50% | `cupid_50` | Keep top 50% of demos by TRAK influence |
| 7 | CUPID 75% | `cupid_75` | Keep top 75% of demos by TRAK influence |
| 14 | BC + Influence Reweighting | `influence_reweight` | BC loss weighted by TRAK influence scores |

Tasks: `pen`, `hammer`, `door`, `relocate`
Trials per (method, task): 10
Total experiments: **5 methods × 4 tasks × 10 trials = 200**

---

## 1. Environment Setup

```bash
conda create -n stride_new python=3.10 -y
conda activate stride_new

pip install mujoco
pip install -r requirements.txt

wandb login
```

## 2. Full Runs

Run all 5 methods across all 4 tasks with 10 trials each.

```bash
export MUJOCO_GL=egl

# Pen
python -m experiments.run_experiments \
    --task pen \
    --method gaussian_50,gaussian_75,cupid_50,cupid_75,influence_reweight \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 50

# Hammer
python -m experiments.run_experiments \
    --task hammer \
    --method gaussian_50,gaussian_75,cupid_50,cupid_75,influence_reweight \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 50

# Door
python -m experiments.run_experiments \
    --task door \
    --method gaussian_50,gaussian_75,cupid_50,cupid_75,influence_reweight \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 50

# Relocate
python -m experiments.run_experiments \
    --task relocate \
    --method gaussian_50,gaussian_75,cupid_50,cupid_75,influence_reweight \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 50
```

## 3. Troubleshooting

- **Minari dataset not found**: Run `python -c "import minari; minari.load_dataset('D4RL/pen/human-v2', download=True)"` for each task (pen, hammer, door, relocate).
- **Gymnasium environment not found**: Make sure `gymnasium-robotics` is installed: `pip install gymnasium-robotics`.
