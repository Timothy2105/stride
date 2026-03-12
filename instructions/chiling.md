# Chiling's Experiment Runbook

Assigned methods:

| # | Method | Config name | Description |
|---|--------|-------------|-------------|
| 8 | CUPID-Quality 25% | `cupid_quality_25` | Keep top 25% by CUPID-Quality ensemble score |
| 9 | CUPID-Quality 50% | `cupid_quality_50` | Keep top 50% by CUPID-Quality ensemble score |
| 10 | CUPID-Quality 75% | `cupid_quality_75` | Keep top 75% by CUPID-Quality ensemble score |
| 12 | STRIDE w/o Influence | `stride_no_influence` | STRIDE pipeline with random influence scores |
| 13 | STRIDE Random Edits | `stride_random_edits` | STRIDE with random latent noise instead of trained editor |

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

---

## 2. Full Runs

Run all 5 methods across all 4 tasks with 10 trials each.

```bash
export MUJOCO_GL=egl

# Pen
python -m experiments.run_experiments \
    --task pen \
    --method cupid_quality_25,cupid_quality_50,cupid_quality_75,stride_no_influence,stride_random_edits \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 50

# Hammer
python -m experiments.run_experiments \
    --task hammer \
    --method cupid_quality_25,cupid_quality_50,cupid_quality_75,stride_no_influence,stride_random_edits \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 50

# Door
python -m experiments.run_experiments \
    --task door \
    --method cupid_quality_25,cupid_quality_50,cupid_quality_75,stride_no_influence,stride_random_edits \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 50

# Relocate
python -m experiments.run_experiments \
    --task relocate \
    --method cupid_quality_25,cupid_quality_50,cupid_quality_75,stride_no_influence,stride_random_edits \
    --device cuda \
    --seed 42 \
    --n-trials 10 \
    --n-eval-episodes 50
```

---

## 3. Troubleshooting

- **Minari dataset not found**: Run `python -c "import minari; minari.load_dataset('D4RL/pen/human-v2', download=True)"` for each task (pen, hammer, door, relocate).
- **Gymnasium environment not found**: Make sure `gymnasium-robotics` is installed: `pip install gymnasium-robotics`.
