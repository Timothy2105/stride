"""Ablation runner: executes the full STRIDE pipeline with a given AblationConfig.

This module wraps every stage of the STRIDE pipeline — VAE training, BC training,
influence computation, editor training, dataset editing, and evaluation — and
threads the AblationConfig through each component.  The result is a single
`run_ablation(cfg) -> dict` function that returns all metrics.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from stride.ablation.config import AblationConfig
from stride.data.loader import load_pen_human, make_datasets, make_dataloaders
from stride.models.policy import BCPolicy
from stride.models.vae import ConditionalVAE
from stride.models.editor import LatentEditor
from stride.training.train_bc import train_bc
from stride.training.train_vae import train_vae
from stride.training.train_editor_dpo import train_editor_dpo, dpo_editor_loss
from stride.editing.edit import apply_stride
from stride.influence.trak import compute_influence_scores_batched, compute_influence_scores
from stride.influence.selection import normalise_influence_scores

# --- Baselines ---
from stride.baselines.gaussian_filter import smooth_actions_per_episode
from stride.baselines.random_latent import random_latent_edit


def run_ablation(
    cfg: AblationConfig,
    ckpt_dir: str = "ablations/checkpoints",
    verbose: bool = True,
) -> dict:
    """Run the full STRIDE pipeline with the given ablation config.

    Parameters
    ----------
    cfg      : AblationConfig with all hyperparameters
    ckpt_dir : directory for intermediate checkpoints
    verbose  : print progress logs

    Returns
    -------
    dict with keys:
        'config'   : serialized AblationConfig
        'metrics'  : dict of evaluation metrics (or empty if skip_eval)
        'timing'   : dict of per-stage wall-clock times
        'training' : dict of final training losses per stage
        'error'    : error message if the run failed, else None
    """
    result = {
        "config": cfg.to_dict(),
        "metrics": {},
        "timing": {},
        "training": {},
        "error": None,
    }

    # Apply smoke test overrides
    if cfg.smoke_test:
        cfg = cfg.apply_smoke_test()

    run_ckpt = Path(ckpt_dir) / cfg.name
    run_ckpt.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    seed = cfg.seed

    try:
        # ====================================================================
        # Stage 0: Load data
        # ====================================================================
        t0 = time.time()
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Ablation: {cfg.name}")
            print(f"  {cfg.description}")
            print(f"{'='*60}")
            print(f"\n[Ablation] Loading data: {cfg.task} ...")

        data = load_pen_human(cfg.task)
        
        # Baseline: Gaussian smoothing
        if cfg.use_gaussian_filter:
            if verbose:
                print(f"  [Ablation] Applying Gaussian smoothing (sigma={cfg.gaussian_sigma}) ...")
            data["actions"] = smooth_actions_per_episode(
                data["actions"], data["episode_ends"], sigma=cfg.gaussian_sigma
            )

        train_ds, val_ds, train_idx, val_idx = make_datasets(data, seed=seed)
        result["timing"]["data_load"] = time.time() - t0

        obs_dim = data["observations"].shape[1]
        act_dim = data["actions"].shape[1]

        # ====================================================================
        # Stage 1: Train initial BC policy (needed for influence computation)
        # ====================================================================
        t0 = time.time()
        if verbose:
            print(f"\n[Ablation] Training initial BC policy ...")

        bc_ckpt = str(run_ckpt / "bc_policy.pt")
        bc_policy = train_bc(
            data=data,
            epochs=cfg.epochs_bc,
            lr=cfg.lr_bc,
            batch_size=cfg.batch_size,
            device_str=device,
            out_path=bc_ckpt,
            use_weights=False,
            hidden=cfg.policy_hidden,
            use_cosine_lr=cfg.use_cosine_lr,
            obs_noise_std=cfg.obs_noise_std,
            seed=seed,
            verbose=verbose,
        )
        result["timing"]["train_bc_initial"] = time.time() - t0

        # ====================================================================
        # If editor is disabled, just use vanilla BC
        # ====================================================================
        if not cfg.use_editor and not cfg.use_random_latent and not cfg.use_influence_weighting:
            if verbose:
                print(f"\n[Ablation] Baseline run — using vanilla BC")
            result["training"]["method"] = "vanilla_bc"
            result["timing"]["total_training"] = sum(result["timing"].values())

            if not cfg.skip_eval:
                result["metrics"] = _evaluate(bc_ckpt, cfg, verbose)
            return result

        # ====================================================================
        # Stage 2: Train VAE
        # ====================================================================
        t0 = time.time()
        if verbose:
            print(f"\n[Ablation] Training VAE ...")

        vae_ckpt = str(run_ckpt / "vae.pt")
        vae = train_vae(
            data=data,
            epochs=cfg.epochs_vae,
            lr=cfg.lr_vae,
            batch_size=cfg.batch_size,
            latent_dim=cfg.vae_latent_dim,
            target_beta=cfg.vae_beta,
            anneal_epochs=cfg.vae_anneal_epochs,
            device_str=device,
            out_path=vae_ckpt,
            seed=seed,
            verbose=verbose,
        )
        result["timing"]["train_vae"] = time.time() - t0

        # ====================================================================
        # Stage 3: Train DPO latent editor
        # ====================================================================
        t0 = time.time()
        if verbose:
            print(f"\n[Ablation] Training DPO editor ...")

        editor_ckpt = str(run_ckpt / "editor.pt")

        # Train editor with config toggles + Ranking DPO support
        editor, influence_raw = _train_editor_with_config(
            cfg, data, vae, bc_policy, editor_ckpt, verbose
        )
        result["timing"]["train_editor"] = time.time() - t0

        # ====================================================================
        # Stage 4: Apply Edits (STRIDE or Random Latent)
        # ====================================================================
        t0 = time.time()
        
        if cfg.use_random_latent:
            if verbose:
                print(f"\n[Ablation] Applying Random Latent perturbations ...")
            edited_actions = random_latent_edit(
                data["observations"][train_idx],
                data["actions"][train_idx],
                vae,
                noise_std=cfg.latent_noise_std,
                device_str=device,
                seed=seed,
            )
            edited_data = {
                "observations": data["observations"][train_idx],
                "actions": edited_actions,
            }
        elif cfg.use_editor:
            if verbose:
                print(f"\n[Ablation] Applying STRIDE edits ...")
            n_aug = cfg.n_aug if cfg.use_augmentation else 0
            edited_data = apply_stride(
                data=data,
                train_idx=train_idx,
                vae=vae,
                editor=editor,
                edit_scale=cfg.edit_scale,
                blend_alpha=cfg.blend_alpha,
                n_aug=n_aug,
                aug_noise_std=cfg.aug_noise_std,
                device_str=device,
                seed=seed,
                verbose=verbose,
            )
        else:
            # No editing (Vanilla or Influence Weighted)
            edited_data = {
                "observations": data["observations"][train_idx],
                "actions": data["actions"][train_idx],
            }
            if verbose:
                print("\n[Ablation] No data editing applied")

        result["timing"]["editing"] = time.time() - t0

        # ====================================================================
        # Stage 5: Train final BC on edited data
        # ====================================================================
        t0 = time.time()
        
        # Handle influence weighting for BC loss if requested
        use_weights = False
        influence_weights = None
        if cfg.use_influence_weighting:
            if verbose:
                print(f"  [Ablation] Using influence weighting for BC loss ...")
            use_weights = True
            _, weights = normalise_influence_scores(influence_raw)
            weights = weights + cfg.soft_min_weight
            
            # Align weights with the full dataset for train_bc
            influence_weights = np.ones(len(data["observations"]), dtype=np.float32)
            influence_weights[train_idx] = weights

        if verbose:
            print(f"\n[Ablation] Training final BC ...")

        stride_bc_ckpt = str(run_ckpt / "stride_bc.pt")
        final_bc = train_bc(
            data=edited_data,
            val_data=data,
            epochs=cfg.epochs_bc,
            lr=cfg.lr_bc,
            batch_size=cfg.batch_size,
            device_str=device,
            out_path=stride_bc_ckpt,
            use_weights=use_weights,
            influence_weights=influence_weights,
            hidden=cfg.policy_hidden,
            use_cosine_lr=cfg.use_cosine_lr,
            obs_noise_std=cfg.obs_noise_std,
            use_lspo=cfg.use_lspo,
            vae=vae if (cfg.use_lspo or cfg.use_random_latent) else None,
            vae_ckpt=vae_ckpt if (cfg.use_lspo or cfg.use_random_latent) else None,
            seed=seed,
            verbose=verbose,
        )
        result["timing"]["train_bc_final"] = time.time() - t0
        result["training"]["method"] = "stride" if (cfg.use_editor or cfg.use_lspo) else "baseline"

        # ====================================================================
        # Stage 6: Evaluate
        # ====================================================================
        if not cfg.skip_eval:
            t0 = time.time()
            result["metrics"] = _evaluate(stride_bc_ckpt, cfg, verbose)
            result["timing"]["evaluation"] = time.time() - t0

        result["timing"]["total_training"] = sum(
            v for k, v in result["timing"].items() if k != "evaluation"
        )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        if verbose:
            print(f"\n[Ablation] ERROR in '{cfg.name}': {e}")

    return result


def _train_editor_with_config(
    cfg: AblationConfig,
    data: dict,
    vae: ConditionalVAE,
    bc_policy: BCPolicy,
    out_path: str,
    verbose: bool,
) -> tuple:
    """Train editor with ablation config loss toggles.

    This wraps train_editor_dpo but passes through the ablation toggles
    for individual loss components.
    """
    from stride.training.train_editor_dpo import (
        DPOEditorDataset,
        train_editor_dpo as _original_train,
    )
    import torch.optim as optim

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu"
                          else "cpu")
    seed = cfg.seed
    torch.manual_seed(seed)

    train_ds, val_ds, train_idx, val_idx = make_datasets(data, seed=seed)
    train_loader, val_loader = make_dataloaders(train_ds, val_ds, batch_size=cfg.batch_size)

    obs_dim = data["observations"].shape[1]
    act_dim = data["actions"].shape[1]

    vae = vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    latent_dim = vae.latent_dim

    bc_policy = bc_policy.to(device)
    bc_policy.eval()

    # -- Compute influence scores --
    if cfg.use_influence:
        if verbose:
            print("  [Editor] Computing TRAK influence scores ...")
        try:
            influence_raw = compute_influence_scores_batched(
                bc_policy, train_loader, val_loader,
                proj_dim=cfg.proj_dim, seed=seed, device=device,
            )
        except Exception:
            influence_raw = compute_influence_scores(
                bc_policy, train_loader, val_loader,
                proj_dim=cfg.proj_dim, seed=seed, device=device,
            )
    else:
        # Random influence scores (no real influence computation)
        if verbose:
            print("  [Editor] Influence disabled — using random scores")
        n_train = len(train_ds)
        influence_raw = np.random.default_rng(seed).standard_normal(n_train).astype(np.float32)

    influence_corrected = -influence_raw

    # -- VAE latent means for KNN --
    train_obs_np = data["observations"][train_idx]
    train_act_np = data["actions"][train_idx]

    with torch.no_grad():
        obs_t = torch.from_numpy(train_obs_np).float().to(device)
        act_t = torch.from_numpy(train_act_np).float().to(device)
        mu, _ = vae.encode(obs_t, act_t)
        latent_means = mu.cpu().numpy()

    # -- Preference pairs --
    from stride.influence.selection import compute_preference_pairs, compute_corrective_directions

    winners, losers, valid = compute_preference_pairs(
        actions=train_act_np,
        influence_scores_corrected=influence_corrected,
        embeddings=latent_means,
        k=cfg.k_neighbors,
    )

    # -- Ranking DPO data --
    if cfg.use_ranking_dpo:
        from stride.influence.selection import compute_ranking_data
        if verbose:
            print("  [Editor] Computing R-DPO ranking data ...")
        rank_act, rank_sco = compute_ranking_data(
            actions=train_act_np,
            influence_scores_corrected=influence_corrected,
            embeddings=latent_means,
            k=cfg.k_neighbors,
        )
    else:
        rank_act, rank_sco = None, None

    _, influence_weights = normalise_influence_scores(-influence_raw)
    directions = compute_corrective_directions(
        observations=train_obs_np,
        actions=train_act_np,
        influence_weights=influence_weights,
        embeddings=latent_means,
        k=cfg.k_neighbors,
    )

    # -- Build editor dataset --
    editor_ds = DPOEditorDataset(
        train_obs_np, train_act_np, winners, losers, valid, directions,
        neigh_actions=rank_act, neigh_scores=rank_sco
    )
    from torch.utils.data import DataLoader
    editor_loader = DataLoader(editor_ds, batch_size=cfg.batch_size,
                                shuffle=True, drop_last=False)

    # -- Initialize editor --
    editor = LatentEditor(
        obs_dim=obs_dim, act_dim=act_dim,
        latent_dim=latent_dim,
        hidden=cfg.editor_hidden,
    ).to(device)
    optimizer = optim.Adam(editor.parameters(), lr=cfg.lr_editor)

    best_loss = float("inf")
    best_state = None
    epochs = cfg.epochs_editor

    # Handle Optional lambda weights: use 0.0 if None
    lambda_reg = cfg.lambda_reg if cfg.lambda_reg is not None else 0.0
    lambda_cos = cfg.lambda_cos if cfg.lambda_cos is not None else 0.0

    for epoch in range(1, epochs + 1):
        editor.train()
        t0 = time.time()
        epoch_info = {"dpo": 0, "rank": 0, "cos": 0, "reg": 0, "total": 0, "pref_acc": 0}
        n = 0

        for batch in editor_loader:
            if cfg.use_ranking_dpo:
                obs_b, act_b, win_b, los_b, valid_b, dir_b, rank_act_b, rank_sco_b = [
                    x.to(device) for x in batch
                ]
            else:
                obs_b, act_b, win_b, los_b, valid_b, dir_b = [
                    x.to(device) for x in batch
                ]
                rank_act_b, rank_sco_b = None, None

            a_prime, dz = editor.edit(obs_b, act_b, vae)

            loss, info = dpo_editor_loss(
                a_prime, act_b, win_b, los_b, valid_b, dir_b, dz,
                neigh_actions=rank_act_b,
                neigh_scores=rank_sco_b,
                beta=cfg.dpo_beta,
                lambda_reg=lambda_reg,
                lambda_cos=lambda_cos,
                use_dpo=cfg.use_dpo_loss,
                use_cosine=cfg.use_cosine_loss,
                use_reg=cfg.use_reg_loss,
                use_ranking=cfg.use_ranking_dpo,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = obs_b.shape[0]
            for key in epoch_info:
                epoch_info[key] += info[key] * bs
            n += bs

        for key in epoch_info:
            epoch_info[key] /= max(n, 1)

        if epoch_info["total"] < best_loss:
            best_loss = epoch_info["total"]
            best_state = {k: v.cpu().clone() for k, v in editor.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  [Editor] epoch {epoch:4d}/{epochs}  "
                  f"total={epoch_info['total']:.4f}  "
                  f"dpo={epoch_info['dpo']:.4f}  "
                  f"cos={epoch_info['cos']:.4f}  "
                  f"reg={epoch_info['reg']:.4f}  "
                  f"pref_acc={epoch_info['pref_acc']:.1%}  "
                  f"({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "latent_dim": latent_dim,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, out_path)

    if best_state is not None:
        editor.load_state_dict(best_state)
    return editor.to("cpu"), influence_raw


def _evaluate(ckpt_path: str, cfg: AblationConfig, verbose: bool) -> dict:
    """Evaluate a trained policy checkpoint."""
    if verbose:
        print(f"\n[Ablation] Evaluating {cfg.name} ...")

    try:
        from stride.eval.evaluate import evaluate_from_checkpoint

        # Determine env name from task
        task = cfg.task
        if "pen" in task:
            env_name = "AdroitHandPen-v1"
        elif "door" in task:
            env_name = "AdroitHandDoor-v1"
        elif "hammer" in task:
            env_name = "AdroitHandHammer-v1"
        elif "relocate" in task:
            env_name = "AdroitHandRelocate-v1"
        else:
            env_name = "AdroitHandPen-v1"

        stats = evaluate_from_checkpoint(
            ckpt_path,
            n_episodes=cfg.n_eval_episodes,
            env_name=env_name,
            seed=cfg.seed,
            device_str=cfg.device,
        )
        if verbose:
            print(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}  "
                  f"Success: {stats['success_rate']:.1%}")
        return stats
    except Exception as e:
        if verbose:
            print(f"  [Ablation] Evaluation failed: {e}")
        return {"error": str(e)}


def save_ablation_result(result: dict, results_dir: str = "ablations/results") -> str:
    """Save a single ablation result to a JSON file.

    Returns the path to the saved file.
    """
    os.makedirs(results_dir, exist_ok=True)
    name = result["config"]["name"]
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return path


def load_all_results(results_dir: str = "ablations/results") -> list[dict]:
    """Load all ablation results from a results directory."""
    results = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return results
    for f in sorted(results_path.glob("*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results
