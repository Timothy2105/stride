"""TRAK-style influence score computation.

We approximate the influence of each training sample (s_i, a_i) on validation
loss using random (Johnson-Lindenstrauss) gradient projections, following the
TRAK framework (Park et al., 2023):

    I(s_i, a_i) ≈ − (G_val · G_train_i)

where G_val  = P^T ∇_θ J(θ)     (projected validation gradient, shape P)
      G_i    = P^T ∇_θ ℓ_i(θ)   (projected per-sample train gradient, shape P)
      P ∈ R^{|θ| × proj_dim}     is a fixed random projection matrix

This avoids explicit Hessian inversion while preserving the sign of influence:
  positive score  → sample helps reduce validation loss (keep / upweight)
  negative score  → sample hurts (down-weight or ignore)

Reference: Park et al., "TRAK: Attributing Model Behaviour at Scale", ICML 2023.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Gradient projection utilities
# ---------------------------------------------------------------------------

def _get_projection_matrix(
    param_dim: int,
    proj_dim: int,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a fixed random JL projection matrix P ∈ R^{param_dim × proj_dim}.

    Each column is drawn i.i.d. from N(0, 1/proj_dim) (scaled Gaussian).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    P = torch.randn(param_dim, proj_dim, generator=rng) / (proj_dim ** 0.5)
    return P.to(device)


def _compute_param_dim(model: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _flat_grad(loss: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Compute gradient of `loss` w.r.t. model parameters → flat vector."""
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                                  create_graph=False, allow_unused=True)
    parts = []
    for g in grads:
        if g is None:
            # Parameter didn't contribute; treat as zero.
            parts.append(torch.zeros_like(next(iter(model.parameters())).view(-1)[:0]))
        else:
            parts.append(g.detach().view(-1))
    return torch.cat(parts)


@torch.no_grad()
def _project(flat_grad: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Project flat gradient using P: result shape = (proj_dim,)."""
    return flat_grad @ P   # (param_dim,) @ (param_dim, proj_dim) → (proj_dim,)


# ---------------------------------------------------------------------------
# Main influence computation
# ---------------------------------------------------------------------------

def compute_influence_scores(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    proj_dim: int = 512,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Compute TRAK-style influence scores for every training sample.

    Parameters
    ----------
    model        : trained BC policy π_θ (called with (obs,) → act)
    train_loader : yields (obs, act, weight) batches from the training set
    val_loader   : yields (obs, act, weight) batches from the validation set
    proj_dim     : projection dimension P (default 512)
    seed         : random seed for the JL projection matrix
    device       : torch device

    Returns
    -------
    influence : np.ndarray of shape (N_train,)
                Higher = more positively influential on validation loss.
    """
    model = model.to(device)
    model.eval()

    param_dim = _compute_param_dim(model)
    P = _get_projection_matrix(param_dim, proj_dim, seed=seed, device=device)

    # ------------------------------------------------------------------
    # 1.  Projected validation gradient: G_val  (proj_dim,)
    # ------------------------------------------------------------------
    val_grad_accum = torch.zeros(proj_dim, device=device)
    n_val = 0

    for obs_b, act_b, _ in val_loader:
        obs_b, act_b = obs_b.to(device), act_b.to(device)
        model.zero_grad()
        pred = model(obs_b)
        loss = F.mse_loss(pred, act_b, reduction="sum")
        loss.backward()

        flat = torch.cat([p.grad.detach().view(-1) if p.grad is not None
                          else torch.zeros(p.numel(), device=device)
                          for p in model.parameters() if p.requires_grad])
        val_grad_accum += flat @ P
        n_val += obs_b.shape[0]

    # Average over validation samples
    G_val = val_grad_accum / max(n_val, 1)   # (proj_dim,)

    # ------------------------------------------------------------------
    # 2.  Per-sample projected training gradients + influence
    # ------------------------------------------------------------------
    all_influences: list[float] = []

    for obs_b, act_b, _ in train_loader:
        obs_b, act_b = obs_b.to(device), act_b.to(device)

        for i in range(obs_b.shape[0]):
            obs_i = obs_b[i : i + 1]
            act_i = act_b[i : i + 1]

            model.zero_grad()
            pred_i = model(obs_i)
            loss_i = F.mse_loss(pred_i, act_i, reduction="mean")
            loss_i.backward()

            flat_i = torch.cat([p.grad.detach().view(-1) if p.grad is not None
                                 else torch.zeros(p.numel(), device=device)
                                 for p in model.parameters() if p.requires_grad])
            G_i = flat_i @ P   # (proj_dim,)

            # influence approximation:  I_i ≈ − G_val · G_i
            score = -(G_val @ G_i).item()
            all_influences.append(score)

    return np.array(all_influences, dtype=np.float32)


def compute_influence_scores_batched(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    proj_dim: int = 512,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Batched variant – projects the per-sample gradient via functional API.

    Significantly faster than the loop version but requires more memory.
    Falls back gracefully to :func:`compute_influence_scores` on OOM.
    """
    try:
        return _batched_impl(model, train_loader, val_loader, proj_dim, seed, device)
    except torch.cuda.OutOfMemoryError:
        print("[TRAK] OOM in batched mode, falling back to per-sample loop.")
        return compute_influence_scores(model, train_loader, val_loader, proj_dim, seed, device)


def _batched_impl(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    proj_dim: int,
    seed: int,
    device: torch.device | str,
) -> np.ndarray:
    """Implementation helper for batched influence computation."""
    from torch.func import grad, vmap, functional_call  # type: ignore[import]

    model = model.to(device)
    model.eval()

    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    param_dim = sum(p.numel() for p in params.values())
    P = _get_projection_matrix(param_dim, proj_dim, seed=seed, device=device)

    def per_sample_loss(params_dict, obs_i, act_i):
        pred = functional_call(model, params_dict, (obs_i.unsqueeze(0),))[0]
        return F.mse_loss(pred, act_i, reduction="mean")

    grad_fn = grad(per_sample_loss)

    def project_grad(params_dict, obs_i, act_i):
        g = grad_fn(params_dict, obs_i, act_i)
        flat = torch.cat([v.view(-1) for v in g.values()])
        return flat @ P

    batched_pg = vmap(project_grad, in_dims=(None, 0, 0))

    # Validation gradient
    G_val = torch.zeros(proj_dim, device=device)
    n_val = 0
    for obs_b, act_b, _ in val_loader:
        obs_b, act_b = obs_b.to(device), act_b.to(device)
        projected = batched_pg(params, obs_b, act_b)  # (B, proj_dim)
        G_val += projected.sum(dim=0)
        n_val += obs_b.shape[0]
    G_val /= max(n_val, 1)

    # Training gradients + influence
    all_influences = []
    for obs_b, act_b, _ in train_loader:
        obs_b, act_b = obs_b.to(device), act_b.to(device)
        projected = batched_pg(params, obs_b, act_b)  # (B, proj_dim)
        scores = -(projected @ G_val)                  # (B,)
        all_influences.append(scores.detach().cpu().numpy())

    return np.concatenate(all_influences, axis=0).astype(np.float32)
