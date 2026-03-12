"""TRAK-based influence scoring for data attribution.

Implements the TRAK algorithm (Park et al., 2023) for computing
per-demonstration influence scores.  Used to power both the CUPID
and CUPID-Quality baselines.

Algorithm overview
------------------
For a trained model f_θ with loss L:

1. **Featurise**: For each sample z_i, compute the gradient
       g_i = ∇_θ L(z_i; θ)
   and project it with a Rademacher random matrix P ∈ ℝ^{k×d}:
       φ_i = P · g_i / √k

2. **Form Gram matrix**: Collect all training features into
       X ∈ ℝ^{n_train × k}
   and compute  Q = (X^T X + λI)^{-1}

3. **Score**: For each test sample z_j, compute projected gradient
       ψ_j = P · ∇_θ L(z_j; θ) / √k
   and the influence of training sample i on test sample j:
       τ(z_i, z_j) = φ_i^T Q ψ_j

CUPID scoring
-------------
Given a (n_train_demos × n_test_episodes) influence matrix S:

  CUPID(d) = Σ_{j ∈ succ} S[d, j]  −  Σ_{j ∈ fail} S[d, j]

CUPID-Quality scoring
---------------------
Three aggregation methods are ensembled:

  Q(d) = 0.5 · sum_of_sum(d) + 0.25 · min_of_max(d) + 0.25 · max_of_min(d)

where the aggregation methods differ in how transition-level scores
are collapsed to the (demo, test_episode) level.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-sample gradient computation
# ---------------------------------------------------------------------------

def _compute_per_sample_grads_loop(
    model: torch.nn.Module,
    obs: torch.Tensor,
    act: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Compute per-sample gradients of MSE loss using a simple loop.

    This is a fallback when torch.func is unavailable. It processes
    samples in small batches with individual backward passes.

    Returns
    -------
    grads : (N, n_params) flattened per-sample gradient vectors.
    """
    model = model.to(device).eval()
    param_list = [p for p in model.parameters() if p.requires_grad]

    all_grads = []
    for start in range(0, len(obs), batch_size):
        end = min(start + batch_size, len(obs))
        obs_b = obs[start:end].to(device)
        act_b = act[start:end].to(device)
        B = obs_b.shape[0]

        batch_grads = []
        for i in range(B):
            model.zero_grad()
            pred = model(obs_b[i: i + 1])
            loss = F.mse_loss(pred, act_b[i: i + 1])
            loss.backward()
            grad_vec = torch.cat([p.grad.detach().flatten() for p in param_list])
            batch_grads.append(grad_vec)

        all_grads.append(torch.stack(batch_grads).cpu())

    return torch.cat(all_grads, dim=0)


def _compute_per_sample_grads_vmap(
    model: torch.nn.Module,
    obs: torch.Tensor,
    act: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Compute per-sample gradients using torch.func.vmap (PyTorch ≥ 2.0).

    This is the efficient vectorised implementation.

    Returns
    -------
    grads : (N, n_params) flattened per-sample gradient vectors.
    """
    from torch.func import grad, vmap, functional_call

    model = model.to(device).eval()
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    # Ordered parameter names (for consistent flattening)
    param_names = list(params.keys())

    def loss_fn(params_dict, buffers_dict, obs_single, act_single):
        pred = functional_call(model, (params_dict, buffers_dict),
                               (obs_single.unsqueeze(0),))
        return F.mse_loss(pred.squeeze(0), act_single)

    ft_grad = grad(loss_fn)
    ft_per_sample = vmap(ft_grad, in_dims=(None, None, 0, 0))

    all_grads = []
    for start in range(0, len(obs), batch_size):
        end = min(start + batch_size, len(obs))
        obs_b = obs[start:end].to(device)
        act_b = act[start:end].to(device)

        sample_grads = ft_per_sample(params, buffers, obs_b, act_b)
        flat = torch.cat(
            [sample_grads[k].reshape(obs_b.shape[0], -1) for k in param_names],
            dim=1,
        )
        all_grads.append(flat.cpu())

    return torch.cat(all_grads, dim=0)


def compute_per_sample_grads(
    model: torch.nn.Module,
    obs: torch.Tensor,
    act: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Compute per-sample gradients, using vmap if available."""
    try:
        return _compute_per_sample_grads_vmap(model, obs, act, device, batch_size)
    except Exception:
        logger.info("[TRAK] vmap not available or failed; falling back to loop method")
        return _compute_per_sample_grads_loop(model, obs, act, device, batch_size)


# ---------------------------------------------------------------------------
# TRAK Scorer
# ---------------------------------------------------------------------------

class TRAKScorer:
    """Compute TRAK influence scores between training and test data.

    Parameters
    ----------
    model      : trained model with a standard ``forward(obs) → action`` API.
    proj_dim   : dimensionality of the random projection.
    lambda_reg : ridge regularisation strength for the Gram inversion.
    seed       : random seed for the projection matrix.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        proj_dim: int = 512,
        lambda_reg: float = 1e-3,
        seed: int = 42,
    ):
        self.model = model
        self.proj_dim = proj_dim
        self.lambda_reg = lambda_reg
        self.seed = seed

        # Total number of trainable parameters
        self.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Rademacher random projection matrix
        rng = np.random.default_rng(seed)
        signs = rng.choice(
            [-1.0, 1.0], size=(proj_dim, self.n_params),
        ).astype(np.float32)
        self.proj_matrix = torch.from_numpy(
            (signs / np.sqrt(proj_dim)).astype(np.float32)
        )

    def _project(self, grads: torch.Tensor) -> torch.Tensor:
        """Project (N, n_params) gradient matrix → (N, proj_dim)."""
        return grads.float() @ self.proj_matrix.T  # (N, proj_dim)

    def compute_transition_scores(
        self,
        train_obs: np.ndarray,
        train_act: np.ndarray,
        test_obs: np.ndarray,
        test_act: np.ndarray,
        device: str = "cpu",
        grad_batch_size: int = 128,
    ) -> np.ndarray:
        """Compute transition-level TRAK influence scores.

        Returns
        -------
        scores : (N_train, N_test) influence matrix.
        """
        dev = torch.device(device)

        logger.info("[TRAK] Computing per-sample gradients for training data …")
        train_grads = compute_per_sample_grads(
            self.model,
            torch.from_numpy(train_obs).float(),
            torch.from_numpy(train_act).float(),
            dev, grad_batch_size,
        )
        train_feats = self._project(train_grads)  # (N_train, k)

        logger.info("[TRAK] Computing per-sample gradients for test data …")
        test_grads = compute_per_sample_grads(
            self.model,
            torch.from_numpy(test_obs).float(),
            torch.from_numpy(test_act).float(),
            dev, grad_batch_size,
        )
        test_feats = self._project(test_grads)  # (N_test, k)

        logger.info("[TRAK] Solving ridge system …")
        XtX = train_feats.T @ train_feats  # (k, k)
        Q = torch.linalg.inv(
            XtX + self.lambda_reg * torch.eye(self.proj_dim)
        )  # (k, k)

        scores = (train_feats @ Q @ test_feats.T).numpy()  # (N_train, N_test)
        logger.info(
            f"[TRAK] Score matrix shape: {scores.shape}  "
            f"range: [{scores.min():.4f}, {scores.max():.4f}]"
        )
        return scores

    def compute_demo_scores(
        self,
        train_obs: np.ndarray,
        train_act: np.ndarray,
        train_episode_ends: list[int],
        test_obs: np.ndarray,
        test_act: np.ndarray,
        test_episode_ends: list[int],
        test_successes: np.ndarray,
        device: str = "cpu",
        grad_batch_size: int = 128,
    ) -> dict[str, np.ndarray]:
        """Compute per-demonstration CUPID and CUPID-Quality scores.

        Returns
        -------
        dict with keys:
            transition_scores : (N_train, N_test) raw transition influence
            cupid             : (n_train_demos,) CUPID scores
            cupid_quality     : (n_train_demos,) CUPID-Quality scores
            demo_test_matrix  : dict of (n_train_demos, n_test_demos) matrices
                                keyed by aggregation method
        """
        transition_scores = self.compute_transition_scores(
            train_obs, train_act, test_obs, test_act,
            device=device, grad_batch_size=grad_batch_size,
        )

        n_train_demos = len(train_episode_ends)
        n_test_demos = len(test_episode_ends)
        success_mask = np.asarray(test_successes, dtype=bool)

        # Handle edge case: no successes → use reward-based proxy
        if success_mask.sum() == 0:
            logger.warning(
                "[TRAK] No successful test episodes! Using top-50% by "
                "total influence as proxy successes."
            )
            proxy_scores = transition_scores.sum(axis=0)  # total influence per test transition
            test_ep_proxy = _aggregate_by_episodes(
                proxy_scores.reshape(1, -1), [0, len(proxy_scores)],
                test_episode_ends, "sum_of_sum",
            )[0]  # (n_test_demos,)
            median = np.median(test_ep_proxy)
            success_mask = test_ep_proxy >= median

        # Compute demo × test_episode matrices for three aggregation methods
        aggr_methods = ["sum_of_sum", "min_of_max", "max_of_min"]
        demo_test_matrices: dict[str, np.ndarray] = {}

        for method in aggr_methods:
            demo_test_matrices[method] = _aggregate_by_episodes(
                transition_scores,
                train_episode_ends,
                test_episode_ends,
                method,
            )

        # CUPID score: sum_of_sum with success/failure weighting
        sos = demo_test_matrices["sum_of_sum"]
        cupid = _net_score(sos, success_mask)

        # CUPID-Quality: weighted ensemble
        minimax = _net_score(demo_test_matrices["min_of_max"], success_mask)
        maximin = _net_score(demo_test_matrices["max_of_min"], success_mask)
        cupid_quality = 0.5 * cupid + 0.25 * minimax + 0.25 * maximin

        logger.info(
            f"[TRAK] CUPID scores: range [{cupid.min():.4f}, {cupid.max():.4f}]"
        )
        logger.info(
            f"[TRAK] CUPID-Q scores: range [{cupid_quality.min():.4f}, {cupid_quality.max():.4f}]"
        )

        return {
            "transition_scores": transition_scores,
            "cupid": cupid,
            "cupid_quality": cupid_quality,
            "demo_test_matrix": demo_test_matrices,
            "success_mask": success_mask,
        }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_by_episodes(
    transition_scores: np.ndarray,
    train_episode_ends: list[int],
    test_episode_ends: list[int],
    method: str,
) -> np.ndarray:
    """Aggregate (N_train_trans × N_test_trans) scores to (n_train_demos × n_test_demos).

    Aggregation methods
    -------------------
    sum_of_sum : Sum over train transitions, sum over test transitions.
    min_of_max : For each test transition, take max over train transitions
                 in a demo; then min over test transitions in a test episode.
    max_of_min : For each test transition, take min over train transitions
                 in a demo; then max over test transitions in a test episode.
    """
    n_train_demos = len(train_episode_ends)
    n_test_demos = len(test_episode_ends)
    result = np.zeros((n_train_demos, n_test_demos), dtype=np.float32)

    train_starts = [0] + train_episode_ends[:-1]
    test_starts = [0] + test_episode_ends[:-1]

    for i, (ts, te) in enumerate(zip(train_starts, train_episode_ends)):
        for j, (us, ue) in enumerate(zip(test_starts, test_episode_ends)):
            block = transition_scores[ts:te, us:ue]  # (T_train_i, T_test_j)

            if block.size == 0:
                continue

            if method == "sum_of_sum":
                result[i, j] = block.sum()
            elif method == "min_of_max":
                # Max over train transitions (axis=0), then min over test
                result[i, j] = block.max(axis=0).min()
            elif method == "max_of_min":
                # Min over train transitions (axis=0), then max over test
                result[i, j] = block.min(axis=0).max()
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

    return result


def _net_score(demo_test_matrix: np.ndarray, success_mask: np.ndarray) -> np.ndarray:
    """Compute net influence: positive on successes, negative on failures.

    Parameters
    ----------
    demo_test_matrix : (n_train_demos, n_test_demos)
    success_mask     : (n_test_demos,) boolean

    Returns
    -------
    scores : (n_train_demos,) net influence per training demo.
    """
    succ = demo_test_matrix[:, success_mask].sum(axis=1)
    fail = demo_test_matrix[:, ~success_mask].sum(axis=1)
    return succ - fail


# ---------------------------------------------------------------------------
# Convenience: broadcast demo scores to transition level
# ---------------------------------------------------------------------------

def demo_scores_to_transition(
    demo_scores: np.ndarray,
    episode_ends: list[int],
    n_transitions: int,
) -> np.ndarray:
    """Broadcast per-demo scores to per-transition scores.

    Parameters
    ----------
    demo_scores   : (n_demos,)
    episode_ends  : list of cumulative step counts
    n_transitions : total number of transitions

    Returns
    -------
    transition_scores : (n_transitions,) with each transition mapped to its demo score.
    """
    trans = np.zeros(n_transitions, dtype=np.float32)
    start = 0
    for ep_idx, end in enumerate(episode_ends):
        trans[start:end] = float(demo_scores[ep_idx])
        start = end
    return trans
