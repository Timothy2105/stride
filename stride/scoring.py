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
from stride import cupid_utils
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)






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
    return cupid_utils.compute_per_sample_grads_loop(model, obs, act, device, batch_size)


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
    return cupid_utils.compute_per_sample_grads_vmap(model, obs, act, device, batch_size)


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

        
        self.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        
        rng = np.random.default_rng(seed)
        signs = rng.choice(
            [-1.0, 1.0], size=(proj_dim, self.n_params),
        ).astype(np.float32)
        self.proj_matrix = torch.from_numpy(
            (signs / np.sqrt(proj_dim)).astype(np.float32)
        )

    def _project(self, grads: torch.Tensor) -> torch.Tensor:
        """Project (N, n_params) gradient matrix → (N, proj_dim)."""
        return grads.float() @ self.proj_matrix.T  

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
        return cupid_utils.compute_transition_scores(
            self.model, self.proj_matrix, self.lambda_reg,
            train_obs, train_act, test_obs, test_act,
            device=device, grad_batch_size=grad_batch_size,
        )

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
        return cupid_utils.compute_demo_scores(
            train_obs, train_act, train_episode_ends,
            test_obs, test_act, test_episode_ends, test_successes,
            device=device, grad_batch_size=grad_batch_size
        )






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
    return cupid_utils.aggregate_by_episodes(
        transition_scores, train_episode_ends, test_episode_ends, method
    )


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
