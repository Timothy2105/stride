"""Influence-guided corrective direction computation.

For each training sample i we:
1. Normalise influence scores: Ĩ_j = (I_j − μ) / σ,  w_j = max(0, Ĩ_j)
2. Find k-nearest neighbours N(i) in VAE latent space (or action space)
3. Compute a corrective target direction in action space:
       Δa_i = Σ_{j∈N(i)} w_j (a_j − a_i)
       Δa_i ← Δa_i / (‖Δa_i‖ + ε)
4. If all neighbour weights are zero (no positive-influence neighbours),
   Δa_i is set to the zero vector (no correction).

The normalised directions are returned as the training targets for the editor.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors






def normalise_influence_scores(
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardise influence scores and clip at zero.

    Parameters
    ----------
    scores : (N,) raw influence values

    Returns
    -------
    scores_norm : (N,) standardised scores  Ĩ_j = (I_j − μ) / σ
    weights     : (N,) non-negative weights  w_j = max(0, Ĩ_j)
    """
    mu = scores.mean()
    sigma = scores.std()
    if sigma < 1e-8:
        sigma = 1e-8
    scores_norm = (scores - mu) / sigma
    weights = np.maximum(0.0, scores_norm)
    return scores_norm, weights






def build_knn_index(
    embeddings: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
) -> NearestNeighbors:
    """Fit a sklearn NearestNeighbors index on the given embeddings.

    Parameters
    ----------
    embeddings : (N, D) array of feature vectors (VAE latent means or actions)
    k          : number of neighbours to retrieve
    metric     : distance metric (default 'euclidean')

    Returns
    -------
    fitted NearestNeighbors object
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto",
                          n_jobs=-1)
    nn.fit(embeddings)
    return nn






def compute_corrective_directions(
    observations: np.ndarray,
    actions: np.ndarray,
    influence_weights: np.ndarray,
    embeddings: np.ndarray | None = None,
    k: int = 10,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute normalised corrective target directions for each training sample.

    For each sample i:
        Δa_i = Σ_{j∈N(i)} w_j (a_j − a_i)
        Δa_i ← Δa_i / (‖Δa_i‖ + ε)

    where N(i) are the k-nearest neighbours (excluding i itself).

    Parameters
    ----------
    observations      : (N, obs_dim) – used for KNN if embeddings is None
    actions           : (N, act_dim) – action vectors
    influence_weights : (N,)  w_j = max(0, Ĩ_j) non-negative
    embeddings        : (N, D) optional VAE latent means; uses actions if None
    k                 : number of neighbours
    eps               : small constant for normalisation

    Returns
    -------
    directions : (N, act_dim) normalised corrective directions
                 Zero vector if sample has no positively-influential neighbours.
    """
    knn_features = embeddings if embeddings is not None else actions
    nn_index = build_knn_index(knn_features, k=k)

    N, act_dim = actions.shape
    directions = np.zeros((N, act_dim), dtype=np.float32)

    
    
    _, indices = nn_index.kneighbors(knn_features)
    
    neighbour_indices = indices[:, 1:]  

    for i in range(N):
        nbrs = neighbour_indices[i]           
        w = influence_weights[nbrs]           
        diff = actions[nbrs] - actions[i]    

        
        delta = (w[:, None] * diff).sum(axis=0)  

        norm = np.linalg.norm(delta)
        if norm > eps:
            directions[i] = delta / (norm + eps)
        

    return directions






def compute_preference_pairs(
    actions: np.ndarray,
    influence_scores_corrected: np.ndarray,
    embeddings: np.ndarray | None = None,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (winner, loser) action pairs for DPO editor training.

    For each sample i, among its k-nearest neighbours:
      - winner a_w = action of the neighbour with highest influence (most helpful)
      - loser  a_l = action of the neighbour with lowest influence (most harmful)

    Parameters
    ----------
    actions                    : (N, act_dim)
    influence_scores_corrected : (N,) corrected sign: positive = helpful
    embeddings                 : (N, D) for KNN; uses actions if None
    k                          : number of neighbours

    Returns
    -------
    winners : (N, act_dim) — best neighbour action per sample
    losers  : (N, act_dim) — worst neighbour action per sample
    valid   : (N,) bool — True if the sample has a meaningful preference pair
              (i.e. winner influence > loser influence)
    """
    knn_features = embeddings if embeddings is not None else actions
    nn_index = build_knn_index(knn_features, k=k)

    N, act_dim = actions.shape
    winners = np.zeros((N, act_dim), dtype=np.float32)
    losers = np.zeros((N, act_dim), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    _, indices = nn_index.kneighbors(knn_features)
    neighbour_indices = indices[:, 1:]  

    for i in range(N):
        nbrs = neighbour_indices[i]
        nbr_scores = influence_scores_corrected[nbrs]

        best_j = nbrs[np.argmax(nbr_scores)]
        worst_j = nbrs[np.argmin(nbr_scores)]

        winners[i] = actions[best_j]
        losers[i] = actions[worst_j]

        
        valid[i] = (influence_scores_corrected[best_j] >
                    influence_scores_corrected[worst_j])

    return winners, losers, valid


def compute_ranking_data(
    actions: np.ndarray,
    influence_scores_corrected: np.ndarray,
    embeddings: np.ndarray | None = None,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract top-k neighbor actions and their influence scores for R-DPO.

    For each sample i, we find its k-nearest neighbors and return their
    actions and influence scores, which can be used for a ranking-based loss.

    Parameters
    ----------
    actions                    : (N, act_dim)
    influence_scores_corrected : (N,) corrected sign: positive = helpful
    embeddings                 : (N, D) for KNN; uses actions if None
    k                          : number of neighbours

    Returns
    -------
    neighbour_actions : (N, k, act_dim)
    neighbour_scores  : (N, k)
    """
    knn_features = embeddings if embeddings is not None else actions
    nn_index = build_knn_index(knn_features, k=k)

    N, act_dim = actions.shape
    neighbour_actions = np.zeros((N, k, act_dim), dtype=np.float32)
    neighbour_scores = np.zeros((N, k), dtype=np.float32)

    _, indices = nn_index.kneighbors(knn_features)
    neighbour_indices = indices[:, 1:]  

    for i in range(N):
        nbrs = neighbour_indices[i]
        neighbour_actions[i] = actions[nbrs]
        neighbour_scores[i] = influence_scores_corrected[nbrs]

    return neighbour_actions, neighbour_scores
