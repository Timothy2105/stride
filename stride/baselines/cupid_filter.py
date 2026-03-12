"""Baseline: CUPID data curation.

Filter training demonstrations by keeping only the top-K% trajectories
according to CUPID influence scores.  Higher CUPID score → more helpful
demonstration → keep it.

This implements the standard CUPID scoring:
  score(d) = Σ_{j ∈ succ} influence(d, j)  −  Σ_{j ∈ fail} influence(d, j)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def filter_by_cupid(
    data: dict,
    demo_scores: np.ndarray,
    keep_ratio: float,
) -> dict:
    """Keep the top-``keep_ratio`` fraction of demonstrations by CUPID score.

    Parameters
    ----------
    data        : raw dataset dict with 'observations', 'actions',
                  'rewards', 'terminals', 'episode_ends'.
    demo_scores : (n_demos,) per-demonstration CUPID scores.
    keep_ratio  : fraction of demos to keep (e.g. 0.50 = top 50%).

    Returns
    -------
    Filtered data dict with updated episode_ends.
    """
    episode_ends = [int(x) for x in data["episode_ends"]]
    n_demos = len(episode_ends)

    if demo_scores.shape[0] != n_demos:
        raise ValueError(
            f"demo_scores length {demo_scores.shape[0]} != "
            f"number of episodes {n_demos}"
        )
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    n_keep = max(1, int(n_demos * keep_ratio))
    keep_ids = set(np.argsort(demo_scores)[-n_keep:].tolist())

    keep_idx: list[int] = []
    curated_ends: list[int] = []
    start = 0
    cursor = 0

    for ep_id, end in enumerate(episode_ends):
        end_i = int(end)
        if ep_id in keep_ids:
            rng = list(range(start, end_i))
            keep_idx.extend(rng)
            cursor += len(rng)
            curated_ends.append(cursor)
        start = end_i

    keep_idx_np = np.asarray(keep_idx, dtype=np.int64)

    filtered = {
        "observations": data["observations"][keep_idx_np],
        "actions": data["actions"][keep_idx_np],
        "episode_ends": curated_ends,
    }
    if "rewards" in data:
        filtered["rewards"] = data["rewards"][keep_idx_np]
    if "terminals" in data:
        filtered["terminals"] = data["terminals"][keep_idx_np]

    logger.info(
        f"[CUPID] Kept {n_keep}/{n_demos} demos (ratio={keep_ratio:.0%})  "
        f"transitions: {len(data['observations'])} → {len(keep_idx_np)}"
    )
    return filtered
