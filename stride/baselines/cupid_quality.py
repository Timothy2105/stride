"""Baseline: CUPID-Quality data curation.

Identical to CUPID filtering but uses the CUPID-Quality score — a
weighted ensemble of three aggregation methods:

  Q(d) = 0.50 · sum_of_sum(d)
       + 0.25 · min_of_max(d)
       + 0.25 · max_of_min(d)

This ensemble is more robust than using a single aggregation method
because it balances total influence, worst-case, and best-case views.
"""

from __future__ import annotations

import logging

import numpy as np

from stride.baselines.cupid_filter import filter_by_cupid

logger = logging.getLogger(__name__)


def filter_by_cupid_quality(
    data: dict,
    quality_scores: np.ndarray,
    keep_ratio: float,
) -> dict:
    """Keep the top-``keep_ratio`` fraction by CUPID-Quality score.

    Parameters
    ----------
    data            : raw dataset dict.
    quality_scores  : (n_demos,) per-demo CUPID-Quality scores.
    keep_ratio      : fraction of demos to keep.

    Returns
    -------
    Filtered data dict with updated episode_ends.
    """
    logger.info(f"[CUPID-Q] Filtering with quality scores, keep_ratio={keep_ratio:.0%}")
    return filter_by_cupid(data, quality_scores, keep_ratio)
