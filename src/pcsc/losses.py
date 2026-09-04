"""Vectorized portfolio losses for historical and stressed return blocks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def path_losses_for_blocks(
    portfolios: ArrayLike, return_blocks: ArrayLike
) -> NDArray[np.float64]:
    """Return a J x B matrix of additive maximum path losses."""

    weights = np.asarray(portfolios, dtype=float)
    blocks = np.asarray(return_blocks, dtype=float)
    if weights.ndim != 2 or blocks.ndim != 3:
        raise ValueError("portfolios must be J x n and return_blocks B x H x n")
    if weights.shape[1] != blocks.shape[2]:
        raise ValueError("Portfolio and return-block asset dimensions differ")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(blocks)):
        raise ValueError("Portfolio weights and return blocks must be finite")
    paths = np.einsum("bhn,jn->bhj", blocks, weights, optimize=True)
    cumulative = np.concatenate(
        [np.zeros((blocks.shape[0], 1, weights.shape[0])), np.cumsum(paths, axis=1)],
        axis=1,
    )
    running_peak = np.maximum.accumulate(cumulative, axis=1)
    losses_by_block_portfolio = np.max(running_peak - cumulative, axis=1)
    return losses_by_block_portfolio.T
