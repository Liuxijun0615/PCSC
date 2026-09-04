"""Portfolio objective components shared by baselines and PCSC."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.portfolio.constraints import one_way_turnover


def portfolio_returns(
    asset_returns: ArrayLike, weights: ArrayLike
) -> NDArray[np.float64]:
    matrix = np.asarray(asset_returns, dtype=float)
    vector = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[1] != vector.size:
        raise ValueError("asset_returns must be T x n and weights must have length n")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise ValueError("asset returns and weights must be finite")
    return matrix @ vector


def historical_cvar_loss(
    asset_returns: ArrayLike, weights: ArrayLike, *, confidence: float = 0.95
) -> float:
    """Return positive historical tail loss using an exact worst-tail count."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    returns = portfolio_returns(asset_returns, weights)
    tail_count = max(1, int(np.ceil((1.0 - confidence) * returns.size)))
    tail = np.partition(returns, tail_count - 1)[:tail_count]
    return float(-tail.mean())


def expected_holding_period_net_return(
    mean_daily_returns: ArrayLike,
    weights: ArrayLike,
    *,
    holding_days: int,
    previous_weights: ArrayLike | None,
    transaction_cost_bps: float,
) -> float:
    """Approximate expected holding-period return net of rebalance cost."""

    mean_returns = np.asarray(mean_daily_returns, dtype=float)
    vector = np.asarray(weights, dtype=float)
    if mean_returns.shape != vector.shape or mean_returns.ndim != 1:
        raise ValueError("mean_daily_returns and weights must share a 1-D shape")
    if holding_days <= 0 or transaction_cost_bps < 0.0:
        raise ValueError("holding_days must be positive and costs nonnegative")
    gross = float(holding_days * mean_returns @ vector)
    if previous_weights is None:
        # Moving from cash to a fully invested portfolio trades one unit of wealth.
        turnover = 1.0
    else:
        turnover = one_way_turnover(vector, previous_weights)
    return gross - transaction_cost_bps / 10_000.0 * turnover
