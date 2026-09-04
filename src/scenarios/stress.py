"""PCSC stress-scenario representation, repair, transform, and path loss."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class StressScenario:
    block_start: int
    market_multiplier: float
    residual_compression: float


@dataclass(frozen=True)
class ScenarioBounds:
    block_length: int
    market_multiplier_min: float
    market_multiplier_max: float
    residual_compression_min: float
    residual_compression_max: float

    def validate(self, history_length: int) -> None:
        if self.block_length < 2 or self.block_length > history_length:
            raise ValueError("Invalid scenario block length")
        if not (
            0.0 < self.market_multiplier_min <= self.market_multiplier_max
        ):
            raise ValueError("Invalid market-multiplier bounds")
        if not (
            0.0
            <= self.residual_compression_min
            <= self.residual_compression_max
            <= 1.0
        ):
            raise ValueError("Invalid residual-compression bounds")


def repair_scenario(
    scenario: StressScenario, *, bounds: ScenarioBounds, history_length: int
) -> StressScenario:
    """Clip a scenario to its finite historical and numerical support."""

    bounds.validate(history_length)
    max_start = history_length - bounds.block_length
    block_start = int(np.clip(round(scenario.block_start), 0, max_start))
    multiplier = float(
        np.clip(
            scenario.market_multiplier,
            bounds.market_multiplier_min,
            bounds.market_multiplier_max,
        )
    )
    compression = float(
        np.clip(
            scenario.residual_compression,
            bounds.residual_compression_min,
            bounds.residual_compression_max,
        )
    )
    return StressScenario(block_start, multiplier, compression)


def stress_return_transform(
    historical_block: ArrayLike,
    *,
    market_multiplier: float,
    residual_compression: float,
) -> NDArray[np.float64]:
    """Amplify the equal-weight market component and compress residuals."""

    block = np.asarray(historical_block, dtype=float)
    if block.ndim != 2 or block.shape[0] < 1 or block.shape[1] < 1:
        raise ValueError("historical_block must be an H x n matrix")
    if not np.all(np.isfinite(block)):
        raise ValueError("historical_block must be finite")
    if market_multiplier <= 0.0:
        raise ValueError("market_multiplier must be positive")
    if not 0.0 <= residual_compression <= 1.0:
        raise ValueError("residual_compression must lie in [0, 1]")
    market = block.mean(axis=1, keepdims=True)
    residual = block - market
    return market_multiplier * market + (1.0 - residual_compression) * residual


def scenario_returns(
    return_history: ArrayLike,
    scenario: StressScenario,
    *,
    bounds: ScenarioBounds,
) -> NDArray[np.float64]:
    """Repair and materialize one scenario's stressed H x n return block."""

    history = np.asarray(return_history, dtype=float)
    if history.ndim != 2 or not np.all(np.isfinite(history)):
        raise ValueError("return_history must be a finite T x n matrix")
    repaired = repair_scenario(
        scenario, bounds=bounds, history_length=history.shape[0]
    )
    start = repaired.block_start
    block = history[start : start + bounds.block_length]
    return stress_return_transform(
        block,
        market_multiplier=repaired.market_multiplier,
        residual_compression=repaired.residual_compression,
    )


def path_loss(weights: ArrayLike, stressed_returns: ArrayLike) -> float:
    """Maximum additive peak-to-trough portfolio loss over one scenario path."""

    vector = np.asarray(weights, dtype=float)
    returns = np.asarray(stressed_returns, dtype=float)
    if returns.ndim != 2 or vector.ndim != 1 or returns.shape[1] != vector.size:
        raise ValueError("stressed_returns must be H x n and weights length n")
    if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(vector)):
        raise ValueError("weights and stressed returns must be finite")
    portfolio = returns @ vector
    cumulative = np.concatenate([[0.0], np.cumsum(portfolio)])
    running_peak = np.maximum.accumulate(cumulative)
    # The zero floor is the empty-interval convention required for a
    # nonnegative maximum-drawdown loss when every daily return is positive.
    return float(max(0.0, np.max(running_peak - cumulative)))


def scenario_loss(
    weights: ArrayLike,
    return_history: ArrayLike,
    scenario: StressScenario,
    *,
    bounds: ScenarioBounds,
) -> float:
    return path_loss(
        weights, scenario_returns(return_history, scenario, bounds=bounds)
    )


def historical_blocks(
    return_history: ArrayLike, *, block_length: int
) -> NDArray[np.float64]:
    history = np.asarray(return_history, dtype=float)
    if history.ndim != 2 or not np.all(np.isfinite(history)):
        raise ValueError("return_history must be a finite T x n matrix")
    if block_length < 2 or block_length > history.shape[0]:
        raise ValueError("Invalid block_length")
    windows = sliding_window_view(history, window_shape=block_length, axis=0)
    return np.moveaxis(windows, -1, 1)


def select_fixed_worst_block_starts(
    return_history: ArrayLike, *, block_length: int, count: int
) -> NDArray[np.int64]:
    """Select fixed blocks by equal-weight market path loss, independent of portfolios."""

    history = np.asarray(return_history, dtype=float)
    blocks = historical_blocks(history, block_length=block_length)
    if count <= 0 or count > blocks.shape[0]:
        raise ValueError("count must be within the available block count")
    equal_weights = np.full(history.shape[1], 1.0 / history.shape[1])
    losses = np.array([path_loss(equal_weights, block) for block in blocks])
    order = np.lexsort((np.arange(len(losses)), -losses))
    return order[:count].astype(np.int64)
