"""Portfolio-wise vulnerability normalization and coverage archive selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CoverageSelection:
    selected_indices: NDArray[np.int64]
    marginal_gains: NDArray[np.float64]
    coverage_trace: NDArray[np.float64]
    final_coverage: float


def portfolio_wise_vulnerability(
    scenario_losses: ArrayLike,
    historical_block_losses: ArrayLike,
    *,
    epsilon: float = 1e-8,
) -> NDArray[np.float64]:
    """Normalize each portfolio against its own historical loss distribution.

    ``scenario_losses`` has shape J x K and ``historical_block_losses`` has shape
    J x B.  The MAD is the unscaled median absolute deviation appearing in the
    paper definition.
    """

    losses = np.asarray(scenario_losses, dtype=float)
    baselines = np.asarray(historical_block_losses, dtype=float)
    if losses.ndim != 2 or baselines.ndim != 2:
        raise ValueError("Loss matrices must be two-dimensional")
    if losses.shape[0] != baselines.shape[0] or baselines.shape[1] == 0:
        raise ValueError("Loss matrices must share portfolios and have baselines")
    if not np.all(np.isfinite(losses)) or not np.all(np.isfinite(baselines)):
        raise ValueError("Loss matrices must be finite")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    medians = np.median(baselines, axis=1, keepdims=True)
    mad = np.median(np.abs(baselines - medians), axis=1, keepdims=True)
    return np.maximum(0.0, (losses - medians) / (mad + epsilon))


def archive_coverage(vulnerability: ArrayLike) -> float:
    """Mean portfolio-wise maximum vulnerability covered by an archive."""

    values = np.asarray(vulnerability, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("vulnerability must be a non-empty J x K matrix")
    if values.shape[1] == 0:
        return 0.0
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("vulnerability must be finite and nonnegative")
    return float(np.max(values, axis=1).mean())


def greedy_coverage_archive(
    vulnerability: ArrayLike, *, archive_size: int
) -> CoverageSelection:
    """Greedily maximize marginal portfolio-wise vulnerability coverage."""

    values = np.asarray(vulnerability, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("vulnerability must be a non-empty J x K matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("vulnerability must be finite and nonnegative")
    if archive_size <= 0 or archive_size > values.shape[1]:
        raise ValueError("archive_size must not exceed candidate scenarios")

    current = np.zeros(values.shape[0], dtype=float)
    remaining = np.ones(values.shape[1], dtype=bool)
    selected: list[int] = []
    gains: list[float] = []
    trace: list[float] = []
    for _ in range(archive_size):
        candidates = np.flatnonzero(remaining)
        candidate_gains = np.array(
            [np.maximum(current, values[:, index]).mean() - current.mean() for index in candidates]
        )
        # np.argmax supplies a stable smallest-index tie break because candidates
        # are sorted in ascending original order.
        chosen_position = int(np.argmax(candidate_gains))
        chosen = int(candidates[chosen_position])
        gain = float(candidate_gains[chosen_position])
        current = np.maximum(current, values[:, chosen])
        remaining[chosen] = False
        selected.append(chosen)
        gains.append(gain)
        trace.append(float(current.mean()))
    return CoverageSelection(
        selected_indices=np.asarray(selected, dtype=np.int64),
        marginal_gains=np.asarray(gains, dtype=float),
        coverage_trace=np.asarray(trace, dtype=float),
        final_coverage=float(current.mean()),
    )


def nondominated_mask(objectives: ArrayLike) -> NDArray[np.bool_]:
    """Return the non-dominated rows of a minimization objective matrix."""

    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("objectives must be a non-empty matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("objectives must be finite")
    mask = np.ones(values.shape[0], dtype=bool)
    for index in range(values.shape[0]):
        if not mask[index]:
            continue
        dominated = np.all(values <= values[index], axis=1) & np.any(
            values < values[index], axis=1
        )
        if np.any(dominated):
            mask[index] = False
    return mask
