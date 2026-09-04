"""Deterministic long-only budget, cap, and turnover repair operators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _upper_bounds(upper: float | ArrayLike, size: int) -> NDArray[np.float64]:
    values = np.asarray(upper, dtype=float)
    if values.ndim == 0:
        values = np.full(size, float(values), dtype=float)
    if values.shape != (size,):
        raise ValueError(f"upper must be scalar or shape ({size},)")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("upper bounds must be finite and nonnegative")
    return values


def project_capped_simplex(
    values: ArrayLike,
    *,
    upper: float | ArrayLike,
    budget: float = 1.0,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> NDArray[np.float64]:
    """Euclidean projection onto ``sum(w)=budget`` and ``0<=w<=upper``."""

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must be finite")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("budget must be positive and finite")
    bounds = _upper_bounds(upper, vector.size)
    if bounds.sum() < budget - tolerance:
        raise ValueError("upper bounds cannot satisfy the budget")

    lower_lambda = float(np.min(vector - bounds))
    upper_lambda = float(np.max(vector))
    projected = np.clip(vector, 0.0, bounds)
    for _ in range(max_iterations):
        lagrange = 0.5 * (lower_lambda + upper_lambda)
        projected = np.clip(vector - lagrange, 0.0, bounds)
        total = float(projected.sum())
        if abs(total - budget) <= tolerance:
            break
        if total > budget:
            lower_lambda = lagrange
        else:
            upper_lambda = lagrange
    residual = budget - float(projected.sum())
    if abs(residual) > 10 * tolerance:
        capacity = bounds - projected if residual > 0 else projected
        eligible = capacity > tolerance
        if not np.any(eligible):
            raise RuntimeError("Capped-simplex projection failed to close the budget")
        projected[eligible] += residual * capacity[eligible] / capacity[eligible].sum()
    projected = np.clip(projected, 0.0, bounds)
    if abs(float(projected.sum()) - budget) > 1e-9:
        raise RuntimeError("Capped-simplex projection violates the budget")
    return projected


def one_way_turnover(weights: ArrayLike, previous_weights: ArrayLike) -> float:
    """Return conventional one-way turnover, one half of the L1 weight change."""

    current = np.asarray(weights, dtype=float)
    previous = np.asarray(previous_weights, dtype=float)
    if current.shape != previous.shape or current.ndim != 1:
        raise ValueError("weights and previous_weights must have the same 1-D shape")
    return 0.5 * float(np.abs(current - previous).sum())


def _feasible_anchor(
    previous: NDArray[np.float64],
    candidate: NDArray[np.float64],
    bounds: NDArray[np.float64],
    *,
    budget: float,
    tolerance: float,
) -> NDArray[np.float64]:
    """Build a capped target with minimum required turnover from prior weights."""

    anchor = np.minimum(previous, bounds)
    deficit = budget - float(anchor.sum())
    if deficit < -tolerance:
        raise ValueError("previous weights exceed the budget after nonnegative capping")
    if deficit <= tolerance:
        return anchor
    capacity = bounds - anchor
    preference = np.maximum(candidate - anchor, 0.0)
    preferred_capacity = np.minimum(preference, capacity)
    if preferred_capacity.sum() > tolerance:
        allocation = deficit * preferred_capacity / preferred_capacity.sum()
        allocation = np.minimum(allocation, capacity)
        anchor += allocation
        deficit = budget - float(anchor.sum())
        capacity = bounds - anchor
    if deficit > tolerance:
        if capacity.sum() < deficit - tolerance:
            raise ValueError("upper bounds cannot absorb redistributed weight")
        anchor += deficit * capacity / capacity.sum()
    return anchor


def repair_weights(
    values: ArrayLike,
    *,
    upper: float | ArrayLike,
    previous_weights: ArrayLike | None = None,
    max_one_way_turnover: float | None = None,
    budget: float = 1.0,
    tolerance: float = 1e-12,
) -> NDArray[np.float64]:
    """Repair a candidate and, when requested, enforce one-way turnover.

    If market drift leaves the prior portfolio above an asset cap, the operator
    first constructs a feasible anchor.  It raises when the turnover budget is
    smaller than the minimum mass that must leave cap-violating positions.
    """

    candidate = project_capped_simplex(
        values, upper=upper, budget=budget, tolerance=tolerance
    )
    if previous_weights is None and max_one_way_turnover is None:
        return candidate
    if previous_weights is None or max_one_way_turnover is None:
        raise ValueError(
            "previous_weights and max_one_way_turnover must be supplied together"
        )
    if max_one_way_turnover < 0.0:
        raise ValueError("max_one_way_turnover must be nonnegative")

    previous = np.asarray(previous_weights, dtype=float)
    if previous.shape != candidate.shape or not np.all(np.isfinite(previous)):
        raise ValueError("previous_weights must be finite and match candidate shape")
    if np.any(previous < -tolerance) or abs(float(previous.sum()) - budget) > 1e-9:
        raise ValueError("previous_weights must be nonnegative and sum to the budget")
    previous = np.maximum(previous, 0.0)
    bounds = _upper_bounds(upper, candidate.size)
    anchor = _feasible_anchor(
        previous,
        candidate,
        bounds,
        budget=budget,
        tolerance=tolerance,
    )
    minimum_turnover = one_way_turnover(anchor, previous)
    if minimum_turnover > max_one_way_turnover + 1e-10:
        raise ValueError(
            "Turnover cap is infeasible after enforcing asset upper bounds: "
            f"minimum={minimum_turnover:.6f}, allowed={max_one_way_turnover:.6f}"
        )
    if one_way_turnover(candidate, previous) <= max_one_way_turnover + tolerance:
        return candidate

    low = 0.0
    high = 1.0
    repaired = anchor.copy()
    for _ in range(200):
        fraction = 0.5 * (low + high)
        trial = anchor + fraction * (candidate - anchor)
        if one_way_turnover(trial, previous) <= max_one_way_turnover:
            low = fraction
            repaired = trial
        else:
            high = fraction
        if high - low <= tolerance:
            break
    if one_way_turnover(repaired, previous) > max_one_way_turnover + 1e-9:
        raise RuntimeError("Turnover repair failed")
    return repaired
