"""NSGA-II Mean--CVaR baseline with the project's shared weight repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from src.portfolio.constraints import one_way_turnover, repair_weights
from src.portfolio.objectives import historical_cvar_loss


@dataclass(frozen=True)
class ParetoPortfolioResult:
    weights: NDArray[np.float64]
    objectives: NDArray[np.float64]
    selected_index: int
    selected_weights: NDArray[np.float64]
    selected_objectives: NDArray[np.float64]
    seed: int


class CappedTurnoverRepair(Repair):
    def __init__(
        self,
        *,
        upper: float | ArrayLike,
        previous_weights: ArrayLike | None,
        max_one_way_turnover: float | None,
    ) -> None:
        super().__init__()
        self.upper = upper
        self.previous_weights = previous_weights
        self.max_one_way_turnover = max_one_way_turnover

    def _do(self, problem, x, **kwargs):
        values = np.asarray(x, dtype=float)
        repaired = np.empty_like(values)
        for index, row in enumerate(values):
            repaired[index] = repair_weights(
                row,
                upper=self.upper,
                previous_weights=self.previous_weights,
                max_one_way_turnover=self.max_one_way_turnover,
            )
        return repaired


class MeanCVarProblem(Problem):
    def __init__(
        self,
        asset_returns: ArrayLike,
        *,
        previous_weights: ArrayLike | None,
        holding_days: int,
        transaction_cost_bps: float,
        cvar_confidence: float,
    ) -> None:
        returns = np.asarray(asset_returns, dtype=float)
        if returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] < 2:
            raise ValueError("asset_returns must be a T x n matrix")
        if not np.all(np.isfinite(returns)):
            raise ValueError("asset_returns must be finite")
        if holding_days <= 0 or transaction_cost_bps < 0.0:
            raise ValueError("Invalid holding days or transaction cost")
        if not 0.0 < cvar_confidence < 1.0:
            raise ValueError("cvar_confidence must lie strictly between zero and one")
        self.asset_returns = returns
        self.mean_daily_returns = returns.mean(axis=0)
        self.previous_weights = (
            None
            if previous_weights is None
            else np.asarray(previous_weights, dtype=float)
        )
        self.holding_days = holding_days
        self.transaction_cost_bps = transaction_cost_bps
        self.cvar_confidence = cvar_confidence
        super().__init__(n_var=returns.shape[1], n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        weights = np.asarray(x, dtype=float)
        gross = self.holding_days * (weights @ self.mean_daily_returns)
        if self.previous_weights is None:
            turnover = np.ones(weights.shape[0])
        else:
            turnover = np.array(
                [one_way_turnover(row, self.previous_weights) for row in weights]
            )
        net = gross - self.transaction_cost_bps / 10_000.0 * turnover
        cvar = np.array(
            [
                historical_cvar_loss(
                    self.asset_returns, row, confidence=self.cvar_confidence
                )
                for row in weights
            ]
        )
        out["F"] = np.column_stack([-net, cvar])


def select_normalized_ideal(objectives: ArrayLike) -> int:
    """Select the front member nearest the normalized ideal point.

    This is a deterministic candidate deployment rule.  It remains preliminary
    until frozen using 2019 validation only.
    """

    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("objectives must be a non-empty matrix")
    minima = values.min(axis=0)
    spans = values.max(axis=0) - minima
    spans = np.where(spans > 0.0, spans, 1.0)
    normalized = (values - minima) / spans
    distances = np.linalg.norm(normalized, axis=1)
    return int(np.argmin(distances))


def solve_mean_cvar_nsga2(
    asset_returns: ArrayLike,
    *,
    upper: float | ArrayLike,
    previous_weights: ArrayLike | None,
    max_one_way_turnover: float | None,
    holding_days: int,
    transaction_cost_bps: float,
    cvar_confidence: float = 0.95,
    population_size: int = 80,
    generations: int = 50,
    seed: int = 1,
) -> ParetoPortfolioResult:
    """Run the baseline and return its non-dominated front and selected member."""

    returns = np.asarray(asset_returns, dtype=float)
    if population_size < 4 or generations < 1:
        raise ValueError("population_size must be >=4 and generations >=1")
    problem = MeanCVarProblem(
        returns,
        previous_weights=previous_weights,
        holding_days=holding_days,
        transaction_cost_bps=transaction_cost_bps,
        cvar_confidence=cvar_confidence,
    )
    repair = CappedTurnoverRepair(
        upper=upper,
        previous_weights=previous_weights,
        max_one_way_turnover=max_one_way_turnover,
    )
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0 / returns.shape[1], eta=20),
        repair=repair,
        eliminate_duplicates=True,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", generations),
        seed=seed,
        verbose=False,
    )
    if result.X is None or result.F is None:
        raise RuntimeError("NSGA-II returned no Pareto portfolios")
    weights = np.atleast_2d(np.asarray(result.X, dtype=float))
    objectives = np.atleast_2d(np.asarray(result.F, dtype=float))
    selected = select_normalized_ideal(objectives)
    return ParetoPortfolioResult(
        weights=weights,
        objectives=objectives,
        selected_index=selected,
        selected_weights=weights[selected].copy(),
        selected_objectives=objectives[selected].copy(),
        seed=seed,
    )
