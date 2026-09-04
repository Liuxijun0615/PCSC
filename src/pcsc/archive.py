"""Pareto-set-aware stress candidate generation and archive feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.pcsc.coverage import (
    CoverageSelection,
    greedy_coverage_archive,
    portfolio_wise_vulnerability,
)
from src.pcsc.losses import path_losses_for_blocks
from src.scenarios.stress import (
    ScenarioBounds,
    StressScenario,
    historical_blocks,
    repair_scenario,
    scenario_returns,
)


@dataclass(frozen=True)
class ScenarioEvaluation:
    candidates: tuple[StressScenario, ...]
    scenario_losses: NDArray[np.float64]
    historical_losses: NDArray[np.float64]
    vulnerability: NDArray[np.float64]


@dataclass(frozen=True)
class ScenarioArchiveUpdate:
    archive: tuple[StressScenario, ...]
    candidates: tuple[StressScenario, ...]
    selected_candidate_indices: NDArray[np.int64]
    marginal_coverage_gains: NDArray[np.float64]
    coverage_trace: NDArray[np.float64]
    final_coverage: float
    scenario_losses: NDArray[np.float64]
    vulnerability: NDArray[np.float64]


def random_scenario(
    rng: np.random.Generator, *, bounds: ScenarioBounds, history_length: int
) -> StressScenario:
    bounds.validate(history_length)
    return StressScenario(
        block_start=int(
            rng.integers(0, history_length - bounds.block_length + 1)
        ),
        market_multiplier=float(
            rng.uniform(
                bounds.market_multiplier_min, bounds.market_multiplier_max
            )
        ),
        residual_compression=float(
            rng.uniform(
                bounds.residual_compression_min,
                bounds.residual_compression_max,
            )
        ),
    )


def mutate_scenario(
    scenario: StressScenario,
    rng: np.random.Generator,
    *,
    bounds: ScenarioBounds,
    history_length: int,
    block_reset_probability: float = 0.20,
    block_shift_radius: int | None = None,
    numeric_scale_fraction: float = 0.10,
) -> StressScenario:
    """Apply bounded mixed discrete/continuous scenario variation."""

    if not 0.0 <= block_reset_probability <= 1.0:
        raise ValueError("block_reset_probability must lie in [0, 1]")
    if numeric_scale_fraction < 0.0:
        raise ValueError("numeric_scale_fraction must be nonnegative")
    bounds.validate(history_length)
    radius = bounds.block_length if block_shift_radius is None else block_shift_radius
    if radius < 0:
        raise ValueError("block_shift_radius must be nonnegative")
    if rng.random() < block_reset_probability:
        block_start = int(
            rng.integers(0, history_length - bounds.block_length + 1)
        )
    else:
        block_start = scenario.block_start + int(rng.integers(-radius, radius + 1))
    multiplier_sigma = numeric_scale_fraction * (
        bounds.market_multiplier_max - bounds.market_multiplier_min
    )
    compression_sigma = numeric_scale_fraction * (
        bounds.residual_compression_max - bounds.residual_compression_min
    )
    mutated = StressScenario(
        block_start=block_start,
        market_multiplier=float(
            scenario.market_multiplier + rng.normal(0.0, multiplier_sigma)
        ),
        residual_compression=float(
            scenario.residual_compression + rng.normal(0.0, compression_sigma)
        ),
    )
    return repair_scenario(mutated, bounds=bounds, history_length=history_length)


def generate_scenario_candidates(
    *,
    prior_archive: Sequence[StressScenario],
    candidate_size: int,
    rng: np.random.Generator,
    bounds: ScenarioBounds,
    history_length: int,
    random_immigrant_fraction: float = 0.25,
    block_reset_probability: float = 0.20,
    block_shift_radius: int | None = None,
    numeric_scale_fraction: float = 0.10,
) -> tuple[StressScenario, ...]:
    if candidate_size <= 0:
        raise ValueError("candidate_size must be positive")
    if not 0.0 <= random_immigrant_fraction <= 1.0:
        raise ValueError("random_immigrant_fraction must lie in [0, 1]")
    repaired_archive = [
        repair_scenario(item, bounds=bounds, history_length=history_length)
        for item in prior_archive
    ][:candidate_size]
    candidates = list(repaired_archive)
    while len(candidates) < candidate_size:
        use_random = not repaired_archive or rng.random() < random_immigrant_fraction
        if use_random:
            child = random_scenario(rng, bounds=bounds, history_length=history_length)
        else:
            parent = repaired_archive[int(rng.integers(0, len(repaired_archive)))]
            child = mutate_scenario(
                parent,
                rng,
                bounds=bounds,
                history_length=history_length,
                block_reset_probability=block_reset_probability,
                block_shift_radius=block_shift_radius,
                numeric_scale_fraction=numeric_scale_fraction,
            )
        candidates.append(child)
    return tuple(candidates)


def evaluate_scenario_candidates(
    return_history: ArrayLike,
    pareto_portfolios: ArrayLike,
    candidates: Sequence[StressScenario],
    *,
    bounds: ScenarioBounds,
    epsilon: float = 1e-8,
) -> ScenarioEvaluation:
    history = np.asarray(return_history, dtype=float)
    portfolios = np.asarray(pareto_portfolios, dtype=float)
    if history.ndim != 2 or portfolios.ndim != 2:
        raise ValueError("return_history and pareto_portfolios must be matrices")
    if history.shape[1] != portfolios.shape[1]:
        raise ValueError("History and portfolios use different asset dimensions")
    if len(candidates) == 0:
        raise ValueError("At least one scenario candidate is required")
    stressed_blocks = np.stack(
        [scenario_returns(history, item, bounds=bounds) for item in candidates]
    )
    scenario_losses = path_losses_for_blocks(portfolios, stressed_blocks)
    baseline_blocks = historical_blocks(history, block_length=bounds.block_length)
    historical_losses = path_losses_for_blocks(portfolios, baseline_blocks)
    vulnerability = portfolio_wise_vulnerability(
        scenario_losses, historical_losses, epsilon=epsilon
    )
    return ScenarioEvaluation(
        candidates=tuple(candidates),
        scenario_losses=scenario_losses,
        historical_losses=historical_losses,
        vulnerability=vulnerability,
    )


def select_average_loss_archive(
    scenario_losses: ArrayLike, *, archive_size: int
) -> NDArray[np.int64]:
    """Ablation/control selector based on mean Pareto-portfolio raw loss."""

    losses = np.asarray(scenario_losses, dtype=float)
    if losses.ndim != 2 or losses.shape[0] == 0 or losses.shape[1] == 0:
        raise ValueError("scenario_losses must be a non-empty J x K matrix")
    if archive_size <= 0 or archive_size > losses.shape[1]:
        raise ValueError("Invalid archive_size")
    mean_loss = losses.mean(axis=0)
    order = np.lexsort((np.arange(losses.shape[1]), -mean_loss))
    return order[:archive_size].astype(np.int64)


def select_mean_vulnerability_archive(
    vulnerability: ArrayLike, *, archive_size: int
) -> NDArray[np.int64]:
    """Coverage ablation: rank normalized scenarios without set marginality."""

    values = np.asarray(vulnerability, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("vulnerability must be a non-empty J x K matrix")
    if archive_size <= 0 or archive_size > values.shape[1]:
        raise ValueError("Invalid archive_size")
    mean_vulnerability = values.mean(axis=0)
    order = np.lexsort((np.arange(values.shape[1]), -mean_vulnerability))
    return order[:archive_size].astype(np.int64)


def update_scenario_archive(
    return_history: ArrayLike,
    pareto_portfolios: ArrayLike,
    *,
    prior_archive: Sequence[StressScenario],
    bounds: ScenarioBounds,
    archive_size: int,
    candidate_size: int,
    rng: np.random.Generator,
    epsilon: float = 1e-8,
    random_immigrant_fraction: float = 0.25,
    block_reset_probability: float = 0.20,
    block_shift_radius: int | None = None,
    numeric_scale_fraction: float = 0.10,
    selection_rule: str = "coverage",
) -> ScenarioArchiveUpdate:
    """Complete portfolio-to-scenario feedback and coverage-based archive update."""

    history = np.asarray(return_history, dtype=float)
    if archive_size <= 0 or candidate_size < archive_size:
        raise ValueError("candidate_size must be at least archive_size > 0")
    candidates = generate_scenario_candidates(
        prior_archive=prior_archive,
        candidate_size=candidate_size,
        rng=rng,
        bounds=bounds,
        history_length=history.shape[0],
        random_immigrant_fraction=random_immigrant_fraction,
        block_reset_probability=block_reset_probability,
        block_shift_radius=block_shift_radius,
        numeric_scale_fraction=numeric_scale_fraction,
    )
    evaluation = evaluate_scenario_candidates(
        history,
        pareto_portfolios,
        candidates,
        bounds=bounds,
        epsilon=epsilon,
    )
    if selection_rule == "coverage":
        selection: CoverageSelection = greedy_coverage_archive(
            evaluation.vulnerability, archive_size=archive_size
        )
    elif selection_rule in {"average_loss", "mean_vulnerability"}:
        indices = (
            select_average_loss_archive(
                evaluation.scenario_losses, archive_size=archive_size
            )
            if selection_rule == "average_loss"
            else select_mean_vulnerability_archive(
                evaluation.vulnerability, archive_size=archive_size
            )
        )
        current = np.zeros(evaluation.vulnerability.shape[0], dtype=float)
        gains = []
        trace = []
        for index in indices:
            previous = float(current.mean())
            current = np.maximum(current, evaluation.vulnerability[:, index])
            trace.append(float(current.mean()))
            gains.append(float(current.mean()) - previous)
        selection = CoverageSelection(
            selected_indices=indices,
            marginal_gains=np.asarray(gains, dtype=float),
            coverage_trace=np.asarray(trace, dtype=float),
            final_coverage=float(current.mean()),
        )
    else:
        raise ValueError(f"Unknown selection_rule: {selection_rule}")
    archive = tuple(candidates[index] for index in selection.selected_indices)
    return ScenarioArchiveUpdate(
        archive=archive,
        candidates=candidates,
        selected_candidate_indices=selection.selected_indices,
        marginal_coverage_gains=selection.marginal_gains,
        coverage_trace=selection.coverage_trace,
        final_coverage=selection.final_coverage,
        scenario_losses=evaluation.scenario_losses,
        vulnerability=evaluation.vulnerability,
    )
