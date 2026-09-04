"""Auditable generational PCSC loop with bidirectional archive feedback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.baselines.nsga2_mean_cvar import select_normalized_ideal
from src.pcsc.archive import ScenarioArchiveUpdate, update_scenario_archive
from src.pcsc.losses import path_losses_for_blocks
from src.portfolio.constraints import one_way_turnover, repair_weights
from src.scenarios.stress import ScenarioBounds, StressScenario, scenario_returns


@dataclass(frozen=True)
class PCSCGenerationRecord:
    generation: int
    archive_updated: bool
    pareto_size: int
    archive_coverage_at_update: float | None
    best_negative_net_return: float
    best_worst_stress_loss: float


@dataclass(frozen=True)
class PCSCEvolutionResult:
    pareto_weights: NDArray[np.float64]
    pareto_objectives: NDArray[np.float64]
    selected_index: int
    selected_weights: NDArray[np.float64]
    selected_objectives: NDArray[np.float64]
    archive: tuple[StressScenario, ...]
    generation_log: tuple[PCSCGenerationRecord, ...]
    archive_update_count: int
    seed: int
    archive_selection_rule: str
    scenario_feedback_set: str
    portfolio_feedback_enabled: bool


def fast_nondominated_sort(
    objectives: ArrayLike,
) -> tuple[NDArray[np.int64], list[NDArray[np.int64]]]:
    """Return Pareto rank for each row and fronts for minimization objectives."""

    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("objectives must be a non-empty finite matrix")
    size = values.shape[0]
    dominates: list[list[int]] = [[] for _ in range(size)]
    dominated_count = np.zeros(size, dtype=int)
    for left in range(size):
        for right in range(left + 1, size):
            left_dominates = np.all(values[left] <= values[right]) and np.any(
                values[left] < values[right]
            )
            right_dominates = np.all(values[right] <= values[left]) and np.any(
                values[right] < values[left]
            )
            if left_dominates:
                dominates[left].append(right)
                dominated_count[right] += 1
            elif right_dominates:
                dominates[right].append(left)
                dominated_count[left] += 1

    ranks = np.full(size, -1, dtype=np.int64)
    current = np.flatnonzero(dominated_count == 0).astype(np.int64)
    fronts: list[NDArray[np.int64]] = []
    rank = 0
    while current.size:
        fronts.append(current)
        ranks[current] = rank
        next_front: list[int] = []
        for member in current:
            for dominated in dominates[int(member)]:
                dominated_count[dominated] -= 1
                if dominated_count[dominated] == 0:
                    next_front.append(dominated)
        current = np.asarray(next_front, dtype=np.int64)
        rank += 1
    if np.any(ranks < 0):
        raise RuntimeError("Non-dominated sorting did not rank every individual")
    return ranks, fronts


def crowding_distances(
    objectives: ArrayLike, fronts: list[NDArray[np.int64]]
) -> NDArray[np.float64]:
    values = np.asarray(objectives, dtype=float)
    distances = np.zeros(values.shape[0], dtype=float)
    for front in fronts:
        if front.size <= 2:
            distances[front] = np.inf
            continue
        front_values = values[front]
        local = np.zeros(front.size, dtype=float)
        for objective in range(values.shape[1]):
            order = np.argsort(front_values[:, objective], kind="mergesort")
            local[order[0]] = np.inf
            local[order[-1]] = np.inf
            span = (
                front_values[order[-1], objective]
                - front_values[order[0], objective]
            )
            if span <= 0.0:
                continue
            for position in range(1, front.size - 1):
                if np.isinf(local[order[position]]):
                    continue
                local[order[position]] += (
                    front_values[order[position + 1], objective]
                    - front_values[order[position - 1], objective]
                ) / span
        distances[front] = local
    return distances


def _survival_indices(objectives: NDArray[np.float64], size: int) -> NDArray[np.int64]:
    _, fronts = fast_nondominated_sort(objectives)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(int(value) for value in front)
            continue
        distances = crowding_distances(objectives, [front])[front]
        order = np.lexsort((front, -distances))
        need = size - len(selected)
        selected.extend(int(front[index]) for index in order[:need])
        break
    return np.asarray(selected, dtype=np.int64)


def _tournament(
    rng: np.random.Generator,
    ranks: NDArray[np.int64],
    crowding: NDArray[np.float64],
) -> int:
    left, right = rng.integers(0, len(ranks), size=2)
    if ranks[left] != ranks[right]:
        return int(left if ranks[left] < ranks[right] else right)
    if crowding[left] != crowding[right]:
        return int(left if crowding[left] > crowding[right] else right)
    return int(left if rng.random() < 0.5 else right)


def _sbx_pair(
    parent_a: NDArray[np.float64],
    parent_b: NDArray[np.float64],
    *,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    probability: float,
    eta: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    if rng.random() > probability:
        return child_a, child_b
    exponent = 1.0 / (eta + 1.0)
    for index in range(parent_a.size):
        if rng.random() > 0.5 or abs(parent_a[index] - parent_b[index]) <= 1e-14:
            continue
        first = min(parent_a[index], parent_b[index])
        second = max(parent_a[index], parent_b[index])
        rand = rng.random()
        beta = 1.0 + 2.0 * (first - lower[index]) / (second - first)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        beta_q = (
            (rand * alpha) ** exponent
            if rand <= 1.0 / alpha
            else (1.0 / (2.0 - rand * alpha)) ** exponent
        )
        first_child = 0.5 * ((first + second) - beta_q * (second - first))
        beta = 1.0 + 2.0 * (upper[index] - second) / (second - first)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        beta_q = (
            (rand * alpha) ** exponent
            if rand <= 1.0 / alpha
            else (1.0 / (2.0 - rand * alpha)) ** exponent
        )
        second_child = 0.5 * ((first + second) + beta_q * (second - first))
        first_child = float(np.clip(first_child, lower[index], upper[index]))
        second_child = float(np.clip(second_child, lower[index], upper[index]))
        if rng.random() < 0.5:
            child_a[index], child_b[index] = second_child, first_child
        else:
            child_a[index], child_b[index] = first_child, second_child
    return child_a, child_b


def _polynomial_mutation(
    child: NDArray[np.float64],
    *,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    probability: float,
    eta: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    mutated = child.copy()
    exponent = 1.0 / (eta + 1.0)
    for index in range(child.size):
        if rng.random() > probability or upper[index] <= lower[index]:
            continue
        value = mutated[index]
        delta_one = (value - lower[index]) / (upper[index] - lower[index])
        delta_two = (upper[index] - value) / (upper[index] - lower[index])
        rand = rng.random()
        mutation_power: float
        if rand <= 0.5:
            value_term = 2.0 * rand + (1.0 - 2.0 * rand) * (
                1.0 - delta_one
            ) ** (eta + 1.0)
            mutation_power = value_term**exponent - 1.0
        else:
            value_term = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (
                1.0 - delta_two
            ) ** (eta + 1.0)
            mutation_power = 1.0 - value_term**exponent
        mutated[index] = np.clip(
            value + mutation_power * (upper[index] - lower[index]),
            lower[index],
            upper[index],
        )
    return mutated


def _upper_array(upper: float | ArrayLike, size: int) -> NDArray[np.float64]:
    bounds = np.asarray(upper, dtype=float)
    if bounds.ndim == 0:
        bounds = np.full(size, float(bounds))
    if bounds.shape != (size,) or np.any(bounds < 0.0) or bounds.sum() < 1.0:
        raise ValueError("Invalid portfolio upper bounds")
    return bounds


def _repair_population(
    values: NDArray[np.float64],
    *,
    upper: NDArray[np.float64],
    previous_weights: NDArray[np.float64] | None,
    max_one_way_turnover: float | None,
) -> NDArray[np.float64]:
    return np.stack(
        [
            repair_weights(
                row,
                upper=upper,
                previous_weights=previous_weights,
                max_one_way_turnover=max_one_way_turnover,
            )
            for row in values
        ]
    )


def _portfolio_objectives(
    population: NDArray[np.float64],
    *,
    mean_daily_returns: NDArray[np.float64],
    archive_blocks: NDArray[np.float64],
    previous_weights: NDArray[np.float64] | None,
    holding_days: int,
    transaction_cost_bps: float,
) -> NDArray[np.float64]:
    gross = holding_days * (population @ mean_daily_returns)
    if previous_weights is None:
        turnover = np.ones(len(population))
    else:
        turnover = np.array(
            [one_way_turnover(row, previous_weights) for row in population]
        )
    net = gross - transaction_cost_bps / 10_000.0 * turnover
    worst_loss = path_losses_for_blocks(population, archive_blocks).max(axis=1)
    return np.column_stack([-net, worst_loss])


def run_pcsc(
    return_history: ArrayLike,
    *,
    upper: float | ArrayLike,
    previous_weights: ArrayLike | None,
    max_one_way_turnover: float | None,
    holding_days: int,
    transaction_cost_bps: float,
    scenario_bounds: ScenarioBounds,
    archive_size: int,
    scenario_candidate_size: int,
    population_size: int,
    generations: int,
    scenario_update_interval: int,
    seed: int,
    archive_selection_rule: str = "coverage",
    scenario_feedback_set: str = "pareto",
    portfolio_feedback_enabled: bool = True,
    random_immigrant_fraction: float = 0.25,
    block_reset_probability: float = 0.20,
    block_shift_radius: int | None = None,
    numeric_mutation_scale_fraction: float = 0.10,
    crossover_probability: float = 0.9,
    crossover_eta: float = 15.0,
    mutation_eta: float = 20.0,
) -> PCSCEvolutionResult:
    """Run PCSC or one of its two core archive-feedback ablations."""

    history = np.asarray(return_history, dtype=float)
    if history.ndim != 2 or not np.all(np.isfinite(history)):
        raise ValueError("return_history must be a finite T x n matrix")
    if population_size < 4 or generations < 1 or scenario_update_interval < 1:
        raise ValueError("Invalid evolutionary sizes or update interval")
    if scenario_feedback_set not in {"pareto", "full_population"}:
        raise ValueError("scenario_feedback_set must be pareto or full_population")
    scenario_bounds.validate(history.shape[0])
    bounds = _upper_array(upper, history.shape[1])
    previous = (
        None if previous_weights is None else np.asarray(previous_weights, dtype=float)
    )
    rng = np.random.default_rng(seed)
    raw_population = rng.uniform(0.0, bounds, size=(population_size, history.shape[1]))
    population = _repair_population(
        raw_population,
        upper=bounds,
        previous_weights=previous,
        max_one_way_turnover=max_one_way_turnover,
    )
    mean_returns = history.mean(axis=0)
    # Before an archive exists, archive-based Pareto ranks are undefined.  The
    # initial archive therefore covers the full feasible initial population;
    # every subsequent update uses the current non-dominated set only.
    initial_update = update_scenario_archive(
        history,
        population,
        prior_archive=[],
        bounds=scenario_bounds,
        archive_size=archive_size,
        candidate_size=scenario_candidate_size,
        rng=rng,
        selection_rule=archive_selection_rule,
        random_immigrant_fraction=random_immigrant_fraction,
        block_reset_probability=block_reset_probability,
        block_shift_radius=block_shift_radius,
        numeric_scale_fraction=numeric_mutation_scale_fraction,
    )
    archive = initial_update.archive
    archive_blocks = np.stack(
        [scenario_returns(history, item, bounds=scenario_bounds) for item in archive]
    )
    objectives = _portfolio_objectives(
        population,
        mean_daily_returns=mean_returns,
        archive_blocks=archive_blocks,
        previous_weights=previous,
        holding_days=holding_days,
        transaction_cost_bps=transaction_cost_bps,
    )
    generation_log: list[PCSCGenerationRecord] = []
    archive_updates = 1

    for generation in range(1, generations + 1):
        ranks, fronts = fast_nondominated_sort(objectives)
        crowding = crowding_distances(objectives, fronts)
        archive_updated = False
        update_coverage: float | None = None
        if portfolio_feedback_enabled and generation % scenario_update_interval == 0:
            feedback_portfolios = (
                population[fronts[0]]
                if scenario_feedback_set == "pareto"
                else population
            )
            update: ScenarioArchiveUpdate = update_scenario_archive(
                history,
                feedback_portfolios,
                prior_archive=archive,
                bounds=scenario_bounds,
                archive_size=archive_size,
                candidate_size=scenario_candidate_size,
                rng=rng,
                selection_rule=archive_selection_rule,
                random_immigrant_fraction=random_immigrant_fraction,
                block_reset_probability=block_reset_probability,
                block_shift_radius=block_shift_radius,
                numeric_scale_fraction=numeric_mutation_scale_fraction,
            )
            archive = update.archive
            archive_blocks = np.stack(
                [
                    scenario_returns(history, item, bounds=scenario_bounds)
                    for item in archive
                ]
            )
            # This re-evaluation is the scenario-to-portfolio feedback: both ranks
            # and tournament pressure immediately use the updated archive losses.
            objectives = _portfolio_objectives(
                population,
                mean_daily_returns=mean_returns,
                archive_blocks=archive_blocks,
                previous_weights=previous,
                holding_days=holding_days,
                transaction_cost_bps=transaction_cost_bps,
            )
            ranks, fronts = fast_nondominated_sort(objectives)
            crowding = crowding_distances(objectives, fronts)
            archive_updated = True
            update_coverage = update.final_coverage
            archive_updates += 1

        offspring_rows: list[NDArray[np.float64]] = []
        lower = np.zeros(history.shape[1], dtype=float)
        while len(offspring_rows) < population_size:
            first_parent = population[_tournament(rng, ranks, crowding)]
            second_parent = population[_tournament(rng, ranks, crowding)]
            first_child, second_child = _sbx_pair(
                first_parent,
                second_parent,
                lower=lower,
                upper=bounds,
                probability=crossover_probability,
                eta=crossover_eta,
                rng=rng,
            )
            for child in (first_child, second_child):
                child = _polynomial_mutation(
                    child,
                    lower=lower,
                    upper=bounds,
                    probability=1.0 / history.shape[1],
                    eta=mutation_eta,
                    rng=rng,
                )
                offspring_rows.append(child)
                if len(offspring_rows) == population_size:
                    break
        offspring = _repair_population(
            np.stack(offspring_rows),
            upper=bounds,
            previous_weights=previous,
            max_one_way_turnover=max_one_way_turnover,
        )
        offspring_objectives = _portfolio_objectives(
            offspring,
            mean_daily_returns=mean_returns,
            archive_blocks=archive_blocks,
            previous_weights=previous,
            holding_days=holding_days,
            transaction_cost_bps=transaction_cost_bps,
        )
        combined_population = np.vstack([population, offspring])
        combined_objectives = np.vstack([objectives, offspring_objectives])
        survive = _survival_indices(combined_objectives, population_size)
        population = combined_population[survive]
        objectives = combined_objectives[survive]
        final_ranks, final_fronts = fast_nondominated_sort(objectives)
        generation_log.append(
            PCSCGenerationRecord(
                generation=generation,
                archive_updated=archive_updated,
                pareto_size=len(final_fronts[0]),
                archive_coverage_at_update=update_coverage,
                best_negative_net_return=float(objectives[:, 0].min()),
                best_worst_stress_loss=float(objectives[:, 1].min()),
            )
        )

    _, fronts = fast_nondominated_sort(objectives)
    pareto_indices = fronts[0]
    pareto_weights = population[pareto_indices]
    pareto_objectives = objectives[pareto_indices]
    selected = select_normalized_ideal(pareto_objectives)
    return PCSCEvolutionResult(
        pareto_weights=pareto_weights,
        pareto_objectives=pareto_objectives,
        selected_index=selected,
        selected_weights=pareto_weights[selected].copy(),
        selected_objectives=pareto_objectives[selected].copy(),
        archive=archive,
        generation_log=tuple(generation_log),
        archive_update_count=archive_updates,
        seed=seed,
        archive_selection_rule=archive_selection_rule,
        scenario_feedback_set=scenario_feedback_set,
        portfolio_feedback_enabled=portfolio_feedback_enabled,
    )
