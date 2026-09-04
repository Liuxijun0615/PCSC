"""Run a small, deterministic PCSC example on synthetic return data.

This script is a smoke test for the optimizer. It does not reproduce the
paper's empirical results, which require the separately prepared A-share data.
"""

from __future__ import annotations

import numpy as np

from src.pcsc.evolution import run_pcsc
from src.portfolio.constraints import one_way_turnover
from src.scenarios.stress import ScenarioBounds


def make_synthetic_returns(
    *, observations: int = 120, assets: int = 10, seed: int = 2026
) -> np.ndarray:
    """Create correlated daily returns for a lightweight demonstration."""

    rng = np.random.default_rng(seed)
    market = rng.normal(0.0002, 0.012, size=(observations, 1))
    idiosyncratic = rng.normal(0.0, 0.015, size=(observations, assets))
    return 0.45 * market + 0.55 * idiosyncratic


def main() -> None:
    returns = make_synthetic_returns()
    asset_count = returns.shape[1]
    previous_weights = np.full(asset_count, 1.0 / asset_count)

    result = run_pcsc(
        returns,
        upper=0.20,
        previous_weights=previous_weights,
        max_one_way_turnover=0.30,
        holding_days=20,
        transaction_cost_bps=10,
        scenario_bounds=ScenarioBounds(
            block_length=20,
            market_multiplier_min=1.0,
            market_multiplier_max=1.5,
            residual_compression_min=0.0,
            residual_compression_max=0.5,
        ),
        archive_size=6,
        scenario_candidate_size=12,
        population_size=24,
        generations=10,
        scenario_update_interval=2,
        seed=7,
    )

    turnover = one_way_turnover(result.selected_weights, previous_weights)
    print("PCSC synthetic smoke test completed")
    print(f"Pareto portfolios: {len(result.pareto_weights)}")
    print(f"Scenario archive size: {len(result.archive)}")
    print(f"Archive updates: {result.archive_update_count}")
    print(f"Selected objectives: {result.selected_objectives.tolist()}")
    print(f"Weight sum: {result.selected_weights.sum():.10f}")
    print(f"Maximum weight: {result.selected_weights.max():.10f}")
    print(f"One-way turnover: {turnover:.10f}")


if __name__ == "__main__":
    main()
