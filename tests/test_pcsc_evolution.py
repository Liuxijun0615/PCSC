from __future__ import annotations

import unittest

import numpy as np

from src.pcsc.evolution import fast_nondominated_sort, run_pcsc
from src.portfolio.constraints import one_way_turnover
from src.scenarios.stress import ScenarioBounds


class PCSCEvolutionTests(unittest.TestCase):
    def test_fast_nondominated_sort(self) -> None:
        ranks, fronts = fast_nondominated_sort(
            np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 2.0], [2.0, 0.0]])
        )
        self.assertEqual(ranks.tolist(), [0, 0, 1, 0])
        self.assertEqual(fronts[0].tolist(), [0, 1, 3])

    def test_small_pcsc_run_is_feasible_and_updates_archive(self) -> None:
        rng = np.random.default_rng(31)
        history = rng.normal(0.0001, 0.015, size=(60, 10))
        previous = np.full(10, 0.1)
        result = run_pcsc(
            history,
            upper=0.2,
            previous_weights=previous,
            max_one_way_turnover=0.20,
            holding_days=20,
            transaction_cost_bps=10,
            scenario_bounds=ScenarioBounds(5, 1.0, 1.5, 0.0, 0.5),
            archive_size=3,
            scenario_candidate_size=7,
            population_size=12,
            generations=4,
            scenario_update_interval=2,
            seed=101,
        )
        self.assertEqual(result.archive_update_count, 3)
        self.assertEqual(len(result.generation_log), 4)
        self.assertEqual(
            [item.archive_updated for item in result.generation_log],
            [False, True, False, True],
        )
        np.testing.assert_allclose(
            result.pareto_weights.sum(axis=1), 1.0, atol=1e-8
        )
        self.assertTrue(np.all(result.pareto_weights <= 0.2 + 1e-8))
        self.assertTrue(
            all(
                one_way_turnover(row, previous) <= 0.20 + 1e-8
                for row in result.pareto_weights
            )
        )

    def test_feedback_ablation_keeps_initial_archive(self) -> None:
        history = np.random.default_rng(2).normal(0, 0.01, size=(40, 5))
        result = run_pcsc(
            history,
            upper=0.4,
            previous_weights=np.full(5, 0.2),
            max_one_way_turnover=0.3,
            holding_days=20,
            transaction_cost_bps=10,
            scenario_bounds=ScenarioBounds(4, 1.0, 1.5, 0.0, 0.5),
            archive_size=2,
            scenario_candidate_size=5,
            population_size=8,
            generations=3,
            scenario_update_interval=1,
            seed=8,
            portfolio_feedback_enabled=False,
        )
        self.assertEqual(result.archive_update_count, 1)
        self.assertFalse(any(item.archive_updated for item in result.generation_log))


if __name__ == "__main__":
    unittest.main()
