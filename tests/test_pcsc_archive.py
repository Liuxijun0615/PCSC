from __future__ import annotations

import unittest

import numpy as np

from src.pcsc.archive import (
    evaluate_scenario_candidates,
    select_average_loss_archive,
    select_mean_vulnerability_archive,
    update_scenario_archive,
)
from src.scenarios.stress import ScenarioBounds, StressScenario


class PCSCArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.history = rng.normal(0.0, 0.02, size=(60, 4))
        self.portfolios = np.array(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
            ]
        )
        self.bounds = ScenarioBounds(5, 1.0, 1.5, 0.0, 0.5)

    def test_candidate_evaluation_shapes(self) -> None:
        candidates = [
            StressScenario(0, 1.0, 0.0),
            StressScenario(10, 1.5, 0.5),
        ]
        result = evaluate_scenario_candidates(
            self.history,
            self.portfolios,
            candidates,
            bounds=self.bounds,
        )
        self.assertEqual(result.scenario_losses.shape, (3, 2))
        self.assertEqual(result.historical_losses.shape, (3, 56))
        self.assertEqual(result.vulnerability.shape, (3, 2))
        self.assertTrue(np.all(result.vulnerability >= 0.0))

    def test_archive_update_is_seed_reproducible_and_bounded(self) -> None:
        first = update_scenario_archive(
            self.history,
            self.portfolios,
            prior_archive=[],
            bounds=self.bounds,
            archive_size=4,
            candidate_size=10,
            rng=np.random.default_rng(99),
        )
        second = update_scenario_archive(
            self.history,
            self.portfolios,
            prior_archive=[],
            bounds=self.bounds,
            archive_size=4,
            candidate_size=10,
            rng=np.random.default_rng(99),
        )
        self.assertEqual(first.archive, second.archive)
        self.assertEqual(len(first.archive), 4)
        self.assertGreaterEqual(first.final_coverage, 0.0)
        self.assertTrue(np.all(np.diff(first.coverage_trace) >= -1e-12))
        for scenario in first.archive:
            self.assertGreaterEqual(scenario.block_start, 0)
            self.assertLessEqual(scenario.block_start, 55)
            self.assertGreaterEqual(scenario.market_multiplier, 1.0)
            self.assertLessEqual(scenario.market_multiplier, 1.5)
            self.assertGreaterEqual(scenario.residual_compression, 0.0)
            self.assertLessEqual(scenario.residual_compression, 0.5)

    def test_average_loss_ablation_is_deterministic(self) -> None:
        losses = np.array([[10.0, 0.0, 4.0], [0.0, 8.0, 4.0]])
        selected = select_average_loss_archive(losses, archive_size=2)
        self.assertEqual(selected.tolist(), [0, 1])

    def test_mean_vulnerability_ablation_uses_normalized_values(self) -> None:
        vulnerability = np.array([[1.0, 4.0, 0.0], [1.0, 0.0, 3.0]])
        selected = select_mean_vulnerability_archive(vulnerability, archive_size=2)
        self.assertEqual(selected.tolist(), [1, 2])

    def test_average_loss_archive_update_runs(self) -> None:
        result = update_scenario_archive(
            self.history,
            self.portfolios,
            prior_archive=[],
            bounds=self.bounds,
            archive_size=3,
            candidate_size=8,
            rng=np.random.default_rng(13),
            selection_rule="average_loss",
        )
        self.assertEqual(len(result.archive), 3)
        self.assertEqual(result.coverage_trace.shape, (3,))

    def test_mean_vulnerability_archive_update_runs(self) -> None:
        result = update_scenario_archive(
            self.history,
            self.portfolios,
            prior_archive=[],
            bounds=self.bounds,
            archive_size=3,
            candidate_size=8,
            rng=np.random.default_rng(13),
            selection_rule="mean_vulnerability",
        )
        self.assertEqual(len(result.archive), 3)


if __name__ == "__main__":
    unittest.main()
