from __future__ import annotations

import unittest

import numpy as np

from src.scenarios.stress import (
    ScenarioBounds,
    StressScenario,
    historical_blocks,
    path_loss,
    repair_scenario,
    scenario_returns,
    select_fixed_worst_block_starts,
    stress_return_transform,
)


class StressScenarioTests(unittest.TestCase):
    def test_stress_transform_matches_definition(self) -> None:
        block = np.array([[0.02, 0.00], [-0.04, -0.02]])
        transformed = stress_return_transform(
            block, market_multiplier=1.5, residual_compression=0.5
        )
        market = block.mean(axis=1, keepdims=True)
        expected = 1.5 * market + 0.5 * (block - market)
        np.testing.assert_allclose(transformed, expected)

    def test_path_loss_is_additive_maximum_drawdown(self) -> None:
        returns = np.array([[0.10], [-0.20], [-0.10], [0.05]])
        self.assertAlmostEqual(path_loss(np.array([1.0]), returns), 0.30)
        self.assertAlmostEqual(
            path_loss(np.array([1.0]), np.array([[0.01], [0.02]])), 0.0
        )

    def test_scenario_repair_and_materialization(self) -> None:
        history = np.arange(20, dtype=float).reshape(10, 2) / 1000.0
        bounds = ScenarioBounds(4, 1.0, 1.5, 0.0, 0.5)
        repaired = repair_scenario(
            StressScenario(99, 2.0, -1.0), bounds=bounds, history_length=10
        )
        self.assertEqual(repaired, StressScenario(6, 1.5, 0.0))
        materialized = scenario_returns(history, repaired, bounds=bounds)
        self.assertEqual(materialized.shape, (4, 2))

    def test_historical_blocks_and_fixed_selection(self) -> None:
        history = np.array([[0.01], [0.01], [-0.10], [-0.10], [0.02]])
        blocks = historical_blocks(history, block_length=2)
        self.assertEqual(blocks.shape, (4, 2, 1))
        starts = select_fixed_worst_block_starts(
            history, block_length=2, count=1
        )
        self.assertEqual(starts.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
