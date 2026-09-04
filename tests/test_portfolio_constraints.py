from __future__ import annotations

import unittest

import numpy as np

from src.portfolio.constraints import (
    one_way_turnover,
    project_capped_simplex,
    repair_weights,
)


class PortfolioConstraintTests(unittest.TestCase):
    def test_capped_simplex_projection(self) -> None:
        projected = project_capped_simplex(
            np.array([2.0, -1.0, 0.2, 0.1]), upper=0.4
        )
        self.assertAlmostEqual(float(projected.sum()), 1.0)
        self.assertTrue(np.all(projected >= 0.0))
        self.assertTrue(np.all(projected <= 0.4 + 1e-12))

    def test_infeasible_cap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            project_capped_simplex(np.ones(4), upper=0.2)

    def test_turnover_repair_respects_cap_and_turnover(self) -> None:
        previous = np.full(10, 0.1)
        candidate = np.arange(1.0, 11.0)
        repaired = repair_weights(
            candidate,
            upper=0.2,
            previous_weights=previous,
            max_one_way_turnover=0.15,
        )
        self.assertAlmostEqual(float(repaired.sum()), 1.0)
        self.assertLessEqual(float(repaired.max()), 0.2 + 1e-10)
        self.assertLessEqual(one_way_turnover(repaired, previous), 0.15 + 1e-9)

    def test_market_drift_cap_violation_has_minimum_turnover(self) -> None:
        previous = np.array([0.4, 0.3, 0.2, 0.1])
        with self.assertRaises(ValueError):
            repair_weights(
                np.ones(4),
                upper=0.3,
                previous_weights=previous,
                max_one_way_turnover=0.05,
            )
        repaired = repair_weights(
            np.ones(4),
            upper=0.3,
            previous_weights=previous,
            max_one_way_turnover=0.10,
        )
        self.assertAlmostEqual(one_way_turnover(repaired, previous), 0.10)
        self.assertLessEqual(float(repaired.max()), 0.3 + 1e-10)


if __name__ == "__main__":
    unittest.main()
