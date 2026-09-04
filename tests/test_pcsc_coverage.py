from __future__ import annotations

import unittest

import numpy as np

from src.pcsc.coverage import (
    archive_coverage,
    greedy_coverage_archive,
    nondominated_mask,
    portfolio_wise_vulnerability,
)
from src.pcsc.losses import path_losses_for_blocks


class PCSCCoverageTests(unittest.TestCase):
    def test_portfolio_wise_normalization_removes_loss_scale(self) -> None:
        historical = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        scenarios = np.array([[4.0, 1.0], [40.0, 10.0]])
        vulnerability = portfolio_wise_vulnerability(
            scenarios, historical, epsilon=1e-12
        )
        np.testing.assert_allclose(vulnerability, [[2.0, 0.0], [2.0, 0.0]])

    def test_greedy_archive_uses_marginal_maximum_coverage(self) -> None:
        vulnerability = np.array([[9.0, 0.0, 4.0], [0.0, 8.0, 4.0]])
        selection = greedy_coverage_archive(vulnerability, archive_size=2)
        self.assertEqual(selection.selected_indices.tolist(), [0, 1])
        self.assertAlmostEqual(selection.final_coverage, 8.5)
        self.assertAlmostEqual(
            archive_coverage(vulnerability[:, selection.selected_indices]), 8.5
        )
        self.assertTrue(np.all(np.diff(selection.coverage_trace) >= 0.0))

    def test_nondominated_mask_for_minimization(self) -> None:
        objectives = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 2.0], [2.0, 0.0]])
        self.assertEqual(
            nondominated_mask(objectives).tolist(), [True, True, False, True]
        )

    def test_vectorized_path_losses(self) -> None:
        portfolios = np.array([[1.0, 0.0], [0.5, 0.5]])
        blocks = np.array(
            [
                [[0.10, 0.0], [-0.20, -0.10], [-0.10, 0.0]],
                [[0.01, 0.01], [0.02, 0.02], [0.01, 0.01]],
            ]
        )
        losses = path_losses_for_blocks(portfolios, blocks)
        np.testing.assert_allclose(losses[:, 0], [0.30, 0.20])
        np.testing.assert_allclose(losses[:, 1], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
