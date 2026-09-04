from __future__ import annotations

import unittest

import numpy as np

from src.portfolio.objectives import (
    expected_holding_period_net_return,
    historical_cvar_loss,
    portfolio_returns,
)


class PortfolioObjectiveTests(unittest.TestCase):
    def test_portfolio_returns_and_historical_cvar(self) -> None:
        returns = np.array([[-0.10, 0.0], [0.0, -0.20], [0.10, 0.10]])
        weights = np.array([0.5, 0.5])
        np.testing.assert_allclose(
            portfolio_returns(returns, weights), [-0.05, -0.10, 0.10]
        )
        self.assertAlmostEqual(
            historical_cvar_loss(returns, weights, confidence=2 / 3), 0.10
        )

    def test_expected_net_return_charges_one_way_cost(self) -> None:
        value = expected_holding_period_net_return(
            np.array([0.001, 0.0]),
            np.array([0.75, 0.25]),
            holding_days=20,
            previous_weights=np.array([0.5, 0.5]),
            transaction_cost_bps=10,
        )
        self.assertAlmostEqual(value, 0.015 - 0.00025)

    def test_initial_investment_charges_full_not_half_turnover(self) -> None:
        value = expected_holding_period_net_return(
            np.zeros(2),
            np.array([0.5, 0.5]),
            holding_days=20,
            previous_weights=None,
            transaction_cost_bps=10,
        )
        self.assertAlmostEqual(value, -0.001)


if __name__ == "__main__":
    unittest.main()
