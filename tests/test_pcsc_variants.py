from __future__ import annotations

import unittest

from src.pcsc.variants import PCSCVariant, VARIANT_MECHANISMS


class PCSCVariantTests(unittest.TestCase):
    def test_each_ablation_changes_exactly_one_core_switch(self) -> None:
        full = VARIANT_MECHANISMS[PCSCVariant.FULL]
        no_feedback = VARIANT_MECHANISMS[
            PCSCVariant.WITHOUT_PORTFOLIO_TO_SCENARIO_FEEDBACK
        ]
        no_coverage = VARIANT_MECHANISMS[
            PCSCVariant.WITHOUT_PORTFOLIO_WISE_COVERAGE
        ]
        self.assertEqual(
            [key for key in full if full[key] != no_feedback[key]],
            ["portfolio_feedback_enabled"],
        )
        self.assertEqual(
            [key for key in full if full[key] != no_coverage[key]],
            ["archive_selection_rule"],
        )


if __name__ == "__main__":
    unittest.main()
