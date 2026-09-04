"""Frozen mechanism switches for PCSC and its two single-factor ablations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from src.pcsc.evolution import PCSCEvolutionResult, run_pcsc


class PCSCVariant(StrEnum):
    FULL = "pcsc"
    WITHOUT_PORTFOLIO_TO_SCENARIO_FEEDBACK = "w_o_portfolio_to_scenario_feedback"
    WITHOUT_PORTFOLIO_WISE_COVERAGE = "w_o_portfolio_wise_coverage"


VARIANT_MECHANISMS = {
    PCSCVariant.FULL: {
        "archive_selection_rule": "coverage",
        "scenario_feedback_set": "pareto",
        "portfolio_feedback_enabled": True,
    },
    PCSCVariant.WITHOUT_PORTFOLIO_TO_SCENARIO_FEEDBACK: {
        "archive_selection_rule": "coverage",
        "scenario_feedback_set": "pareto",
        "portfolio_feedback_enabled": False,
    },
    PCSCVariant.WITHOUT_PORTFOLIO_WISE_COVERAGE: {
        "archive_selection_rule": "mean_vulnerability",
        "scenario_feedback_set": "pareto",
        "portfolio_feedback_enabled": True,
    },
}


def run_pcsc_variant(
    variant: PCSCVariant | str, *args: Any, **kwargs: Any
) -> PCSCEvolutionResult:
    selected = PCSCVariant(variant)
    forbidden = set(VARIANT_MECHANISMS[selected]).intersection(kwargs)
    if forbidden:
        raise ValueError(
            "Variant mechanism switches cannot be overridden: "
            + ", ".join(sorted(forbidden))
        )
    return run_pcsc(*args, **kwargs, **VARIANT_MECHANISMS[selected])
