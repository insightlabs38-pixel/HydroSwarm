"""Grounded deterministic and optional constrained language explanations."""

from hydroswarm.explanation.grounded import (
    EvidenceBundle,
    ExplanationIntent,
    GroundedExplanation,
    deterministic_operational_summary,
    explain,
    plan_outcome_sensitivity,
    remove_one_sensor_sensitivity,
)
from hydroswarm.explanation.language import ConstrainedLanguageDecoder

__all__ = [
    "ConstrainedLanguageDecoder",
    "EvidenceBundle",
    "ExplanationIntent",
    "GroundedExplanation",
    "deterministic_operational_summary",
    "explain",
    "plan_outcome_sensitivity",
    "remove_one_sensor_sensitivity",
]

