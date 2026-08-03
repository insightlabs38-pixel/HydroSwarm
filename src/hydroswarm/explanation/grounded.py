"""Fixed explanation intents rendered only from verified structured evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Mapping, Sequence


class ExplanationIntent(StrEnum):
    WHY_SOURCE = "WHY_SOURCE"
    WHY_SAMPLE = "WHY_SAMPLE"
    WHAT_CHANGED = "WHAT_CHANGED"
    WHY_PLAN_REJECTED = "WHY_PLAN_REJECTED"
    COMPARE_PLANS = "COMPARE_PLANS"
    UNCERTAINTY_REMAINS = "UNCERTAINTY_REMAINS"
    WHICH_SENSOR_MATTERED = "WHICH_SENSOR_MATTERED"


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    source_node: str | None
    source_probability: float | None
    candidate_region: tuple[str, ...]
    candidate_coverage: float | None
    recommended_sample: str | None
    information_gain_bits: float | None
    candidates_before: int | None
    candidates_after: int | None
    selected_plan: str | None
    rejected_plan: str | None
    rejection_codes: tuple[str, ...]
    exposure_reduction_mg: float | None
    pressure_violation_minutes: float | None
    service_availability: float | None
    disagreement_js: float | None
    ood_level: str
    approval_pending: bool
    abstention_reason: str | None
    supporting_sensors: tuple[str, ...] = ()
    removed_candidates: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class GroundedExplanation:
    intent: ExplanationIntent
    text: str
    facts: Mapping[str, str | float | int | bool | tuple[str, ...] | None]
    limitations: tuple[str, ...]


def _pct(value: float | None) -> str:
    return "not measured" if value is None else f"{100 * value:.1f}%"


def explain(intent: ExplanationIntent, evidence: EvidenceBundle) -> GroundedExplanation:
    facts: dict[str, str | float | int | bool | tuple[str, ...] | None] = {
        "source_node": evidence.source_node,
        "source_probability": evidence.source_probability,
        "candidate_region": evidence.candidate_region,
        "candidate_coverage": evidence.candidate_coverage,
        "recommended_sample": evidence.recommended_sample,
        "information_gain_bits": evidence.information_gain_bits,
        "disagreement_js": evidence.disagreement_js,
        "ood_level": evidence.ood_level,
        "approval_pending": evidence.approval_pending,
    }
    if intent == ExplanationIntent.WHY_SOURCE:
        text = (
            f"{evidence.source_node or 'No single node'} is the leading source at "
            f"{_pct(evidence.source_probability)} probability. The calibrated candidate region is "
            f"{', '.join(evidence.candidate_region) or 'empty'}, supported by sensors "
            f"{', '.join(evidence.supporting_sensors) or 'none'}."
        )
    elif intent == ExplanationIntent.WHY_SAMPLE:
        text = (
            f"Sample {evidence.recommended_sample or 'none'} because its expected information gain is "
            f"{evidence.information_gain_bits if evidence.information_gain_bits is not None else 'not measured'} bits."
        )
    elif intent == ExplanationIntent.WHAT_CHANGED:
        text = (
            f"The candidate set changed from {evidence.candidates_before} to {evidence.candidates_after}. "
            + (" ".join(f"{node} was removed: {reason}." for node, reason in (evidence.removed_candidates or {}).items()))
        )
    elif intent == ExplanationIntent.WHY_PLAN_REJECTED:
        text = (
            f"Plan {evidence.rejected_plan or 'unknown'} was rejected by exact simulation: "
            f"{', '.join(evidence.rejection_codes) or 'no verified reason supplied'}."
        )
    elif intent == ExplanationIntent.COMPARE_PLANS:
        text = (
            f"Selected plan {evidence.selected_plan or 'none'} changes modeled contaminant consumption by "
            f"{evidence.exposure_reduction_mg if evidence.exposure_reduction_mg is not None else 'not measured'} mg, "
            f"with {evidence.pressure_violation_minutes if evidence.pressure_violation_minutes is not None else 'not measured'} "
            f"pressure-violation minutes and {_pct(evidence.service_availability)} service availability."
        )
    elif intent == ExplanationIntent.UNCERTAINTY_REMAINS:
        text = (
            f"Remaining uncertainty: disagreement={evidence.disagreement_js}, OOD={evidence.ood_level}, "
            f"candidate region={', '.join(evidence.candidate_region) or 'none'}. "
            f"{evidence.abstention_reason or 'No abstention is active.'}"
        )
    else:
        text = (
            f"Sensors {', '.join(evidence.supporting_sensors) or 'none'} had the greatest measured "
            "remove-one sensitivity. This is sensitivity evidence, not causal proof."
        )
    return GroundedExplanation(
        intent=intent,
        text=text,
        facts=facts,
        limitations=(
            "Simulator outcomes depend on network and demand assumptions.",
            "Sensitivity evidence is not causal proof.",
            "A human operator must approve every response plan.",
        ),
    )


def deterministic_operational_summary(evidence: EvidenceBundle) -> str:
    approval = "OPERATOR APPROVAL PENDING" if evidence.approval_pending else "No approval pending"
    return (
        f"Source: {evidence.source_node or 'unresolved'} ({_pct(evidence.source_probability)}). "
        f"Candidate region: {', '.join(evidence.candidate_region) or 'none'}. "
        f"Next sample: {evidence.recommended_sample or 'not required'}. "
        f"Plan: {evidence.selected_plan or 'none verified'}. Exposure reduction: "
        f"{evidence.exposure_reduction_mg if evidence.exposure_reduction_mg is not None else 'not measured'} mg. "
        f"Pressure violations: {evidence.pressure_violation_minutes if evidence.pressure_violation_minutes is not None else 'not measured'} min. "
        f"Service: {_pct(evidence.service_availability)}. {approval}. "
        "Research decision support only; follow utility procedures."
    )


def remove_one_sensor_sensitivity(
    sensor_ids: Sequence[str],
    baseline_probabilities: Mapping[str, float],
    rerun_without_sensor: Callable[[str], Mapping[str, float]],
) -> Mapping[str, float]:
    """Measure total-variation belief change after removing each sensor."""
    scores: dict[str, float] = {}
    nodes = set(baseline_probabilities)
    for sensor_id in sensor_ids:
        changed = rerun_without_sensor(sensor_id)
        union = nodes | set(changed)
        scores[sensor_id] = 0.5 * sum(
            abs(baseline_probabilities.get(node, 0.0) - changed.get(node, 0.0)) for node in union
        )
    return scores


def plan_outcome_sensitivity(
    baseline: Mapping[str, float], alternatives: Sequence[Mapping[str, float]]
) -> Mapping[str, tuple[float, float]]:
    """Return min/max simulated outcome under alternate assumptions."""
    keys = set(baseline).union(*(set(item) for item in alternatives))
    return {
        key: (
            min([baseline.get(key, 0.0), *[item.get(key, 0.0) for item in alternatives]]),
            max([baseline.get(key, 0.0), *[item.get(key, 0.0) for item in alternatives]]),
        )
        for key in keys
    }

