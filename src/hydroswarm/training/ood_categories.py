"""Governed OOD categories, expected behavior, and abstention outcomes
(overnight-plan.txt Task 2.5).

hydroswarm.inference.ood.OODDetector already computes a coarse *severity*
level (NORMAL/CAUTION/OUTSIDE_VALIDATED_RANGE) from a weighted combination
of signals. This module adds the *category* taxonomy the plan asks for --
which specific thing is novel -- plus the governed, per-category statement
of what the controller must do about it, and a small classifier for
scoring abstention decisions against ground truth (false abstention vs.
unsafe non-abstention), separate from severity.

In-distribution uncertainty (the model is unsure but the input is
ordinary) and OOD novelty (the input itself is outside the validated
operating range) are deliberately kept as different concepts: severity
covers the former, category membership covers the latter. An example can
have high in-distribution uncertainty with OODCategory.NONE, or be clearly
novel (e.g. UNSEEN_TOPOLOGY) with a confident-looking but untrustworthy
raw prediction.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum


class OODCategory(str, Enum):
    NONE = "NONE"  # in-distribution
    UNSEEN_TOPOLOGY = "UNSEEN_TOPOLOGY"
    UNSEEN_SENSOR_LAYOUT = "UNSEEN_SENSOR_LAYOUT"
    EXTREME_DEMAND = "EXTREME_DEMAND"
    TANK_STATE_SHIFT = "TANK_STATE_SHIFT"
    VALVE_PUMP_MISMATCH = "VALVE_PUMP_MISMATCH"
    ROUGHNESS_MISMATCH = "ROUGHNESS_MISMATCH"
    SEVERE_MISSINGNESS = "SEVERE_MISSINGNESS"
    FROZEN_DRIFTING_SENSOR = "FROZEN_DRIFTING_SENSOR"
    TIMING_OUTSIDE_TRAINING_RANGE = "TIMING_OUTSIDE_TRAINING_RANGE"
    UNSUPPORTED_NETWORK_ELEMENT_OR_INVALID_CALIBRATION = (
        "UNSUPPORTED_NETWORK_ELEMENT_OR_INVALID_CALIBRATION"
    )


@dataclass(frozen=True, slots=True)
class ExpectedBehavior:
    category: OODCategory
    planning_permitted: bool
    calibration_valid: bool
    description: str

    def __post_init__(self) -> None:
        if not self.description:
            raise ValueError(f"expected behavior for {self.category} is missing a description")
        if self.planning_permitted and not self.calibration_valid:
            raise ValueError(
                f"{self.category}: planning may never be permitted while calibration is "
                "invalid -- an uncalibrated candidate set cannot bound risk for planning"
            )


#: Fail-closed by default: every real OOD category suppresses planning and
#: invalidates calibration unless explicitly reasoned otherwise below.
OOD_CATEGORY_BEHAVIOR: dict[OODCategory, ExpectedBehavior] = {
    OODCategory.NONE: ExpectedBehavior(
        category=OODCategory.NONE,
        planning_permitted=True,
        calibration_valid=True,
        description="In-distribution: normal calibrated operation.",
    ),
    OODCategory.UNSEEN_TOPOLOGY: ExpectedBehavior(
        category=OODCategory.UNSEEN_TOPOLOGY,
        planning_permitted=False,
        calibration_valid=False,
        description="Topology-specific calibration does not transfer to an unvalidated "
        "network; CAUTION and planning suppression until the topology is validated or a "
        "broader validated calibration artifact explicitly covers it.",
    ),
    OODCategory.UNSEEN_SENSOR_LAYOUT: ExpectedBehavior(
        category=OODCategory.UNSEEN_SENSOR_LAYOUT,
        planning_permitted=False,
        calibration_valid=False,
        description="Signature library and calibration were fit against a specific sensor "
        "layout; a materially different layout invalidates both until re-validated.",
    ),
    OODCategory.EXTREME_DEMAND: ExpectedBehavior(
        category=OODCategory.EXTREME_DEMAND,
        planning_permitted=False,
        calibration_valid=False,
        description="Demand far outside the training distribution shifts travel times and "
        "concentrations in ways the fitted signatures do not cover.",
    ),
    OODCategory.TANK_STATE_SHIFT: ExpectedBehavior(
        category=OODCategory.TANK_STATE_SHIFT,
        planning_permitted=False,
        calibration_valid=False,
        description="Tank levels far outside the training range change hydraulic gradients "
        "and flow directions beyond the fitted operating range.",
    ),
    OODCategory.VALVE_PUMP_MISMATCH: ExpectedBehavior(
        category=OODCategory.VALVE_PUMP_MISMATCH,
        planning_permitted=False,
        calibration_valid=False,
        description="An unexpected valve/pump configuration changes flow direction and "
        "connectivity; classical signatures for the assumed configuration no longer apply.",
    ),
    OODCategory.ROUGHNESS_MISMATCH: ExpectedBehavior(
        category=OODCategory.ROUGHNESS_MISMATCH,
        planning_permitted=False,
        calibration_valid=False,
        description="Pipe roughness far from the calibrated assumption changes travel time "
        "and attenuation beyond the fitted signature library.",
    ),
    OODCategory.SEVERE_MISSINGNESS: ExpectedBehavior(
        category=OODCategory.SEVERE_MISSINGNESS,
        planning_permitted=False,
        calibration_valid=False,
        description="Insufficient observed evidence to trust the calibrated candidate set; "
        "evidence_sufficiency should already be false in this state.",
    ),
    OODCategory.FROZEN_DRIFTING_SENSOR: ExpectedBehavior(
        category=OODCategory.FROZEN_DRIFTING_SENSOR,
        planning_permitted=False,
        calibration_valid=False,
        description="A faulty sensor can masquerade as contamination evidence; planning "
        "must wait for sensor-fault triage (INSPECT_SENSOR), not proceed on tainted evidence.",
    ),
    OODCategory.TIMING_OUTSIDE_TRAINING_RANGE: ExpectedBehavior(
        category=OODCategory.TIMING_OUTSIDE_TRAINING_RANGE,
        planning_permitted=False,
        calibration_valid=False,
        description="Start-time/duration far outside the training bin range is not covered "
        "by the fitted ordinal heads or signature library.",
    ),
    OODCategory.UNSUPPORTED_NETWORK_ELEMENT_OR_INVALID_CALIBRATION: ExpectedBehavior(
        category=OODCategory.UNSUPPORTED_NETWORK_ELEMENT_OR_INVALID_CALIBRATION,
        planning_permitted=False,
        calibration_valid=False,
        description="An element type the feature schema does not model, or a calibration "
        "artifact that failed its own hash/schema validation, must fail closed.",
    ),
}


def topology_calibration_is_valid(
    topology_hash: str,
    validated_topology_hashes: Collection[str],
    *,
    broader_validated_artifact_exists: bool = False,
) -> bool:
    """Unknown topology hashes invalidate topology-specific calibration
    unless a broader validated artifact explicitly covers them (an opt-in
    the caller must assert, never inferred)."""

    if topology_hash in validated_topology_hashes:
        return True
    return broader_validated_artifact_exists


class AbstentionOutcome(str, Enum):
    CORRECT_ABSTENTION = "CORRECT_ABSTENTION"
    FALSE_ABSTENTION = "FALSE_ABSTENTION"
    CORRECT_PROCEED = "CORRECT_PROCEED"
    UNSAFE_NON_ABSTENTION = "UNSAFE_NON_ABSTENTION"


def classify_abstention_outcome(*, abstained: bool, safe_to_proceed: bool) -> AbstentionOutcome:
    """Score one decision against ground truth, per the plan's explicit
    metrics: false abstention (abstained when it was actually safe to
    proceed) and unsafe non-abstention (proceeded when it was not safe --
    the metric the plan treats as the more serious failure mode)."""

    if abstained:
        return AbstentionOutcome.FALSE_ABSTENTION if safe_to_proceed else AbstentionOutcome.CORRECT_ABSTENTION
    return AbstentionOutcome.CORRECT_PROCEED if safe_to_proceed else AbstentionOutcome.UNSAFE_NON_ABSTENTION
