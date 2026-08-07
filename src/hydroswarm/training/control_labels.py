"""Control target label generation: evidence_sufficiency and next_step
(core-issues2.txt Phase 5).

## evidence_sufficiency

targets_v2's governed definition requires combining calibrated candidate-set
size, posterior entropy, classical-neural disagreement, sensor health and
missingness, OOD state, remaining sample budget, and calibration validity.
corpus.py's `_evidence_sufficiency` only ever implemented the sensor-health
subset (documented there as an accepted interim state, not an oversight).

This module extends that with two more signals that are now genuinely
available (Phase 2/3/4's trajectory and OOD-label infrastructure):
posterior entropy (the classical localizer's own uncertainty, a real
signal this repo already computes) and OOD-category calibration validity
(Phase 4's classify_ood_category + ood_categories.OOD_CATEGORY_BEHAVIOR).

Two signals from the full governed definition are still NOT included here,
documented rather than silently dropped:

- **calibrated candidate-set size / candidate coverage**: these require a
  CalibrationArtifact already fitted (via conformal calibration) against a
  *trained* Sentinel checkpoint. Evidence_sufficiency is itself a training
  target for that same checkpoint -- there is no calibration artifact to
  read from until after Stage 1 of Phase 8's staged training sequence
  ("1. Train the corrected Sentinel backbone... 6. Add validated
  auxiliary objectives... Stage 6: Calibration and final selection" in
  overnight-plan.txt). This is a structural ordering dependency, not
  something this module can work around by fabricating an uncalibrated
  substitute and calling it "calibrated."
- **classical-neural disagreement**: requires both a classical prediction
  (available) and a *trained* neural prediction (not available before
  Stage 1 training) to compare -- same structural dependency.

Both become available once a Stage-1-trained checkpoint exists; extending
this rule at that point is a natural, well-scoped follow-up, not attempted
here.

A third, independently-defined "evidence sufficient" notion exists in
hydroswarm.agents.sentinel.HydroSentinel.deterministic_fallback
(`len(region) <= 2 and bool(usable)`) -- an FSM-fallback-path
simplification, not this module's training-label rule. They are
deliberately NOT unified here: the FSM fallback only runs when the neural
Sentinel path itself is unavailable (a narrower, different circumstance),
and changing live controller fallback behavior is out of a label-
generation module's scope. Documented as a known open reconciliation, not
hidden.

## next_step

hydroswarm.agents.controller's EVIDENCE_CHECK state (the FSM's actual
deterministic policy) already encodes 3 of NextStep's 4 values:

    if ood_level == OUTSIDE_VALIDATED_RANGE: ABSTAIN
    elif evidence_sufficient: GENERATE_PLANS
    elif sample_count >= min(5, max_sampling_rounds): ABSTAIN
    else: COLLECT_SAMPLE

classify_next_step below mirrors this exactly (same branch order, same
budget bound) and adds the one case the live FSM has no distinct state
for: INSPECT_SENSOR. There is no dormant/partial implementation of this
branch anywhere in the repo (confirmed by inspection of controller.py) --
it is derived here from event_cause == SENSOR_FAULT (an already-governed,
already-trained target that exists specifically to distinguish "this
evidence pattern is a faulty sensor, not real contamination"), inserted
between the evidence-insufficient and budget-exhausted checks. This is
this module's own reasoned extension beyond the current literal FSM code,
not a claim that the live controller already branches this way -- flagged
plainly rather than presented as a straight port.
"""

from __future__ import annotations

import torch

from .ood_categories import OODCategory, OOD_CATEGORY_BEHAVIOR
from .targets_v2 import EventCause, NextStep

#: core-issues3.txt Phase 8 item 8: hydroswarm.agents.controller's FSMState
#: has no INSPECT_SENSOR-equivalent state at all (confirmed by inspection
#: of agents/schemas.py's FSMState and controller.py's ALLOWED_TRANSITIONS
#: -- EVIDENCE_CHECK only ever transitions to SAMPLE_SELECTION,
#: PLAN_GENERATION, or COMPLETE), so NextStep.INSPECT_SENSOR has no
#: matching branch in the specific controller (the agent FSM) this target
#: is meant to advise.
#:
#: This is NOT the same as "no live inspect-sensor concept exists
#: anywhere" -- hydroswarm.inference.pipeline already computes a DIFFERENT,
#: already-authoritative ControlAction.INSPECT_SENSORS via
#: inference.fusion.uncertainty_control(), triggered by
#: disagreement_js >= 0.5 (classical/neural disagreement), not by
#: event_cause == SENSOR_FAULT the way classify_next_step below derives
#: this target. Two independently-triggered "inspect the sensor" signals
#: exist in this codebase today, in different subsystems, agreeing only in
#: name -- reconciling them (or deciding they are genuinely different
#: questions that both deserve to exist) is real design work belonging to
#: Phase 9's granular output-gating/architecture-v4 contract, not
#: something to resolve unilaterally in a label-generation pass. Until
#: that reconciliation happens, NextStep.INSPECT_SENSOR remains a valid,
#: reproducible TRAINING label (classify_next_step's event_cause ==
#: SENSOR_FAULT derivation is a real governed target, not an invented
#: label) but is NOT runtime-enabled for the agent-FSM controller
#: specifically -- that controller has no state to fulfill it, regardless
#: of what the separately-authoritative inference-pipeline policy does.
NEXT_STEP_RUNTIME_ENABLED: frozenset[NextStep] = frozenset(NextStep) - frozenset({NextStep.INSPECT_SENSOR})

#: Posterior entropy (bits) at or below which the classical localizer's
#: belief is concentrated enough to count toward "sufficient" -- one
#: component of the governed rule; sensor health and OOD-category validity
#: are the other two implemented here. Tunable, like
#: _evidence_sufficiency's health_threshold; not fit from data.
DEFAULT_ENTROPY_THRESHOLD_BITS = 2.0

#: Same budget bound hydroswarm.agents.controller.SwarmController's
#: EVIDENCE_CHECK state applies (min(5, limits.max_sampling_rounds)).
MAXIMUM_SAMPLING_ROUNDS = 5

_NEXT_STEP_INDEX = {member: index for index, member in enumerate(NextStep)}


def classify_evidence_sufficiency(
    *,
    healthy_fraction: float,
    sensors_ever_healthy: int,
    posterior_entropy_bits: float,
    ood_category: OODCategory,
    health_threshold: float = 0.5,
    minimum_healthy_sensors: int = 2,
    entropy_threshold_bits: float = DEFAULT_ENTROPY_THRESHOLD_BITS,
) -> bool:
    """Sensor health (corpus.py's original rule) AND a concentrated
    classical posterior AND OOD-category calibration validity. All three
    must hold -- any one failing means the evidence is not trustworthy
    enough to plan against, matching the governed rule's fail-closed
    framing ("evidence_sufficiency should already be false" for OOD
    categories, per ood_categories.OOD_CATEGORY_BEHAVIOR's own
    SEVERE_MISSINGNESS description)."""

    if not OOD_CATEGORY_BEHAVIOR[ood_category].calibration_valid:
        return False
    if healthy_fraction < health_threshold or sensors_ever_healthy < minimum_healthy_sensors:
        return False
    return posterior_entropy_bits <= entropy_threshold_bits


def classify_next_step(
    *,
    ood_level_outside_validated_range: bool,
    evidence_sufficient: bool,
    sample_count: int,
    event_cause: EventCause,
    maximum_sampling_rounds: int = MAXIMUM_SAMPLING_ROUNDS,
) -> NextStep:
    """Mirrors hydroswarm.agents.controller's EVIDENCE_CHECK branch order
    exactly, extended with INSPECT_SENSOR (see module docstring)."""

    if ood_level_outside_validated_range:
        return NextStep.ABSTAIN
    if evidence_sufficient:
        return NextStep.GENERATE_PLANS
    if sample_count >= maximum_sampling_rounds:
        return NextStep.ABSTAIN
    if event_cause == EventCause.SENSOR_FAULT:
        return NextStep.INSPECT_SENSOR
    return NextStep.COLLECT_SAMPLE


def next_step_target(step: NextStep) -> dict[str, torch.Tensor]:
    return {"next_step": torch.tensor(_NEXT_STEP_INDEX[step])}
