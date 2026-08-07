"""Canonical semantic output names and granular output governance
(core-issues4.txt Section C / Phase 9.2).

Prior to this module, runtime gating was role-only: `hydroswarm.tasks.
RUNTIME_TASKS` gates by role ("sentinel"/"scout"/"strategist"/"ood"), so a
single trained head (e.g. source_node) being validated silently authorized
every other head sharing that role (e.g. source_region, sensor_fault) even
if THAT head never received a real gradient. This module names every
individual learned output HydroCore can produce and defines the strict
governance invariant a v4 checkpoint identity must satisfy:

    runtime_enabled_outputs <= validated_outputs <= trained_outputs

No output name may appear in any of the governance sets unless it is a
member of CANONICAL_OUTPUT_NAMES -- an unknown name is a defect in whatever
produced the set (e.g. a typo, or a name that was renamed on one side and
not the other), not a new output to silently accept.

This module intentionally has no opinion on which outputs a given
checkpoint actually enables -- that is the checkpoint identity's job (see
`hydroswarm.training.checkpoint_identity`). It only defines the fixed
vocabulary and the invariant every checkpoint's governance sets must obey.
"""

from __future__ import annotations

#: Sentinel: incident/node-level scientific outputs Sentinel supervision
#: covers today (core-issues4.txt Section C).
SENTINEL_OUTPUTS: frozenset[str] = frozenset({
    "source_node",
    "source_region",
    "start_time",
    "duration",
    "relative_strength",
    "event_presence",
    "event_cause",
    "sensor_fault",
    "evidence_sufficiency",
})

#: Scout: next-sample selection outputs.
SCOUT_OUTPUTS: frozenset[str] = frozenset({
    "sample_node",
    "information_gain",
    "candidate_reduction",
    "should_continue_sampling",
})

#: Strategist: bounded-plan ranking/prescreening outputs. Deliberately does
#: NOT include action_logits/action_pointer_logits -- core-issues4.txt
#: Section D.6's preferred decision is that deterministic candidate plans
#: own action-template/target identity and the learned model only ranks
#: the actual candidates it is given, so those two heads are diagnostic-only
#: at best in v4 (see checkpoint_identity.DIAGNOSTIC_ONLY_OUTPUTS_V4).
STRATEGIST_OUTPUTS: frozenset[str] = frozenset({
    "plan_validity",
    "plan_value",
    "exposure_proxy",
    "pressure_risk_proxy",
    "service_loss_proxy",
    "containment_time_proxy",
    "plan_regret_proxy",
})

#: OOD / control: incident-level advisory outputs. `ood_category` is the
#: learned 11-class advisory signal (hydroswarm.training.ood_categories.
#: OODCategory); it is explicitly NOT the deterministic three-level
#: severity hydroswarm.inference.ood.OODDetector computes and remains
#: authoritative regardless of this head's validation state.
OOD_CONTROL_OUTPUTS: frozenset[str] = frozenset({
    "ood_category",
    "next_step",
})

#: Auxiliary: never operator-authoritative (core-issues2.txt Phase 6 rule
#: 6, core-issues4.txt Section D.11-12). `future_concentration` is included
#: in the canonical vocabulary even though it is currently unsupported
#: (Phase 7.4: disabled, all-masked) -- see checkpoint_identity's
#: TRAINING_ONLY_OUTPUTS_V4/UNSUPPORTED set for why it cannot appear in any
#: v4 checkpoint's trained_outputs today.
AUXILIARY_OUTPUTS: frozenset[str] = frozenset({
    "sensor_reconstruction",
    "travel_time",
    "future_concentration",
})

#: Every name any HydroCore output can canonically be governed under.
CANONICAL_OUTPUT_NAMES: frozenset[str] = (
    SENTINEL_OUTPUTS | SCOUT_OUTPUTS | STRATEGIST_OUTPUTS | OOD_CONTROL_OUTPUTS | AUXILIARY_OUTPUTS
)

#: Which role each canonical output name belongs to -- used only for
#: reporting/grouping (e.g. a human-readable governance report), never for
#: gating itself (that is exactly the role-only granularity this module
#: replaces).
OUTPUT_ROLE: dict[str, str] = {
    **{name: "sentinel" for name in SENTINEL_OUTPUTS},
    **{name: "scout" for name in SCOUT_OUTPUTS},
    **{name: "strategist" for name in STRATEGIST_OUTPUTS},
    **{name: "ood_control" for name in OOD_CONTROL_OUTPUTS},
    **{name: "auxiliary" for name in AUXILIARY_OUTPUTS},
}


class OutputGovernanceError(Exception):
    """Raised when a set of trained/validated/runtime-enabled output names
    violates the canonical-vocabulary or subset invariants."""


def validate_output_governance(
    *,
    trained_outputs: frozenset[str],
    validated_outputs: frozenset[str],
    runtime_enabled_outputs: frozenset[str],
    diagnostic_only_outputs: frozenset[str],
    training_only_outputs: frozenset[str],
) -> None:
    """Fail closed on any governance-set violation. Does not check that
    every canonical output is *covered* by exactly one set -- an output can
    legitimately belong to none of them (e.g. removed/unimplemented)."""

    all_sets = {
        "trained_outputs": trained_outputs,
        "validated_outputs": validated_outputs,
        "runtime_enabled_outputs": runtime_enabled_outputs,
        "diagnostic_only_outputs": diagnostic_only_outputs,
        "training_only_outputs": training_only_outputs,
    }
    for set_name, names in all_sets.items():
        unknown = names - CANONICAL_OUTPUT_NAMES
        if unknown:
            raise OutputGovernanceError(
                f"{set_name} contains unknown output name(s) not in CANONICAL_OUTPUT_NAMES: "
                f"{sorted(unknown)}"
            )
    if not runtime_enabled_outputs <= validated_outputs:
        raise OutputGovernanceError(
            "runtime_enabled_outputs is not a subset of validated_outputs: "
            f"{sorted(runtime_enabled_outputs - validated_outputs)} are runtime-enabled but "
            "not validated"
        )
    if not validated_outputs <= trained_outputs:
        raise OutputGovernanceError(
            "validated_outputs is not a subset of trained_outputs: "
            f"{sorted(validated_outputs - trained_outputs)} are validated but not trained"
        )
    diagnostic_and_trained = diagnostic_only_outputs & trained_outputs
    if diagnostic_and_trained:
        raise OutputGovernanceError(
            "diagnostic_only_outputs overlaps trained_outputs -- an output is either a real "
            f"trained/validated/runtime-enabled candidate or diagnostic-only, never both: "
            f"{sorted(diagnostic_and_trained)}"
        )
    training_and_runtime = training_only_outputs & runtime_enabled_outputs
    if training_and_runtime:
        raise OutputGovernanceError(
            "training_only_outputs overlaps runtime_enabled_outputs -- a training-only output "
            f"(e.g. an auxiliary head) must never be runtime-enabled: {sorted(training_and_runtime)}"
        )
