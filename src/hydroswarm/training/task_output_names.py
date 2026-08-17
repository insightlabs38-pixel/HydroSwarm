"""Canonical `task_name -> output_name` mapping, extracted as shared,
importable data from `hydroswarm.training.losses.compute_multitask_loss`'s
own internal routing dicts/branches (never redefined independently -- this
module is the single place both `run_m10_2_supervision_audit.py` and
`hydroswarm.training.gradient_coverage` import it from, so the two stay
provably consistent with each other and with `compute_multitask_loss`
itself).

`assert_consistent_with_all_task_names()` fails importing this module if a
future change to `losses.py`'s own task vocabulary (`ALL_TASK_NAMES`) is not
mirrored here -- a drift between this table and the real routing logic would
silently make every downstream consumer's classification wrong.
"""

from __future__ import annotations

from .losses import ALL_TASK_NAMES, PROFILE_CLASS_COUNTS

_CLASSIFICATION_OUTPUTS: dict[str, str] = {
    "source_node": "source_node_logits",
    "source_region": "source_region_logits",
    "sample_node": "sample_node_logits",
    "action_template": "action_logits",
    "target_pointer": "action_pointer_logits",
    "plan_validity": "plan_validity_logits",
    "ood_class": "ood_category_logits",
    "event_cause": "event_cause_logits",
    "next_step": "next_step_logits",
}
_PROFILE_OUTPUTS: dict[str, str] = {task: f"{task}_logits" for task in PROFILE_CLASS_COUNTS}
_REGRESSION_OUTPUTS: dict[str, str] = {
    "plan_value": "plan_value",
    "information_gain": "expected_information_gain",
    "candidate_reduction": "candidate_reduction_prediction",
    "sensor_reconstruction": "sensor_reconstruction_prediction",
    "future_concentration": "future_concentration_prediction",
    "travel_time": "travel_time_prediction",
    "exposure_proxy": "exposure_proxy",
    "pressure_risk_proxy": "pressure_risk_proxy",
    "service_loss_proxy": "service_loss_proxy",
    "containment_time_proxy": "containment_time_proxy",
    "plan_regret_proxy": "plan_regret_proxy",
}
_SPECIAL_OUTPUTS: dict[str, str] = {
    "sensor_fault": "sensor_fault_logits",
    "event_presence": "event_presence_logits",
    "evidence_sufficiency": "evidence_sufficiency",
    "should_continue_sampling": "should_continue_sampling_logits",
}

TASK_OUTPUT_NAMES: dict[str, str] = {
    **_CLASSIFICATION_OUTPUTS,
    **_PROFILE_OUTPUTS,
    **_REGRESSION_OUTPUTS,
    **_SPECIAL_OUTPUTS,
}


def assert_consistent_with_all_task_names() -> None:
    if set(TASK_OUTPUT_NAMES) != ALL_TASK_NAMES:
        raise AssertionError(
            "TASK_OUTPUT_NAMES has drifted from hydroswarm.training.losses.ALL_TASK_NAMES: "
            f"missing={sorted(ALL_TASK_NAMES - set(TASK_OUTPUT_NAMES))}, "
            f"extra={sorted(set(TASK_OUTPUT_NAMES) - ALL_TASK_NAMES)}"
        )


assert_consistent_with_all_task_names()

#: Governed target names excluded from `hydroswarm.training.output_governance.
#: CANONICAL_OUTPUT_NAMES` by design (checkpoint_identity.py Section D item 6)
#: -- deterministic candidate plans own action-template/target identity, the
#: model only ranks/validates candidates it is given.
LEGACY_EXCLUDED_TASK_NAMES: frozenset[str] = frozenset({"action_template", "target_pointer"})

#: Raw, always-present-in-a-forward-pass output keys with no governed target
#: name at all (checkpoint_identity.py Section D items 1/2/9).
LEGACY_UNGOVERNED_OUTPUT_KEYS: tuple[str, ...] = ("uncertainty", "ood_logits", "sentinel", "scout", "strategist")
