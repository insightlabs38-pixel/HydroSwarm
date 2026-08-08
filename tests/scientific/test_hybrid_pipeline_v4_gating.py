"""core-issues3.txt Phase 15 item 2: granular runtime_enabled_outputs
gating on HybridInferencePipeline's v4-only advisory SemanticPredictions
fields (event_presence/event_cause/next_step), additive to the pre-existing
coarse trained_tasks role gating (tests/scientific/test_hybrid_pipeline.py
already covers that). Reuses that file's own PriorFollowingModel/_pipeline
fixtures."""

from __future__ import annotations

from uuid import uuid4

import torch

from hydroswarm.inference import HybridInferencePipeline

from test_hybrid_pipeline import PriorFollowingModel, _pipeline, _series


class EventAwareModel(PriorFollowingModel):
    """Adds event_presence/event_cause/next_step heads on top of
    PriorFollowingModel's existing sentinel/scout/strategist/ood outputs.
    Defaults to confident, SUPPORTED predictions (event_presence=True,
    event_cause=CONTAMINATION [index 0], next_step=GENERATE_PLANS
    [index 2]) so a test only needs to override what it is checking."""

    def __call__(self, batch):
        output = super().__call__(batch)
        output["event_presence_logits"] = torch.tensor([[4.0]])
        output["event_cause_logits"] = torch.tensor([[8.0, -8.0, -8.0, -8.0, -8.0]])
        output["next_step_logits"] = torch.tensor([[-8.0, -8.0, 8.0, -8.0]])
        # index 1 == UNSEEN_TOPOLOGY (OODCategory enum declaration order:
        # NONE, UNSEEN_TOPOLOGY, UNSEEN_SENSOR_LAYOUT, EXTREME_DEMAND,
        # TANK_STATE_SHIFT, VALVE_PUMP_MISMATCH, ROUGHNESS_MISMATCH,
        # SEVERE_MISSINGNESS, FROZEN_DRIFTING_SENSOR,
        # TIMING_OUTSIDE_TRAINING_RANGE,
        # UNSUPPORTED_NETWORK_ELEMENT_OR_INVALID_CALIBRATION) -- a
        # supported class, per ood_labels.SUPPORTED_OOD_CATEGORIES.
        output["ood_category_logits"] = torch.tensor(
            [[-8.0, 8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0]]
        )
        return output


def _analyze(model, *, runtime_enabled_outputs):
    pipeline, network = _pipeline(model)
    pipeline.runtime_enabled_outputs = runtime_enabled_outputs
    return pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])


def test_runtime_enabled_outputs_none_preserves_legacy_behavior() -> None:
    """The default (no v4 checkpoint identity) must behave exactly as
    before this pass -- every existing caller/test that never sets
    runtime_enabled_outputs keeps working unchanged."""

    result = _analyze(EventAwareModel(), runtime_enabled_outputs=None)
    semantics = result.semantic_predictions
    assert semantics.event_presence is True
    assert semantics.event_presence_probability is not None
    assert semantics.event_cause == "CONTAMINATION"
    assert semantics.next_step == "GENERATE_PLANS"
    assert semantics.evidence_sufficiency is not None
    assert semantics.sensor_fault_probability is not None
    assert semantics.ood_category == "UNSEEN_TOPOLOGY"


def test_output_not_in_runtime_enabled_outputs_is_suppressed_to_none() -> None:
    """A v4 identity that has NOT validated event_cause/next_step/
    evidence_sufficiency/sensor_fault must produce None for every one of
    them, even though the raw model output contains confident values for
    all of them -- Phase 14's own governance invariant, "unvalidated heads
    must contribute exactly zero to runtime decisions", enforced here at
    the SemanticPredictions boundary."""

    result = _analyze(EventAwareModel(), runtime_enabled_outputs=frozenset({"event_presence"}))
    semantics = result.semantic_predictions
    assert semantics.event_presence is True
    assert semantics.event_presence_probability is not None
    assert semantics.event_cause is None
    assert semantics.next_step is None
    assert semantics.evidence_sufficiency is None
    assert semantics.sensor_fault_probability is None
    assert semantics.ood_category is None


def test_empty_runtime_enabled_outputs_suppresses_everything_gated() -> None:
    result = _analyze(EventAwareModel(), runtime_enabled_outputs=frozenset())
    semantics = result.semantic_predictions
    assert semantics.event_presence is None
    assert semantics.event_presence_probability is None
    assert semantics.event_cause is None
    assert semantics.next_step is None
    assert semantics.evidence_sufficiency is None
    assert semantics.sensor_fault_probability is None
    assert semantics.ood_category is None


def test_unsupported_ood_category_class_never_surfaces_even_when_enabled() -> None:
    """4 of the 11 governed OOD categories have no real, reproducible
    training examples yet (ood_labels.UNSUPPORTED_OOD_CATEGORIES) -- even a
    v4 identity that validated ood_category overall must never surface one
    of them as a live prediction (same convention as event_cause's
    AMBIGUOUS/HYDRAULIC_MISMATCH, core-issues5.txt Section 18.1)."""

    class UnseenSensorLayoutModel(EventAwareModel):
        def __call__(self, batch):
            output = super().__call__(batch)
            # index 2 == UNSEEN_SENSOR_LAYOUT, an unsupported category.
            output["ood_category_logits"] = torch.tensor(
                [[-8.0, -8.0, 8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0]]
            )
            return output

    result = _analyze(UnseenSensorLayoutModel(), runtime_enabled_outputs=frozenset({"ood_category"}))
    assert result.semantic_predictions.ood_category is None
    # This test's own deterministic OOD severity must remain untouched by
    # the (suppressed) learned category prediction -- the two are
    # independent authority tiers, never coupled.
    assert result.ood_level is not None


def test_unsupported_event_cause_class_never_surfaces_even_when_enabled() -> None:
    """AMBIGUOUS/HYDRAULIC_MISMATCH have zero real training examples
    (reports/results/v4/phase13-metrics-and-baselines.md) -- even a v4
    identity that validated event_cause overall must never surface either
    class as a live prediction (core-issues3.txt Phase 6.5/9.3)."""

    class HydraulicMismatchModel(EventAwareModel):
        def __call__(self, batch):
            output = super().__call__(batch)
            # index 2 == HYDRAULIC_MISMATCH (EventCause enum declaration
            # order: CONTAMINATION, SENSOR_FAULT, HYDRAULIC_MISMATCH,
            # AMBIGUOUS, NORMAL).
            output["event_cause_logits"] = torch.tensor([[-8.0, -8.0, 8.0, -8.0, -8.0]])
            return output

    result = _analyze(HydraulicMismatchModel(), runtime_enabled_outputs=frozenset({"event_cause"}))
    assert result.semantic_predictions.event_cause is None


def test_inspect_faulty_sensor_next_step_never_surfaces_even_when_enabled() -> None:
    """NextStep.INSPECT_FAULTY_SENSOR is a real, reproducible training
    label but is not runtime-enabled for the agent-FSM controller (no
    matching FSM state -- see hydroswarm.training.control_labels.
    NEXT_STEP_RUNTIME_ENABLED's own docstring)."""

    class InspectSensorModel(EventAwareModel):
        def __call__(self, batch):
            output = super().__call__(batch)
            # index 1 == INSPECT_FAULTY_SENSOR (NextStep enum declaration
            # order: COLLECT_SAMPLE, INSPECT_FAULTY_SENSOR, GENERATE_PLANS,
            # ABSTAIN).
            output["next_step_logits"] = torch.tensor([[-8.0, 8.0, -8.0, -8.0]])
            return output

    result = _analyze(InspectSensorModel(), runtime_enabled_outputs=frozenset({"next_step"}))
    assert result.semantic_predictions.next_step is None


def test_low_confidence_event_presence_is_false_not_none() -> None:
    """event_presence is a boolean decision (>= 0.5 sigmoid threshold),
    distinct from "no trustworthy value available" -- a validated head
    predicting low probability must surface False, not None."""

    class NoEventModel(EventAwareModel):
        def __call__(self, batch):
            output = super().__call__(batch)
            output["event_presence_logits"] = torch.tensor([[-4.0]])
            return output

    result = _analyze(NoEventModel(), runtime_enabled_outputs=frozenset({"event_presence"}))
    semantics = result.semantic_predictions
    assert semantics.event_presence is False
    assert semantics.event_presence_probability is not None
    assert semantics.event_presence_probability < 0.5


def test_runtime_enabled_outputs_defaults_to_none_when_unset() -> None:
    """HybridInferencePipeline's constructor default must not silently
    gate every caller that predates this parameter."""

    pipeline, _network = _pipeline(EventAwareModel())
    assert pipeline.runtime_enabled_outputs is None
    assert isinstance(pipeline, HybridInferencePipeline)


# core-issues5.txt Section 11 (P0 governance fix): an output absent from
# runtime_enabled_outputs must be incapable of changing any operationally
# relevant final result -- proven here by mutating the disabled head's
# raw logits to EXTREME values and asserting the result is identical to a
# moderate/default run, not merely "look plausible."


class ExtremeScoutModel(PriorFollowingModel):
    """expected_information_gain pinned to an extreme, adversarially large
    value at a specific node -- if this ever leaked into a governed
    decision, it would dominate any real classical signal."""

    def __call__(self, batch):
        output = super().__call__(batch)
        nodes = output["expected_information_gain"].shape[1]
        output["expected_information_gain"] = torch.full((1, nodes), 1_000.0)
        return output


def test_disabled_scout_information_gain_never_reaches_semantic_predictions() -> None:
    moderate = _analyze(PriorFollowingModel(), runtime_enabled_outputs=frozenset())
    extreme = _analyze(ExtremeScoutModel(), runtime_enabled_outputs=frozenset())

    assert moderate.semantic_predictions.expected_information_gain is None
    assert extreme.semantic_predictions.expected_information_gain is None


def test_enabled_scout_information_gain_does_reach_semantic_predictions() -> None:
    """Sanity check for the test above: when actually governed as
    runtime-enabled, the same field IS populated -- proves the None result
    above is real suppression, not an unrelated code path that always
    returns None."""

    result = _analyze(PriorFollowingModel(), runtime_enabled_outputs=frozenset({"information_gain"}))
    assert result.semantic_predictions.expected_information_gain is not None


class ExtremeStrategistModel(PriorFollowingModel):
    """plan_value/plan_validity pinned to extreme, maximally-confident
    values -- if this ever leaked past governance, it would dominate
    generate_response_plans' deterministic baseline scores."""

    def __call__(self, batch):
        output = super().__call__(batch)
        output["plan_value"] = torch.full_like(output["plan_value"], 1_000.0)
        output["plan_validity_logits"] = torch.stack(
            [
                torch.full(output["plan_value"].shape, -1_000.0),
                torch.full(output["plan_value"].shape, 1_000.0),
            ],
            dim=-1,
        )
        return output


def test_disabled_strategist_plan_scores_never_reach_semantic_predictions() -> None:
    moderate = _analyze(PriorFollowingModel(), runtime_enabled_outputs=frozenset())
    extreme = _analyze(ExtremeStrategistModel(), runtime_enabled_outputs=frozenset())

    assert moderate.semantic_predictions.plan_values == ()
    assert moderate.semantic_predictions.plan_validity == ()
    assert extreme.semantic_predictions.plan_values == ()
    assert extreme.semantic_predictions.plan_validity == ()


def test_enabled_strategist_plan_scores_do_reach_semantic_predictions() -> None:
    result = _analyze(
        PriorFollowingModel(), runtime_enabled_outputs=frozenset({"plan_value", "plan_validity"})
    )
    assert result.semantic_predictions.plan_values != ()
    assert result.semantic_predictions.plan_validity != ()
