"""M10.2 preflight: schema, masking, leakage, and governance regression
tests for `hydroswarm.evaluation.scout_state`/`scout_readiness`.

Frozen correction document:
docs/evaluation/HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from hydroswarm.evaluation.live_robustness import locked_test_opened
from hydroswarm.evaluation.scout_readiness import (
    M10_2_PREFLIGHT_BLOCKED,
    M10_2_READY_FOR_SCIENTIFIC_EVALUATION,
    M9_6_REQUIRED_SCOUT_TARGET_KEYS,
    M9_6_SCOUT_HEAD_AUDIT,
    ScoutHeadTrainingAudit,
    m10_2_readiness,
)
from hydroswarm.evaluation.scout_state import (
    SCOUT_EVAL_STATE_SCHEMA_VERSION,
    ScoutEvaluationState,
    ScoutStateLeakageError,
    apply_scout_candidate_mask,
    assert_finite_scout_outputs,
    assert_no_target_only_keys,
    build_scout_evaluation_state,
    decode_learned_scout_recommendation,
    select_candidate_node,
)
from hydroswarm.inference.authority import scout_certificate
from hydroswarm.model import HydroCore
from hydroswarm.training import checkpoint_identity
from hydroswarm.training.losses import compute_multitask_loss
from hydroswarm.training.output_governance import (
    OutputGovernanceError,
    SCOUT_OUTPUTS,
    validate_output_governance,
)

ROOT = Path(__file__).resolve().parents[2]


def _tiny_scout_model() -> HydroCore:
    return HydroCore(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        adapter_dims=(32, 32, 32),
        dropout=0.0,
        scout_control_heads=True,
    )


def _tiny_batch(nodes: int = 4, batch: int = 2, *, node_mask: torch.Tensor | None = None) -> dict:
    generator = torch.Generator().manual_seed(31)
    out = {
        "node_features": torch.randn(batch, nodes, 3, generator=generator),
        "temporal_features": torch.randn(batch, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(batch, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(batch, nodes, dtype=torch.bool),
    }
    if node_mask is not None:
        out["node_mask"] = node_mask
    return out


# --------------------------------------------------------------------------
# 1. Schema identity / versioning.
# --------------------------------------------------------------------------


def test_eval_schema_version_is_distinct_from_the_training_corpus_placeholder() -> None:
    assert SCOUT_EVAL_STATE_SCHEMA_VERSION == "scout-eval-state-v1"
    assert SCOUT_EVAL_STATE_SCHEMA_VERSION != checkpoint_identity.SCOUT_STATE_SCHEMA_VERSION
    assert checkpoint_identity.SCOUT_STATE_SCHEMA_VERSION == "scout-state-v1-unbuilt"


# --------------------------------------------------------------------------
# 2. Basic construction / shape validation.
# --------------------------------------------------------------------------


def test_build_scout_evaluation_state_basic_shapes() -> None:
    node_ids = [("J1", "J2", "J3", "J4")]
    batch = _tiny_batch(nodes=4, batch=1)
    state = build_scout_evaluation_state(
        node_ids=node_ids,
        batch=batch,
        already_sampled=[["J1"]],
        sampling_round=[1],
        sample_budget_total=[5],
    )
    assert state.batch_size == 1
    assert state.nodes == 4
    assert state.already_sampled_mask.tolist() == [[True, False, False, False]]
    assert state.accessible_mask.tolist() == [[True, True, True, True]]
    assert state.sampling_round.tolist() == [1]
    assert state.sample_budget_remaining.tolist() == [4]


def test_mismatched_node_ids_batch_length_raises() -> None:
    with pytest.raises(ValueError, match="node_ids"):
        build_scout_evaluation_state(
            node_ids=[("J1", "J2")],
            batch=_tiny_batch(nodes=2, batch=2),
            already_sampled=[[], []],
            sampling_round=[0, 0],
            sample_budget_total=[5, 5],
        )


def test_state_post_init_rejects_wrong_mask_dtype() -> None:
    batch = _tiny_batch(nodes=2, batch=1)
    with pytest.raises(ValueError, match="boolean"):
        ScoutEvaluationState(
            node_ids=(("J1", "J2"),),
            batch=batch,
            already_sampled_mask=torch.zeros(1, 2, dtype=torch.float32),
            accessible_mask=torch.ones(1, 2, dtype=torch.bool),
            sampling_round=torch.tensor([0]),
            sample_budget_remaining=torch.tensor([5]),
        )


# --------------------------------------------------------------------------
# 3/6. Candidate eligibility / stable node-index mapping.
# --------------------------------------------------------------------------


def test_candidate_mask_excludes_already_sampled_and_inaccessible_and_padding() -> None:
    node_ids = [("J1", "J2", "J3", "J4")]
    node_mask = torch.tensor([[True, True, True, False]])  # J4 is padding
    batch = _tiny_batch(nodes=4, batch=1, node_mask=node_mask)
    state = build_scout_evaluation_state(
        node_ids=node_ids,
        batch=batch,
        already_sampled=[["J1"]],
        accessible=[["J1", "J2"]],  # J3 marked inaccessible
        sampling_round=[1],
        sample_budget_total=[5],
    )
    # eligible: only J2 (J1 sampled, J3 inaccessible, J4 padding)
    assert state.candidate_mask().tolist() == [[False, True, False, False]]


def test_node_index_maps_deterministically_to_the_same_physical_node_as_batch_tensors() -> None:
    """core requirement: sample_node_logits[b, i] must refer to the same
    physical node as node_ids[b][i] and batch['node_features'][b, i]."""

    node_ids = [("A", "B", "C")]
    batch = _tiny_batch(nodes=3, batch=1)
    state = build_scout_evaluation_state(
        node_ids=node_ids, batch=batch, already_sampled=[[]], sampling_round=[0], sample_budget_total=[3]
    )
    model = _tiny_scout_model().eval()
    with torch.no_grad():
        output = model(state.batch)
    # Force position 1 ("B") to be the unique maximum raw logit.
    logits = output["sample_node_logits"].clone()
    logits[0, 1] = logits.max() + 100.0
    output = dict(output)
    output["sample_node_logits"] = logits
    recommendation = decode_learned_scout_recommendation(output, state)
    assert recommendation.node_index == 1
    assert recommendation.node_id == "B"


# --------------------------------------------------------------------------
# 4. Masking helper -- adversarial: an enormous invalid logit must never win.
# --------------------------------------------------------------------------


def test_apply_scout_candidate_mask_blocks_an_enormous_invalid_logit_from_winning_argmax() -> None:
    node_ids = [("J1", "J2", "J3")]
    batch = _tiny_batch(nodes=3, batch=1)
    state = build_scout_evaluation_state(
        node_ids=node_ids,
        batch=batch,
        already_sampled=[["J2"]],  # J2 ineligible
        sampling_round=[1],
        sample_budget_total=[5],
    )
    output = {
        "sample_node_logits": torch.tensor([[0.0, 1e30, 0.5]]),  # J2's raw logit is enormous
        "expected_information_gain": torch.tensor([[0.1, 1e30, 0.2]]),
        "candidate_reduction_prediction": torch.tensor([[0.1, 1e30, 0.2]]),
    }
    mask = state.candidate_mask()
    masked = apply_scout_candidate_mask(output, mask)
    indices, ids = select_candidate_node(masked["sample_node_logits"], mask, state.node_ids)
    assert ids[0] == "J3"
    assert indices[0] == 2
    assert masked["expected_information_gain"][0, 1].item() == 0.0
    assert masked["candidate_reduction_prediction"][0, 1].item() == 0.0


def test_select_candidate_node_returns_none_when_nothing_is_eligible() -> None:
    node_ids = [("J1", "J2")]
    mask = torch.tensor([[False, False]])
    logits = torch.tensor([[5.0, 9.0]])
    indices, ids = select_candidate_node(logits, mask, node_ids)
    assert indices == [None]
    assert ids == [None]


def test_apply_scout_candidate_mask_requires_sample_node_logits_present() -> None:
    with pytest.raises(KeyError):
        apply_scout_candidate_mask({}, torch.ones(1, 2, dtype=torch.bool))


# --------------------------------------------------------------------------
# 5. Already-sampled nodes can never be selected (real model forward pass).
# --------------------------------------------------------------------------


def test_already_sampled_node_is_never_selected_through_a_real_forward_pass() -> None:
    torch.manual_seed(7)
    node_ids = [("J1", "J2", "J3", "J4")]
    batch = _tiny_batch(nodes=4, batch=1)
    model = _tiny_scout_model().eval()
    with torch.no_grad():
        output = model(batch)
    for held_out in ("J1", "J2", "J3", "J4"):
        state = build_scout_evaluation_state(
            node_ids=node_ids, batch=batch, already_sampled=[[held_out]], sampling_round=[1], sample_budget_total=[5]
        )
        recommendation = decode_learned_scout_recommendation(output, state)
        assert recommendation.node_id != held_out


# --------------------------------------------------------------------------
# 7. Multi-topology / batch-size > 1 shape handling.
# --------------------------------------------------------------------------


def test_batch_size_greater_than_one_with_heterogeneous_node_mask() -> None:
    node_ids = [("J1", "J2", "J3"), ("K1", "K2", "K3")]
    node_mask = torch.tensor([[True, True, True], [True, True, False]])  # item 1's K3 is padding
    batch = _tiny_batch(nodes=3, batch=2, node_mask=node_mask)
    state = build_scout_evaluation_state(
        node_ids=node_ids,
        batch=batch,
        already_sampled=[[], ["K1"]],
        sampling_round=[0, 1],
        sample_budget_total=[5, 5],
    )
    model = _tiny_scout_model().eval()
    with torch.no_grad():
        output = model(state.batch)
    assert_finite_scout_outputs(output)
    for batch_index in (0, 1):
        recommendation = decode_learned_scout_recommendation(output, state, batch_index=batch_index)
        assert recommendation.node_id in state.node_ids[batch_index]
    # item 1's padding node (K3) and already-sampled node (K1) must never be picked.
    second = decode_learned_scout_recommendation(output, state, batch_index=1)
    assert second.node_id == "K2"


# --------------------------------------------------------------------------
# 8. Future-evidence leakage: content outside this schema's declared inputs
#    cannot change the decision.
# --------------------------------------------------------------------------


def test_extraneous_batch_keys_do_not_affect_model_output_or_recommendation() -> None:
    node_ids = [("J1", "J2", "J3")]
    batch = _tiny_batch(nodes=3, batch=1)
    state_clean = build_scout_evaluation_state(
        node_ids=node_ids, batch=batch, already_sampled=[[]], sampling_round=[0], sample_budget_total=[3]
    )
    leaky_batch = dict(batch)
    # Simulate a careless caller appending an unrelated, unknown "future
    # observation" style key HydroCore.forward()/this schema never declares
    # or reads -- must have zero effect.
    leaky_batch["future_debug_observations"] = torch.full((1, 3), 999.0)  # type: ignore[typeddict-item]
    state_leaky = build_scout_evaluation_state(
        node_ids=node_ids, batch=leaky_batch, already_sampled=[[]], sampling_round=[0], sample_budget_total=[3]
    )
    model = _tiny_scout_model().eval()
    with torch.no_grad():
        output_clean = model(state_clean.batch)
        output_leaky = model(state_leaky.batch)
    torch.testing.assert_close(output_clean["sample_node_logits"], output_leaky["sample_node_logits"])
    rec_clean = decode_learned_scout_recommendation(output_clean, state_clean)
    rec_leaky = decode_learned_scout_recommendation(output_leaky, state_leaky)
    assert rec_clean.node_id == rec_leaky.node_id


# --------------------------------------------------------------------------
# 9. Target-label leakage: ground-truth/offline-scoring fields must never
#    reach model input.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leaked_key", ["sample_node", "information_gain", "candidate_reduction", "should_continue_sampling"]
)
def test_target_only_scout_keys_in_batch_raise_before_reaching_the_model(leaked_key: str) -> None:
    batch = _tiny_batch(nodes=3, batch=1)
    batch[leaked_key] = torch.tensor([0])  # type: ignore[literal-required]
    with pytest.raises(ScoutStateLeakageError):
        assert_no_target_only_keys(batch)
    with pytest.raises(ScoutStateLeakageError):
        build_scout_evaluation_state(
            node_ids=[("J1", "J2", "J3")],
            batch=batch,
            already_sampled=[[]],
            sampling_round=[0],
            sample_budget_total=[3],
        )


def test_travel_time_is_a_legitimate_input_field_not_blocked_as_leakage() -> None:
    """travel_time collides by name between targets_v2's governed target
    vocabulary and HydroBatch's own real deterministic input feature -- must
    NOT be treated as a leaked label (see scout_state.py's _TARGET_ONLY_KEYS
    docstring)."""

    batch = _tiny_batch(nodes=3, batch=1)
    batch["travel_time"] = torch.zeros(1, 3)  # type: ignore[typeddict-item]
    assert_no_target_only_keys(batch)  # must not raise
    build_scout_evaluation_state(
        node_ids=[("J1", "J2", "J3")], batch=batch, already_sampled=[[]], sampling_round=[0], sample_budget_total=[3]
    )  # must not raise


def test_target_only_key_set_excludes_every_real_hydrobatch_field() -> None:
    from hydroswarm.model.core import HydroBatch

    from hydroswarm.evaluation.scout_state import _TARGET_ONLY_KEYS  # type: ignore[attr-defined]

    assert _TARGET_ONLY_KEYS.isdisjoint(set(HydroBatch.__annotations__))
    assert "travel_time" not in _TARGET_ONLY_KEYS
    assert "sample_node" in _TARGET_ONLY_KEYS


def test_mutating_realized_ground_truth_labels_does_not_change_a_state_built_without_them() -> None:
    """Holding current revealed state fixed (the same batch/already_sampled/
    round/budget), mutating an offline realized-outcome label value (as if
    computed by hydroswarm.training.scout_labels.generate_scout_label for
    scoring purposes) must not change the resulting ScoutEvaluationState at
    all, since the label is never passed to the builder in the first
    place."""

    node_ids = [("J1", "J2", "J3")]
    batch = _tiny_batch(nodes=3, batch=1)
    state_a = build_scout_evaluation_state(
        node_ids=node_ids, batch=batch, already_sampled=[["J1"]], sampling_round=[1], sample_budget_total=[3]
    )
    # A wildly different "realized outcome" (e.g. information_gain_bits=99.0,
    # a different eventual source) computed OFFLINE never enters the builder
    # at all -- there is no parameter to pass it through even adversarially.
    state_b = build_scout_evaluation_state(
        node_ids=node_ids, batch=batch, already_sampled=[["J1"]], sampling_round=[1], sample_budget_total=[3]
    )
    assert torch.equal(state_a.already_sampled_mask, state_b.already_sampled_mask)
    assert torch.equal(state_a.candidate_mask(), state_b.candidate_mask())


# --------------------------------------------------------------------------
# 10. Finite outputs.
# --------------------------------------------------------------------------


def test_assert_finite_scout_outputs_passes_on_a_real_forward_pass() -> None:
    model = _tiny_scout_model().eval()
    with torch.no_grad():
        output = model(_tiny_batch())
    assert_finite_scout_outputs(output)  # must not raise


def test_assert_finite_scout_outputs_fails_closed_on_nan() -> None:
    output = {"sample_node_logits": torch.tensor([[0.0, float("nan")]])}
    with pytest.raises(FloatingPointError):
        assert_finite_scout_outputs(output)


# --------------------------------------------------------------------------
# 11. Checkpoint/output-governance consistency.
# --------------------------------------------------------------------------


def test_scout_outputs_absent_from_trained_outputs_is_governance_valid() -> None:
    """The M10.2 audit's own finding (no Scout task ever trained) is itself
    a governance-valid state: nothing requires every canonical output to be
    trained."""

    validate_output_governance(
        trained_outputs=frozenset(),
        validated_outputs=frozenset(),
        runtime_enabled_outputs=frozenset(),
        diagnostic_only_outputs=frozenset(),
        training_only_outputs=frozenset(),
    )  # must not raise


def test_claiming_an_untrained_scout_output_as_runtime_enabled_fails_closed() -> None:
    one_output = next(iter(SCOUT_OUTPUTS))
    with pytest.raises(OutputGovernanceError):
        validate_output_governance(
            trained_outputs=frozenset(),
            validated_outputs=frozenset(),
            runtime_enabled_outputs=frozenset({one_output}),
            diagnostic_only_outputs=frozenset(),
            training_only_outputs=frozenset(),
        )


def test_m9_6_scout_head_audit_matches_required_target_keys() -> None:
    assert M9_6_SCOUT_HEAD_AUDIT.scout_heads_present is True
    assert M9_6_SCOUT_HEAD_AUDIT.scout_heads_trained is False
    assert set(M9_6_SCOUT_HEAD_AUDIT.missing_scout_target_keys) == M9_6_REQUIRED_SCOUT_TARGET_KEYS
    assert set(M9_6_SCOUT_HEAD_AUDIT.required_scout_target_keys) == set(SCOUT_OUTPUTS)


def test_readiness_is_blocked_for_the_real_m9_6_audit() -> None:
    assert m10_2_readiness() == M10_2_PREFLIGHT_BLOCKED


def test_readiness_would_be_ready_if_heads_were_actually_trained() -> None:
    hypothetical = ScoutHeadTrainingAudit(
        checkpoint_label="hypothetical",
        scout_heads_present=True,
        scout_heads_trained=True,
        required_scout_target_keys=(),
        observed_corpus_target_keys=(),
        missing_scout_target_keys=(),
        finding="hypothetical",
    )
    assert m10_2_readiness(hypothetical) == M10_2_READY_FOR_SCIENTIFIC_EVALUATION


# --------------------------------------------------------------------------
# 6/7 (deterministic fallback + no authority bypass).
# --------------------------------------------------------------------------


def test_scout_certificate_signature_has_no_learned_recommendation_input() -> None:
    """Structural, not merely behavioral, guarantee: scout_certificate
    cannot even be CALLED with a learned recommendation -- it accepts only
    an IncidentAnalysisResult, so nothing in hydroswarm.evaluation.scout_state
    can be wired into it to bypass deterministic authority."""

    parameters = inspect.signature(scout_certificate).parameters
    assert set(parameters) == {"analysis"}


def test_decode_learned_scout_recommendation_is_never_promotable() -> None:
    node_ids = [("J1", "J2")]
    batch = _tiny_batch(nodes=2, batch=1)
    state = build_scout_evaluation_state(
        node_ids=node_ids, batch=batch, already_sampled=[[]], sampling_round=[0], sample_budget_total=[3]
    )
    model = _tiny_scout_model().eval()
    with torch.no_grad():
        output = model(state.batch)
    recommendation = decode_learned_scout_recommendation(output, state)
    assert recommendation.promotable is False


def test_deterministic_scout_fallback_is_callable_without_any_model() -> None:
    from hydroswarm.agents.scout import HydroScout

    scout = HydroScout()
    state = {
        "candidate_region": ("J1", "J2", "J3"),
        "candidate_probabilities": {"J1": 0.5, "J2": 0.3, "J3": 0.2},
        "sampling_history": (),
        "node_ids": ("J1", "J2", "J3"),
    }
    output = scout.deterministic_fallback(state)
    assert output is not None


# --------------------------------------------------------------------------
# 13. locked_test_opened invariant.
# --------------------------------------------------------------------------


def test_locked_test_opened_remains_false() -> None:
    assert locked_test_opened(ROOT) is False


# --------------------------------------------------------------------------
# Loss-guard mechanism this preflight's finding depends on.
# --------------------------------------------------------------------------


def test_compute_multitask_loss_skips_a_task_absent_from_targets() -> None:
    """The exact mechanism the M10.2 checkpoint-governance audit cites: a
    task absent from `targets` contributes NOTHING (not even a zero-valued
    connected term) to compute_multitask_loss's result -- this is what let
    sample_node/information_gain/candidate_reduction/should_continue_sampling
    silently receive no gradient throughout M9.6 training."""

    model = _tiny_scout_model()
    output = model(_tiny_batch(nodes=4, batch=2))
    targets_without_any_scout_key = {
        "source_node": torch.tensor([0, 1]),
        "source_node_mask": torch.tensor([True, True]),
    }
    result = compute_multitask_loss(output, targets_without_any_scout_key)
    assert "sample_node" not in result.tasks
    assert "information_gain" not in result.tasks
    assert "candidate_reduction" not in result.tasks
    assert "should_continue_sampling" not in result.tasks
