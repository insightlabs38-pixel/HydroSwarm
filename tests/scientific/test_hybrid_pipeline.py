from __future__ import annotations

from uuid import uuid4

import pytest
import torch

from hydroswarm.classical import (
    SignatureArtifact,
    SignatureCacheKey,
    SignatureLibrary,
    SourceHypothesis,
)
from hydroswarm.calibration import CalibrationArtifact, CalibrationReport
from hydroswarm.inference import (
    HybridInferencePipeline,
    HybridRuntimeMode,
    OODDetector,
    OODReference,
)
from hydroswarm.model.core import HydroCore
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_INDEX
from hydroswarm.preprocessing import DEFAULT_FEATURE_SCHEMA, SensorSeries
from hydroswarm.simulation import HydraulicSimulator, build_wntr_network
from hydroswarm.simulation.wrapper import FEATURE_SNAPSHOT_TIME_SECONDS


def _artifact() -> SignatureArtifact:
    hypotheses = tuple(
        SourceHypothesis(node, 0, 60, 1.0, "nominal") for node in ("J1", "J2", "J3")
    )
    signatures = {
        "J1": [[0.0, 0.0, 0.0], [1.00, 0.15, 0.05]],
        "J2": [[0.0, 0.0, 0.0], [0.78, 1.00, 0.10]],
        "J3": [[0.0, 0.0, 0.0], [0.65, 0.20, 1.00]],
    }
    library = SignatureLibrary()
    for hypothesis in hypotheses:
        library.add(hypothesis.identifier, signatures[hypothesis.source_node])
    return SignatureArtifact(
        key=SignatureCacheKey("a" * 64, "b" * 64, "test", "c" * 64, "d" * 64),
        library=library,
        hypotheses=hypotheses,
        sensor_nodes=("J1", "J2", "J3"),
        sample_times_seconds=(0, 3600),
        cache_hit=True,
        artifact_hash="e" * 64,
    )


def _series(node: str, final: float) -> SensorSeries:
    return SensorSeries(
        node_id=node,
        timestamps_seconds=(0.0, 3600.0),
        concentration_mg_l=(0.0, final),
        pressure_m=(25.0, 24.0),
        health=(1.0, 1.0),
        missing=(False, False),
        drift=(False, False),
        delayed=(False, False),
    )


class PriorFollowingModel:
    def eval(self):
        return self

    def __call__(self, batch):
        prior = batch["classical_prior"].clamp_min(1e-7)
        nodes = prior.shape[1]
        return {
            "source_node_logits": prior.log(),
            "evidence_sufficiency": torch.tensor([[0.9]]),
            "uncertainty": torch.tensor([[0.1]]),
            "expected_information_gain": torch.full((1, nodes), 0.05),
            "sensor_fault_logits": torch.full((1, nodes), -4.0),
            "ood_logits": torch.tensor([[4.0, 0.0, -4.0]]),
            "plan_value": torch.zeros(1, 8),
            "plan_validity_logits": torch.tensor([[[0.0, 2.0]] * 8]),
        }


class CandidateAwareModel(PriorFollowingModel):
    """Same PASS-1 (incident-only) behavior as PriorFollowingModel, but
    also handles PASS-2 candidate-conditioned calls (plan tensors
    present) by returning a controlled plan_value/plan_validity keyed to
    plan_template_ids -- lets tests verify the resulting deltas are
    correctly attributed by ACTION_TEMPLATE identity through the full
    analyze() -> _score_candidate_plans -> generate_response_plans round
    trip, not merely "doesn't crash"."""

    def __call__(self, batch):
        if "plan_template_ids" not in batch:
            return super().__call__(batch)
        template_ids = batch["plan_template_ids"]
        plans = template_ids.shape[1]
        isolate_index = ACTION_TEMPLATE_INDEX["ISOLATE_SOURCE"]
        values = torch.where(template_ids == isolate_index, 10.0, -10.0)
        validity_logits = torch.stack(
            [torch.zeros(1, plans), torch.full((1, plans), 4.0)], dim=-1
        )
        return {"plan_value": values.float(), "plan_validity_logits": validity_logits}


def _pipeline(model) -> tuple[HybridInferencePipeline, object]:
    network = build_wntr_network()
    network.options.time.duration = 3600
    model_hash = HybridInferencePipeline._fingerprint_model(model)
    calibration = CalibrationArtifact(
        schema_version="hydroswarm-calibration-v1",
        alpha=0.1,
        model_hash=model_hash,
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        dataset_manifest_hash="f" * 64,
        global_scores=(0.8, 0.8, 0.8, 0.8),
        mondrian_scores={},
        network_scores={},
        report=CalibrationReport(1.0, 1.0, 0.0, {}, {}, 4),
    )
    pipeline = HybridInferencePipeline(
        simulator=HydraulicSimulator(network),
        signature_artifact=_artifact(),
        model=model,
        model_hash=model_hash,
        calibration_artifact=calibration,
        maximum_planning_candidates=1,
    )
    return pipeline, network


def test_hybrid_result_aligns_native_beliefs_and_records_provenance() -> None:
    pipeline, network = _pipeline(PriorFollowingModel())
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    assert result.node_alignment == tuple(sorted(network.node_name_list))
    assert set(result.classical_belief) == set(result.node_alignment)
    assert sum(result.classical_belief.values()) == pytest.approx(1.0)
    assert result.neural_belief is not None
    assert sum(result.fused_belief.values()) == pytest.approx(1.0)
    assert len(set(round(value, 6) for value in result.classical_belief.values())) > 2
    assert result.fusion_diagnostics is not None
    assert "classical_trust=" in result.trust_rationale
    assert result.runtime_mode == HybridRuntimeMode.FULL_HYBRID
    assert result.provenance_hashes["model"] == pipeline._model_hash
    assert all(len(value) == 64 for value in result.provenance_hashes.values() if value != "none")


def test_analyze_snapshots_hydraulic_state_at_the_feature_snapshot_time_not_the_last_step() -> None:
    # core-issues.txt Phase 3 item 18 discovery: calculate_state() with no
    # argument defaults to the network's LAST simulated timestamp, not the
    # FEATURE_SNAPSHOT_TIME_SECONDS snapshot hydroswarm.training.corpus.
    # build_feature_context (training) and hydroswarm.cli's fixed-inference
    # verification both use -- silently feeding the model a hydraulic state
    # far outside its training distribution for any simulation whose
    # duration differs from FEATURE_SNAPSHOT_TIME_SECONDS. Regression test:
    # a network whose duration is deliberately NOT
    # FEATURE_SNAPSHOT_TIME_SECONDS must still be snapshotted at that time.
    network = build_wntr_network()
    network.options.time.duration = FEATURE_SNAPSHOT_TIME_SECONDS * 4
    simulator = HydraulicSimulator(network)
    calls: list[int | None] = []
    original = HydraulicSimulator.calculate_state

    def spy(self, at_time=None):
        calls.append(at_time)
        return original(self, at_time)

    HydraulicSimulator.calculate_state = spy
    try:
        pipeline = HybridInferencePipeline(
            simulator=simulator,
            signature_artifact=_artifact(),
            model=PriorFollowingModel(),
            maximum_planning_candidates=1,
        )
        pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])
    finally:
        HydraulicSimulator.calculate_state = original

    assert calls == [FEATURE_SNAPSHOT_TIME_SECONDS]


def test_neural_failure_is_explicit_and_falls_back_to_classical_posterior() -> None:
    class BrokenModel:
        def __call__(self, batch):
            raise RuntimeError("checkpoint unavailable")

    pipeline, network = _pipeline(BrokenModel())
    result = pipeline.analyze(uuid4(), network, [_series("J1", 1.0)])

    assert result.runtime_mode == HybridRuntimeMode.CLASSICAL_SAFE
    assert result.neural_failure == "RuntimeError"
    assert result.neural_belief is None
    assert result.fused_belief == result.classical_belief
    assert result.fusion_diagnostics is None


def test_calibrated_by_default_when_neither_side_declares_a_fusion_config() -> None:
    # core-issues.txt repair item 10: every existing CalibrationArtifact
    # fixture (and this test's own _pipeline() helper) predates
    # fusion_config_hash -- both default to None, so the check must be
    # skipped rather than spuriously invalidating calibration.
    pipeline, network = _pipeline(PriorFollowingModel())
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])
    assert result.calibrated is True


def test_uncalibrated_when_pipeline_declares_a_fusion_config_the_artifact_never_recorded() -> None:
    pipeline, network = _pipeline(PriorFollowingModel())
    pipeline.fusion_config_hash = "fuse_source_probabilities-v1"
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])
    assert result.calibrated is False


def test_scout_strategist_and_ood_neural_outputs_are_ignored_when_untrained() -> None:
    # core-issues.txt repair item 8: PriorFollowingModel's
    # expected_information_gain/plan_value/plan_validity_logits/ood_logits
    # are exactly the outputs that must never influence a runtime decision
    # until a checkpoint declares those tasks trained -- today no checkpoint
    # does, because Scout/Strategist/OOD target generators do not exist.
    pipeline, network = _pipeline(PriorFollowingModel())
    pipeline.trained_tasks = frozenset({"sentinel"})
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    assert result.semantic_predictions.expected_information_gain is None
    assert result.semantic_predictions.plan_values == ()
    assert result.semantic_predictions.plan_validity == ()
    # ood_logits=[4.0, 0.0, -4.0] softmaxes its last class near zero, which
    # would otherwise pull ood_components.energy toward 0. Gated, energy
    # must instead fall back to the deterministic 1 - max(neural belief).
    assert result.neural_belief is not None
    expected_energy = 1.0 - max(result.neural_belief.values())
    assert result.ood_components.energy == pytest.approx(expected_energy, abs=1e-6)


def test_scout_strategist_and_ood_neural_outputs_pass_through_when_declared_trained() -> None:
    pipeline, network = _pipeline(PriorFollowingModel())
    pipeline.trained_tasks = frozenset({"sentinel", "scout", "strategist", "ood"})
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    assert result.semantic_predictions.expected_information_gain is not None
    assert result.semantic_predictions.plan_values != ()
    assert result.semantic_predictions.plan_validity != ()
    # With "ood" declared trained, ood_logits' near-zero last class must
    # actually drive down ood_components.energy rather than falling back.
    assert result.ood_components.energy < 0.05


def test_candidate_conditioned_model_does_not_fall_back_to_classical_safe_without_plan_tensors() -> None:
    """core-issues5.txt Section 2 (P0 blocker) integration regression.

    A real candidate-conditioned HydroCore forward pass over the actual
    live HydraulicFeatureBuilder-shaped batch (which never contains plan
    tensors -- planning has not happened at this point in the pipeline)
    must produce real Sentinel/localization outputs, not silently
    degrade the whole incident analysis to
    HybridRuntimeMode.CLASSICAL_SAFE. Before the fix, HydroCore.forward()
    unconditionally raised KeyError for a candidate_conditioned model
    given no plan tensors, and _run_model's broad except-Exception
    converted that architecture mismatch into an indistinguishable,
    silent classical-only fallback.
    """
    model = HydroCore(
        d_model=32, nhead=4, dim_feedforward=64, num_layers=1, modality_layers=1,
        latent_tokens=64, plan_queries=2, adapter_dims=(32, 32, 32),
        strategist_mode="candidate_conditioned",
    ).eval()
    pipeline, network = _pipeline(model)

    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    assert result.runtime_mode == HybridRuntimeMode.FULL_HYBRID
    assert result.neural_failure is None
    assert result.neural_belief is not None
    assert result.semantic_predictions.plan_values == ()
    assert result.semantic_predictions.plan_validity == ()


def test_disagreement_and_ood_fail_closed_before_planning() -> None:
    class ContrarianModel(PriorFollowingModel):
        def __call__(self, batch):
            output = super().__call__(batch)
            output["source_node_logits"] = torch.tensor([[-20.0, -20.0, -20.0, -20.0, -20.0, 20.0]])
            output["ood_logits"] = torch.tensor([[-4.0, -2.0, 5.0]])
            return output

    pipeline, network = _pipeline(ContrarianModel())
    pipeline.ood_detector = OODDetector(
        OODReference(caution_threshold=0.10, outside_threshold=0.15)
    )
    result = pipeline.analyze(uuid4(), network, [_series("J1", 1.0)])

    assert result.fusion_diagnostics is not None
    assert result.fusion_diagnostics.disagreement_js >= 0.5
    assert result.ood_level.value == "OUTSIDE_VALIDATED_RANGE"
    assert not result.planning_allowed
    assert not result.plan_proposals
    assert "HIGH_CLASSICAL_NEURAL_DISAGREEMENT" in result.planning_suppression_reasons
    assert any(reason.startswith("OOD_") for reason in result.planning_suppression_reasons)


# core-issues5.txt Section 6 (P0 blocker): live planning must use the real
# candidate-conditioned Strategist architecture, keyed by ACTION_TEMPLATE
# identity, never an anonymous positional-delta approximation.


def test_learned_prescreen_promotes_the_template_it_actually_scored_highest() -> None:
    """End-to-end: a controlled candidate-conditioned model that strongly
    favors ISOLATE_SOURCE by ACTION_TEMPLATE identity must cause
    ISOLATE_SOURCE to rank at the top of the final plan proposals --
    proof the real PASS-2 scores flow through generate_response_plans
    correctly, keyed by template name, not tensor position."""

    pipeline, network = _pipeline(CandidateAwareModel())
    pipeline.trained_tasks = frozenset({"sentinel", "scout", "strategist", "ood"})
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    assert result.plan_proposals
    templates = [proposal.template for proposal in result.plan_proposals]
    assert "ISOLATE_SOURCE" in templates
    by_value = sorted(result.plan_proposals, key=lambda proposal: -proposal.predicted_value)
    assert by_value[0].template == "ISOLATE_SOURCE"


def test_disabled_strategist_produces_deterministic_ordering_not_failure() -> None:
    """With "strategist" excluded from trained_tasks, PASS-2 scoring must
    be skipped entirely -- proposals must match the pure deterministic
    baseline (no neural influence), and analysis must not fail."""

    pipeline, network = _pipeline(CandidateAwareModel())
    pipeline.trained_tasks = frozenset({"sentinel"})
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    assert result.plan_proposals
    by_template = {proposal.template: proposal for proposal in result.plan_proposals}
    # generate_response_plans' own baseline value for ISOLATE_SOURCE
    # (response.py) is 0.65 -- unperturbed by any neural delta.
    if "ISOLATE_SOURCE" in by_template:
        assert by_template["ISOLATE_SOURCE"].predicted_value == pytest.approx(0.65)


def test_runtime_enabled_outputs_excluding_plan_value_and_validity_disables_scoring() -> None:
    """Granular v4 governance must be able to suppress Strategist PASS-2
    scoring even when the coarser trained_tasks role switch allows it --
    matches every checkpoint built so far (plan_value/plan_validity are
    validated but not runtime-enabled, Phase 14 gate 7 unmet)."""

    pipeline, network = _pipeline(CandidateAwareModel())
    pipeline.trained_tasks = frozenset({"sentinel", "scout", "strategist", "ood"})
    pipeline.runtime_enabled_outputs = frozenset({"source_node", "event_presence"})
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    by_template = {proposal.template: proposal for proposal in result.plan_proposals}
    if "ISOLATE_SOURCE" in by_template:
        assert by_template["ISOLATE_SOURCE"].predicted_value == pytest.approx(0.65)


def test_learned_prescreen_never_marks_a_plan_verified() -> None:
    """PASS-2 scoring only ever ranks/adjusts PlanProposal.predicted_value/
    predicted_validity -- it has no mechanism to mark anything VERIFIED;
    that remains exclusively WNTR's authority, applied downstream of this
    pipeline entirely."""

    pipeline, network = _pipeline(CandidateAwareModel())
    pipeline.trained_tasks = frozenset({"sentinel", "scout", "strategist", "ood"})
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    for proposal in result.plan_proposals:
        assert not hasattr(proposal, "decision")
        assert not hasattr(proposal.plan, "decision")


def test_a_broken_pass_two_model_falls_back_to_deterministic_ordering() -> None:
    """A real PASS-2 failure (here: a model that raises on the plan-tensor
    batch) must fall back to deterministic ordering, never crash the
    whole incident analysis."""

    class BrokenPassTwoModel(PriorFollowingModel):
        def __call__(self, batch):
            if "plan_template_ids" in batch:
                raise RuntimeError("pass-2 broken")
            return super().__call__(batch)

    pipeline, network = _pipeline(BrokenPassTwoModel())
    pipeline.trained_tasks = frozenset({"sentinel", "scout", "strategist", "ood"})
    result = pipeline.analyze(uuid4(), network, [_series("J1", 0.78)])

    assert result.plan_proposals
    by_template = {proposal.template: proposal for proposal in result.plan_proposals}
    if "ISOLATE_SOURCE" in by_template:
        assert by_template["ISOLATE_SOURCE"].predicted_value == pytest.approx(0.65)
