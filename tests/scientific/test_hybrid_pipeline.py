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
from hydroswarm.preprocessing import DEFAULT_FEATURE_SCHEMA, SensorSeries
from hydroswarm.simulation import HydraulicSimulator, build_wntr_network


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
