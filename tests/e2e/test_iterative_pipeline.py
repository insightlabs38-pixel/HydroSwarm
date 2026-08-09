from __future__ import annotations

import pytest

from uuid import uuid4

import torch

from hydroswarm.classical import (
    SignatureArtifact,
    SignatureCacheKey,
    SignatureLibrary,
    SourceHypothesis,
)
from hydroswarm.calibration import CalibrationArtifact, CalibrationReport
from hydroswarm.inference import HybridInferencePipeline
from hydroswarm.inference import ControlAction
from hydroswarm.preprocessing import DEFAULT_FEATURE_SCHEMA, SensorSeries
from hydroswarm.simulation import HydraulicSimulator, build_wntr_network


pytestmark = pytest.mark.real_simulation


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
            "expected_information_gain": torch.full((1, nodes), 0.05),
            "ood_logits": torch.tensor([[4.0, 0.0, -4.0]]),
        }


def _pipeline(model):
    network = build_wntr_network()
    network.options.time.duration = 3600
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
    artifact = SignatureArtifact(
        key=SignatureCacheKey("a" * 64, "b" * 64, "test", "c" * 64, "d" * 64),
        library=library,
        hypotheses=hypotheses,
        sensor_nodes=("J1", "J2", "J3"),
        sample_times_seconds=(0, 3600),
        cache_hit=True,
        artifact_hash="e" * 64,
    )
    calibration = CalibrationArtifact(
        schema_version="hydroswarm-calibration-v1", alpha=0.1,
        model_hash=HybridInferencePipeline._fingerprint_model(model),
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint, dataset_manifest_hash="f" * 64,
        global_scores=(0.8, 0.8, 0.8, 0.8), mondrian_scores={}, network_scores={},
        report=CalibrationReport(1.0, 1.0, 0.0, {}, {}, 4),
    )
    return HybridInferencePipeline(
        simulator=HydraulicSimulator(network), signature_artifact=artifact, model=model,
        calibration_artifact=calibration,
        maximum_planning_candidates=1,
    ), network


def test_new_sample_triggers_complete_reanalysis_and_candidate_contraction() -> None:
    pipeline, network = _pipeline(PriorFollowingModel())
    incident_id = uuid4()
    before = pipeline.analyze(incident_id, network, [_series("J1", 0.78)], noise_scale=0.30)

    assert before.control_action == ControlAction.REQUEST_SAMPLE
    assert before.sample_result is not None
    assert before.sample_result.recommended_node in {"J2", "J3"}
    assert not before.planning_allowed

    after = pipeline.reanalyze_after_sample(
        before,
        network,
        [_series("J1", 0.78), _series("J2", 1.0)],
        noise_scale=0.30,
    )
    assert len(after.posterior_history) == 2
    assert len(after.evidence_history) == 2
    assert len(after.comparison_history) == 1
    assert after.before_after is not None
    assert after.before_after.candidate_contraction > 0
    assert after.before_after.removed_candidates
    assert after.fused_belief["J2"] > before.fused_belief["J2"]
    assert after.planning_allowed
    assert after.control_action == ControlAction.GENERATE_PLANS
    assert len(after.plan_proposals) >= 3

    # Identical evidence is idempotent and does not append another posterior round.
    repeated = pipeline.reanalyze_after_sample(
        after,
        network,
        [_series("J1", 0.78), _series("J2", 1.0)],
        noise_scale=0.30,
    )
    assert repeated is after
    assert len(repeated.posterior_history) == 2
