"""core-issues5.txt Section 12 (P0 governance fix): a truthful V4 inference
release bundle -- separate from a resumable training checkpoint, real
content hashes throughout, no fabricated resumability metadata.

Builds a self-contained tiny v4 checkpoint in tmp_path (mirrors
tests/integration/test_v4_pipeline_factory.py's own
_tiny_model/_identity_for/_save_v4_checkpoint pattern) rather than
depending on any real trained checkpoint under the gitignored
experiments/runs/ tree.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from hydroswarm.model.core import HydroCore
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats
from hydroswarm.training import checkpoint_identity as ci

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_v4_inference_release_bundle as bundle_script  # noqa: E402

_TINY_KWARGS = dict(
    d_model=32,
    nhead=2,
    dim_feedforward=64,
    num_layers=1,
    modality_layers=1,
    latent_tokens=64,
    plan_queries=1,
    action_vocabulary_size=ACTION_TEMPLATE_COUNT,
)


def _tiny_model() -> HydroCore:
    model = HydroCore(**_TINY_KWARGS)  # type: ignore[arg-type]
    model.variant_name = "test-tiny"
    return model


def _write_normalization_dir(directory: Path) -> str:
    """Writes a real, fitted node/edge NormalizationStats pair and returns
    the combined fingerprint (HydraulicFeatureBuilder.normalization_
    fingerprint's own formula) callers use as the checkpoint identity's
    normalization_hash."""

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    node_stats = NormalizationStats.fit(
        rng.normal(size=(20, len(DEFAULT_FEATURE_SCHEMA.node_features))),
        DEFAULT_FEATURE_SCHEMA.node_features,
    )
    edge_stats = NormalizationStats.fit(
        rng.normal(size=(10, len(DEFAULT_FEATURE_SCHEMA.edge_features))),
        DEFAULT_FEATURE_SCHEMA.edge_features,
    )
    node_stats.save(directory / "node-normalization.json")
    edge_stats.save(directory / "edge-normalization.json")
    from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder

    return HydraulicFeatureBuilder(
        node_normalization=node_stats, edge_normalization=edge_stats
    ).normalization_fingerprint


def _save_checkpoint(directory: Path, *, normalization_hash: str) -> ci.CheckpointIdentity:
    model = _tiny_model()
    identity = ci.build_checkpoint_identity(
        model,
        normalization_hash=normalization_hash,
        fusion_policy_hash="fixed-weight-v1:neural=0.5",
        source_corpus_manifest_hashes=("abc123",),
        trained_outputs=frozenset({"source_node", "event_presence", "sensor_reconstruction", "travel_time"}),
        validated_outputs=frozenset({"source_node", "event_presence"}),
        runtime_enabled_outputs=frozenset({"event_presence"}),
        training_only_outputs=frozenset({"sensor_reconstruction", "travel_time"}),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    ci.save_v4_checkpoint(
        directory,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=1,
        best_validation_loss=1.0,
        identity=identity,
        resolved_training_config={},
        dataset_manifest_hashes={"train": "abc123"},
        task_weights={"source_node": 1.0},
    )
    return identity


def test_bundle_contains_only_inference_files_no_resumable_state(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    normalization_dir = tmp_path / "normalization"
    normalization_fingerprint = _write_normalization_dir(normalization_dir)
    _save_checkpoint(checkpoint_dir, normalization_hash=normalization_fingerprint)

    output_dir = tmp_path / "bundle"
    bundle_script.build(
        checkpoint_dir=checkpoint_dir,
        normalization_dir=normalization_dir,
        output_dir=output_dir,
        workdir=Path(__file__).resolve().parents[2],
    )

    names = {p.name for p in output_dir.iterdir()}
    assert "model.safetensors" in names
    assert "checkpoint_identity.json" in names
    assert "node-normalization.json" in names
    assert "edge-normalization.json" in names
    assert "output_governance.json" in names
    assert "signature-policy-manifest.json" in names
    assert "runtime_manifest.json" in names
    assert "artifact-manifest.json" in names
    assert "SHA256SUMS" in names
    # No fabricated resumability metadata -- optimizer/scheduler/RNG state
    # from the source checkpoint directory must never be copied in.
    assert "optimizer_state.pt" not in names
    assert "rng_state.pt" not in names
    assert "trainer_state.json" not in names


def test_every_manifest_hash_is_a_real_content_hash(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    normalization_dir = tmp_path / "normalization"
    normalization_fingerprint = _write_normalization_dir(normalization_dir)
    _save_checkpoint(checkpoint_dir, normalization_hash=normalization_fingerprint)

    output_dir = tmp_path / "bundle"
    bundle_script.build(
        checkpoint_dir=checkpoint_dir,
        normalization_dir=normalization_dir,
        output_dir=output_dir,
        workdir=Path(__file__).resolve().parents[2],
    )

    manifest = json.loads((output_dir / "artifact-manifest.json").read_text(encoding="utf-8"))
    for name, recorded_hash in manifest["files"].items():
        actual_hash = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        assert recorded_hash == actual_hash, f"{name}: manifest hash does not match real file content"

    runtime_manifest = json.loads((output_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert runtime_manifest["model_sha256"] == hashlib.sha256(
        (output_dir / "model.safetensors").read_bytes()
    ).hexdigest()
    assert len(runtime_manifest["source_git_sha"]) == 40  # a real git SHA, not "unavailable"/"none"


def test_output_governance_matches_checkpoint_identity_verbatim(tmp_path: Path) -> None:
    """No re-deciding governance at bundle-build time -- the bundle must
    reflect exactly what the checkpoint identity already declared."""

    checkpoint_dir = tmp_path / "checkpoint"
    normalization_dir = tmp_path / "normalization"
    normalization_fingerprint = _write_normalization_dir(normalization_dir)
    identity = _save_checkpoint(checkpoint_dir, normalization_hash=normalization_fingerprint)

    output_dir = tmp_path / "bundle"
    bundle_script.build(
        checkpoint_dir=checkpoint_dir,
        normalization_dir=normalization_dir,
        output_dir=output_dir,
        workdir=Path(__file__).resolve().parents[2],
    )

    governance = json.loads((output_dir / "output_governance.json").read_text(encoding="utf-8"))
    assert set(governance["trained_outputs"]) == identity.trained_outputs
    assert set(governance["validated_outputs"]) == identity.validated_outputs
    assert set(governance["runtime_enabled_outputs"]) == identity.runtime_enabled_outputs
    assert set(governance["training_only_outputs"]) == identity.training_only_outputs


def test_calibration_is_honestly_recorded_as_not_yet_fit(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    normalization_dir = tmp_path / "normalization"
    normalization_fingerprint = _write_normalization_dir(normalization_dir)
    _save_checkpoint(checkpoint_dir, normalization_hash=normalization_fingerprint)

    output_dir = tmp_path / "bundle"
    bundle_script.build(
        checkpoint_dir=checkpoint_dir,
        normalization_dir=normalization_dir,
        output_dir=output_dir,
        workdir=Path(__file__).resolve().parents[2],
    )

    calibration_status = json.loads((output_dir / "calibration-status.json").read_text(encoding="utf-8"))
    assert calibration_status["status"] == "NOT_YET_FIT"
    runtime_manifest = json.loads((output_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert runtime_manifest["calibration_status"] == "NOT_YET_FIT"


def test_refuses_to_package_mismatched_normalization(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    normalization_dir = tmp_path / "normalization"
    _write_normalization_dir(normalization_dir)
    # Deliberately wrong hash: the checkpoint claims a normalization
    # identity that does not match what's actually in normalization_dir.
    _save_checkpoint(checkpoint_dir, normalization_hash="0" * 64)

    output_dir = tmp_path / "bundle"
    try:
        bundle_script.build(
            checkpoint_dir=checkpoint_dir,
            normalization_dir=normalization_dir,
            output_dir=output_dir,
            workdir=Path(__file__).resolve().parents[2],
        )
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_refuses_to_overwrite_a_nonempty_output_directory(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    normalization_dir = tmp_path / "normalization"
    normalization_fingerprint = _write_normalization_dir(normalization_dir)
    _save_checkpoint(checkpoint_dir, normalization_hash=normalization_fingerprint)

    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    (output_dir / "preexisting.txt").write_text("do not clobber", encoding="utf-8")

    try:
        bundle_script.build(
            checkpoint_dir=checkpoint_dir,
            normalization_dir=normalization_dir,
            output_dir=output_dir,
            workdir=Path(__file__).resolve().parents[2],
        )
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_clean_runtime_loads_and_analyzes_from_the_bundle_alone(tmp_path: Path) -> None:
    """core-issues5.txt delta item 3 (P0 blocker): end-to-end clean-runtime
    proof -- build inference bundle -> load ONLY from that bundle ->
    construct V4PipelineFactory/runtime -> analyze a real non-locked
    incident -> neural path loads successfully, with no dependency on the
    original training corpus/checkpoint directory.

    The original checkpoint_dir/normalization_dir are DELETED after the
    bundle is built and BEFORE the bundle is loaded, so a real, provable
    dependency on either would surface as a hard failure here, not merely
    an untested assumption."""

    checkpoint_dir = tmp_path / "checkpoint"
    normalization_dir = tmp_path / "normalization"
    normalization_fingerprint = _write_normalization_dir(normalization_dir)
    # _save_checkpoint's own fixed governance (event_presence
    # runtime-enabled, source_node validated but NOT runtime-enabled) is
    # reused as-is -- this test's job is proving the NEURAL PATH ITSELF
    # loads and runs from a clean bundle (neural_failure is None,
    # runtime_mode FULL_HYBRID), not re-testing source_node governance
    # specifically (delta item 2's own dedicated tests already cover that).
    _save_checkpoint(checkpoint_dir, normalization_hash=normalization_fingerprint)

    bundle_dir = tmp_path / "bundle"
    bundle_script.build(
        checkpoint_dir=checkpoint_dir,
        normalization_dir=normalization_dir,
        output_dir=bundle_dir,
        workdir=Path(__file__).resolve().parents[2],
    )

    # Provable clean-runtime independence: nothing under checkpoint_dir/
    # normalization_dir survives past this point.
    shutil.rmtree(checkpoint_dir)
    shutil.rmtree(normalization_dir)
    assert not checkpoint_dir.exists()
    assert not normalization_dir.exists()

    from hydroswarm.runtime.v4_defaults import V4PipelineFactory
    from hydroswarm.simulation.wrapper import wntr

    if wntr is None:
        return

    factory = V4PipelineFactory(bundle_dir)
    assert factory.trained_assets_ready is True
    assert factory.fallback_reason is None
    assert factory.identity is not None
    assert factory.identity.runtime_enabled_outputs == frozenset({"event_presence"})

    network_path = Path(__file__).resolve().parents[2] / "data" / "frozen" / "golden_network.inp"
    pipeline = factory(None, network_path)

    from uuid import uuid4

    from hydroswarm.preprocessing import SensorSeries

    series = SensorSeries(
        node_id="J1",
        timestamps_seconds=(0.0, 3600.0),
        concentration_mg_l=(0.0, 0.78),
        pressure_m=(25.0, 24.0),
        health=(1.0, 1.0),
        missing=(False, False),
        drift=(False, False),
        delayed=(False, False),
    )
    result = pipeline.analyze(uuid4(), wntr.network.WaterNetworkModel(str(network_path)), [series])

    assert result.neural_failure is None
    assert result.runtime_mode.value == "FULL_HYBRID"
    assert result.semantic_predictions is not None
