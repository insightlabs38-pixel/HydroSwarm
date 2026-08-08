"""core-issues3.txt Phase 15 item 11: runtime tests for the v4-aware
loader (`hydroswarm.runtime.v4_defaults.V4PipelineFactory`). Builds
self-contained tiny v4 checkpoints in tmp_path via
`hydroswarm.training.checkpoint_identity.save_v4_checkpoint` -- mirrors
tests/unit/test_checkpoint_identity.py's own `_tiny_model`/`_identity_for`/
`_optimizer_scheduler` construction pattern (duplicated, not imported: the
two files live in different test directories with no shared package
`__init__.py`, matching this project's established per-directory test
module convention) rather than depending on any real trained checkpoint
under the gitignored `experiments/runs/` tree (which does not persist
across sessions/clones -- see this project's own established persistence
caveat)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from hydroswarm.model.core import HydroCore
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.runtime.v4_defaults import V4PipelineFactory
from hydroswarm.simulation.wrapper import wntr
from hydroswarm.training import checkpoint_identity as ci

_REPO_ROOT = Path(__file__).resolve().parents[2]

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


def _identity_for(model: HydroCore, **overrides: object) -> ci.CheckpointIdentity:
    kwargs = dict(
        normalization_hash="no-normalization",
        fusion_policy_hash="fixed-weight-v1:neural=0.5",
        source_corpus_manifest_hashes=("abc123",),
        trained_outputs=frozenset({"source_node"}),
        validated_outputs=frozenset({"source_node"}),
        runtime_enabled_outputs=frozenset({"source_node"}),
    )
    kwargs.update(overrides)
    return ci.build_checkpoint_identity(model, **kwargs)  # type: ignore[arg-type]


def _optimizer_scheduler(model: HydroCore):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    return optimizer, scheduler


def _save_v4_checkpoint(directory: Path, **identity_overrides: object) -> ci.CheckpointIdentity:
    model = _tiny_model()
    identity = _identity_for(model, **identity_overrides)
    optimizer, scheduler = _optimizer_scheduler(model)
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


def test_missing_checkpoint_directory_fails_closed(tmp_path: Path) -> None:
    factory = V4PipelineFactory(tmp_path / "does-not-exist")
    assert factory.trained_assets_ready is False
    assert factory.identity is None
    assert factory.fallback_reason == "v4_trained_assets_unavailable:NotAV4CheckpointError"


def test_v4_checkpoint_loads_successfully_and_exposes_identity(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    identity = _save_v4_checkpoint(directory)

    factory = V4PipelineFactory(directory)
    assert factory.trained_assets_ready is True
    assert factory.fallback_reason is None
    assert factory.identity is not None
    assert factory.identity.runtime_enabled_outputs == identity.runtime_enabled_outputs


def test_mismatched_architecture_metadata_fails_closed(tmp_path: Path) -> None:
    """A checkpoint_identity.json hand-edited to declare a different
    architecture_version than this build's ARCHITECTURE_VERSION_V4 must be
    rejected, not silently loaded against a mismatched live model."""

    directory = tmp_path / "checkpoint"
    _save_v4_checkpoint(directory)

    identity_path = directory / "checkpoint_identity.json"
    raw = json.loads(identity_path.read_text(encoding="utf-8"))
    raw["architecture_version"] = "hydrocore-v3"
    identity_path.write_text(json.dumps(raw), encoding="utf-8")

    factory = V4PipelineFactory(directory)
    assert factory.trained_assets_ready is False
    assert factory.fallback_reason is not None
    assert "unavailable" in factory.fallback_reason


def test_corrupted_artifact_manifest_fingerprint_fails_closed(tmp_path: Path) -> None:
    """artifact_manifest.json's recorded checkpoint_identity_fingerprint
    must match a fresh recomputation from checkpoint_identity.json -- a
    real tamper/corruption signal (core-issues3.txt Phase 9.1 "fail closed
    on architecture or hash mismatch")."""

    directory = tmp_path / "checkpoint"
    _save_v4_checkpoint(directory)

    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_identity_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    factory = V4PipelineFactory(directory)
    assert factory.trained_assets_ready is False


def test_no_calibration_artifact_configured_falls_back_to_uncalibrated(tmp_path: Path) -> None:
    """V4PipelineFactory never sets a calibration_path by default (no v4
    checkpoint has passed Phase 19 selection/calibration fitting yet) --
    the resulting pipeline must still construct successfully, with
    calibration_artifact=None (matches DefaultPipelineFactory's own
    unsupported-topology fallback path, not a new failure mode)."""

    directory = tmp_path / "checkpoint"
    _save_v4_checkpoint(directory)
    factory = V4PipelineFactory(directory)
    assert factory.trained_assets_ready is True

    if wntr is None:
        return
    network_path = _REPO_ROOT / "data" / "frozen" / "golden_network.inp"
    pipeline = factory(None, network_path)
    assert pipeline.calibration_artifact is None


def test_pipeline_from_v4_factory_uses_sentinel_only_trained_tasks_and_declared_runtime_outputs(
    tmp_path: Path,
) -> None:
    """Scout/Strategist/OOD stay excluded from trained_tasks regardless of
    what a v4 identity's finer-grained runtime_enabled_outputs says --
    core-issues3.txt Phase 14: none of those roles have passed promotion
    for any v4 checkpoint built so far."""

    directory = tmp_path / "checkpoint"
    identity = _save_v4_checkpoint(
        directory,
        trained_outputs=frozenset({"source_node", "event_presence"}),
        validated_outputs=frozenset({"source_node", "event_presence"}),
        runtime_enabled_outputs=frozenset({"source_node", "event_presence"}),
    )
    factory = V4PipelineFactory(directory)

    if wntr is None:
        return
    network_path = _REPO_ROOT / "data" / "frozen" / "golden_network.inp"
    pipeline = factory(None, network_path)
    assert pipeline.trained_tasks == frozenset({"sentinel"})
    assert pipeline.runtime_enabled_outputs == identity.runtime_enabled_outputs
    assert pipeline.runtime_enabled_outputs == frozenset({"source_node", "event_presence"})
