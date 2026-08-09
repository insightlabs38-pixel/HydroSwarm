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

import pytest
import torch

from hydroswarm.model.core import HydroCore
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.preprocessing.builder import NO_NORMALIZATION_SENTINEL
from hydroswarm.runtime.v4_defaults import V4PipelineFactory
from hydroswarm.runtime.v4_normalization import load_runtime_normalization_bundle
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
    # core-issues5.txt Section 3: these tiny test fixtures genuinely train
    # without any governed normalization artifact, so the honest value is
    # the real sentinel V4PipelineFactory checks for -- NOT an arbitrary
    # placeholder string, which (after the Section 3 fix) V4PipelineFactory
    # would instead interpret as "a real normalization artifact must be
    # loaded and hash-verified", spuriously failing closed for every test
    # here that does not care about normalization at all.
    kwargs = dict(
        normalization_hash=NO_NORMALIZATION_SENTINEL,
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


#: Masked locally by a warm data/generated/signatures cache (gitignored,
#: session-accumulated); found by the real_simulation runtime audit in
#: tests/conftest.py on a clean-cache CI run.
@pytest.mark.real_simulation
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


#: Constructs a real HybridInferencePipeline against a real network
#: (factory(None, network_path)) -- masked locally by a warm
#: data/generated/signatures cache (gitignored, session-accumulated); a
#: fresh checkout/CI run hits a real cache miss here (382s on the CI run
#: that found this). Found by the real_simulation runtime audit in
#: tests/conftest.py, not the static call-count audit (itself run against
#: a warm local cache).
@pytest.mark.real_simulation
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


# core-issues5.txt Section 3 (P0 blocker): a checkpoint that declares real
# normalized training input must have that exact train-owned normalization
# artifact loaded, hash-verified, and injected into the live feature-
# building path -- never silently served with an unnormalized
# HydraulicFeatureBuilder. These tests use the real, committed
# data/learning-v2/cycle-b2/normalization artifact (the one Stage F's
# actual joint-v4 training corpus was built from), not a synthetic fixture,
# so a real fingerprint mismatch would be caught the same way it would in
# production.
_REAL_NORMALIZATION_DIR = _REPO_ROOT / "data" / "learning-v2" / "cycle-b2" / "normalization"


#: Masked locally by a warm data/generated/signatures cache (gitignored,
#: session-accumulated); found by the real_simulation runtime audit in
#: tests/conftest.py on a clean-cache CI run.
@pytest.mark.real_simulation
def test_real_normalization_bundle_is_loaded_and_wired_into_feature_builder(tmp_path: Path) -> None:
    bundle = load_runtime_normalization_bundle(_REAL_NORMALIZATION_DIR)
    directory = tmp_path / "checkpoint"
    _save_v4_checkpoint(directory, normalization_hash=bundle.fingerprint)

    factory = V4PipelineFactory(directory, normalization_dir=_REAL_NORMALIZATION_DIR)
    assert factory.trained_assets_ready is True
    assert factory.fallback_reason is None
    assert factory._feature_builder is not None
    assert factory._feature_builder.normalization_fingerprint == bundle.fingerprint

    if wntr is None:
        return
    network_path = _REPO_ROOT / "data" / "frozen" / "golden_network.inp"
    pipeline = factory(None, network_path)
    assert pipeline.feature_builder.normalization_fingerprint == bundle.fingerprint


def test_missing_normalization_artifact_directory_fails_closed(tmp_path: Path) -> None:
    """A checkpoint that declares real (non-sentinel) normalized training
    input, pointed at a normalization_dir with no artifacts at all, must
    fail closed rather than silently serve unnormalized features."""

    directory = tmp_path / "checkpoint"
    _save_v4_checkpoint(directory, normalization_hash="a" * 64)

    factory = V4PipelineFactory(directory, normalization_dir=tmp_path / "no-such-normalization-dir")
    assert factory.trained_assets_ready is False
    assert factory.fallback_reason == "v4_trained_assets_unavailable:NormalizationBundleError"
    assert factory._feature_builder is None


def test_normalization_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    """The real committed artifact loads fine on its own, but if the
    checkpoint identity's recorded normalization_hash does not match its
    real fingerprint (a stale identity, or a checkpoint actually trained
    against a different normalization artifact), the factory must fail
    closed rather than silently serve a mismatched normalization."""

    directory = tmp_path / "checkpoint"
    _save_v4_checkpoint(directory, normalization_hash="0" * 64)

    factory = V4PipelineFactory(directory, normalization_dir=_REAL_NORMALIZATION_DIR)
    assert factory.trained_assets_ready is False
    assert factory.fallback_reason == "v4_trained_assets_unavailable:NormalizationBundleError"


# core-issues5.txt Section 4 (P0 blocker): the served network's signature-
# policy mode and the real policy hash used to generate its
# SignatureArtifact must be recorded, and the policy itself must be the
# real governed training policy -- not an independently-invented, smaller
# runtime hypothesis grid (start_time_bins=(0, 60) etc, a real defect this
# fix closes).


#: Masked locally by a warm data/generated/signatures cache (gitignored,
#: session-accumulated); found by the real_simulation runtime audit in
#: tests/conftest.py on a clean-cache CI run.
@pytest.mark.real_simulation
def test_committed_training_topology_file_resolves_to_governed_mode(tmp_path: Path) -> None:
    """`data/topology-transfer/branched-loop.inp`, loaded directly (no
    write/re-read round trip), is one of the exact committed .inp files
    scripts/generate_cycle_b_corpus.py's own TOPOLOGY_BUILDERS load this
    same way -- must resolve to GOVERNED_KNOWN_NETWORK.

    Deliberately NOT testing this via `hydroswarm.simulation.
    build_wntr_network()` (golden-reference) written out with
    `wntr.network.write_inpfile` and reloaded: WNTR's own SI/US round trip
    introduces ~1e-9 relative floating-point noise into
    length/diameter/roughness (confirmed directly -- e.g. `0.3` becomes
    `0.2999999999988`), which changes `network_sha256`'s hash despite
    describing the physically identical network. golden-reference has no
    single canonical `.inp` file to load byte-identically the way
    branched-loop/loop-grid do (it is only ever constructed in-memory at
    both training and evaluation time), so it cannot be used for this
    specific "loaded twice, same bytes on disk" integration test --
    covered instead at the unit level in
    tests/unit/test_signature_policy.py, which computes
    `network_sha256(build_wntr_network())` directly with no disk round
    trip at all.
    """

    if wntr is None:
        return
    from hydroswarm.classical import GOVERNED_TRAINING_SIGNATURE_POLICY

    directory = tmp_path / "checkpoint"
    _save_v4_checkpoint(directory)
    factory = V4PipelineFactory(directory)
    network_path = _REPO_ROOT / "data" / "topology-transfer" / "branched-loop.inp"
    factory(None, network_path)

    assert factory.signature_mode == "GOVERNED_KNOWN_NETWORK"
    assert factory.signature_policy_hash == GOVERNED_TRAINING_SIGNATURE_POLICY.policy_hash


def test_golden_fixture_snapshot_resolves_to_runtime_generated_mode(tmp_path: Path) -> None:
    """data/frozen/golden_network.inp is a single frozen RANDOMIZED
    scenario snapshot (see data/frozen/manifest.json's
    hydroswarm.evaluation.golden.freeze_golden_inputs generator), not the
    pristine base network -- it must NOT be claimed as governed/training-
    owned provenance this exact-hash check cannot actually verify."""

    if wntr is None:
        return

    checkpoint_dir = tmp_path / "checkpoint"
    _save_v4_checkpoint(checkpoint_dir)
    factory = V4PipelineFactory(checkpoint_dir)
    network_path = _REPO_ROOT / "data" / "frozen" / "golden_network.inp"
    factory(None, network_path)

    assert factory.signature_mode == "RUNTIME_GENERATED_IMPORTED_NETWORK"


def test_governed_policy_produces_the_full_training_hypothesis_grid(tmp_path: Path) -> None:
    """The runtime-generated SignatureArtifact's own recorded hypotheses
    must reflect the REAL governed training policy's bins
    (start_time_bins=(0, 60, 120, 240) etc), not a smaller ad-hoc runtime
    grid -- the concrete train/serve mismatch this fix closes."""

    if wntr is None:
        return
    from hydroswarm.classical import GOVERNED_TRAINING_SIGNATURE_POLICY

    directory = tmp_path / "checkpoint"
    _save_v4_checkpoint(directory)
    factory = V4PipelineFactory(directory)
    network_path = _REPO_ROOT / "data" / "frozen" / "golden_network.inp"
    pipeline = factory(None, network_path)

    junction_count = len({h.source_node for h in pipeline.signature_artifact.hypotheses})
    assert len(pipeline.signature_artifact.hypotheses) == junction_count * (
        len(GOVERNED_TRAINING_SIGNATURE_POLICY.start_time_bins)
        * len(GOVERNED_TRAINING_SIGNATURE_POLICY.duration_bins)
        * len(GOVERNED_TRAINING_SIGNATURE_POLICY.strength_bins)
        * len(GOVERNED_TRAINING_SIGNATURE_POLICY.demand_regimes)
    )
    assert pipeline.signature_artifact.sample_times_seconds == GOVERNED_TRAINING_SIGNATURE_POLICY.sample_times_seconds


def test_sentinel_normalization_hash_skips_bundle_loading(tmp_path: Path) -> None:
    """A checkpoint that honestly declares NO_NORMALIZATION_SENTINEL (truly
    trained without governed normalization) must load successfully with a
    plain, unnormalized HydraulicFeatureBuilder -- normalization_dir is
    never even consulted, so a missing/wrong directory there must not
    matter."""

    directory = tmp_path / "checkpoint"
    _save_v4_checkpoint(directory)  # default normalization_hash is the sentinel

    factory = V4PipelineFactory(directory, normalization_dir=tmp_path / "does-not-exist")
    assert factory.trained_assets_ready is True
    assert factory._feature_builder is not None
    assert factory._feature_builder.normalization_fingerprint == NO_NORMALIZATION_SENTINEL
