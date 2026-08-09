"""DefaultPipelineFactory composes the actual promoted checkpoint,
calibration artifact, and (now) normalization identity used in production.
No prior test exercised it directly -- these are the tests that would have
caught core-issues.txt repair item 2's real regression (the promoted
checkpoint's source_region_head became shape-incompatible) before it ever
reached a real deployment."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hydroswarm.runtime.defaults import DefaultPipelineFactory
from hydroswarm.simulation.wrapper import wntr

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _copied_checkpoint_metadata(tmp_path: Path) -> tuple[Path, dict]:
    """Copy the real promoted checkpoint's safetensors into an isolated
    project root, and return that root plus a mutable copy of its real
    metadata.json so a test can corrupt one field without touching the
    tracked models/ directory."""

    (tmp_path / "models").mkdir()
    shutil.copyfile(
        _REPO_ROOT / "models" / "hydrocore-s-learning-v1.safetensors",
        tmp_path / "models" / "hydrocore-s-learning-v1.safetensors",
    )
    metadata = json.loads(
        (_REPO_ROOT / "models" / "hydrocore-s-learning-v1.metadata.json").read_text(encoding="utf-8")
    )
    return tmp_path, metadata


def _write_metadata(root: Path, metadata: dict) -> None:
    (root / "models" / "hydrocore-s-learning-v1.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_promoted_checkpoint_loads_successfully() -> None:
    factory = DefaultPipelineFactory()
    assert factory.trained_assets_ready is True
    assert factory.fallback_reason is None


def test_promoted_checkpoint_required_the_known_v2_source_region_migration() -> None:
    """Documents, rather than hides, that the currently promoted checkpoint
    predates repair item 2's architecture fix -- exactly the two
    source_region_head parameters, nothing else, were re-initialized
    rather than loaded from the checkpoint's own (shape-incompatible,
    never-meaningfully-trained) weights."""

    factory = DefaultPipelineFactory()
    assert factory.trained_assets_ready is True
    assert set(factory.migrated_parameters) == {
        "source_region_head.network.1.weight",
        "source_region_head.network.1.bias",
    }


def test_promoted_checkpoint_calibration_still_validates_with_no_normalization() -> None:
    """core-issues.txt repair item 6: the promoted calibration artifact was
    fit with no governed normalization layer (the only state that has ever
    existed), and the runtime feature builder also defaults to none -- they
    must still agree, not silently diverge because one side started
    recording an explicit identity and the other did not."""

    factory = DefaultPipelineFactory()
    factory._load_assets()
    assert factory._calibrator is not None
    assert factory._feature_builder is not None
    from hydroswarm.preprocessing.builder import NO_NORMALIZATION_SENTINEL

    assert factory._calibrator.artifact.normalization_hash == NO_NORMALIZATION_SENTINEL
    assert factory._feature_builder.normalization_fingerprint == NO_NORMALIZATION_SENTINEL


#: Constructs a real HybridInferencePipeline against a real network and
#: calls it -- masked locally by a warm data/generated/signatures cache
#: (gitignored, session-accumulated); a fresh checkout/CI run hits a real
#: cache miss here. Found by the real_simulation runtime audit in
#: tests/conftest.py on a clean-cache CI run, not the static call-count
#: audit (which was itself run against a warm local cache).
@pytest.mark.real_simulation
def test_promoted_checkpoint_declares_only_sentinel_as_a_trained_task() -> None:
    """core-issues.txt repair item 8: the real promoted checkpoint's own
    metadata.json now declares trained_tasks, and the real hybrid pipeline
    built from it must actually gate Scout/Strategist/OOD accordingly --
    not just the constant in isolation."""

    factory = DefaultPipelineFactory()
    assert factory.trained_assets_ready is True
    assert factory.trained_tasks == frozenset({"sentinel"})

    if wntr is None:
        return
    network_path = _REPO_ROOT / "data" / "frozen" / "golden_network.inp"
    pipeline = factory(None, network_path)
    assert pipeline.trained_tasks == frozenset({"sentinel"})


def test_missing_architecture_config_fails_closed(tmp_path: Path) -> None:
    """core-issues.txt repair item 9: a checkpoint that does not declare its
    own architecture_config must never fall back to instantiating the
    model with hardcoded constructor defaults -- it must refuse to load."""

    root, metadata = _copied_checkpoint_metadata(tmp_path)
    del metadata["architecture_config"]
    _write_metadata(root, metadata)

    factory = DefaultPipelineFactory(root)
    assert factory.trained_assets_ready is False
    assert factory.fallback_reason == "trained_assets_unavailable:KeyError"


def test_stale_architecture_version_fails_closed(tmp_path: Path) -> None:
    """architecture_version is the one architecture_config field the
    runtime does not pass to HydroCore.from_variant (it is always this
    build's ARCHITECTURE_VERSION constant, not a constructor knob), so a
    checkpoint recorded against an older code version is the one mismatch
    verify_architecture_compatibility can still actually observe here --
    every other field is now structurally guaranteed to match, because the
    model is built directly from them (repair item 9)."""

    root, metadata = _copied_checkpoint_metadata(tmp_path)
    metadata["architecture_config"]["architecture_version"] = "hydrocore-v2"
    _write_metadata(root, metadata)

    factory = DefaultPipelineFactory(root)
    assert factory.trained_assets_ready is False
    assert factory.fallback_reason == "trained_assets_unavailable:ArchitectureCompatibilityError"


def test_normalization_hash_mismatch_against_checkpoint_metadata_fails_closed(tmp_path: Path) -> None:
    """core-issues.txt repair item 9: normalization_hash is now also
    checked directly against the checkpoint's own declared identity, not
    only indirectly through calibration validation."""

    root, metadata = _copied_checkpoint_metadata(tmp_path)
    metadata["normalization_hash"] = "0" * 64
    _write_metadata(root, metadata)

    factory = DefaultPipelineFactory(root)
    assert factory.trained_assets_ready is False
    assert factory.fallback_reason == "trained_assets_unavailable:ValueError"
