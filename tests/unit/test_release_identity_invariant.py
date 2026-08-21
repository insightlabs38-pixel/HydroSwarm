"""V5 release-engineering rebase, hard release identity invariant (no
warning-only drift): every consumer of the frozen HydroCore-v5 identity --
the release bundle's own runtime manifest, the native setup verifier, the
strict self-test, the release-manifest generator, and the packaged runtime
ZIP -- must agree on the canonical frozen model and calibration identity.

Distinguishes file SHA-256 from artifact hash throughout: `model_sha256` is
the checksum of `model.safetensors` itself; `calibration_file_sha256` is the
checksum of `calibration.json` itself; `calibration_artifact_hash` is the
distinct governed identity `SplitConformalCalibrator`'s own artifact carries
(fit population/grouping/alpha, not merely "these bytes"). Comparing the
wrong pair against each other would pass by accident even if one consumer
actually drifted.

The container strict-self-test leg of this invariant (the same model/
calibration identity, verified inside the actual built image) is enforced
by `.github/workflows/docker-verify.yml` and `.github/workflows/release.yml`
directly against a running container, which this offline unit suite cannot
build; those workflows assert the exact same FROZEN_MODEL_SHA256 /
FROZEN_CALIBRATION_ARTIFACT_HASH constants defined here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = PROJECT_ROOT / "models" / "hydrocore-v5-release"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
build_release_manifest = importlib.import_module("build_release_manifest")
setup_common = importlib.import_module("setup_common")

# The single canonical frozen identity every consumer below is checked
# against -- itself verified directly against the committed bundle files,
# not hand-typed in isolation (see test_frozen_constants_match_the_committed_bundle_files).
FROZEN_MODEL_SHA256 = "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5"
FROZEN_CALIBRATION_FILE_SHA256 = "8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d"
FROZEN_CALIBRATION_ARTIFACT_HASH = "f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd"
FROZEN_RUNTIME_MANIFEST_SHA256 = "f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34"


def test_frozen_constants_match_the_committed_bundle_files() -> None:
    assert hashlib.sha256((BUNDLE_DIR / "model.safetensors").read_bytes()).hexdigest() == FROZEN_MODEL_SHA256
    assert hashlib.sha256((BUNDLE_DIR / "calibration.json").read_bytes()).hexdigest() == FROZEN_CALIBRATION_FILE_SHA256
    assert (
        hashlib.sha256((BUNDLE_DIR / "runtime_manifest.json").read_bytes()).hexdigest()
        == FROZEN_RUNTIME_MANIFEST_SHA256
    )


def test_v5_runtime_manifest_model_and_calibration_identity() -> None:
    manifest = json.loads((BUNDLE_DIR / "runtime_manifest.json").read_text())
    assert manifest["model_sha256"] == FROZEN_MODEL_SHA256
    assert manifest["calibration_file_sha256"] == FROZEN_CALIBRATION_FILE_SHA256
    assert manifest["calibration_artifact_hash"] == FROZEN_CALIBRATION_ARTIFACT_HASH


def test_native_setup_verify_bundle_reports_the_frozen_model_identity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    monkeypatch.setenv("HYDROSWARM_V5_BUNDLE_DIR", str(BUNDLE_DIR))
    assert setup_common.cmd_verify_bundle(argparse.Namespace()) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_sha256"] == FROZEN_MODEL_SHA256


@pytest.mark.real_simulation
def test_strict_self_test_reports_the_frozen_model_identity() -> None:
    from hydroswarm.cli import run_self_test

    report = run_self_test(strict=True)
    assert report["trained_assets"]["model_sha256"] == FROZEN_MODEL_SHA256


def test_release_manifest_reports_the_frozen_model_and_calibration_identity() -> None:
    manifest = build_release_manifest.build_manifest()
    assert manifest["model_hash"] == FROZEN_MODEL_SHA256
    assert manifest["calibration_artifact_hash"] == FROZEN_CALIBRATION_ARTIFACT_HASH
    assert manifest["runtime_manifest_sha256"] == FROZEN_RUNTIME_MANIFEST_SHA256


def test_release_zip_contains_the_frozen_model_identity(tmp_path: Path) -> None:
    build_release_bundle = importlib.import_module("build_release_bundle")

    output = tmp_path / "test-runtime.zip"
    build_release_bundle.build_bundle(output, release_version="v-test")

    with zipfile.ZipFile(output) as archive:
        model_bytes = archive.read("models/hydrocore-v5-release/model.safetensors")
        manifest_bytes = archive.read("models/hydrocore-v5-release/runtime_manifest.json")

    assert hashlib.sha256(model_bytes).hexdigest() == FROZEN_MODEL_SHA256
    assert hashlib.sha256(manifest_bytes).hexdigest() == FROZEN_RUNTIME_MANIFEST_SHA256


def test_no_consumer_falls_back_to_the_historical_v4_identity() -> None:
    """Sanity check that the frozen V5 identity is not coincidentally also
    a valid V4 identity -- the invariant above is meaningful only if V4's
    own model hash genuinely differs."""
    v4_manifest_path = PROJECT_ROOT / "models" / "hydrocore-v4-release" / "runtime_manifest.json"
    if not v4_manifest_path.is_file():
        pytest.skip("historical V4 bundle not present in this checkout")
    v4_manifest = json.loads(v4_manifest_path.read_text())
    assert v4_manifest.get("model_sha256") != FROZEN_MODEL_SHA256
