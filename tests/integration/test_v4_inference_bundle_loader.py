"""core-issues5.txt delta item 3 (P0 blocker): dedicated unit coverage for
`hydroswarm.runtime.v4_inference_bundle.load_v4_inference_bundle`, the
loader that makes a `scripts/build_v4_inference_release_bundle.py` bundle
actually bootable (previously nothing could load it -- see that module's
own docstring for the two real defects this closes).

Reuses `test_v4_release_bundle.py`'s own bundle-building fixtures rather
than duplicating them (same directory, no shared package `__init__.py` --
matches this project's established per-directory test import convention,
e.g. `test_hybrid_pipeline_v4_gating.py` importing from
`test_hybrid_pipeline`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hydroswarm.runtime.v4_inference_bundle import (
    InferenceBundleError,
    load_v4_inference_bundle,
)

from test_v4_release_bundle import (  # noqa: E402
    _save_checkpoint,
    _write_normalization_dir,
    bundle_script,
)


def _build_bundle(tmp_path: Path) -> Path:
    checkpoint_dir = tmp_path / "checkpoint"
    normalization_dir = tmp_path / "normalization"
    normalization_fingerprint = _write_normalization_dir(normalization_dir)
    _save_checkpoint(checkpoint_dir, normalization_hash=normalization_fingerprint)

    bundle_dir = tmp_path / "bundle"
    bundle_script.build(
        checkpoint_dir=checkpoint_dir,
        normalization_dir=normalization_dir,
        output_dir=bundle_dir,
        workdir=Path(__file__).resolve().parents[2],
    )
    return bundle_dir


def test_loads_successfully_and_reproduces_the_real_model_and_identity(tmp_path: Path) -> None:
    bundle_dir = _build_bundle(tmp_path)
    bundle = load_v4_inference_bundle(bundle_dir)

    assert bundle.model is not None
    assert bundle.model_hash == hashlib.sha256((bundle_dir / "model.safetensors").read_bytes()).hexdigest()
    assert bundle.identity.runtime_enabled_outputs == frozenset({"event_presence"})
    assert bundle.calibration is None
    assert bundle.calibration_status == "NOT_YET_FIT"
    assert bundle.bundle_dir == bundle_dir


def test_missing_required_file_fails_closed(tmp_path: Path) -> None:
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "output_governance.json").unlink()

    with pytest.raises(InferenceBundleError):
        load_v4_inference_bundle(bundle_dir)


def test_tampered_file_content_fails_sha256sums_check(tmp_path: Path) -> None:
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "output_governance.json").write_text('{"tampered": true}', encoding="utf-8")

    with pytest.raises(InferenceBundleError, match="does not match SHA256SUMS"):
        load_v4_inference_bundle(bundle_dir)


def test_artifact_manifest_disagreeing_with_sha256sums_fails_closed(tmp_path: Path) -> None:
    bundle_dir = _build_bundle(tmp_path)
    manifest_path = bundle_dir / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A hand-tampered manifest that still matches SHA256SUMS for every OTHER
    # file, but disagrees on this one -- must be caught even though
    # SHA256SUMS itself was not touched.
    manifest["files"]["output_governance.json"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # Re-sign SHA256SUMS for the tampered manifest file itself so ONLY the
    # cross-check (not the raw file-hash check) is what catches this.
    sums_path = bundle_dir / "SHA256SUMS"
    lines = [line for line in sums_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [line for line in lines if not line.endswith("artifact-manifest.json")]
    new_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    lines.append(f"{new_hash}  artifact-manifest.json")
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(InferenceBundleError, match="do not match SHA256SUMS"):
        load_v4_inference_bundle(bundle_dir)


def test_governance_drift_between_identity_and_output_governance_json_fails_closed(tmp_path: Path) -> None:
    """A tampered output_governance.json that is ALSO re-signed consistently
    in artifact-manifest.json/SHA256SUMS (so the file-integrity layer alone
    cannot catch it) must still be rejected by the cross-check against
    checkpoint_identity.json's own governance sets -- the real
    load-bearing consistency guarantee, not merely file-content
    integrity."""

    bundle_dir = _build_bundle(tmp_path)
    governance_path = bundle_dir / "output_governance.json"
    governance = json.loads(governance_path.read_text(encoding="utf-8"))
    governance["runtime_enabled_outputs"] = sorted({*governance["runtime_enabled_outputs"], "source_node"})
    new_text = json.dumps(governance, indent=2, sort_keys=True, default=str) + "\n"
    governance_path.write_text(new_text, encoding="utf-8")
    new_hash = hashlib.sha256(new_text.encode()).hexdigest()

    # artifact-manifest.json never records a hash for itself (the real
    # builder computes it before its own file exists -- see
    # scripts/build_v4_inference_release_bundle.py), so only
    # output_governance.json's entries need updating here, matching the
    # real bundle's own convention exactly.
    manifest_path = bundle_dir / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["output_governance.json"] = new_hash
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    sums_path = bundle_dir / "SHA256SUMS"
    lines = [line for line in sums_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [line for line in lines if not line.endswith("output_governance.json")]
    lines.append(f"{new_hash}  output_governance.json")
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(InferenceBundleError, match="runtime_enabled_outputs"):
        load_v4_inference_bundle(bundle_dir)


def test_missing_directory_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(InferenceBundleError):
        load_v4_inference_bundle(tmp_path / "does-not-exist")
