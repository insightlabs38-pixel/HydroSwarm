from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from tests.historical_artifact_portability import require_historical_artifact

from hydroswarm.runtime.v5_defaults import V5PipelineFactory, V5_RUNTIME_ENABLED_OUTPUTS, V5_TRAINED_TASKS

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "models/hydrocore-v5-release"
sys.path.insert(0, str(ROOT / "scripts/hydrocore_v5"))
import m10_4_common as m104  # noqa: E402


def test_v5_bundle_loads_selected_model_and_m10_5b_calibration() -> None:
    factory = V5PipelineFactory(BUNDLE, project_root=ROOT)
    assert factory.trained_assets_ready
    assert factory.model_hash == "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5"
    manifest = json.loads((BUNDLE / "runtime_manifest.json").read_text())
    assert manifest["calibration_artifact_hash"] == "f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd"
    assert set(manifest["runtime_enabled_outputs"]) == V5_RUNTIME_ENABLED_OUTPUTS
    assert set(manifest["trained_tasks"]) == V5_TRAINED_TASKS
    assert "next_step" not in V5_RUNTIME_ENABLED_OUTPUTS


def test_v5_release_matches_m10_4_authoritative_model_calibration_and_role_governance() -> None:
    record = m104.m10.canonical_s_checkpoint(20260814)
    require_historical_artifact(record["canonical_export_path"], record["canonical_export_sha256"], repo_root=ROOT)
    historical_model, historical_hash = m104.load_canonical_model(20260814)
    historical = m104.fit_frozen_calibrator(model_hash=historical_hash, topology_hashes=m104.trained_family_topology_hashes())
    release = V5PipelineFactory(BUNDLE, project_root=ROOT)
    assert release.trained_assets_ready
    assert historical_hash == release.model_hash
    assert historical.artifact.artifact_hash == release._calibrator.artifact.artifact_hash
    assert V5_TRAINED_TASKS == m104.M10_4_TRAINED_TASKS
    assert V5_RUNTIME_ENABLED_OUTPUTS == m104.M10_4_RUNTIME_ENABLED_OUTPUTS - {"next_step"}
    assert sum(parameter.numel() for parameter in historical_model.parameters()) == sum(parameter.numel() for parameter in release._model.parameters())


def test_v5_bundle_failures_do_not_fall_back_to_v4(tmp_path: Path) -> None:
    broken = tmp_path / "v5"
    shutil.copytree(BUNDLE, broken)
    (broken / "calibration.json").unlink()
    factory = V5PipelineFactory(broken, project_root=ROOT)
    assert factory.trained_assets_ready is False
    assert factory.model_hash is None
    assert factory.fallback_reason and factory.fallback_reason.startswith("v5_trained_assets_unavailable:")
    assert "v4" not in factory.fallback_reason.lower()


def test_v5_loader_does_not_reconstruct_calibration() -> None:
    source = (ROOT / "src/hydroswarm/runtime/v5_defaults.py").read_text()
    assert "SplitConformalCalibrator.load" in source
    assert "SplitConformalCalibrator.fit" not in source
    assert "rank_sample_locations" in (ROOT / "models/hydrocore-v5-release/runtime_manifest.json").read_text()
    assert "generate_response_plans" in (ROOT / "models/hydrocore-v5-release/runtime_manifest.json").read_text()


def test_current_status_index_names_only_additive_completion() -> None:
    status = json.loads((ROOT / "reports/evaluation/hydrocore-v5/m10/m10-current-status.json").read_text())
    closure = json.loads((ROOT / status["authoritative_m10_5_closure_path"]).read_text())
    assert status["m10_complete"] is True
    assert closure["closure_state"] == status["authoritative_m10_5_state"]
    assert status["m10_5b_state"] == "M10_5B_CALIBRATION_ARTIFACT_MATERIALIZED"
    assert status["locked_test_opened"] is False


@pytest.mark.parametrize("failure", [
    "missing_checkpoint", "corrupt_checkpoint", "checkpoint_sha_mismatch",
    "missing_calibration", "corrupt_calibration", "calibration_sha_mismatch",
    "feature_schema_mismatch", "fusion_mismatch", "manifest_corrupt",
])
def test_v5_identity_failure_matrix_is_fail_closed_without_v4_fallback(tmp_path: Path, failure: str) -> None:
    broken = tmp_path / failure
    shutil.copytree(BUNDLE, broken)
    manifest_path = broken / "runtime_manifest.json"
    if failure == "missing_checkpoint":
        (broken / "model.safetensors").unlink()
    elif failure == "corrupt_checkpoint":
        with (broken / "model.safetensors").open("ab") as handle:
            handle.write(b"corrupt")
    elif failure == "checkpoint_sha_mismatch":
        manifest = json.loads(manifest_path.read_text())
        manifest["model_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "missing_calibration":
        (broken / "calibration.json").unlink()
    elif failure == "corrupt_calibration":
        with (broken / "calibration.json").open("a") as handle:
            handle.write("corrupt")
    elif failure == "calibration_sha_mismatch":
        (broken / "calibration.json.sha256").write_text("0" * 64)
    elif failure in {"feature_schema_mismatch", "fusion_mismatch"}:
        manifest = json.loads(manifest_path.read_text())
        manifest["feature_schema_hash" if failure == "feature_schema_mismatch" else "fusion_config_hash"] = "mismatch"
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest_path.write_text("not-json")
    factory = V5PipelineFactory(broken, project_root=ROOT)
    assert factory.trained_assets_ready is False
    assert factory.model_hash is None
    assert factory.fallback_reason and factory.fallback_reason.startswith("v5_trained_assets_unavailable:")
    assert "v4" not in factory.fallback_reason.lower()
