from __future__ import annotations

import json
import shutil
from pathlib import Path

from hydroswarm.runtime.v5_defaults import V5PipelineFactory, V5_RUNTIME_ENABLED_OUTPUTS, V5_TRAINED_TASKS

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "models/hydrocore-v5-release"


def test_v5_bundle_loads_selected_model_and_m10_5b_calibration() -> None:
    factory = V5PipelineFactory(BUNDLE, project_root=ROOT)
    assert factory.trained_assets_ready
    assert factory.model_hash == "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5"
    manifest = json.loads((BUNDLE / "runtime_manifest.json").read_text())
    assert manifest["calibration_artifact_hash"] == "f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd"
    assert set(manifest["runtime_enabled_outputs"]) == V5_RUNTIME_ENABLED_OUTPUTS
    assert set(manifest["trained_tasks"]) == V5_TRAINED_TASKS
    assert "next_step" not in V5_RUNTIME_ENABLED_OUTPUTS


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
