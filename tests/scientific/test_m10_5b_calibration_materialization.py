from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/hydrocore_v5"
sys.path[:0] = [str(SCRIPTS)]

import m10_4_common as m104  # noqa: E402
from hydroswarm.calibration.conformal import SplitConformalCalibrator  # noqa: E402


def test_m10_5b_materialized_calibration_is_exact_m10_4_reconstruction() -> None:
    artifact_path = ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5b-calibration/calibration.json"
    report = json.loads((artifact_path.parent / "m10-5b-materialization.json").read_text())
    _model, model_hash = m104.load_canonical_model(20260814)
    original = m104.fit_frozen_calibrator(model_hash=model_hash, topology_hashes=m104.trained_family_topology_hashes())
    reloaded = SplitConformalCalibrator.load(artifact_path)
    assert asdict(original.artifact) == asdict(reloaded.artifact)
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == report["serialized_file_sha256"]
    assert report["support_file_sha256"] == hashlib.sha256(m104.CALIBRATION_SUPPORT_PATH.read_bytes()).hexdigest()
    assert report["all_support_candidate_sets_equal"] is True
    assert report["closure_state"] == "M10_5B_CALIBRATION_ARTIFACT_MATERIALIZED"
