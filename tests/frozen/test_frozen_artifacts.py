from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_manifest_matches_checked_in_inputs() -> None:
    frozen = ROOT / "data" / "frozen"
    manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    for filename, metadata in manifest["artifacts"].items():
        path = frozen / filename
        assert path.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]


def test_checked_in_golden_result_contains_measured_safety_story() -> None:
    result = json.loads(
        (ROOT / "reports" / "results" / "golden_result.json").read_text(encoding="utf-8")
    )
    assert result["source"]["true"] == "J2"
    assert result["localization"]["candidate_contraction"] > 0
    assert result["plans"]["unsafe"]["verification"]["decision"] == "REJECTED"
    assert result["plans"]["safe"]["verification"]["decision"] == "VERIFIED"
    assert result["workflow"]["approval_pause_state"] == "HUMAN_APPROVAL"
    assert result["workflow"]["completed_replay_state"] == "COMPLETE"

