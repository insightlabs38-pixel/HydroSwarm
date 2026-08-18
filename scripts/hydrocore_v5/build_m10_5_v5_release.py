"""Build the immutable M10.5 v5 serving bundle from governed artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
import m10_4_common as m104  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402

SEED = 20260814
MODEL_SHA = "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5"
BUNDLE = ROOT / "models/hydrocore-v5-release"
CALIBRATION = ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5b-calibration/calibration.json"
PROTOCOL = ROOT / "docs/evaluation/HYDROCORE_V5_M10_5_COMPLETION_PROTOCOL.md"
M105B = ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5b-calibration/m10-5b-materialization.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict[str, object]:
    assert sha(CALIBRATION) == json.loads(M105B.read_text())["serialized_file_sha256"]
    source = Path(m104.m10.canonical_s_checkpoint(SEED)["canonical_export_path"])
    assert sha(source) == MODEL_SHA
    BUNDLE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, BUNDLE / "model.safetensors")
    shutil.copyfile(CALIBRATION, BUNDLE / "calibration.json")
    shutil.copyfile(CALIBRATION.with_suffix(".json.sha256"), BUNDLE / "calibration.json.sha256")
    materialization = json.loads(M105B.read_text())
    manifest = {
        "release_schema_version": "hydroswarm-v5-release-v1",
        "selected_seed": SEED, "model_sha256": MODEL_SHA,
        "calibration_file_sha256": materialization["serialized_file_sha256"],
        "calibration_artifact_hash": materialization["artifact_hash"],
        "feature_schema_hash": materialization["feature_schema_hash"],
        "fusion_config_hash": materialization["fusion_config_hash"],
        "validated_topology_hashes": materialization["validated_topology_hashes"],
        "trained_tasks": ["sentinel"],
        "runtime_enabled_outputs": ["event_cause", "event_presence", "evidence_sufficiency", "relative_strength", "source_node"],
        "deterministic_authority": {"ood": "OODDetector", "scout": "rank_sample_locations", "planner": "generate_response_plans", "physical_verification": "WNTR/EPANET", "human_approval_required": True, "autonomous_actuation": False},
        "m10_4_protocol_hash": "cd0ac1f2d5a12a771cc441b4ea19bf0d76c672809b35d3d178f8893b768a177c",
        "m10_5a_selection_commit": "6829c676bf9aa074dbc9e62150b256efd0475335",
        "m10_5b_protocol_hash": materialization["protocol_hash"],
        "m10_5b_artifact_identity": materialization["artifact_hash"],
        "m10_5_protocol_hash": sha(PROTOCOL),
        "source_git_provenance": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_config": SHARED_MODEL_CONFIG,
        "feature_semantics": "M10.4-tested default builder; unobserved_age_sentinel=incident_elapsed retained despite M9.6 fixed-age training record",
    }
    manifest["files"] = {name: sha(BUNDLE / name) for name in ("model.safetensors", "calibration.json", "calibration.json.sha256")}
    (BUNDLE / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
