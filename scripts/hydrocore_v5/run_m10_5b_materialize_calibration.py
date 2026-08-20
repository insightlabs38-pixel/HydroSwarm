"""Materialize (never re-fit) the exact M10.4 frozen calibration artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(SCRIPTS)]

import m10_4_common as m104  # noqa: E402
from hydroswarm.calibration.conformal import SplitConformalCalibrator  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

SEED = 20260814
EXPECTED_MODEL_SHA = "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5"
REPORT_DIR = ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5b-calibration"
PROTOCOL_DOC = ROOT / "docs/evaluation/HYDROCORE_V5_M10_5B_CALIBRATION_MATERIALIZATION_AMENDMENT.md"
ARTIFACT = REPORT_DIR / "calibration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def support_rows() -> tuple[list[dict[str, Any]], int]:
    rows = [json.loads(line) for line in m104.CALIBRATION_SUPPORT_PATH.read_text().splitlines() if line]
    return rows, sum(row["arm"] == m104.CALIBRATION_SUPPORT_ARM for row in rows)


def construct_exact_m10_4_calibrator() -> SplitConformalCalibrator:
    # This is intentionally the exact frozen M10.4 construction, not a local fit.
    _model, model_hash = m104.load_canonical_model(SEED)
    assert model_hash == EXPECTED_MODEL_SHA
    return m104.fit_frozen_calibrator(
        model_hash=model_hash, topology_hashes=m104.trained_family_topology_hashes()
    )


def run() -> dict[str, Any]:
    assert locked_test_opened(ROOT) is False
    assert PROTOCOL_DOC.exists()
    rows, arm_count = support_rows()
    support_hash = sha256(m104.CALIBRATION_SUPPORT_PATH)
    first = construct_exact_m10_4_calibrator()
    second = construct_exact_m10_4_calibrator()
    first_artifact, second_artifact = first.artifact, second.artifact
    independent_equal = asdict(first_artifact) == asdict(second_artifact)
    assert independent_equal
    assert first_artifact.artifact_hash == second_artifact.artifact_hash
    assert first_artifact.global_scores == second_artifact.global_scores
    assert first_artifact.mondrian_scores == second_artifact.mondrian_scores
    assert first_artifact.network_scores == second_artifact.network_scores
    assert first_artifact.report == second_artifact.report
    assert first_artifact.validated_topology_hashes == second_artifact.validated_topology_hashes

    first.save(ARTIFACT)
    reloaded = SplitConformalCalibrator.load(ARTIFACT)
    reload_equal = asdict(reloaded.artifact) == asdict(first_artifact)
    assert reload_equal
    assert sha256(ARTIFACT) == ARTIFACT.with_suffix(".json.sha256").read_text().strip()

    candidate_equal = True
    group_equal = True
    for row in rows:
        if row["arm"] != m104.CALIBRATION_SUPPORT_ARM:
            continue
        kwargs = {"condition": row["condition"], "network_id": f"{row['family']}:{row['depth_bucket']}"}
        if first.selection(**kwargs)[:2] != reloaded.selection(**kwargs)[:2]:
            group_equal = False
        if first.candidate_set(row["probabilities"], **kwargs) != reloaded.candidate_set(row["probabilities"], **kwargs):
            candidate_equal = False
    assert group_equal and candidate_equal
    assert locked_test_opened(ROOT) is False

    payload = {
        "kind": "M10_5B_CALIBRATION_MATERIALIZATION",
        "closure_state": "M10_5B_CALIBRATION_ARTIFACT_MATERIALIZED",
        "protocol_hash": sha256(PROTOCOL_DOC),
        "constructor": "m10_4_common.fit_frozen_calibrator",
        "constructor_invocation": "load_canonical_model(20260814); fit_frozen_calibrator(model_hash=model_hash, topology_hashes=trained_family_topology_hashes())",
        "selected_seed": SEED,
        "selected_model_sha256": EXPECTED_MODEL_SHA,
        "support_file": str(m104.CALIBRATION_SUPPORT_PATH.relative_to(ROOT)),
        "support_file_sha256": support_hash,
        "support_row_count": len(rows),
        "arm": m104.CALIBRATION_SUPPORT_ARM,
        "arm_filtered_row_count": arm_count,
        "alpha": m104.CALIBRATION_ALPHA,
        "minimum_group_size": m104.CALIBRATION_MINIMUM_GROUP_SIZE,
        "feature_schema_hash": first_artifact.feature_schema_hash,
        "fusion_config_hash": first_artifact.fusion_config_hash,
        "validated_topology_hashes": list(first_artifact.validated_topology_hashes),
        "artifact_schema_version": first_artifact.schema_version,
        "artifact_path": str(ARTIFACT.relative_to(ROOT)),
        "artifact_hash": first_artifact.artifact_hash,
        "serialized_file_sha256": sha256(ARTIFACT),
        "independent_reconstruction_equal": independent_equal,
        "reload_structural_equal": reload_equal,
        "all_support_group_selection_equal": group_equal,
        "all_support_candidate_sets_equal": candidate_equal,
        "locked_test_opened_before": False,
        "locked_test_opened_after": False,
        "new_calibration_examples": 0,
        "statistical_parameter_changes": 0,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "m10-5b-materialization.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "m10-5b-closure.json").write_text(json.dumps({
        "kind": "M10_5B_CLOSURE", "closure_state": payload["closure_state"],
        "protocol_hash": payload["protocol_hash"], "artifact_path": payload["artifact_path"],
        "artifact_hash": payload["artifact_hash"], "locked_test_opened": False,
    }, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
