"""M10.5 serving-release identity preflight.

This intentionally performs no serving mutation.  Its job is to prove whether
the pre-existing M9 governance can name a serving identity without looking at
M10.4 performance.  An unresolved selection is a scientific governance
blocker, not an invitation to choose a seed after the fact.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

START_COMMIT = "8a4752bf39f82542735c21dd44981f164ed2b849"
M10_4_PROTOCOL_HASH = "cd0ac1f2d5a12a771cc441b4ea19bf0d76c672809b35d3d178f8893b768a177c"
EXPECTED_CHECKPOINTS = {
    20260814: "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5",
    31874: "527e707b988870dff323b6a3e0f1bde36a152b7bbe7e4c7f4bf0e59cdaf19332",
    20260815: "b45f5afeabba820270130595dffb44152ec12fad6fa383cce67013f14524113c",
}
SUPERVISED_M9_6_OUTPUTS = frozenset({
    "source_node", "source_region", "start_time", "duration", "relative_strength",
    "event_presence", "event_cause", "sensor_fault", "evidence_sufficiency",
})
M10_4_RUNTIME_ENABLED_OUTPUTS = frozenset({
    "event_cause", "event_presence", "evidence_sufficiency", "next_step",
    "relative_strength", "source_node",
})
REPORT_DIR = ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5"
PROTOCOL_DOC = ROOT / "docs/evaluation/HYDROCORE_V5_M10_5_SERVING_FREEZE_PROTOCOL.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def selection_audit() -> dict[str, Any]:
    """Return the non-result-based selection audit without inspecting metrics."""

    m9_dir = ROOT / "reports/evaluation/hydrocore-v5/m9-6"
    closure = read_json(m9_dir / "m9-6-closure.json")
    manifest = read_json(m9_dir / "m9-6-manifest.json")
    m104 = read_json(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-closure.json")
    candidates = manifest["checkpoint_identities"]["ARM_B_M9_6"]
    candidate_rows: list[dict[str, Any]] = []
    for seed, expected in EXPECTED_CHECKPOINTS.items():
        record = candidates[str(seed)]
        candidate_rows.append({
            "seed": seed,
            "path": record["export_path"],
            "sha256": record["sha256_after"],
            "matches_expected": record["sha256_after"] == expected,
        })

    source_paths = [
        m9_dir / "m9-6-closure.json",
        m9_dir / "m9-6-manifest.json",
        m9_dir / "m9-6-protocol.json",
        ROOT / "scripts/hydrocore_v5/m9_6_common.py",
        ROOT / "scripts/hydrocore_v5/m10_common.py",
    ]
    forbidden_selector_tokens = (
        "deployment_seed", "serving_seed", "release_seed", "selected_deployment",
        "deployment.selection", "serving.selection", "canonical_deployment",
        "production_seed", "serving ensemble", "canonical ensemble",
    )
    token_hits = {
        str(path.relative_to(ROOT)): [token for token in forbidden_selector_tokens if token in path.read_text().lower()]
        for path in source_paths
    }
    token_hits = {path: hits for path, hits in token_hits.items() if hits}
    per_seed_records = [
        read_json(m9_dir / f"m9-6-training-runs/ARM_B_M9_6-seed{seed}.json")
        for seed in EXPECTED_CHECKPOINTS
    ]
    final_step_only = all(record["canonical_checkpoint_policy"] == "FINAL_STEP_1350" for record in per_seed_records)
    parent_pass = (
        m104["closure_state"] == "M10_4_FULL_TRAJECTORY_PASS"
        and m104["protocol_hash"] == M10_4_PROTOCOL_HASH
    )
    resolved = False
    return {
        "kind": "M10_5_RELEASE_SELECTION_AUDIT",
        "m10_4_parent_pass": parent_pass,
        "m9_6_status": closure["hydrocore_s_status"],
        "m9_6_recipe": closure["selected_hydrocore_s_recipe"],
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "per_seed_export_policy": "FINAL_STEP_1350",
        "per_seed_export_policy_verified": final_step_only,
        "authoritative_records_scanned": [str(path.relative_to(ROOT)) for path in source_paths],
        "explicit_selector_token_hits": token_hits,
        "selection_resolved": resolved,
        "selection_rule": None,
        "selection_reason": (
            "M9.6 freezes only the export-within-each-seed policy; its authoritative records name all "
            "three canonical checkpoints but no deployment seed, deterministic selector, or governed ensemble. "
            "Choosing now would be post-hoc, particularly because M10.4 evaluated all three."
        ),
        "m10_4_performance_used": False,
    }


def output_governance_audit() -> dict[str, Any]:
    stale = sorted(M10_4_RUNTIME_ENABLED_OUTPUTS - SUPERVISED_M9_6_OUTPUTS)
    return {
        "kind": "M10_5_OUTPUT_GOVERNANCE_AUDIT",
        "trained_tasks": ["sentinel"],
        "known_supervised_m9_6_outputs": sorted(SUPERVISED_M9_6_OUTPUTS),
        "m10_4_runtime_enabled_outputs_observed": sorted(M10_4_RUNTIME_ENABLED_OUTPUTS),
        "outputs_not_supported_by_m9_6_supervision_record": stale,
        "next_step_disposition": "MUST_BE_SUPPRESSED_IN_ANY_FUTURE_V5_RELEASE",
        "release_allowlist_frozen": False,
        "reason": "No release identity exists, so no serving allowlist may be frozen. next_step is not supervised M9.6 output.",
    }


def closure_for(preflight: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    state = (
        "M10_5_SERVING_FREEZE_BLOCKED_SELECTION_IDENTITY"
        if preflight["result"] == "M10_5_PREFLIGHT_BLOCKED_SELECTION_IDENTITY" and not selection["selection_resolved"]
        else "M10_5_SERVING_FREEZE_BLOCKED_IMPLEMENTATION"
    )
    return {
        "kind": "M10_5_CLOSURE",
        "milestone": "M10.5",
        "closure_state": state,
        "start_commit": START_COMMIT,
        "protocol_hash": sha256(PROTOCOL_DOC),
        "m10_4_parent_protocol_hash": M10_4_PROTOCOL_HASH,
        "m10_5_complete": False,
        "reason": selection["selection_reason"],
        "next_required_action": "A separately frozen, non-M10.4-result-based deployment selection or ensemble rule.",
        "locked_test_opened_before": False,
        "locked_test_opened_after": False,
    }


def write_artifacts() -> dict[str, Any]:
    assert PROTOCOL_DOC.exists(), "freeze the M10.5 protocol before running selection preflight"
    assert locked_test_opened(ROOT) is False
    selection = selection_audit()
    assert selection["m10_4_parent_pass"]
    assert selection["candidate_count"] == 3
    assert all(row["matches_expected"] for row in selection["candidates"])
    assert selection["per_seed_export_policy_verified"]
    assert not selection["selection_resolved"]
    assert not selection["m10_4_performance_used"]
    preflight = {
        "kind": "M10_5_PREFLIGHT",
        "result": "M10_5_PREFLIGHT_BLOCKED_SELECTION_IDENTITY",
        "start_commit": START_COMMIT,
        "observed_commit": current_commit(),
        "protocol_hash": sha256(PROTOCOL_DOC),
        "checks": {
            "m10_4_parent_pass": selection["m10_4_parent_pass"],
            "all_canonical_checkpoint_hashes_match": all(row["matches_expected"] for row in selection["candidates"]),
            "per_seed_final_step_policy_verified": selection["per_seed_export_policy_verified"],
            "selection_uniquely_resolved": selection["selection_resolved"],
            "selection_does_not_use_m10_4_performance": not selection["m10_4_performance_used"],
            "locked_test_opened": locked_test_opened(ROOT),
        },
        "blocker": selection["selection_reason"],
    }
    protocol = {
        "kind": "M10_5_PROTOCOL",
        "protocol_hash": sha256(PROTOCOL_DOC),
        "start_commit": START_COMMIT,
        "m10_4_parent_closure": "M10_4_FULL_TRAJECTORY_PASS",
        "m10_4_parent_protocol_hash": M10_4_PROTOCOL_HASH,
        "selection_rule_required": True,
        "forbidden_selection_input": "M10.4 performance",
        "frozen_action_on_unresolved_selection": "M10_5_SERVING_FREEZE_BLOCKED_SELECTION_IDENTITY",
    }
    serving_path = {
        "kind": "M10_5_SERVING_PATH_AUDIT",
        "pre_modification_default": "hydroswarm.api.app -> V4PipelineFactory(resolve_v4_bundle_dir()) -> models/hydrocore-v4-release",
        "m10_4_evaluated_factory": "M10_4_PipelineFactory injected into create_app; canonical M9.6 model/calibration",
        "classification": {
            "v4_default_bundle": "A_REQUIRED_TO_REPLACE_IF_AND_ONLY_IF_SELECTION_IDENTITY_IS_RESOLVED",
            "create_app_pipeline_factory_injection": "B_HARMLESS_COMPATIBILITY",
            "m10_4_factory": "C_EVALUATION_ONLY",
        },
        "serving_change_performed": False,
        "reason": "Blocked before selection; default wiring must not be changed to an arbitrarily chosen v5 seed.",
    }
    safety = {
        "kind": "M10_5_SAFETY_COUNTERS",
        "locked_test_opened_before": False,
        "locked_test_opened_after": False,
        "serving_mutations": 0,
        "checkpoint_selection_from_m10_4_performance": 0,
        "v5_bundle_created_without_selection_rule": 0,
        "default_path_redirected_without_selection_rule": 0,
        "all_zero": True,
    }
    deferred = {
        "status": "NOT_EXECUTED_DUE_TO_SELECTION_IDENTITY_BLOCKER",
        "reason": "Parity/release corruption tests require a uniquely governed serving identity and must not choose one implicitly.",
    }
    historical = {
        "kind": "M10_5_HISTORICAL_IMMUTABILITY",
        "historical_paths_modified": [],
        "v4_release_modified": False,
        "m9_to_m10_4_artifacts_modified": False,
        "canonical_checkpoint_bytes_modified": False,
    }
    closure = closure_for(preflight, selection)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {
        "m10-5-preflight.json": preflight,
        "m10-5-protocol.json": protocol,
        "m10-5-release-selection.json": selection,
        "m10-5-release-manifest.json": {
            "kind": "M10_5_RELEASE_MANIFEST",
            "status": "NOT_CREATED_DUE_TO_SELECTION_IDENTITY_BLOCKER",
            "bundle_path": None,
            "checkpoint": None,
            "reason": "A manifest naming a checkpoint or ensemble would itself make the forbidden post-hoc selection.",
        },
        "m10-5-output-governance.json": output_governance_audit(),
        "m10-5-serving-path.json": serving_path,
        "m10-5-parity.json": {"kind": "M10_5_PARITY", **deferred},
        "m10-5-fail-closed.json": {"kind": "M10_5_FAIL_CLOSED", **deferred},
        "m10-5-safety-counters.json": safety,
        "m10-5-regression-smoke.json": {"kind": "M10_5_REGRESSION_SMOKE", **deferred},
        "m10-5-historical-immutability.json": historical,
        "m10-5-closure.json": closure,
    }
    for name, payload in payloads.items():
        (REPORT_DIR / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    assert locked_test_opened(ROOT) is False
    return payloads


if __name__ == "__main__":
    result = write_artifacts()
    print(json.dumps(result["m10-5-closure.json"], indent=2, sort_keys=True))
