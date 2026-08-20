"""Create the additive, non-locked M11.1 finalist-selection record.

This is a synthesis guard, not an evaluator: it consumes only closed tracked
records, performs no training/calibration/threshold work, and refuses any
evidence manifest containing a locked-split token.
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
from hydroswarm.runtime.paths import resolve_v5_bundle_dir  # noqa: E402

REPORT_DIR = ROOT / "reports/evaluation/hydrocore-v5/m11/m11-1"
PROTOCOL_DOC = ROOT / "docs/evaluation/HYDROCORE_V5_M11_1_FINALIST_SELECTION_PROTOCOL.md"
V5_BUNDLE = ROOT / "models/hydrocore-v5-release"
V4_BUNDLE = ROOT / "models/hydrocore-v4-release"
V5_CHECKPOINT = "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5"
V5_MANIFEST = "f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34"
V5_CALIBRATION = "8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d"
V5_CALIBRATION_ARTIFACT = "f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd"
V5_OUTPUTS = ["event_cause", "event_presence", "evidence_sufficiency", "relative_strength", "source_node"]
LOCKED_TOKENS = ("locked_final_test", "locked_topology_test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _no_locked_source(value: str) -> bool:
    lowered = value.lower()
    return not any(token in lowered for token in LOCKED_TOKENS)


def authoritative_sources() -> list[Path]:
    """Closed, tracked, development-only records permitted by the protocol."""
    return [
        ROOT / "reports/results/v4/architecture-freeze.json",
        ROOT / "reports/evaluation/hydrocore-v5/m0-baseline.json",
        ROOT / "reports/evaluation/hydrocore-v5/m9-final/m9-final-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m9-8/m9-8-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-0/m10-0-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-gate.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-trajectory-summary.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5-completion/m10-5-completion-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-current-status.json",
    ]


def preflight() -> dict[str, Any]:
    status = read_json(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-current-status.json")
    closure = read_json(ROOT / status["authoritative_m10_5_closure_path"])
    closure_5a = read_json(ROOT / status["m10_5a_closure_path"])
    closure_5b = read_json(ROOT / status["m10_5b_closure_path"])
    manifest = read_json(V5_BUNDLE / "runtime_manifest.json")
    app_source = (ROOT / "src/hydroswarm/api/app.py").read_text(encoding="utf-8")
    v5_source = (ROOT / "src/hydroswarm/runtime/v5_defaults.py").read_text(encoding="utf-8")
    paths_source = (ROOT / "src/hydroswarm/runtime/paths.py").read_text(encoding="utf-8")
    expected_authority = {
        "ood": "OODDetector", "scout": "rank_sample_locations", "planner": "generate_response_plans",
        "physical_verification": "WNTR/EPANET", "human_approval_required": True, "autonomous_actuation": False,
    }
    checks = {
        "m10_complete": status.get("m10_complete") is True,
        "m10_5_completion_pass": closure.get("closure_state") == "M10_5_SERVING_FREEZE_PASS",
        "m10_5a_selection_frozen": closure_5a.get("closure_state") == "M10_5A_DEPLOYMENT_SELECTION_FROZEN",
        "m10_5b_calibration_materialized": closure_5b.get("closure_state") == "M10_5B_CALIBRATION_ARTIFACT_MATERIALIZED",
        "v5_bundle_exists": V5_BUNDLE.is_dir(),
        "selected_checkpoint_identity": sha256(V5_BUNDLE / "model.safetensors") == V5_CHECKPOINT == manifest.get("model_sha256"),
        "release_manifest_identity": sha256(V5_BUNDLE / "runtime_manifest.json") == V5_MANIFEST,
        "calibration_file_identity": sha256(V5_BUNDLE / "calibration.json") == V5_CALIBRATION == manifest.get("calibration_file_sha256"),
        "calibration_artifact_identity": manifest.get("calibration_artifact_hash") == V5_CALIBRATION_ARTIFACT,
        "default_runtime_v5": resolve_v5_bundle_dir(ROOT) == V5_BUNDLE.resolve() and "V5PipelineFactory(DEFAULT_V5_RELEASE_BUNDLE_DIR" in app_source,
        "no_silent_v4_fallback": "never fall back to v4" in paths_source and "never selects the historical v4 release" in v5_source,
        "runtime_enabled_output_allowlist": manifest.get("runtime_enabled_outputs") == V5_OUTPUTS,
        "next_step_suppressed": "next_step" not in manifest.get("runtime_enabled_outputs", []),
        "learned_ood_non_authoritative": closure.get("learned_ood") == "NON_AUTHORITATIVE_DETERMINISTIC_OODDetector",
        "learned_scout_non_authoritative": closure.get("learned_scout") == "NON_AUTHORITATIVE_DETERMINISTIC_rank_sample_locations",
        "learned_strategist_non_authoritative": closure.get("learned_strategist") == "NON_AUTHORITATIVE_DETERMINISTIC_generate_response_plans",
        "deterministic_authority": manifest.get("deterministic_authority") == expected_authority,
        "feature_semantics_unchanged": "M10.4 tested" in closure.get("feature_semantics", "") and "M9.6 fixed-age" in closure.get("feature_semantics", ""),
        "locked_test_unopened": locked_test_opened(ROOT) is False and status.get("locked_test_opened") is False,
    }
    return {
        "kind": "M11_1_PREFLIGHT", "milestone": "M11.1", "code_under_test_commit": current_commit(),
        "checks": checks, "all_checks_pass": all(checks.values()), "locked_test_opened_before": False,
        "locked_test_opened_after": False,
    }


def candidate_eligibility() -> dict[str, Any]:
    v4_manifest = read_json(V4_BUNDLE / "runtime_manifest.json")
    v4_freeze = read_json(ROOT / "reports/results/v4/architecture-freeze.json")
    v5 = {
        "name": "HydroCore-v5 M10 frozen release", "version": "hydrocore-v5-release-v1",
        "eligible": True, "checkpoint_sha256": V5_CHECKPOINT, "calibration_file_sha256": V5_CALIBRATION,
        "calibration_artifact_hash": V5_CALIBRATION_ARTIFACT,
        "feature_schema_hash": read_json(V5_BUNDLE / "runtime_manifest.json")["feature_schema_hash"],
        "release_bundle": _relative(V5_BUNDLE), "release_manifest_sha256": V5_MANIFEST,
        "runtime_authority": "deterministic OOD/Scout/planning; WNTR/EPANET verification; human approval; no autonomous actuation",
        "upstream_closure": "M10_5_SERVING_FREEZE_PASS",
        "eligibility_reason": "Complete M10-selected, materialized, default-serving v5 identity with reproducible release bytes.",
        "disqualification_reason": None,
    }
    v4 = {
        "name": "HydroCore-v4 frozen incumbent", "version": v4_manifest["architecture_version"], "eligible": True,
        "checkpoint_sha256": v4_manifest["model_sha256"], "calibration_file_sha256": sha256(V4_BUNDLE / "calibration.json"),
        "calibration_artifact_hash": v4_manifest["calibration_artifact_hash"], "feature_schema_hash": v4_manifest["feature_schema_hash"],
        "release_bundle": _relative(V4_BUNDLE), "release_manifest_sha256": sha256(V4_BUNDLE / "runtime_manifest.json"),
        "runtime_authority": "deterministic OOD/Scout/planning; WNTR/EPANET verification; human approval; no autonomous actuation",
        "upstream_closure": v4_freeze["status"],
        "eligibility_reason": "Historical frozen incumbent has a selected checkpoint, calibration, authority policy, and tracked reproducible release bundle.",
        "disqualification_reason": None,
    }
    excluded = [
        {"name": "HydroCore-M capacity variant", "reason": "M9 final closure: NOT_PROMOTED; no meaningful capacity gain."},
        {"name": "HydroCore-L capacity variant", "reason": "M9 final closure: unauthorized."},
        {"name": "Continuous-time variants", "reason": "M9 final closure: temporal architecture search closed with classical model retained."},
        {"name": "M10.2 Scout-refit checkpoints", "reason": "M10.2 closure: learned Scout not promoted; runtime authority unchanged."},
        {"name": "M10.3 Strategist-refit checkpoints", "reason": "M10.3 closure: blocked/rejected; broader retraining would be required."},
        {"name": "Learned OOD/fusion variants", "reason": "M10.1 closure: learned OOD not promoted; deterministic OOD retained."},
    ]
    return {"kind": "M11_1_CANDIDATE_ELIGIBILITY", "eligible_candidates": [v5, v4], "excluded_candidates": excluded,
            "candidate_set_is_system_level": True, "experimental_checkpoint_promotion_forbidden": True}


def evidence_manifest() -> dict[str, Any]:
    sources = authoritative_sources()
    rows = [{"path": _relative(path), "sha256": sha256(path), "locked_source": False} for path in sources]
    if not all(_no_locked_source(row["path"]) for row in rows):
        raise ValueError("locked source prohibited from M11.1 evidence manifest")
    return {"kind": "M11_1_EVIDENCE_MANIFEST", "permitted_only": True, "locked_source_count": 0,
            "sources": rows, "forbidden_sources": ["locked final evaluation", "locked topology evaluation", "future M11 evidence", "newly tuned variants"]}


def comparison(eligibility: dict[str, Any]) -> dict[str, Any]:
    v5, v4 = eligibility["eligible_candidates"]
    gates = {
        "A_predictive_incident_intelligence": {"v5": "PASS: M9 final selected causal predictor; M10.4 source trajectory closed PASS.", "v4": "PASS: frozen historical calibrated incident-intelligence evidence."},
        "B_robustness_fail_closed": {"v5": "PASS: M10 release corruption/mismatch fail-closed evidence; development topology/OOD evidence retained.", "v4": "PASS: frozen v4 deterministic fallback and release self-test evidence."},
        "C_end_to_end_decision_utility": {"v5": "PASS: M10.4 full trajectory PASS with exact verification and deterministic sampling/planning.", "v4": "PASS: historical end-to-end authority evidence; no v5-era trajectory matrix."},
        "D_safety_authority": {"v5": "PASS: deterministic authorities, WNTR/EPANET, human approval, no actuation.", "v4": "PASS: same deterministic authority boundary."},
        "E_release_readiness": {"v5": "PASS: M10.5 current default immutable bundle and no-v4-fallback matrix.", "v4": "PASS: reproducible incumbent bundle, but not the current normal serving identity."},
        "F_complexity_scientific_justification": {"v5": "PASS: selected S model; capacity/learned-component negative results retained rather than bypassing deterministic controls.", "v4": "PASS: simpler historical incumbent; retained as reference."},
        "G_known_limitations": {"v5": "DISCLOSED: retained M10 limitations listed in m11-1-limitations.json.", "v4": "DISCLOSED: historical calibration/provenance limitations remain in v4 freeze record."},
    }
    return {
        "kind": "M11_1_COMPARISON", "selection_method": "frozen gate-based rubric; no weighted score", "gates": gates,
        "tie_break_applied": "Most recent complete non-locked system-level governed development evidence on normal serving identity.",
        "selected": v5["name"], "not_selected": v4["name"],
        "decision_reason": "Both frozen system candidates clear the safety/reproducibility gates. The frozen tie-break selects v5 because M9 selected its predictor and M10.0-M10.5 supplied the complete, closed, non-locked system-level development validation and immutable current serving identity. No M11.1 metric, tuning, or locked evidence was used.",
    }


def limitations() -> dict[str, Any]:
    return {"kind": "M11_1_LIMITATIONS", "limitations": [
        "M10.4 selected-plan-vs-NO_ACTION Gate E was vacuous: NO_ACTION was absent from the bounded generated candidate set.",
        "Deterministic active sampling modestly improved localization but did not change the final approved action in the M10.4 population.",
        "Development unseen-topology evidence is limited; unsupported/unseen topology appropriately suppresses calibration/actionability.",
        "M10.4 tested incident-elapsed unobserved-age runtime behavior differs from M9.6 fixed-age training behavior; it is intentionally frozen, not resolved.",
        "Learned OOD was not promoted.", "Learned Scout was not promoted.", "Learned Strategist was not justified or promoted.",
    ]}


def build_artifacts(output_dir: Path = REPORT_DIR) -> dict[str, dict[str, Any]]:
    if locked_test_opened(ROOT):
        raise RuntimeError("M11.1 must not run after locked evaluation access")
    p = preflight()
    if not p["all_checks_pass"]:
        raise RuntimeError("M11_1_FINALIST_SELECTION_BLOCKED_UPSTREAM_IDENTITY")
    protocol = {"kind": "M11_1_PROTOCOL", "milestone": "M11.1", "protocol_path": _relative(PROTOCOL_DOC),
                "protocol_sha256": sha256(PROTOCOL_DOC), "candidate_set": ["HydroCore-v5 M10 frozen release", "HydroCore-v4 frozen incumbent"],
                "locked_test_prohibition": True, "closure_vocabulary": ["M11_1_FINALIST_SELECTED", "M11_1_FINALIST_SELECTION_BLOCKED_UPSTREAM_IDENTITY", "M11_1_FINALIST_SELECTION_BLOCKED_REQUIRES_SYSTEM_CHANGE", "M11_1_FINALIST_SELECTION_BLOCKED_AMBIGUOUS"]}
    eligibility = candidate_eligibility()
    evidence = evidence_manifest()
    result = comparison(eligibility)
    carried_limitations = limitations()
    selected = eligibility["eligible_candidates"][0]
    final_selection = {"kind": "HYDROCORE_V5_FINAL_SELECTION", "schema_version": 1, "milestone": "M11.1",
        "selected_finalist_system": selected["name"], "release_bundle_path": selected["release_bundle"],
        "checkpoint_sha256": selected["checkpoint_sha256"], "calibration_file_sha256": selected["calibration_file_sha256"],
        "calibration_artifact_hash": selected["calibration_artifact_hash"], "release_manifest_sha256": selected["release_manifest_sha256"],
        "runtime_enabled_outputs": V5_OUTPUTS, "trained_tasks": ["sentinel"], "authority_policy": selected["runtime_authority"],
        "candidate_set": [candidate["name"] for candidate in eligibility["eligible_candidates"]], "candidate_eligibility_path": "m11-1-candidate-eligibility.json",
        "candidate_eligibility_decisions": [
            {"name": candidate["name"], "eligible": candidate["eligible"], "reason": candidate["eligibility_reason"]}
            for candidate in eligibility["eligible_candidates"]
        ],
        "selection_evidence_manifest_path": "m11-1-evidence-manifest.json", "protocol_sha256": protocol["protocol_sha256"],
        "known_limitations_path": "m11-1-limitations.json", "m10_parent_identity": "M10_5_SERVING_FREEZE_PASS", "source_commit": current_commit(),
        "locked_test_opened": False, "locked_test_opened_before": False, "locked_test_opened_after": False,
        "finalist_selected": True, "finalist_frozen": False, "locked_evaluation_authorized": False}
    closure = {"kind": "M11_1_CLOSURE", "milestone": "M11.1", "closure_state": "M11_1_FINALIST_SELECTED",
               "finalist": selected["name"], "finalist_selected": True, "finalist_frozen": False,
               "locked_evaluation_authorized": False, "locked_test_opened_before": False, "locked_test_opened_after": False,
               "next_authorized_milestone": "M11.2 (not executed)", "code_under_test_commit": current_commit()}
    payloads = {"m11-1-preflight.json": p, "m11-1-protocol.json": protocol, "m11-1-candidate-eligibility.json": eligibility,
        "m11-1-evidence-manifest.json": evidence, "m11-1-comparison.json": result, "m11-1-limitations.json": carried_limitations,
        "final-selection.json": final_selection, "m11-1-closure.json": closure}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if locked_test_opened(ROOT):
        raise RuntimeError("locked-test state changed during M11.1")
    return payloads


if __name__ == "__main__":
    artifacts = build_artifacts()
    print(json.dumps(artifacts["m11-1-closure.json"], indent=2, sort_keys=True))
