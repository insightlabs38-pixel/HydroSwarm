"""Execute the non-locked M11.5 validation matrix for the frozen finalist."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/hydrocore_v5")]

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
import run_m11_2_finalist_freeze as m112  # noqa: E402

REPORT_DIR = ROOT / "reports/evaluation/hydrocore-v5/m11/m11-5"
PROTOCOL_DOC = ROOT / "docs/evaluation/HYDROCORE_V5_M11_5_FULL_VALIDATION_PROTOCOL.md"
FREEZE = ROOT / "reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json"
M11_STATUS = ROOT / "reports/evaluation/hydrocore-v5/m11/m11-current-status.json"
M10_STATUS = ROOT / "reports/evaluation/hydrocore-v5/m10/m10-current-status.json"
EXPECTED = m112.EXPECTED
OUTPUTS = m112.OUTPUTS


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def verify_finalist() -> dict[str, Any]:
    freeze = read_json(FREEZE)
    identity = m112.finalist_identity()
    checks = {
        "finalist": freeze.get("finalist_system") == EXPECTED["system"],
        "checkpoint": freeze.get("checkpoint_sha256") == identity["assets"]["checkpoint"]["sha256"] == EXPECTED["checkpoint"],
        "calibration": freeze.get("calibration", {}).get("sha256") == identity["assets"]["calibration"]["sha256"] == EXPECTED["calibration"],
        "calibration_artifact": freeze.get("calibration", {}).get("artifact_hash") == EXPECTED["calibration_artifact"],
        "release_manifest": freeze.get("release_manifest_sha256") == identity["assets"]["release_manifest"]["sha256"] == EXPECTED["manifest"],
        "tasks": freeze.get("trained_tasks") == ["sentinel"], "outputs": freeze.get("runtime_enabled_outputs") == OUTPUTS,
        "tuning_closed": freeze.get("tuning_closed") is True, "identity_violations": not m112.identity_violations(identity),
    }
    return {"checks": checks, "all_checks_pass": all(checks.values()), "freeze_sha256": sha256(FREEZE)}


def preflight() -> dict[str, Any]:
    status, m10 = read_json(M11_STATUS), read_json(M10_STATUS)
    m112_closure = read_json(ROOT / "reports/evaluation/hydrocore-v5/m11/m11-2/m11-2-closure.json")
    identity = verify_finalist()
    checks = {
        "m10_complete": m10.get("m10_complete") is True,
        "m11_1_selected": status.get("m11_1_state") == "M11_1_FINALIST_SELECTED",
        "m11_2_frozen": status.get("m11_2_state") == "M11_2_FINALIST_FROZEN" and m112_closure.get("closure_state") == "M11_2_FINALIST_FROZEN",
        "current_flags": all(status.get(key) is value for key, value in {"finalist_selected": True, "finalist_frozen": True, "tuning_closed": True, "locked_test_opened": False, "locked_evaluation_authorized": False}.items()),
        "next_authorized": status.get("next_authorized_milestone") == "M11.5",
        "finalist_identity": identity["all_checks_pass"], "locked_unopened": locked_test_opened(ROOT) is False,
    }
    return {"kind": "M11_5_PREFLIGHT", "milestone": "M11.5", "code_under_test_commit": current_commit(), "checks": checks,
            "all_checks_pass": all(checks.values()), "locked_test_opened_before": False, "locked_test_opened_after": False}


def evidence_paths() -> list[Path]:
    return [
        FREEZE, ROOT / "reports/evaluation/hydrocore-v5/m11/m11-2/m11-2-reproducibility.json",
        ROOT / "reports/evaluation/hydrocore-v5/m11/m11-2/m11-2-output-governance.json",
        ROOT / "reports/evaluation/hydrocore-v5/m9-final/m9-final-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-gate.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-trajectory-summary.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-safety-counters.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5-completion/m10-5-completion-closure.json",
        ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5/m10-5-fail-closed.json",
    ]


def matrix_definition() -> dict[str, Any]:
    rows = [
        ("A", "Finalist identity / reproducibility", "FRESH_FROZEN_FINALIST_RUN", "exact parity; clean load; no fallback", "M11.2 freeze invariants"),
        ("B", "Predictive / source localization", "CLOSED_EVIDENCE_REUSE", "M9 selected predictor and M10.4 trajectory", "M9 final/M10.4 frozen gates"),
        ("C", "Calibration / actionability", "CLOSED_EVIDENCE_REUSE", "B_DEPTH_AWARE artifact and M10.4 behavior", "M9 coverage floor and M10 fail-closed invariant"),
        ("D", "Robustness / sensor quality", "CLOSED_EVIDENCE_REUSE", "M10.1 development conditions", "finite output and authority invariants"),
        ("E", "Development OOD / topology", "CLOSED_EVIDENCE_REUSE", "M10.1 development-only OOD", "deterministic OOD and fail-closed invariants"),
        ("F", "Scout / active sampling", "CLOSED_EVIDENCE_REUSE", "M10.2 and M10.4 trajectories", "zero Scout safety counters"),
        ("G", "Planning / physical verification", "CLOSED_EVIDENCE_REUSE", "M10.4 physical/safety evidence", "WNTR verification and zero unsafe-plan counters"),
        ("H", "Human approval / actuation safety", "CLOSED_EVIDENCE_REUSE", "M10.4 safety evidence", "zero bypass/actuation counters"),
        ("I", "End-to-end incident trajectory", "CLOSED_EVIDENCE_REUSE", "M10.4 full trajectory", "M10_4_FULL_TRAJECTORY_PASS"),
        ("J", "Serving / release parity", "FRESH_FROZEN_FINALIST_RUN", "M11.2 clean load plus M10.5 serving freeze", "exact parity/no-v4 fallback"),
        ("K", "Fail-closed matrix", "CLOSED_EVIDENCE_REUSE", "M10.5 failure matrix", "classical-safe failure/no v4 fallback"),
        ("L", "Output governance", "FRESH_FROZEN_FINALIST_RUN", "M11.2 output freeze", "exact allowlist and non-authority"),
        ("M", "Known limitations", "CLOSED_EVIDENCE_REUSE", "M11.2 limitations", "descriptive only"),
        ("N", "Software / release quality", "FRESH_FROZEN_FINALIST_RUN", "full pytest, strict self-test, Docker, frontend, static checks", "established release gates"),
    ]
    return {"kind": "M11_5_MATRIX_DEFINITION", "m11_3_m11_4": "UNUSED_RESERVED_SUBSUMED_BY_M10_M11_2", "fresh_seed_namespace": None,
            "rows": [{"row_id": a, "domain": b, "evidence_mode": c, "scope": d, "gate_source": e, "hard_gating": a != "M"} for a, b, c, d, e in rows]}


def run_gate(name: str, command: list[str], *, cwd: Path = ROOT) -> dict[str, Any]:
    started = perf_counter()
    try:
        completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600, check=False)
        output = completed.stdout
        return {"name": name, "command": command, "exit_code": completed.returncode, "passed": completed.returncode == 0,
                "duration_seconds": round(perf_counter() - started, 3), "output_tail": output[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"name": name, "command": command, "exit_code": None, "passed": False, "duration_seconds": round(perf_counter() - started, 3), "error": f"{type(error).__name__}: {error}"}


def software_gates() -> dict[str, Any]:
    gates = [
        run_gate("full_pytest", [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q"]),
        run_gate("pyright", [str(ROOT / ".venv/bin/pyright")]),
        run_gate("ruff_changed_m11_5", [str(ROOT / ".venv/bin/ruff"), "check", "scripts/hydrocore_v5/run_m11_5_full_validation.py", "tests/scientific/test_m11_5_full_validation.py"]),
        run_gate("strict_self_test", [str(ROOT / ".venv/bin/hydroswarm"), "self-test", "--strict"]),
        run_gate("frontend_lint", ["npm", "run", "lint"], cwd=ROOT / "frontend"),
        run_gate("frontend_typecheck", ["npm", "run", "typecheck"], cwd=ROOT / "frontend"),
        run_gate("frontend_test", ["npm", "run", "test"], cwd=ROOT / "frontend"),
        run_gate("frontend_build", ["npm", "run", "build"], cwd=ROOT / "frontend"),
        run_gate("docker_build_strict_self_test", ["docker", "build", "-t", "hydroswarm-m11-5-validation:local", "."]),
    ]
    return {"kind": "M11_5_SOFTWARE_GATES", "gates": gates, "all_required_pass": all(gate["passed"] for gate in gates)}


def reused_results() -> dict[str, dict[str, Any]]:
    m9 = read_json(ROOT / "reports/evaluation/hydrocore-v5/m9-final/m9-final-closure.json")
    ood = read_json(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-closure.json")
    scout = read_json(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-closure.json")
    trajectory = read_json(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-closure.json")
    gate = read_json(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-gate.json")
    safety = read_json(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-safety-counters.json")
    completion = read_json(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5-completion/m10-5-completion-closure.json")
    return {
        "predictive": {"pass": m9["M9_STATUS"] == "CLOSED" and trajectory["closure_state"] == "M10_4_FULL_TRAJECTORY_PASS", "source": relative(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-trajectory-summary.json")},
        "calibration": {"pass": gate["all_checks_pass"] is True, "source": relative(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-gate.json")},
        "robustness": {"pass": ood["guardrails"]["all_outputs_finite"] is True and ood["guardrails"]["no_authority_boundary_regression"] is True, "source": relative(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-closure.json")},
        "ood": {"pass": ood["M10_1_DECISION"] == "LEARNED_OOD_NOT_PROMOTED_DETERMINISTIC_RETAINED", "source": relative(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-closure.json")},
        "scout": {"pass": scout["deterministic_scout_fallback_preserved"] is True and scout["hard_gates_passed"] is True, "source": relative(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-closure.json")},
        "planning": {"pass": trajectory["closure_state"] == "M10_4_FULL_TRAJECTORY_PASS" and safety["counters"]["unverified_plan_surfaced_as_actionable"] == 0, "source": relative(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-safety-counters.json")},
        "end_to_end": {"pass": trajectory["closure_state"] == "M10_4_FULL_TRAJECTORY_PASS", "source": relative(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-closure.json")},
        "fail_closed": {"pass": completion["fail_closed"].startswith("PASS:"), "source": relative(ROOT / "reports/evaluation/hydrocore-v5/m10/m10-5-completion/m10-5-completion-closure.json")},
        "safety": safety["counters"],
    }


def build_artifacts(software: dict[str, Any], output_dir: Path = REPORT_DIR) -> dict[str, dict[str, Any]]:
    if locked_test_opened(ROOT):
        raise RuntimeError("M11.5 must not run after locked evaluation access")
    protocol = {"kind": "M11_5_PROTOCOL", "milestone": "M11.5", "protocol_path": relative(PROTOCOL_DOC), "protocol_sha256": sha256(PROTOCOL_DOC),
                "parent_freeze_path": relative(FREEZE), "parent_freeze_sha256": sha256(FREEZE), "locked_test_prohibition": True,
                "m11_3_m11_4": "UNUSED_RESERVED_SUBSUMED_BY_M10_M11_2", "no_tuning": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m11-5-protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    p, identity, reuse, definition = preflight(), verify_finalist(), reused_results(), matrix_definition()
    clean, output = m112.clean_load_reproducibility(), read_json(ROOT / "reports/evaluation/hydrocore-v5/m11/m11-2/m11-2-output-governance.json")
    rows: list[dict[str, Any]] = []
    outcome = {"A": identity["all_checks_pass"] and clean["all_checks_pass"], "B": reuse["predictive"]["pass"], "C": reuse["calibration"]["pass"], "D": reuse["robustness"]["pass"], "E": reuse["ood"]["pass"], "F": reuse["scout"]["pass"], "G": reuse["planning"]["pass"], "H": reuse["safety"]["human_approval_bypassed"] == 0 and reuse["safety"]["autonomous_actuation_detected"] == 0, "I": reuse["end_to_end"]["pass"], "J": clean["all_checks_pass"], "K": reuse["fail_closed"]["pass"], "L": output["allowlist_matches"] is True and output["next_step_disposition"] == "SUPPRESSED_UNSUPERVISED", "M": True, "N": software["all_required_pass"]}
    for definition_row in definition["rows"]:
        row_id = definition_row["row_id"]
        rows.append({**definition_row, "finalist_identity_verified": verify_finalist()["all_checks_pass"], "result": "PASS" if outcome[row_id] else "FAIL", "status": "DESCRIPTIVE" if row_id == "M" else ("PASS" if outcome[row_id] else "FAIL"), "limitation": "carried forward; non-gating" if row_id == "M" else None})
    hard_rows = [row for row in rows if row["hard_gating"]]
    matrix_pass = p["all_checks_pass"] and all(row["result"] == "PASS" for row in hard_rows)
    counters = {key: reuse["safety"].get(key, 0) for key in ("learned_ood_overrode_deterministic", "learned_scout_selected_sample", "learned_strategist_selected_plan", "inaccessible_sample_selected", "sampled_node_reselected", "sampling_budget_exceeded", "unverified_plan_surfaced_as_actionable", "rejected_plan_surfaced_as_safe", "human_approval_bypassed", "stale_approval_accepted", "autonomous_actuation_detected", "nonfinite_value_reached_decision")}
    counters.update({"silent_v4_fallback": 0 if clean["all_checks_pass"] else 1, "finalist_identity_drift": 0 if identity["all_checks_pass"] else 1, "locked_test_opened": 0, "invariant_failures": sum(1 for row in hard_rows if row["result"] != "PASS")})
    readiness = {"kind": "M11_5_READINESS", "finalist_selected": True, "finalist_frozen": True, "final_selection_record_exists": True, "m11_5_matrix_green": matrix_pass, "tuning_closed": True, "locked_test_unopened": locked_test_opened(ROOT) is False, "m11_6_preconditions_satisfied": matrix_pass, "locked_evaluation_authorized": False, "locked_test_opened": False, "awaiting_explicit_human_authorization": matrix_pass}
    closure_state = "M11_5_FULL_VALIDATION_PASS" if matrix_pass else "M11_5_FULL_VALIDATION_FAIL"
    closure = {"kind": "M11_5_CLOSURE", "milestone": "M11.5", "closure_state": closure_state, "matrix_green": matrix_pass, "locked_test_opened_before": False, "locked_test_opened_after": False, "locked_evaluation_authorized": False, "next_action": "AWAIT_EXPLICIT_HUMAN_AUTHORIZATION_FOR_M11_6" if matrix_pass else "DO_NOT_PROCEED_TO_M11_6", "code_under_test_commit": current_commit()}
    evidence = {"kind": "M11_5_EVIDENCE_MANIFEST", "sources": [{"path": relative(path), "sha256": sha256(path), "locked": False} for path in evidence_paths()], "locked_source_count": 0}
    payloads = {"m11-5-preflight.json": p, "m11-5-finalist-identity.json": identity, "m11-5-matrix-definition.json": definition,
        "m11-5-gate-provenance.json": {"kind": "M11_5_GATE_PROVENANCE", "no_post_hoc_numeric_gates": True, "rows": definition["rows"]}, "m11-5-evidence-manifest.json": evidence,
        "m11-5-predictive.json": reuse["predictive"], "m11-5-calibration.json": reuse["calibration"], "m11-5-robustness.json": reuse["robustness"], "m11-5-ood.json": reuse["ood"], "m11-5-scout.json": reuse["scout"], "m11-5-planning.json": reuse["planning"], "m11-5-end-to-end.json": reuse["end_to_end"], "m11-5-serving.json": clean, "m11-5-fail-closed.json": reuse["fail_closed"], "m11-5-output-governance.json": output, "m11-5-safety-counters.json": counters, "m11-5-limitations.json": m112.limitations(), "m11-5-software-gates.json": software, "m11-5-matrix.json": {"kind": "M11_5_MATRIX", "rows": rows, "matrix_green": matrix_pass}, "m11-5-readiness.json": readiness, "m11-5-closure.json": closure}
    for name, payload in payloads.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if locked_test_opened(ROOT):
        raise RuntimeError("locked-test state changed during M11.5")
    return {"m11-5-protocol.json": protocol, **payloads}


if __name__ == "__main__":
    artifacts = build_artifacts(software_gates())
    print(json.dumps(artifacts["m11-5-closure.json"], indent=2, sort_keys=True))
