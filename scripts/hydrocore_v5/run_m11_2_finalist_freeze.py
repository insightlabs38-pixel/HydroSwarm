"""Freeze and verify the already-selected M11.1 HydroCore-v5 finalist."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA  # noqa: E402
from hydroswarm.runtime.paths import resolve_v5_bundle_dir  # noqa: E402
from hydroswarm.runtime.v5_defaults import V5_RUNTIME_ENABLED_OUTPUTS  # noqa: E402

REPORT_DIR = ROOT / "reports/evaluation/hydrocore-v5/m11/m11-2"
PROTOCOL_DOC = ROOT / "docs/evaluation/HYDROCORE_V5_M11_2_FINALIST_FREEZE_PROTOCOL.md"
PARENT_SELECTION = ROOT / "reports/evaluation/hydrocore-v5/m11/m11-1/final-selection.json"
PARENT_CLOSURE = ROOT / "reports/evaluation/hydrocore-v5/m11/m11-1/m11-1-closure.json"
PARENT_PROTOCOL = ROOT / "docs/evaluation/HYDROCORE_V5_M11_1_FINALIST_SELECTION_PROTOCOL.md"
M10_STATUS = ROOT / "reports/evaluation/hydrocore-v5/m10/m10-current-status.json"
BUNDLE = ROOT / "models/hydrocore-v5-release"
EXPECTED = {
    "system": "HydroCore-v5 M10 frozen release",
    "seed": 20260814,
    "checkpoint": "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5",
    "manifest": "f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34",
    "calibration": "8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d",
    "calibration_artifact": "f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd",
    "m11_1_protocol": "52d911b86b37c7de095643cf02415601e5e1b198cf17c849239b97da8e94264d",
}
OUTPUTS = ["event_cause", "event_presence", "evidence_sufficiency", "relative_strength", "source_node"]
AUTHORITY = {
    "ood": "OODDetector", "scout": "rank_sample_locations", "planner": "generate_response_plans",
    "physical_verification": "WNTR/EPANET", "human_approval_required": True, "autonomous_actuation": False,
}
TRAINING_RECIPE = [
    "CLASSICAL_HYDROCORE_S", "AGE_FIX_ONLY", "EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING",
]
RUNTIME_SOURCES = [
    ROOT / "src/hydroswarm/runtime/v5_defaults.py", ROOT / "src/hydroswarm/runtime/paths.py",
    ROOT / "src/hydroswarm/api/app.py", ROOT / "src/hydroswarm/inference/pipeline.py",
    ROOT / "src/hydroswarm/inference/ood.py", ROOT / "src/hydroswarm/agents/scout.py",
    ROOT / "src/hydroswarm/agents/strategist.py", ROOT / "src/hydroswarm/simulation/verifier.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parent_selection_verification() -> dict[str, Any]:
    selection, closure = read_json(PARENT_SELECTION), read_json(PARENT_CLOSURE)
    checks = {
        "parent_closure": closure.get("closure_state") == "M11_1_FINALIST_SELECTED",
        "one_finalist_selected": closure.get("finalist_selected") is True and selection.get("finalist_selected") is True,
        "parent_not_already_frozen": closure.get("finalist_frozen") is False and selection.get("finalist_frozen") is False,
        "locked_evaluation_not_authorized": selection.get("locked_evaluation_authorized") is False,
        "selected_system": selection.get("selected_finalist_system") == EXPECTED["system"],
        "release_bundle": selection.get("release_bundle_path") == relative(BUNDLE),
        "checkpoint": selection.get("checkpoint_sha256") == EXPECTED["checkpoint"],
        "manifest": selection.get("release_manifest_sha256") == EXPECTED["manifest"],
        "calibration": selection.get("calibration_file_sha256") == EXPECTED["calibration"],
        "calibration_artifact": selection.get("calibration_artifact_hash") == EXPECTED["calibration_artifact"],
        "m11_1_protocol": selection.get("protocol_sha256") == EXPECTED["m11_1_protocol"] == sha256(PARENT_PROTOCOL),
    }
    return {
        "kind": "M11_2_PARENT_SELECTION_VERIFICATION", "parent_selection_path": relative(PARENT_SELECTION),
        "parent_selection_sha256": sha256(PARENT_SELECTION), "parent_closure_path": relative(PARENT_CLOSURE),
        "parent_closure_sha256": sha256(PARENT_CLOSURE), "checks": checks, "all_checks_pass": all(checks.values()),
    }


def finalist_identity() -> dict[str, Any]:
    manifest = read_json(BUNDLE / "runtime_manifest.json")
    model_config = manifest["model_config"]
    return {
        "kind": "M11_2_FINALIST_IDENTITY", "schema_version": 1, "system": EXPECTED["system"],
        "release_bundle": relative(BUNDLE), "model_variant": "small", "selected_seed": manifest["selected_seed"],
        "parameter_count": 4_182_612, "training_recipe": TRAINING_RECIPE,
        "assets": {
            "checkpoint": {"path": relative(BUNDLE / "model.safetensors"), "sha256": sha256(BUNDLE / "model.safetensors")},
            "calibration": {"path": relative(BUNDLE / "calibration.json"), "sha256": sha256(BUNDLE / "calibration.json"),
                            "artifact_hash": manifest["calibration_artifact_hash"], "alpha": 0.1,
                            "grouping": "B_DEPTH_AWARE"},
            "calibration_checksum": {"path": relative(BUNDLE / "calibration.json.sha256"), "sha256": sha256(BUNDLE / "calibration.json.sha256")},
            "release_manifest": {"path": relative(BUNDLE / "runtime_manifest.json"), "sha256": sha256(BUNDLE / "runtime_manifest.json")},
        },
        "model_config": model_config, "model_config_sha256": canonical_hash(model_config),
        "feature_schema": {"fingerprint": manifest["feature_schema_hash"], "runtime_fingerprint": DEFAULT_FEATURE_SCHEMA.fingerprint},
        "feature_semantics": manifest["feature_semantics"], "fusion_config_hash": manifest["fusion_config_hash"],
        "trained_tasks": manifest["trained_tasks"], "runtime_enabled_outputs": manifest["runtime_enabled_outputs"],
        "deterministic_authority": manifest["deterministic_authority"],
        "runtime_governance_source_hashes": {relative(path): sha256(path) for path in RUNTIME_SOURCES},
    }


def identity_violations(identity: dict[str, Any]) -> list[str]:
    expected = finalist_identity()
    fields = {
        "system": (identity["system"], expected["system"]), "selected_seed": (identity["selected_seed"], expected["selected_seed"]),
        "checkpoint_sha256": (identity["assets"]["checkpoint"]["sha256"], expected["assets"]["checkpoint"]["sha256"]),
        "calibration_sha256": (identity["assets"]["calibration"]["sha256"], expected["assets"]["calibration"]["sha256"]),
        "calibration_artifact_hash": (identity["assets"]["calibration"]["artifact_hash"], expected["assets"]["calibration"]["artifact_hash"]),
        "release_manifest_sha256": (identity["assets"]["release_manifest"]["sha256"], expected["assets"]["release_manifest"]["sha256"]),
        "feature_schema": (identity["feature_schema"], expected["feature_schema"]), "fusion_config_hash": (identity["fusion_config_hash"], expected["fusion_config_hash"]),
        "trained_tasks": (identity["trained_tasks"], expected["trained_tasks"]), "runtime_enabled_outputs": (identity["runtime_enabled_outputs"], expected["runtime_enabled_outputs"]),
        "deterministic_authority": (identity["deterministic_authority"], expected["deterministic_authority"]),
        "model_config": (identity["model_config"], expected["model_config"]), "feature_semantics": (identity["feature_semantics"], expected["feature_semantics"]),
    }
    return [name for name, (actual, required) in fields.items() if actual != required]


def preflight(parent: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    status = read_json(M10_STATUS)
    m10_closure = read_json(ROOT / status["authoritative_m10_5_closure_path"])
    manifest = read_json(BUNDLE / "runtime_manifest.json")
    checks = {
        "m10_complete": status.get("m10_complete") is True,
        "m10_serving_freeze": m10_closure.get("closure_state") == "M10_5_SERVING_FREEZE_PASS",
        "m11_1_parent": parent["all_checks_pass"], "release_exists": BUNDLE.is_dir(),
        "checkpoint": identity["assets"]["checkpoint"]["sha256"] == EXPECTED["checkpoint"],
        "selected_seed": identity["selected_seed"] == EXPECTED["seed"],
        "release_manifest": identity["assets"]["release_manifest"]["sha256"] == EXPECTED["manifest"],
        "calibration": identity["assets"]["calibration"]["sha256"] == EXPECTED["calibration"],
        "calibration_artifact": identity["assets"]["calibration"]["artifact_hash"] == EXPECTED["calibration_artifact"],
        "feature_schema": identity["feature_schema"]["fingerprint"] == identity["feature_schema"]["runtime_fingerprint"],
        "feature_semantics": "incident_elapsed" in identity["feature_semantics"] and "M9.6 fixed-age" in identity["feature_semantics"],
        "fusion": identity["fusion_config_hash"] == "fuse_source_probabilities-v1",
        "trained_tasks": identity["trained_tasks"] == ["sentinel"],
        "output_allowlist": identity["runtime_enabled_outputs"] == OUTPUTS == sorted(V5_RUNTIME_ENABLED_OUTPUTS),
        "next_step_suppressed": "next_step" not in identity["runtime_enabled_outputs"],
        "authority": identity["deterministic_authority"] == AUTHORITY,
        "default_serving": resolve_v5_bundle_dir(ROOT) == BUNDLE.resolve(),
        "no_v4_fallback": "v4" not in str(resolve_v5_bundle_dir(ROOT)).lower() and manifest["release_schema_version"] == "hydroswarm-v5-release-v1",
        "locked_unopened": locked_test_opened(ROOT) is False and status.get("locked_test_opened") is False,
    }
    return {"kind": "M11_2_PREFLIGHT", "milestone": "M11.2", "code_under_test_commit": current_commit(), "checks": checks,
            "all_checks_pass": all(checks.values()), "locked_test_opened_before": False, "locked_test_opened_after": False}


def clean_load_reproducibility() -> dict[str, Any]:
    child = """
import json
from pathlib import Path
from hydroswarm.runtime.v5_defaults import V5PipelineFactory, V5_RUNTIME_ENABLED_OUTPUTS, V5_TRAINED_TASKS
root = Path.cwd(); factory = V5PipelineFactory(root / 'models/hydrocore-v5-release', project_root=root)
manifest = factory.manifest
print(json.dumps({'ready': factory.trained_assets_ready, 'model_hash': factory.model_hash,
 'calibration_artifact_hash': None if manifest is None else manifest.get('calibration_artifact_hash'),
 'feature_schema_hash': None if manifest is None else manifest.get('feature_schema_hash'),
 'fusion_config_hash': None if manifest is None else manifest.get('fusion_config_hash'),
 'outputs': sorted(V5_RUNTIME_ENABLED_OUTPUTS), 'trained_tasks': sorted(V5_TRAINED_TASKS),
 'fallback_reason': factory.fallback_reason}, sort_keys=True))
"""
    child_environment = os.environ.copy()
    source_path = str(ROOT / "src")
    child_environment["PYTHONPATH"] = source_path + (os.pathsep + child_environment["PYTHONPATH"] if child_environment.get("PYTHONPATH") else "")
    observed = json.loads(subprocess.check_output([sys.executable, "-c", child], cwd=ROOT, text=True, env=child_environment))
    checks = {
        "clean_process_loaded": observed["ready"] is True and observed["fallback_reason"] is None,
        "checkpoint": observed["model_hash"] == EXPECTED["checkpoint"],
        "calibration_artifact": observed["calibration_artifact_hash"] == EXPECTED["calibration_artifact"],
        "feature_schema": observed["feature_schema_hash"] == DEFAULT_FEATURE_SCHEMA.fingerprint,
        "fusion": observed["fusion_config_hash"] == "fuse_source_probabilities-v1",
        "outputs": observed["outputs"] == OUTPUTS,
        "trained_tasks": observed["trained_tasks"] == ["sentinel"],
    }
    return {"kind": "M11_2_REPRODUCIBILITY", "method": "fresh Python process release load; no calibration fit or performance metric",
            "observed": observed, "checks": checks, "all_checks_pass": all(checks.values())}


def negative_identity_tests(identity: dict[str, Any]) -> dict[str, Any]:
    mutations: dict[str, tuple[list[str], Any]] = {
        "checkpoint_sha": (["assets", "checkpoint", "sha256"], "0" * 64),
        "calibration_sha": (["assets", "calibration", "sha256"], "0" * 64),
        "calibration_artifact": (["assets", "calibration", "artifact_hash"], "mismatch"),
        "release_manifest": (["assets", "release_manifest", "sha256"], "0" * 64),
        "feature_schema": (["feature_schema", "fingerprint"], "mismatch"),
        "fusion": (["fusion_config_hash"], "mismatch"), "selected_seed": (["selected_seed"], 1),
        "trained_tasks": (["trained_tasks"], ["sentinel", "scout"]),
        "add_next_step": (["runtime_enabled_outputs"], OUTPUTS + ["next_step"]),
        "enable_learned_ood": (["deterministic_authority", "ood"], "learned_ood_category"),
        "enable_learned_scout": (["deterministic_authority", "scout"], "learned_scout"),
        "enable_learned_strategist": (["deterministic_authority", "planner"], "learned_strategist"),
        "disable_human_approval": (["deterministic_authority", "human_approval_required"], False),
        "enable_autonomous_actuation": (["deterministic_authority", "autonomous_actuation"], True),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, (path, value) in mutations.items():
        candidate = copy.deepcopy(identity)
        target: dict[str, Any] = candidate
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        violations = identity_violations(candidate)
        results[name] = {"accepted_as_same_finalist": not violations, "violations": violations}
    return {"kind": "M11_2_IDENTITY_NEGATIVE_TESTS", "tests": results,
            "all_mutations_rejected": all(not result["accepted_as_same_finalist"] for result in results.values())}


def historical_immutability(changes: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    base = "330bff8be18433e441b9382e14860ff02a19b2f5"
    if changes is None:
        raw_changes = subprocess.check_output(
            ["git", "diff", "--no-renames", "--name-status", base, "HEAD"], cwd=ROOT, text=True,
        ).splitlines()
        change_entries = [tuple(line.split("\t", maxsplit=1)) for line in raw_changes]
    else:
        change_entries = changes
    changed = [path for _, path in change_entries]
    protected_prefixes = (
        "models/hydrocore-v5-release/", "models/hydrocore-v4-release/",
        *(f"reports/evaluation/hydrocore-v5/m{milestone}-" for milestone in range(10)),
        "reports/evaluation/hydrocore-v5/m10/", "reports/evaluation/hydrocore-v5/m11/m11-1/",
        *(f"docs/evaluation/HYDROCORE_V5_M{milestone}_" for milestone in range(10)),
        "docs/evaluation/HYDROCORE_V5_M10_", "docs/evaluation/HYDROCORE_V5_M11_1_",
        "src/hydroswarm/preprocessing/schema.py",
        *tuple(relative(path) for path in RUNTIME_SOURCES),
    )
    violations = [path for path in changed if path.startswith(protected_prefixes)]
    later_milestone_additions = [
        path for status, path in change_entries
        if status == "A" and path.startswith((
            "docs/evaluation/HYDROCORE_V5_M11_5_", "scripts/hydrocore_v5/run_m11_5_",
            "tests/scientific/test_m11_5_", "reports/evaluation/hydrocore-v5/m11/m11-5/",
            "reports/evaluation/hydrocore-v5/m11/m11-current-status.json",
        ))
    ]
    return {"kind": "M11_2_HISTORICAL_IMMUTABILITY", "baseline_commit": base, "changed_since_baseline": changed,
            "protected_path_violations": violations, "historical_artifacts_unchanged": not violations,
            "later_milestone_additions": later_milestone_additions,
            "no_system_tuning_or_runtime_change": not violations}


def authority_freeze(identity: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "M11_2_AUTHORITY_FREEZE", "deterministic_ood": "OODDetector", "learned_ood": "NON_AUTHORITATIVE",
            "deterministic_scout": "rank_sample_locations", "learned_scout": "NON_AUTHORITATIVE",
            "deterministic_planning": "generate_response_plans", "learned_strategist": "NON_AUTHORITATIVE",
            "physical_authority": "WNTR/EPANET", "human_approval_required": True, "autonomous_actuation": False,
            "matches_release_manifest": identity["deterministic_authority"] == AUTHORITY}


def limitations() -> dict[str, Any]:
    return {"kind": "M11_2_LIMITATIONS", "limitations": [
        "M10.4 selected-plan-vs-NO_ACTION Gate E was vacuous because NO_ACTION was absent from the bounded candidate set.",
        "Deterministic active sampling modestly improved localization but did not change the final approved action in M10.4.",
        "Development unseen-topology evidence is limited; unsupported topology suppresses calibration/actionability.",
        "M9.6 fixed-age versus M10.4 incident-elapsed unobserved-age behavior remains unresolved and frozen.",
        "Learned OOD, Scout, and Strategist were not promoted.",
    ]}


def build_artifacts(output_dir: Path = REPORT_DIR) -> dict[str, dict[str, Any]]:
    if locked_test_opened(ROOT):
        raise RuntimeError("M11.2 must not run after locked evaluation access")
    protocol = {"kind": "M11_2_PROTOCOL", "milestone": "M11.2", "protocol_path": relative(PROTOCOL_DOC),
                "protocol_sha256": sha256(PROTOCOL_DOC), "parent_selection": relative(PARENT_SELECTION),
                "parent_m11_1_protocol_sha256": EXPECTED["m11_1_protocol"], "selected_finalist": EXPECTED["system"],
                "locked_test_prohibition": True, "tuning_closed_on_success": True,
                "closure_vocabulary": ["M11_2_FINALIST_FROZEN", "M11_2_FINALIST_FREEZE_BLOCKED_SELECTION_IDENTITY", "M11_2_FINALIST_FREEZE_BLOCKED_IDENTITY_DRIFT", "M11_2_FINALIST_FREEZE_BLOCKED_REQUIRES_SYSTEM_CHANGE", "M11_2_FINALIST_FREEZE_BLOCKED_REPRODUCIBILITY"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m11-2-protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    parent, identity = parent_selection_verification(), finalist_identity()
    p = preflight(parent, identity)
    reproducibility, negative, history = clean_load_reproducibility(), negative_identity_tests(identity), historical_immutability()
    if not (p["all_checks_pass"] and reproducibility["all_checks_pass"] and negative["all_mutations_rejected"] and history["historical_artifacts_unchanged"] and history["no_system_tuning_or_runtime_change"]):
        raise RuntimeError("M11_2_FINALIST_FREEZE_BLOCKED_IDENTITY_DRIFT")
    authority = authority_freeze(identity)
    output = {"kind": "M11_2_OUTPUT_GOVERNANCE", "trained_tasks": identity["trained_tasks"], "runtime_enabled_outputs": identity["runtime_enabled_outputs"],
              "suppressed_output": "next_step", "next_step_disposition": "SUPPRESSED_UNSUPERVISED", "allowlist_matches": identity["runtime_enabled_outputs"] == OUTPUTS}
    freeze = {"kind": "HYDROCORE_V5_FINALIST_FREEZE", "schema_version": 1, "milestone": "M11.2", "finalist_system": EXPECTED["system"],
        "parent_m11_1_selection_path": relative(PARENT_SELECTION), "parent_m11_1_selection_sha256": parent["parent_selection_sha256"],
        "m11_1_protocol_sha256": EXPECTED["m11_1_protocol"], "m11_2_protocol_sha256": protocol["protocol_sha256"], "selected_seed": identity["selected_seed"],
        "checkpoint_path": identity["assets"]["checkpoint"]["path"], "checkpoint_sha256": identity["assets"]["checkpoint"]["sha256"],
        "architecture": {"variant": identity["model_variant"], "parameter_count": identity["parameter_count"], "model_config_sha256": identity["model_config_sha256"]},
        "training_recipe": TRAINING_RECIPE, "calibration": identity["assets"]["calibration"], "release_bundle": identity["release_bundle"],
        "release_manifest_sha256": identity["assets"]["release_manifest"]["sha256"], "feature_schema": identity["feature_schema"],
        "feature_semantics": identity["feature_semantics"], "fusion_config_hash": identity["fusion_config_hash"], "trained_tasks": identity["trained_tasks"],
        "runtime_enabled_outputs": identity["runtime_enabled_outputs"], "suppressed_untrained_outputs": ["next_step", "ood_category", "sample_node", "information_gain", "candidate_reduction", "should_continue_sampling", "plan_validity", "plan_value", "exposure_proxy", "pressure_risk_proxy", "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy"],
        "authority": authority, "default_serving_identity": "V5PipelineFactory(resolve_v5_bundle_dir())", "no_v4_fallback": True,
        "finalist_identity_manifest_path": "m11-2-finalist-identity.json", "authority_freeze_path": "m11-2-authority-freeze.json",
        "output_governance_path": "m11-2-output-governance.json", "reproducibility_path": "m11-2-reproducibility.json",
        "known_limitations_path": "m11-2-limitations.json", "source_commit": current_commit(), "historical_artifacts_unchanged": True,
        "locked_test_opened": False, "finalist_selected": True, "finalist_frozen": True, "tuning_closed": True, "locked_evaluation_authorized": False}
    closure = {"kind": "M11_2_CLOSURE", "milestone": "M11.2", "closure_state": "M11_2_FINALIST_FROZEN", "finalist": EXPECTED["system"],
               "finalist_selected": True, "finalist_frozen": True, "tuning_closed": True, "locked_evaluation_authorized": False,
               "locked_test_opened_before": False, "locked_test_opened_after": False, "next_authorized_milestone": "M11.5 (not executed)", "code_under_test_commit": current_commit()}
    current = {"kind": "HYDROCORE_V5_M11_CURRENT_STATUS", "m11_1_state": "M11_1_FINALIST_SELECTED", "m11_2_state": "M11_2_FINALIST_FROZEN",
               "current_finalist": EXPECTED["system"], "freeze_certificate": "reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json", "finalist_selected": True,
               "finalist_frozen": True, "tuning_closed": True, "locked_evaluation_authorized": False, "locked_test_opened": False, "next_authorized_milestone": "M11.5"}
    payloads = {"m11-2-preflight.json": p, "m11-2-parent-selection-verification.json": parent, "m11-2-finalist-identity.json": identity,
        "m11-2-authority-freeze.json": authority, "m11-2-output-governance.json": output, "m11-2-reproducibility.json": reproducibility,
        "m11-2-identity-negative-tests.json": negative, "m11-2-historical-immutability.json": history, "m11-2-limitations.json": limitations(),
        "finalist-freeze.json": freeze, "m11-2-closure.json": closure}
    for name, payload in payloads.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir.parent / "m11-current-status.json").write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    if locked_test_opened(ROOT):
        raise RuntimeError("locked-test state changed during M11.2")
    return {"m11-2-protocol.json": protocol, **payloads, "m11-current-status.json": current}


if __name__ == "__main__":
    records = build_artifacts()
    print(json.dumps(records["m11-2-closure.json"], indent=2, sort_keys=True))
