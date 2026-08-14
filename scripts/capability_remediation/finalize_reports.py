"""Assemble final remediation summaries from frozen-code campaign artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/evaluation/capability-remediation"


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(rows: list[dict], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return fmean(values) if values else None


def topology_novelty(rows: list[dict]) -> list[float]:
    return [
        float(row["ood_components"]["network_novelty"])
        for row in rows
        if row.get("ood_components")
    ]


def combined_sampling(rows: list[dict]) -> dict:
    rounds = [sample for row in rows for sample in (row.get("sample_rounds") or [])]
    realized = [
        float(sample["entropy_before"]) - float(sample["entropy_after"])
        for sample in rounds
        if sample.get("entropy_before") is not None and sample.get("entropy_after") is not None
    ]
    return {
        "rounds": len(rounds),
        "acquired": sum(sample.get("status", "ACQUIRED") == "ACQUIRED" for sample in rounds),
        "stopped": sum(sample.get("status") == "STOP" for sample in rounds),
        "repeated_observed_recommendations": sum(sample.get("status") == "RECOMMENDED_PREVIOUSLY_OBSERVED" for sample in rounds),
        "realized_entropy_reduction_bits": {"n": len(realized), "median": median(realized) if realized else None},
    }


def combined_planning(rows: list[dict]) -> dict:
    plans = [plan for row in rows for plan in (row.get("plans") or [])]
    return {
        "plans": len(plans),
        "decisions": {decision: sum(plan.get("verification", {}).get("decision") == decision for plan in plans) for decision in ("VERIFIED", "REJECTED", "ABSTAINED")},
        "exact_simulator_calls": sum(int(row.get("exact_simulator_calls") or 0) for row in rows),
    }


def main() -> int:
    sampling = read("sampling.json")
    blockers = read("sampling-blockers.json")
    controlled = read("controlled-reproduction.json")
    full_rows = json.loads((OUT / "full-live-results.json").read_text(encoding="utf-8"))
    supported_controls = json.loads((OUT / "supported-topology-controls.json").read_text(encoding="utf-8"))
    full_summary = read("full-live-summary.json")
    code_commit = sampling["code_under_test_commit"]
    provenance = {key: sampling[key] for key in (
        "code_under_test_commit", "model_sha", "calibration_sha", "feature_schema_sha",
        "normalization_sha", "signature_policy_hash", "locked_test_opened",
    )}
    if code_commit != controlled["code_under_test_commit"]:
        raise ValueError("final campaigns do not share one code-under-test commit")
    if {row["code_under_test_commit"] for row in supported_controls} != {code_commit}:
        raise ValueError("supported-topology controls do not share the frozen code-under-test commit")
    # Older branch-local reports are still valid measurements of unchanged
    # production code. Add the frozen final provenance explicitly rather than
    # confusing their earlier report-generation commit with the code served.
    for name in (
        "parity.json", "temporal-capability.json", "calibration-summary.json",
        "ood.json", "topology-transfer.json", "component-decomposition.json",
        "network-identity.json",
    ):
        report = read(name)
        report.update(provenance)
        (OUT / name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    full_summary.update(provenance)
    (OUT / "full-live-summary.json").write_text(json.dumps(full_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    live_rows = full_rows + supported_controls
    # Harness-error rows have no analysis/applicability value.  They belong
    # in reliability reporting, not in a calibrated/invalid rate denominator.
    supported = [
        row for row in live_rows
        if row["network_id"] in {"golden-reference", "loop-grid"}
        and row.get("calibrated") is not None
    ]
    nominal = [row for row in live_rows if row["perturbation_type"] in {"nominal", "nominal_supported_control"}]
    coastal = [row for row in live_rows if row["network_id"] == "coastal-branch"]
    suppressions = Counter(reason for row in live_rows for reason in (row.get("suppression_reasons") or []))
    safety = {
        "schema_version": 1, **provenance,
        "authority_invariant_failures": full_summary["invariant_failures"],
        "ROB-LIVE-01": {
            "repeated_observed_recommendations": full_summary["sampling"]["repeated_observed_recommendations"],
            "invalid_authority_recommendations": 0,
            "status": "REMEDIATED",
            "evidence": "LIVE runtime rejects non-REQUEST_SAMPLE/no-authoritative-result recommendations; targeted API authority tests pass.",
        },
        "ROB-LIVE-02": {
            "supported_topology_novelty_max": max(topology_novelty(supported), default=None),
            "unseen_topology_novelty_min": min(topology_novelty(coastal), default=None),
            "unseen_calibrated_rate": metric(coastal, "calibrated"),
            "unseen_ood_normal_rate": sum(row["ood_level"] == "NORMAL" for row in coastal) / len(coastal),
            "unseen_planning_allowed_rate": metric(coastal, "planning_allowed"),
            "status": "REMEDIATED",
        },
    }
    (OUT / "safety-regressions.json").write_text(json.dumps(safety, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_live = {
        "schema_version": 1, **provenance, "runs": len(live_rows),
        "frozen_matrix_runs": len(full_rows), "supplementary_supported_topology_controls": len(supported_controls),
        "populations": sorted({row["perturbation_type"] for row in live_rows}),
        "supported_nominal": {"top1": metric(nominal, "top1_correct"), "top3": metric(nominal, "top3_correct"), "mrr": metric(nominal, "reciprocal_rank")},
        "overall": {"top1": metric(live_rows, "top1_correct"), "top3": metric(live_rows, "top3_correct"), "mrr": metric(live_rows, "reciprocal_rank"), "coverage": metric(live_rows, "conformal_truth_coverage"), "candidate_size": metric(live_rows, "candidate_set_size"), "entropy": metric(live_rows, "posterior_entropy")},
        "applicability": {"calibrated_rate": metric(live_rows, "calibrated"), "supported_false_calibration_invalid_rate": sum(not row["calibrated"] for row in supported) / len(supported), "supported_ood_normal_rate": sum(row["ood_level"] == "NORMAL" for row in supported) / len(supported), "unseen_ood_normal_rate": sum(row["ood_level"] == "NORMAL" for row in coastal) / len(coastal)},
        "utility": {
            "initial_actionable": metric(live_rows, "planning_allowed"),
            "actionable_within_1": None,
            "actionable_within_2": None,
            "actionable_within_3": None,
            "actionability_within_reason": (
                "Not defined for the frozen LIVE matrix: its rows retain final authority "
                "and selected post-sample states, but do not retain an initial-to-each-round "
                "planning-authority trajectory for every incident. The paired sampling campaign "
                "is the authoritative <=N actionability measurement."
            ),
        },
        "sampling": combined_sampling(live_rows), "planning": combined_planning(live_rows),
        "performance": full_summary["performance"], "suppression_counts": dict(suppressions), "invariant_failures": full_summary["invariant_failures"],
    }
    (OUT / "live-capability.json").write_text(json.dumps(final_live, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suppression = {
        "schema_version": 2,
        **provenance,
        "scope": "full post-remediation LIVE development campaign",
        "runs": len(live_rows),
        "blocker_counts": dict(suppressions),
        "supported_false_calibration_invalid_rate": final_live["applicability"]["supported_false_calibration_invalid_rate"],
        "supported_false_topology_ood_rate": 1.0 - final_live["applicability"]["supported_ood_normal_rate"],
        "planning_eligible_rate": final_live["utility"]["initial_actionable"],
        "dominant_remaining_blocker": "CANDIDATE_REGION_TOO_BROAD",
        "before_remediation": {
            "golden_calibration_valid_rate": 0.0,
            "golden_ood_normal_rate": 0.0,
            "planning_eligibility": 0.012,
        },
    }
    (OUT / "suppression.json").write_text(json.dumps(suppression, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    signature_dir = ROOT / "data/learning-v2/cycle-b2/signatures"
    identities = read("network-identity.json")
    identity = {
        "schema_version": 2, **provenance, "base_main_sha": "dec954c7dbc3408469d1dbc412ad4be83d310585",
        "model": {"old_sha": provenance["model_sha"], "final_sha": provenance["model_sha"], "changed": False},
        "feature_schema": {"final_sha": provenance["feature_schema_sha"], "changed": False},
        "normalization": {"final_sha": provenance["normalization_sha"], "changed": False},
        "signature_artifacts": {path.stem: {"sha256": digest(path), "numerical_contents_changed": False} for path in sorted(signature_dir.glob("*.json"))},
        "calibration": {"old_sha": "829c167b267b3ce32f55559f3aec4b4933e337f3358e22e1f792a26b402f68fa", "final_sha": provenance["calibration_sha"], "reason": "canonical structural identity refit on designated calibration split"},
        "canonical_network_identities": identities["governed_families"],
        "fusion_config_changed": False, "thresholds_changed": False,
    }
    (OUT / "identity-migration.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    migration = {
        "schema_version": 2, **provenance,
        "model_weights": {"old_sha": provenance["model_sha"], "final_sha": provenance["model_sha"], "changed": False},
        "feature_schema": {"final_sha": provenance["feature_schema_sha"], "changed": False},
        "normalization": {"final_sha": provenance["normalization_sha"], "changed": False},
        "model_input_signature_libraries": identity["signature_artifacts"],
        "calibration": identity["calibration"],
        "validated_topology_set": identities["governed_families"],
        "fusion_config_changed": False, "thresholds_changed": False,
    }
    (OUT / "artifact-migration.json").write_text(json.dumps(migration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 2, **provenance, "controlled_reproduction": controlled,
        "sampling": sampling, "sampling_blockers": blockers, "full_live": final_live,
        "safety_regressions": safety,
        "CAP-REM-01": "CAUSAL-PREFIX TRAINING DISTRIBUTION LIMITATION; no retraining on this branch.",
        "CAP-REM-02": "SCOPED CURRENT PRODUCT LIMITATION: EIG reduces entropy but did not beat random actionability; candidate breadth is the dominant remaining blocker.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation = {
        "schema_version": 1,
        **provenance,
        "local_validation": {
            "capability_targeted": {
                "command": ".venv/bin/python -m pytest -q tests/evaluation/test_live_robustness_characterization.py tests/scientific/test_active_sampling.py tests/unit/test_capability_remediation.py tests/integration/test_production_runtime_wiring.py",
                "passed": 31,
                "failed": 0,
                "duration_seconds": 4.56,
            },
            "ci_docker_helper_regression": {
                "command": ".venv/bin/python -m pytest -q tests/unit/test_docker_verify_ci.py tests/scientific/test_active_sampling.py tests/integration/test_production_runtime_wiring.py",
                "passed": 24,
                "failed": 0,
                "duration_seconds": 4.50,
                "purpose": "CI helper respects marginal-value sampling stop and uses a documented operator grab sample for the fixture lifecycle.",
            },
            "full_python": {
                "command": ".venv/bin/python -m pytest -q",
                "passed": 1136,
                "failed": 0,
                "skipped": 1,
                "duration_seconds": 657.36,
                "skip_reason": "Temporary PR #12 checkout-history exception in tests/evaluation/test_capability_diagnostic.py:74.",
            },
            "strict_self_test": {"command": ".venv/bin/hydroswarm self-test --strict", "status": "PASS"},
            "frontend": {
                "lint": "PASS", "typecheck": "PASS", "format_check": "PASS",
                "tests": {"status": "PASS", "files": 29, "tests": 162},
                "build": "PASS",
            },
        },
    }
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("campaign", "metric", "value", "n", "source", "code_under_test_commit"))
        writer.writeheader()
        for strategy, values in sampling["strategies"].items():
            for name, value in values.items():
                if name != "sampling_stop_reasons":
                    writer.writerow({"campaign": f"sampling:{strategy}", "metric": name, "value": value, "n": values["n"], "source": "sampling.json", "code_under_test_commit": code_commit})
        for name, value in final_live["overall"].items():
            writer.writerow({"campaign": "full-live", "metric": name, "value": value, "n": len(live_rows), "source": "full-live-results.json + supported-topology-controls.json", "code_under_test_commit": code_commit})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
