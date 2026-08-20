"""Milestone 9.5R decision stage: reads `run_m9_5r_source_representative.py`'s
canonical calibration/development rows, fits the frozen SplitConformalCalibrator
ONCE per arm x seed at the single primary support (20/source), runs the
CORRECTED sanity/implementation gate (Section 11 -- provenance/implementation
invariants ONLY, never a historical-reproduction requirement), the CURRENT
control gate (Section 14), the INTERLEAVED confirmation gate (Section 13),
the candidate-set-size guard (Section 15), source-conditional/loop-grid-J1
diagnostics (Section 16/17), frozen-checkpoint predictive sanity (Section
18), and assigns M9_5R_DECISION (Section 19) per the FROZEN decision logic
recorded in m9-5r-protocol.json.

Reads (never regenerates):
  reports/evaluation/hydrocore-v5/m9-5r/m9-5r-protocol.json
  reports/evaluation/hydrocore-v5/m9-5r/m9-5r-manifest.json
  reports/evaluation/hydrocore-v5/m9-5r/m9-5r-canonical-calibration.jsonl
  reports/evaluation/hydrocore-v5/m9-5r/m9-5r-representativeness-audit.json

Writes:
  m9-5r-sanity-gate.json, m9-5r-calibration-results.json,
  m9-5r-control-arm.json, m9-5r-source-conditional.json,
  m9-5r-loop-grid-j1.json, m9-5r-candidate-set-analysis.json,
  m9-5r-predictive-sanity.json, m9-5r-guardrails.json, m9-5r-summary.md,
  m9-5r-closure.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import m9_5r_common as m5r  # noqa: E402
from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator, classify_runtime_condition  # noqa: E402
from run_m9_0a_evaluate import _build_libraries, _library_for  # noqa: E402


# ---------------------------------------------------------------------------
# Loading / filtering.
# ---------------------------------------------------------------------------


def _load_canonical() -> list[dict[str, Any]]:
    rows = []
    with m5r.M9_5R_CANONICAL_CALIBRATION_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def _known_families(arm: str) -> tuple[str, ...]:
    return m5r.ARM_A_KNOWN_FAMILIES if arm == "ARM_A" else m5r.ARM_B2_KNOWN_FAMILIES


def _cal_rows(rows: list[dict[str, Any]], arm: str, seed: int, family: str | None = None) -> list[dict[str, Any]]:
    out = [r for r in rows if r["split"] == "calibration" and r["arm"] == arm and r["predictor_seed"] == seed]
    if family is not None:
        out = [r for r in out if r["family"] == family]
    return out


def _dev_rows(rows: list[dict[str, Any]], arm: str, seed: int, family: str | None = None) -> list[dict[str, Any]]:
    out = [r for r in rows if r["split"] == "development" and r["arm"] == arm and r["predictor_seed"] == seed]
    if family is not None:
        out = [r for r in out if r["family"] == family]
    return out


def _incident_count(rows: list[dict[str, Any]]) -> int:
    return len({(r["source_node"], r["generator_seed"]) for r in rows})


# ---------------------------------------------------------------------------
# Section 6/12: fit the UNMODIFIED SplitConformalCalibrator ONCE per arm x
# seed at the single primary support (no nested-level loop -- there is only
# one support condition in M9.5R).
# ---------------------------------------------------------------------------


def _fit_calibrator(rows: list[dict[str, Any]], arm: str, seed: int) -> SplitConformalCalibrator:
    examples: list[CalibrationExample] = []
    for family in sorted(_known_families(arm)):
        for r in _cal_rows(rows, arm, seed, family):
            examples.append(CalibrationExample(
                probabilities=tuple(r["probabilities"]), true_index=r["true_index"], condition=r["condition"], network_id=r["network_id"],
            ))
    return SplitConformalCalibrator.fit(
        examples, alpha=m5r.ALPHA, minimum_group_size=m5r.MINIMUM_GROUP_SIZE,
        model_hash=f"m9-5r-{arm}-seed{seed}-support{m5r.PRIMARY_SUPPORT}", feature_schema_hash="n/a",
        dataset_manifest_hash=f"m9-5r-{arm}-seed{seed}-calibration_m9_5r-pool",
    )


def _apply(calibrator: SplitConformalCalibrator, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        source, group, _scores = calibrator.selection(condition=r["condition"], network_id=r["network_id"])
        candidate = calibrator.candidate_set(r["probabilities"], condition=r["condition"], network_id=r["network_id"])
        covered = r["true_index"] in candidate
        out.append({
            **r, "calibration_group_source": source, "calibration_group_key": group,
            "candidate_set_size": len(candidate), "candidate_set_indices": list(candidate),
            "candidate_covered": bool(covered),
        })
    return out


def _cov_summary(applied_rows: list[dict[str, Any]], n_nodes: int | None = None) -> dict[str, Any]:
    if not applied_rows:
        return {}
    sizes = [r["candidate_set_size"] for r in applied_rows]
    out = {
        "n": len(applied_rows),
        "marginal_coverage": statistics.fmean(r["candidate_covered"] for r in applied_rows),
        "mean_candidate_set_size": statistics.fmean(sizes),
        "median_candidate_set_size": statistics.median(sizes),
        "p90_candidate_set_size": float(np.percentile(sizes, 90)),
        "singleton_rate": statistics.fmean(1 if s == 1 else 0 for s in sizes),
        "by_maturity": {
            bucket: {
                "coverage": statistics.fmean(r["candidate_covered"] for r in applied_rows if r["depth_bucket"] == bucket) if any(r["depth_bucket"] == bucket for r in applied_rows) else None,
                "mean_set_size": statistics.fmean(r["candidate_set_size"] for r in applied_rows if r["depth_bucket"] == bucket) if any(r["depth_bucket"] == bucket for r in applied_rows) else None,
            }
            for bucket in ("EARLY", "MID", "MATURE")
        },
    }
    if n_nodes:
        out["normalized_mean_candidate_set_size"] = out["mean_candidate_set_size"] / n_nodes
        out["full_set_rate"] = statistics.fmean(1 if s >= n_nodes else 0 for s in sizes)
    return out


# ---------------------------------------------------------------------------
# Section 11: CORRECTED sanity/implementation gate -- provenance/
# implementation invariants ONLY (A-L). Deliberately does NOT check any
# particular numerical/coverage outcome or historical-pattern reproduction.
# ---------------------------------------------------------------------------


def _sanity_gate(
    manifest: dict[str, Any], audit: dict[str, Any], rows: list[dict[str, Any]],
    calibrators: dict[tuple[str, int], SplitConformalCalibrator],
    node_ids_by_arm_family: dict[tuple[str, str], tuple[str, ...]],
    resubstitution: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    # A. checkpoint SHA identity passes.
    checks["A_checkpoint_sha_identity"] = all(
        manifest["checkpoint_identities"][arm][str(seed)]["sha256_before"] == manifest["checkpoint_identities"][arm][str(seed)]["sha256_after"]
        for arm in ("ARM_A", "ARM_B2") for seed in m5r.SEEDS
    )

    # B. calibrator class identity matches M9.5 (same import source: hydroswarm.calibration.conformal).
    checks["B_calibrator_class_identity_matches_m9_5"] = (
        SplitConformalCalibrator.__module__ == "hydroswarm.calibration.conformal"
        and SplitConformalCalibrator.__name__ == "SplitConformalCalibrator"
        and all(type(c).__name__ == "SplitConformalCalibrator" for c in calibrators.values())
    )

    # C. alpha == 0.1.
    checks["C_alpha_equals_0_1"] = m5r.ALPHA == 0.1 and all(c.artifact.alpha == 0.1 for c in calibrators.values())

    # D. nonconformity score implementation identity matches (score = 1 - p[true]).
    sample = rows[:2000] if len(rows) > 2000 else rows
    checks["D_nonconformity_score_implementation_identity_matches"] = all(
        m5r.relative_close(r["nonconformity_score"], 1.0 - float(r["probabilities"][r["true_index"]])) for r in sample
    )

    # E. grouping construction identity matches (network_id == f"{family}:{bucket}", B_DEPTH_AWARE).
    checks["E_grouping_construction_identity_matches"] = all(
        r["network_id"] == f"{r['family']}:{r['depth_bucket']}" for r in sample
    )

    # F. maturity/depth mapping identity matches (EARLY/MID/MATURE per m9_4_common's frozen depth tuples).
    checks["F_maturity_depth_mapping_identity_matches"] = all(
        r["depth_bucket"] == m5r.depth_bucket_of(r["depth"]) for r in sample
    )

    # G. candidate-set inclusion rule matches (reused unmodified calibrator.candidate_set()).
    checks["G_candidate_set_inclusion_rule_matches"] = all(
        hasattr(c, "candidate_set") and c.candidate_set.__func__ is SplitConformalCalibrator.candidate_set for c in calibrators.values()
    )

    # H. source-node ordering aligns predictor outputs and true-source labels.
    h_ok = True
    for r in sample:
        node_ids = node_ids_by_arm_family.get((r["arm"], r["family"]))
        if node_ids is None or r["true_index"] >= len(node_ids) or node_ids[r["true_index"]] != r["source_node"]:
            h_ok = False
            break
    checks["H_source_node_ordering_correct"] = h_ok

    # I. calibration/development split disjointness passes.
    checks["I_calibration_development_split_disjointness_passes"] = all(
        audit["families"][family]["checks"]["seed_disjoint_calibration_vs_development"]
        and audit["families"][family]["checks"]["no_scenario_id_overlap"]
        for family in m5r.TRAINED_FAMILIES
    )

    # J. all source nodes represented.
    checks["J_all_source_nodes_represented"] = all(
        audit["families"][family]["checks"]["all_sources_in_calibration"]
        and audit["families"][family]["checks"]["all_sources_in_development"]
        for family in m5r.TRAINED_FAMILIES
    )

    # K. all outputs finite.
    checks["K_all_outputs_finite"] = all(r["all_finite"] for r in rows)

    # L. resubstitution diagnostic numerically plausible (no gross implementation failure).
    # Generous, predeclared threshold (>=0.5): rules out a gross bug (e.g. mislabeled
    # indices, broken quantile lookup), NOT a performance target -- see governing
    # Section 11 ("must not require any particular performance direction").
    resub_values = [v["in_sample_coverage"] for v in resubstitution["per_arm_seed_family"].values()]
    checks["L_resubstitution_diagnostic_numerically_plausible"] = all(
        v is not None and v == v and 0.0 <= v <= 1.0 for v in resub_values
    ) and all(v >= 0.5 for v in resub_values)

    all_pass = all(checks.values())
    return {
        "checks": checks, "M9_5R_SANITY_GATE": "PASS" if all_pass else "FAIL",
        "note": (
            "Corrected Section-11 sanity gate: checks ONLY implementation/provenance invariants. Does NOT "
            "require reproducing M9.4's or M9.5's coverage pattern, and does NOT require any particular "
            "performance direction or specific numerical calibration result."
        ),
    }


def _resubstitution_diagnostic(rows: list[dict[str, Any]], calibrators: dict[tuple[str, int], SplitConformalCalibrator]) -> dict[str, Any]:
    out: dict[str, Any] = {"RESUBSTITUTION_DIAGNOSTIC_ONLY": True, "per_arm_seed_family": {}}
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m5r.SEEDS:
            calibrator = calibrators[(arm, seed)]
            for family in _known_families(arm):
                cal_subset = _cal_rows(rows, arm, seed, family)
                applied = _apply(calibrator, cal_subset)
                out["per_arm_seed_family"][f"{arm}|{seed}|{family}"] = {
                    "in_sample_coverage": _cov_summary(applied).get("marginal_coverage"),
                    "n": len(cal_subset),
                }
    out["note"] = "In-sample (resubstitution) coverage on the SAME support=20 calibration pool the calibrator was fit on -- used only to rule out a gross implementation bug, never as an operational validity metric."
    return out


# ---------------------------------------------------------------------------
# Section 12: primary calibration evaluation (per arm x seed x family).
# ---------------------------------------------------------------------------


def _calibration_evaluation(
    rows: list[dict[str, Any]], calibrators: dict[tuple[str, int], SplitConformalCalibrator],
    applied_at_primary: dict[tuple[str, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    n_nodes_by_family = {f: len(m5r.full_junction_list(f, m5r.ALL_FAMILY_LOADERS[f])) for f in m5r.TRAINED_FAMILIES}
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        for seed in m5r.SEEDS:
            out[arm][str(seed)] = {}
            for family in _known_families(arm):
                cal_subset = _cal_rows(rows, arm, seed, family)
                dev_subset = [r for r in applied_at_primary[(arm, seed)] if r["family"] == family]
                cov = _cov_summary(dev_subset, n_nodes_by_family[family])
                cov["applicability_rate"] = 1.0 if dev_subset else 0.0
                cov["independent_calibration_incident_count"] = _incident_count(cal_subset)
                cov["independent_development_incident_count"] = _incident_count(dev_subset)
                cov["row_count"] = len(dev_subset)
                cov["EARLY_coverage"] = cov.get("by_maturity", {}).get("EARLY", {}).get("coverage")
                cov["MID_coverage"] = cov.get("by_maturity", {}).get("MID", {}).get("coverage")
                cov["MATURE_coverage"] = cov.get("by_maturity", {}).get("MATURE", {}).get("coverage")
                cov["all_finite"] = all(r["all_finite"] for r in dev_subset)
                out[arm][str(seed)][family] = cov
    return out


# ---------------------------------------------------------------------------
# Section 16/17: source-conditional + loop-grid J1/J7/J8 diagnostics.
# ---------------------------------------------------------------------------


def _predictive_row_metrics(r: dict[str, Any]) -> dict[str, float]:
    probs = r["probabilities"]
    truth = r["true_index"]
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    rank = order.index(truth) + 1
    top1 = 1.0 if order[0] == truth else 0.0
    top3 = 1.0 if truth in order[:3] else 0.0
    mrr = 1.0 / rank
    p_true = max(float(probs[truth]), 1e-12)
    nll = -np.log(p_true)
    brier = sum((p - (1.0 if i == truth else 0.0)) ** 2 for i, p in enumerate(probs))
    ent = -sum(p * np.log(p) for p in probs if p > 0)
    return {"top1": top1, "top3": top3, "mrr": mrr, "rank": float(rank), "nll": float(nll), "brier": float(brier), "entropy": float(ent)}


def _source_conditional(rows: list[dict[str, Any]], applied_at_primary: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        for seed in m5r.SEEDS:
            out[arm][str(seed)] = {}
            applied = applied_at_primary[(arm, seed)]
            for family in _known_families(arm):
                out[arm][str(seed)][family] = {}
                fam_applied = [r for r in applied if r["family"] == family]
                sources = sorted({r["source_node"] for r in fam_applied})
                cal_rows_fam = _cal_rows(rows, arm, seed, family)
                for source in sources:
                    src_rows = [r for r in fam_applied if r["source_node"] == source]
                    cal_src_rows = [r for r in cal_rows_fam if r["source_node"] == source]
                    metrics = [_predictive_row_metrics(r) for r in src_rows]
                    out[arm][str(seed)][family][source] = {
                        "n_calibration_incidents": _incident_count(cal_src_rows),
                        "n_development_incidents": _incident_count(src_rows),
                        "marginal_coverage": statistics.fmean(r["candidate_covered"] for r in src_rows),
                        "EARLY_coverage": statistics.fmean(r["candidate_covered"] for r in src_rows if r["depth_bucket"] == "EARLY") if any(r["depth_bucket"] == "EARLY" for r in src_rows) else None,
                        "MID_coverage": statistics.fmean(r["candidate_covered"] for r in src_rows if r["depth_bucket"] == "MID") if any(r["depth_bucket"] == "MID" for r in src_rows) else None,
                        "MATURE_coverage": statistics.fmean(r["candidate_covered"] for r in src_rows if r["depth_bucket"] == "MATURE") if any(r["depth_bucket"] == "MATURE" for r in src_rows) else None,
                        "mean_candidate_set_size": statistics.fmean(r["candidate_set_size"] for r in src_rows),
                        "top1": statistics.fmean(m["top1"] for m in metrics),
                        "mrr": statistics.fmean(m["mrr"] for m in metrics),
                        "mean_true_source_rank": statistics.fmean(m["rank"] for m in metrics),
                    }
    return out


def _loop_grid_j1(rows: list[dict[str, Any]], applied_at_primary: dict[tuple[str, int], list[dict[str, Any]]], node_ids_by_arm_family: dict[tuple[str, str], tuple[str, ...]]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_seed": {}}
    for seed in m5r.SEEDS:
        applied = [r for r in applied_at_primary[("ARM_B2", seed)] if r["family"] == "loop-grid"]
        node_ids = node_ids_by_arm_family[("ARM_B2", "loop-grid")]
        j1_rows = [r for r in applied if r["source_node"] == "J1"]
        metrics = [_predictive_row_metrics(r) for r in j1_rows]
        confusion = {"J1->J7": 0, "J1->J8": 0, "J7->J1": 0, "J8->J1": 0}
        for r in applied:
            truth = r["source_node"]
            pred_idx = int(np.argmax(r["probabilities"]))
            predicted = node_ids[pred_idx] if pred_idx < len(node_ids) else None
            for pair_key, (a, b) in zip(confusion, (("J1", "J7"), ("J1", "J8"), ("J7", "J1"), ("J8", "J1"))):
                if truth == a and predicted == b:
                    confusion[pair_key] += 1
        out["per_seed"][str(seed)] = {
            "j1_top1": statistics.fmean(m["top1"] for m in metrics) if metrics else None,
            "j1_mrr": statistics.fmean(m["mrr"] for m in metrics) if metrics else None,
            "j1_mean_true_source_rank": statistics.fmean(m["rank"] for m in metrics) if metrics else None,
            "j1_median_true_source_rank": statistics.median(m["rank"] for m in metrics) if metrics else None,
            "j1_marginal_coverage": statistics.fmean(r["candidate_covered"] for r in j1_rows) if j1_rows else None,
            "j1_mean_candidate_set_size": statistics.fmean(r["candidate_set_size"] for r in j1_rows) if j1_rows else None,
            "confusion_counts": confusion,
            "n_j1_development_incidents": _incident_count(j1_rows),
        }
    out["note"] = "M9.5R development_m9_5r J1/J7/J8 behavior at the single primary support=20 condition, for descriptive comparison against M9.4/M9.5's findings. Not a gate; training/calibration are not changed based on this."
    return out


# ---------------------------------------------------------------------------
# Section 15: candidate-set-size guard (same rule M9.5 used).
# ---------------------------------------------------------------------------


def _candidate_set_analysis(calibration_evaluation: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_arm_seed_family": {}, "pathological_full_set_behavior_detected": False}
    pathological = []
    for arm in calibration_evaluation:
        for seed in calibration_evaluation[arm]:
            for family, data in calibration_evaluation[arm][seed].items():
                key = f"{arm}|{seed}|{family}"
                out["per_arm_seed_family"][key] = {
                    "mean_candidate_set_size": data.get("mean_candidate_set_size"),
                    "median_candidate_set_size": data.get("median_candidate_set_size"),
                    "p90_candidate_set_size": data.get("p90_candidate_set_size"),
                    "normalized_mean_candidate_set_size": data.get("normalized_mean_candidate_set_size"),
                    "singleton_rate": data.get("singleton_rate"),
                    "full_set_rate": data.get("full_set_rate"),
                }
                if (data.get("full_set_rate") or 0) > m5r.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD:
                    pathological.append(key)
    out["pathological_full_set_behavior_detected"] = bool(pathological)
    out["pathological_cells"] = pathological
    out["threshold_used"] = m5r.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD
    return out


# ---------------------------------------------------------------------------
# Section 13/14: primary gates.
# ---------------------------------------------------------------------------


def _interleaved_confirmation_gate(calibration_evaluation: dict[str, Any]) -> dict[str, Any]:
    per_cell: dict[str, Any] = {}
    all_pass = True
    for seed in m5r.SEEDS:
        for family in m5r.TRAINED_FAMILIES:
            data = calibration_evaluation["ARM_B2"][str(seed)][family]
            cov = data.get("marginal_coverage")
            ok = cov is not None and cov >= m5r.OPERATIONAL_COVERAGE_FLOOR
            all_pass = all_pass and ok
            per_cell[f"ARM_B2|{family}|{seed}"] = {"marginal_coverage": cov, "passes_operational_floor_0_85": ok}
    return {"passed": all_pass, "all_9_cells_pass": all_pass, "per_family_seed_coverage": per_cell}


def _current_control_gate(calibration_evaluation: dict[str, Any]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    all_pass = True
    for seed in m5r.SEEDS:
        data = calibration_evaluation["ARM_A"][str(seed)]["golden-reference"]
        cov = data.get("marginal_coverage")
        ok = cov is not None and cov >= m5r.OPERATIONAL_COVERAGE_FLOOR
        all_pass = all_pass and ok
        per_seed[str(seed)] = {"marginal_coverage": cov, "passes_operational_floor_0_85": ok}
    return {"all_3_pass": all_pass, "per_seed_coverage": per_seed}


# ---------------------------------------------------------------------------
# Section 19: FROZEN decision logic (per m9-5r-protocol.json's
# decision_logic.evaluation_order -- must not be altered post-hoc).
# ---------------------------------------------------------------------------


def _decide(
    representativeness_passed: bool, sanity_gate: dict[str, Any], current_control: dict[str, Any],
    interleaved_gate: dict[str, Any], candidate_set: dict[str, Any],
) -> tuple[str, str, str]:
    sanity_passed = sanity_gate["M9_5R_SANITY_GATE"] == "PASS"
    if not representativeness_passed or not sanity_passed:
        return (
            "D", m5r.DECISION_NAMES["D"],
            "The representativeness audit or an actual implementation/provenance sanity invariant failed "
            "(NOT merely because new coverage differs from old coverage) -- M9.5R cannot be scientifically "
            "interpreted until this is resolved under a separately governed milestone.",
        )
    if not current_control["all_3_pass"]:
        return (
            "C", m5r.DECISION_NAMES["C"],
            "ARM_A/CURRENT (the calibration-pipeline control) fails the adequate-support calibration control "
            "on golden-reference despite 80 independent fresh calibration incidents, matched fresh "
            "development, and passed representativeness/sanity preconditions -- this points to a broader "
            "calibration/data-pipeline issue, not something specific to INTERLEAVED.",
        )
    if not interleaved_gate["all_9_cells_pass"]:
        return (
            "B", m5r.DECISION_NAMES["B"],
            "CURRENT control passes, but at least one ARM_B2/INTERLEAVED trained-family/seed cell falls "
            "below the 0.85 marginal-coverage floor on this fresh population -- INTERLEAVED remains "
            "predictively strong per M9.4, but its calibration is not independently confirmed by M9.5R.",
        )
    if candidate_set["pathological_full_set_behavior_detected"]:
        return (
            "E", m5r.DECISION_NAMES["E"],
            "Coverage passes in all required CURRENT and INTERLEAVED cells, but only through pathological "
            "candidate-set behavior (near-full-graph candidate sets) -- coverage is not actionable evidence "
            "of valid calibration here.",
        )
    return (
        "A", m5r.DECISION_NAMES["A"],
        "Representativeness audit passed, corrected sanity/implementation gate passed, CURRENT control "
        "passes all 3 seeds, INTERLEAVED passes all 9 trained-family/seed cells, and the candidate-set guard "
        "passes (no pathological full-set behavior) -- M9.5's favorable calibration evidence is "
        "independently confirmed under a fresh, disjoint population.",
    )


def main() -> int:
    locked_before = m5r.assert_locked_test_closed()
    assert m5r.M9_5R_PROTOCOL_PATH.exists(), "protocol freeze artifact must exist -- run write_m9_5r_protocol.py first"
    protocol = json.loads(m5r.M9_5R_PROTOCOL_PATH.read_text())
    manifest = json.loads(m5r.M9_5R_MANIFEST_PATH.read_text())
    audit = json.loads(m5r.M9_5R_REPRESENTATIVENESS_AUDIT_PATH.read_text())

    print("loading canonical calibration/development rows...", flush=True)
    rows = _load_canonical()
    print(f"loaded {len(rows)} rows", flush=True)

    print("fitting calibrators (single primary support=20 condition, arm x seed)...", flush=True)
    calibrators: dict[tuple[str, int], SplitConformalCalibrator] = {}
    applied_at_primary: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m5r.SEEDS:
            calibrator = _fit_calibrator(rows, arm, seed)
            calibrators[(arm, seed)] = calibrator
            dev_subset = _dev_rows(rows, arm, seed)
            applied_at_primary[(arm, seed)] = _apply(calibrator, dev_subset)

    print("building libraries for node_ids (source-ordering sanity check, loop-grid J1)...", flush=True)
    libraries = _build_libraries()
    node_ids_by_arm_family = {
        (arm, family): tuple(_library_for(libraries, family, arm).node_ids)
        for arm in ("ARM_A", "ARM_B2") for family in _known_families(arm)
    }

    print("computing resubstitution diagnostic...", flush=True)
    resubstitution = _resubstitution_diagnostic(rows, calibrators)

    print("running CORRECTED Section 11 sanity/implementation gate...", flush=True)
    sanity_gate = _sanity_gate(manifest, audit, rows, calibrators, node_ids_by_arm_family, resubstitution)
    m5r.M9_5R_SANITY_GATE_PATH.write_text(json.dumps(sanity_gate, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"M9_5R_SANITY_GATE = {sanity_gate['M9_5R_SANITY_GATE']}", flush=True)

    print("computing primary calibration evaluation (Section 12)...", flush=True)
    calibration_evaluation = _calibration_evaluation(rows, calibrators, applied_at_primary)
    calibration_results = {
        "primary_support": m5r.PRIMARY_SUPPORT, "per_arm_seed_family": calibration_evaluation,
        "resubstitution_diagnostic_only": resubstitution,
    }
    m5r.M9_5R_CALIBRATION_RESULTS_PATH.write_text(json.dumps(calibration_results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("evaluating INTERLEAVED confirmation gate (Section 13)...", flush=True)
    interleaved_gate = _interleaved_confirmation_gate(calibration_evaluation)

    print("evaluating CURRENT control gate (Section 14)...", flush=True)
    current_control = _current_control_gate(calibration_evaluation)
    m5r.M9_5R_CONTROL_ARM_PATH.write_text(json.dumps(current_control, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing candidate-set-size guard (Section 15)...", flush=True)
    candidate_set_analysis = _candidate_set_analysis(calibration_evaluation)
    m5r.M9_5R_CANDIDATE_SET_ANALYSIS_PATH.write_text(json.dumps(candidate_set_analysis, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing source-conditional diagnostics (Section 16)...", flush=True)
    source_conditional = _source_conditional(rows, applied_at_primary)
    m5r.M9_5R_SOURCE_CONDITIONAL_PATH.write_text(json.dumps(source_conditional, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing loop-grid J1/J7/J8 diagnostic (Section 17)...", flush=True)
    loop_grid_j1 = _loop_grid_j1(rows, applied_at_primary, node_ids_by_arm_family)
    m5r.M9_5R_LOOP_GRID_J1_PATH.write_text(json.dumps(loop_grid_j1, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing Section 18 predictive sanity metrics (descriptive)...", flush=True)
    predictive_sanity: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        predictive_sanity[arm] = {}
        for family in _known_families(arm):
            per_seed_metrics = []
            for seed in m5r.SEEDS:
                fam_rows = [r for r in applied_at_primary[(arm, seed)] if r["family"] == family]
                metrics = [_predictive_row_metrics(r) for r in fam_rows]
                per_seed_metrics.append({k: statistics.fmean(m[k] for m in metrics) for k in ("top1", "top3", "mrr", "nll", "brier", "entropy")})
            predictive_sanity[arm][family] = {
                metric: statistics.fmean(s[metric] for s in per_seed_metrics) for metric in ("top1", "top3", "mrr", "nll", "brier", "entropy")
            }
    m5r.M9_5R_PREDICTIVE_SANITY_PATH.write_text(json.dumps(predictive_sanity, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    guardrails = {
        "note": "M9.5R does not retest predictive generalization/known-family guardrails -- M9.4 remains authoritative. Section 18 metrics here are descriptive only.",
        "no_safety_authority_regression": True,
        "neural_outputs_never_bypass_deterministic_authority": True,
        "alpha_unchanged": m5r.ALPHA == 0.1,
        "coverage_floor_unchanged": m5r.OPERATIONAL_COVERAGE_FLOOR == 0.85,
        "sanity_gate": sanity_gate,
        "current_control": current_control,
        "interleaved_confirmation_gate": interleaved_gate,
        "candidate_set_analysis": candidate_set_analysis,
    }
    m5r.M9_5R_GUARDRAILS_PATH.write_text(json.dumps(guardrails, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    decision_code, decision_name, decision_reason = _decide(
        audit["representativeness_audit_passed"], sanity_gate, current_control, interleaved_gate, candidate_set_analysis,
    )
    print(f"M9_5R_DECISION = {decision_code} ({decision_name})", flush=True)

    provisional_recipe = None
    if decision_code == "A":
        provisional_recipe = "CLASSICAL_HYDROCORE_S + AGE_FIX_ONLY + STEP_MATCHED_INTERLEAVED_MULTI_FAMILY"

    next_milestone = "M9.6_EXACT_COMPUTE_PARITY_CONFIRMATION" if decision_code == "A" else {
        "B": "diagnose remaining INTERLEAVED calibration failure under a separately governed milestone (do not change architecture)",
        "C": "audit the broader calibration/data pipeline under a separately governed milestone",
        "D": "fix the actual representativeness/implementation problem under a separately governed milestone",
        "E": "reconsider calibration actionability (candidate-set design), not architecture, under a separately governed milestone",
        "F": "collect additional evidence to resolve the unforeseen ambiguity under a separately governed milestone",
    }[decision_code]

    locked_after = m5r.assert_locked_test_closed()
    end_commit = m5r.current_commit()

    closure = {
        "milestone": "M9.5R", "kind": "INDEPENDENT_CALIBRATION_CONFIRMATION",
        "branch": manifest["branch"], "start_commit": manifest["start_commit"],
        "protocol_frozen_at_commit": manifest["protocol_frozen_at_commit"], "execution_commit": end_commit,
        "end_commit_note": (
            f"{end_commit} is the commit at decide-stage execution time, BEFORE this milestone's own code+"
            "artifact commit exists (a commit cannot embed its own SHA at authoring time -- see M9.4/M9.5's "
            "closure/manifest end_commit_note for the same issue). A metadata-only follow-up commit records "
            "the true final SHA after the initial M9.5R commit is made, mirroring M9.4/M9.5's fix."
        ),
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "no_training_performed": True, "no_predictor_modified": True,
        "alpha": m5r.ALPHA, "coverage_floor": m5r.OPERATIONAL_COVERAGE_FLOOR,
        "calibration_repeats_per_source": m5r.CALIBRATION_REPEATS_PER_SOURCE,
        "development_repeats_per_source": m5r.DEVELOPMENT_REPEATS_PER_SOURCE,
        "representativeness_audit_passed": audit["representativeness_audit_passed"],
        "sanity_gate_passed": sanity_gate["M9_5R_SANITY_GATE"] == "PASS",
        "current_control": {"all_3_pass": current_control["all_3_pass"], "per_seed_coverage": current_control["per_seed_coverage"]},
        "interleaved": {"all_9_cells_pass": interleaved_gate["all_9_cells_pass"], "per_family_seed_coverage": interleaved_gate["per_family_seed_coverage"]},
        "candidate_set_guard_passed": not candidate_set_analysis["pathological_full_set_behavior_detected"],
        "pathological_full_set_behavior": candidate_set_analysis["pathological_full_set_behavior_detected"],
        "M9_5R_DECISION": decision_code, "M9_5R_DECISION_NAME": decision_name,
        "provisional_best_hydrocore_s_recipe": provisional_recipe,
        "next_recommended_milestone": next_milestone,
        "strongest_evidence": (
            f"ARM_B2/INTERLEAVED trained-family/seed cells passing >=0.85: "
            f"{sum(1 for v in interleaved_gate['per_family_seed_coverage'].values() if v['passes_operational_floor_0_85'])}/9. "
            f"ARM_A/CURRENT control all-3-seeds-pass: {current_control['all_3_pass']}. "
            f"Representativeness PASS: {audit['representativeness_audit_passed']}. Sanity gate PASS: "
            f"{sanity_gate['M9_5R_SANITY_GATE'] == 'PASS'}."
        ),
        "evidence_against": decision_reason if decision_code != "A" else "None identified against the primary confirmation question at this run.",
        "limitations": [
            "M9.5R does not repeat M9.5's 4/8/12/20 support curve -- exactly one primary support condition (20/source) was run, per governing Section 7.",
            "Unseen-family (coastal-branch/tree-branch/dense-loop) inference is out of scope for M9.5R -- calibration validity concerns trained families only.",
            "Section 18 predictive sanity metrics are descriptive only; M9.5R does not redefine or retest M9.4's predictive-generalization decision.",
            "seed20260814's ARM_B2 checkpoint has 1200 optimizer steps vs 1350 for the other two seeds and vs ARM_A's 1350 (known M9.0a optimizer-step-parity gap) -- preserved, not fixed, here; unaffected by M9.5R since no training occurs.",
            "M9.5R does not reinterpret or overwrite M9.5 -- M9.5 remains formally closed as M9_5_DECISION=E (REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER).",
            "This is a calibration confirmation only; no field-performance, production-readiness, or locked-test claim is made.",
        ],
        "decision_reason": decision_reason,
    }
    m5r.M9_5R_CLOSURE_PATH.write_text(json.dumps(closure, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    _write_summary(protocol, manifest, audit, sanity_gate, current_control, interleaved_gate, candidate_set_analysis, loop_grid_j1, closure)

    print(json.dumps({"M9_5R_DECISION": decision_code, "decision_name": decision_name}, indent=2))
    return 0


def _write_summary(protocol, manifest, audit, sanity_gate, current_control, interleaved_gate, candidate_set_analysis, loop_grid_j1, closure) -> None:
    lines = [
        "# Milestone 9.5R summary: independent, one-shot calibration confirmation",
        "",
        "Independent, one-shot confirmation of HydroCore-S calibration at the already-predeclared adequate "
        "support level (20 independent calibration incidents/source), using fresh, disjoint calibration and "
        "development populations. Does NOT reinterpret or overwrite M9.5, which remains formally closed as "
        "`M9_5_DECISION=E` (`REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER`).",
        "",
        f"**Representativeness audit passed**: {audit['representativeness_audit_passed']}",
        f"**Corrected sanity/implementation gate**: {sanity_gate['M9_5R_SANITY_GATE']}",
        "",
        "## CURRENT control (ARM_A, golden-reference)",
        "",
        f"All 3 seeds pass >=0.85: **{current_control['all_3_pass']}**",
        f"Per-seed coverage: {current_control['per_seed_coverage']}",
        "",
        "## INTERLEAVED confirmation (ARM_B2, 3 trained families x 3 seeds)",
        "",
        f"All 9 cells pass >=0.85: **{interleaved_gate['all_9_cells_pass']}**",
        "",
        "## Candidate-set-size guard",
        "",
        f"Pathological full-set behavior detected: **{candidate_set_analysis['pathological_full_set_behavior_detected']}**",
        "",
        f"## M9_5R_DECISION: {closure['M9_5R_DECISION']} ({closure['M9_5R_DECISION_NAME']})",
        "",
        closure["decision_reason"],
        "",
        f"Provisional best HydroCore-S recipe: {closure['provisional_best_hydrocore_s_recipe']}",
        f"Next recommended milestone: {closure['next_recommended_milestone']}",
        "",
        f"locked tests opened: before={closure['locked_test_opened_before']}, after={closure['locked_test_opened_after']}. "
        "No model promoted to production. No safety/authority semantics changed. No architecture/training/"
        "calibration-method change performed. No field-performance claim made.",
    ]
    m5r.M9_5R_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
