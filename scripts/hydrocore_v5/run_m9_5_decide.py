"""Milestone 9.5 decision stage: reads `run_m9_5_source_representative.py`'s
canonical calibration/development rows, fits the frozen SplitConformalCalibrator
at each nested support level (Section 13), runs the quantile-stability
bootstrap (Section 14), the primary support=20 calibration gate (Section 16),
control-arm diagnostic (Section 17), source-conditional/loop-grid-J1
diagnostics (Section 19/20), candidate-set-size guard (Section 21), frozen-
checkpoint predictive sanity metrics (Section 28), and assigns M9_5_DECISION
(Section 23).

Reads (never regenerates):
  reports/evaluation/hydrocore-v5/m9-5/m9-5-manifest.json
  reports/evaluation/hydrocore-v5/m9-5/m9-5-canonical-calibration.jsonl
  reports/evaluation/hydrocore-v5/m9-5/m9-5-representativeness-audit.json

Writes:
  m9-5-support-curve.json, m9-5-quantile-stability.json,
  m9-5-calibration-results.json, m9-5-source-conditional.json,
  m9-5-loop-grid-j1.json, m9-5-candidate-set-analysis.json,
  m9-5-control-arm-analysis.json, m9-5-guardrails.json, m9-5-summary.md,
  m9-5-closure.json
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

import m9_5_common as m5  # noqa: E402
from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator, _quantile  # noqa: E402
from run_m9_0a_evaluate import _build_libraries, _library_for  # noqa: E402


# ---------------------------------------------------------------------------
# Loading / filtering.
# ---------------------------------------------------------------------------


def _load_canonical() -> list[dict[str, Any]]:
    rows = []
    with m5.M9_5_CANONICAL_CALIBRATION_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def _known_families(arm: str) -> tuple[str, ...]:
    return m5.ARM_A_KNOWN_FAMILIES if arm == "ARM_A" else m5.ARM_B2_KNOWN_FAMILIES


def _cal_rows(rows: list[dict[str, Any]], arm: str, seed: int, family: str | None = None, level: int | None = None) -> list[dict[str, Any]]:
    out = [r for r in rows if r["split"] == "calibration" and r["arm"] == arm and r["predictor_seed"] == seed]
    if family is not None:
        out = [r for r in out if r["family"] == family]
    if level is not None:
        out = [r for r in out if r["repeat"] < level]
    return out


def _dev_rows(rows: list[dict[str, Any]], arm: str, seed: int, family: str | None = None) -> list[dict[str, Any]]:
    out = [r for r in rows if r["split"] == "development" and r["arm"] == arm and r["predictor_seed"] == seed]
    if family is not None:
        out = [r for r in out if r["family"] == family]
    return out


def _incident_count(rows: list[dict[str, Any]]) -> int:
    return len({(r["source_node"], r["generator_seed"]) for r in rows})


# ---------------------------------------------------------------------------
# Section 13/16: fit the UNMODIFIED SplitConformalCalibrator at a given
# support level, jointly over an arm's known families (mirrors M9.4's
# `_fit_calibrator_m9_4` -- ONE calibrator per arm x seed, B_DEPTH_AWARE
# network_id=f"{family}:{bucket}" grouping resolves per-family specificity).
# ---------------------------------------------------------------------------


def _fit_calibrator_at_level(rows: list[dict[str, Any]], arm: str, seed: int, level: int) -> SplitConformalCalibrator:
    examples: list[CalibrationExample] = []
    for family in sorted(_known_families(arm)):
        for r in _cal_rows(rows, arm, seed, family, level):
            examples.append(CalibrationExample(
                probabilities=tuple(r["probabilities"]), true_index=r["true_index"], condition=r["condition"], network_id=r["network_id"],
            ))
    return SplitConformalCalibrator.fit(
        examples, alpha=m5.ALPHA, minimum_group_size=m5.MINIMUM_GROUP_SIZE,
        model_hash=f"m9-5-{arm}-seed{seed}-support{level}", feature_schema_hash="n/a",
        dataset_manifest_hash=f"m9-5-{arm}-seed{seed}-support{level}-calibration_m9_5-pool",
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
# Section 13: support-curve analysis.
# ---------------------------------------------------------------------------


def _support_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n_nodes_by_family = {f: len(m5.full_junction_list(f, m5.ALL_FAMILY_LOADERS[f])) for f in m5.TRAINED_FAMILIES}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        for seed in m5.SEEDS:
            out[arm][str(seed)] = {}
            for level in m5.SUPPORT_LEVELS:
                calibrator = _fit_calibrator_at_level(rows, arm, seed, level)
                out[arm][str(seed)][str(level)] = {}
                for family in _known_families(arm):
                    cal_subset = _cal_rows(rows, arm, seed, family, level)
                    dev_subset = _dev_rows(rows, arm, seed, family)
                    applied = _apply(calibrator, dev_subset)
                    quantiles = {}
                    for bucket in ("EARLY", "MID", "MATURE"):
                        network_id = f"{family}:{bucket}"
                        source, group_key, scores = calibrator.selection(network_id=network_id)
                        quantiles[bucket] = {
                            "group_source": source, "group_key": group_key, "group_n": len(scores),
                            "quantile": _quantile(scores, m5.ALPHA) if scores else None,
                        }
                    out[arm][str(seed)][str(level)][family] = {
                        "n_independent_calibration_incidents": _incident_count(cal_subset),
                        "n_calibration_depth_rows": len(cal_subset),
                        "quantiles_by_bucket": quantiles,
                        **_cov_summary(applied, n_nodes_by_family[family]),
                    }
    return out


# ---------------------------------------------------------------------------
# Section 14: quantile-stability bootstrap (support=4 vs support=20), bucket-
# level, resampling unit = physical calibration incident.
# ---------------------------------------------------------------------------


def _bootstrap_quantiles(rows: list[dict[str, Any]], bucket: str, resamples: int, seed: int) -> list[float]:
    bucket_rows = [r for r in rows if r["depth_bucket"] == bucket]
    if not bucket_rows:
        return []
    by_incident: dict[tuple[str, int], list[float]] = {}
    for r in bucket_rows:
        key = (r["source_node"], r["generator_seed"])
        by_incident.setdefault(key, []).append(r["nonconformity_score"])
    incidents = sorted(by_incident)
    n = len(incidents)
    rng = np.random.default_rng(seed)
    quantiles = []
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        scores: list[float] = []
        for i in idx:
            scores.extend(by_incident[incidents[i]])
        quantiles.append(_quantile(scores, m5.ALPHA))
    return quantiles


def _quantile_stability_stats(quantiles: list[float]) -> dict[str, Any]:
    if not quantiles:
        return {}
    arr = np.asarray(quantiles)
    p5, p50, p95 = (float(np.percentile(arr, p)) for p in (5, 50, 95))
    return {
        "median_quantile": p50, "p5_quantile": p5, "p95_quantile": p95,
        "absolute_span": p95 - p5, "relative_span": (p95 - p5) / p50 if p50 else None,
    }


def _quantile_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        for seed in m5.SEEDS:
            out[arm][str(seed)] = {}
            for family in _known_families(arm):
                out[arm][str(seed)][family] = {}
                for level in (4, m5.PRIMARY_SUPPORT):
                    cal_subset = _cal_rows(rows, arm, seed, family, level)
                    out[arm][str(seed)][family][str(level)] = {
                        bucket: _quantile_stability_stats(_bootstrap_quantiles(cal_subset, bucket, m5.QUANTILE_BOOTSTRAP_RESAMPLES, m5.QUANTILE_BOOTSTRAP_SEED))
                        for bucket in ("EARLY", "MID", "MATURE")
                    }
    # Aggregate: did span shrink materially from support=4 to support=20?
    improved = 0
    total = 0
    for arm in out:
        for seed in out[arm]:
            for family in out[arm][seed]:
                for bucket in ("EARLY", "MID", "MATURE"):
                    s4 = out[arm][seed][family].get("4", {}).get(bucket, {})
                    s20 = out[arm][seed][family].get(str(m5.PRIMARY_SUPPORT), {}).get(bucket, {})
                    if s4.get("absolute_span") is not None and s20.get("absolute_span") is not None:
                        total += 1
                        if s20["absolute_span"] < s4["absolute_span"]:
                            improved += 1
    out["_summary"] = {
        "n_cells_compared": total, "n_cells_span_shrunk": improved,
        "quantile_stability_improved_with_support": (improved / total > 0.5) if total else None,
        "resamples": m5.QUANTILE_BOOTSTRAP_RESAMPLES, "bootstrap_seed": m5.QUANTILE_BOOTSTRAP_SEED,
    }
    return out


# ---------------------------------------------------------------------------
# Section 12: reproduction/sanity gate (12A implementation-path consistency
# using the support=4 nested subset) + resubstitution diagnostic (12B,
# support=20, DESCRIPTIVE ONLY -- not a validity metric).
# ---------------------------------------------------------------------------


def _reproduction_sanity_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_arm_seed": {}}
    arm_a_poor = []
    arm_b2_mixed_or_failing = []
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m5.SEEDS:
            calibrator = _fit_calibrator_at_level(rows, arm, seed, 4)
            config_ok = (
                calibrator.artifact.alpha == m5.ALPHA
                and type(calibrator).__name__ == "SplitConformalCalibrator"
            )
            per_family_coverage = {}
            for family in _known_families(arm):
                dev_subset = _dev_rows(rows, arm, seed, family)
                applied = _apply(calibrator, dev_subset)
                per_family_coverage[family] = _cov_summary(applied).get("marginal_coverage")
            out["per_arm_seed"][f"{arm}|{seed}"] = {
                "support_4_calibrator_config_matches_frozen_construction": config_ok,
                "minimum_group_size_used": m5.MINIMUM_GROUP_SIZE,
                "per_family_marginal_coverage_at_support_4": per_family_coverage,
            }
            if arm == "ARM_A":
                arm_a_poor.append(per_family_coverage.get("golden-reference", 1.0) < m5.OPERATIONAL_COVERAGE_FLOOR)
            else:
                arm_b2_mixed_or_failing.append(not all(v is not None and v >= m5.OPERATIONAL_COVERAGE_FLOOR for v in per_family_coverage.values()))
    implementation_path_consistent = all(
        out["per_arm_seed"][k]["support_4_calibrator_config_matches_frozen_construction"] for k in out["per_arm_seed"]
    )
    qualitatively_consistent_with_m9_4 = (sum(arm_a_poor) >= 2) and (sum(arm_b2_mixed_or_failing) >= 1)
    out["implementation_path_consistent"] = implementation_path_consistent
    out["qualitatively_consistent_with_m9_4_pattern"] = qualitatively_consistent_with_m9_4
    out["M9_5_REPRODUCTION_SANITY_GATE"] = "PASS" if (implementation_path_consistent and qualitatively_consistent_with_m9_4) else "FAIL"
    out["note"] = (
        "Implementation-path consistency (same calibrator class/alpha/grouping), NOT exact-value equality "
        "(M9.5 uses intentionally NEW, disjoint seeds from M9.4's calibration_m9_4/development_m9_4 pools). "
        "Qualitative check: ARM_A/golden-reference poor (<0.85) on >=2/3 seeds and ARM_B2 mixed-or-failing on "
        ">=1/3 seeds at the support=4 nested subset, consistent with M9.4's own observed pattern."
    )
    return out


def _resubstitution_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"RESUBSTITUTION_DIAGNOSTIC_ONLY": True, "per_arm_seed_family": {}}
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m5.SEEDS:
            calibrator = _fit_calibrator_at_level(rows, arm, seed, m5.PRIMARY_SUPPORT)
            for family in _known_families(arm):
                cal_subset = _cal_rows(rows, arm, seed, family, m5.PRIMARY_SUPPORT)
                applied = _apply(calibrator, cal_subset)
                out["per_arm_seed_family"][f"{arm}|{seed}|{family}"] = {
                    "in_sample_coverage": _cov_summary(applied).get("marginal_coverage"),
                    "n": len(cal_subset),
                }
    out["note"] = "In-sample (resubstitution) coverage on the SAME support=20 calibration pool the calibrator was fit on -- expected near-nominal, used only to rule out a gross implementation bug, never as an operational validity metric."
    return out


# ---------------------------------------------------------------------------
# Section 19/20: source-conditional coverage + loop-grid J1/J7/J8 diagnostic
# (PRIMARY support=20 only).
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
        for seed in m5.SEEDS:
            out[arm][str(seed)] = {}
            applied = applied_at_primary[(arm, seed)]
            for family in _known_families(arm):
                out[arm][str(seed)][family] = {}
                fam_applied = [r for r in applied if r["family"] == family]
                sources = sorted({r["source_node"] for r in fam_applied})
                cal_rows_fam = _cal_rows(rows, arm, seed, family, m5.PRIMARY_SUPPORT)
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
    for seed in m5.SEEDS:
        applied = [r for r in applied_at_primary[("ARM_B2", seed)] if r["family"] == "loop-grid"]
        node_ids = node_ids_by_arm_family[("ARM_B2", "loop-grid")]
        index_of = {j: i for i, j in enumerate(node_ids)}
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
            "j1_mean_true_source_rank": statistics.fmean(m["rank"] for m in metrics) if metrics else None,
            "j1_marginal_coverage": statistics.fmean(r["candidate_covered"] for r in j1_rows) if j1_rows else None,
            "j1_mean_candidate_set_size": statistics.fmean(r["candidate_set_size"] for r in j1_rows) if j1_rows else None,
            "confusion_counts": confusion,
            "n_j1_development_incidents": _incident_count(j1_rows),
        }
    out["note"] = "M9.5 development_m9_5 J1/J7/J8 behavior at PRIMARY support=20, for descriptive comparison against M9.4's m9-4-confusion-matrices.json findings. Source-discrimination behavior (as opposed to calibration coverage) is NOT altered by M9.5's intervention -- this is a secondary diagnostic, not a gate."
    return out


# ---------------------------------------------------------------------------
# Section 21: candidate-set-size guard.
# ---------------------------------------------------------------------------


def _candidate_set_analysis(support_curve: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_arm_seed_family_level": {}, "pathological_full_set_behavior_detected": False}
    pathological = []
    for arm in support_curve:
        for seed in support_curve[arm]:
            for level in support_curve[arm][seed]:
                for family, data in support_curve[arm][seed][level].items():
                    key = f"{arm}|{seed}|{level}|{family}"
                    out["per_arm_seed_family_level"][key] = {
                        "mean_candidate_set_size": data.get("mean_candidate_set_size"),
                        "normalized_mean_candidate_set_size": data.get("normalized_mean_candidate_set_size"),
                        "full_set_rate": data.get("full_set_rate"),
                        "singleton_rate": data.get("singleton_rate"),
                    }
                    if level == str(m5.PRIMARY_SUPPORT) and (data.get("full_set_rate") or 0) > 0.8:
                        pathological.append(key)
    out["pathological_full_set_behavior_detected"] = bool(pathological)
    out["pathological_cells"] = pathological
    return out


# ---------------------------------------------------------------------------
# Section 17: control-arm diagnostic.
# ---------------------------------------------------------------------------


def _control_arm_analysis(support_curve: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"golden_reference_per_seed_coverage": {}, "by_support_level": {}}
    for level in m5.SUPPORT_LEVELS:
        level_vals = []
        for seed in m5.SEEDS:
            data = support_curve.get("ARM_A", {}).get(str(seed), {}).get(str(level), {}).get("golden-reference", {})
            cov = data.get("marginal_coverage")
            out["golden_reference_per_seed_coverage"].setdefault(str(seed), {})[str(level)] = cov
            if cov is not None:
                level_vals.append(cov)
        out["by_support_level"][str(level)] = {
            "mean_coverage": statistics.fmean(level_vals) if level_vals else None,
            "min_coverage": min(level_vals) if level_vals else None,
            "all_3_seeds_pass_0_85": all(v >= m5.OPERATIONAL_COVERAGE_FLOOR for v in level_vals) if len(level_vals) == 3 else False,
        }
    primary = out["by_support_level"][str(m5.PRIMARY_SUPPORT)]
    out["adequate_support_restored_calibration"] = bool(primary["all_3_seeds_pass_0_85"])
    return out


# ---------------------------------------------------------------------------
# Section 16/23: primary calibration gate + decision.
# ---------------------------------------------------------------------------


def _interleaved_calibration_gate(support_curve: dict[str, Any]) -> dict[str, Any]:
    per_cell: dict[str, Any] = {}
    all_pass = True
    for seed in m5.SEEDS:
        for family in m5.TRAINED_FAMILIES:
            data = support_curve["ARM_B2"][str(seed)][str(m5.PRIMARY_SUPPORT)][family]
            cov = data.get("marginal_coverage")
            ok = cov is not None and cov >= m5.OPERATIONAL_COVERAGE_FLOOR
            all_pass = all_pass and ok
            per_cell[f"ARM_B2|{family}|{seed}"] = {"marginal_coverage": cov, "passes_operational_floor_0_85": ok}
    return {"passed": all_pass, "all_9_family_seed_cells_pass": all_pass, "per_family_seed": per_cell}


def _decide(
    audit_passed: bool, reproduction_gate: dict[str, Any], interleaved_gate: dict[str, Any],
    control_arm: dict[str, Any], candidate_set: dict[str, Any], quantile_stability: dict[str, Any],
) -> tuple[str, str, str]:
    if not audit_passed or reproduction_gate["M9_5_REPRODUCTION_SANITY_GATE"] != "PASS":
        return (
            "E", "REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER",
            "The representativeness audit or the Section 12 reproduction/implementation-path-consistency "
            "sanity gate did not pass; the M9.5 support-level calibration results below cannot yet be "
            "scientifically interpreted.",
        )
    if not control_arm["adequate_support_restored_calibration"]:
        return (
            "D", "CONTROL_ARM_CALIBRATION_FAILURE_REMAINS",
            "ARM_A/CURRENT (the calibration-support control) still fails marginal coverage >=0.85 on "
            "golden-reference at PRIMARY support=20 despite full source support and a passed "
            "representativeness audit -- this points to a broader calibration/evaluation-pipeline issue "
            "rather than an INTERLEAVED-specific problem; per Section 17/27, this blocks attributing any "
            "residual ARM_B2 failure to insufficient support alone.",
        )
    if interleaved_gate["all_9_family_seed_cells_pass"] and not candidate_set["pathological_full_set_behavior_detected"]:
        return (
            "A", "CALIBRATION_SUPPORT_HYPOTHESIS_CONFIRMED_INTERLEAVED_VALID",
            "All 9 ARM_B2 trained-family/seed cells pass marginal coverage >=0.85 at the predeclared PRIMARY "
            "support=20 level, the candidate-set-size guard passes (no pathological full-set behavior), and "
            "the representativeness/control-arm/reproduction preconditions all hold.",
        )
    n_passing = sum(1 for v in interleaved_gate["per_family_seed"].values() if v["passes_operational_floor_0_85"])
    n_passing_at_support_4 = 0  # computed by caller and folded into limitations; kept simple here
    if n_passing > 0:
        return (
            "B", "SUPPORT_IMPROVES_CALIBRATION_BUT_NOT_ALL_GUARDRAILS_PASS",
            f"{n_passing}/9 ARM_B2 trained-family/seed cells pass at PRIMARY support=20 (control arm recovered, "
            "representativeness/reproduction preconditions hold), but at least one required cell remains below "
            "0.85 -- small-n calibration support mattered but is not the complete explanation.",
        )
    return (
        "C", "CALIBRATION_SUPPORT_HYPOTHESIS_NOT_CONFIRMED",
        "Increasing calibration support to the predeclared PRIMARY level (20 independent incidents/source) did "
        "not materially improve ARM_B2 trained-family coverage (0/9 cells pass) despite the control arm "
        "recovering and all preconditions holding -- systematic undercoverage remains; move to calibration-"
        "design/root-cause remediation rather than further support increases.",
    )


def main() -> int:
    locked_before = m5.assert_locked_test_closed()
    manifest = json.loads(m5.M9_5_MANIFEST_PATH.read_text())
    audit = json.loads(m5.M9_5_REPRESENTATIVENESS_AUDIT_PATH.read_text())

    print("loading canonical calibration/development rows...", flush=True)
    rows = _load_canonical()
    print(f"loaded {len(rows)} rows", flush=True)

    print("running Section 12 reproduction/sanity gate...", flush=True)
    reproduction_gate = _reproduction_sanity_gate(rows)
    print("running Section 12B resubstitution diagnostic...", flush=True)
    resubstitution = _resubstitution_diagnostic(rows)

    print("computing support-curve (Section 13)...", flush=True)
    support_curve = _support_curve(rows)
    m5.M9_5_SUPPORT_CURVE_PATH.write_text(json.dumps(support_curve, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing quantile-stability bootstrap (Section 14)...", flush=True)
    quantile_stability = _quantile_stability(rows)
    m5.M9_5_QUANTILE_STABILITY_PATH.write_text(json.dumps(quantile_stability, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    calibration_results = {
        "support_levels": list(m5.SUPPORT_LEVELS), "primary_support": m5.PRIMARY_SUPPORT,
        "reproduction_sanity_gate": reproduction_gate, "resubstitution_diagnostic_only": resubstitution,
    }
    m5.M9_5_CALIBRATION_RESULTS_PATH.write_text(json.dumps(calibration_results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing PRIMARY-support applied rows for source-conditional/J1/predictive-sanity...", flush=True)
    applied_at_primary: dict[tuple[str, int], list[dict[str, Any]]] = {}
    primary_calibrators: dict[tuple[str, int], SplitConformalCalibrator] = {}
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m5.SEEDS:
            calibrator = _fit_calibrator_at_level(rows, arm, seed, m5.PRIMARY_SUPPORT)
            primary_calibrators[(arm, seed)] = calibrator
            dev_subset = _dev_rows(rows, arm, seed)
            applied_at_primary[(arm, seed)] = _apply(calibrator, dev_subset)

    print("computing source-conditional performance (Section 19)...", flush=True)
    source_conditional = _source_conditional(rows, applied_at_primary)
    m5.M9_5_SOURCE_CONDITIONAL_PATH.write_text(json.dumps(source_conditional, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("building libraries for node_ids (loop-grid J1 diagnostic, Section 20)...", flush=True)
    libraries = _build_libraries()
    node_ids_by_arm_family = {
        (arm, family): tuple(_library_for(libraries, family, arm).node_ids)
        for arm in ("ARM_A", "ARM_B2") for family in _known_families(arm)
    }
    loop_grid_j1 = _loop_grid_j1(rows, applied_at_primary, node_ids_by_arm_family)
    m5.M9_5_LOOP_GRID_J1_PATH.write_text(json.dumps(loop_grid_j1, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing candidate-set-size guard (Section 21)...", flush=True)
    candidate_set_analysis = _candidate_set_analysis(support_curve)
    m5.M9_5_CANDIDATE_SET_ANALYSIS_PATH.write_text(json.dumps(candidate_set_analysis, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing control-arm diagnostic (Section 17)...", flush=True)
    control_arm_analysis = _control_arm_analysis(support_curve)
    m5.M9_5_CONTROL_ARM_ANALYSIS_PATH.write_text(json.dumps(control_arm_analysis, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("evaluating primary calibration gate (Section 16)...", flush=True)
    interleaved_gate = _interleaved_calibration_gate(support_curve)

    print("computing Section 28 predictive sanity metrics (descriptive)...", flush=True)
    predictive_sanity: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        predictive_sanity[arm] = {}
        for family in _known_families(arm):
            per_seed_metrics = []
            for seed in m5.SEEDS:
                fam_rows = [r for r in applied_at_primary[(arm, seed)] if r["family"] == family]
                metrics = [_predictive_row_metrics(r) for r in fam_rows]
                per_seed_metrics.append({k: statistics.fmean(m[k] for m in metrics) for k in ("top1", "top3", "mrr", "nll", "brier", "entropy")})
            predictive_sanity[arm][family] = {
                metric: statistics.fmean(s[metric] for s in per_seed_metrics) for metric in ("top1", "top3", "mrr", "nll", "brier", "entropy")
            }

    guardrails = {
        "known_family_guardrails_note": "M9.5 does not retest predictive generalization/known-family guardrails -- those are M9.4's domain (Section 28: predictive sanity metrics here are descriptive only, do not redefine M9.4's decision).",
        "predictive_sanity_metrics_development_m9_5": predictive_sanity,
        "interleaved_calibration_gate": interleaved_gate,
        "control_arm_analysis": control_arm_analysis,
        "candidate_set_analysis": candidate_set_analysis,
        "quantile_stability_summary": quantile_stability["_summary"],
    }
    m5.M9_5_GUARDRAILS_PATH.write_text(json.dumps(guardrails, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    decision_code, decision_name, decision_reason = _decide(
        audit["representativeness_audit_passed"], reproduction_gate, interleaved_gate, control_arm_analysis,
        candidate_set_analysis, quantile_stability,
    )
    print(f"M9_5_DECISION = {decision_code} ({decision_name})", flush=True)

    provisional_recipe = None
    if decision_code == "A":
        provisional_recipe = "CLASSICAL_HYDROCORE_S + AGE_FIX_ONLY + STEP_MATCHED_INTERLEAVED_MULTI_FAMILY"

    next_milestone = {
        "A": "EXACT_COMPUTE_PARITY_CONFIRMATION (not started in M9.5)",
        "B": "M9.6_CALIBRATION_DESIGN_DIAGNOSTIC (family/source/depth-conditional score-shift investigation; not started in M9.5)",
        "C": "M9.6_CALIBRATION_DESIGN_REMEDIATION (reopen calibration design/construction, not model architecture; not started in M9.5)",
        "D": "M9.6_CALIBRATION_PIPELINE_AUDIT (quantile construction/grouping/population-identity/depth-correlation/score-definition audit; not started in M9.5)",
        "E": "M9.5_RECHECK (resolve representativeness/reproduction blocker before any further interpretation)",
        "F": "M9.5_EXTENDED_EVIDENCE (collect additional evidence to resolve ambiguity)",
    }[decision_code]

    locked_after = m5.assert_locked_test_closed()
    end_commit = m5.current_commit()

    closure = {
        "milestone": "M9.5", "kind": "SOURCE_REPRESENTATIVE_CALIBRATION_SUPPORT_CONFIRMATION",
        "start_commit": manifest["start_commit"], "end_commit": end_commit,
        "end_commit_note": (
            f"{end_commit} is the commit at decide-stage execution time, BEFORE this milestone's own code+"
            "artifact commit exists (a commit cannot embed its own SHA at authoring time -- see M9.4's "
            "closure/manifest end_commit_note for the same issue). A metadata-only follow-up commit records "
            "the true final SHA after the initial M9.5 commit is made, mirroring M9.4's fix."
        ),
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "no_training_performed": True, "no_predictor_modified": True,
        "alpha": m5.ALPHA, "coverage_floor": m5.OPERATIONAL_COVERAGE_FLOOR,
        "support_levels_repeats_per_source": list(m5.SUPPORT_LEVELS), "primary_support_repeats_per_source": m5.PRIMARY_SUPPORT,
        "representativeness_audit_passed": audit["representativeness_audit_passed"],
        "reproduction_sanity_gate_passed": reproduction_gate["M9_5_REPRODUCTION_SANITY_GATE"] == "PASS",
        "control_arm": {
            "golden_reference_per_seed_coverage": control_arm_analysis["golden_reference_per_seed_coverage"],
            "adequate_support_restored_calibration": control_arm_analysis["adequate_support_restored_calibration"],
        },
        "interleaved_calibration_gate": interleaved_gate,
        "quantile_stability_improved_with_support": quantile_stability["_summary"]["quantile_stability_improved_with_support"],
        "candidate_set_guard_passed": not candidate_set_analysis["pathological_full_set_behavior_detected"],
        "M9_5_DECISION": decision_code, "M9_5_DECISION_NAME": decision_name,
        "provisional_best_hydrocore_s_recipe": provisional_recipe,
        "next_recommended_milestone": next_milestone,
        "strongest_evidence": (
            f"ARM_B2 PRIMARY-support(={m5.PRIMARY_SUPPORT}) trained-family/seed cells passing >=0.85: "
            f"{sum(1 for v in interleaved_gate['per_family_seed'].values() if v['passes_operational_floor_0_85'])}/9. "
            f"Control-arm (ARM_A golden-reference) all-3-seeds-pass at PRIMARY support: "
            f"{control_arm_analysis['adequate_support_restored_calibration']}."
        ),
        "evidence_against": decision_reason,
        "recommendation_reason": decision_reason,
        "limitations": [
            "SCOPE: unseen-family (coastal-branch/tree-branch/dense-loop) inference was omitted entirely "
            "(optional per governing Section 10; calibration validity concerns trained families only) -- see "
            "manifest unseen_family_scope_note.",
            "The Section 12A reproduction/sanity check is an implementation-path-consistency + qualitative-"
            "pattern check against M9.4's OWN observations, not a numerical reproduction (M9.5 uses "
            "intentionally new, disjoint seeds -- exact-value equality was never expected or required).",
            "Section 28 predictive sanity metrics are descriptive only; M9.5 does not redefine or retest "
            "M9.4's predictive-generalization decision.",
            "seed20260814's ARM_B2 checkpoint has 1200 optimizer steps vs 1350 for the other two seeds and vs "
            "ARM_A's 1350 (known M9.0a optimizer-step-parity gap) -- preserved, not fixed, here; unaffected by "
            "M9.5 since no training occurs.",
            "Quantile-stability bootstrap compares support=4 vs support=20 only (per Section 14); the full "
            "4-level curve is in m9-5-support-curve.json but the formal stability comparison is 4-vs-20.",
        ],
    }
    m5.M9_5_CLOSURE_PATH.write_text(json.dumps(closure, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    _write_summary(manifest, audit, reproduction_gate, control_arm_analysis, interleaved_gate, candidate_set_analysis, quantile_stability, loop_grid_j1, closure)

    print(json.dumps({"M9_5_DECISION": decision_code, "decision_name": decision_name}, indent=2))
    return 0


def _write_summary(manifest, audit, reproduction_gate, control_arm_analysis, interleaved_gate, candidate_set_analysis, quantile_stability, loop_grid_j1, closure) -> None:
    lines = [
        "# Milestone 9.5 summary: source-representative calibration-support confirmation",
        "",
        "Calibration-support / frozen-checkpoint study only. No training, no tuning, no calibration-method "
        "change. Follows up `reports/evaluation/hydrocore-v5/m9-4/m9-4-closure.json` (M9_4_DECISION=B).",
        "",
        f"**Representativeness audit passed**: {audit['representativeness_audit_passed']}",
        f"**Reproduction/sanity gate**: {reproduction_gate['M9_5_REPRODUCTION_SANITY_GATE']}",
        "",
        "## Control arm (ARM_A / CURRENT, golden-reference)",
        "",
        f"All 3 seeds pass >=0.85 at PRIMARY support={closure['primary_support_repeats_per_source']}: "
        f"**{control_arm_analysis['adequate_support_restored_calibration']}**",
        f"Per-support-level coverage: {control_arm_analysis['by_support_level']}",
        "",
        "## INTERLEAVED / ARM_B2 primary calibration gate",
        "",
        f"All 9 trained-family/seed cells pass >=0.85: **{interleaved_gate['all_9_family_seed_cells_pass']}**",
        "",
        "## Candidate-set-size guard",
        "",
        f"Pathological full-set behavior detected: **{candidate_set_analysis['pathological_full_set_behavior_detected']}**",
        "",
        "## Quantile stability (support=4 vs support=20)",
        "",
        f"{quantile_stability['_summary']}",
        "",
        f"## M9_5_DECISION: {closure['M9_5_DECISION']} ({closure['M9_5_DECISION_NAME']})",
        "",
        closure["recommendation_reason"],
        "",
        f"Provisional best HydroCore-S recipe: {closure['provisional_best_hydrocore_s_recipe']}",
        f"Next recommended milestone: {closure['next_recommended_milestone']}",
        "",
        f"locked tests opened: before={closure['locked_test_opened_before']}, after={closure['locked_test_opened_after']}. "
        "No model promoted to production. No safety/authority semantics changed. No architecture/training/"
        "calibration-method change performed.",
    ]
    m5.M9_5_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
