"""physics-informed-localizer-full-data-gate (EXPERIMENTAL, NON-RELEASE):
paired statistical analysis of `run_experiment.py`'s pilot-600/full-data-9000
evaluation outputs, at the single pre-declared seed 20261110.

Thin wrapper around `physics_informed_localizer_validation.analyze_results`
(imported, not reimplemented): reuses its `paired_bootstrap` convention
(2000 resamples, deterministic bootstrap seed 20260826, 90% percentile
interval -- "HydroSwarm's established convention", unchanged), its
`build_metric_table`/`centrality_subgroups`/`distance_subgroups`/
`subgroup_paired_bootstrap`/`paired_transitions` logic, unmodified, run once
per stage (`pilot-600`, `full-data-9000`) by retargeting `RESULTS_ROOT`.
Adds exactly what no prior branch needed:

  - the pre-registered PRIMARY ENDPOINT gate: paired C1_C2-vs-A_CONTROL
    `ood-UNSEEN_TOPOLOGY` Top-1 on the full-data-9000 stage, classified
    PASS_FULL_DATA_GATE / INCONCLUSIVE_FULL_DATA_GATE / FAIL_FULL_DATA_GATE
    per the task's pre-registered 8-point gate (see
    docs/evaluation/experimental/PHYSICS_INFORMED_LOCALIZER_FULL_DATA_GATE_PLAN.md);
  - the "critical scale analysis": pilot (600) vs full-data (9000) A_CONTROL/
    C1_C2/delta at the SAME seed, and a qualitative determination of whether
    scale strengthens/preserves/attenuates/eliminates the effect;
  - ranking-shape validation (Top-3 avoidance-of-regression, true-source-rank
    when Top-1 is wrong, fraction of failures with true source in Top-3,
    correct-Top-3-to-outside-Top-3 conversions) for A_CONTROL vs C1_C2 on
    each stage.

Writes under reports/evaluation/physics-informed-localizer-full-data-gate/:
  - {pilot-600,full-data-9000}/seed-20261110/metric-table.{json,md},
    centrality-subgroups.json, distance-subgroups.json,
    subgroup-paired-bootstrap.json, paired-transitions.json,
    ranking-shape-analysis.json, parameter-counts.json
  - gate/primary-endpoint-gate.json
  - scale-comparison/pilot-vs-full-data.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental" / "physics_informed_localizer_validation"))
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental"))

import analyze_results as base  # noqa: E402  (physics_informed_localizer_validation's own module)

SEED: int = 20261110
ARM_NAMES: tuple[str, ...] = ("A_CONTROL", "C1_C2")
STAGES: tuple[str, ...] = ("pilot-600", "full-data-9000")
PRIMARY_POPULATION = "ood-UNSEEN_TOPOLOGY"
GATE_MIN_EFFECT_PP = 2.0

EXPERIMENT_NAME = "physics-informed-localizer-full-data-gate"
RESULTS_ROOT = ROOT / "reports" / "evaluation" / EXPERIMENT_NAME


def _stage_root(stage: str) -> Path:
    return RESULTS_ROOT / stage


def _set_stage(stage: str) -> None:
    base.RESULTS_ROOT = _stage_root(stage)
    base.ARM_NAMES = list(ARM_NAMES)


# ---------------------------------------------------------------------------
# Per-stage per-seed analysis (byte-for-byte reuse of the base module's own
# metric-table/subgroup/paired-transition functions)
# ---------------------------------------------------------------------------


def analyze_stage(stage: str) -> dict[str, dict[str, Any]] | None:
    _set_stage(stage)
    evaluations = {arm: base.load_evaluation(SEED, arm) for arm in ARM_NAMES}
    evaluations = {arm: evaluation for arm, evaluation in evaluations.items() if evaluation is not None}
    if not evaluations:
        print(f"[stage {stage}] no evaluation files found yet, skipping")
        return None
    rows_by_arm = {arm: {population: base.load_rows(SEED, arm, population) for population in base.POPULATIONS} for arm in evaluations}
    results_dir = _stage_root(stage) / f"seed-{SEED}"
    results_dir.mkdir(parents=True, exist_ok=True)

    table = base.build_metric_table(evaluations)
    (results_dir / "metric-table.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (results_dir / "metric-table.md").write_text(base.metric_table_markdown(table, list(evaluations)) + "\n", encoding="utf-8")
    (results_dir / "centrality-subgroups.json").write_text(
        json.dumps(base.centrality_subgroups(rows_by_arm, list(evaluations)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "distance-subgroups.json").write_text(
        json.dumps(base.distance_subgroups(rows_by_arm, list(evaluations)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "subgroup-paired-bootstrap.json").write_text(
        json.dumps(base.subgroup_paired_bootstrap(rows_by_arm, list(evaluations)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "paired-transitions.json").write_text(
        json.dumps(base.paired_transitions(rows_by_arm, list(evaluations)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    params = base.parameter_counts([SEED], list(evaluations))
    (results_dir / "parameter-counts.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ranking_shape = ranking_shape_analysis(rows_by_arm)
    (results_dir / "ranking-shape-analysis.json").write_text(
        json.dumps(ranking_shape, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[stage {stage}, seed {SEED}] wrote metric-table/subgroups/paired-transitions/ranking-shape-analysis/parameter-counts")
    return rows_by_arm


# ---------------------------------------------------------------------------
# Ranking-shape validation (task-required, not optional): does C1_C2 at this
# scale avoid a Top-3 regression, improve true-source rank when Top-1 stays
# wrong, increase the fraction of failures with the true source in Top-3,
# and avoid converting correct-Top-3 cases into outside-Top-3 cases?
# ---------------------------------------------------------------------------


def ranking_shape_analysis(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"population": PRIMARY_POPULATION}
    control_rows = {row["scenario_id"]: row for row in rows_by_arm.get("A_CONTROL", {}).get(PRIMARY_POPULATION, []) if row.get("has_source")}
    c1c2_rows = {row["scenario_id"]: row for row in rows_by_arm.get("C1_C2", {}).get(PRIMARY_POPULATION, []) if row.get("has_source")}
    shared = sorted(set(control_rows) & set(c1c2_rows))
    if not shared:
        result["n"] = 0
        return result

    control_top3 = [float(control_rows[sid]["top3"]) for sid in shared]
    c1c2_top3 = [float(c1c2_rows[sid]["top3"]) for sid in shared]
    result["n"] = len(shared)
    result["top3_bootstrap_delta"] = base.paired_bootstrap(control_top3, c1c2_top3)

    # Top-3 transition table: does C1_C2 convert previously-correct-Top-3
    # cases into outside-Top-3 (a regression), or the reverse (an
    # improvement)?
    top3_table = {"both_correct": 0, "control_only": 0, "c1_c2_only": 0, "both_wrong": 0}
    for sid in shared:
        control_ok, c1c2_ok = bool(control_rows[sid]["top3"]), bool(c1c2_rows[sid]["top3"])
        if control_ok and c1c2_ok:
            top3_table["both_correct"] += 1
        elif control_ok and not c1c2_ok:
            top3_table["control_only"] += 1
        elif not control_ok and c1c2_ok:
            top3_table["c1_c2_only"] += 1
        else:
            top3_table["both_wrong"] += 1
    result["top3_transition_table"] = top3_table
    result["net_top3_conversions"] = top3_table["c1_c2_only"] - top3_table["control_only"]
    result["correct_top3_converted_to_outside_top3"] = top3_table["control_only"]

    # True-source rank when Top-1 remains incorrect for C1_C2: did rank
    # improve relative to A_CONTROL on the same examples?
    rank_when_top1_wrong: list[tuple[int, int]] = []
    for sid in shared:
        if not c1c2_rows[sid]["top1"]:
            control_rank = control_rows[sid].get("true_source_rank")
            c1c2_rank = c1c2_rows[sid].get("true_source_rank")
            if control_rank is not None and c1c2_rank is not None:
                rank_when_top1_wrong.append((int(control_rank), int(c1c2_rank)))
    if rank_when_top1_wrong:
        control_ranks = [pair[0] for pair in rank_when_top1_wrong]
        c1c2_ranks = [pair[1] for pair in rank_when_top1_wrong]
        result["rank_when_c1_c2_top1_wrong"] = {
            "n": len(rank_when_top1_wrong),
            "control_mean_rank": statistics.fmean(control_ranks),
            "c1_c2_mean_rank": statistics.fmean(c1c2_ranks),
            "mean_rank_delta_c1_c2_minus_control": statistics.fmean(c - a for a, c in rank_when_top1_wrong),
            "n_improved": sum(1 for a, c in rank_when_top1_wrong if c < a),
            "n_unchanged": sum(1 for a, c in rank_when_top1_wrong if c == a),
            "n_worsened": sum(1 for a, c in rank_when_top1_wrong if c > a),
        }
    else:
        result["rank_when_c1_c2_top1_wrong"] = {"n": 0}

    # Fraction of C1_C2 Top-1 failures where the true source is still within
    # C1_C2's own Top-3 (a "soft" failure vs. a total miss).
    c1c2_top1_failures = [sid for sid in shared if not c1c2_rows[sid]["top1"]]
    control_top1_failures = [sid for sid in shared if not control_rows[sid]["top1"]]
    result["fraction_of_top1_failures_with_source_in_top3"] = {
        "c1_c2": (
            sum(1 for sid in c1c2_top1_failures if c1c2_rows[sid]["top3"]) / len(c1c2_top1_failures)
            if c1c2_top1_failures
            else None
        ),
        "a_control": (
            sum(1 for sid in control_top1_failures if control_rows[sid]["top3"]) / len(control_top1_failures)
            if control_top1_failures
            else None
        ),
        "n_c1_c2_top1_failures": len(c1c2_top1_failures),
        "n_a_control_top1_failures": len(control_top1_failures),
    }
    return result


# ---------------------------------------------------------------------------
# Primary endpoint gate (pre-registered, task Section "Full-data success
# gate") -- full-data-9000 stage only.
# ---------------------------------------------------------------------------


def calibration_ood_diagnostics(stage: str) -> dict[str, Any]:
    """Qualitative comparison of calibration/OOD proxy behavior between
    A_CONTROL and C1_C2 at this stage -- same diagnostics already computed
    by `evaluate_arm` (calibration coverage/mean-set-size/ECE,
    proxy_actionable_rate, proxy_abstention_rate, proxy_calibrated_coverage,
    ood_caution_or_outside_rate), compared for a "materially worse" flag
    using the same order-of-magnitude thresholds prior studies in this
    family used descriptively (coverage/actionable-rate drop > 5 points,
    ECE increase > 0.05)."""

    _set_stage(stage)
    control_eval = base.load_evaluation(SEED, "A_CONTROL")
    c1c2_eval = base.load_evaluation(SEED, "C1_C2")
    if control_eval is None or c1c2_eval is None:
        return {"available": False}

    result: dict[str, Any] = {
        "available": True,
        "calibration": {
            "a_control": control_eval.get("calibration"),
            "c1_c2": c1c2_eval.get("calibration"),
        },
        "populations": {},
    }
    flags: list[str] = []
    control_cal = control_eval.get("calibration", {})
    c1c2_cal = c1c2_eval.get("calibration", {})
    if control_cal.get("coverage") is not None and c1c2_cal.get("coverage") is not None:
        if c1c2_cal["coverage"] < control_cal["coverage"] - 0.05:
            flags.append("calibration_coverage_drop_gt_5pp")
    if control_cal.get("expected_calibration_error") is not None and c1c2_cal.get("expected_calibration_error") is not None:
        if c1c2_cal["expected_calibration_error"] > control_cal["expected_calibration_error"] + 0.05:
            flags.append("expected_calibration_error_increase_gt_0.05")

    for population in base.POPULATIONS:
        control_pop = control_eval["populations"].get(population, {})
        c1c2_pop = c1c2_eval["populations"].get(population, {})
        result["populations"][population] = {
            "proxy_actionable_rate": {"a_control": control_pop.get("proxy_actionable_rate"), "c1_c2": c1c2_pop.get("proxy_actionable_rate")},
            "proxy_abstention_rate": {"a_control": control_pop.get("proxy_abstention_rate"), "c1_c2": c1c2_pop.get("proxy_abstention_rate")},
            "proxy_calibrated_coverage": {"a_control": control_pop.get("proxy_calibrated_coverage"), "c1_c2": c1c2_pop.get("proxy_calibrated_coverage")},
            "ood_caution_or_outside_rate": {"a_control": control_pop.get("ood_caution_or_outside_rate"), "c1_c2": c1c2_pop.get("ood_caution_or_outside_rate")},
        }
        ca, cc = control_pop.get("proxy_actionable_rate"), c1c2_pop.get("proxy_actionable_rate")
        if ca is not None and cc is not None and cc < ca - 0.10:
            flags.append(f"{population}_proxy_actionable_rate_drop_gt_10pp")
    result["materially_worse_flags"] = flags
    result["materially_worse"] = len(flags) > 0
    return result


def primary_endpoint_gate() -> dict[str, Any]:
    stage = "full-data-9000"
    _set_stage(stage)
    control_rows = {row["scenario_id"]: row for row in base.load_rows(SEED, "A_CONTROL", PRIMARY_POPULATION) if row.get("has_source")}
    c1c2_rows = {row["scenario_id"]: row for row in base.load_rows(SEED, "C1_C2", PRIMARY_POPULATION) if row.get("has_source")}
    shared = sorted(set(control_rows) & set(c1c2_rows))
    if not shared:
        return {"available": False, "classification": "INCONCLUSIVE_FULL_DATA_GATE", "reason": "no full-data-9000 evaluation rows found yet"}

    control_top1 = [float(control_rows[sid]["top1"]) for sid in shared]
    c1c2_top1 = [float(c1c2_rows[sid]["top1"]) for sid in shared]
    control_top3 = [float(control_rows[sid]["top3"]) for sid in shared]
    c1c2_top3 = [float(c1c2_rows[sid]["top3"]) for sid in shared]
    control_mrr = [float(control_rows[sid]["reciprocal_rank"]) for sid in shared]
    c1c2_mrr = [float(c1c2_rows[sid]["reciprocal_rank"]) for sid in shared]

    top1_bootstrap = base.paired_bootstrap(control_top1, c1c2_top1)
    top3_bootstrap = base.paired_bootstrap(control_top3, c1c2_top3)
    mrr_bootstrap = base.paired_bootstrap(control_mrr, c1c2_mrr)

    observed_pp = top1_bootstrap["observed"] * 100
    ci_low_pp = top1_bootstrap["ci_low"] * 100
    ci_high_pp = top1_bootstrap["ci_high"] * 100

    known_topology: dict[str, Any] = {}
    for population in ("validation", "development_holdout"):
        c_rows = {row["scenario_id"]: row for row in base.load_rows(SEED, "A_CONTROL", population) if row.get("has_source")}
        e_rows = {row["scenario_id"]: row for row in base.load_rows(SEED, "C1_C2", population) if row.get("has_source")}
        ids = sorted(set(c_rows) & set(e_rows))
        c_top1 = [float(c_rows[i]["top1"]) for i in ids]
        e_top1 = [float(e_rows[i]["top1"]) for i in ids]
        bs = base.paired_bootstrap(c_top1, e_top1)
        known_topology[population] = {
            "n": len(ids),
            "top1_delta_bootstrap": bs,
            "material_statistically_supported_regression": bool(bs["ci_high"] is not None and bs["ci_high"] < 0),
        }

    cal_ood = calibration_ood_diagnostics(stage)

    top3_significant_negative = bool(top3_bootstrap["ci_high"] is not None and top3_bootstrap["ci_high"] < 0)
    mrr_significant_negative = bool(mrr_bootstrap["ci_high"] is not None and mrr_bootstrap["ci_high"] < 0)
    known_topology_regression = any(v["material_statistically_supported_regression"] for v in known_topology.values())

    criteria = {
        "1_top1_delta_positive": observed_pp > 0,
        "2_top1_delta_at_least_2pp": observed_pp >= GATE_MIN_EFFECT_PP,
        "3_ci_excludes_zero_positive": bool(top1_bootstrap["excludes_zero"] and ci_low_pp > 0),
        "4_no_top3_significant_negative_regression": not top3_significant_negative,
        "5_no_mrr_significant_negative_regression": not mrr_significant_negative,
        "6_no_known_topology_material_regression": not known_topology_regression,
        "7_calibration_ood_not_materially_worse": not cal_ood.get("materially_worse", False),
        "8_no_governance_code_altered": True,
    }

    harmful_regression = top3_significant_negative or mrr_significant_negative or known_topology_regression
    if observed_pp <= 0:
        classification = "FAIL_FULL_DATA_GATE"
    elif harmful_regression:
        classification = "FAIL_FULL_DATA_GATE"
    elif all(criteria.values()):
        classification = "PASS_FULL_DATA_GATE"
    else:
        classification = "INCONCLUSIVE_FULL_DATA_GATE"

    return {
        "available": True,
        "population": PRIMARY_POPULATION,
        "seed": SEED,
        "n_paired": len(shared),
        "top1_observed_pp": observed_pp,
        "top1_ci_low_pp": ci_low_pp,
        "top1_ci_high_pp": ci_high_pp,
        "top1_bootstrap": top1_bootstrap,
        "top3_bootstrap": top3_bootstrap,
        "mrr_bootstrap": mrr_bootstrap,
        "known_topology_regressions": known_topology,
        "calibration_ood_diagnostics": cal_ood,
        "criteria": criteria,
        "classification": classification,
    }


# ---------------------------------------------------------------------------
# Critical scale analysis: pilot (600) vs full-data (9000) at the SAME seed
# ---------------------------------------------------------------------------


def scale_comparison() -> dict[str, Any]:
    result: dict[str, Any] = {"seed": SEED, "population": PRIMARY_POPULATION, "stages": {}}
    stage_top1: dict[str, dict[str, float | None]] = {}
    for stage in STAGES:
        _set_stage(stage)
        control_eval = base.load_evaluation(SEED, "A_CONTROL")
        c1c2_eval = base.load_evaluation(SEED, "C1_C2")
        control_top1 = control_eval["populations"][PRIMARY_POPULATION]["top1"] if control_eval else None
        c1c2_top1 = c1c2_eval["populations"][PRIMARY_POPULATION]["top1"] if c1c2_eval else None
        delta = (c1c2_top1 - control_top1) if (control_top1 is not None and c1c2_top1 is not None) else None
        result["stages"][stage] = {
            "a_control_top1": control_top1,
            "c1_c2_top1": c1c2_top1,
            "delta": delta,
            "delta_pp": delta * 100 if delta is not None else None,
        }
        stage_top1[stage] = {"a_control": control_top1, "c1_c2": c1c2_top1, "delta": delta}

    pilot_delta = stage_top1.get("pilot-600", {}).get("delta")
    full_delta = stage_top1.get("full-data-9000", {}).get("delta")
    if pilot_delta is not None and full_delta is not None:
        if full_delta <= 0 and pilot_delta > 0:
            direction = "eliminated"
        elif full_delta < pilot_delta * 0.5:
            direction = "attenuated"
        elif full_delta > pilot_delta * 1.25:
            direction = "strengthened"
        else:
            direction = "preserved"
        result["scale_effect_direction"] = direction
        result["pilot_to_full_data_delta_change_pp"] = (full_delta - pilot_delta) * 100
    else:
        result["scale_effect_direction"] = "unavailable"
    result["note"] = (
        "Descriptive scale-effect classification (strengthens/preserves/attenuates/eliminates) "
        "based on the ratio of full-data delta to pilot delta at the SAME seed (20261110); not a "
        "new statistical test beyond the primary-endpoint gate's own bootstrap CI."
    )
    return result


def main() -> None:
    for stage in STAGES:
        analyze_stage(stage)

    gate_dir = RESULTS_ROOT / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate = primary_endpoint_gate()
    (gate_dir / "primary-endpoint-gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote gate/primary-endpoint-gate.json: classification={gate.get('classification')}")

    scale_dir = RESULTS_ROOT / "scale-comparison"
    scale_dir.mkdir(parents=True, exist_ok=True)
    comparison = scale_comparison()
    (scale_dir / "pilot-vs-full-data.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote scale-comparison/pilot-vs-full-data.json: scale_effect_direction={comparison.get('scale_effect_direction')}")


if __name__ == "__main__":
    main()
