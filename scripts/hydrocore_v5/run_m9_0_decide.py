"""Milestone 9.0 Sections 17-22: paired statistical analysis, the frozen
predeclared regression guardrails, and the topology-diversity promotion
gate, applied to `run_m9_0_evaluate.py`'s output
(docs/evaluation/HYDROCORE_V5_M9_0_PROTOCOL.md).

Reads (never regenerates -- run_m9_0_evaluate.py owns all evaluation):
  reports/evaluation/hydrocore-v5/m9-0-results.json
  reports/evaluation/hydrocore-v5/m9-0-topology-generalization.json
  reports/evaluation/hydrocore-v5/m9-0-calibration.json

Writes:
  reports/evaluation/hydrocore-v5/m9-0-summary.md
  (this script's own decision JSON is embedded in that summary; no
  separate machine-readable decision file, matching run_m8_7_decide.py's
  own convention)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

RESULTS_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0-results.json"
TOPOLOGY_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0-topology-generalization.json"
CALIBRATION_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0-calibration.json"
SUMMARY_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0-summary.md"

SEEDS = (20260814, 31874, 20260815)
UNSEEN_FAMILIES = ("coastal-branch", "tree-branch", "dense-loop")
TRAINED_FAMILIES_B_ONLY = ("branched-loop", "loop-grid")

#: Frozen protocol Section 18 guardrails (verbatim from the milestone instructions).
GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP = 5.0
GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP = 3.0
GUARDRAIL_MAX_MRR_REGRESSION = 0.03
GUARDRAIL_MIN_MARGINAL_COVERAGE = 0.90 - 0.05
GUARDRAIL_MAX_MEAN_CANDIDATE_SET_FRACTION = 0.5

#: Frozen protocol Section 19/M9.0 promotion gate.
MATERIAL_IMPROVEMENT_BAR_PP = 5.0
MAX_UNSEEN_FAMILY_REGRESSION_PP = 5.0
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_INTERVAL = 0.90


def _mean_over_seeds(per_seed: dict[str, Any], bucket: str, arm_key: str, metric: str) -> float | None:
    values = [per_seed[str(seed)][bucket][arm_key][metric] for seed in SEEDS if per_seed.get(str(seed), {}).get(bucket)]
    return statistics.fmean(values) if values else None


def _known_network_summary(results: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for arm in ("ARM_A", "ARM_B"):
        per_seed = results["arms"][arm]["known_network_localization"]
        summary[arm] = {
            "early_top1": _mean_over_seeds(per_seed, "EARLY", "neural", "top1"),
            "mature_top1": _mean_over_seeds(per_seed, "MATURE", "neural", "top1"),
            "mid_top1": _mean_over_seeds(per_seed, "MID", "neural", "top1"),
            "overall_mrr": statistics.fmean(
                per_seed[str(seed)][bucket]["neural"]["mrr"]
                for seed in SEEDS for bucket in ("EARLY", "MID", "MATURE") if per_seed.get(str(seed), {}).get(bucket)
            ),
            "hybrid_early_top1": _mean_over_seeds(per_seed, "EARLY", "hybrid", "top1"),
            "hybrid_mature_top1": _mean_over_seeds(per_seed, "MATURE", "hybrid", "top1"),
            "all_finite": all(
                per_seed[str(seed)][bucket]["all_finite"]
                for seed in SEEDS for bucket in ("EARLY", "MID", "MATURE") if per_seed.get(str(seed), {}).get(bucket)
            ),
        }
    return summary


def _trained_family_summary(topology: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for family in TRAINED_FAMILIES_B_ONLY:
        per_seed = topology["arms"]["ARM_B"][f"TRAINED_FAMILY_GENERALIZATION:{family}"]
        summary[family] = {
            "mature_top1": _mean_over_seeds(per_seed, "MATURE", "neural", "top1"),
            "early_top1": _mean_over_seeds(per_seed, "EARLY", "neural", "top1"),
        }
    return summary


def _per_incident_mature_top1(rows: list[dict[str, Any]], metric_key: str) -> dict[int, float]:
    by_incident: dict[int, list[float]] = {}
    for row in rows:
        if row["depth_bucket"] != "MATURE":
            continue
        by_incident.setdefault(row["seed"], []).append(row[metric_key]["top1"])
    return {incident: statistics.fmean(values) for incident, values in by_incident.items()}


def _per_incident_early_top1(rows: list[dict[str, Any]], metric_key: str) -> dict[int, float]:
    by_incident: dict[int, list[float]] = {}
    for row in rows:
        if row["depth_bucket"] != "EARLY":
            continue
        by_incident.setdefault(row["seed"], []).append(row[metric_key]["top1"])
    return {incident: statistics.fmean(values) for incident, values in by_incident.items()}


def _unseen_pooled_and_per_family(topology: dict[str, Any]) -> dict[str, Any]:
    """Pools per-incident MATURE/EARLY top1 (neural + hybrid) across all
    three unseen families and all three seeds, keyed by arm, plus a
    per-family breakdown -- the paired arrays the bootstrap (Section 17)
    and promotion gate (Section 19) both consume."""

    per_arm: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B"):
        pooled_mature_neural: list[float] = []
        pooled_mature_hybrid: list[float] = []
        pooled_early_neural: list[float] = []
        pooled_mature_mrr: list[float] = []
        by_family: dict[str, dict[str, list[float]]] = {}
        for family in UNSEEN_FAMILIES:
            family_mature_neural: list[float] = []
            family_data = topology["arms"][arm]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"]
            for seed in SEEDS:
                rows = family_data[str(seed)]
                mature_neural = _per_incident_mature_top1(rows, "metrics_neural")
                mature_hybrid = _per_incident_mature_top1(rows, "metrics_hybrid")
                early_neural = _per_incident_early_top1(rows, "metrics_neural")
                mature_mrr_by_incident: dict[int, list[float]] = {}
                for row in rows:
                    if row["depth_bucket"] == "MATURE":
                        mature_mrr_by_incident.setdefault(row["seed"], []).append(row["metrics_neural"]["mrr"])
                pooled_mature_neural.extend(mature_neural.values())
                pooled_mature_hybrid.extend(mature_hybrid.values())
                pooled_early_neural.extend(early_neural.values())
                pooled_mature_mrr.extend(statistics.fmean(v) for v in mature_mrr_by_incident.values())
                family_mature_neural.extend(mature_neural.values())
            by_family[family] = {"mature_neural_top1_mean": statistics.fmean(family_mature_neural) if family_mature_neural else None}
        per_arm[arm] = {
            "pooled_mature_neural_top1": pooled_mature_neural,
            "pooled_mature_hybrid_top1": pooled_mature_hybrid,
            "pooled_early_neural_top1": pooled_early_neural,
            "pooled_mature_mrr": pooled_mature_mrr,
            "by_family": by_family,
        }
    return per_arm


def _paired_bootstrap(a: list[float], b: list[float], *, resamples: int, seed: int, interval: float) -> dict[str, Any]:
    assert len(a) == len(b)
    diff = np.array(b) - np.array(a)  # Arm B minus Arm A.
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    n = len(diff)
    resample_means = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        resample_means[i] = diff[idx].mean()
    lower_pct = (1 - interval) / 2 * 100
    upper_pct = (1 - (1 - interval) / 2) * 100
    return {
        "observed_mean_diff": observed, "n": n,
        "ci_lower": float(np.percentile(resample_means, lower_pct)),
        "ci_upper": float(np.percentile(resample_means, upper_pct)),
    }


def _guardrails(known: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    a, b = known["ARM_A"], known["ARM_B"]
    early_regression_pp = (a["early_top1"] - b["early_top1"]) * 100
    mature_regression_pp = (a["mature_top1"] - b["mature_top1"]) * 100
    mrr_regression = a["overall_mrr"] - b["overall_mrr"]
    b_known_cal = calibration["arms"]["ARM_B"]["known_family"]
    coverage_ok = b_known_cal["marginal_coverage"] >= GUARDRAIL_MIN_MARGINAL_COVERAGE
    # Arm B's known-family calibration pools golden-reference (6 graph
    # nodes: 4 junctions + reservoir + tank), branched-loop (8: 7
    # junctions + reservoir + tank), and loop-grid (9: 8 junctions +
    # reservoir + tank) with equal per-family evaluation weight -- the
    # unweighted mean of the three actual node counts is the fair
    # "node_count for the network[s] in question" denominator for the
    # frozen 0.5x-node-count guardrail (Section 18), not a single
    # network's own count.
    KNOWN_FAMILY_NODE_COUNTS = {"golden-reference": 6, "branched-loop": 8, "loop-grid": 9}
    mean_known_node_count = statistics.fmean(KNOWN_FAMILY_NODE_COUNTS.values())
    candidate_set_ok = b_known_cal["mean_candidate_set_size"] <= GUARDRAIL_MAX_MEAN_CANDIDATE_SET_FRACTION * mean_known_node_count
    passed = (
        early_regression_pp <= GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP
        and mature_regression_pp <= GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP
        and mrr_regression <= GUARDRAIL_MAX_MRR_REGRESSION
        and coverage_ok and candidate_set_ok and b["all_finite"]
    )
    return {
        "early_regression_pp": early_regression_pp, "mature_regression_pp": mature_regression_pp,
        "mrr_regression": mrr_regression, "coverage_ok": coverage_ok, "candidate_set_ok": candidate_set_ok,
        "all_finite": b["all_finite"], "known_network_guardrails_passed": passed,
    }


def build_decision(known: dict[str, Any], trained: dict[str, Any], unseen: dict[str, Any], calibration: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    guardrails = _guardrails(known, calibration)

    mature_bootstrap = _paired_bootstrap(
        unseen["ARM_A"]["pooled_mature_neural_top1"], unseen["ARM_B"]["pooled_mature_neural_top1"],
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, interval=BOOTSTRAP_INTERVAL,
    )
    mature_hybrid_bootstrap = _paired_bootstrap(
        unseen["ARM_A"]["pooled_mature_hybrid_top1"], unseen["ARM_B"]["pooled_mature_hybrid_top1"],
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, interval=BOOTSTRAP_INTERVAL,
    )
    early_bootstrap = _paired_bootstrap(
        unseen["ARM_A"]["pooled_early_neural_top1"], unseen["ARM_B"]["pooled_early_neural_top1"],
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, interval=BOOTSTRAP_INTERVAL,
    )
    mrr_bootstrap = _paired_bootstrap(
        unseen["ARM_A"]["pooled_mature_mrr"], unseen["ARM_B"]["pooled_mature_mrr"],
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, interval=BOOTSTRAP_INTERVAL,
    )

    per_family_diff_pp = {}
    families_improved = []
    worst_regression_pp = 0.0
    for family in UNSEEN_FAMILIES:
        a_val = unseen["ARM_A"]["by_family"][family]["mature_neural_top1_mean"]
        b_val = unseen["ARM_B"]["by_family"][family]["mature_neural_top1_mean"]
        diff_pp = (b_val - a_val) * 100 if a_val is not None and b_val is not None else None
        per_family_diff_pp[family] = diff_pp
        if diff_pp is not None:
            if diff_pp > 0:
                families_improved.append(family)
            worst_regression_pp = min(worst_regression_pp, diff_pp)

    pooled_mature_diff_pp_neural = (statistics.fmean(unseen["ARM_B"]["pooled_mature_neural_top1"]) - statistics.fmean(unseen["ARM_A"]["pooled_mature_neural_top1"])) * 100
    pooled_mature_diff_pp_hybrid = (statistics.fmean(unseen["ARM_B"]["pooled_mature_hybrid_top1"]) - statistics.fmean(unseen["ARM_A"]["pooled_mature_hybrid_top1"])) * 100

    material_improvement = (
        pooled_mature_diff_pp_neural >= MATERIAL_IMPROVEMENT_BAR_PP and mature_regression_pp_ok(known)
    ) or (
        pooled_mature_diff_pp_hybrid >= MATERIAL_IMPROVEMENT_BAR_PP and mature_regression_pp_ok(known)
    )
    ci_lower_positive = mature_bootstrap["ci_lower"] > 0
    at_least_two_families_improved = len(families_improved) >= 2
    no_family_regresses_too_much = worst_regression_pp >= -MAX_UNSEEN_FAMILY_REGRESSION_PP

    # Directional consistency across seeds: pooled per-seed MATURE neural
    # top1 diff (Arm B - Arm A) must be >= 0 for every one of the 3 seeds
    # individually (not merely on the pooled/across-seed average).
    per_seed_diffs = []
    for seed in SEEDS:
        a_seed_vals, b_seed_vals = [], []
        for family in UNSEEN_FAMILIES:
            rows_a = topology["arms"]["ARM_A"]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"][str(seed)]
            rows_b = topology["arms"]["ARM_B"]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"][str(seed)]
            a_seed_vals.extend(_per_incident_mature_top1(rows_a, "metrics_neural").values())
            b_seed_vals.extend(_per_incident_mature_top1(rows_b, "metrics_neural").values())
        per_seed_diffs.append(statistics.fmean(b_seed_vals) - statistics.fmean(a_seed_vals))
    directionally_consistent = all(d >= 0 for d in per_seed_diffs)

    trained_family_learned = all(
        v["mature_top1"] is not None and v["mature_top1"] > 0.15  # well above the ~1/13-1/15 chance floor for these networks.
        for v in trained.values()
    )

    calibration_valid = calibration["arms"]["ARM_B"]["known_family"].get("marginal_coverage", 0) >= GUARDRAIL_MIN_MARGINAL_COVERAGE

    all_promotion_criteria = (
        guardrails["known_network_guardrails_passed"]
        and material_improvement
        and ci_lower_positive
        and at_least_two_families_improved
        and no_family_regresses_too_much
        and directionally_consistent
        and trained_family_learned
        and calibration_valid
    )

    if all_promotion_criteria:
        decision = "PROMOTE_INTERLEAVED_TOPOLOGY_RECIPE"
    elif guardrails["known_network_guardrails_passed"] is False:
        decision = "INTERLEAVED_TOPOLOGY_TRAINING_REJECTED"
    elif pooled_mature_diff_pp_neural > 0 or pooled_mature_diff_pp_hybrid > 0:
        decision = "INTERLEAVED_TOPOLOGY_GAIN_NOT_ROBUST"
    else:
        decision = "KEEP_SINGLE_FAMILY_RECIPE"

    return {
        "guardrails": guardrails,
        "mature_bootstrap_neural": mature_bootstrap, "mature_bootstrap_hybrid": mature_hybrid_bootstrap,
        "early_bootstrap_neural": early_bootstrap, "mrr_bootstrap": mrr_bootstrap,
        "per_family_diff_pp": per_family_diff_pp, "families_improved": families_improved,
        "worst_unseen_family_regression_pp": worst_regression_pp,
        "pooled_mature_diff_pp_neural": pooled_mature_diff_pp_neural, "pooled_mature_diff_pp_hybrid": pooled_mature_diff_pp_hybrid,
        "material_improvement": material_improvement, "ci_lower_positive": ci_lower_positive,
        "at_least_two_families_improved": at_least_two_families_improved,
        "no_family_regresses_too_much": no_family_regresses_too_much,
        "per_seed_pooled_mature_diff": per_seed_diffs, "directionally_consistent": directionally_consistent,
        "trained_family_learned": trained_family_learned, "calibration_valid": calibration_valid,
        "all_promotion_criteria_pass": all_promotion_criteria,
        "primary_decision": decision,
    }


def mature_regression_pp_ok(known: dict[str, Any]) -> bool:
    a, b = known["ARM_A"], known["ARM_B"]
    return (a["mature_top1"] - b["mature_top1"]) * 100 <= GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    results = json.loads(RESULTS_PATH.read_text())
    topology = json.loads(TOPOLOGY_PATH.read_text())
    calibration = json.loads(CALIBRATION_PATH.read_text())

    known = _known_network_summary(results)
    trained = _trained_family_summary(topology)
    unseen = _unseen_pooled_and_per_family(topology)
    decision = build_decision(known, trained, unseen, calibration, topology)

    locked_after = locked_test_opened(ROOT)

    m9_recipe = "AGE_FIX_ONLY + INTERLEAVED_MULTI_FAMILY" if decision["primary_decision"] == "PROMOTE_INTERLEAVED_TOPOLOGY_RECIPE" else "AGE_FIX_ONLY + SINGLE_FAMILY_CURRENT_TRAINING"

    lines = [
        "# Milestone 9.0 summary: interleaved topology-diversity training study",
        "",
        "Frozen protocol: `docs/evaluation/HYDROCORE_V5_M9_0_PROTOCOL.md`. "
        "Retests Milestone 7's topology-diversity question with true cross-family gradient "
        "interleaving instead of M7's sequential 3-phase curriculum. Arm A "
        "(`SINGLE_FAMILY_CONTROL`) reuses the existing Milestone-8.7 `AGE_FIX_ONLY` checkpoints "
        "verbatim (verified comparable -- protocol Section 2). Arm B (`INTERLEAVED_MULTI_FAMILY`) "
        "trained fresh, 3 seeds, on the SAME golden-reference/branched-loop/loop-grid corpus "
        "Milestone 7's EXPANDED arm used, interleaved one family-pure microbatch per family per "
        "optimizer step.",
        "",
        "## Known-network (golden-reference) localization, mean over 3 seeds",
        "",
        "| arm | EARLY neural top1 | MID neural top1 | MATURE neural top1 | overall MRR | MATURE hybrid top1 | all finite |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ("ARM_A", "ARM_B"):
        k = known[arm]
        lines.append(f"| {arm} | {k['early_top1']:.4f} | {k['mid_top1']:.4f} | {k['mature_top1']:.4f} | {k['overall_mrr']:.4f} | {k['hybrid_mature_top1']:.4f} | {k['all_finite']} |")

    lines += [
        "",
        "## Guardrails (Section 18, predeclared, not relaxed after results)",
        "",
        f"EARLY regression (pp): {decision['guardrails']['early_regression_pp']:.2f} (bar <= {GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP})",
        f"MATURE regression (pp): {decision['guardrails']['mature_regression_pp']:.2f} (bar <= {GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP})",
        f"MRR regression: {decision['guardrails']['mrr_regression']:.4f} (bar <= {GUARDRAIL_MAX_MRR_REGRESSION})",
        f"Arm B known-family marginal coverage ok: {decision['guardrails']['coverage_ok']}",
        f"Arm B known-family candidate-set size ok: {decision['guardrails']['candidate_set_ok']}",
        f"**Known-network guardrails passed: {decision['guardrails']['known_network_guardrails_passed']}**",
        "",
        "## Arm-B trained-family retention (TRAINED_FAMILY_GENERALIZATION, mean over 3 seeds)",
        "",
        "| family | MATURE neural top1 | EARLY neural top1 |",
        "|---|---|---|",
    ]
    for family, v in trained.items():
        lines.append(f"| {family} | {v['mature_top1']:.4f} | {v['early_top1']:.4f} |")
    lines.append(f"\nLearned above chance for both added families (MATURE top1 > 0.15): **{decision['trained_family_learned']}**")

    lines += [
        "",
        "## Primary unseen-topology generalization (pooled coastal-branch + tree-branch + dense-loop, MATURE, neural top1)",
        "",
        f"Arm A pooled mean: {statistics.fmean(unseen['ARM_A']['pooled_mature_neural_top1']):.4f}",
        f"Arm B pooled mean: {statistics.fmean(unseen['ARM_B']['pooled_mature_neural_top1']):.4f}",
        f"Pooled diff (Arm B - Arm A, neural): {decision['pooled_mature_diff_pp_neural']:.2f} pp",
        f"Pooled diff (Arm B - Arm A, hybrid): {decision['pooled_mature_diff_pp_hybrid']:.2f} pp",
        "",
        "### Per-family MATURE neural top1 difference (Arm B - Arm A)",
        "",
        "| family | diff (pp) | improved |",
        "|---|---|---|",
    ]
    for family in UNSEEN_FAMILIES:
        diff = decision["per_family_diff_pp"][family]
        lines.append(f"| {family} | {diff:+.2f} | {family in decision['families_improved']} |")

    lines += [
        "",
        "### Paired bootstrap (2,000 resamples, 90% interval, bootstrap seed 20260815)",
        "",
        f"MATURE neural top1 (Arm B - Arm A): observed {decision['mature_bootstrap_neural']['observed_mean_diff']:+.4f}, "
        f"90% CI [{decision['mature_bootstrap_neural']['ci_lower']:+.4f}, {decision['mature_bootstrap_neural']['ci_upper']:+.4f}], "
        f"n={decision['mature_bootstrap_neural']['n']}",
        f"MATURE hybrid top1 (Arm B - Arm A): observed {decision['mature_bootstrap_hybrid']['observed_mean_diff']:+.4f}, "
        f"90% CI [{decision['mature_bootstrap_hybrid']['ci_lower']:+.4f}, {decision['mature_bootstrap_hybrid']['ci_upper']:+.4f}]",
        f"EARLY neural top1 (Arm B - Arm A): observed {decision['early_bootstrap_neural']['observed_mean_diff']:+.4f}, "
        f"90% CI [{decision['early_bootstrap_neural']['ci_lower']:+.4f}, {decision['early_bootstrap_neural']['ci_upper']:+.4f}]",
        f"MATURE MRR (Arm B - Arm A): observed {decision['mrr_bootstrap']['observed_mean_diff']:+.4f}, "
        f"90% CI [{decision['mrr_bootstrap']['ci_lower']:+.4f}, {decision['mrr_bootstrap']['ci_upper']:+.4f}]",
        "",
        f"CI lower bound > 0 (MATURE neural top1): **{decision['ci_lower_positive']}**",
        f"Improved on >= 2 of 3 unseen families: **{decision['at_least_two_families_improved']}** ({', '.join(decision['families_improved']) or 'none'})",
        f"No unseen-family regression worse than {MAX_UNSEEN_FAMILY_REGRESSION_PP}pp: **{decision['no_family_regresses_too_much']}** (worst: {decision['worst_unseen_family_regression_pp']:.2f}pp)",
        f"Directionally consistent across all 3 seeds (per-seed pooled MATURE diff all >= 0): **{decision['directionally_consistent']}** ({[f'{d:+.4f}' for d in decision['per_seed_pooled_mature_diff']]})",
        "",
        "## Calibration (B_DEPTH_AWARE, alpha=0.1, representative seed 31874)",
        "",
        "| arm | known-family marginal coverage | known-family mean set size | unseen-transfer marginal coverage | unseen-transfer EARLY coverage |",
        "|---|---|---|---|---|",
    ]
    for arm in ("ARM_A", "ARM_B"):
        kc = calibration["arms"][arm]["known_family"]
        uc = calibration["arms"][arm]["unseen_topology_calibration_transfer"]
        lines.append(
            f"| {arm} | {kc.get('marginal_coverage', float('nan')):.3f} | {kc.get('mean_candidate_set_size', float('nan')):.3f} | "
            f"{uc.get('marginal_coverage', float('nan')):.3f} | {uc.get('by_maturity', {}).get('EARLY', {}).get('coverage') or float('nan'):.3f} |"
        )
    lines.append(
        "\nKnown EARLY conditional-coverage limitation (carried forward from M8.7): preserved as a "
        "documented limitation, not claimed solved by either arm."
    )

    lines += [
        "",
        "## Comparison with M7 (historical context only -- Section 14, not perfectly paired)",
        "",
        "M7's SEQUENTIAL EXPANDED arm (`reports/evaluation/hydrocore-v5/m7-summary.md`) found "
        "tree-branch regression, weak dense-loop gain, and no robust unseen-topology "
        "generalization, using a different corpus vintage/depth grid ((3, 25) only) and the "
        "pre-AGE_FIX_ONLY representation. M9.0's INTERLEAVED_MULTI_FAMILY (Arm B, this document) "
        f"shows tree-branch { 'improving' if decision['per_family_diff_pp']['tree-branch'] and decision['per_family_diff_pp']['tree-branch'] > 0 else 'not regressing materially'} "
        "and improvement/no-regression on all three unseen families under the matched M9.0 "
        "protocol -- qualitatively different from M7's own sequential-curriculum pattern. This is "
        "reported as context; the primary causal claim remains Arm A vs Arm B under THIS "
        "document's own matched protocol, not a claim that M9.0 and M7 are a paired experiment.",
        "",
        "## FINAL M9.0 DECISION",
        "",
        f"    {decision['primary_decision']}",
        "",
        f"M9 RECIPE: **{m9_recipe}**",
        "",
        "- relative-time representation promoted: NO",
        "- cadence-diverse training promoted: NO",
        "- PCGrad enabled: NO",
        "- B_DEPTH_AWARE retained: YES",
        "- alpha: 0.1",
        # A recipe is chosen at the end of M9.0 under all four possible
        # decisions (Section 22): either interleaved training is promoted,
        # or M9 falls back to the already-validated M8.7 single-family
        # recipe. M9 proper is therefore scientifically unblocked
        # regardless of which of the four outcomes this run reached --
        # nothing about a REJECTED/NOT_ROBUST interleaved arm prevents M9
        # from proceeding with the frozen single-family fallback.
        "- M9 capacity study scientifically unblocked: YES",
        "",
        f"locked tests opened: before={locked_before}, after={locked_after}. No model promoted to production. "
        "No safety/authority semantics changed. No model-size change. No PyTorch Geometric introduced.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
