"""Milestone 9.0b Sections 9-17, 24: safety/actionability gates, the M9.0a
anomaly diagnosis, and the final Outcome A-E promotion decision, applied to
`run_m9_0b_evaluate.py`'s output (docs/evaluation/HYDROCORE_V5_M9_0B_PROTOCOL.md).

Reads (never regenerates):
  reports/evaluation/hydrocore-v5/m9-0b-results.json
  reports/evaluation/hydrocore-v5/m9-0b-group-support.json
  reports/evaluation/hydrocore-v5/m9-0b-unseen-transfer.json
  reports/evaluation/hydrocore-v5/m9-0b-calibration-by-seed.json

Writes:
  reports/evaluation/hydrocore-v5/m9-0b-summary.md
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

RESULTS_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0b-results.json"
GROUP_SUPPORT_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0b-group-support.json"
UNSEEN_TRANSFER_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0b-unseen-transfer.json"
SUMMARY_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0b-summary.md"

SEEDS = (20260814, 31874, 20260815)
KNOWN_FAMILIES = ("golden-reference", "branched-loop", "loop-grid")
MATURITY_BUCKETS = ("EARLY", "MID", "MATURE")
UNSEEN_FAMILIES = ("coastal-branch", "tree-branch", "dense-loop")
SCHEME_NAMES = ("CURRENT_FAMILY_DEPTH", "POOLED_DEPTH_AWARE", "BROAD_FALLBACK_CONTROL", "HIERARCHICAL_CONSERVATIVE")

MIN_ACCEPTABLE_COVERAGE = 0.85
NORMALIZED_SET_SIZE_BAR = 0.5
MINIMUM_FAMILY_SUPPORT = 10


def _scheme_seed_safety(per_seed: dict[str, Any]) -> dict[str, Any]:
    """Section 10: safety_valid for one scheme/seed."""
    overall_ok = per_seed["overall"].get("marginal_coverage", 0.0) >= MIN_ACCEPTABLE_COVERAGE
    maturity_results = {}
    maturity_ok = True
    for bucket in MATURITY_BUCKETS:
        cov = per_seed["by_maturity"].get(bucket, {}).get("marginal_coverage")
        maturity_results[bucket] = cov
        if cov is None or cov < MIN_ACCEPTABLE_COVERAGE:
            maturity_ok = False
    family_results = {}
    family_ok = True
    for family in KNOWN_FAMILIES:
        entry = per_seed["by_family"].get(family, {})
        n = entry.get("n", 0)
        cov = entry.get("marginal_coverage")
        family_results[family] = {"n": n, "marginal_coverage": cov, "adequate_support": n >= MINIMUM_FAMILY_SUPPORT}
        if n >= MINIMUM_FAMILY_SUPPORT and (cov is None or cov < MIN_ACCEPTABLE_COVERAGE):
            family_ok = False
    return {
        "overall_marginal_coverage": per_seed["overall"].get("marginal_coverage"),
        "overall_ok": overall_ok, "maturity": maturity_results, "maturity_ok": maturity_ok,
        "family": family_results, "family_ok": family_ok,
        "safety_valid": overall_ok and maturity_ok and family_ok,
        "mean_normalized_set_size": per_seed["overall"].get("mean_normalized_set_size"),
        "singleton_rate": per_seed["overall"].get("singleton_rate"),
    }


def _scheme_summary(results: dict[str, Any], scheme: str) -> dict[str, Any]:
    per_seed_safety = {seed: _scheme_seed_safety(results["schemes"][scheme]["per_seed"][str(seed)]) for seed in SEEDS}
    all_seeds_safety_valid = all(v["safety_valid"] for v in per_seed_safety.values())
    normalized_sizes = [v["mean_normalized_set_size"] for v in per_seed_safety.values() if v["mean_normalized_set_size"] is not None]
    candidate_set_ok_per_seed = {seed: (v["mean_normalized_set_size"] is not None and v["mean_normalized_set_size"] <= NORMALIZED_SET_SIZE_BAR) for seed, v in per_seed_safety.items()}
    candidate_set_guardrail_pass = all(candidate_set_ok_per_seed.values())
    return {
        "per_seed": per_seed_safety,
        "all_seeds_safety_valid": all_seeds_safety_valid,
        "candidate_set_guardrail_pass": candidate_set_guardrail_pass,
        "mean_normalized_set_size_across_seeds": statistics.fmean(normalized_sizes) if normalized_sizes else None,
        "operationally_valid": all_seeds_safety_valid and candidate_set_guardrail_pass,
        "mean_singleton_rate": statistics.fmean(v["singleton_rate"] for v in per_seed_safety.values() if v["singleton_rate"] is not None),
    }


def _diagnose(group_support: dict[str, Any]) -> dict[str, Any]:
    """Section 13: compare Scheme A's family:depth quantiles to Scheme B's
    pooled-depth quantiles, and Scheme C's condition/global quantiles, to
    distinguish hypotheses A-E."""
    family_depth_minus_pooled: list[float] = []
    per_cell: dict[str, list[float]] = {}
    for seed in SEEDS:
        current = group_support["schemes"]["CURRENT_FAMILY_DEPTH"]["per_seed"][str(seed)]["network_groups"]
        pooled = group_support["schemes"]["POOLED_DEPTH_AWARE"]["per_seed"][str(seed)]["network_groups"]
        for key, entry in current.items():
            family, bucket = key.split(":", 1)
            if bucket not in pooled:
                continue
            diff = entry["quantile"] - pooled[bucket]["quantile"]
            family_depth_minus_pooled.append(diff)
            per_cell.setdefault(key, []).append(diff)

    broad_quantiles: list[float] = []
    for seed in SEEDS:
        broad = group_support["schemes"]["BROAD_FALLBACK_CONTROL"]["per_seed"][str(seed)]
        if broad["global_quantile"] is not None:
            broad_quantiles.append(broad["global_quantile"])
        for cond_entry in broad["condition_groups"].values():
            broad_quantiles.append(cond_entry["quantile"])

    pooled_quantiles: list[float] = []
    for seed in SEEDS:
        pooled = group_support["schemes"]["POOLED_DEPTH_AWARE"]["per_seed"][str(seed)]["network_groups"]
        pooled_quantiles.extend(entry["quantile"] for entry in pooled.values())

    family_depth_systematically_smaller = bool(family_depth_minus_pooled) and statistics.fmean(family_depth_minus_pooled) < 0
    broad_more_conservative_than_pooled = (
        bool(broad_quantiles) and bool(pooled_quantiles) and statistics.fmean(broad_quantiles) > statistics.fmean(pooled_quantiles)
    )
    return {
        "mean_family_depth_quantile_minus_pooled_quantile": statistics.fmean(family_depth_minus_pooled) if family_depth_minus_pooled else None,
        "per_cell_mean_diff": {key: statistics.fmean(values) for key, values in per_cell.items()},
        "family_depth_quantiles_systematically_smaller_than_pooled": family_depth_systematically_smaller,
        "mean_broad_fallback_quantile": statistics.fmean(broad_quantiles) if broad_quantiles else None,
        "mean_pooled_depth_quantile": statistics.fmean(pooled_quantiles) if pooled_quantiles else None,
        "broad_fallback_more_conservative_than_pooled_depth": broad_more_conservative_than_pooled,
    }


def _aps_raps_gate(scheme_summaries: dict[str, Any]) -> bool:
    """Section 24: only YES if at least one scheme passes ALL coverage
    gates but EVERY coverage-passing scheme fails the candidate-set
    guardrail solely because sets are too broad."""
    coverage_passing = [name for name, s in scheme_summaries.items() if s["all_seeds_safety_valid"]]
    if not coverage_passing:
        return False
    return all(not scheme_summaries[name]["candidate_set_guardrail_pass"] for name in coverage_passing)


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    results = json.loads(RESULTS_PATH.read_text())
    group_support = json.loads(GROUP_SUPPORT_PATH.read_text())
    unseen_transfer = json.loads(UNSEEN_TRANSFER_PATH.read_text())

    scheme_summaries = {scheme: _scheme_summary(results, scheme) for scheme in SCHEME_NAMES}
    diagnosis = _diagnose(group_support)
    aps_raps_warranted = _aps_raps_gate(scheme_summaries)

    # --- Selection logic (Section 16, Outcomes A-E) -----------------------
    pooled = scheme_summaries["POOLED_DEPTH_AWARE"]
    hierarchical = scheme_summaries["HIERARCHICAL_CONSERVATIVE"]

    operationally_valid = {name: s["operationally_valid"] for name, s in scheme_summaries.items()}
    decision: str
    selected_scheme: str | None

    pooled_valid = operationally_valid["POOLED_DEPTH_AWARE"]
    hierarchical_valid = operationally_valid["HIERARCHICAL_CONSERVATIVE"]
    broad_valid = operationally_valid["BROAD_FALLBACK_CONTROL"]

    # Efficiency comparison only among schemes that are actually valid.
    hierarchical_materially_smaller = (
        pooled_valid and hierarchical_valid
        and pooled["mean_normalized_set_size_across_seeds"] is not None
        and hierarchical["mean_normalized_set_size_across_seeds"] is not None
        and hierarchical["mean_normalized_set_size_across_seeds"] < pooled["mean_normalized_set_size_across_seeds"] - 0.02
    )

    if pooled_valid and not hierarchical_materially_smaller:
        decision = "PROMOTE_POOLED_DEPTH_AWARE"
        selected_scheme = "POOLED_DEPTH_AWARE"
    elif hierarchical_valid and hierarchical_materially_smaller:
        decision = "PROMOTE_HIERARCHICAL_CONSERVATIVE"
        selected_scheme = "HIERARCHICAL_CONSERVATIVE"
    elif broad_valid:
        decision = "PROMOTE_BROAD_CONSERVATIVE_CALIBRATION"
        selected_scheme = "BROAD_FALLBACK_CONTROL"
    elif hierarchical_valid:
        decision = "PROMOTE_HIERARCHICAL_CONSERVATIVE"
        selected_scheme = "HIERARCHICAL_CONSERVATIVE"
    else:
        decision = "INTERLEAVED_PREDICTOR_CALIBRATION_NOT_RESOLVED"
        selected_scheme = None

    # A second, simplicity-driven pass: if more than one scheme is
    # operationally valid and their efficiency differs only trivially,
    # prefer the simplest (Outcome D), in the predeclared order
    # POOLED_DEPTH_AWARE > BROAD_FALLBACK_CONTROL > HIERARCHICAL_CONSERVATIVE.
    valid_schemes = [name for name in ("POOLED_DEPTH_AWARE", "BROAD_FALLBACK_CONTROL", "HIERARCHICAL_CONSERVATIVE") if operationally_valid[name]]
    if len(valid_schemes) > 1 and not hierarchical_materially_smaller:
        decision = "PROMOTE_SIMPLEST_VALID_CALIBRATOR"
        selected_scheme = valid_schemes[0]

    locked_after = locked_test_opened(ROOT)

    if decision == "INTERLEAVED_PREDICTOR_CALIBRATION_NOT_RESOLVED":
        m9_1_representation = "AGE_FIX_ONLY"
        m9_1_topology = "SINGLE_FAMILY_CURRENT_TRAINING"
        m9_1_calibration = "B_DEPTH_AWARE (existing)"
        interleaved_promoted = False
    else:
        m9_1_representation = "AGE_FIX_ONLY"
        m9_1_topology = "STEP_MATCHED_INTERLEAVED_MULTI_FAMILY"
        m9_1_calibration = selected_scheme or "N/A"
        interleaved_promoted = True

    lines = [
        "# Milestone 9.0b summary: multi-topology calibration grouping study",
        "",
        "Frozen protocol: `docs/evaluation/HYDROCORE_V5_M9_0B_PROTOCOL.md`. Tests whether a "
        "different Mondrian grouping/fallback construction over the unmodified "
        "`SplitConformalCalibrator`, at fixed alpha=0.1 and the frozen (unretrained) M9.0a "
        "`ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY` predictor checkpoints, can restore "
        "safety-valid known-family coverage.",
        "",
        "## Scheme results (safety + actionability, per seed)",
        "",
    ]
    for scheme in SCHEME_NAMES:
        s = scheme_summaries[scheme]
        lines.append(f"### {scheme}")
        lines.append("")
        lines.append("| seed | marginal | EARLY | MID | MATURE | mean norm. set size | safety valid |")
        lines.append("|---|---|---|---|---|---|---|")
        for seed in SEEDS:
            v = s["per_seed"][seed]
            lines.append(
                f"| {seed} | {v['overall_marginal_coverage']:.4f} | {v['maturity'].get('EARLY') or float('nan'):.4f} | "
                f"{v['maturity'].get('MID') or float('nan'):.4f} | {v['maturity'].get('MATURE') or float('nan'):.4f} | "
                f"{v['mean_normalized_set_size']:.4f} | {v['safety_valid']} |"
            )
        lines.append("")
        lines.append(f"All 3 seeds safety valid: **{s['all_seeds_safety_valid']}** | Candidate-set guardrail pass: **{s['candidate_set_guardrail_pass']}** | "
                      f"Operationally valid: **{s['operationally_valid']}** | mean normalized set size (across seeds): "
                      f"{s['mean_normalized_set_size_across_seeds']:.4f}" if s['mean_normalized_set_size_across_seeds'] is not None else "N/A")
        lines.append("")

    lines += [
        "## Trained-family check (per scheme, min marginal coverage across seeds)",
        "",
        "| scheme | golden-reference | branched-loop | loop-grid |",
        "|---|---|---|---|",
    ]
    for scheme in SCHEME_NAMES:
        s = scheme_summaries[scheme]
        mins = {}
        for family in KNOWN_FAMILIES:
            values = [v["family"][family]["marginal_coverage"] for v in s["per_seed"].values() if v["family"][family]["marginal_coverage"] is not None]
            mins[family] = min(values) if values else None
        lines.append(f"| {scheme} | {mins['golden-reference']:.4f} | {mins['branched-loop']:.4f} | {mins['loop-grid']:.4f} |" if all(v is not None for v in mins.values()) else f"| {scheme} | n/a | n/a | n/a |")

    lines += [
        "",
        "## Diagnosis of the M9.0a anomaly (Section 13)",
        "",
        f"Mean (family:depth quantile - pooled-depth quantile), across all cells/seeds: "
        f"{diagnosis['mean_family_depth_quantile_minus_pooled_quantile']:+.4f}" if diagnosis['mean_family_depth_quantile_minus_pooled_quantile'] is not None else "n/a",
        f"family:depth quantiles systematically SMALLER than pooled-depth (Hypothesis A support): "
        f"**{diagnosis['family_depth_quantiles_systematically_smaller_than_pooled']}**",
        f"Mean BROAD_FALLBACK_CONTROL quantile: {diagnosis['mean_broad_fallback_quantile']:.4f}" if diagnosis['mean_broad_fallback_quantile'] is not None else "n/a",
        f"Mean POOLED_DEPTH_AWARE quantile: {diagnosis['mean_pooled_depth_quantile']:.4f}" if diagnosis['mean_pooled_depth_quantile'] is not None else "n/a",
        f"Broad-fallback quantiles more conservative than pooled-depth (Hypothesis C support): "
        f"**{diagnosis['broad_fallback_more_conservative_than_pooled_depth']}**",
        "",
        "## Unseen-topology calibration transfer (diagnostic only, not used to select a scheme)",
        "",
        "| scheme | family | mean marginal coverage (3 seeds) |",
        "|---|---|---|",
    ]
    for scheme in SCHEME_NAMES:
        for family in UNSEEN_FAMILIES:
            values = [unseen_transfer["schemes"][scheme]["per_seed"][str(seed)][family].get("marginal_coverage") for seed in SEEDS]
            values = [v for v in values if v is not None]
            mean_cov = statistics.fmean(values) if values else None
            lines.append(f"| {scheme} | {family} | {mean_cov:.4f} |" if mean_cov is not None else f"| {scheme} | {family} | n/a |")

    lines += [
        "",
        "## FINAL M9.0b DECISION",
        "",
        f"    {decision}",
        "",
        f"Selected scheme: **{selected_scheme or 'NONE'}**",
        "",
        f"APS_RAPS_CALIBRATION_FOLLOWUP_WARRANTED: **{aps_raps_warranted}**",
        "",
        "## M9.1 RECIPE",
        "",
        f"- representation: {m9_1_representation}",
        f"- topology training: {m9_1_topology}",
        f"- calibration: {m9_1_calibration}",
        "- alpha: 0.1",
        f"- interleaved predictor operationally promoted: {'YES' if interleaved_promoted else 'NO'}",
        "- M9_1_SCIENTIFICALLY_UNBLOCKED: YES",
        "",
        f"locked tests opened: before={locked_before}, after={locked_after}. No predictor retrained. "
        "No production calibration code changed. No M9.1/M9 capacity work begun.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "decision": decision, "selected_scheme": selected_scheme,
        "aps_raps_warranted": aps_raps_warranted,
        "scheme_summaries": {
            name: {"all_seeds_safety_valid": s["all_seeds_safety_valid"], "candidate_set_guardrail_pass": s["candidate_set_guardrail_pass"], "operationally_valid": s["operationally_valid"]}
            for name, s in scheme_summaries.items()
        },
        "diagnosis": diagnosis,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
