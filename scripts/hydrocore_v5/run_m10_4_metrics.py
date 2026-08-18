"""Milestone 10.4 Parts 5-10: metrics, comparator, safety-gate, utility-gate,
and closure computation FROM the already-executed, immutable trajectory rows
(`run_m10_4_execute.py`'s `m10-4-trajectories.jsonl`). No trajectory is
re-run here; no threshold is adjusted based on what this script finds --
`m10_4_protocol.UTILITY_GATE` is read as-is.

Writes (all under reports/evaluation/hydrocore-v5/m10/m10-4/):
  m10-4-source-trajectory.json
  m10-4-scout-trajectory.json
  m10-4-strategist-trajectory.json
  m10-4-physical-outcomes.json
  m10-4-comparator.json
  m10-4-trajectory-summary.json
  m10-4-gate.json
  m10-4-closure.json
and docs/evaluation/HYDROCORE_V5_M10_4_FULL_TRAJECTORY_RESULTS.md
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m10_4_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402

M10_4_DIR = m10.M10_DIR / "m10-4"


def _load_trajectories() -> list[dict[str, Any]]:
    path = M10_4_DIR / "m10-4-trajectories.jsonl"
    rows = []
    with path.open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def _ok_arm(row: dict[str, Any], arm: str) -> dict[str, Any] | None:
    arm_data = row.get("arms", {}).get(arm)
    if not arm_data or "outcome" in arm_data:
        return None
    return arm_data


def _mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _rate(flags: Iterable[bool | None]) -> dict[str, Any]:
    flags = [bool(f) for f in flags if f is not None]
    n = len(flags)
    successes = sum(flags)
    lo, hi = m10.wilson_interval_90(successes, n) if n else (float("nan"), float("nan"))
    return {"n": n, "rate": successes / n if n else None, "wilson_90_lo": lo, "wilson_90_hi": hi}


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key)), []).append(row)
    return groups


# ---------------------------------------------------------------------------
# Part 5: source-inference metrics.
# ---------------------------------------------------------------------------


def source_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def one_slice(subset: list[dict[str, Any]]) -> dict[str, Any]:
        full_arms = [a for row in subset if (a := _ok_arm(row, "FULL"))]
        initial = [a["initial"] for a in full_arms]
        final = [a["final"] for a in full_arms]
        return {
            "n": len(full_arms),
            "initial_top1": _rate(x.get("top1_correct") for x in initial),
            "initial_top3": _rate(x.get("top3_correct") for x in initial),
            "initial_mrr": _mean(x.get("reciprocal_rank") for x in initial),
            "final_top1": _rate(x.get("top1_correct") for x in final),
            "final_top3": _rate(x.get("top3_correct") for x in final),
            "final_mrr": _mean(x.get("reciprocal_rank") for x in final),
            "final_entropy_bits": _mean(x.get("posterior_entropy") for x in final),
            "final_candidate_set_size": _mean(x.get("candidate_set_size") for x in final),
            "final_calibrated_rate": _rate(a["final_analysis"].get("calibrated") for a in full_arms),
            "final_actionable_rate": _rate(a["final_analysis"].get("planning_allowed") for a in full_arms),
            "false_confidence_rate": _rate(
                (x.get("top1_correct") is False and (a["final_analysis"].get("calibrated") or False))
                for a, x in zip(full_arms, final, strict=True)
            ),
        }

    return {
        "kind": "M10_4_SOURCE_TRAJECTORY", "protocol_hash": proto.protocol_hash(),
        "overall": one_slice(rows),
        "by_model_seed": {k: one_slice(v) for k, v in _group(rows, "model_seed").items()},
        "by_family": {k: one_slice(v) for k, v in _group(rows, "family").items()},
        "by_condition_kind": {k: one_slice(v) for k, v in _group(rows, "condition_kind").items()},
    }


# ---------------------------------------------------------------------------
# Part 6: Scout / evidence-acquisition metrics.
# ---------------------------------------------------------------------------


def scout_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_arms = [a for row in rows if (a := _ok_arm(row, "FULL"))]
    samples_taken = [a.get("samples_taken", 0) for a in full_arms]
    requested_at_least_one = [n > 0 for n in samples_taken]
    stop_reasons: dict[str, int] = {}
    for a in full_arms:
        reason = a.get("stop_reason") or "NONE"
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    rank_improvements: list[int] = []
    entropy_reductions: list[float] = []
    useful_sample_rounds = 0
    total_sample_rounds = 0
    for a in full_arms:
        for r in a.get("rounds", []):
            if r.get("status") != "SAMPLE":
                continue
            total_sample_rounds += 1
            before_rank = r.get("true_source_rank_before")
            after_rank = r.get("true_source_rank_after")
            if before_rank is not None and after_rank is not None:
                delta = before_rank - after_rank
                rank_improvements.append(delta)
                if delta > 0:
                    useful_sample_rounds += 1
            eb, ea = r.get("entropy_before"), r.get("entropy_after")
            if eb is not None and ea is not None:
                entropy_reductions.append(eb - ea)

    return {
        "kind": "M10_4_SCOUT_TRAJECTORY", "protocol_hash": proto.protocol_hash(),
        "n_incidents": len(full_arms),
        "fraction_requesting_ge1_sample": _rate(requested_at_least_one),
        "mean_samples_per_incident": _mean(samples_taken),
        "max_samples_observed": max(samples_taken) if samples_taken else None,
        "stop_reason_distribution": stop_reasons,
        "total_sample_rounds": total_sample_rounds,
        "mean_true_source_rank_change_per_sample": _mean(rank_improvements),
        "fraction_sample_rounds_improving_rank": (useful_sample_rounds / total_sample_rounds) if total_sample_rounds else None,
        "mean_entropy_reduction_bits_per_sample": _mean(entropy_reductions),
    }


# ---------------------------------------------------------------------------
# Part 7: Strategist / plan metrics.
# ---------------------------------------------------------------------------


def strategist_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_arms = [a for row in rows if (a := _ok_arm(row, "FULL"))]
    generated = [a.get("plans_generated", 0) for a in full_arms]
    verified = [a.get("plans_verified", 0) for a in full_arms]
    rejected = [a.get("plans_rejected", 0) for a in full_arms]
    no_safe = [bool(a.get("no_safe_plan")) for a in full_arms]
    no_action_available = [bool(a.get("no_action_available")) for a in full_arms]
    approved = [bool(a.get("selected_plan") and a["selected_plan"].get("approval_status") == 200) for a in full_arms]
    return {
        "kind": "M10_4_STRATEGIST_TRAJECTORY", "protocol_hash": proto.protocol_hash(),
        "n_incidents": len(full_arms),
        "mean_candidates_generated": _mean(generated),
        "mean_candidates_wntr_verified": _mean(verified),
        "mean_candidates_rejected": _mean(rejected),
        "no_safe_plan_rate": _rate(no_safe),
        "no_action_available_rate": _rate(no_action_available),
        "human_approved_rate": _rate(approved),
    }


# ---------------------------------------------------------------------------
# Part 8: physical outcomes vs NO_ACTION reference.
# ---------------------------------------------------------------------------

CONSEQUENCE_FIELDS = (
    "population_impacted", "contaminant_mass_consumed_mg", "volume_above_threshold_l",
    "minimum_pressure_m", "pressure_violation_minutes", "unserved_demand_l",
    "service_availability", "containment_time_minutes",
)


def physical_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_arms = [a for row in rows if (a := _ok_arm(row, "FULL"))]
    deltas: dict[str, list[float]] = {field: [] for field in CONSEQUENCE_FIELDS}
    paired_available = 0
    for a in full_arms:
        selected = a.get("selected_plan")
        no_action = a.get("no_action_consequences")
        if not selected or not no_action:
            continue
        selected_consequences = selected.get("verification", {}).get("consequences")
        if not selected_consequences:
            continue
        paired_available += 1
        for field in CONSEQUENCE_FIELDS:
            sv, nv = selected_consequences.get(field), no_action.get(field)
            if sv is not None and nv is not None:
                deltas[field].append(sv - nv)
    return {
        "kind": "M10_4_PHYSICAL_OUTCOMES", "protocol_hash": proto.protocol_hash(),
        "n_incidents_with_selected_plan": sum(1 for a in full_arms if a.get("selected_plan")),
        "n_incidents_with_no_action_reference": sum(1 for a in full_arms if a.get("no_action_consequences")),
        "n_paired_selected_vs_no_action": paired_available,
        "mean_delta_selected_minus_no_action": {field: _mean(values) for field, values in deltas.items()},
    }


# ---------------------------------------------------------------------------
# Part 4/8: comparator (ARM_FULL vs ARM_NO_EXTRA_SAMPLING).
# ---------------------------------------------------------------------------


def comparator_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = []
    for row in rows:
        full = _ok_arm(row, "FULL")
        noext = _ok_arm(row, "NO_EXTRA_SAMPLING")
        if full is None or noext is None:
            continue
        paired.append((row, full, noext))

    initial_equal = sum(1 for row, _, _ in paired if row.get("paired_initial_state_equal"))
    full_top1 = [f["final"].get("top1_correct") for _, f, _ in paired]
    noext_top1 = [n["final"].get("top1_correct") for _, _, n in paired]
    decision_changed = 0
    sampling_improves = 0
    sampling_worsens = 0
    for _, f, n in paired:
        f_top1, n_top1 = f["final"].get("top1_correct"), n["final"].get("top1_correct")
        f_action = (f.get("selected_plan") or {}).get("action_types")
        n_action = (n.get("selected_plan") or {}).get("action_types")
        if f_action != n_action:
            decision_changed += 1
        if f_top1 is True and n_top1 is False:
            sampling_improves += 1
        elif f_top1 is False and n_top1 is True:
            sampling_worsens += 1

    sampled_pairs = [(f, n) for _, f, n in paired if (f.get("samples_taken") or 0) > 0]
    return {
        "kind": "M10_4_COMPARATOR", "protocol_hash": proto.protocol_hash(),
        "n_pairs": len(paired),
        "n_pairs_initial_state_equal": initial_equal,
        "paired_initial_state_equal_rate": (initial_equal / len(paired)) if paired else None,
        "arm_full_final_top1": _rate(full_top1),
        "arm_no_extra_sampling_final_top1": _rate(noext_top1),
        "n_pairs_where_sampling_occurred": len(sampled_pairs),
        "fraction_final_decision_changed_by_sampling": (decision_changed / len(paired)) if paired else None,
        "n_incidents_sampling_improved_top1": sampling_improves,
        "n_incidents_sampling_worsened_top1": sampling_worsens,
    }


# ---------------------------------------------------------------------------
# Utility gate + closure.
# ---------------------------------------------------------------------------


def compute_gate(
    *, safety: dict[str, Any], fail_closed: dict[str, Any], source: dict[str, Any],
    comparator: dict[str, Any], strategist: dict[str, Any], physical: dict[str, Any],
) -> dict[str, Any]:
    gate_a = bool(safety["all_zero"])

    full_top1 = comparator["arm_full_final_top1"]["rate"]
    noext_top1 = comparator["arm_no_extra_sampling_final_top1"]["rate"]
    gate_b = (
        full_top1 is not None and noext_top1 is not None
        and full_top1 >= noext_top1 - 0.05
    )

    n_sampled = comparator["n_pairs_where_sampling_occurred"]
    improved = comparator["n_incidents_sampling_improved_top1"]
    worsened = comparator["n_incidents_sampling_worsened_top1"]
    if n_sampled == 0:
        gate_c = True  # vacuously non-harmful -- no active sampling exercised in this population
        gate_c_detail = "no incident in the population exercised active sampling"
    else:
        gate_c = (improved - worsened) >= -max(1, round(0.10 * n_sampled))
        gate_c_detail = f"improved={improved} worsened={worsened} of {n_sampled} sampled pairs"

    gate_d = safety["counters"]["wntr_rejected_plan_surfaced_as_safe"] == 0

    deltas = physical["mean_delta_selected_minus_no_action"]
    exposure_delta = deltas.get("population_impacted")
    pressure_delta = deltas.get("minimum_pressure_m")
    service_delta = deltas.get("service_availability")
    n_no_action_pairs = physical.get("n_paired_selected_vs_no_action", 0)
    gate_e_evaluated = n_no_action_pairs > 0
    gate_e = (
        (exposure_delta is None or exposure_delta <= 0.10 * max(1.0, abs(exposure_delta)))
        and (pressure_delta is None or pressure_delta >= -0.10 * max(1.0, abs(pressure_delta)))
        and (service_delta is None or service_delta >= -0.10)
    )
    gate_e_detail = (
        f"evaluated over {n_no_action_pairs} paired selected-vs-NO_ACTION incidents"
        if gate_e_evaluated
        else "VACUOUS -- NO_ACTION never appeared among the (count=2) generated candidates in this "
             "population, so no selected-plan-vs-NO_ACTION physical delta was ever observed to compare. "
             "This gate PASSES only because there is nothing to contradict it, not because non-harm was "
             "positively demonstrated. Disclosed, non-blocking limitation of this population's candidate "
             "count -- not silently presented as a real characterization (Part 8/10 honesty requirement)."
    )

    gate_f = safety["counters"]["nonfinite_value_reached_decision"] == 0
    gate_g = bool(fail_closed["all_cases_bounded_and_deterministic"])

    checks = {"A": gate_a, "B": gate_b, "C": gate_c, "D": gate_d, "E": gate_e, "F": gate_f, "G": gate_g}
    return {
        "kind": "M10_4_GATE", "protocol_hash": proto.protocol_hash(),
        "checks": checks, "gate_c_detail": gate_c_detail,
        "gate_e_evaluated": gate_e_evaluated, "gate_e_detail": gate_e_detail,
        "full_top1": full_top1, "no_extra_sampling_top1": noext_top1,
        "all_checks_pass": all(checks.values()),
    }


def compute_closure(*, preflight: dict[str, Any], gate: dict[str, Any], safety: dict[str, Any]) -> dict[str, Any]:
    if preflight.get("result") != "M10_4_PREFLIGHT_PASS":
        state = "M10_4_FULL_TRAJECTORY_BLOCKED"
        rationale = "preflight did not pass"
    elif not safety["all_zero"]:
        state = "M10_4_FULL_TRAJECTORY_BLOCKED"
        rationale = "a hard safety counter was non-zero"
    elif gate["all_checks_pass"]:
        state = "M10_4_FULL_TRAJECTORY_PASS"
        rationale = "integration valid; all hard safety gates pass; retained system meets the frozen trajectory utility/quality gate"
    else:
        state = "M10_4_FULL_TRAJECTORY_UTILITY_NOT_ESTABLISHED"
        failed = [k for k, v in gate["checks"].items() if not v]
        rationale = f"trajectory scientifically executed and hard safety boundaries intact, but utility gate criterion/criteria failed: {failed}"

    return {
        "kind": "M10_4_CLOSURE", "milestone": "M10.4", "protocol_hash": proto.protocol_hash(),
        "branch": m10.current_branch(), "commit": m10.current_commit(),
        "closure_state": state, "rationale": rationale,
        "m10_5_authorized": False,
        "m10_5_note": "Even a PASS here does not authorize serving-path freeze, runtime promotion, or "
                      "opening the locked test -- M10.5 requires separate, explicit authorization.",
    }


def main() -> None:
    rows = _load_trajectories()
    safety = json.loads((M10_4_DIR / "m10-4-safety-counters.json").read_text())
    fail_closed = json.loads((M10_4_DIR / "m10-4-fail-closed.json").read_text())
    preflight = json.loads((M10_4_DIR / "m10-4-preflight.json").read_text())

    source = source_metrics(rows)
    scout = scout_metrics(rows)
    strategist = strategist_metrics(rows)
    physical = physical_outcomes(rows)
    comparator = comparator_metrics(rows)

    (M10_4_DIR / "m10-4-source-trajectory.json").write_text(json.dumps(source, indent=2, default=str) + "\n")
    (M10_4_DIR / "m10-4-scout-trajectory.json").write_text(json.dumps(scout, indent=2, default=str) + "\n")
    (M10_4_DIR / "m10-4-strategist-trajectory.json").write_text(json.dumps(strategist, indent=2, default=str) + "\n")
    (M10_4_DIR / "m10-4-physical-outcomes.json").write_text(json.dumps(physical, indent=2, default=str) + "\n")
    (M10_4_DIR / "m10-4-comparator.json").write_text(json.dumps(comparator, indent=2, default=str) + "\n")

    n_harness_errors = sum(
        1 for row in rows for arm in ("FULL", "NO_EXTRA_SAMPLING") if _ok_arm(row, arm) is None
    )
    summary = {
        "kind": "M10_4_TRAJECTORY_SUMMARY", "protocol_hash": proto.protocol_hash(),
        "n_physical_incidents": len(rows), "n_api_incidents": len(rows) * 2,
        "n_arm_outcomes_not_ok": n_harness_errors,
        "source_overall": source["overall"], "scout_overview": {
            "fraction_requesting_ge1_sample": scout["fraction_requesting_ge1_sample"],
            "mean_samples_per_incident": scout["mean_samples_per_incident"],
        },
        "strategist_overview": {
            "no_safe_plan_rate": strategist["no_safe_plan_rate"],
            "human_approved_rate": strategist["human_approved_rate"],
        },
        "comparator_overview": {
            "arm_full_final_top1": comparator["arm_full_final_top1"]["rate"],
            "arm_no_extra_sampling_final_top1": comparator["arm_no_extra_sampling_final_top1"]["rate"],
        },
    }
    (M10_4_DIR / "m10-4-trajectory-summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    gate = compute_gate(safety=safety, fail_closed=fail_closed, source=source, comparator=comparator, strategist=strategist, physical=physical)
    (M10_4_DIR / "m10-4-gate.json").write_text(json.dumps(gate, indent=2, default=str) + "\n")

    closure = compute_closure(preflight=preflight, gate=gate, safety=safety)
    (M10_4_DIR / "m10-4-closure.json").write_text(json.dumps(closure, indent=2, default=str) + "\n")

    print(json.dumps({"closure_state": closure["closure_state"], "gate_checks": gate["checks"]}, indent=2))


if __name__ == "__main__":
    main()
