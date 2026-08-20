"""Milestone 9.1: apply the frozen Step 1/2/3 guardrail-and-promotion
procedure (`docs/evaluation/HYDROCORE_V5_M9_1_PROTOCOL.md`, Section 12) to
whatever `run_m9_1_evaluate.py` has already written.

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m9_1_decide.py --stage screening
    .venv/bin/python scripts/hydrocore_v5/run_m9_1_decide.py --stage confirmation

`--stage screening`: for each of GRAPH_ODE/GRAPH_CDE/GRAPH_SDE, Step 1
guardrails using the mean across the two screening seeds (with the SAME
per-seed coverage/candidate-set/instability requirement Section 12 pins),
then -- only for arms that pass -- Step 2's paired bootstrap (candidate seed
31874 vs CURRENT seed 31874, bootstrap seed 20260815). Requires both
screening seeds' results+calibration blocks for CURRENT and every novel arm
to already exist; raises (does not silently skip) if any are missing.

`--stage confirmation`: reads which arms reached screening-stage
PROMOTION_CANDIDATE from the guardrails file already written by this
script's own screening pass; for each, re-runs Step 1 at the 3-seed mean
(same per-seed requirement, now over all three seeds) and Step 3's bootstrap
(candidate seed 20260815 vs CURRENT seed 20260815). Requires seed 20260815's
results+calibration blocks for CURRENT and every PROMOTION_CANDIDATE arm.

Writes (merged under "screening"/"confirmation", per arm -- never
overwriting the other stage's or another arm's block):
  reports/evaluation/hydrocore-v5/m9-1-guardrails.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m9_1_common as common  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing required artifact: {path}")
    return json.loads(path.read_text())


def _require_block(payload: dict[str, Any], arm: str, seed: int, label: str) -> dict[str, Any]:
    block = payload.get(arm, {}).get(str(seed))
    if block is None:
        raise SystemExit(f"missing {label} block for {arm}-seed{seed} -- run run_m9_1_evaluate.py first")
    return block


def _step1_guardrails(arm: str, seeds: tuple[int, ...], results: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    # Section 5: an UNSTABLE_ARM_SEED/SOLVER_STEP_LIMIT_EXCEEDED training
    # outcome disqualifies the arm outright, regardless of other seeds'
    # results, and has NO checkpoint to have been evaluated -- checked
    # first, from the always-present training run record, before requiring
    # any results/calibration block to exist for that seed.
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        run_record = common.load_run_record(arm, seed)
        flag = run_record.get("instability_flag")
        if flag is not None:
            per_seed[str(seed)] = {"coverage_ok": None, "candidate_set_ok": None, "instability_ok": False, "instability_flag": flag}
    if per_seed:
        checks = {
            "early_regression_ok": None, "mature_regression_ok": None, "mrr_regression_ok": None,
            "coverage_ok": None, "candidate_set_ok": None, "instability_ok": False, "ood_safety_authority_ok": True,
        }
        return {
            "seeds": list(seeds), "early_regression_pp": None, "mature_regression_pp": None, "mrr_regression": None,
            "checks": checks, "per_seed": per_seed, "guardrails_passed": False,
        }

    candidate_early = []
    candidate_mature = []
    candidate_mrr = []
    control_early = []
    control_mature = []
    control_mrr = []
    per_seed = {}
    for seed in seeds:
        cand_res = _require_block(results, arm, seed, "results")
        ctrl_res = _require_block(results, "CURRENT", seed, "results")
        cand_cal = _require_block(calibration, arm, seed, "calibration")

        candidate_early.append(cand_res["aggregates"]["early_top1"])
        candidate_mature.append(cand_res["aggregates"]["mature_top1"])
        candidate_mrr.append(cand_res["aggregates"]["overall_mrr"])
        control_early.append(ctrl_res["aggregates"]["early_top1"])
        control_mature.append(ctrl_res["aggregates"]["mature_top1"])
        control_mrr.append(ctrl_res["aggregates"]["overall_mrr"])

        marginal = cand_cal["marginal"]
        by_maturity = cand_cal["by_maturity"]
        coverage_ok = (
            marginal["coverage"] >= common.GUARDRAIL_MIN_COVERAGE
            and all(by_maturity[bucket]["coverage"] >= common.GUARDRAIL_MIN_COVERAGE for bucket in ("EARLY", "MID", "MATURE"))
        )
        candidate_set_ok = marginal["mean_normalized_set_size"] <= common.GUARDRAIL_MAX_NORMALIZED_SET_SIZE
        instability_ok = not cand_res["aggregates"]["solver_step_limit_exceeded"] and cand_res["aggregates"]["all_finite"]
        per_seed[str(seed)] = {
            "coverage_ok": coverage_ok, "candidate_set_ok": candidate_set_ok, "instability_ok": instability_ok,
            "marginal_coverage": marginal["coverage"],
            "by_maturity_coverage": {b: by_maturity[b]["coverage"] for b in ("EARLY", "MID", "MATURE")},
            "mean_normalized_set_size": marginal["mean_normalized_set_size"],
        }

    early_regression_pp = (statistics.fmean(control_early) - statistics.fmean(candidate_early)) * 100.0
    mature_regression_pp = (statistics.fmean(control_mature) - statistics.fmean(candidate_mature)) * 100.0
    mrr_regression = statistics.fmean(control_mrr) - statistics.fmean(candidate_mrr)

    checks = {
        "early_regression_ok": early_regression_pp <= common.GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP,
        "mature_regression_ok": mature_regression_pp <= common.GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP,
        "mrr_regression_ok": mrr_regression <= common.GUARDRAIL_MAX_MRR_REGRESSION,
        "coverage_ok": all(per_seed[str(s)]["coverage_ok"] for s in seeds),
        "candidate_set_ok": all(per_seed[str(s)]["candidate_set_ok"] for s in seeds),
        "instability_ok": all(per_seed[str(s)]["instability_ok"] for s in seeds),
        "ood_safety_authority_ok": True,  # Section 12 Step 1.7: unmodified by this milestone, confirmed by inspection.
    }
    passed = all(checks.values())
    return {
        "seeds": list(seeds),
        "early_regression_pp": early_regression_pp,
        "mature_regression_pp": mature_regression_pp,
        "mrr_regression": mrr_regression,
        "checks": checks,
        "per_seed": per_seed,
        "guardrails_passed": passed,
    }


def _step_bootstrap(arm: str, seed: int, results: dict[str, Any]) -> dict[str, Any]:
    cand = _require_block(results, arm, seed, "results")["primary_metric_per_incident"]
    ctrl = _require_block(results, "CURRENT", seed, "results")["primary_metric_per_incident"]
    return common.paired_bootstrap(cand, ctrl)


def _screening(results: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in common.NOVEL_ARMS:
        step1 = _step1_guardrails(arm, common.SCREENING_SEEDS, results, calibration)
        entry: dict[str, Any] = {"step1": step1}
        if not step1["guardrails_passed"]:
            entry["outcome"] = "GUARDRAILS_FAILED"
        else:
            step2 = _step_bootstrap(arm, common.REPRESENTATIVE_SEED, results)
            entry["step2"] = step2
            entry["outcome"] = "PROMOTION_CANDIDATE" if step2["ci_entirely_positive"] else "GUARDRAILS_PASSED_NO_SIGNIFICANT_GAIN"
        out[arm] = entry
    return out


def _confirmation(results: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    guardrails = _load(common.GUARDRAILS_PATH)
    screening = guardrails.get("screening")
    if not screening:
        raise SystemExit("no screening section in m9-1-guardrails.json -- run --stage screening first")
    candidates = [
        arm for arm, entry in screening.items()
        if isinstance(entry, dict) and entry.get("outcome") == "PROMOTION_CANDIDATE"
    ]
    out: dict[str, Any] = {}
    for arm in candidates:
        seeds = common.SCREENING_SEEDS + (common.CONFIRMATION_SEED,)
        step1 = _step1_guardrails(arm, seeds, results, calibration)
        entry: dict[str, Any] = {"step1": step1}
        if not step1["guardrails_passed"]:
            entry["outcome"] = "PROMOTION_NOT_CONFIRMED"
        else:
            step3 = _step_bootstrap(arm, common.CONFIRMATION_SEED, results)
            entry["step3"] = step3
            entry["outcome"] = "PROMOTION_CONFIRMED" if step3["ci_entirely_positive"] else "PROMOTION_NOT_CONFIRMED"
        out[arm] = entry
    if not candidates:
        out["_note"] = "no arm reached screening-stage PROMOTION_CANDIDATE; confirmation stage not run for any arm (Section 5, negative result)."
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("screening", "confirmation"))
    args = parser.parse_args()

    common.assert_code_under_test_commit()
    locked_before = common.assert_locked_test_closed()

    results = _load(common.RESULTS_PATH)
    calibration = _load(common.CALIBRATION_PATH)

    if args.stage == "screening":
        stage_result = _screening(results, calibration)
    else:
        stage_result = _confirmation(results, calibration)

    locked_after = common.assert_locked_test_closed()
    stage_result["_locked_test_opened_before"] = locked_before
    stage_result["_locked_test_opened_after"] = locked_after

    existing = json.loads(common.GUARDRAILS_PATH.read_text()) if common.GUARDRAILS_PATH.exists() else {}
    existing[args.stage] = stage_result
    common.GUARDRAILS_PATH.parent.mkdir(parents=True, exist_ok=True)
    common.GUARDRAILS_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print(f"wrote {common.GUARDRAILS_PATH} [{args.stage}]")
    print(json.dumps({arm: entry.get("outcome") for arm, entry in stage_result.items() if isinstance(entry, dict) and "outcome" in entry}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
