"""Milestone 9.3: run all calibration-diagnostic analyses (Sections 7-20)
over the canonical table and write the required machine-readable artifacts.
Interpretation (root-cause evidence ratings, recommendation, summary.md) is
written separately/by hand after reviewing these outputs.

DIAGNOSTIC / ANALYSIS-ONLY. Reads `m9-3-canonical-calibration-diagnostics.jsonl`
(built by `run_m9_3_build_table.py`); no model inference, no training, no
locked-test access, no checkpoint modification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import m9_3_analysis_lib as lib  # noqa: E402
import m9_3_common as m93  # noqa: E402


def load_canonical_table() -> pd.DataFrame:
    rows = [json.loads(line) for line in m93.M9_3_CANONICAL_PATH.read_text().splitlines()]
    df = pd.DataFrame(rows)
    df["top1_correct"] = df["top1_correct"].astype(bool)
    return df


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    locked_before = m93.assert_locked_test_closed()
    df = load_canonical_table()
    assert set(df["training_seed"].unique().tolist()) == set(m93.SEEDS), "unexpected seeds in canonical table"
    print(f"loaded canonical table: {len(df)} rows")

    known = df[(df["split"] != "development_unseen_diagnostic")]

    print("Section 7: support analysis...")
    support = {"group_support": lib.support_analysis(known), "fallback_frequency": lib.fallback_frequency(known)}
    write_json(m93.M9_3_SUPPORT_ANALYSIS_PATH, support)

    print("Section 8: empirical coverage uncertainty...")
    coverage_unc = lib.coverage_uncertainty(known)
    write_json(m93.M9_3_DIR / "m9-3-coverage-uncertainty.json", coverage_unc)

    print("Section 9: score-shift diagnostics...")
    write_json(m93.M9_3_SCORE_SHIFT_PATH, lib.score_shift(known))

    print("Section 10: quantile stability (bootstrap)...")
    write_json(m93.M9_3_QUANTILE_STABILITY_PATH, lib.quantile_stability(known))

    print("Section 11: calibration-support learning curves...")
    write_json(m93.M9_3_LEARNING_CURVES_PATH, lib.support_learning_curves(known))

    print("Section 12: family heterogeneity...")
    write_json(m93.M9_3_FAMILY_HETEROGENEITY_PATH, lib.family_heterogeneity(known))

    print("Section 13: depth root-cause analysis...")
    write_json(m93.M9_3_DEPTH_ANALYSIS_PATH, lib.depth_root_cause(known))

    print("Section 14: confidence / overconfidence analysis...")
    confidence = {"by_depth_correctness": lib.confidence_overconfidence(known), "reliability_bins": lib.reliability_bins(known)}
    write_json(m93.M9_3_CONFIDENCE_ANALYSIS_PATH, confidence)

    print("Section 15: source-conditional analysis...")
    write_json(m93.M9_3_SOURCE_CONDITIONAL_PATH, lib.source_conditional(known))

    print("Section 16: case studies...")
    write_json(m93.M9_3_MISCOVERAGE_CASES_PATH, lib.case_studies(known))

    print("Section 17: miscoverage severity...")
    severity = lib.miscoverage_severity(known)
    existing = json.loads(m93.M9_3_MISCOVERAGE_CASES_PATH.read_text())
    existing["severity"] = severity
    write_json(m93.M9_3_MISCOVERAGE_CASES_PATH, existing)

    print("Section 18: counterfactual decomposition...")
    write_json(m93.M9_3_COUNTERFACTUAL_PATH, lib.counterfactual_decomposition(known))

    print("Section 19: sample-size estimation...")
    sample_size = lib.sample_size_estimation(known)
    existing_cf = json.loads(m93.M9_3_COUNTERFACTUAL_PATH.read_text())
    existing_cf["sample_size_estimation"] = sample_size
    write_json(m93.M9_3_COUNTERFACTUAL_PATH, existing_cf)

    print("Section 20: exchangeability audit...")
    write_json(m93.M9_3_EXCHANGEABILITY_PATH, lib.exchangeability_audit(known))

    locked_after = m93.assert_locked_test_closed()
    print(json.dumps({"locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
