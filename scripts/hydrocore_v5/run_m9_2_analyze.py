"""Milestone 9.2: run all diagnostic analyses over the canonical table and
write the required machine-readable artifacts (Sections 5-13 of the M9.2
brief). Interpretation (hypothesis mapping, closure recommendation,
m9-2-summary.md) is written separately/by hand after reviewing these
outputs -- this script computes numbers only, no promotion/selection logic.

DIAGNOSTIC / ANALYSIS-ONLY. Reads `m9-2-canonical-diagnostics.jsonl` (built
by `run_m9_2_build_table.py`); does not touch model/training code, does not
open locked_final_test/locked_topology_test, does not retrain or tune
anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import m9_2_analysis_lib as lib  # noqa: E402
import m9_2_common as m92  # noqa: E402


def load_canonical_table() -> pd.DataFrame:
    rows = [json.loads(line) for line in m92.M9_2_CANONICAL_PATH.read_text().splitlines()]
    df = pd.DataFrame(rows)
    df["top1_correct"] = df["top1_correct"].astype(bool)
    df["true_source_covered"] = df["true_source_covered"].astype("boolean")
    return df


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    locked_before = m92.assert_locked_test_closed()
    df = load_canonical_table()
    assert set(df["training_seed"].unique().tolist()) == set(m92.SCREENING_SEEDS), "unexpected seeds in canonical table"
    assert m92.EXCLUDED_UNPAIRED_SEED not in df["training_seed"].unique().tolist()

    print(f"loaded canonical table: {len(df)} rows")

    print("Section 5: depth metrics + paired deltas + bootstrap...")
    depth_metrics = {
        "per_arm_seed_depth": lib.depth_metrics_by_arm_seed(df),
        "paired_deltas_by_depth": lib.paired_deltas_by_depth(df),
    }
    # Section 13 cross-seed classification applied to the primary MATURE-depth top1 delta finding.
    cross_seed = {}
    for arm in m92.NOVEL_ARMS:
        cross_seed[arm] = {}
        for depth in m92.CAUSAL_PREFIX_DEPTHS:
            per_seed = depth_metrics["paired_deltas_by_depth"][arm][str(depth)]["per_seed"]
            deltas = [per_seed[str(s)]["top1_delta"] for s in m92.SCREENING_SEEDS]
            cross_seed[arm][str(depth)] = {"seed_deltas": deltas, "classification": lib.classify_cross_seed(deltas)}
    depth_metrics["cross_seed_consistency_top1_delta"] = cross_seed
    write_json(m92.M9_2_DEPTH_METRICS_PATH, depth_metrics)

    print("Section 6: disagreement / complementarity...")
    disagreements = lib.disagreement_tables(df)
    write_json(m92.M9_2_DISAGREEMENTS_PATH, disagreements)

    print("Section 7: rank movement...")
    rank_analysis = lib.rank_movement(df)
    write_json(m92.M9_2_RANK_ANALYSIS_PATH, rank_analysis)

    print("Section 8: topology / spatial error analysis...")
    topology_analysis = lib.topology_error_analysis(df)
    write_json(m92.M9_2_TOPOLOGY_ANALYSIS_PATH, topology_analysis)

    print("Sections 9-10: missingness stratification + difficulty analysis...")
    missingness_analysis = {
        "covariate_stratified_deltas": lib.covariate_stratified_deltas(df.copy()),
        "current_conditioned_difficulty": lib.current_conditioned_difficulty(df.copy()),
    }
    write_json(m92.M9_2_MISSINGNESS_ANALYSIS_PATH, missingness_analysis)

    print("Section 11: calibration diagnostics...")
    calibration_diag = lib.calibration_diagnostics(df)
    write_json(m92.M9_2_CALIBRATION_DIAGNOSTICS_PATH, calibration_diag)

    print("Section 12: case studies...")
    case_studies = lib.case_studies(df)
    write_json(m92.M9_2_CASE_STUDIES_PATH, case_studies)

    locked_after = m92.assert_locked_test_closed()
    print(json.dumps({"locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
