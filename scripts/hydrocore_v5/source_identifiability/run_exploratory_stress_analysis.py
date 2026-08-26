#!/usr/bin/env python3
"""Exploratory-tier clean-vs-stress identifiability comparison (higher
power than the confirmatory tier's n=15/condition, Section 8 of the
protocol). Pools all 7 networks (525 incidents, 175 per condition)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_identifiability import common, stats_utils  # noqa: E402

EXPLORATORY_PATH = common.OUTPUT_ROOT / "exploratory" / "exploratory-identifiability.jsonl"
OUTPUT_PATH = common.OUTPUT_ROOT / "exploratory" / "exploratory-stress-comparison.json"


def main() -> None:
    rows = [json.loads(line) for line in EXPLORATORY_PATH.read_text().splitlines() if line.strip()]
    clean = [r["identifiability_normalized_correlation"]["identifiability_score"] for r in rows if r["condition_kind"] == "CLEAN"]
    result = {"kind": "SOURCE_IDENTIFIABILITY_EXPLORATORY_STRESS_COMPARISON", "n_clean": len(clean)}
    for condition in sorted({r["condition_kind"] for r in rows} - {"CLEAN"}):
        stressed = [
            r["identifiability_normalized_correlation"]["identifiability_score"]
            for r in rows
            if r["condition_kind"] == condition
        ]
        result[condition] = stats_utils.unpaired_bootstrap_diff(stressed, clean)

    # oracle top1 rate under clean vs stress, same pooling
    oracle_clean = [r["oracle_top1"] for r in rows if r["condition_kind"] == "CLEAN"]
    result["oracle_top1_rates"] = {
        condition: float(
            sum(r["oracle_top1"] for r in rows if r["condition_kind"] == condition)
            / sum(1 for r in rows if r["condition_kind"] == condition)
        )
        for condition in sorted({r["condition_kind"] for r in rows})
    }
    result["oracle_top1_stress_minus_clean"] = {
        condition: stats_utils.unpaired_bootstrap_diff(
            [r["oracle_top1"] for r in rows if r["condition_kind"] == condition], oracle_clean
        )
        for condition in sorted({r["condition_kind"] for r in rows} - {"CLEAN"})
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
