#!/usr/bin/env python3
"""Phase 8: known-vs-unseen topology decomposition.

Combines two tiers:
  - confirmatory (n=125, real HydroCore-v5 outcomes attached, small):
    known-vs-unseen HydroCore-v5 top1/MRR AND known-vs-unseen physical
    identifiability, on the exact frozen M11.6 locked evaluation.
  - exploratory (n up to ~525, self-generated, no HydroCore prediction,
    much larger): known-vs-unseen physical identifiability only, at higher
    statistical power, across CLEAN/MEASUREMENT_NOISE/SENSOR_DROPOUT.

Answers whether any observed known/unseen HydroCore-v5 gap is explained by
(A) worse physical identifiability in unseen networks, (B) comparable
identifiability but poorer representation, or (C) both -- never claiming
more than the evidence in each tier supports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_identifiability import common, stats_utils  # noqa: E402

JOINED_PATH = common.OUTPUT_ROOT / "joined" / "joined-incidents.jsonl"
EXPLORATORY_PATH = common.OUTPUT_ROOT / "exploratory" / "exploratory-identifiability.jsonl"
OUTPUT_PATH = common.OUTPUT_ROOT / "joined" / "topology-split-decision.json"


def main() -> None:
    confirmatory = [json.loads(line) for line in JOINED_PATH.read_text().splitlines() if line.strip()]
    known_conf = [r for r in confirmatory if r["known_topology"]]
    unseen_conf = [r for r in confirmatory if not r["known_topology"]]

    confirmatory_block = {
        "n_known": len(known_conf),
        "n_unseen": len(unseen_conf),
        "hydrocore_top1_known_minus_unseen": stats_utils.unpaired_bootstrap_diff(
            [r["hydrocore_top1_correct"] for r in known_conf],
            [r["hydrocore_top1_correct"] for r in unseen_conf],
        ),
        "hydrocore_mrr_known_minus_unseen": stats_utils.unpaired_bootstrap_diff(
            [r["hydrocore_reciprocal_rank"] for r in known_conf],
            [r["hydrocore_reciprocal_rank"] for r in unseen_conf],
        ),
        "identifiability_known_minus_unseen": stats_utils.unpaired_bootstrap_diff(
            [r["identifiability_score"] for r in known_conf],
            [r["identifiability_score"] for r in unseen_conf],
        ),
        "oracle_top1_known_minus_unseen": stats_utils.unpaired_bootstrap_diff(
            [r["oracle_top1"] for r in known_conf], [r["oracle_top1"] for r in unseen_conf]
        ),
    }

    exploratory_block = None
    if EXPLORATORY_PATH.exists():
        exploratory = [json.loads(line) for line in EXPLORATORY_PATH.read_text().splitlines() if line.strip()]
        known_exp = [r for r in exploratory if r["known_topology"]]
        unseen_exp = [r for r in exploratory if not r["known_topology"]]
        exploratory_block = {
            "n_known": len(known_exp),
            "n_unseen": len(unseen_exp),
            "identifiability_known_minus_unseen": stats_utils.unpaired_bootstrap_diff(
                [r["identifiability_normalized_correlation"]["identifiability_score"] for r in known_exp],
                [r["identifiability_normalized_correlation"]["identifiability_score"] for r in unseen_exp],
            ),
            "oracle_top1_known_minus_unseen": stats_utils.unpaired_bootstrap_diff(
                [r["oracle_top1"] for r in known_exp], [r["oracle_top1"] for r in unseen_exp]
            ),
            "by_condition": {
                condition: {
                    "identifiability_known_minus_unseen": stats_utils.unpaired_bootstrap_diff(
                        [
                            r["identifiability_normalized_correlation"]["identifiability_score"]
                            for r in known_exp
                            if r["condition_kind"] == condition
                        ],
                        [
                            r["identifiability_normalized_correlation"]["identifiability_score"]
                            for r in unseen_exp
                            if r["condition_kind"] == condition
                        ],
                    )
                }
                for condition in sorted({r["condition_kind"] for r in exploratory})
            },
        }

    def _decide(hydrocore_gap: dict, identifiability_gap: dict) -> str:
        hydrocore_ci_excludes_zero = bool(hydrocore_gap.get("ci_entirely_positive") or hydrocore_gap.get("ci_entirely_non_positive"))
        identifiability_worse_for_unseen = bool(identifiability_gap.get("ci_entirely_positive"))  # known - unseen > 0
        if not hydrocore_ci_excludes_zero:
            return "NO_ROBUST_HYDROCORE_GAP_DETECTED_ON_THIS_EVIDENCE"
        if identifiability_worse_for_unseen:
            return "A_WORSE_IDENTIFIABILITY"
        return "B_COMPARABLE_IDENTIFIABILITY_POORER_REPRESENTATION"

    decision = _decide(
        confirmatory_block["hydrocore_top1_known_minus_unseen"],
        confirmatory_block["identifiability_known_minus_unseen"],
    )

    result = {
        "kind": "SOURCE_IDENTIFIABILITY_TOPOLOGY_SPLIT_DECISION",
        "confirmatory_tier": confirmatory_block,
        "exploratory_tier": exploratory_block,
        "decision_from_confirmatory_tier": decision,
        "caveat": (
            "Confirmatory tier has only n_unseen={} locked incidents -- a "
            "null or noisy result here does not positively establish 'no "
            "gap exists', only that this small locked sample cannot "
            "robustly detect one. The exploratory tier's much larger, "
            "condition-stratified identifiability comparison is the more "
            "statistically trustworthy read on whether unseen topologies "
            "are intrinsically less identifiable; the confirmatory tier "
            "remains the only one tied to real HydroCore-v5 predictions."
        ).format(confirmatory_block["n_unseen"]),
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
