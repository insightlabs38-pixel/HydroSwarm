#!/usr/bin/env python3
"""Phase 6: failure taxonomy over the confirmatory M11.6 HydroCore-v5 failures.

Categories (overlap allowed, never forced to exactly one):
  A - Information-limited: the physics-based oracle, using the SAME real
      (stressed) observation, also fails -- the strongest available test
      that evidence itself was insufficient.
  B - Representation-limited: the oracle succeeds with a residual margin
      clearly outside the sensor noise floor -- physically decisive
      evidence was available and HydroCore still missed it.
  C - Stress-induced identifiability collapse: a non-NOMINAL condition,
      good CLEAN physical separability (identifiability_score > 1.0, i.e.
      better than this incident's own typical candidate separation), but
      the oracle still fails on the real observation -- a subset of A
      specifically attributable to the applied stress rather than
      intrinsic shape ambiguity.
  D - Ranking-limited: HydroCore's official Top-3 flag is true but Top-1 is
      false -- the true source was nearly recovered.
  E - OOD/governance-limited: HydroCore's own recorded outcome shows the
      system abstained (SUPPRESSED) or the prediction was not calibrated --
      governance, not the raw ranking, is the binding constraint.
  F - Inconclusive: the oracle "succeeds" but its residual margin does not
      clearly clear the sensor noise floor -- too close to call from this
      evidence alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_identifiability import common  # noqa: E402

JOINED_PATH = common.OUTPUT_ROOT / "joined" / "joined-incidents.jsonl"
CONFIRMATORY_PATH = common.OUTPUT_ROOT / "confirmatory" / "confirmatory-identifiability.jsonl"
OUTPUT_DIR = common.OUTPUT_ROOT / "taxonomy"
OUTPUT_PATH = OUTPUT_DIR / "failure-taxonomy.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "failure-taxonomy-summary.json"


def classify(record: dict, noise_floor: float) -> dict:
    categories = []
    oracle_fails = record["oracle_top1"] == 0.0
    oracle_margin = record["oracle_residual_margin"]
    confident_oracle_success = (not oracle_fails) and (oracle_margin is not None) and (oracle_margin >= noise_floor)
    borderline_oracle_success = (not oracle_fails) and (oracle_margin is not None) and (oracle_margin < noise_floor)

    if oracle_fails:
        categories.append("A_information_limited")
        if record["condition_kind"] != "NOMINAL" and record["identifiability_score"] > 1.0:
            categories.append("C_stress_induced_collapse")
    if confident_oracle_success:
        categories.append("B_representation_limited")
    if record["hydrocore_top3_correct"] and not record["hydrocore_top1_correct"]:
        categories.append("D_ranking_limited")
    if record["hydrocore_abstained"] or not record["hydrocore_calibrated"]:
        categories.append("E_ood_governance_limited")
    if borderline_oracle_success:
        categories.append("F_inconclusive")
    if not categories:
        categories.append("F_inconclusive")
    return {"categories": categories}


def main() -> None:
    joined = [json.loads(line) for line in JOINED_PATH.read_text().splitlines() if line.strip()]
    noise_floor_by_key = {}
    for line in CONFIRMATORY_PATH.read_text().splitlines():
        record = json.loads(line)
        noise_floor_by_key[(record["split"], record["seed"])] = record["noise_floor_distance"]

    failures = [r for r in joined if not r["hydrocore_top1_correct"]]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with OUTPUT_PATH.open("w") as handle:
        for record in failures:
            noise_floor = noise_floor_by_key[(record["split"], record["seed"])]
            classification = classify(record, noise_floor)
            row = {**record, **classification}
            rows.append(row)
            handle.write(json.dumps(row) + "\n")

    n = len(rows)
    prevalence = {}
    for category in (
        "A_information_limited",
        "B_representation_limited",
        "C_stress_induced_collapse",
        "D_ranking_limited",
        "E_ood_governance_limited",
        "F_inconclusive",
    ):
        count = sum(1 for r in rows if category in r["categories"])
        prevalence[category] = {"n": count, "fraction_of_failures": count / n if n else None}

    # known vs unseen breakdown of the dominant category
    by_topology = {}
    for known_label, flag in (("known", True), ("unseen", False)):
        subset = [r for r in rows if r["known_topology"] == flag]
        by_topology[known_label] = {
            "n_failures": len(subset),
            "fraction_representation_limited": (
                sum(1 for r in subset if "B_representation_limited" in r["categories"]) / len(subset)
                if subset
                else None
            ),
            "fraction_information_limited": (
                sum(1 for r in subset if "A_information_limited" in r["categories"]) / len(subset)
                if subset
                else None
            ),
        }

    summary = {
        "kind": "SOURCE_IDENTIFIABILITY_FAILURE_TAXONOMY",
        "n_total_incidents": len(joined),
        "n_hydrocore_top1_failures": n,
        "prevalence": prevalence,
        "by_known_vs_unseen_topology": by_topology,
        "note": "Categories overlap by design (see module docstring); counts do not sum to n.",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
