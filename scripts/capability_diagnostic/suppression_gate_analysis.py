"""Capability diagnostic Sections 27-29: suppression-reason decomposition
(HIGHEST PRIORITY per protocol), evidence-sufficiency diagnostic, and
actionable-within-N metric, mined entirely from the already-executed
264-run LIVE post-remediation dataset.

All counterfactuals in this script are OFFLINE, pure re-analysis of
already-recorded per-incident gate outcomes (the `suppression_reasons`
list, `calibrated`, `ood_level`, `disagreement_js`, `candidate_set_size`,
`evidence_sufficient`, and `sample_rounds`). No production pipeline code
(hydroswarm.inference.pipeline / fusion.py) is imported or called, and no
production threshold/config is changed. This is diagnostic-only, per the
protocol's "never expose these counterfactuals in production code" rule.

Sanity-checked invariant this script relies on (verified directly against
the dataset before writing this script): for every one of the 255
analyzable records, `len(suppression_reasons) == 0` exactly iff
`planning_allowed is True`. This means `suppression_reasons` is a complete,
literal log of which individual gate(s) fired for that incident, and can be
used directly (not re-derived from thresholds) to reconstruct per-gate
counterfactuals.

Outputs:
  reports/evaluation/capability-diagnostic/suppression-analysis.json
  (keys: suppression_reason_decomposition, gate_counterfactual_actionability,
  evidence_sufficiency_diagnostic, actionable_within_n)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

DATA_PATH = ROOT / "reports/evaluation/live-robustness/post-remediation-results.json"
OUT_PATH = ROOT / "reports/evaluation/capability-diagnostic/suppression-analysis.json"

DISAGREEMENT_THRESHOLD = 0.50  # src/hydroswarm/inference/pipeline.py:160 default
MAXIMUM_PLANNING_CANDIDATES = 3  # src/hydroswarm/inference/pipeline.py:159 default


def gate_bucket(reason: str) -> str:
    """Map a raw suppression_reasons string to a stable gate-family key.
    OOD_* reasons (e.g. OOD_CAUTION, OOD_OUTSIDE_VALIDATED_RANGE) collapse
    to a single 'OOD' family since they're all instances of the same
    `ood_level.value != "NORMAL"` gate in src/hydroswarm/inference/pipeline.py."""
    if reason.startswith("OOD_"):
        return "OOD"
    return reason


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed for this diagnostic"
    locked_before = locked_test_opened(ROOT)

    all_records: list[dict[str, Any]] = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    total_records = len(all_records)

    analyzable = [r for r in all_records if r.get("suppression_reasons") is not None]
    non_analyzable = total_records - len(analyzable)

    # ---------------- Section 27: suppression-reason decomposition ----------------
    raw_reason_counts: Counter[str] = Counter()
    bucketed_reason_counts: Counter[str] = Counter()
    first_blocker_counts: Counter[str] = Counter()
    first_blocker_bucketed_counts: Counter[str] = Counter()
    n_blockers_histogram: Counter[int] = Counter()
    pairwise_cooccurrence: Counter[tuple[str, str]] = Counter()

    for r in analyzable:
        reasons: list[str] = r["suppression_reasons"]
        for reason in reasons:
            raw_reason_counts[reason] += 1
            bucketed_reason_counts[gate_bucket(reason)] += 1
        if reasons:
            first_blocker_counts[reasons[0]] += 1
            first_blocker_bucketed_counts[gate_bucket(reasons[0])] += 1
        n_blockers_histogram[len(reasons)] += 1
        bucketed_set = sorted({gate_bucket(x) for x in reasons})
        for a, b in combinations(bucketed_set, 2):
            pairwise_cooccurrence[(a, b)] += 1

    n_analyzable = len(analyzable)
    suppression_decomposition = {
        "n_analyzable_records": n_analyzable,
        "n_non_analyzable_records": non_analyzable,
        "non_analyzable_reason": (
            "outcome=ABSTAINED runs with error_class set (e.g. ANALYZE_409) where analysis "
            "never completed and suppression_reasons is null; excluded from gate accounting "
            "since no gate ever ran."
        ),
        "raw_reason_frequency": dict(raw_reason_counts),
        "raw_reason_rate_of_analyzable": {k: v / n_analyzable for k, v in raw_reason_counts.items()},
        "gate_family_frequency": dict(bucketed_reason_counts),
        "gate_family_rate_of_analyzable": {k: v / n_analyzable for k, v in bucketed_reason_counts.items()},
        "first_blocker_raw_frequency": dict(first_blocker_counts),
        "first_blocker_gate_family_frequency": dict(first_blocker_bucketed_counts),
        "n_simultaneous_blockers_histogram": {str(k): v for k, v in sorted(n_blockers_histogram.items())},
        "pairwise_gate_family_cooccurrence": {f"{a}+{b}": v for (a, b), v in pairwise_cooccurrence.items()},
        "dominant_blocker_gate_family": (
            bucketed_reason_counts.most_common(1)[0][0] if bucketed_reason_counts else None
        ),
    }

    # ---------------- Gate counterfactual actionability ----------------
    # Ground-truth gate sets per record, read directly from suppression_reasons
    # (verified 1:1 consistent with the recorded planning_allowed field before
    # this script was written: len(suppression_reasons)==0 iff planning_allowed==True).
    gate_sets: dict[str, set[str]] = {
        r["run_id"]: {gate_bucket(x) for x in r["suppression_reasons"]} for r in analyzable
    }
    observed_gate_families = sorted({g for gs in gate_sets.values() for g in gs})

    removable_scenarios = {
        "all_gates_active_observed": set(),  # remove nothing
        "all_except_calibration": {"CALIBRATION_INVALID_OR_MISSING"},
        "all_except_candidate_size": {"CANDIDATE_REGION_TOO_BROAD"},
        "all_except_evidence_sufficiency": {"MODEL_EVIDENCE_INSUFFICIENT"},
        "all_except_disagreement": {"HIGH_CLASSICAL_NEURAL_DISAGREEMENT"},
        "all_except_ood": {"OOD"},
    }

    def eligible_count_removing(remove: set[str]) -> int:
        count = 0
        for run_id, gates in gate_sets.items():
            remaining = gates - remove
            if not remaining:
                count += 1
        return count

    gate_counterfactuals: dict[str, Any] = {}
    for label, remove_set in removable_scenarios.items():
        eligible_analyzable = eligible_count_removing(remove_set)
        gate_counterfactuals[label] = {
            "gates_removed": sorted(remove_set),
            "eligible_count_of_264_total": eligible_analyzable,  # non-analyzable can never become eligible
            "eligible_fraction_of_264_total": eligible_analyzable / total_records,
            "eligible_count_of_255_analyzable": eligible_analyzable,
            "eligible_fraction_of_255_analyzable": eligible_analyzable / n_analyzable,
        }

    # "inference-quality-only": the only counterfactual gate scheme where
    # ALL current conservative proxy gates are removed and the sole blocking
    # criterion becomes "was the top-1 localization actually wrong". This is
    # the idealized ceiling: what if suppression tracked ground-truth
    # correctness instead of proxies.
    top1_known = [r for r in analyzable if r.get("top1_correct") is not None]
    inference_quality_only_eligible = sum(1 for r in top1_known if r["top1_correct"] is True)
    gate_counterfactuals["inference_quality_only"] = {
        "description": (
            "Idealized counterfactual: suppress ONLY when the top-1 localization guess is "
            "actually wrong (oracle correctness gate), removing calibration/candidate-size/"
            "evidence-sufficiency/disagreement/OOD gates entirely. This is a ceiling, not an "
            "achievable policy (correctness is not known at decision time), used only to show "
            "how much of current suppression is attributable to genuine localization failure "
            "versus conservative proxies."
        ),
        "n_with_known_top1_correct": len(top1_known),
        "eligible_count_of_264_total": inference_quality_only_eligible,
        "eligible_fraction_of_264_total": inference_quality_only_eligible / total_records,
        "eligible_fraction_of_known_top1": (
            inference_quality_only_eligible / len(top1_known) if top1_known else None
        ),
    }

    observed_eligible = eligible_count_removing(set())
    gate_counterfactuals["_headline_comparison"] = {
        "observed_planning_eligible_count_of_264": observed_eligible,
        "observed_planning_eligible_rate_of_264": observed_eligible / total_records,
        "matches_protocol_confirmed_premise_rate_0_012": (
            abs(observed_eligible / total_records - 0.012) < 0.02
        ),
    }

    gate_counterfactual_actionability = {
        "disagreement_threshold_used": DISAGREEMENT_THRESHOLD,
        "maximum_planning_candidates_used": MAXIMUM_PLANNING_CANDIDATES,
        "source_of_thresholds": "src/hydroswarm/inference/pipeline.py:159-160 (defaults, unchanged/read-only)",
        "method": (
            "Gate-firing sets are read directly from each record's recorded suppression_reasons "
            "list (verified equivalent to the pipeline's own gate logic: empty list iff "
            "planning_allowed==True across all 255 analyzable records). For each scenario, one "
            "gate family is removed from every record's fired-gate set and the incident is "
            "counted eligible iff no gate remains. Denominator is always all 264 records per "
            "protocol instruction (non-analyzable/ABSTAINED records can never become eligible "
            "under any gate-relaxation, since analysis never completed for them)."
        ),
        "observed_gate_families_present_in_data": observed_gate_families,
        "scenarios": gate_counterfactuals,
    }

    # ---------------- Section 28: evidence-sufficiency diagnostic ----------------
    es_records = [r for r in all_records if r.get("evidence_sufficient") is not None and r.get("top1_correct") is not None]
    tt = sum(1 for r in es_records if r["evidence_sufficient"] is True and r["top1_correct"] is True)
    tf = sum(1 for r in es_records if r["evidence_sufficient"] is True and r["top1_correct"] is False)
    ft = sum(1 for r in es_records if r["evidence_sufficient"] is False and r["top1_correct"] is True)
    ff = sum(1 for r in es_records if r["evidence_sufficient"] is False and r["top1_correct"] is False)
    n_es = len(es_records)
    # Conservatism: among incidents that were ACTUALLY top1-correct, what
    # fraction did the head still call evidence_sufficient=False?
    correct_total = tt + ft
    conservative_rate_among_correct = ft / correct_total if correct_total else None
    # Usefulness correlation: among evidence_sufficient=True incidents, what
    # fraction were actually correct (precision of the "sufficient" signal)?
    sufficient_total = tt + tf
    precision_of_sufficient_signal = tt / sufficient_total if sufficient_total else None
    insufficient_total = ft + ff
    precision_of_insufficient_signal = ff / insufficient_total if insufficient_total else None

    evidence_sufficiency_diagnostic = {
        "n_records_with_both_fields": n_es,
        "confusion_table": {
            "evidence_sufficient_true_top1_correct_true": tt,
            "evidence_sufficient_true_top1_correct_false": tf,
            "evidence_sufficient_false_top1_correct_true": ft,
            "evidence_sufficient_false_top1_correct_false": ff,
        },
        "precision_of_evidence_sufficient_true_signal": precision_of_sufficient_signal,
        "precision_of_evidence_sufficient_false_signal": precision_of_insufficient_signal,
        "conservatism_rate_among_actually_correct_incidents": conservative_rate_among_correct,
        "interpretation": (
            "conservatism_rate_among_actually_correct_incidents is the fraction of incidents "
            "where top1 localization was ALREADY correct but evidence_sufficient was still "
            "False (i.e. the incident was blocked from planning despite being right). A high "
            "value here indicates the evidence-sufficiency head/gate combination is excessively "
            "conservative rather than well-correlated with real usefulness; a low value (most "
            "correct incidents are also flagged sufficient) indicates good correlation."
        ),
        "verdict": (
            "EXCESSIVELY_CONSERVATIVE"
            if (conservative_rate_among_correct is not None and conservative_rate_among_correct >= 0.5)
            else "NOT_CLEARLY_EXCESSIVELY_CONSERVATIVE"
        ),
    }

    # ---------------- Section 29: actionable-within-N ----------------
    def rich_round(sr: dict[str, Any]) -> bool:
        return "planning_allowed_after" in sr

    records_with_rich_rounds = 0
    total_rich_rounds = 0
    per_record_transitions: list[dict[str, Any]] = []

    for r in all_records:
        rounds = r.get("sample_rounds") or []
        rich_rounds_sorted = sorted([sr for sr in rounds if rich_round(sr)], key=lambda x: x["round"])
        if rich_rounds_sorted:
            records_with_rich_rounds += 1
            total_rich_rounds += len(rich_rounds_sorted)
        initial = r.get("planning_allowed")
        if initial is None:
            # non-analyzable (ABSTAINED) record: no baseline, cannot be
            # made actionable at any horizon from this data.
            per_record_transitions.append(
                {
                    "run_id": r["run_id"],
                    "actionable_initial": None,
                    "actionable_within_1": None,
                    "actionable_within_2": None,
                    "actionable_within_3": None,
                    "n_rich_rounds_observed": 0,
                    "became_actionable_at_round": None,
                }
            )
            continue

        state = bool(initial)
        by_round_state = [state]
        rich_by_index = {sr["round"]: sr for sr in rich_rounds_sorted}
        became_at = 0 if state else None
        for k in range(3):
            if k in rich_by_index and rich_by_index[k].get("planning_allowed_after") is True:
                state = True
                if became_at is None:
                    became_at = k + 1
            by_round_state.append(state)
        per_record_transitions.append(
            {
                "run_id": r["run_id"],
                "actionable_initial": bool(initial),
                "actionable_within_1": by_round_state[1],
                "actionable_within_2": by_round_state[2],
                "actionable_within_3": by_round_state[3],
                "n_rich_rounds_observed": len(rich_rounds_sorted),
                "became_actionable_at_round": became_at,
            }
        )

    def rate(key: str) -> dict[str, Any]:
        known = [t[key] for t in per_record_transitions if t[key] is not None]
        return {
            "n_known": len(known),
            "eligible_count": sum(1 for x in known if x),
            "eligible_rate_of_264_total": sum(1 for x in known if x) / total_records,
            "eligible_rate_of_known": (sum(1 for x in known if x) / len(known)) if known else None,
        }

    became_at_values = [t["became_actionable_at_round"] for t in per_record_transitions if t["became_actionable_at_round"] is not None]
    became_at_values_sorted = sorted(became_at_values)
    median_samples_to_actionability = (
        became_at_values_sorted[len(became_at_values_sorted) // 2] if became_at_values_sorted else None
    )

    actionable_within_n = {
        "data_support": (
            "PARTIALLY SUPPORTED BY REAL DATA: 213/264 records' sample_rounds contain only a "
            "single immediate STOP entry (http_status=409, sampling request rejected, no real "
            "sample incorporated) so no round-level transition is observable for them beyond "
            "the initial planning_allowed state. Only records whose sample_rounds entries carry "
            "the richer schema (entropy_before/after, candidate_size_before/after, "
            "planning_allowed_after, recommended_node -- i.e. a real sample was actually "
            "incorporated and the incident was reanalyzed) provide genuine round-by-round "
            "transition evidence."
        ),
        "records_with_at_least_one_rich_round": records_with_rich_rounds,
        "records_with_at_least_one_rich_round_fraction_of_264": records_with_rich_rounds / total_records,
        "total_rich_rounds_observed": total_rich_rounds,
        "actionable_initial": rate("actionable_initial"),
        "actionable_within_1": rate("actionable_within_1"),
        "actionable_within_2": rate("actionable_within_2"),
        "actionable_within_3": rate("actionable_within_3"),
        "median_samples_to_actionability_among_those_that_became_actionable_via_sampling": (
            median_samples_to_actionability
        ),
        "n_that_became_actionable_via_sampling_1_to_3_rounds": len(became_at_values),
        "top1_correct_within_n": (
            "NOT RUN -- per-round sample_rounds entries record entropy_before/after, "
            "candidate_size_before/after, expected_information_gain, and recommended_node (the "
            "next node chosen to sample), but do NOT record a per-round belief/posterior "
            "snapshot or an explicit top-1 node identity after each round. There is therefore "
            "no way to determine, from this dataset alone, whether the top-1 localization guess "
            "would be correct after N rounds -- only whether the incident would be "
            "planning-ELIGIBLE (a distinct, gate-based notion) after N rounds, which is reported "
            "above. Recomputing true per-round top-1 correctness would require re-running the "
            "live harness with per-round belief capture, which is out of scope for this "
            "data-mining task."
        ),
        "per_record_transitions_sample": per_record_transitions[:5],
    }

    report = {
        "schema_version": 1,
        "sections": "27_suppression_decomposition, 28_evidence_sufficiency_diagnostic, 29_actionable_within_n",
        "locked_test_opened": locked_before,
        "source_data": str(DATA_PATH.relative_to(ROOT)),
        "total_records_in_dataset": total_records,
        "suppression_reason_decomposition": suppression_decomposition,
        "gate_counterfactual_actionability": gate_counterfactual_actionability,
        "evidence_sufficiency_diagnostic": evidence_sufficiency_diagnostic,
        "actionable_within_n": actionable_within_n,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    locked_after = locked_test_opened(ROOT)
    print(
        json.dumps(
            {
                "n_analyzable": n_analyzable,
                "dominant_gate_family": suppression_decomposition["dominant_blocker_gate_family"],
                "observed_eligible_rate_264": gate_counterfactuals["_headline_comparison"][
                    "observed_planning_eligible_rate_of_264"
                ],
                "all_except_calibration_eligible_rate": gate_counterfactuals["all_except_calibration"][
                    "eligible_fraction_of_264_total"
                ],
                "all_except_candidate_size_eligible_rate": gate_counterfactuals["all_except_candidate_size"][
                    "eligible_fraction_of_264_total"
                ],
                "inference_quality_only_eligible_rate": gate_counterfactuals["inference_quality_only"][
                    "eligible_fraction_of_264_total"
                ],
                "evidence_sufficiency_verdict": evidence_sufficiency_diagnostic["verdict"],
                "locked_test_opened_after": locked_after,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
