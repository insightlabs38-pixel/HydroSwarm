"""M10.3B -- write the final root-cause classification and closure
artifacts from the already-collected diagnostic data (no training, no
checkpoint access, no locked data). Run once, after
`run_m10_3b_diagnosis.py` has produced every other m10-3b-*.json artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m10_common as m10  # noqa: E402

M10_3B_DIR = m10.M10_DIR / "m10-3b-diagnosis"

CATEGORIES = (
    "MECHANICAL_TARGET_OR_RANKING_DEFECT", "TARGET_DEGENERACY", "CANDIDATE_POPULATION_DEGENERACY",
    "LEGAL_INPUT_INFORMATION_INSUFFICIENT", "FROZEN_REPRESENTATION_INSUFFICIENT",
    "METRIC_OR_GATE_MISINTERPRETATION", "GENUINE_MODEL_CAPACITY_LIMIT", "INCONCLUSIVE",
)


def main() -> None:
    locked_before = m10.assert_locked_test_closed()

    root_cause = {
        "kind": "M10_3B_ROOT_CAUSE_CLASSIFICATION",
        "valid_categories": list(CATEGORIES),
        "per_criterion": {
            "plan_value": {
                "categories": ["CANDIDATE_POPULATION_DEGENERACY"],
                "evidence": [
                    "82.5% of validation incidents (240/291 with >=2 valid candidates) have ALL valid candidates' plan_value effectively tied at the 0.05 near-tie tolerance (m10-3b-within-incident-variance.json); 0% of incidents have 3+ meaningfully distinguishable candidates.",
                    "Root physical mechanism: the three link-target templates capable of a materially different physical outcome (ISOLATE_SOURCE/ISOLATE_AND_FLUSH/ALTERNATE_VALVE_CUT) are REJECTED with rejection_codes=('PRESSURE_BELOW_MINIMUM',) in 100% of sampled incidents on this network topology -- a genuine, exact-WNTR-verified safety-gate rejection, not a bug (m10-3b-candidate-diversity.json's isolation_template_rejection_diagnostic).",
                    "The 5 templates that DO pass verification besides FLUSH_DOWNSTREAM (NO_ACTION/PROTECT_CRITICAL/INCREASE_MONITORING/REQUEST_SAMPLE/WAIT_OBSERVE) never modify the network and are numerically IDENTICAL to NO_ACTION's own consequences (exposure_proxy_mean=1.0, containment_time_proxy_mean=1.9441580756013745 for all five, to machine precision).",
                    "Legal decision-time template identity explains only 0.11% of plan_value's total variance (m10-3b-feature-identifiability.json) -- consistent with there being almost nothing for that signal to predict, not with the signal being unavailable to the model.",
                    "Oracle utility: even a perfect oracle beats trivial NO_ACTION by more than the near-tie tolerance in only 9.6% of incidents; NO_ACTION is already within tolerance of the pool optimum in 90.4% of incidents (m10-3b-oracle-utility.json).",
                ],
                "explicitly_ruled_out": {
                    "MECHANICAL_TARGET_OR_RANKING_DEFECT": "ranking-alignment-audit found zero mechanical sign/order/mask/index defects (all 6 checks passed); formula matches the frozen definition exactly (formula audit + synthetic monotonicity + real-code unit tests).",
                    "FROZEN_REPRESENTATION_INSUFFICIENT": "not invoked -- there is essentially no within-incident signal for any representation, frozen or not, to extract in 82.5% of incidents; a richer representation cannot rank candidates whose true outcomes are tied.",
                },
            },
            "exposure_proxy": {
                "categories": ["CANDIDATE_POPULATION_DEGENERACY"],
                "evidence": [
                    "90.4% of valid candidates have exposure_proxy EXACTLY 1.0 (m10-3b-target-identifiability.json) -- the same physically-inert-template mechanism as plan_value.",
                    "Only 23.0% of incidents have 2+ meaningfully distinguishable (tolerance 0.01) exposure_proxy values among valid candidates; 0% have 3+ distinguishable clusters.",
                ],
                "explicitly_ruled_out": {"MECHANICAL_TARGET_OR_RANKING_DEFECT": "same as plan_value."},
            },
            "pressure_risk_proxy": {
                "categories": ["TARGET_DEGENERACY", "CANDIDATE_POPULATION_DEGENERACY"],
                "evidence": [
                    "MATHEMATICALLY PROVEN degenerate: PlanVerifier rejects (PRESSURE_BELOW_MINIMUM) any plan whose pressure_violation_minutes would be nonzero, using the SAME minimum_pressure_m threshold for both the rejection decision and the violation-minutes computation (src/hydroswarm/simulation/wrapper.py:1341/1349, 1477). Therefore pressure_risk_proxy == 0.0 for every plan whose plan_validity=True, by construction of the safety gate -- confirmed both by direct source-code trace and empirically (std=0.0, n_unique=1 across all 1734 valid candidates).",
                    "Also CANDIDATE_POPULATION_DEGENERACY: the only 3 templates that could ever exercise nonzero pressure risk (the isolation templates) never reach VERIFIED status in this population at all.",
                ],
                "explicitly_ruled_out": {
                    "MECHANICAL_TARGET_OR_RANKING_DEFECT": "the formula correctly incorporates a nonzero pressure_risk_proxy when one reaches it (test_pressure_risk_proxy_cost_component_present_when_nonzero) -- the zero is a population/gate property, not a formula bug.",
                    "FROZEN_REPRESENTATION_INSUFFICIENT": "the target is mathematically constant for every example this task's own gate ever evaluates; no representation can learn a nonconstant function of a constant.",
                },
                "governance_note": "This finding must NOT be used to justify weakening the pressure safety threshold -- forbidden by frozen governance rules regardless of any capability-metric benefit.",
            },
            "service_loss_proxy": {
                "categories": ["CANDIDATE_POPULATION_DEGENERACY"],
                "evidence": [
                    "Bounded (not mathematically forced, unlike pressure_risk_proxy) to [0, 0.10] for any valid candidate by the SAME verifier gate (SERVICE_BELOW_MINIMUM, floor 0.90), but empirically ~5.99e-13 mean (effectively exactly 0) across all 1734 valid candidates in this population -- a candidate/population characteristic (the templates that pass verification here don't meaningfully disrupt service), not a mathematical necessity like pressure_risk_proxy.",
                ],
                "explicitly_ruled_out": {"MECHANICAL_TARGET_OR_RANKING_DEFECT": "formula correctly reflects 1-service_availability; no clipping/masking defect found."},
            },
            "plan_regret_proxy": {
                "categories": ["CANDIDATE_POPULATION_DEGENERACY"],
                "evidence": [
                    "EXACT deterministic bijective transform of plan_value (Spearman -0.99999910, max reconstruction error 3.2e-8) -- carries identical information and an identical root cause to plan_value's own entry.",
                ],
                "explicitly_ruled_out": {"MECHANICAL_TARGET_OR_RANKING_DEFECT": "verified by direct unit test (test_plan_value_is_exact_deterministic_function_of_regret) and empirical reconstruction check."},
            },
            "pairwise_ranking": {
                "categories": ["CANDIDATE_POPULATION_DEGENERACY"],
                "evidence": [
                    "This criterion only has a non-trivial denominator in incidents with a real, non-tied, both-valid pair -- exactly the incidents the within-incident-variance artifact shows are rare and low-margin for plan_value (17.5% distinguishable-at-all, 0% with 3+ distinguishable clusters). A near-chance/sub-chance pairwise score on a population where most 'pairs' are near-ties by construction is the EXPECTED behavior of a correctly-trained model facing noise-dominated labels, not evidence the model failed to learn a real signal.",
                    "Ranking-alignment-audit (mechanical) found zero sign/order/mask/index defects in the pairwise comparator itself.",
                ],
                "explicitly_ruled_out": {
                    "MECHANICAL_TARGET_OR_RANKING_DEFECT": "6/6 mechanical checks passed (m10-3b-ranking-alignment-audit.json).",
                    "FROZEN_REPRESENTATION_INSUFFICIENT": "cannot be concluded from a metric whose own denominator is dominated by near-ties; see decision-tree reasoning in the closure document.",
                },
            },
            "containment_time_proxy": {
                "categories": ["METRIC_OR_GATE_MISINTERPRETATION", "TARGET_DEGENERACY"],
                "evidence": [
                    "M10.3A's Level-A gate criterion 4 (MSE + Spearman 'pass') is computed POOLED across ALL validation candidates (run_m10_3_level_a_gate.py's own `_spearman_ci(p, t, ...)` on the full concatenated arrays), NOT per-incident -- unlike the dedicated within-incident pairwise-ranking criterion, which exists ONLY for plan_value.",
                    "Within-incident, only 5.15% of incidents have a meaningfully distinguishable (tolerance 1/240) containment_time_proxy spread among valid candidates; 94.8% are effectively all-tied within-incident (m10-3b-within-incident-variance.json).",
                    "containment_time_proxy DOES have real BETWEEN-incident variance (14 distinct global values, IQR=1.75 on validation, m10-3b-target-identifiability.json) -- plausibly explained by incident-level severity/topology differences rather than by which candidate template is chosen.",
                    "This means containment_time_proxy's strong pooled-Spearman 'pass' in M10.3A most likely reflects the model learning to predict per-INCIDENT severity (a real, legitimate, but between-incident signal available even from NO_ACTION-only candidates), not genuine within-incident candidate discrimination -- the SAME candidate-population degeneracy documented above, just not caught by this particular (pooled) gate criterion the way it was caught by plan_value's dedicated pairwise-ranking criterion.",
                ],
                "explicitly_ruled_out": {"MECHANICAL_TARGET_OR_RANKING_DEFECT": "formula/masking verified correct; the 'pass' is real given the metric as specified, the concern is what the metric as specified actually measures."},
            },
        },
        "no_criterion_supports_frozen_representation_insufficient": True,
        "no_criterion_supports_mechanical_defect": True,
    }
    (M10_3B_DIR / "m10-3b-root-cause.json").write_text(json.dumps(root_cause, indent=2, default=str) + "\n")

    closure = {
        "kind": "M10_3B_DIAGNOSIS_CLOSURE",
        "milestone": "M10.3B",
        "branch": m10.current_branch(),
        "commit_at_closure_time": m10.current_commit(),
        "amends": "docs/evaluation/HYDROCORE_V5_M10_3B_STRATEGIST_DIAGNOSIS.md -- additive to M10.3A, does not reopen/reverse/alter M10.3A's closure",
        "central_question": "Did M10.3A fail because HydroCore needs broader retraining, or because the current Strategist objective/population does not contain enough correct, legally observable, within-incident decision signal to justify such retraining?",
        "answer": (
            "The current Strategist objective/population does not contain enough within-incident "
            "decision signal, for a concrete, mechanistically-proven physical reason specific to "
            "this task's single-family/single-depth pilot scope -- NOT because HydroCore's frozen "
            "representation is insufficient. No broader/full retrain is scientifically justified by "
            "this evidence."
        ),
        "M10_3B_DECISION": "M10_3B_POPULATION_AMENDMENT_REQUIRED",
        "decision_reasoning": (
            "Section 20(D) (M10_3B_LEARNED_STRATEGIST_NOT_JUSTIFIED) does not yet apply: this task's "
            "own evidence traces the degeneracy to a SPECIFIC, narrow, already-disclosed pilot-scope "
            "choice (M10.3A Part 7's own frozen protocol: family='golden-reference' ONLY, depth=25 "
            "'MATURE' ONLY -- mirroring M10.2's precedent, not an exhaustive population) rather than "
            "a fundamental property of every realistic candidate population the system could "
            "legitimately generate. Two already-governed, already-trained-on TRAINED_FAMILIES "
            "(branched-loop, loop-grid -- see scripts/hydrocore_v5/run_m7_topology.py) and five "
            "already-governed depth buckets (1,2,3,4,6 -- EARLY/MID severity, never sampled here) "
            "exist and were never evaluated for Strategist candidate diversity. Because the isolation-"
            "template rejection mechanism this task identified (PRESSURE_BELOW_MINIMUM, a hydraulic-"
            "redundancy property of THIS SPECIFIC network under MATURE-depth severity) is plausibly, "
            "but not yet confirmedly, specific to golden-reference/MATURE, declaring D "
            "(LEARNED_STRATEGIST_NOT_JUSTIFIED) now would extrapolate beyond this task's own evidence. "
            "Declaring C (BROADER_REFIT_SCIENTIFICALLY_JUSTIFIED) is directly contradicted by the "
            "evidence: root-cause classification found CANDIDATE_POPULATION_DEGENERACY and mathematically-"
            "forced TARGET_DEGENERACY for every failed criterion, and explicitly found no support for "
            "FROZEN_REPRESENTATION_INSUFFICIENT anywhere. Declaring A (CORRECTION_REQUIRED) is directly "
            "contradicted: the ranking-alignment/target-formula/leakage audits found zero mechanical "
            "defects across 6 independent mechanical checks plus real-code unit tests."
        ),
        "specified_population_amendment_scope": {
            "principle": "Development-only, must remain within already-existing, already-governed system semantics (existing TRAINED_FAMILIES, existing depth buckets, existing deterministic candidate templates) -- no new/invented candidate templates, no weakened safety thresholds, no hand-crafted implausible plans.",
            "recommended_additions_for_a_future_M10_3C_population_amendment": [
                "Include the OTHER 2 already-trained families (branched-loop, loop-grid) in addition to golden-reference -- different topology redundancy may allow the isolation templates to remain WNTR-VERIFIED (not systematically PRESSURE_BELOW_MINIMUM) on at least some incidents, restoring real exposure/pressure/plan_value tradeoffs.",
                "Include EARLY/MID depth buckets (1,2,3,4,6) in addition to MATURE (25) -- lower-severity incidents may need less aggressive isolation, making PROTECT_CRITICAL/FLUSH_DOWNSTREAM-class actions genuinely differentiate from NO_ACTION even where isolation remains infeasible.",
                "The amendment must FREEZE this expanded population and its own frozen protocol BEFORE inspecting any Level-A result, per this task's own governance rules -- this diagnosis provides the motivating evidence, not the frozen protocol itself.",
                "The amendment should re-run this SAME within-incident-identifiability/oracle-utility diagnostic on the expanded population BEFORE any training, to confirm the expansion actually restores meaningful within-incident decision signal (never assume it will).",
            ],
            "if_the_amendment_finds_the_same_degeneracy_on_the_expanded_population": (
                "Then M10_3B_LEARNED_STRATEGIST_NOT_JUSTIFIED (Section 20-D) becomes the well-evidenced "
                "conclusion, and the system should retain the deterministic candidate generator + "
                "deterministic Strategist + exact WNTR verification permanently for this decision, "
                "proceeding to M10.4 on that basis."
            ),
        },
        "does_not_authorize": [
            "true M10.3", "Strategist retraining of any kind", "another Level A or Level B",
            "a full/shared HydroCore retrain", "opening locked final/topology tests",
            "altering closed M9/M10 results", "the M10.3C population amendment itself (a separately authorized future task)",
        ],
        "m9_m10_historical_artifacts_unchanged": True,
        "m10_3a_closure_unaltered": True,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": None,  # filled below
    }
    locked_after = m10.assert_locked_test_closed()
    closure["locked_test_opened_after"] = locked_after
    (M10_3B_DIR / "m10-3b-closure.json").write_text(json.dumps(closure, indent=2, default=str) + "\n")
    print("wrote root-cause and closure artifacts")
    print(json.dumps({"decision": closure["M10_3B_DECISION"], "locked_before": locked_before, "locked_after": locked_after}, indent=2))


if __name__ == "__main__":
    main()
