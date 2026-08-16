"""Writes the M9.5R PRE-EXECUTION PROTOCOL FREEZE artifact
(`reports/evaluation/hydrocore-v5/m9-5r/m9-5r-protocol.json`).

Per the governing M9.5R prompt Section 3/20 ("Prefer a protocol/freeze
commit BEFORE running fresh calibration/development inference" /
"Before generating new data: record ... Then execute against that frozen
protocol"), this script MUST be run -- and its output committed -- before
`run_m9_5r_source_representative.py` or `run_m9_5r_decide.py` ever perform
model inference or generate scenario data. It captures the frozen decision
logic, gates, seed ranges, and calibration identity in a form that a human
reviewer can diff against later to confirm nothing was changed post-hoc.

This script performs NO model inference, NO scenario generation, and reads
NO locked test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m9_5r_common as m5r  # noqa: E402


def main() -> int:
    m5r.M9_5R_DIR.mkdir(parents=True, exist_ok=True)

    locked_before = m5r.assert_locked_test_closed()
    start_commit = m5r.current_commit()
    start_branch = m5r.current_branch()
    assert start_branch == m5r.FROZEN_BRANCH

    protocol = {
        "milestone": "M9.5R",
        "kind": "INDEPENDENT_CALIBRATION_CONFIRMATION",
        "primary_question": (
            "Does the exact existing conformal calibration method, at 20 independent calibration incidents "
            "per source, independently reproduce valid calibration for the frozen INTERLEAVED HydroCore-S "
            "predictor on a fresh, source-representative development population?"
        ),
        "secondary_question": (
            "Does ARM_A/CURRENT also continue to calibrate normally on golden-reference under the same "
            "fresh population policy?"
        ),
        "not_reinterpretation_of_m9_5": True,
        "m9_5_decision_preserved": "E",
        "m9_5_decision_name_preserved": "REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER",
        "branch": start_branch,
        "start_commit": start_commit,
        "provenance": {
            "m9_4_code_commit": m5r.M9_4_CODE_COMMIT,
            "m9_4_metadata_fix_commit": m5r.M9_4_METADATA_FIX_COMMIT,
            "m9_5_code_commit": m5r.M9_5_CODE_COMMIT,
            "m9_5_metadata_fix_commit": m5r.M9_5_METADATA_FIX_COMMIT,
            "m9_4_closure_sha256": m5r.checkpoint_sha256(str(m5r.M9_4_CLOSURE_PATH)),
            "m9_5_closure_sha256": m5r.checkpoint_sha256(str(m5r.M9_5_CLOSURE_PATH)),
            "m9_5_manifest_sha256": m5r.checkpoint_sha256(str(m5r.M9_5_MANIFEST_PATH)),
        },
        "hard_non_goals": [
            "no training of ARM_A or ARM_B2", "no optimizer steps", "no backward()", "no fine-tuning",
            "no architecture change", "no AGE_FIX_ONLY change", "no interleaved-recipe change",
            "no alpha change", "no coverage-floor weakening", "no nonconformity-score change",
            "no grouping/fallback/minimum_group_size change", "no APS/RAPS/temperature/isotonic/Platt scaling",
            "no source- or complexity-conditioned conformal extensions", "no support-size sweep",
            "no hyperparameter sweep", "no tuning the calibrator on development data",
            "no tuning trained-family calibration on unseen-topology data", "no opening locked data",
            "no omitting difficult source nodes", "no dropping seeds", "no rerolling an unfavorable draw",
        ],
        "frozen_predictors": {
            "seeds": list(m5r.SEEDS),
            "arm_a": "CURRENT / SINGLE_FAMILY HydroCore-S (AGE_FIX_ONLY export, M8.7)",
            "arm_b2": "STEP_MATCHED_INTERLEAVED_MULTI_FAMILY HydroCore-S (M9.0a export)",
            "checkpoint_verification": "SHA256 before and after inference; must be byte-identical; eval()+no_grad() only",
        },
        "calibration_method_frozen": {
            "calibrator_class": "SplitConformalCalibrator",
            "grouping": "B_DEPTH_AWARE / CURRENT_FAMILY_DEPTH",
            "alpha": m5r.ALPHA,
            "minimum_group_size": m5r.MINIMUM_GROUP_SIZE,
            "coverage_floor": m5r.OPERATIONAL_COVERAGE_FLOOR,
            "nominal_coverage_target": m5r.NOMINAL_COVERAGE_TARGET,
            "reused_from": "hydroswarm.calibration.conformal (unmodified; imported directly, not through m9_5_common)",
        },
        "primary_support_condition": {
            "calibration_repeats_per_source": m5r.CALIBRATION_REPEATS_PER_SOURCE,
            "development_repeats_per_source": m5r.DEVELOPMENT_REPEATS_PER_SOURCE,
            "no_support_sweep": True,
            "note": "Exactly one primary calibration-support condition: 20 independent physical incidents per source. No 4/8/12 levels exist in this milestone.",
        },
        "trained_families": list(m5r.TRAINED_FAMILIES),
        "expected_incident_counts": {
            family: {
                "n_sources": len(m5r.full_junction_list(family, m5r.ALL_FAMILY_LOADERS[family])),
                "calibration_incidents": len(m5r.full_junction_list(family, m5r.ALL_FAMILY_LOADERS[family])) * m5r.CALIBRATION_REPEATS_PER_SOURCE,
                "development_incidents": len(m5r.full_junction_list(family, m5r.ALL_FAMILY_LOADERS[family])) * m5r.DEVELOPMENT_REPEATS_PER_SOURCE,
            }
            for family in m5r.TRAINED_FAMILIES
        },
        "seed_namespace": {
            "seed_base_floor": m5r.M9_5R_SEED_BASE_FLOOR,
            "seed_base_step": m5r.M9_5R_SEED_BASE_STEP,
            "source_stride": m5r.M9_5R_SOURCE_STRIDE,
            "roles": list(m5r.M9_5R_ROLES),
            "seed_bases": {f"{k[0]}|{k[1]}": v for k, v in m5r.M9_5R_SEED_BASES.items()},
            "disjoint_from": ["M7", "M9.4 (calibration_m9_4/development_m9_4)", "M9.5 (calibration_m9_5/development_m9_5)", "locked_final_test", "locked_topology_test"],
        },
        "representativeness_gate": {
            "required_checks": [
                "identical_complete_source_support_calibration_and_development", "20_calibration_incidents_per_source",
                "20_development_incidents_per_source", "balanced_source_distribution", "no_zero_support_source",
                "no_incident_overlap", "no_generator_seed_overlap", "same_topology_definition",
                "same_event_generation_mechanism", "same_event_severity_distribution_policy",
                "same_contamination_timing_policy", "same_sensor_availability_mechanism", "same_missingness_mechanism",
                "same_noise_mechanism", "same_cadence_timing_mechanism", "same_physical_perturbation_mechanism",
            ],
            "on_fail": "STOP -- do not calculate a model-selection conclusion",
        },
        "corrected_sanity_gate": {
            "purpose": "Establish the intended pipeline was executed correctly, NOT that fresh stochastic data reproduces any previous numerical outcome.",
            "required_checks": [
                "A_checkpoint_sha_identity", "B_calibrator_class_identity_matches_m9_5",
                "C_alpha_equals_0_1", "D_nonconformity_score_implementation_identity_matches",
                "E_grouping_construction_identity_matches", "F_maturity_depth_mapping_identity_matches",
                "G_candidate_set_inclusion_rule_matches", "H_source_node_ordering_correct",
                "I_calibration_development_split_disjointness_passes", "J_all_source_nodes_represented",
                "K_all_outputs_finite", "L_resubstitution_diagnostic_numerically_plausible",
            ],
            "must_not_require": [
                "m9_4_like_bad_coverage", "m9_5_like_good_coverage", "any_particular_performance_direction",
                "any_specific_numerical_calibration_result",
            ],
            "on_fail": "STOP before scientific interpretation -- M9_5R_DECISION = D",
        },
        "candidate_set_guard": {
            "rule": "pathological if full_set_rate > threshold at the primary support level (same rule M9.5 used)",
            "threshold": m5r.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD,
        },
        "interleaved_confirmation_gate": {
            "required_cells": [f"{family}|{seed}" for family in m5r.TRAINED_FAMILIES for seed in m5r.SEEDS],
            "n_required_cells": len(m5r.TRAINED_FAMILIES) * len(m5r.SEEDS),
            "rule": "marginal coverage >= coverage_floor in ALL cells, no averaging away failures",
        },
        "current_control_gate": {
            "required_cells": [f"golden-reference|{seed}" for seed in m5r.SEEDS],
            "n_required_cells": len(m5r.SEEDS),
            "rule": "marginal coverage >= coverage_floor in ALL cells",
        },
        "decision_logic": {
            "codes": m5r.DECISION_NAMES,
            "evaluation_order": [
                "D if representativeness_audit FAILS or sanity_gate FAILS",
                "C if CURRENT control does not pass 3/3 cells",
                "B if INTERLEAVED does not pass 9/9 cells",
                "E if coverage passes (C and B conditions clear) but candidate_set_guard is pathological",
                "A if representativeness PASS, sanity PASS, CURRENT 3/3, INTERLEAVED 9/9, candidate_set_guard PASS, all finite, no safety issue",
                "F only if a genuinely unforeseen statistical/engineering condition prevents interpretation despite valid execution",
            ],
            "frozen_before_data_generation": True,
            "anti_post_hoc_rule": "Decision logic, gates, thresholds, and seed ranges above are committed in this protocol artifact BEFORE any M9.5R inference is run. If a flaw is discovered after results, the result is preserved and the flaw is reported -- not silently patched and rerun.",
        },
        "artifacts_directory": str(m5r.M9_5R_DIR.relative_to(m5r.ROOT_PATH)),
        "locked_test_opened_before_protocol_freeze": locked_before,
    }

    m5r.M9_5R_PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {m5r.M9_5R_PROTOCOL_PATH}", flush=True)
    print(json.dumps({"start_commit": start_commit, "locked_test_opened_before": locked_before}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
