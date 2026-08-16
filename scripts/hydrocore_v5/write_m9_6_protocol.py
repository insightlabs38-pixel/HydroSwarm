"""Writes the M9.6 PRE-TRAINING PROTOCOL FREEZE artifact
(`reports/evaluation/hydrocore-v5/m9-6/m9-6-protocol.json`).

Per the governing M9.6 prompt Section 5 ("Before any M9.6 training: create
and commit a frozen M9.6 protocol artifact ... Do not begin training before
this protocol is frozen. Once data/results exist: DO NOT change the
protocol or decision thresholds"), this script MUST be run -- and its
output committed -- before any M9.6 training, evaluation, or calibration
script executes. No model construction, no scenario generation, no locked
test access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m9_6_common as m6  # noqa: E402


def main() -> int:
    m6.M9_6_DIR.mkdir(parents=True, exist_ok=True)

    locked_before = m6.assert_locked_test_closed()
    start_commit = m6.current_commit()
    start_branch = m6.current_branch()
    assert start_branch == m6.FROZEN_BRANCH

    protocol = {
        "milestone": "M9.6",
        "kind": "EXACT_COMPUTE_PARITY_FINAL_HYDROCORE_S_CONFIRMATION",
        "primary_question": (
            "Under a freshly trained, exactly matched 1350-optimizer-step protocol, does multi-topology "
            "INTERLEAVED HydroCore-S still outperform CURRENT HydroCore-S on source-representative unseen "
            "topology localization without materially regressing known-family performance, while satisfying "
            "the independently confirmed M9.5R calibration policy?"
        ),
        "not_an_architecture_search": True,
        "closed_axes": [
            "GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE", "Mamba", "S4", "Mamba-2", "SSMs", "transformer variants",
            "new graph architectures", "HydroCore-M", "HydroCore-L", "another temporal encoder",
            "causal-depth policy search", "task loss/weight search", "optimizer/LR/scheduler/batch search",
            "calibration-method search",
        ],
        "branch": start_branch,
        "start_commit": start_commit,
        "provenance": {
            "m9_4_closure_sha256": m6.checkpoint_sha256(str(m6.M9_4_CLOSURE_PATH)),
            "m9_5_closure_sha256": m6.checkpoint_sha256(str(m6.M9_5_CLOSURE_PATH)),
            "m9_5r_closure_sha256": m6.checkpoint_sha256(str(m6.M9_5R_CLOSURE_PATH)),
            "m9_5r_manifest_sha256": m6.checkpoint_sha256(str(m6.M9_5R_MANIFEST_PATH)),
            "m9_5r_code_commit": m6.M9_5R_CODE_COMMIT,
            "m9_5r_artifact_commit": m6.M9_5R_ARTIFACT_COMMIT,
            "m9_5r_metadata_fix_commit": m6.M9_5R_METADATA_FIX_COMMIT,
        },
        "arms": {
            "ARM_A_M9_6": {
                "name": "CURRENT / SINGLE_FAMILY", "trained_families": ["golden-reference"],
                "code_path": "Trainer-based, mirrors run_m8_7_arm.py's AGE_FIX_ONLY arm exactly (build_scenario_pool, ARM_POLICIES['A'] full-history depth policy)",
                "train_scenario_count": 600, "validation_scenario_count": 100,
            },
            "ARM_B_M9_6": {
                "name": "INTERLEAVED_MULTI_FAMILY", "trained_families": list(m6.TRAINED_FAMILIES),
                "code_path": "hand-rolled step-matched interleaved loop, mirrors run_m9_0a_arm_b2.py's train_arm_b2 exactly (_build_family_pools, step_matched_interleaved_optimizer_step, fixed 3-update rotation)",
                "train_scenario_count_per_family": 200, "total_train_scenario_count": 600,
                "family_weighting": {family: round(1.0 / 3, 6) for family in m6.TRAINED_FAMILIES},
            },
            "architecture_identical": True, "no_architectural_difference_whatsoever": True,
        },
        "model_configuration": {
            "variant": "small", "use_adapters": False, "representation": "AGE_FIX_ONLY",
            "feature_kwargs": {"unobserved_age_sentinel": "fixed", "include_relative_gap_feature": False},
            "model_kwargs": {"temporal_feature_dim": 6, "quality_feature_dim": 4, "elapsed_time_normalization": "window_relative"},
            "expected_approximate_param_count": 4_000_000,
            "pcgrad_enabled": False, "note": "actual parameter count verified and recorded per-run in m9-6-training-parity.json; both arms MUST match exactly",
        },
        "seeds": list(m6.SEEDS),
        "exact_compute_parity": {
            "total_optimizer_steps_required": m6.TOTAL_OPTIMIZER_STEPS,
            "optimizer_steps_per_epoch": list(m6.ARM_A_OPTIMIZER_STEPS_PER_EPOCH),
            "scheduler_total_steps": m6.SCHEDULER_TOTAL_STEPS,
            "no_exceptions": True,
            "early_stopping_disabled_via": "TrainingConfig.early_stopping_patience=0 (existing, already-validated config field; 0 is falsy so Trainer.fit()'s/train_arm_b2()'s own early-exit becomes a structural no-op -- no training-loop code is modified)",
            "canonical_checkpoint_policy": m6.CANONICAL_CHECKPOINT_POLICY,
            "canonical_checkpoint_policy_note": (
                "Trainer.fit()/train_arm_b2() both always re-load the BEST-VALIDATION checkpoint into the "
                "model right before exporting model-export.safetensors, even when all 20 epochs ran without "
                "early stopping (confirmed historically: m8-7-runs/AGE_FIX_ONLY-seed20260815.json has "
                "stopped_early=false, epochs_completed=20, but exported global_steps=1275, not 1350). M9.6 "
                "preserves this best-validation export UNCHANGED for the record, and additionally reloads "
                "the LAST periodic checkpoint (guaranteed epoch 20 with early stopping disabled) into a "
                "fresh model instance and exports it separately as the canonical, promotion-authoritative "
                "M9.6 checkpoint -- FINAL STEP 1350, per this Section's predeclaration."
            ),
            "optimizer": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.01, "gradient_clip_norm": 1.0,
            "warmup_steps": 10, "scheduler": "cosine", "batch_size": 2, "gradient_accumulation_steps": 4,
            "microbatches_per_optimizer_update": 4, "checkpoint_every_epochs": 1, "deterministic": True, "precision": "fp32",
            "config_source": "configs/training-v5-causal.yaml (unmodified; only early_stopping_patience overridden at the TrainingConfig instantiation call site, not in the yaml file)",
        },
        "training_exposure_parity": {
            "arm_a_total_train_scenarios": 600, "arm_a_composition": {"golden-reference": 600},
            "arm_b_total_train_scenarios": 600, "arm_b_composition": {family: 200 for family in m6.TRAINED_FAMILIES},
            "equal_total_exposure_budget": True,
            "family_rotation_policy": "reused unmodified from run_m9_0a_arm_b2.py: fixed 3-update rotation (golden/branched/loop/EXTRA, cycling which family gets the 4th microbatch slot), no new curriculum invented",
        },
        "train_validation_data": {
            "reused_unmodified_from": ["hydroswarm.training.causal_prefix.build_scenario_pool (ARM_A)", "run_m9_0_arm_b._build_family_pools (ARM_B)"],
            "arm_a_seed_range_source": "causal_prefix.SPLIT_SEED_RANGES (~900,000,000-903,999,999)",
            "arm_b_seed_range_source": "run_m7_topology.SEED_BASES[(family,'train'/'validation')] (940,000,000-970,000,000)",
            "disjoint_from_m9_6_calibration_development": True,
            "scope_decision_note": (
                "M9.6 reuses the EXACT historical train/validation pools (same physical scenarios M8.7/M9.0/"
                "M9.0a already used) rather than generating new train/validation seed ranges from scratch. "
                "This is a documented interpretation of governing Section 11: those pools' seed ranges are "
                "already, by construction, physically disjoint from every calibration/development namespace "
                "in this repo including M9.6's own (998,000,000+) -- reusing them satisfies the disjointness "
                "requirement without inventing a new curriculum (Section 6) and keeps ARM_A/ARM_B exposure-"
                "budget comparison apples-to-apples with the historical M9.0a definition, isolating TOPOLOGY "
                "DIVERSITY as the only intended difference (Section 11's own stated goal)."
            ),
        },
        "calibration_method_frozen": {
            "calibrator_class": "SplitConformalCalibrator", "grouping": "B_DEPTH_AWARE / CURRENT_FAMILY_DEPTH",
            "alpha": m6.ALPHA, "minimum_group_size": m6.MINIMUM_GROUP_SIZE, "coverage_floor": m6.OPERATIONAL_COVERAGE_FLOOR,
            "reused_from": "hydroswarm.calibration.conformal (unmodified), method independently confirmed in M9.5R",
        },
        "calibration_population": {
            "repeats_per_source": m6.CALIBRATION_REPEATS_PER_SOURCE, "families": list(m6.TRAINED_FAMILIES),
            "expected_counts": {family: len(m6.full_junction_list(family, m6.ALL_FAMILY_LOADERS[family])) * m6.CALIBRATION_REPEATS_PER_SOURCE for family in m6.TRAINED_FAMILIES},
        },
        "development_population": {
            "repeats_per_source": m6.DEVELOPMENT_REPEATS_PER_SOURCE, "families": list(m6.ALL_FAMILIES),
            "trained_vs_unseen": {"trained": list(m6.TRAINED_FAMILIES), "unseen": list(m6.UNSEEN_DEVELOPMENT_FAMILIES)},
            "expected_counts": {family: len(m6.full_junction_list(family, m6.ALL_FAMILY_LOADERS[family])) * m6.DEVELOPMENT_REPEATS_PER_SOURCE for family in m6.ALL_FAMILIES},
        },
        "seed_namespace": {
            "seed_base_floor": m6.M9_6_SEED_BASE_FLOOR, "seed_base_step": m6.M9_6_SEED_BASE_STEP,
            "source_stride": m6.M9_6_SOURCE_STRIDE, "roles": list(m6.M9_6_ROLES),
            "seed_bases": {f"{k[0]}|{k[1]}": v for k, v in m6.M9_6_SEED_BASES.items()},
            "disjoint_from": ["M7 train/validation", "M9.4", "M9.5", "M9.5R", "locked_final_test", "locked_topology_test"],
        },
        "depths": list(m6.DEPTHS), "maturity_groups": {"EARLY": list(m6.EARLY_DEPTHS), "MID": list(m6.MID_DEPTHS), "MATURE": list(m6.MATURE_DEPTHS)},
        "paired_bootstrap": {
            "resamples": m6.BOOTSTRAP_RESAMPLES, "seed": m6.BOOTSTRAP_SEED, "interval": m6.BOOTSTRAP_INTERVAL,
            "resampling_unit": "physical base incident (all depths stay attached)", "pairing": "family, source, incident_id, physical seed -- same physical scenarios scored under both arms",
        },
        "primary_predictive_promotion_gate": {
            "unseen_macro_family_mature_delta_must_be_positive": True,
            "bootstrap_ci90_lower_bound_must_be_positive": True,
            "min_unseen_families_improved": m6.GENERALIZATION_MIN_UNSEEN_FAMILIES_IMPROVED,
            "max_unseen_family_regression_pp": m6.GENERALIZATION_MAX_UNSEEN_FAMILY_REGRESSION_PP,
            "per_seed_macro_delta_must_be_nonnegative_all_3_seeds": True,
            "all_outputs_finite": True, "no_safety_authority_regression": True,
        },
        "known_family_guardrails": {
            "max_early_top1_regression_pp": m6.GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP,
            "max_mature_top1_regression_pp": m6.GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP,
            "max_mrr_regression": m6.GUARDRAIL_MAX_MRR_REGRESSION,
            "family": "golden-reference (shared, trained by both arms)",
        },
        "calibration_gate": {
            "interleaved_required_cells": [f"{family}|{seed}" for family in m6.TRAINED_FAMILIES for seed in m6.SEEDS],
            "current_required_cells": [f"golden-reference|{seed}" for seed in m6.SEEDS],
            "coverage_floor": m6.OPERATIONAL_COVERAGE_FLOOR,
            "candidate_set_guard_threshold": m6.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD,
        },
        "decision_logic": {
            "codes": m6.DECISION_NAMES,
            "evaluation_order": [
                "E if exact 1350-step parity (or another training invariant: scheduler/optimizer/exposure/param-count/checkpoint-corruption/topology-exposure-mismatch/locked-data-access) fails for ANY arm/seed",
                "F if calibration representativeness OR the M9.5R-identical implementation/sanity invariants fail before a clean decision can be reached",
                "B if compute parity holds but the primary predictive promotion gate (Section 22) fails",
                "D if the predictive gate passes but known-family preservation (Section 23) fails",
                "C if the predictive gate passes, known-family guardrails pass, but the calibration gate (Section 28/29/30) fails",
                "A if exact compute parity PASS, predictive generalization gate PASS, known-family guardrails PASS, calibration representativeness PASS, CURRENT control PASS, INTERLEAVED 9/9 PASS, candidate-set guard PASS, all finite, no safety/authority regression, locks unopened",
                "G only if a genuinely valid but statistically unresolved result remains after all the above are checked",
            ],
            "frozen_before_training": True,
            "anti_post_hoc_rule": "This protocol (arms, seeds, 1350-step requirement, exposure budget, family rotation, seed ranges, promotion gates, decision logic) is committed BEFORE any M9.6 training/evaluation/calibration executes. If a flaw is discovered after results, the result is preserved and the flaw is reported -- not silently patched and rerun.",
        },
        "hard_non_goals": [
            "no architecture change", "no AGE_FIX_ONLY change", "no model width/depth change", "no HydroCore-M/L",
            "no other temporal encoder", "no causal-depth policy change", "no task loss/weight change",
            "no confidence regularization", "no auxiliary losses", "no optimizer change", "no hyperparameter search",
            "no Optuna", "no LR/batch/scheduler tuning", "no post-hoc family-sampling tuning",
            "no early-stopping-driven compute-budget change (other than disabling it entirely, per Section 9)",
            "no alpha change from 0.1", "no conformal score change", "no calibration-method change", "no APS/RAPS",
            "no temperature scaling", "no tuning calibration on development", "no topology-definition change",
            "no omitting difficult source nodes", "no dropping seeds", "no rerunning only failed seeds",
            "no opening locked_final_test", "no opening locked_topology_test", "no automatic production promotion",
        ],
        "artifacts_directory": str(m6.M9_6_DIR.relative_to(m6.ROOT_PATH)),
        "locked_test_opened_before_protocol_freeze": locked_before,
    }

    m6.M9_6_PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {m6.M9_6_PROTOCOL_PATH}", flush=True)
    print(json.dumps({"start_commit": start_commit, "locked_test_opened_before": locked_before}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
