"""Contract tests for Milestone 9.6 (`scripts/hydrocore_v5/m9_6_common.py`,
`write_m9_6_protocol.py`, `run_m9_6_train_arm_a.py`, `run_m9_6_train_arm_b.py`,
`run_m9_6_evaluate.py`, `run_m9_6_calibrate.py`, `run_m9_6_decide.py`) --
the final exact-compute-parity confirmation of the selected HydroCore-S
training recipe.

M9.6 is a CONFIRMATORY TRAINING milestone, not an architecture search.
These tests cover: governance/protocol-freeze correctness, exact
1350-optimizer-step parity (no early-stopping escape hatch, canonical
final-step checkpoint policy), training/validation/calibration/development
seed-range disjointness, no locked-data access, no architecture/
hyperparameter changes, paired-incident bootstrap correctness, macro-family
equal weighting, and decision-logic correctness -- never a promotion
decision by themselves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT / "src"))

import m9_4_common as m4  # noqa: E402
import m9_5_common as m5  # noqa: E402
import m9_5r_common as m5r  # noqa: E402
import m9_6_common as m6  # noqa: E402
import run_m7_topology as m7  # noqa: E402


def _training_run_present(arm: str, seed: int) -> bool:
    return (m6.M9_6_TRAINING_RUNS_DIR / f"{arm}-seed{seed}.json").exists()


def _all_training_runs_present() -> bool:
    return all(_training_run_present(arm, seed) for arm in ("ARM_A_M9_6", "ARM_B_M9_6") for seed in m6.SEEDS)


def _protocol_present() -> bool:
    return m6.M9_6_PROTOCOL_PATH.exists()


def _closure_present() -> bool:
    return m6.M9_6_CLOSURE_PATH.exists()


needs_protocol = pytest.mark.skipif(not _protocol_present(), reason="M9.6 protocol has not been frozen yet in this environment")
needs_training = pytest.mark.skipif(not _all_training_runs_present(), reason="M9.6 training has not completed yet in this environment")
needs_closure = pytest.mark.skipif(not _closure_present(), reason="M9.6 pipeline has not been run yet in this environment")


# ---------------------------------------------------------------------------
# Governance constants.
# ---------------------------------------------------------------------------


def test_alpha_frozen_at_0_1():
    assert m6.ALPHA == 0.1
    assert m6.MINIMUM_GROUP_SIZE == 10
    assert m6.OPERATIONAL_COVERAGE_FLOOR == 0.85


def test_exactly_1350_optimizer_steps_required():
    assert m6.TOTAL_OPTIMIZER_STEPS == 1350
    assert sum(m6.ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == 1350
    assert len(m6.ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == 20
    assert m6.SCHEDULER_TOTAL_STEPS == 1500


def test_exactly_two_arms_three_seeds():
    assert m6.ARMS == ("ARM_A_M9_6", "ARM_B_M9_6")
    assert m6.SEEDS == (20260814, 31874, 20260815)
    assert len(m6.SEEDS) == 3


def test_canonical_checkpoint_policy_is_final_step():
    assert m6.CANONICAL_CHECKPOINT_POLICY == "FINAL_STEP_1350"


def test_no_locked_test_access():
    assert m6.assert_locked_test_closed() is False


def test_decision_codes_frozen():
    assert set(m6.DECISION_NAMES.keys()) == {"A", "B", "C", "D", "E", "F", "G"}
    assert m6.DECISION_NAMES["A"] == "EXACT_COMPUTE_INTERLEAVED_CONFIRMATION_PASS"
    assert m6.DECISION_NAMES["E"] == "COMPUTE_PARITY_OR_TRAINING_PROTOCOL_BLOCKER"


def test_trained_and_unseen_families():
    assert set(m6.TRAINED_FAMILIES) == {"golden-reference", "branched-loop", "loop-grid"}
    assert set(m6.UNSEEN_DEVELOPMENT_FAMILIES) == {"coastal-branch", "tree-branch", "dense-loop"}
    assert set(m6.ALL_FAMILIES) == set(m6.TRAINED_FAMILIES) | set(m6.UNSEEN_DEVELOPMENT_FAMILIES)


def test_depths_and_maturity_groups_unchanged():
    assert m6.DEPTHS == (1, 2, 3, 4, 6, 12, 25)
    assert m6.EARLY_DEPTHS == (1, 2, 3)
    assert m6.MID_DEPTHS == (4, 6)
    assert m6.MATURE_DEPTHS == (12, 25)


def test_candidate_set_guard_reused_from_m9_5r_unchanged():
    assert m6.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD == m5r.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD == 0.8


def test_bootstrap_constants_predeclared():
    assert m6.BOOTSTRAP_RESAMPLES == 2000
    assert m6.BOOTSTRAP_SEED == 20260818
    assert m6.BOOTSTRAP_INTERVAL == 0.90


def test_promotion_gate_thresholds_predeclared_unchanged_from_m9_4():
    assert m6.GENERALIZATION_MIN_UNSEEN_FAMILIES_IMPROVED == 2
    assert m6.GENERALIZATION_MAX_UNSEEN_FAMILY_REGRESSION_PP == 5.0
    assert m6.GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP == 5.0
    assert m6.GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP == 3.0
    assert m6.GUARDRAIL_MAX_MRR_REGRESSION == 0.03


# ---------------------------------------------------------------------------
# Seed-range disjointness (train/validation reuse M7's pools; calibration/
# development get a fresh M9.6-only namespace disjoint from M9.4/M9.5/M9.5R).
# ---------------------------------------------------------------------------


def test_m9_6_seed_bases_disjoint_from_m7_m9_4_m9_5_m9_5r_ranges():
    assert min(m7.SEED_BASES.values()) >= 940_000_000
    assert max(m7.SEED_BASES.values()) < m6.M9_6_SEED_BASE_FLOOR
    m9_4_ceiling = max(m4.M9_4_SEED_BASES.values()) + m4.M9_4_SEED_BASE_STEP
    m9_5_ceiling = max(m5.M9_5_SEED_BASES.values()) + m5.M9_5_SEED_BASE_STEP
    m9_5r_ceiling = max(m5r.M9_5R_SEED_BASES.values()) + m5r.M9_5R_SEED_BASE_STEP
    assert m9_4_ceiling <= m5.M9_5_SEED_BASE_FLOOR
    assert m9_5_ceiling <= m5r.M9_5R_SEED_BASE_FLOOR
    assert m9_5r_ceiling <= m6.M9_6_SEED_BASE_FLOOR
    assert min(m6.M9_6_SEED_BASES.values()) >= m6.M9_6_SEED_BASE_FLOOR


def test_m9_6_calibration_and_development_seed_ranges_disjoint_per_family():
    for family in m6.ALL_FAMILIES:
        cal_base = m6.m9_6_seed_base(family, "calibration_m9_6")
        dev_base = m6.m9_6_seed_base(family, "development_m9_6")
        junctions = m6.full_junction_list(family, m6.ALL_FAMILY_LOADERS[family])
        cal_max = cal_base + (len(junctions) - 1) * m6.M9_6_SOURCE_STRIDE + (m6.CALIBRATION_REPEATS_PER_SOURCE - 1)
        assert cal_max < dev_base or dev_base + (len(junctions) - 1) * m6.M9_6_SOURCE_STRIDE + (m6.DEVELOPMENT_REPEATS_PER_SOURCE - 1) < cal_base


def test_m9_6_seed_base_table_covers_every_family_and_role_exactly_once():
    expected_keys = {(family, role) for family in m6.ALL_FAMILIES for role in m6.M9_6_ROLES}
    assert set(m6.M9_6_SEED_BASES.keys()) == expected_keys
    assert len(set(m6.M9_6_SEED_BASES.values())) == len(m6.M9_6_SEED_BASES)


def test_train_validation_pools_use_seed_ranges_disjoint_from_calibration_development():
    """ARM_A reuses causal_prefix.SPLIT_SEED_RANGES (~900M-904M); ARM_B
    reuses run_m7_topology.SEED_BASES train/validation roles (940M-970M).
    Both must stay clear of M9.6's own calibration/development floor."""

    from hydroswarm.training.causal_prefix import SPLIT_SEED_RANGES

    for _split, (start_seed, count) in SPLIT_SEED_RANGES.items():
        max_seed = start_seed + (count - 1) * 100
        assert max_seed < m6.M9_6_SEED_BASE_FLOOR
    for (family, role), base in m7.SEED_BASES.items():
        if role in ("train", "validation"):
            assert base < m6.M9_6_SEED_BASE_FLOOR


# ---------------------------------------------------------------------------
# Training-code governance: no early-stopping escape hatch, no resume from
# historical checkpoints, no hyperparameter search, no architecture change.
# ---------------------------------------------------------------------------


def test_arm_a_forces_early_stopping_disabled():
    source = (SCRIPTS_DIR / "run_m9_6_train_arm_a.py").read_text()
    assert "early_stopping_patience=0" in source
    assert "config.early_stopping_patience == 0" in source


def test_arm_b_forces_early_stopping_disabled():
    source = (SCRIPTS_DIR / "run_m9_6_train_arm_b.py").read_text()
    assert "early_stopping_patience=0" in source
    assert "config.early_stopping_patience == 0" in source


def test_no_resume_from_historical_checkpoints():
    for path in ("run_m9_6_train_arm_a.py", "run_m9_6_train_arm_b.py"):
        source = (SCRIPTS_DIR / path).read_text()
        assert "resume_from" not in source
        assert "m9-0a-runs" not in source
        assert "m8-7-runs" not in source


def test_no_hyperparameter_or_architecture_search():
    for path in ("run_m9_6_train_arm_a.py", "run_m9_6_train_arm_b.py"):
        source = (SCRIPTS_DIR / path).read_text()
        for forbidden in ("Optuna", "optuna", "lr_sweep", "learning_rate=0.0", "GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE", "Mamba"):
            assert forbidden not in source, f"{path}: forbidden search/architecture-change token found: {forbidden!r}"


def test_arm_a_single_family_arm_b_three_family():
    source_a = (SCRIPTS_DIR / "run_m9_6_train_arm_a.py").read_text()
    source_b = (SCRIPTS_DIR / "run_m9_6_train_arm_b.py").read_text()
    assert "ARM_DEFINITIONS[\"AGE_FIX_ONLY\"]" in source_a
    assert "FAMILY_NAMES" in source_b and "_build_family_pools" in source_b


def test_canonical_checkpoint_is_final_step_not_best_validation():
    for path in ("run_m9_6_train_arm_a.py", "run_m9_6_train_arm_b.py"):
        source = (SCRIPTS_DIR / path).read_text()
        assert "model-export-final-step.safetensors" in source
        assert "load_checkpoint(final_checkpoint_dir" in source


def test_training_scripts_never_open_locked_test():
    for path in ("run_m9_6_train_arm_a.py", "run_m9_6_train_arm_b.py"):
        source = (SCRIPTS_DIR / path).read_text()
        assert "locked_final_test" not in source
        assert "locked_topology_test" not in source
        assert "assert_locked_test_closed" in source


def test_training_scripts_assert_exact_epoch_and_step_counts():
    source_a = (SCRIPTS_DIR / "run_m9_6_train_arm_a.py").read_text()
    source_b = (SCRIPTS_DIR / "run_m9_6_train_arm_b.py").read_text()
    assert "summary.epochs_completed == config.epochs" in source_a
    assert "summary.stopped_early is False" in source_a
    assert "epochs_completed == config.epochs" in source_b
    assert "stopped_early is False" in source_b
    assert "global_step == m6.TOTAL_OPTIMIZER_STEPS" in source_b


# ---------------------------------------------------------------------------
# Protocol-freeze artifact (must exist BEFORE training/results).
# ---------------------------------------------------------------------------


@needs_protocol
def test_protocol_freeze_predeclares_arms_parity_seeds_gates_decision():
    protocol = json.loads(m6.M9_6_PROTOCOL_PATH.read_text())
    assert protocol["exact_compute_parity"]["total_optimizer_steps_required"] == 1350
    assert protocol["exact_compute_parity"]["canonical_checkpoint_policy"] == "FINAL_STEP_1350"
    assert protocol["calibration_method_frozen"]["alpha"] == 0.1
    assert set(protocol["arms"].keys()) >= {"ARM_A_M9_6", "ARM_B_M9_6"}
    assert protocol["arms"]["architecture_identical"] is True
    assert "decision_logic" in protocol
    assert protocol["not_an_architecture_search"] is True


@needs_protocol
def test_protocol_lists_all_closed_architecture_axes():
    protocol = json.loads(m6.M9_6_PROTOCOL_PATH.read_text())
    closed = protocol["closed_axes"]
    for axis in ("GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE", "Mamba", "HydroCore-M"):
        assert axis in closed


# ---------------------------------------------------------------------------
# Training-run artifact contract tests (require training to have completed).
# ---------------------------------------------------------------------------


@needs_training
def test_all_six_training_runs_hit_exactly_1350_optimizer_steps():
    for arm in ("ARM_A_M9_6", "ARM_B_M9_6"):
        for seed in m6.SEEDS:
            record = json.loads((m6.M9_6_TRAINING_RUNS_DIR / f"{arm}-seed{seed}.json").read_text())
            assert record["actual_total_optimizer_steps"] == 1350, f"{arm} seed{seed}: {record['actual_total_optimizer_steps']} != 1350"
            assert record["matches_required_total_optimizer_steps"] is True
            assert record["stopped_early"] is False
            assert record["epochs_completed"] == 20


@needs_training
def test_all_six_canonical_checkpoints_are_final_step():
    for arm in ("ARM_A_M9_6", "ARM_B_M9_6"):
        for seed in m6.SEEDS:
            record = json.loads((m6.M9_6_TRAINING_RUNS_DIR / f"{arm}-seed{seed}.json").read_text())
            assert record["canonical_global_step"] == 1350
            assert record["canonical_epoch"] == 19
            assert record["canonical_checkpoint_policy"] == "FINAL_STEP_1350"


@needs_training
def test_both_arms_identical_parameter_count():
    counts = set()
    for arm in ("ARM_A_M9_6", "ARM_B_M9_6"):
        for seed in m6.SEEDS:
            record = json.loads((m6.M9_6_TRAINING_RUNS_DIR / f"{arm}-seed{seed}.json").read_text())
            counts.add(record["model_architecture"]["param_count"])
    assert len(counts) == 1, f"param counts differ across arms/seeds: {counts}"


@needs_training
def test_arm_b_family_exposure_equal_weighted_and_matches_arm_a_total_volume():
    for seed in m6.SEEDS:
        record = json.loads((m6.M9_6_TRAINING_RUNS_DIR / f"ARM_B_M9_6-seed{seed}.json").read_text())
        assert record["train_scenario_count_per_family"] == 200
        assert record["total_train_scenario_count"] == 600
        assert set(record["family_weighting"].keys()) == set(m6.TRAINED_FAMILIES)
        for weight in record["family_weighting"].values():
            assert weight == pytest.approx(1.0 / 3)


@needs_training
def test_arm_a_and_arm_b_checkpoints_are_distinct_files():
    shas = set()
    for arm in ("ARM_A_M9_6", "ARM_B_M9_6"):
        for seed in m6.SEEDS:
            record = json.loads((m6.M9_6_TRAINING_RUNS_DIR / f"{arm}-seed{seed}.json").read_text())
            shas.add(record["canonical_export_sha256"])
    assert len(shas) == 6


# ---------------------------------------------------------------------------
# Paired-bootstrap / macro-family aggregation correctness (pure functions,
# testable without the full pipeline having run).
# ---------------------------------------------------------------------------


def test_decide_module_paired_bootstrap_resamples_incidents_not_depth_rows():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_6_decide as decide  # noqa: E402

    rows_a = [
        {"family": "coastal-branch", "source_node": "J1", "generator_seed": 1, "depth_bucket": "MATURE", "metrics_neural": {"top1": 1.0}},
        {"family": "coastal-branch", "source_node": "J1", "generator_seed": 1, "depth_bucket": "MATURE", "metrics_neural": {"top1": 0.0}},
    ]
    rows_b = [
        {"family": "coastal-branch", "source_node": "J1", "generator_seed": 1, "depth_bucket": "MATURE", "metrics_neural": {"top1": 1.0}},
        {"family": "coastal-branch", "source_node": "J1", "generator_seed": 1, "depth_bucket": "MATURE", "metrics_neural": {"top1": 1.0}},
    ]
    incidents_a = decide._group_by_incident(rows_a, bucket="MATURE")
    incidents_b = decide._group_by_incident(rows_b, bucket="MATURE")
    assert len(incidents_a) == 1  # ONE physical incident, both depth rows attached
    assert len(incidents_b) == 1


def test_decide_module_bootstrap_deterministic_given_fixed_seed():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_6_decide as decide  # noqa: E402

    def make_rows(top1_values):
        return [
            {"family": "coastal-branch", "source_node": f"J{i}", "generator_seed": i, "depth_bucket": "MATURE", "metrics_neural": {"top1": v}}
            for i, v in enumerate(top1_values)
        ]

    rows_a = make_rows([0.0, 1.0, 0.0, 1.0, 0.0])
    rows_b = make_rows([1.0, 1.0, 1.0, 1.0, 1.0])
    d1 = decide._paired_bootstrap_family_mature_delta(rows_a, rows_b, resamples=100, seed=m6.BOOTSTRAP_SEED)
    d2 = decide._paired_bootstrap_family_mature_delta(rows_a, rows_b, resamples=100, seed=m6.BOOTSTRAP_SEED)
    assert d1 == d2


def test_decide_module_never_opens_locked_test():
    source = (SCRIPTS_DIR / "run_m9_6_decide.py").read_text()
    assert "locked_final_test" not in source
    assert "locked_topology_test" not in source


def test_no_alternative_calibration_method_introduced():
    for path in ("run_m9_6_calibrate.py", "run_m9_6_decide.py"):
        candidate = SCRIPTS_DIR / path
        if not candidate.exists():
            continue
        source = candidate.read_text()
        for forbidden in ("APS", "RAPS", "TemperatureScal", "IsotonicRegression", "PlattScal", "alpha_sweep", "alpha=0.05", "alpha=0.15"):
            assert forbidden not in source, f"{path}: forbidden calibration-method deviation found: {forbidden!r}"


# ---------------------------------------------------------------------------
# Historical artifact immutability (M9.0a/M9.3/M9.4/M9.5/M9.5R).
# ---------------------------------------------------------------------------


def test_historical_training_and_evaluation_scripts_untouched_in_place():
    import subprocess

    for path in (
        "scripts/hydrocore_v5/run_m7_topology.py",
        "scripts/hydrocore_v5/run_m8_7_arm.py",
        "scripts/hydrocore_v5/run_m9_0_arm_b.py",
        "scripts/hydrocore_v5/run_m9_0a_arm_b2.py",
        "scripts/hydrocore_v5/run_m9_0a_evaluate.py",
        "scripts/hydrocore_v5/run_m9_0a_decide.py",
        "scripts/hydrocore_v5/m9_4_common.py",
        "scripts/hydrocore_v5/run_m9_4_source_representative.py",
        "scripts/hydrocore_v5/run_m9_4_decide.py",
        "scripts/hydrocore_v5/m9_5_common.py",
        "scripts/hydrocore_v5/run_m9_5_source_representative.py",
        "scripts/hydrocore_v5/run_m9_5_decide.py",
        "scripts/hydrocore_v5/m9_5r_common.py",
        "scripts/hydrocore_v5/run_m9_5r_source_representative.py",
        "scripts/hydrocore_v5/run_m9_5r_decide.py",
        "src/hydroswarm/training/trainer.py",
        "src/hydroswarm/training/config.py",
    ):
        result = subprocess.run(["git", "diff", "--stat", "HEAD", "--", path], cwd=ROOT, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "", f"{path} was modified relative to HEAD: {result.stdout}"


def test_m9_5r_closure_still_reads_decision_a_unaltered():
    closure = json.loads(m5r.M9_5R_CLOSURE_PATH.read_text())
    assert closure["M9_5R_DECISION"] == "A"


# ---------------------------------------------------------------------------
# Closure-dependent contract tests.
# ---------------------------------------------------------------------------


@needs_closure
def test_closure_alpha_coverage_floor_and_step_count_never_weakened():
    closure = json.loads(m6.M9_6_CLOSURE_PATH.read_text())
    assert closure["calibration"]["alpha"] == 0.1
    assert closure["calibration"]["coverage_floor"] == 0.85
    assert closure["locked_test_opened_before"] is False
    assert closure["locked_test_opened_after"] is False
    assert closure["training_parity"]["optimizer_steps_required"] == 1350
    assert closure["M9_6_DECISION"] in ("A", "B", "C", "D", "E", "F", "G")


@needs_closure
def test_closure_decision_e_iff_training_parity_failed():
    closure = json.loads(m6.M9_6_CLOSURE_PATH.read_text())
    if closure["M9_6_DECISION"] == "E":
        assert closure["training_parity"]["passed"] is False
    else:
        assert closure["training_parity"]["passed"] is True


@needs_closure
def test_closure_decision_a_requires_all_gates_pass():
    closure = json.loads(m6.M9_6_CLOSURE_PATH.read_text())
    if closure["M9_6_DECISION"] == "A":
        assert closure["training_parity"]["passed"] is True
        assert closure["predictive_generalization"]["passed"] is True
        assert closure["known_family_guardrails"]["passed"] is True
        assert closure["calibration"]["representativeness_passed"] is True
        assert closure["calibration"]["current_control_all_3_pass"] is True
        assert closure["calibration"]["interleaved_all_9_pass"] is True
        assert closure["calibration"]["candidate_set_guard_passed"] is True
        assert closure["hydrocore_s_status"] == "FROZEN"
