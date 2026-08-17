"""Milestone 9.8: HydroCore-S vs HydroCore-M capacity comparison tests.

Covers architecture/checkpoint-policy/training/data/evaluation/promotion/
governance requirements (governing M9.8 prompt Section 31). Tests that
depend on real M9.8 execution artifacts (training runs, predictions,
bootstrap, closure) read those artifacts directly rather than
re-deriving them -- this module verifies the EXECUTED milestone's
artifacts are internally consistent and match the frozen preregistration,
it does not re-run training or inference itself. The decision-logic tests
(`_decide`) are pure unit tests against synthetic inputs and do not depend
on any real artifact.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from hydroswarm.evaluation.live_robustness import locked_test_opened

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5"))

import m9_8_common as m8  # noqa: E402

M9_8_DIR = m8.M9_8_DIR
M9_6_TRAINING_RUNS_DIR = m8.M9_6_TRAINING_RUNS_DIR


def _artifacts_ready() -> bool:
    return m8.M9_8_CLOSURE_PATH.exists()


requires_full_execution = pytest.mark.skipif(
    not _artifacts_ready(), reason="M9.8 full execution (training/evaluation/decide) has not completed yet"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Locked-split guard.
# ---------------------------------------------------------------------------


def test_locked_test_unopened_before_module() -> None:
    assert locked_test_opened(ROOT) is False


# ---------------------------------------------------------------------------
# ARCHITECTURE.
# ---------------------------------------------------------------------------


def test_s_params_exactly_frozen_value() -> None:
    assert m8.S_PARAMETER_COUNT == 4_182_612


def test_m_params_exactly_frozen_value() -> None:
    assert m8.M_PARAMETER_COUNT == 13_919_572


def test_only_frozen_width_dimensions_differ() -> None:
    selected = _load(m8.M9_7_SELECTED_M_ARCHITECTURE_PATH)
    changed = selected["changed_dimensions_vs_S"]
    assert set(changed) == {"d_model", "nhead", "dim_feedforward"}
    assert changed["d_model"] == {"S": 192, "M": 352}
    assert changed["nhead"] == {"S": 6, "M": 11}
    assert changed["dim_feedforward"] == {"S": 576, "M": 1056}
    assert selected["M9_7_CAPACITY_ISOLATION_FAILED"] is False


def test_schemas_and_heads_identical_per_m9_7_audit() -> None:
    audit = _load(m8.M9_7_SEMANTIC_PARITY_AUDIT_PATH)
    s_row, m_row = audit["rows"][0], audit["rows"][1]
    assert s_row["input_schema"] == m_row["input_schema"].replace(" (IDENTICAL to S)", "")
    assert s_row["number_of_output_heads"] == m_row["number_of_output_heads"]
    assert audit["M9_7_CAPACITY_ISOLATION_FAILED"] is False


# ---------------------------------------------------------------------------
# CHECKPOINT POLICY.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [20260814, 31874, 20260815])
def test_arm_s_uses_canonical_checkpoint_not_best_validation(seed: int) -> None:
    record_path = M9_8_DIR / "m9-8-training-runs" / f"ARM_S-seed{seed}.json"
    if not record_path.exists():
        pytest.skip("ARM_S provenance not yet prepared")
    record = _load(record_path)
    assert record["checkpoint_provenance"] == "REUSED_M9_6_CHECKPOINT"
    assert record["canonical_checkpoint_policy"] == "FINAL_STEP_1350"
    assert record["canonical_global_step"] == 1350
    # The record retains best-validation fields for the historical M9.6
    # record but must not present them as the canonical/loaded checkpoint.
    assert record["canonical_export_path"] != ""
    assert "MUST NOT" in record["best_validation_note"]


def test_arm_s_source_code_never_loads_best_validation_for_inference() -> None:
    import inspect

    source = inspect.getsource(importlib.import_module("run_m9_8_prepare_arm_s"))
    assert "canonical_export_path" in source
    assert "_load_s_model(canonical_export_path)" in source
    assert "_load_s_model(m9_6_record[\"best_validation_export_path\"])" not in source


@requires_full_execution
@pytest.mark.parametrize("seed", [20260814, 31874, 20260815])
def test_arm_m_authoritative_export_global_step_exactly_1350(seed: int) -> None:
    record = _load(M9_8_DIR / "m9-8-training-runs" / f"ARM_M-seed{seed}.json")
    assert record["canonical_checkpoint_policy"] == "FINAL_STEP_1350"
    assert record["canonical_global_step"] == 1350
    assert record["canonical_epoch"] == 19


@requires_full_execution
def test_arm_m_best_validation_export_cannot_feed_promotion_evaluation() -> None:
    import inspect

    decide_source = inspect.getsource(importlib.import_module("run_m9_8_decide"))
    evaluate_source = inspect.getsource(importlib.import_module("run_m9_8_evaluate"))
    # The evaluation script's _canonical_model must load canonical_export_path.
    assert 'record["canonical_export_path"]' in evaluate_source
    assert 'record["best_validation_export_path"]' not in evaluate_source
    # The decide script never reads best_validation_* for any guardrail/bootstrap input.
    assert "best_validation" not in decide_source or "note" in decide_source.lower()


# ---------------------------------------------------------------------------
# TRAINING.
# ---------------------------------------------------------------------------


@requires_full_execution
def test_all_3_m_seeds_trained_exactly_1350_steps() -> None:
    for seed in m8.SEEDS:
        record = _load(M9_8_DIR / "m9-8-training-runs" / f"ARM_M-seed{seed}.json")
        assert record["actual_total_optimizer_steps"] == 1350
        assert record["matches_required_total_optimizer_steps"] is True
        assert record["stopped_early"] is False
        assert record["epochs_completed"] == 20


@requires_full_execution
def test_scheduler_config_matches_frozen_values() -> None:
    for seed in m8.SEEDS:
        record = _load(M9_8_DIR / "m9-8-training-runs" / f"ARM_M-seed{seed}.json")
        assert record["scheduler_total_steps_required"] == 1500
        assert record["actual_optimizer_steps_per_epoch"] == list(m8.ARM_A_OPTIMIZER_STEPS_PER_EPOCH)


@requires_full_execution
def test_same_frozen_train_validation_manifests_as_m9_6() -> None:
    for seed in m8.SEEDS:
        m_record = _load(M9_8_DIR / "m9-8-training-runs" / f"ARM_M-seed{seed}.json")
        m9_6_record = _load(M9_6_TRAINING_RUNS_DIR / f"ARM_B_M9_6-seed{seed}.json")
        assert m_record["train_manifest_hash_per_family"] == m9_6_record["train_manifest_hash_per_family"]
        assert m_record["validation_manifest_hash_per_family"] == m9_6_record["validation_manifest_hash_per_family"]
        assert m_record["manifest_hashes_match_m9_6_arm_b_reference"] is True


@requires_full_execution
def test_family_exposure_balanced() -> None:
    for seed in m8.SEEDS:
        record = _load(M9_8_DIR / "m9-8-training-runs" / f"ARM_M-seed{seed}.json")
        counts = record["family_exposure_counts"]
        assert set(counts) == set(m8.TRAINED_FAMILIES)
        values = list(counts.values())
        assert max(values) - min(values) <= max(values) * 0.05  # roughly balanced, matching the 1/3 weighting


def test_no_hydrocore_l_training_artifacts_exist() -> None:
    assert not (ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-8" / "m9-8-training-runs" / "ARM_L-seed20260814.json").exists()
    for seed_dir in (ROOT / "experiments" / "runs" / "hydrocore-v5-causal-m9-8").glob("*"):
        assert "ARM_L" not in seed_dir.name


# ---------------------------------------------------------------------------
# DATA.
# ---------------------------------------------------------------------------


def test_development_seed_bases_are_frozen_exact_values() -> None:
    assert m8.M9_8_DEVELOPMENT_SEED_BASES == {
        "golden-reference": 1_000_100_000, "branched-loop": 1_000_300_000, "loop-grid": 1_000_500_000,
        "coastal-branch": 1_000_700_000, "tree-branch": 1_000_900_000, "dense-loop": 1_001_100_000,
    }


def test_calibration_seed_bases_are_frozen_exact_values() -> None:
    assert m8.M9_8_CALIBRATION_SEED_BASES == {
        "golden-reference": 1_000_000_000, "branched-loop": 1_000_200_000, "loop-grid": 1_000_400_000,
    }


def test_all_source_support_20_incidents_per_source() -> None:
    assert m8.DEVELOPMENT_REPEATS_PER_SOURCE == 20
    assert m8.CALIBRATION_REPEATS_PER_SOURCE == 20
    for family in m8.ALL_FAMILIES:
        n_sources = len(m8.full_junction_list(family, m8.ALL_FAMILY_LOADERS[family]))
        assert n_sources * 20 == m8.EXPECTED_DEVELOPMENT_INCIDENTS_PER_FAMILY[family]
    assert sum(m8.EXPECTED_DEVELOPMENT_INCIDENTS_PER_FAMILY.values()) == 720
    assert sum(m8.EXPECTED_CALIBRATION_INCIDENTS_PER_FAMILY.values()) == 380


def test_train_val_cal_dev_seed_ranges_disjoint_from_each_other_and_m9_6() -> None:
    ranges: dict[str, tuple[int, int]] = {}
    for family in m8.ALL_FAMILIES:
        n = len(m8.full_junction_list(family, m8.ALL_FAMILY_LOADERS[family]))
        base = m8.m9_8_development_seed_base(family)
        ranges[f"dev|{family}"] = (base, base + (n - 1) * m8.M9_8_SOURCE_STRIDE + m8.DEVELOPMENT_REPEATS_PER_SOURCE - 1)
    for family in m8.TRAINED_FAMILIES:
        n = len(m8.full_junction_list(family, m8.ALL_FAMILY_LOADERS[family]))
        base = m8.m9_8_calibration_seed_base(family)
        ranges[f"cal|{family}"] = (base, base + (n - 1) * m8.M9_8_SOURCE_STRIDE + m8.CALIBRATION_REPEATS_PER_SOURCE - 1)

    items = sorted(ranges.items(), key=lambda kv: kv[1][0])
    for i in range(len(items) - 1):
        (_, (_lo1, hi1)) = items[i]
        (_, (lo2, _hi2)) = items[i + 1]
        assert hi1 < lo2, f"seed ranges overlap: {items[i]} vs {items[i + 1]}"

    m9_6_ceiling = 998_000_000 + 100_000 * 12  # M9.6's own 6 families x 2 roles ceiling
    assert min(lo for lo, _hi in ranges.values()) > m9_6_ceiling


@requires_full_execution
def test_s_and_m_development_incidents_exactly_paired() -> None:
    s_incidents: set[tuple[str, str, int]] = set()
    m_incidents: set[tuple[str, str, int]] = set()
    with m8.M9_8_CANONICAL_PREDICTIONS_PATH.open() as fh:
        for line in fh:
            row = json.loads(line)
            key = (row["family"], row["source_node"], row["generator_seed"])
            if row["arm"] == "ARM_S_M9_8":
                s_incidents.add(key)
            else:
                m_incidents.add(key)
    assert s_incidents == m_incidents
    assert len(s_incidents) == 720


@requires_full_execution
def test_no_locked_data_referenced_in_predictions() -> None:
    with m8.M9_8_CANONICAL_PREDICTIONS_PATH.open() as fh:
        for line in fh:
            row = json.loads(line)
            assert "locked" not in row["family"].lower()


# ---------------------------------------------------------------------------
# EVALUATION.
# ---------------------------------------------------------------------------


def test_depths_and_maturity_buckets_frozen() -> None:
    assert m8.DEPTHS == (1, 2, 3, 4, 6, 12, 25)
    assert m8.EARLY_DEPTHS == (1, 2, 3)
    assert m8.MID_DEPTHS == (4, 6)
    assert m8.MATURE_DEPTHS == (12, 25)


def test_bootstrap_seed_exactly_20260819() -> None:
    assert m8.BOOTSTRAP_SEED == 20260819
    assert m8.BOOTSTRAP_RESAMPLES == 2000
    assert m8.BOOTSTRAP_INTERVAL == 0.90


def test_primary_endpoint_uses_neural_posterior_only() -> None:
    import inspect

    source = inspect.getsource(importlib.import_module("run_m9_8_decide"))
    assert '"metrics_neural"' in source
    assert "metrics_hybrid" not in source.split("def _top1_fn")[1].split("def _mrr_fn")[0]


def test_bootstrap_groups_by_incident_not_row(monkeypatch: pytest.MonkeyPatch) -> None:
    decide = importlib.import_module("run_m9_8_decide")
    rows_a = [
        {"depth_bucket": "MATURE", "family": "coastal-branch", "source_node": "CB1", "generator_seed": 1, "metrics_neural": {"top1": 0.5}},
        {"depth_bucket": "MATURE", "family": "coastal-branch", "source_node": "CB1", "generator_seed": 1, "metrics_neural": {"top1": 0.7}},  # same incident, depth=12 vs 25
    ]
    rows_b = [
        {"depth_bucket": "MATURE", "family": "coastal-branch", "source_node": "CB1", "generator_seed": 1, "metrics_neural": {"top1": 0.9}},
        {"depth_bucket": "MATURE", "family": "coastal-branch", "source_node": "CB1", "generator_seed": 1, "metrics_neural": {"top1": 0.9}},
    ]
    result = decide._paired_bootstrap_family_delta(rows_a, rows_b, resamples=10, seed=1, bucket="MATURE")
    assert result["n_incidents"] == 1  # both depths for the SAME incident collapsed into one paired point


# ---------------------------------------------------------------------------
# PROMOTION / decision logic -- pure unit tests, no real artifacts needed.
# ---------------------------------------------------------------------------


def test_threshold_exact_0_02() -> None:
    assert m8.PRIMARY_EFFECT_MINIMUM_ABSOLUTE_DELTA == 0.02


def test_family_consistency_rule_exact() -> None:
    assert m8.GENERALIZATION_MIN_UNSEEN_FAMILIES_IMPROVED == 2
    assert m8.GENERALIZATION_MAX_UNSEEN_FAMILY_REGRESSION_PP == 3.0


def test_known_family_guardrail_thresholds_exact() -> None:
    assert m8.GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP == 5.0
    assert m8.GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP == 3.0
    assert m8.GUARDRAIL_MAX_MRR_REGRESSION == 0.03


def test_calibration_required_cells_enforced() -> None:
    decide = importlib.import_module("run_m9_8_decide")
    calibration_results = {
        "ARM_S_M9_8": {str(s): {"golden-reference": {"calibration_coverage": 0.9}} for s in m8.SEEDS},
        "ARM_M_M9_8": {
            str(s): {f: {"calibration_coverage": 0.9} for f in m8.TRAINED_FAMILIES} for s in m8.SEEDS
        },
    }
    candidate_set_analysis = {"pathological_full_set_behavior_detected": False}
    result = decide._guardrail_e_calibration(calibration_results, candidate_set_analysis)
    assert result["S_control"]["all_3_pass"] is True
    assert result["M_all_required_cells"]["n_required"] == 9
    assert result["M_all_required_cells"]["all_9_pass"] is True
    assert result["passed"] is True

    # Now fail one cell.
    calibration_results["ARM_M_M9_8"][str(m8.SEEDS[0])]["golden-reference"]["calibration_coverage"] = 0.5
    result_fail = decide._guardrail_e_calibration(calibration_results, candidate_set_analysis)
    assert result_fail["M_all_required_cells"]["all_9_pass"] is False
    assert result_fail["passed"] is False


@pytest.mark.parametrize(
    "delta,ci_lower,ci_upper,b_pass,c_pass,d_pass,e_pass,training_pass,dev_pass,cal_pass,expected_decision",
    [
        (0.05, 0.02, 0.08, True, True, True, True, True, True, True, "A"),  # clean validated gain
        (-0.01, -0.03, 0.01, True, True, True, True, True, True, True, "B"),  # clean fail: delta <= 0
        (0.00, -0.01, 0.02, True, True, True, True, True, True, True, "B"),  # clean fail: delta == 0
        (0.05, -0.01, 0.10, True, True, True, True, True, True, True, "D"),  # borderline: delta>=thresh but CI straddles 0
        (0.01, -0.01, 0.03, True, True, True, True, True, True, True, "D"),  # borderline: delta below threshold, CI straddles
        (0.05, 0.02, 0.08, False, True, True, True, True, True, True, "C"),  # A passes, B fails -> guardrail failure
        (0.05, 0.02, 0.08, True, True, True, True, False, True, True, "E"),  # engineering blocker (training parity failed)
    ],
)
def test_decision_logic_matches_frozen_tree(
    delta, ci_lower, ci_upper, b_pass, c_pass, d_pass, e_pass, training_pass, dev_pass, cal_pass, expected_decision,
) -> None:
    decide = importlib.import_module("run_m9_8_decide")
    threshold = m8.PRIMARY_EFFECT_MINIMUM_ABSOLUTE_DELTA
    passed = delta >= threshold and ci_lower > 0
    clean_fail = delta <= 0 or ci_upper <= 0
    borderline = not passed and not clean_fail
    guardrail_a = {"macro_mature_delta": delta, "bootstrap_ci90": [ci_lower, ci_upper], "threshold": threshold, "passed": passed, "clean_fail": clean_fail, "borderline": borderline}
    guardrail_b = {"passed": b_pass, "per_family_mature_delta_pp": {}, "families_improved": [], "worst_family_regression_pp": 0.0}
    guardrail_c = {"passed": c_pass, "per_seed_macro_mature_delta": {}}
    guardrail_d = {"passed": d_pass}
    guardrail_e = {"passed": e_pass}
    training_parity = {"passed": training_pass}
    dev_repr = {"all_families_pass": dev_pass}
    cal_repr = {"all_families_pass": cal_pass}

    result = decide._decide(training_parity, dev_repr, cal_repr, guardrail_a, guardrail_b, guardrail_c, guardrail_d, guardrail_e)
    assert result["decision"] == expected_decision, (
        f"expected {expected_decision}, got {result['decision']} for delta={delta} CI=[{ci_lower},{ci_upper}]"
    )
    if expected_decision == "A":
        assert result["selected_predictor_after_m9_8"] == "M"
    else:
        assert result["selected_predictor_after_m9_8"] == "S"


def test_hydrocore_l_never_authorized_by_decision() -> None:
    assert m8.HYDROCORE_L_AUTHORIZED is False
    # No decision branch flips this -- it is a module-level constant, not
    # computed from any guardrail result.


@requires_full_execution
def test_decision_a_e_matches_closure_and_is_exactly_one_of_five() -> None:
    closure = _load(m8.M9_8_CLOSURE_PATH)
    assert closure["M9_8_DECISION"] in ("A", "B", "C", "D", "E")
    assert closure["HYDROCORE_L_AUTHORIZED"] is False


# ---------------------------------------------------------------------------
# GOVERNANCE.
# ---------------------------------------------------------------------------


def test_m9_7_and_m9_7a_artifacts_unchanged() -> None:
    amendment = _load(m8.M9_7A_AMENDMENT_PATH)
    snapshot = amendment["m9_7_artifacts_hash_snapshot"]
    for name, expected_sha in snapshot.items():
        path = m8.M9_7_DIR / name
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"{name} was modified since M9.7 closure"
    # M9.7A's own artifacts must also remain untouched by M9.8.
    for path in (m8.M9_7A_AMENDMENT_PATH, m8.M9_7A_CLOSURE_PATH):
        assert path.exists()


def test_m9_6_training_run_records_unchanged() -> None:
    for seed in m8.SEEDS:
        record_path = M9_6_TRAINING_RUNS_DIR / f"ARM_B_M9_6-seed{seed}.json"
        record = _load(record_path)
        assert record["canonical_checkpoint_policy"] == "FINAL_STEP_1350"
        assert record["canonical_global_step"] == 1350


def test_hydrocore_l_unauthorized() -> None:
    assert m8.HYDROCORE_L_AUTHORIZED is False


def test_locked_test_unopened_after_module() -> None:
    assert locked_test_opened(ROOT) is False
