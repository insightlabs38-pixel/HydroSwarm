"""TRUE Milestone 10.2 evaluation harness: fast, no-real-simulation unit
tests for `scripts/hydrocore_v5/run_m10_2_true_evaluation.py` and
`run_m10_2_true_decide.py`.

Frozen protocol: `docs/evaluation/HYDROCORE_V5_M10_2_TRUE_EVALUATION_PROTOCOL.md`.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
from tests.historical_artifact_portability import require_historical_artifact

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

import m10_2_true_protocol as proto  # noqa: E402
import run_m10_2_true_decide as decide  # noqa: E402
import run_m10_2_true_evaluation as ev  # noqa: E402

from hydroswarm.agents.scout import HydroScout  # noqa: E402
from hydroswarm.agents.schemas import ScoutAction  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.inference.authority import scout_certificate  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Protocol freeze / consistency.
# --------------------------------------------------------------------------


def test_protocol_hash_is_deterministic() -> None:
    assert proto.protocol_hash() == proto.protocol_hash()


def test_eval_seed_range_disjoint_from_refit_train_and_validation_ranges() -> None:
    import m10_2_refit_protocol as refit_proto

    eval_range = range(proto.EVAL_SEED_BASE, proto.EVAL_SEED_BASE + proto.EVAL_COUNT * 100, 100)
    train_range = set(range(refit_proto.TRAIN_SEED_BASE, refit_proto.TRAIN_SEED_BASE + refit_proto.TRAIN_COUNT * 100, 100))
    validation_range = set(
        range(refit_proto.VALIDATION_SEED_BASE, refit_proto.VALIDATION_SEED_BASE + refit_proto.VALIDATION_COUNT_AMENDMENT_1 * 100, 100)
    )
    assert not (set(eval_range) & train_range)
    assert not (set(eval_range) & validation_range)


def test_candidate_gate_k_matches_governed_production_constant() -> None:
    from hydroswarm.simulation.wrapper import MAXIMUM_EVALUATION_HYPOTHESES

    assert proto.CANDIDATE_GATE_K == MAXIMUM_EVALUATION_HYPOTHESES


def test_population_scope_matches_accepted_level_a_scope() -> None:
    import m10_2_refit_protocol as refit_proto

    assert proto.FAMILY == refit_proto.FAMILY
    assert proto.DEPTH == refit_proto.DEPTH
    assert proto.MAXIMUM_SAMPLES == refit_proto.MAXIMUM_SAMPLES
    assert proto.NOISE_SCALE_MG_L == refit_proto.NOISE_SCALE_MG_L


# --------------------------------------------------------------------------
# Checkpoint hash fail-closed behavior.
# --------------------------------------------------------------------------


def test_checkpoint_hash_mismatch_fails_closed(tmp_path) -> None:
    fake_path = tmp_path / "model.safetensors"
    fake_path.write_bytes(b"not the real checkpoint bytes")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ev.verify_and_load_checkpoint(20260814, checkpoint_path=fake_path)


def test_checkpoint_missing_fails_closed_never_substitutes_teacher(tmp_path) -> None:
    missing_path = tmp_path / "does-not-exist" / "model.safetensors"
    with pytest.raises(FileNotFoundError, match="never substituted with the original M9.6 teacher"):
        ev.verify_and_load_checkpoint(20260814, checkpoint_path=missing_path)


def test_teacher_checkpoint_hash_mismatch_fails_closed(monkeypatch) -> None:
    original = ev.m10.canonical_s_checkpoint

    def _tampered(seed: int) -> dict:
        record = original(seed)
        return {**record, "canonical_export_sha256": "0" * 64}

    monkeypatch.setattr(ev.m10, "canonical_s_checkpoint", _tampered)
    with pytest.raises(ValueError, match="Parent M9.6 teacher checkpoint changed"):
        ev.verify_teacher_checkpoints_unchanged()


def test_real_approved_checkpoint_hashes_verify() -> None:
    """The three approved Level-A refit checkpoint hashes in the frozen
    protocol must match the actual, currently-committed local weight files
    -- this is the task's own step 1/2 (locate + verify) exercised as a
    regression test, not just a one-off manual check."""

    for seed, expected in proto.LEVEL_A_REFIT_CHECKPOINT_SHA256.items():
        path = require_historical_artifact(
            ev.m10.M10_DIR / "m10-2-refit" / "checkpoints" / f"level-a-seed{seed}" / "model.safetensors",
            expected,
            repo_root=ROOT,
        )
        model, actual_sha256, path = ev.verify_and_load_checkpoint(seed, checkpoint_path=path)
        assert actual_sha256 == expected
        assert isinstance(model, HydroCore)


# --------------------------------------------------------------------------
# Authority invariance / learned-OOD suppression / no runtime promotion.
# --------------------------------------------------------------------------


def test_scout_certificate_signature_unaffected_by_true_evaluation_module() -> None:
    """`scout_certificate` still accepts only `analysis` -- nothing this
    module defines can be wired into it."""

    parameters = set(inspect.signature(scout_certificate).parameters)
    assert parameters == {"analysis"}


def test_learned_scout_recommendation_is_never_promotable() -> None:
    from hydroswarm.evaluation.scout_state import decode_learned_scout_recommendation

    assert inspect.signature(decode_learned_scout_recommendation).return_annotation is not None
    # promotable is a frozen dataclass field with a fixed default and no setter --
    # structurally checked in tests/unit/test_scout_evaluation_state.py already;
    # here we additionally confirm this module never overrides it.
    assert "promotable" not in inspect.getsource(ev)


def test_this_module_never_imports_ood_or_strategist_training_paths() -> None:
    source = inspect.getsource(ev)
    for forbidden in ("ood_category_head", "strategist", "next_step", "plan_validity", "CandidatePlanEncoder"):
        assert forbidden not in source, f"TRUE M10.2 execution module unexpectedly references {forbidden!r}"


# --------------------------------------------------------------------------
# Promotion-rule mechanics (pure function, no model/simulation needed).
# --------------------------------------------------------------------------


def _make_arm(*, resolved_at_step, final_samples_taken, voluntary_stop, top1_final):
    return {
        "resolved_at_step": resolved_at_step,
        "final_samples_taken": final_samples_taken,
        "voluntary_stop": voluntary_stop,
        "rounds": [{"top1": top1_final}],
    }


def test_actionable_within_budget_true_when_resolved() -> None:
    arm = _make_arm(resolved_at_step=1, final_samples_taken=1, voluntary_stop=False, top1_final=True)
    assert decide._actionable_within_budget(arm) is True


def test_actionable_within_budget_false_when_never_resolved() -> None:
    arm = _make_arm(resolved_at_step=None, final_samples_taken=3, voluntary_stop=False, top1_final=False)
    assert decide._actionable_within_budget(arm) is False


def test_false_stop_requires_deterministic_counterfactual_resolution() -> None:
    arm_l = _make_arm(resolved_at_step=None, final_samples_taken=1, voluntary_stop=True, top1_final=False)
    arm_d_resolved = _make_arm(resolved_at_step=2, final_samples_taken=2, voluntary_stop=True, top1_final=True)
    arm_d_never = _make_arm(resolved_at_step=None, final_samples_taken=3, voluntary_stop=False, top1_final=False)
    assert decide._false_stop(arm_l, arm_d_resolved) is True
    assert decide._false_stop(arm_l, arm_d_never) is False


def test_false_stop_does_not_fire_when_learned_already_resolved() -> None:
    arm_l = _make_arm(resolved_at_step=0, final_samples_taken=0, voluntary_stop=True, top1_final=True)
    arm_d = _make_arm(resolved_at_step=2, final_samples_taken=2, voluntary_stop=True, top1_final=True)
    assert decide._false_stop(arm_l, arm_d) is False


def test_unnecessary_sampling_detects_over_sampling_past_resolution() -> None:
    arm_l = _make_arm(resolved_at_step=0, final_samples_taken=3, voluntary_stop=False, top1_final=True)
    assert decide._unnecessary_sampling(arm_l) is True
    arm_l_efficient = _make_arm(resolved_at_step=2, final_samples_taken=2, voluntary_stop=True, top1_final=True)
    assert decide._unnecessary_sampling(arm_l_efficient) is False


def test_promotion_rule_blocks_on_any_hard_gate_failure() -> None:
    import m10_common as m10

    safety_audit = {"all_hard_gates_passed": False}
    per_seed = {
        seed: {
            "actionable_within_budget": {"diff_point_estimate": 1.0, "diff_ci_lower": 0.5},
            "never_actionable_fraction": {"diff_ci_lower": -0.1},
            "source_top1_final_round": {"diff_ci_upper": 0.5},
        }
        for seed in m10.SEEDS
    }
    decision = decide.apply_promotion_rule(safety_audit, per_seed)
    assert decision["result"] == "M10_2_SCIENTIFIC_EVALUATION_BLOCKED"


def test_promotion_rule_requires_consistency_across_seeds() -> None:
    import m10_common as m10

    safety_audit = {"all_hard_gates_passed": True}
    # Only ONE seed shows a positive/CI-excluding-zero improvement -- must NOT promote.
    per_seed = {
        m10.SEEDS[0]: {
            "actionable_within_budget": {"diff_point_estimate": 0.2, "diff_ci_lower": 0.05},
            "never_actionable_fraction": {"diff_ci_lower": -0.1},
            "source_top1_final_round": {"diff_ci_upper": 0.5},
        },
        m10.SEEDS[1]: {
            "actionable_within_budget": {"diff_point_estimate": -0.05, "diff_ci_lower": -0.2},
            "never_actionable_fraction": {"diff_ci_lower": -0.1},
            "source_top1_final_round": {"diff_ci_upper": 0.5},
        },
        m10.SEEDS[2]: {
            "actionable_within_budget": {"diff_point_estimate": -0.02, "diff_ci_lower": -0.15},
            "never_actionable_fraction": {"diff_ci_lower": -0.1},
            "source_top1_final_round": {"diff_ci_upper": 0.5},
        },
    }
    decision = decide.apply_promotion_rule(safety_audit, per_seed)
    assert decision["result"] == "M10_2_LEARNED_SCOUT_NOT_PROMOTED_DETERMINISTIC_RETAINED"


def test_promotion_rule_blocks_on_ci_confident_regression_even_if_primary_metric_wins() -> None:
    import m10_common as m10

    safety_audit = {"all_hard_gates_passed": True}
    per_seed = {
        seed: {
            "actionable_within_budget": {"diff_point_estimate": 0.3, "diff_ci_lower": 0.1},
            "never_actionable_fraction": {"diff_ci_lower": 0.05},  # CI-confident regression (worse).
            "source_top1_final_round": {"diff_ci_upper": 0.5},
        }
        for seed in m10.SEEDS
    }
    decision = decide.apply_promotion_rule(safety_audit, per_seed)
    assert decision["result"] == "M10_2_LEARNED_SCOUT_NOT_PROMOTED_DETERMINISTIC_RETAINED"
    assert decision["criterion_3_no_regression_passed"] is False


def test_promotion_rule_supports_when_all_criteria_pass() -> None:
    import m10_common as m10

    safety_audit = {"all_hard_gates_passed": True}
    per_seed = {
        seed: {
            "actionable_within_budget": {"diff_point_estimate": 0.3, "diff_ci_lower": 0.1},
            "never_actionable_fraction": {"diff_ci_lower": -0.1},
            "source_top1_final_round": {"diff_ci_upper": 0.5},
        }
        for seed in m10.SEEDS
    }
    decision = decide.apply_promotion_rule(safety_audit, per_seed)
    assert decision["result"] == "M10_2_LEARNED_SCOUT_PROMOTION_SUPPORTED"


# --------------------------------------------------------------------------
# STOP semantics / deterministic-policy sanity (no real simulation needed).
# --------------------------------------------------------------------------


def test_hydroscout_deterministic_fallback_stops_when_no_pool() -> None:
    output = HydroScout().deterministic_fallback({
        "sampling_history": ("A", "B"), "candidate_probabilities": {}, "candidate_region": ("A", "B"),
        "node_ids": ("A", "B"),
    })
    assert output.action == ScoutAction.STOP


def test_hydroscout_deterministic_fallback_never_selects_already_sampled() -> None:
    output = HydroScout().deterministic_fallback({
        "sampling_history": ("A",), "candidate_probabilities": {"A": 0.9, "B": 0.1},
        "candidate_region": ("A", "B"), "node_ids": ("A", "B"),
    })
    assert output.node_id == "B"


def test_locked_test_opened_remains_false() -> None:
    assert locked_test_opened(ROOT) is False
