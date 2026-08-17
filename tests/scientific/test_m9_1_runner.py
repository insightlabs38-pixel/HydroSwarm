"""Contract tests for the Milestone 9.1 scientific runner (`scripts/
hydrocore_v5/m9_1_common.py`, `run_m9_1_train_arm.py`, `run_m9_1_evaluate.py`,
`run_m9_1_decide.py`, `run_m9_1_closure.py`).

Frozen protocol: `docs/evaluation/HYDROCORE_V5_M9_1_PROTOCOL.md`, including
its 2026-08-16 Section-21 addendum. These tests cover ONLY the runner's own
governance logic (SHA/lock assertions, the GRAPH_SDE seed-injection
mechanism, the Brownian-seed and bootstrap formulas, guardrail sign
convention, artifact merge safety, final-selection tie-break) -- never
predictive accuracy, never development_holdout/calibration/locked data.
Model-level correctness (permutation-equivariance, causality, solver
correctness) is already covered by `test_m9_1_preflight.py` and is not
re-tested here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT / "src"))

import m9_1_common as common  # noqa: E402
import run_m9_1_closure as closure  # noqa: E402
import run_m9_1_decide as decide  # noqa: E402

torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# Section 6: frozen architecture widths must never drift from the
# preflight-correction artifact this protocol cites verbatim.
# ---------------------------------------------------------------------------


def test_arm_widths_match_preflight_correction_artifact():
    matching = json.loads((ROOT / "reports/evaluation/hydrocore-v5/m9-1-preflight-correction-results.json").read_text())
    corrected = matching["parameter_matching"]["corrected"]
    for arm in ("GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE"):
        assert common.ARM_MLP_WIDTH[arm] == corrected[arm]["mlp_width"]
    assert matching["parameter_matching"]["baseline_total_params"] == common.CURRENT_BASELINE_TOTAL_PARAMS


@pytest.mark.parametrize("arm", ["GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE"])
def test_novel_arm_param_count_within_frozen_bound(arm):
    model = common.build_novel_model(arm)
    total = sum(p.numel() for p in model.parameters())
    delta_pct = (total - common.CURRENT_BASELINE_TOTAL_PARAMS) / common.CURRENT_BASELINE_TOTAL_PARAMS * 100.0
    assert abs(delta_pct) <= 5.0


def test_current_baseline_param_count_matches_m8_7():
    model = common.build_current_model()
    assert sum(p.numel() for p in model.parameters()) == common.CURRENT_BASELINE_TOTAL_PARAMS


def test_ode_max_num_steps_is_frozen_2000():
    dynamics = common.build_dynamics("GRAPH_ODE")
    assert dynamics.max_num_steps == common.ODE_MAX_NUM_STEPS == 2000


# ---------------------------------------------------------------------------
# Section 9: Brownian seed formula and incident-id convention.
# ---------------------------------------------------------------------------


def test_brownian_seed_matches_frozen_formula_verbatim():
    def reference(predictor_training_seed, incident_id, prefix_depth, mc_index):
        key = f"{predictor_training_seed}:{incident_id}:{prefix_depth}:{mc_index}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**31)

    cases = [
        (20260814, 902_000_000, 1, 0),
        (31874, 902_000_100, 25, 3),
        (20260815, 903_000_000, 12, 2),
    ]
    for args in cases:
        assert common.brownian_seed(*args) == reference(*args)


def test_brownian_seed_is_deterministic_and_mc_index_sensitive():
    a = common.brownian_seed(20260814, 902_000_000, 1, 0)
    b = common.brownian_seed(20260814, 902_000_000, 1, 0)
    c = common.brownian_seed(20260814, 902_000_000, 1, 1)
    assert a == b
    assert a != c


def test_incident_id_uses_each_splits_own_generation_seed():
    assert common.incident_id_for("development_holdout", 0) == 902_000_000
    assert common.incident_id_for("development_holdout", 5) == 902_000_500
    assert common.incident_id_for("calibration", 0) == 903_000_000
    assert common.incident_id_for("calibration", 3) == 903_000_300


# ---------------------------------------------------------------------------
# GRAPH_SDE call-time seed injection (never touches continuous_time.py /
# core.py -- verified byte-exact against a direct seeded call).
# ---------------------------------------------------------------------------


def _sde_batch(batch=1, steps=3, nodes=5, edges=6, tfd=6, qfd=4, edge_feature_dim=13, seed=5):
    torch.manual_seed(seed)
    return dict(
        temporal_features=torch.randn(batch, steps, nodes, tfd),
        quality_features=torch.randn(batch, steps, nodes, qfd),
        sensor_mask=torch.ones(batch, steps, nodes, dtype=torch.bool),
        quality_mask=torch.ones(batch, steps, nodes, dtype=torch.bool),
        timestamps=torch.stack([torch.arange(steps).float() * 3600.0 for _ in range(batch)]),
        node_mask=torch.ones(batch, nodes, dtype=torch.bool),
        edge_index=torch.randint(0, nodes, (batch, 2, edges)),
        edge_features=torch.randn(batch, edges, edge_feature_dim),
        edge_mask=torch.ones(batch, edges, dtype=torch.bool),
    )


def test_sde_forward_seed_matches_direct_seeded_call():
    dynamics = common.build_dynamics("GRAPH_SDE").eval()
    batch = _sde_batch()
    args = (
        batch["temporal_features"], batch["quality_features"], batch["sensor_mask"], batch["quality_mask"],
        batch["timestamps"], batch["node_mask"], batch["edge_index"], batch["edge_features"], batch["edge_mask"],
    )
    with torch.no_grad():
        direct = dynamics(*args, seed=555)
    with common.sde_forward_seed(dynamics, 555), torch.no_grad():
        patched = dynamics(*args)
    assert torch.allclose(direct[0], patched[0])
    assert torch.allclose(direct[1], patched[1])


def test_sde_forward_seed_restores_default_behavior_on_exit():
    dynamics = common.build_dynamics("GRAPH_SDE").eval()
    batch = _sde_batch(seed=9)
    args = (
        batch["temporal_features"], batch["quality_features"], batch["sensor_mask"], batch["quality_mask"],
        batch["timestamps"], batch["node_mask"], batch["edge_index"], batch["edge_features"], batch["edge_mask"],
    )
    with torch.no_grad():
        before = dynamics(*args)
    with common.sde_forward_seed(dynamics, 111), torch.no_grad():
        dynamics(*args)
    assert "forward" not in dynamics.__dict__
    with torch.no_grad():
        after = dynamics(*args)
    assert torch.allclose(before[0], after[0])


def test_sde_forward_seed_different_seeds_diverge():
    dynamics = common.build_dynamics("GRAPH_SDE").eval()
    batch = _sde_batch(seed=13)
    args = (
        batch["temporal_features"], batch["quality_features"], batch["sensor_mask"], batch["quality_mask"],
        batch["timestamps"], batch["node_mask"], batch["edge_index"], batch["edge_features"], batch["edge_mask"],
    )
    with common.sde_forward_seed(dynamics, 1), torch.no_grad():
        a = dynamics(*args)
    with common.sde_forward_seed(dynamics, 2), torch.no_grad():
        b = dynamics(*args)
    assert not torch.allclose(a[0], b[0])


# ---------------------------------------------------------------------------
# Section 11: per-row metric block.
# ---------------------------------------------------------------------------


def test_per_row_metrics_top1_and_rank_on_confident_correct_prediction():
    probs = torch.tensor([0.05, 0.9, 0.05])
    metrics = common.per_row_metrics(probs, truth=1)
    assert metrics["top1"] == 1.0
    assert metrics["true_source_rank"] == 1
    assert metrics["nll"] < 0.2


def test_per_row_metrics_top1_zero_and_rank_last_on_wrong_confident_prediction():
    probs = torch.tensor([0.9, 0.05, 0.05])
    metrics = common.per_row_metrics(probs, truth=1)
    assert metrics["top1"] == 0.0
    assert metrics["true_source_rank"] == 2  # one class (index 0) strictly exceeds truth's probability.


def test_primary_metric_restricted_to_early_plus_mid_depths_only():
    rows_by_depth = {
        1: {"metrics": {"top1": 1.0}}, 2: {"metrics": {"top1": 1.0}}, 3: {"metrics": {"top1": 1.0}},
        4: {"metrics": {"top1": 0.0}}, 6: {"metrics": {"top1": 0.0}},
        12: {"metrics": {"top1": 0.0}}, 25: {"metrics": {"top1": 0.0}},  # MATURE: must be excluded.
    }
    # mean of (1,1,1,0,0) over depths (1,2,3,4,6) == 0.6, not diluted by the
    # two MATURE-depth zeros the protocol explicitly excludes (Section 11.1).
    assert common.primary_metric_per_incident(rows_by_depth) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Section 14: bootstrap -- reused resampling loop, determinism, CI math.
# ---------------------------------------------------------------------------


def test_paired_bootstrap_is_deterministic_given_fixed_seed():
    candidate = [0.9, 0.8, 0.95, 0.7, 0.85, 0.6, 0.75, 0.88]
    control = [0.5, 0.4, 0.55, 0.3, 0.45, 0.2, 0.35, 0.48]
    a = common.paired_bootstrap(candidate, control)
    b = common.paired_bootstrap(candidate, control)
    assert a == b


def test_paired_bootstrap_detects_clear_positive_gain():
    candidate = [0.9] * 20
    control = [0.1] * 20
    result = common.paired_bootstrap(candidate, control)
    assert result["ci_entirely_positive"] is True
    assert result["observed_mean_diff"] == pytest.approx(0.8)


def test_paired_bootstrap_no_gain_when_arms_identical():
    values = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.6, 0.4]
    result = common.paired_bootstrap(values, values)
    assert result["ci_entirely_positive"] is False
    assert result["observed_mean_diff"] == 0.0


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        common.paired_bootstrap([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Section 12 Step 1: guardrail sign convention (CURRENT minus candidate;
# positive = candidate worse) via run_m9_1_decide.py's own helper, using
# hand-built results/calibration payload shapes (no real model/data).
# ---------------------------------------------------------------------------


def _fake_results_block(early_top1, mature_top1, overall_mrr, primary_per_incident, all_finite=True, step_limit=False):
    return {
        "aggregates": {
            "early_top1": early_top1, "mature_top1": mature_top1, "overall_mrr": overall_mrr,
            "all_finite": all_finite, "solver_step_limit_exceeded": step_limit,
        },
        "primary_metric_per_incident": primary_per_incident,
    }


def _fake_calibration_block(marginal_coverage=0.9, bucket_coverage=0.9, mean_normalized_set_size=0.2):
    return {
        "marginal": {"coverage": marginal_coverage, "mean_normalized_set_size": mean_normalized_set_size},
        "by_maturity": {b: {"coverage": bucket_coverage} for b in ("EARLY", "MID", "MATURE")},
    }


def test_guardrail_sign_convention_candidate_worse_is_positive_regression(monkeypatch, tmp_path):
    # candidate strictly worse than CURRENT on EARLY top1 -> positive regression.
    results = {
        "CURRENT": {"20260814": _fake_results_block(0.9, 0.9, 0.9, [0.9] * 5), "31874": _fake_results_block(0.9, 0.9, 0.9, [0.9] * 5)},
        "GRAPH_ODE": {"20260814": _fake_results_block(0.7, 0.9, 0.9, [0.7] * 5), "31874": _fake_results_block(0.7, 0.9, 0.9, [0.7] * 5)},
    }
    calibration = {
        "CURRENT": {"20260814": _fake_calibration_block(), "31874": _fake_calibration_block()},
        "GRAPH_ODE": {"20260814": _fake_calibration_block(), "31874": _fake_calibration_block()},
    }
    monkeypatch.setattr(common, "load_run_record", lambda arm, seed: {"instability_flag": None})
    step1 = decide._step1_guardrails("GRAPH_ODE", common.SCREENING_SEEDS, results, calibration)
    assert step1["early_regression_pp"] == pytest.approx(20.0)  # (0.9 - 0.7) * 100, CURRENT minus candidate.
    assert step1["checks"]["early_regression_ok"] is False  # 20pp > 5pp bound.
    assert step1["guardrails_passed"] is False


def test_guardrail_candidate_better_never_penalized_negative_regression_passes(monkeypatch):
    results = {
        "CURRENT": {"20260814": _fake_results_block(0.7, 0.7, 0.7, [0.7] * 5), "31874": _fake_results_block(0.7, 0.7, 0.7, [0.7] * 5)},
        "GRAPH_ODE": {"20260814": _fake_results_block(0.9, 0.9, 0.9, [0.9] * 5), "31874": _fake_results_block(0.9, 0.9, 0.9, [0.9] * 5)},
    }
    calibration = {
        "CURRENT": {"20260814": _fake_calibration_block(), "31874": _fake_calibration_block()},
        "GRAPH_ODE": {"20260814": _fake_calibration_block(), "31874": _fake_calibration_block()},
    }
    monkeypatch.setattr(common, "load_run_record", lambda arm, seed: {"instability_flag": None})
    step1 = decide._step1_guardrails("GRAPH_ODE", common.SCREENING_SEEDS, results, calibration)
    assert step1["early_regression_pp"] < 0  # candidate beats CURRENT -> negative regression.
    assert step1["checks"]["early_regression_ok"] is True
    assert step1["guardrails_passed"] is True


def test_guardrail_coverage_failure_on_either_seed_disqualifies_regardless_of_accuracy(monkeypatch):
    results = {
        "CURRENT": {"20260814": _fake_results_block(0.5, 0.5, 0.5, [0.5] * 5), "31874": _fake_results_block(0.5, 0.5, 0.5, [0.5] * 5)},
        "GRAPH_ODE": {"20260814": _fake_results_block(0.99, 0.99, 0.99, [0.99] * 5), "31874": _fake_results_block(0.99, 0.99, 0.99, [0.99] * 5)},
    }
    calibration = {
        "CURRENT": {"20260814": _fake_calibration_block(), "31874": _fake_calibration_block()},
        # one screening seed's coverage falls below 0.85 -> must disqualify even though accuracy is far better.
        "GRAPH_ODE": {"20260814": _fake_calibration_block(marginal_coverage=0.99), "31874": _fake_calibration_block(marginal_coverage=0.5, bucket_coverage=0.5)},
    }
    monkeypatch.setattr(common, "load_run_record", lambda arm, seed: {"instability_flag": None})
    step1 = decide._step1_guardrails("GRAPH_ODE", common.SCREENING_SEEDS, results, calibration)
    assert step1["checks"]["coverage_ok"] is False
    assert step1["guardrails_passed"] is False


def test_guardrail_instability_flag_disqualifies_regardless_of_other_seed(monkeypatch):
    results = {
        "CURRENT": {"20260814": _fake_results_block(0.5, 0.5, 0.5, [0.5] * 5), "31874": _fake_results_block(0.5, 0.5, 0.5, [0.5] * 5)},
        "GRAPH_SDE": {"20260814": _fake_results_block(0.99, 0.99, 0.99, [0.99] * 5), "31874": _fake_results_block(0.99, 0.99, 0.99, [0.99] * 5)},
    }
    calibration = {
        "CURRENT": {"20260814": _fake_calibration_block(), "31874": _fake_calibration_block()},
        "GRAPH_SDE": {"20260814": _fake_calibration_block(), "31874": _fake_calibration_block()},
    }

    def fake_load(arm, seed):
        return {"instability_flag": "UNSTABLE_ARM_SEED" if seed == 20260814 else None}

    monkeypatch.setattr(common, "load_run_record", fake_load)
    step1 = decide._step1_guardrails("GRAPH_SDE", common.SCREENING_SEEDS, results, calibration)
    assert step1["checks"]["instability_ok"] is False
    assert step1["guardrails_passed"] is False


# ---------------------------------------------------------------------------
# Section 12 final selection: tie-break order among multiple
# PROMOTION_CONFIRMED arms.
# ---------------------------------------------------------------------------


def test_final_selection_single_confirmed_arm_wins_outright():
    confirmed = {"GRAPH_CDE": {"step3": {"observed_mean_diff": 0.01, "ci_lower": 0.001, "ci_upper": 0.02}}}
    winner, _reason = closure._select_winner(confirmed)
    assert winner == "GRAPH_CDE"


def test_final_selection_picks_largest_point_estimate():
    confirmed = {
        "GRAPH_ODE": {"step3": {"observed_mean_diff": 0.03, "ci_lower": 0.01, "ci_upper": 0.05}},
        "GRAPH_SDE": {"step3": {"observed_mean_diff": 0.07, "ci_lower": 0.02, "ci_upper": 0.09}},
    }
    winner, _reason = closure._select_winner(confirmed)
    assert winner == "GRAPH_SDE"


def test_final_selection_tie_break_by_narrower_ci_then_fixed_arm_order():
    # Exactly tied point estimates -> narrower CI wins.
    confirmed = {
        "GRAPH_ODE": {"step3": {"observed_mean_diff": 0.05, "ci_lower": 0.01, "ci_upper": 0.09}},  # width 0.08
        "GRAPH_CDE": {"step3": {"observed_mean_diff": 0.05, "ci_lower": 0.03, "ci_upper": 0.07}},  # width 0.04, narrower.
    }
    winner, _reason = closure._select_winner(confirmed)
    assert winner == "GRAPH_CDE"

    # Exactly tied point estimate AND CI width -> Section 1's fixed listing order (ODE > CDE > SDE).
    confirmed_full_tie = {
        "GRAPH_SDE": {"step3": {"observed_mean_diff": 0.05, "ci_lower": 0.01, "ci_upper": 0.09}},
        "GRAPH_ODE": {"step3": {"observed_mean_diff": 0.05, "ci_lower": 0.01, "ci_upper": 0.09}},
    }
    winner, _reason = closure._select_winner(confirmed_full_tie)
    assert winner == "GRAPH_ODE"


def test_final_selection_zero_confirmed_arms_returns_none():
    winner, reason = closure._select_winner({})
    assert winner is None
    assert "zero" in reason


def test_confirmation_stage_ignores_non_arm_metadata_keys_in_screening_section(monkeypatch, tmp_path):
    # Regression test: m9-1-guardrails.json's "screening" block carries
    # top-level "_locked_test_opened_before"/"_after" boolean keys alongside
    # per-arm dict entries (run_m9_1_decide.py's own main()) -- _confirmation
    # must not choke on iterating those non-dict values when scanning for
    # PROMOTION_CANDIDATE arms.
    guardrails_path = tmp_path / "m9-1-guardrails.json"
    guardrails_path.write_text(json.dumps({
        "screening": {
            "GRAPH_ODE": {"outcome": "GUARDRAILS_FAILED"},
            "GRAPH_CDE": {"outcome": "GUARDRAILS_FAILED"},
            "GRAPH_SDE": {"outcome": "GUARDRAILS_FAILED"},
            "_locked_test_opened_before": False,
            "_locked_test_opened_after": False,
        }
    }))
    monkeypatch.setattr(common, "GUARDRAILS_PATH", guardrails_path)
    result = decide._confirmation({}, {})
    assert "_note" in result  # zero PROMOTION_CANDIDATE arms -> confirmation stage not run for any arm.


# ---------------------------------------------------------------------------
# Section 19(d) / 21(c): SHA and lock-state assertions.
# ---------------------------------------------------------------------------


def test_assert_code_under_test_commit_passes_at_current_head():
    # Smoke test against the real repo -- must not raise (this IS the
    # milestone's own executing commit).
    head = common.assert_code_under_test_commit()
    assert len(head) == 40


def test_frozen_paths_diff_mechanism_detects_a_real_prior_change():
    # 154605180f2a950d86452cfc8ec7202990aba8cf ("implement frozen GRAPH_ODE
    # max_num_steps bound") DID touch continuous_time.py relative to its
    # parent -- regression-guards that assert_code_under_test_commit's
    # underlying git-diff mechanism actually detects frozen-path changes
    # rather than vacuously passing.
    diff = subprocess.run(
        ["git", "diff", "--name-only", "49058beb19cdb4c4ed51fc1afd1e77626c65f3b4",
         "154605180f2a950d86452cfc8ec7202990aba8cf", "--", *common.FROZEN_UNCHANGED_PATHS],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert diff != ""


def test_assert_locked_test_closed_passes_currently():
    assert common.assert_locked_test_closed() is False


def test_code_under_test_commit_floor_v2_audit_is_additive_only():
    # 2026-08-17 Section-21 amendment: CODE_UNDER_TEST_COMMIT_FLOOR was
    # re-superseded from CODE_UNDER_TEST_COMMIT_FLOOR_V1
    # (154605180f2a950d86452cfc8ec7202990aba8cf) to M9.7 commit
    # 475874d8977d0952e8fc3626eb2bd6580cc3c2f7. Mechanically re-prove, on
    # every run, that the ONLY frozen-path change in between is the single
    # additive MODEL_VARIANTS["small_v5_capacity_m"] registration and that
    # it does not touch the existing "small" line -- if a future rebase or
    # history rewrite ever changes what that commit range actually
    # contains, this test (not just the amendment prose) catches it.
    assert common.CODE_UNDER_TEST_COMMIT_FLOOR_V1 == "154605180f2a950d86452cfc8ec7202990aba8cf"
    assert common.CODE_UNDER_TEST_COMMIT_FLOOR == "475874d8977d0952e8fc3626eb2bd6580cc3c2f7"

    touching = subprocess.run(
        ["git", "log", "--format=%H",
         f"{common.CODE_UNDER_TEST_COMMIT_FLOOR_V1}..{common.CODE_UNDER_TEST_COMMIT_FLOOR}",
         "--", *common.FROZEN_UNCHANGED_PATHS],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert touching == [common.CODE_UNDER_TEST_COMMIT_FLOOR]

    diff = subprocess.run(
        ["git", "diff", common.CODE_UNDER_TEST_COMMIT_FLOOR_V1, common.CODE_UNDER_TEST_COMMIT_FLOOR,
         "--", "src/hydroswarm/model/core.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    assert '+    "small_v5_capacity_m": ModelVariant(352, 11, 1056, 4, 64),' in diff
    added = [ln for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    removed = [ln for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    # Pure addition -- matches 475874d's own "17 insertions(+)"; zero removed
    # lines proves the existing "small"/"medium"/"large" entries (which
    # appear only as unchanged context in the unified diff above) were not
    # touched, only a new dict entry appended after them.
    assert len(added) == 17 and len(removed) == 0

    for path in ("src/hydroswarm/model/continuous_time.py", "configs/training-v5-causal.yaml"):
        empty_diff = subprocess.run(
            ["git", "diff", common.CODE_UNDER_TEST_COMMIT_FLOOR_V1, common.CODE_UNDER_TEST_COMMIT_FLOOR, "--", path],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
        assert empty_diff == ""


def test_frozen_small_variant_param_count_unchanged_at_current_head():
    # The M9.7 additive MODEL_VARIANTS["small_v5_capacity_m"] entry must not
    # change "small"'s own resolved configuration or trainable parameter
    # count under M9.1's exact frozen construction recipe.
    from hydroswarm.model.core import HydroCore

    model = HydroCore.from_variant("small", use_adapters=False, **common.SHARED_MODEL_CONFIG, **common.CURRENT_MODEL_KWARGS)
    n = sum(p.numel() for p in model.parameters())
    assert n == common.CURRENT_BASELINE_TOTAL_PARAMS == 4182612


# ---------------------------------------------------------------------------
# Artifact merge safety: never silently drops an unrelated prior entry.
# ---------------------------------------------------------------------------


def test_merge_nested_json_preserves_unrelated_entries(tmp_path):
    path = tmp_path / "artifact.json"
    common.merge_nested_json(path, "CURRENT", "20260814", {"value": 1})
    common.merge_nested_json(path, "GRAPH_ODE", "20260814", {"value": 2})
    common.merge_nested_json(path, "GRAPH_ODE", "31874", {"value": 3})
    data = json.loads(path.read_text())
    assert data["CURRENT"]["20260814"]["value"] == 1
    assert data["GRAPH_ODE"]["20260814"]["value"] == 2
    assert data["GRAPH_ODE"]["31874"]["value"] == 3

    # Re-running the SAME (arm, seed) overwrites only that block.
    common.merge_nested_json(path, "GRAPH_ODE", "20260814", {"value": 99})
    data = json.loads(path.read_text())
    assert data["GRAPH_ODE"]["20260814"]["value"] == 99
    assert data["GRAPH_ODE"]["31874"]["value"] == 3  # untouched.
    assert data["CURRENT"]["20260814"]["value"] == 1  # untouched.
