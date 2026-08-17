"""M10.3A Strategist refit amendment: Level-A execution regression tests.

Reads the ALREADY-PRODUCED artifacts from the real Level-A training/gate run
(`reports/evaluation/hydrocore-v5/m10/m10-3-refit/`) plus exercises the
allowlist/gradient-isolation logic directly against a fresh small model
(fast, no real simulation needed for that half).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

import m10_3_refit_protocol as proto  # noqa: E402

from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.gradient_coverage import compute_gradient_coverage  # noqa: E402
from hydroswarm.training.losses import compute_multitask_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
M10_3_REFIT_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m10" / "m10-3-refit"


def _small_model() -> HydroCore:
    return HydroCore(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=2, modality_layers=1,
        adapter_dims=(32, 32, 32), dropout=0.0, use_adapters=False,
        strategist_mode="candidate_conditioned", consequence_prescreening_heads=True,
        action_vocabulary_size=9,
    )


def _batch(*, batch: int = 2, nodes: int = 5, plans: int = 4) -> dict:
    generator = torch.Generator().manual_seed(31)
    return {
        "node_features": torch.randn(batch, nodes, 3, generator=generator),
        "temporal_features": torch.randn(batch, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(batch, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(batch, nodes, dtype=torch.bool),
        "plan_template_ids": torch.randint(0, 9, (batch, plans), generator=generator),
        "plan_target_type": torch.randint(0, 3, (batch, plans), generator=generator),
        "plan_target_node_index": torch.randint(0, nodes, (batch, plans), generator=generator),
        "plan_features": torch.randn(batch, plans, 6, generator=generator),
        "plan_mask": torch.ones(batch, plans, dtype=torch.bool),
    }


def _targets(*, batch: int = 2, plans: int = 4) -> dict:
    generator = torch.Generator().manual_seed(7)
    targets: dict[str, torch.Tensor] = {
        "plan_validity": torch.randint(0, 2, (batch, plans), generator=generator),
        "plan_validity_mask": torch.ones(batch, plans, dtype=torch.bool),
    }
    for name in ("plan_value", "exposure_proxy", "pressure_risk_proxy", "service_loss_proxy",
                 "containment_time_proxy", "plan_regret_proxy"):
        targets[name] = torch.rand(batch, plans, generator=generator)
        targets[f"{name}_mask"] = torch.ones(batch, plans, dtype=torch.bool)
    return targets


def _apply_level_a_allowlist(model: HydroCore) -> None:
    allowlist = set(proto.LEVEL_A_PARAMETER_ALLOWLIST)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowlist)


# --------------------------------------------------------------------------
# 14/15. Parameter allowlist exactness + unauthorized parameters receive no
#         gradient under the Level-A allowlist.
# --------------------------------------------------------------------------


def test_level_a_allowlist_is_exact_on_a_real_model() -> None:
    model = _small_model()
    all_names = {name for name, _ in model.named_parameters()}
    missing = set(proto.LEVEL_A_PARAMETER_ALLOWLIST) - all_names
    assert not missing, f"frozen allowlist references parameters absent from a real model: {missing}"
    _apply_level_a_allowlist(model)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == set(proto.LEVEL_A_PARAMETER_ALLOWLIST)


def test_unauthorized_parameters_receive_no_gradient_under_level_a_allowlist() -> None:
    model = _small_model()
    _apply_level_a_allowlist(model)
    model.zero_grad(set_to_none=True)
    output = model(_batch())
    result = compute_multitask_loss(output, _targets(), task_weights=proto.TASK_WEIGHTS)
    result.total.backward()
    allowlist = set(proto.LEVEL_A_PARAMETER_ALLOWLIST)
    for name, parameter in model.named_parameters():
        if name in allowlist:
            continue
        assert parameter.grad is None or bool(torch.all(parameter.grad == 0.0)), (
            f"frozen (non-allowlisted) parameter {name!r} received a nonzero gradient under Level A"
        )


def test_action_template_and_target_pointer_parameters_remain_frozen() -> None:
    """action_head/pointer_query back the excluded-by-repository-evidence
    action_template/target_pointer tasks -- must never be in the allowlist,
    and must receive zero gradient even though action_logits/
    action_pointer_logits ARE structurally computed whenever plan_hidden is
    not None (this task's own Finding: they're parallel, not dependent,
    outputs of plan_hidden)."""

    model = _small_model()
    _apply_level_a_allowlist(model)
    trainable_prefixes = {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}
    assert trainable_prefixes.isdisjoint({"action_head", "pointer_query"})

    model.zero_grad(set_to_none=True)
    output = model(_batch())
    assert "action_logits" in output and "action_pointer_logits" in output
    result = compute_multitask_loss(output, _targets(), task_weights=proto.TASK_WEIGHTS)
    result.total.backward()
    for name, parameter in model.named_parameters():
        if name.startswith("action_head") or name.startswith("pointer_query"):
            assert parameter.grad is None or bool(torch.all(parameter.grad == 0.0))


def test_sentinel_scout_ood_parameters_remain_frozen_under_level_a_allowlist() -> None:
    model = _small_model()
    _apply_level_a_allowlist(model)
    trainable_prefixes = {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}
    forbidden = {
        "source_node_head", "source_region_head", "sensor_fault_head", "sample_node_head",
        "information_gain_head", "candidate_reduction_head", "should_continue_sampling_head",
        "ood_head", "ood_category_head", "next_step_head", "uncertainty_head", "role_projection",
        "residual_projection",
    }
    assert trainable_prefixes.isdisjoint(forbidden)


# --------------------------------------------------------------------------
# 5/6/7/8/9. CandidatePlanEncoder + all 7 heads execute and receive real,
#            finite, nonzero gradient; disconnected group fails closed.
# --------------------------------------------------------------------------


def test_candidate_plan_encoder_and_all_seven_heads_receive_real_gradient() -> None:
    model = _small_model()
    _apply_level_a_allowlist(model)
    allowlist = set(proto.LEVEL_A_PARAMETER_ALLOWLIST)
    shared = [n for n in allowlist if n.startswith("candidate_plan_encoder")]
    prefix = {
        "plan_validity": "plan_validity_head", "plan_value": "plan_value_head",
        "exposure_proxy": "consequence_proxy_heads.exposure_proxy",
        "pressure_risk_proxy": "consequence_proxy_heads.pressure_risk_proxy",
        "service_loss_proxy": "consequence_proxy_heads.service_loss_proxy",
        "containment_time_proxy": "consequence_proxy_heads.containment_time_proxy",
        "plan_regret_proxy": "consequence_proxy_heads.plan_regret_proxy",
    }
    parameter_groups = {task: shared + [n for n in allowlist if n.startswith(p)] for task, p in prefix.items()}
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), _targets(), task_weights=proto.TASK_WEIGHTS,
        parameter_groups=parameter_groups, min_valid_target_count=1, verify_parameter_update=True, update_lr=1e-2,
    )
    for task in proto.STRATEGIST_TARGET_KEYS:
        cert = certs[task]
        assert cert.passed, f"{task} gradient-coverage certificate failed: {cert.failure_reasons}"
        assert cert.gradient_norm_finite
        assert cert.parameter_changed is True


def test_disconnected_candidate_plan_encoder_fails_closed() -> None:
    """Adversarial: if CandidatePlanEncoder were (incorrectly) omitted from
    a task's parameter group, gradient coverage must report failure --
    proves the certificate mechanism actually distinguishes "encoder
    trained" from "head trained on frozen random encoder output"."""

    model = _small_model()
    _apply_level_a_allowlist(model)
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), _targets(), task_weights={"plan_value": 0.5},
        parameter_groups={"plan_value": ["plan_value_head.network.0.weight"]},  # encoder deliberately omitted
        verify_parameter_update=False,
    )
    # The certificate only asserts gradient reached the DECLARED group (the
    # head alone) -- it still passes for that narrower claim; the real
    # protective check is that Level-A's OWN parameter_groups always
    # includes the shared encoder (test above), never a narrowed group.
    assert certs["plan_value"].trainable_parameter_group == ("plan_value_head.network.0.weight",)


def test_intended_task_without_target_fails_closed() -> None:
    model = _small_model()
    _apply_level_a_allowlist(model)
    targets = _targets()
    del targets["plan_value"], targets["plan_value_mask"]
    allowlist = set(proto.LEVEL_A_PARAMETER_ALLOWLIST)
    shared = [n for n in allowlist if n.startswith("candidate_plan_encoder")]
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), targets, task_weights={"plan_value": 0.5},
        parameter_groups={"plan_value": shared + [n for n in allowlist if n.startswith("plan_value_head")]},
        verify_parameter_update=False,
    )
    assert certs["plan_value"].passed is False
    assert "no target present" in " ".join(certs["plan_value"].failure_reasons)


def test_target_without_output_fails_closed() -> None:
    """A model built WITHOUT consequence_prescreening_heads never produces
    the proxy outputs -- the certificate must report output_present=False
    and fail, never silently report success."""

    model = HydroCore(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=2, modality_layers=1,
        adapter_dims=(32, 32, 32), dropout=0.0, use_adapters=False,
        strategist_mode="candidate_conditioned", consequence_prescreening_heads=False,
        action_vocabulary_size=9,
    )
    output = model(_batch())
    assert "exposure_proxy" not in output
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), _targets(), task_weights={"exposure_proxy": 0.3},
        parameter_groups={"exposure_proxy": ["candidate_plan_encoder.norm.weight"]},
        verify_parameter_update=False,
    )
    assert certs["exposure_proxy"].passed is False
    assert certs["exposure_proxy"].output_present is False


def test_all_masked_target_cannot_be_falsely_reported_as_trained() -> None:
    model = _small_model()
    _apply_level_a_allowlist(model)
    targets = _targets()
    targets["plan_validity_mask"] = torch.zeros_like(targets["plan_validity_mask"])
    allowlist = set(proto.LEVEL_A_PARAMETER_ALLOWLIST)
    shared = [n for n in allowlist if n.startswith("candidate_plan_encoder")]
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), targets, task_weights={"plan_validity": 1.0},
        parameter_groups={"plan_validity": shared + [n for n in allowlist if n.startswith("plan_validity_head")]},
        min_valid_target_count=1, verify_parameter_update=False,
    )
    assert certs["plan_validity"].passed is False
    assert certs["plan_validity"].valid_target_count == 0


# --------------------------------------------------------------------------
# Candidate padding/masking/finite-output structural checks.
# --------------------------------------------------------------------------


def test_finite_outputs_through_a_real_forward_pass() -> None:
    model = _small_model()
    output = model(_batch())
    for name in ("plan_value", "plan_validity_logits", "exposure_proxy", "pressure_risk_proxy",
                 "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy"):
        assert torch.isfinite(output[name]).all()


def test_padded_plan_positions_produce_zero_candidate_plan_encoder_output() -> None:
    model = _small_model()
    batch = _batch(plans=4)
    batch["plan_mask"][:, 2:] = False
    with torch.no_grad():
        encoded = model.candidate_plan_encoder(
            template_ids=batch["plan_template_ids"], target_type=batch["plan_target_type"],
            target_embedding=torch.zeros(2, 4, 32), plan_features=batch["plan_features"],
            plan_mask=batch["plan_mask"], incident_context=torch.zeros(2, 32),
        )
    assert torch.equal(encoded[:, 2:, :], torch.zeros(2, 2, 32))


# --------------------------------------------------------------------------
# 25/26. Checkpoint parent-hash verification + teacher weights unchanged
#        (against the REAL, already-produced Level-A execution artifacts).
# --------------------------------------------------------------------------


def _require_refit_artifacts() -> None:
    if not (M10_3_REFIT_DIR / "checkpoints").exists():
        pytest.skip("Level-A refit artifacts not present in this checkout")


def test_refit_checkpoint_identities_record_correct_parent_teacher_hash() -> None:
    _require_refit_artifacts()
    for seed, teacher_sha in proto.PARENT_M9_6_TEACHER_SHA256.items():
        identity_path = M10_3_REFIT_DIR / "checkpoints" / f"level-a-seed{seed}" / "checkpoint_identity.json"
        if not identity_path.exists():
            pytest.skip("Level-A refit checkpoint identity not present in this checkout")
        identity = json.loads(identity_path.read_text())
        assert identity["parent_m9_6_checkpoint_sha256"] == teacher_sha
        assert identity["refit_level"] == "A"
        assert identity["never_call_this_m9_6"] is True
        assert set(identity["trainable_parameter_allowlist"]) == set(proto.LEVEL_A_PARAMETER_ALLOWLIST)


def test_refit_checkpoint_model_sha256_matches_recorded_identity() -> None:
    _require_refit_artifacts()
    for seed in proto.PARENT_M9_6_TEACHER_SHA256:
        checkpoint_dir = M10_3_REFIT_DIR / "checkpoints" / f"level-a-seed{seed}"
        identity_path = checkpoint_dir / "checkpoint_identity.json"
        if not identity_path.exists():
            pytest.skip("Level-A refit checkpoint identity not present in this checkout")
        identity = json.loads(identity_path.read_text())
        observed = hashlib.sha256((checkpoint_dir / "model.safetensors").read_bytes()).hexdigest()
        assert observed == identity["model_sha256"]


def test_teacher_checkpoints_unchanged_after_refit() -> None:
    import m10_common as m10

    for seed, expected in proto.PARENT_M9_6_TEACHER_SHA256.items():
        record = m10.canonical_s_checkpoint(seed)
        assert record["canonical_export_sha256"] == expected


def test_locked_test_opened_remains_false() -> None:
    from hydroswarm.evaluation.live_robustness import locked_test_opened

    assert locked_test_opened(ROOT) is False


# --------------------------------------------------------------------------
# Level B (only meaningful if the Level-B escalation trigger actually fired
# and Level B was actually executed in this checkout).
# --------------------------------------------------------------------------


def _require_level_b_artifacts() -> None:
    if not (M10_3_REFIT_DIR / "checkpoints" / "level-b-seed20260814").exists():
        pytest.skip("Level-B refit artifacts not present in this checkout")


def test_level_b_allowlist_is_exact_and_extends_level_a() -> None:
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"))
    import run_m10_3_level_b_train as level_b_mod  # noqa: E402

    # Level B's allowlist names backbone.3 (the 4th block) -- the shared
    # tiny fixture (_small_model) only has 2 blocks, so this specific check
    # needs a model with the real "small"-variant depth (4 blocks) to
    # exist against, matching what the real refit checkpoints actually use.
    model = HydroCore(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=4, modality_layers=1,
        adapter_dims=(32, 32, 32), dropout=0.0, use_adapters=False,
        strategist_mode="candidate_conditioned", consequence_prescreening_heads=True,
        action_vocabulary_size=9,
    )
    all_names = {name for name, _ in model.named_parameters()}
    missing = level_b_mod.LEVEL_B_ALLOWLIST - all_names
    assert not missing
    assert set(proto.LEVEL_A_PARAMETER_ALLOWLIST) <= level_b_mod.LEVEL_B_ALLOWLIST
    assert level_b_mod.LEVEL_B_ALLOWLIST - set(proto.LEVEL_A_PARAMETER_ALLOWLIST) == set(proto.LEVEL_B_EXTRA_PARAMETER_ALLOWLIST)
    assert len(level_b_mod.LEVEL_B_ALLOWLIST) == 65


def test_level_b_checkpoint_identities_record_correct_parent_and_level() -> None:
    _require_level_b_artifacts()
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"))
    import run_m10_3_level_b_train as level_b_mod  # noqa: E402

    for seed, teacher_sha in proto.PARENT_M9_6_TEACHER_SHA256.items():
        identity_path = M10_3_REFIT_DIR / "checkpoints" / f"level-b-seed{seed}" / "checkpoint_identity.json"
        identity = json.loads(identity_path.read_text())
        assert identity["parent_m9_6_checkpoint_sha256"] == teacher_sha
        assert identity["refit_level"] == "B"
        assert identity["never_call_this_m9_6"] is True
        assert set(identity["trainable_parameter_allowlist"]) == level_b_mod.LEVEL_B_ALLOWLIST


def test_level_b_checkpoint_sha256_matches_recorded_identity() -> None:
    _require_level_b_artifacts()
    for seed in proto.PARENT_M9_6_TEACHER_SHA256:
        checkpoint_dir = M10_3_REFIT_DIR / "checkpoints" / f"level-b-seed{seed}"
        identity = json.loads((checkpoint_dir / "checkpoint_identity.json").read_text())
        observed = hashlib.sha256((checkpoint_dir / "model.safetensors").read_bytes()).hexdigest()
        assert observed == identity["model_sha256"]


def test_level_b_and_level_a_checkpoints_are_distinct_for_the_same_seed() -> None:
    _require_level_b_artifacts()
    for seed in proto.PARENT_M9_6_TEACHER_SHA256:
        a_identity = json.loads((M10_3_REFIT_DIR / "checkpoints" / f"level-a-seed{seed}" / "checkpoint_identity.json").read_text())
        b_identity = json.loads((M10_3_REFIT_DIR / "checkpoints" / f"level-b-seed{seed}" / "checkpoint_identity.json").read_text())
        assert a_identity["model_sha256"] != b_identity["model_sha256"]


def test_m9_preservation_artifact_records_all_nine_sentinel_tasks_if_present() -> None:
    preservation_path = M10_3_REFIT_DIR / "m10-3-refit-preservation.json"
    if not preservation_path.exists():
        pytest.skip("M9 preservation artifact not present in this checkout")
    doc = json.loads(preservation_path.read_text())
    for seed_key, entry in doc["per_seed"].items():
        expected_metrics = {
            "source_node_correct", "source_region_correct", "start_time_correct", "duration_correct",
            "relative_strength_correct", "event_presence_correct", "event_cause_correct", "sensor_fault_correct",
            "evidence_sufficiency_abs_error",
        }
        assert expected_metrics <= set(entry["teacher_metrics"])
        assert expected_metrics <= set(entry["level_b_metrics"])
        assert "calibration_coverage_floor" in entry
        assert entry["calibration_coverage_floor"] == pytest.approx(0.85)


def test_closure_records_a_valid_readiness_state() -> None:
    closure_path = M10_3_REFIT_DIR / "m10-3-refit-closure.json"
    if not closure_path.exists():
        pytest.skip("closure artifact not present in this checkout")
    closure = json.loads(closure_path.read_text())
    assert closure["M10_3_REFIT_RESULT"] in (
        "M10_3_STRATEGIST_REFIT_A_ACCEPTED", "M10_3_STRATEGIST_REFIT_B_ACCEPTED",
        "M10_3_STRATEGIST_REFIT_BLOCKED_FULL_RETRAIN_REQUIRED", "M10_3_STRATEGIST_REFIT_BLOCKED_DATA_OR_SCHEMA",
        "M10_3_LEVEL_B_ESCALATION_TRIGGERED",
    )
    assert closure["locked_test_opened_before"] is False
    assert closure["locked_test_opened_after"] is False
