"""M10.2 Scout refit amendment: Level-A execution regression tests.

Reads the ALREADY-PRODUCED artifacts from the real Level-A training/gate run
(`reports/evaluation/hydrocore-v5/m10/m10-2-refit/`) plus exercises the
allowlist/gradient-isolation logic directly against a fresh tiny model (fast,
no real simulation needed for that half).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from tests.historical_artifact_portability import require_historical_artifact

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

import m10_2_refit_protocol as proto  # noqa: E402

from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.gradient_coverage import compute_gradient_coverage  # noqa: E402
from hydroswarm.training.losses import compute_multitask_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
M10_2_REFIT_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m10" / "m10-2-refit"


def _tiny_model_with_scout_and_role_residual() -> HydroCore:
    return HydroCore(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=2, modality_layers=1,
        adapter_dims=(32, 32, 32), dropout=0.0, scout_control_heads=True,
        residual_feature_dim=4, role_feature_dim=8,
    )


def _batch(nodes: int = 4, batch: int = 2) -> dict:
    generator = torch.Generator().manual_seed(31)
    return {
        "node_features": torch.randn(batch, nodes, 3, generator=generator),
        "temporal_features": torch.randn(batch, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(batch, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(batch, nodes, dtype=torch.bool),
        "role_features": torch.zeros(batch, 8),
        "residual_features": torch.zeros(batch, nodes, 4),
    }


def _apply_level_a_allowlist(model: HydroCore) -> None:
    allowlist = set(proto.LEVEL_A_PARAMETER_ALLOWLIST)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowlist)


# --------------------------------------------------------------------------
# 6/7. Parameter allowlist exactness + unrelated parameters receive no
#      gradient under the Level-A allowlist.
# --------------------------------------------------------------------------


def test_level_a_allowlist_is_exact_on_a_real_model() -> None:
    model = _tiny_model_with_scout_and_role_residual()
    all_names = {name for name, _ in model.named_parameters()}
    missing = set(proto.LEVEL_A_PARAMETER_ALLOWLIST) - all_names
    assert not missing, f"frozen allowlist references parameters absent from a real model: {missing}"
    _apply_level_a_allowlist(model)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == set(proto.LEVEL_A_PARAMETER_ALLOWLIST)


def test_unrelated_parameters_receive_no_gradient_under_level_a_allowlist() -> None:
    model = _tiny_model_with_scout_and_role_residual()
    _apply_level_a_allowlist(model)
    model.zero_grad(set_to_none=True)
    output = model(_batch())
    targets = {
        "sample_node": torch.tensor([0, 1]), "sample_node_mask": torch.tensor([True, True]),
        "information_gain": torch.rand(2, 4), "information_gain_mask": torch.ones(2, 4, dtype=torch.bool),
        "candidate_reduction": torch.rand(2, 4), "candidate_reduction_mask": torch.ones(2, 4, dtype=torch.bool),
        "should_continue_sampling": torch.tensor([1.0, 0.0]),
    }
    result = compute_multitask_loss(output, targets, task_weights={
        "sample_node": 1.0, "information_gain": 0.5, "candidate_reduction": 0.5, "should_continue_sampling": 0.5,
    })
    result.total.backward()
    allowlist = set(proto.LEVEL_A_PARAMETER_ALLOWLIST)
    for name, parameter in model.named_parameters():
        if name in allowlist:
            continue
        assert parameter.grad is None or bool(torch.all(parameter.grad == 0.0)), (
            f"frozen (non-allowlisted) parameter {name!r} received a nonzero gradient under Level A"
        )


def test_strategist_and_ood_parameters_remain_frozen_under_level_a_allowlist() -> None:
    model = _tiny_model_with_scout_and_role_residual()
    _apply_level_a_allowlist(model)
    trainable_prefixes = {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}
    forbidden = {
        "action_head", "plan_value_head", "plan_validity_head", "pointer_query", "candidate_plan_encoder",
        "consequence_proxy_heads", "ood_category_head", "ood_head", "next_step_head", "uncertainty_head",
    }
    assert trainable_prefixes.isdisjoint(forbidden)


def test_gradient_coverage_module_reports_disconnected_group_when_allowlist_omits_scout_heads() -> None:
    """Sanity cross-check: if the allowlist were (incorrectly) missing a
    Scout head, gradient_coverage would catch it -- proves the allowlist
    check and the gradient-coverage machinery are actually load-bearing
    together, not merely both present."""

    model = _tiny_model_with_scout_and_role_residual()
    weights = {"sample_node": 1.0}
    targets = {"sample_node": torch.tensor([0, 1]), "sample_node_mask": torch.tensor([True, True])}
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), targets, task_weights=weights,
        parameter_groups={"sample_node": []},  # incorrectly empty, as if omitted from an allowlist
        verify_parameter_update=False,
    )
    assert certs["sample_node"].passed is False
    assert any("no trainable parameter group" in reason for reason in certs["sample_node"].failure_reasons)


# --------------------------------------------------------------------------
# 18/19. Checkpoint parent-hash verification + teacher weights unchanged
#        (against the REAL, already-produced Level-A execution artifacts).
# --------------------------------------------------------------------------


def _require_refit_artifacts() -> None:
    if not (M10_2_REFIT_DIR / "m10-2-refit-closure.json").exists():
        pytest.skip("Level-A refit artifacts not present in this checkout")


def test_refit_checkpoint_identities_record_correct_parent_teacher_hash() -> None:
    _require_refit_artifacts()
    for seed, teacher_sha in proto.TEACHER_CHECKPOINT_SHA256.items():
        identity_path = M10_2_REFIT_DIR / "checkpoints" / f"level-a-seed{seed}" / "checkpoint_identity.json"
        identity = json.loads(identity_path.read_text())
        assert identity["parent_m9_6_checkpoint_sha256"] == teacher_sha
        assert identity["refit_level"] == "A"
        assert identity["never_call_this_m9_6"] is True
        assert set(identity["trainable_parameter_allowlist"]) == set(proto.LEVEL_A_PARAMETER_ALLOWLIST)


def test_refit_checkpoint_model_sha256_matches_recorded_identity() -> None:
    _require_refit_artifacts()
    import hashlib

    for seed in proto.TEACHER_CHECKPOINT_SHA256:
        checkpoint_dir = M10_2_REFIT_DIR / "checkpoints" / f"level-a-seed{seed}"
        identity = json.loads((checkpoint_dir / "checkpoint_identity.json").read_text())
        model_path = require_historical_artifact(
            checkpoint_dir / "model.safetensors", identity["model_sha256"], repo_root=ROOT,
        )
        observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert observed == identity["model_sha256"]


def test_closure_confirms_teacher_checkpoints_unchanged() -> None:
    _require_refit_artifacts()
    closure = json.loads((M10_2_REFIT_DIR / "m10-2-refit-closure.json").read_text())
    assert all(closure["teacher_checkpoints_unchanged"].values())
    assert closure["teacher_checkpoint_sha256"] == {str(k): v for k, v in proto.TEACHER_CHECKPOINT_SHA256.items()}
    refit_hashes = closure["refit_checkpoints_per_seed"]
    teacher_hashes = closure["teacher_checkpoint_sha256"]
    for seed in teacher_hashes:
        assert refit_hashes[seed] != teacher_hashes[seed], "refit checkpoint must not be byte-identical to its teacher"


def test_closure_records_a_valid_readiness_state() -> None:
    _require_refit_artifacts()
    closure = json.loads((M10_2_REFIT_DIR / "m10-2-refit-closure.json").read_text())
    assert closure["M10_2_SCOUT_REFIT_RESULT"] in (
        "M10_2_SCOUT_REFIT_A_ACCEPTED", "M10_2_SCOUT_REFIT_B_ACCEPTED", "M10_2_SCOUT_REFIT_BLOCKED_FULL_RETRAIN_REQUIRED",
    )
    assert closure["locked_test_opened_before"] is False
    assert closure["locked_test_opened_after"] is False


def test_no_fabricated_level_b_artifact_when_level_b_was_not_triggered() -> None:
    _require_refit_artifacts()
    closure = json.loads((M10_2_REFIT_DIR / "m10-2-refit-closure.json").read_text())
    if closure["level_b_triggered"] is False:
        assert not (M10_2_REFIT_DIR / "m10-2-refit-level-b.json").exists()


# --------------------------------------------------------------------------
# 1. Supervision-coverage classification (reads the real, already-produced
#    mechanical audit artifact).
# --------------------------------------------------------------------------


def test_supervision_audit_artifact_matches_expected_classification_counts() -> None:
    audit_path = M10_2_REFIT_DIR / "m10-2-refit-supervision-audit.json"
    if not audit_path.exists():
        pytest.skip("supervision audit artifact not present in this checkout")
    audit = json.loads(audit_path.read_text())
    assert audit["classification_counts"] == {
        "TRAINED_WITH_REAL_TARGETS": 9,
        "PRESENT_BUT_UNSUPERVISED": 6,
        "LEGACY_UNGOVERNED": 7,
        "STRUCTURALLY_NOT_EXERCISED": 8,
        "NOT_INSTANTIATED": 3,
    }
    assert audit["cross_checked_against_real_compute_multitask_loss_call"] is True
    for scout_task in ("sample_node", "information_gain", "candidate_reduction", "should_continue_sampling"):
        assert audit["records"][scout_task]["classification"] == "PRESENT_BUT_UNSUPERVISED"
    for sentinel_task in (
        "source_node", "source_region", "start_time", "duration", "relative_strength",
        "event_presence", "event_cause", "sensor_fault", "evidence_sufficiency",
    ):
        assert audit["records"][sentinel_task]["classification"] == "TRAINED_WITH_REAL_TARGETS"
