"""Unit tests for physics-informed-localizer-validation's Phase 4/5 ablation
wiring (`_mask_physics_columns` in `scripts/hydrocore_v5_experimental/
physics_informed_localizer_validation/run_experiment.py`).

The C-family ablation arms (C1/C2/C3 and the pairwise combinations) must
change ONLY which physics-feature columns reach the model, never
`CandidateConditionedLocalizer`'s architecture or parameter count (Phase
5). These tests verify the masking helper itself: it zeroes exactly the
non-selected columns, leaves selected columns numerically unchanged, is a
true no-op for the full column set (C_FULL), and every declared arm's
`physics_columns` in `ARMS` names only real columns from
`physics_features.PHYSICS_FEATURE_COLUMNS`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5_experimental"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5_experimental" / "physics_informed_localizer_validation"))

import run_experiment as experiment  # noqa: E402
from candidate_conditioned_localizer_v1 import physics_features as physf  # noqa: E402


def _sample_features() -> torch.Tensor:
    # [batch=1, nodes=2, 3] all-nonzero so masking is unambiguous.
    return torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])


class TestMaskPhysicsColumns:
    def test_none_is_a_no_op(self) -> None:
        features = _sample_features()
        out = experiment._mask_physics_columns(features, None)
        assert torch.equal(out, features)

    def test_full_column_set_is_a_no_op(self) -> None:
        features = _sample_features()
        out = experiment._mask_physics_columns(features, physf.PHYSICS_FEATURE_COLUMNS)
        assert torch.equal(out, features)

    def test_single_column_zeroes_the_rest(self) -> None:
        features = _sample_features()
        out = experiment._mask_physics_columns(features, ("hop_magnitude_compatibility",))
        expected = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 5.0, 0.0]]])
        assert torch.equal(out, expected)

    def test_pairwise_columns_zero_only_the_third(self) -> None:
        features = _sample_features()
        out = experiment._mask_physics_columns(
            features, ("nearest_sensor_log_concentration", "hop_arrival_time_compatibility")
        )
        expected = torch.tensor([[[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]]])
        assert torch.equal(out, expected)

    def test_does_not_mutate_input(self) -> None:
        features = _sample_features()
        original = features.clone()
        experiment._mask_physics_columns(features, ("nearest_sensor_log_concentration",))
        assert torch.equal(features, original)


class TestArmRegistryConsistency:
    def test_every_c_family_arm_shares_identical_model_kwargs(self) -> None:
        """Phase 5: C_FULL/C1/C2/C3/pairwise must construct the exact same
        CandidateConditionedLocalizer (same physics_feature_dim=3) --
        ablation is input-only."""

        c_family = ("C_FULL", "C1", "C2", "C3", "C1_C2", "C1_C3", "C2_C3")
        reference_kwargs = experiment.ARMS["C_FULL"]["model_kwargs"]
        for arm in c_family:
            assert experiment.ARMS[arm]["model_kwargs"] == reference_kwargs, arm
            assert experiment.ARMS[arm]["localizer_mode"] == "candidate_conditioned"

    def test_every_declared_physics_column_is_real(self) -> None:
        for arm, spec in experiment.ARMS.items():
            columns = spec["physics_columns"]
            if columns is None:
                continue
            for name in columns:
                assert name in physf.PHYSICS_FEATURE_COLUMNS, f"{arm} references unknown column {name!r}"

    def test_single_feature_arms_select_exactly_one_distinct_column(self) -> None:
        c1, c2, c3 = experiment.ARMS["C1"]["physics_columns"], experiment.ARMS["C2"]["physics_columns"], experiment.ARMS["C3"]["physics_columns"]
        assert len(c1) == len(c2) == len(c3) == 1
        assert {c1[0], c2[0], c3[0]} == set(physf.PHYSICS_FEATURE_COLUMNS)

    def test_a_control_and_a_capacity_matched_have_no_candidate_conditioning(self) -> None:
        for arm in ("A_CONTROL", "A_CAPACITY_MATCHED"):
            assert experiment.ARMS[arm]["localizer_mode"] == "default"
            assert experiment.ARMS[arm]["physics_columns"] is None
            assert "localizer_structural_feature_dim" not in experiment.ARMS[arm]["model_kwargs"]

    def test_capacity_matched_is_the_only_arm_with_extra_generic_capacity(self) -> None:
        for arm, spec in experiment.ARMS.items():
            has_capacity = spec["model_kwargs"].get("localizer_capacity_hidden_dim", 0) > 0
            assert has_capacity == (arm == "A_CAPACITY_MATCHED"), arm
