"""Unit tests for physics-informed-localizer-scale-validation's focused
3-arm wrapper (`scripts/hydrocore_v5_experimental/
physics_informed_localizer_scale_validation/run_experiment.py`).

This branch's own harness is a thin wrapper around the completed
`physics_informed_localizer_validation` branch's `run_experiment.py`
(imported, not reimplemented) -- these tests verify the wrapper itself:
that it retargets seeds/results roots without mutating the completed
branch's own arm registry, that `C1_C2` activates exactly
`nearest_sensor_log_concentration` + `hop_magnitude_compatibility` with
`hop_arrival_time_compatibility` (C3) zeroed, that `C2` and `C1_C2` share
an identical architecture/parameter count (ablation is input-only, per the
completed branch's own Phase 5 requirement), that the shared
`_mask_physics_columns` masking helper does not mutate its input, and that
`A_CONTROL`'s default HydroCore construction is unchanged from ordinary
(non-experimental) `HydroCore.from_variant` behavior.

The wrapper module is loaded under a unique module name (not the bare
`run_experiment` the completed branch's own tests also use) so this file
can be run in the same pytest session as
`tests/unit/test_physics_feature_ablation.py` without the two same-named
`run_experiment.py` files colliding in `sys.modules`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "scripts" / "hydrocore_v5_experimental" / "physics_informed_localizer_validation"
SCALE_DIR = REPO_ROOT / "scripts" / "hydrocore_v5_experimental" / "physics_informed_localizer_scale_validation"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "hydrocore_v5_experimental"))
sys.path.insert(0, str(BASE_DIR))

from candidate_conditioned_localizer_v1 import physics_features as physf  # noqa: E402
from hydroswarm.model.core import HydroCore  # noqa: E402


def _load_scale_validation_module():
    """Loads the scale-validation wrapper under a private module name so it
    never collides with the completed branch's own `run_experiment` module
    in `sys.modules` (both files are literally named `run_experiment.py`)."""

    spec = importlib.util.spec_from_file_location(
        "physics_informed_localizer_scale_validation_run_experiment", SCALE_DIR / "run_experiment.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scale = _load_scale_validation_module()


class TestScopeRetargeting:
    def test_fresh_seeds_are_disjoint_from_completed_branch_seeds(self) -> None:
        completed_seeds = {20260814, 20260901, 20260915}
        assert set(scale.SEEDS) == {20260929, 20261013, 20261027}
        assert set(scale.SEEDS).isdisjoint(completed_seeds)

    def test_priority_order_is_exactly_the_three_required_arms(self) -> None:
        assert scale.PRIORITY_ORDER == ("A_CONTROL", "C2", "C1_C2")

    def test_results_and_run_roots_are_a_new_directory_not_the_completed_branch(self) -> None:
        assert scale.RESULTS_ROOT.name == "physics-informed-localizer-scale-validation"
        assert scale.RUN_ROOT.parts[-2] == "physics-informed-localizer-scale-validation"
        assert "physics-informed-localizer-validation" not in scale.RESULTS_ROOT.parts[-1]

    def test_out_of_scope_arms_are_rejected(self) -> None:
        for arm in ("C_FULL", "C3", "B_CANDIDATE_CONDITIONED", "A_CAPACITY_MATCHED", "C1"):
            assert arm not in scale.PRIORITY_ORDER

    def test_retargeting_does_not_mutate_the_completed_branchs_arm_registry(self) -> None:
        # The wrapper only reassigns SEEDS/RUN_ROOT/RESULTS_ROOT on the imported
        # base module; ARMS itself (arm definitions) must be untouched.
        assert scale.base.ARMS["C1_C2"]["physics_columns"] == (
            "nearest_sensor_log_concentration",
            "hop_magnitude_compatibility",
        )
        assert scale.base.ARMS["C2"]["physics_columns"] == ("hop_magnitude_compatibility",)
        assert scale.base.ARMS["C3"]["physics_columns"] == ("hop_arrival_time_compatibility",)


class TestC1C2ActivatesExactlyC1AndC2:
    def test_c1_c2_columns_are_exactly_c1_plus_c2(self) -> None:
        c1_c2_columns = set(scale.base.ARMS["C1_C2"]["physics_columns"])
        assert c1_c2_columns == {"nearest_sensor_log_concentration", "hop_magnitude_compatibility"}

    def test_c3_is_excluded_from_c1_c2(self) -> None:
        c1_c2_columns = set(scale.base.ARMS["C1_C2"]["physics_columns"])
        assert "hop_arrival_time_compatibility" not in c1_c2_columns

    def test_c3_column_is_zeroed_after_masking_for_c1_c2_and_c2(self) -> None:
        c3_index = physf.PHYSICS_FEATURE_COLUMNS.index("hop_arrival_time_compatibility")
        features = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        for arm in ("C2", "C1_C2"):
            masked = scale.base._mask_physics_columns(features, scale.base.ARMS[arm]["physics_columns"])
            assert torch.all(masked[..., c3_index] == 0.0), f"{arm} did not zero C3"

    def test_c1_c2_keeps_c1_and_c2_columns_numerically_unchanged(self) -> None:
        c1_index = physf.PHYSICS_FEATURE_COLUMNS.index("nearest_sensor_log_concentration")
        c2_index = physf.PHYSICS_FEATURE_COLUMNS.index("hop_magnitude_compatibility")
        features = torch.tensor([[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]])
        masked = scale.base._mask_physics_columns(features, scale.base.ARMS["C1_C2"]["physics_columns"])
        assert torch.equal(masked[..., c1_index], features[..., c1_index])
        assert torch.equal(masked[..., c2_index], features[..., c2_index])


class TestC2AndC1C2ShareIdenticalArchitecture:
    def test_identical_model_kwargs(self) -> None:
        assert scale.base.ARMS["C2"]["model_kwargs"] == scale.base.ARMS["C1_C2"]["model_kwargs"]
        assert scale.base.ARMS["C2"]["localizer_mode"] == scale.base.ARMS["C1_C2"]["localizer_mode"] == "candidate_conditioned"

    def test_identical_instantiated_parameter_count(self) -> None:
        c2_model = scale.base.build_model(arm_name="C2", seed=20260929)
        c1_c2_model = scale.base.build_model(arm_name="C1_C2", seed=20260929)
        c2_report = c2_model.parameter_report_dict()
        c1_c2_report = c1_c2_model.parameter_report_dict()
        assert c2_report["total"] == c1_c2_report["total"]

        c2_shapes = [tuple(p.shape) for p in c2_model.parameters()]
        c1_c2_shapes = [tuple(p.shape) for p in c1_c2_model.parameters()]
        assert c2_shapes == c1_c2_shapes


class TestMaskingDoesNotMutateInput:
    def test_c1_c2_masking_leaves_original_tensor_untouched(self) -> None:
        features = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        original = features.clone()
        scale.base._mask_physics_columns(features, scale.base.ARMS["C1_C2"]["physics_columns"])
        assert torch.equal(features, original)

    def test_c2_masking_leaves_original_tensor_untouched(self) -> None:
        features = torch.tensor([[[7.0, 8.0, 9.0]]])
        original = features.clone()
        scale.base._mask_physics_columns(features, scale.base.ARMS["C2"]["physics_columns"])
        assert torch.equal(features, original)


class TestDefaultHydroCoreBehaviorUnchanged:
    def test_a_control_uses_plain_default_localizer_mode(self) -> None:
        assert scale.base.ARMS["A_CONTROL"]["localizer_mode"] == "default"
        assert scale.base.ARMS["A_CONTROL"]["model_kwargs"] == {}
        assert scale.base.ARMS["A_CONTROL"]["physics_columns"] is None

    def test_a_control_model_matches_a_plain_from_variant_call(self) -> None:
        """`build_model(arm_name="A_CONTROL", ...)` must be indistinguishable
        (same parameter count/shapes) from calling `HydroCore.from_variant`
        directly with no experimental kwargs -- the experimental arms opt in,
        they never change the default construction path."""

        experimental = scale.base.build_model(arm_name="A_CONTROL", seed=20260929)
        torch.manual_seed(20260929)
        plain = HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13, event_control_heads=True)
        assert experimental.parameter_report_dict()["total"] == plain.parameter_report_dict()["total"]
        assert [tuple(p.shape) for p in experimental.parameters()] == [tuple(p.shape) for p in plain.parameters()]
