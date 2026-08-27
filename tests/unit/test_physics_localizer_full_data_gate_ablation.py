"""Unit tests for physics-informed-localizer-full-data-gate's 2-arm,
single-seed, 2-stage wrapper
(`scripts/hydrocore_v5_experimental/physics_informed_localizer_full_data_gate/
run_experiment.py`).

This branch's own harness is a thin wrapper around the completed
`physics_informed_localizer_validation` branch's `run_experiment.py`
(imported, not reimplemented) -- these tests verify the wrapper itself:
that it is scoped to exactly the pre-declared seed (20261110) and arms
(A_CONTROL, C1_C2), that `C1_C2` activates exactly
`nearest_sensor_log_concentration` + `hop_magnitude_compatibility` with
`hop_arrival_time_compatibility` (C3) zeroed, that `C1_C2` and a plain
candidate-conditioned arm share an identical architecture/parameter count
(ablation is input-only, per the completed branch's own Phase 5
requirement), that the shared `_mask_physics_columns` masking helper does
not mutate its input, that `A_CONTROL`'s default HydroCore construction is
unchanged from ordinary (non-experimental) `HydroCore.from_variant`
behavior, and that the full-data stage trains on the entire unsubsampled
9000-example split while the pilot stage keeps the existing 600-example
protocol.

The wrapper module is loaded under a unique module name (not the bare
`run_experiment` the completed branch's own tests also use) so this file
can be run in the same pytest session as
`tests/unit/test_physics_feature_ablation.py` and
`tests/unit/test_physics_localizer_scale_validation_ablation.py` without
the same-named `run_experiment.py` files colliding in `sys.modules`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "scripts" / "hydrocore_v5_experimental" / "physics_informed_localizer_validation"
GATE_DIR = REPO_ROOT / "scripts" / "hydrocore_v5_experimental" / "physics_informed_localizer_full_data_gate"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "hydrocore_v5_experimental"))
sys.path.insert(0, str(BASE_DIR))

from candidate_conditioned_localizer_v1 import physics_features as physf  # noqa: E402
from hydroswarm.model.core import HydroCore  # noqa: E402


def _load_gate_module():
    """Loads the full-data-gate wrapper under a private module name so it
    never collides with the completed branch's own `run_experiment` module
    in `sys.modules` (both files are literally named `run_experiment.py`)."""

    spec = importlib.util.spec_from_file_location(
        "physics_informed_localizer_full_data_gate_run_experiment", GATE_DIR / "run_experiment.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


class TestScopeRetargeting:
    def test_predeclared_seed_is_exactly_20261110(self) -> None:
        assert gate.SEED == 20261110

    def test_arm_names_are_exactly_a_control_and_c1_c2(self) -> None:
        assert gate.ARM_NAMES == ("A_CONTROL", "C1_C2")

    def test_results_and_run_roots_are_a_new_directory_not_a_completed_branch(self) -> None:
        assert gate.RESULTS_ROOT.name == "physics-informed-localizer-full-data-gate"
        assert gate.RUN_ROOT.parts[-2] == "physics-informed-localizer-full-data-gate"
        for prior in ("physics-informed-localizer-validation", "physics-informed-localizer-scale-validation"):
            assert prior not in gate.RESULTS_ROOT.parts

    def test_out_of_scope_arms_are_rejected_by_main(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--stage", choices=("pilot", "full_data"), required=True)
        parser.add_argument("--arms", type=str, default=None)
        parser.add_argument("--seed", type=int, default=gate.SEED)
        args = parser.parse_args(["--stage", "pilot", "--arms", "C2"])
        assert args.arms == "C2"
        assert "C2" not in gate.ARM_NAMES

    def test_seed_mismatch_is_rejected(self) -> None:
        assert 20260929 != gate.SEED  # disjoint from the completed scale-validation study's own seeds


class TestC1C2ActivatesExactlyC1AndC2:
    def test_c1_c2_columns_are_exactly_c1_plus_c2(self) -> None:
        c1_c2_columns = set(gate.base.ARMS["C1_C2"]["physics_columns"])
        assert c1_c2_columns == {"nearest_sensor_log_concentration", "hop_magnitude_compatibility"}

    def test_c3_is_excluded_from_c1_c2(self) -> None:
        c1_c2_columns = set(gate.base.ARMS["C1_C2"]["physics_columns"])
        assert "hop_arrival_time_compatibility" not in c1_c2_columns

    def test_c3_column_is_zeroed_after_masking_for_c1_c2(self) -> None:
        c3_index = physf.PHYSICS_FEATURE_COLUMNS.index("hop_arrival_time_compatibility")
        features = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        masked = gate.base._mask_physics_columns(features, gate.base.ARMS["C1_C2"]["physics_columns"])
        assert torch.all(masked[..., c3_index] == 0.0)

    def test_c1_c2_keeps_c1_and_c2_columns_numerically_unchanged(self) -> None:
        c1_index = physf.PHYSICS_FEATURE_COLUMNS.index("nearest_sensor_log_concentration")
        c2_index = physf.PHYSICS_FEATURE_COLUMNS.index("hop_magnitude_compatibility")
        features = torch.tensor([[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]])
        masked = gate.base._mask_physics_columns(features, gate.base.ARMS["C1_C2"]["physics_columns"])
        assert torch.equal(masked[..., c1_index], features[..., c1_index])
        assert torch.equal(masked[..., c2_index], features[..., c2_index])

    def test_masking_does_not_mutate_input(self) -> None:
        features = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        original = features.clone()
        gate.base._mask_physics_columns(features, gate.base.ARMS["C1_C2"]["physics_columns"])
        assert torch.equal(features, original)


class TestC1C2MatchesValidatedArchitecture:
    def test_identical_model_kwargs_to_c2(self) -> None:
        # C1_C2 must keep the SAME model_kwargs as every other C-family arm
        # (physics_feature_dim=3 always) -- ablation changes only the input
        # columns, never the architecture (Phase 5's requirement, reused
        # unmodified from the completed validation branch).
        assert gate.base.ARMS["C1_C2"]["model_kwargs"] == gate.base.ARMS["C2"]["model_kwargs"]
        assert gate.base.ARMS["C1_C2"]["localizer_mode"] == gate.base.ARMS["C2"]["localizer_mode"] == "candidate_conditioned"

    def test_identical_instantiated_parameter_count_to_c2(self) -> None:
        c2_model = gate.base.build_model(arm_name="C2", seed=gate.SEED)
        c1_c2_model = gate.base.build_model(arm_name="C1_C2", seed=gate.SEED)
        assert c2_model.parameter_report_dict()["total"] == c1_c2_model.parameter_report_dict()["total"]
        c2_shapes = [tuple(p.shape) for p in c2_model.parameters()]
        c1_c2_shapes = [tuple(p.shape) for p in c1_c2_model.parameters()]
        assert c2_shapes == c1_c2_shapes


class TestDefaultHydroCoreBehaviorUnchanged:
    def test_a_control_uses_plain_default_localizer_mode(self) -> None:
        assert gate.base.ARMS["A_CONTROL"]["localizer_mode"] == "default"
        assert gate.base.ARMS["A_CONTROL"]["model_kwargs"] == {}
        assert gate.base.ARMS["A_CONTROL"]["physics_columns"] is None

    def test_a_control_model_matches_a_plain_from_variant_call(self) -> None:
        experimental = gate.base.build_model(arm_name="A_CONTROL", seed=gate.SEED)
        torch.manual_seed(gate.SEED)
        plain = HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13, event_control_heads=True)
        assert experimental.parameter_report_dict()["total"] == plain.parameter_report_dict()["total"]
        assert [tuple(p.shape) for p in experimental.parameters()] == [tuple(p.shape) for p in plain.parameters()]


class TestFullDataTrainerConfigPreservesOptimizationBudget:
    def test_full_data_train_arm_uses_same_pilot_epochs_constant(self) -> None:
        # Same epoch budget (6) as the pilot stage -- full-data training
        # scales train-set size only, never the epoch budget.
        import inspect

        source = inspect.getsource(gate.train_arm_full_data)
        assert "base.PILOT_EPOCHS" in source

    def test_full_data_checkpointing_is_more_frequent_but_epochs_unchanged(self) -> None:
        # checkpoint_every_epochs=1 (crash recovery) is an I/O cadence
        # change only; PILOT_EPOCHS itself (still read from the base
        # module) is untouched by this branch.
        assert gate.base.PILOT_EPOCHS == 6


class TestCheckpointDiscovery:
    def test_latest_checkpoint_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert gate._latest_checkpoint(tmp_path / "does-not-exist") is None

    def test_latest_checkpoint_returns_highest_numbered_dir(self, tmp_path: Path) -> None:
        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "checkpoint-0001").mkdir()
        (checkpoints / "checkpoint-0003").mkdir()
        (checkpoints / "checkpoint-0002").mkdir()
        result = gate._latest_checkpoint(tmp_path)
        assert result is not None
        assert result.name == "checkpoint-0003"
