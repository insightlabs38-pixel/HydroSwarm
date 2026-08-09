import json

import numpy as np
import pytest

from hydroswarm.data.scenarios import (
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioDatasetWriter,
    ScenarioGenerationConfig,
    SplitPlanner,
    WNTRScenarioGenerator,
    load_generated_scenarios,
    validate_split_integrity,
)
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.corpus import fit_signature_library, scenario_to_example


@pytest.mark.real_simulation
def test_exact_scenario_generation_governance_and_replay(tmp_path) -> None:
    config = ScenarioGenerationConfig(
        seed=1234, network_id="demo-v1", network_family="Net1", stage=CurriculumStage.DEGRADED,
        sensor_count=2, start_time_bins_min=(0,), duration_bins_min=(30,), strength_bins=(1.0,),
        demand_regimes=(1.0,), frozen_probability=1.0, communication_outage_probability=1.0,
    )
    generator = WNTRScenarioGenerator(SplitPlanner(held_out_network_families=("C-Town",)))
    first = generator.generate(build_wntr_network(), config)
    second = generator.generate(build_wntr_network(), config)
    assert first.manifest.replay_sha256 == second.manifest.replay_sha256
    assert np.array_equal(first.truth_concentration, second.truth_concentration)
    assert first.frozen_mask.any()
    assert first.communication_outage_mask.any()
    artifact = ScenarioDatasetWriter(tmp_path / "dataset").write(first)
    assert artifact.exists()
    manifest_line = (tmp_path / "dataset" / "manifests" / f"{first.manifest.split.value}.jsonl").read_text()
    assert json.loads(manifest_line)["network_sha256"] == first.manifest.network_sha256
    loaded = load_generated_scenarios(tmp_path / "dataset", first.manifest.split)
    assert loaded[0].manifest == first.manifest
    assert np.array_equal(loaded[0].truth_concentration, first.truth_concentration)
    validate_split_integrity([first.manifest, second.manifest])


def test_leave_one_network_family_out_is_assigned_before_simulation() -> None:
    planner = SplitPlanner(held_out_network_families=("C-Town",))
    assert planner.assign("C-Town", 1) == DatasetSplit.TEST
    assert planner.assign("Net1", 1234) in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}


#: 19 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
@pytest.mark.real_simulation
def test_deterministic_ids_calibration_split_and_tensor_bridge() -> None:
    network = build_wntr_network()
    generator = WNTRScenarioGenerator()
    scenarios = []
    for index, source in enumerate(sorted(network.junction_name_list)):
        config = ScenarioGenerationConfig(
            seed=91_000 + index * 100,
            network_id="reference",
            network_family="reference",
            split=DatasetSplit.TRAIN,
            stage=CurriculumStage.CLEAN,
            source_node=source,
            sensor_count=3,
            pipe_outage_probability=0.0,
        )
        first = generator.generate(network, config)
        second = generator.generate(network, config)
        assert first.manifest.scenario_id == second.manifest.scenario_id
        assert first.manifest.replay_sha256 == second.manifest.replay_sha256
        scenarios.append(first)

    calibration = generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=99_900,
            network_id="reference",
            network_family="reference",
            split=DatasetSplit.CALIBRATION,
        ),
    )
    assert calibration.manifest.split == DatasetSplit.CALIBRATION

    node_ids = tuple(sorted(network.junction_name_list))
    library = fit_signature_library(scenarios, node_ids)
    example = scenario_to_example(scenarios[0], network, library)
    assert example.inputs["node_features"].shape == (6, 19)
    assert example.inputs["temporal_features"].shape == (25, 6, 6)
    assert example.inputs["edge_features"].shape[-1] == 13
    assert example.inputs["classical_prior"].sum().item() == pytest.approx(1.0)


#: 11 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
@pytest.mark.real_simulation
def test_development_holdout_split_is_distinct_from_locked_test() -> None:
    # overnight-plan.txt Phase 5's Cycle A/B "development holdout" is the
    # governed architecture-comparison iteration surface
    # (configs/evaluation_policy_v3.json's "development_holdout" role),
    # never the once-only locked final test -- generating it under
    # DatasetSplit.TEST would make ordinary iteration indistinguishable
    # from opening the locked test.
    assert DatasetSplit.DEVELOPMENT_HOLDOUT != DatasetSplit.TEST
    assert DatasetSplit.DEVELOPMENT_HOLDOUT.value == "development_holdout"

    network = build_wntr_network()
    generator = WNTRScenarioGenerator()
    scenario = generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=77_000,
            network_id="reference",
            network_family="reference",
            split=DatasetSplit.DEVELOPMENT_HOLDOUT,
            stage=CurriculumStage.CLEAN,
        ),
    )
    assert scenario.manifest.split == DatasetSplit.DEVELOPMENT_HOLDOUT

    train_scenarios = [
        generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=78_000 + index * 100,
                network_id="reference",
                network_family="reference",
                split=DatasetSplit.TRAIN,
                stage=CurriculumStage.CLEAN,
                source_node=source,
            ),
        )
        for index, source in enumerate(sorted(network.junction_name_list))
    ]
    library = fit_signature_library(train_scenarios, tuple(sorted(network.junction_name_list)))
    example = scenario_to_example(scenario, network, library)
    assert example.split == "development_holdout"


@pytest.mark.real_simulation
def test_normal_event_type_produces_negligible_concentration() -> None:
    network = build_wntr_network()
    generator = WNTRScenarioGenerator()
    scenario = generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=500, network_id="ref", network_family="reference", event_type=EventType.NORMAL,
            frozen_probability=0.0, communication_outage_probability=0.0,
        ),
    )
    assert scenario.manifest.event_type == "normal"
    assert scenario.truth_concentration.max() < 1e-4
    assert not scenario.frozen_mask.any()
    assert not scenario.communication_outage_mask.any()


@pytest.mark.real_simulation
def test_sensor_fault_only_event_type_forces_a_fault_with_negligible_concentration() -> None:
    network = build_wntr_network()
    generator = WNTRScenarioGenerator()
    scenario = generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=501, network_id="ref", network_family="reference", event_type=EventType.SENSOR_FAULT_ONLY,
            frozen_probability=0.0, communication_outage_probability=0.0,
        ),
    )
    assert scenario.manifest.event_type == "sensor_fault_only"
    assert scenario.truth_concentration.max() < 1e-4
    assert scenario.frozen_mask.any()


@pytest.mark.real_simulation
def test_drift_and_unit_mismatch_faults_are_recorded_in_their_own_masks() -> None:
    """core-issues.txt repair item 3: drift and unit-mismatch faults were
    injected into observed readings but never recorded in any mask, so
    they were structurally invisible to sensor_fault supervision no matter
    how large they were. Forces both deterministically (a large
    drift_per_hour, unit_mismatch_probability=1.0) rather than relying on
    the small defaults, which can sit right at the quantization boundary."""

    network = build_wntr_network()
    generator = WNTRScenarioGenerator()
    scenario = generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=502, network_id="ref", network_family="reference", event_type=EventType.NORMAL,
            frozen_probability=0.0, communication_outage_probability=0.0,
            drift_per_hour=1.0, unit_mismatch_probability=1.0,
        ),
    )
    assert scenario.drift_mask.any()
    assert scenario.unit_mismatch_mask.any()
    assert scenario.drift_mask.shape == scenario.frozen_mask.shape
    assert scenario.unit_mismatch_mask.shape == scenario.frozen_mask.shape
    # unit_mismatch multiplies the whole column by 1000x from the first
    # timestep, unlike frozen/outage which can start partway through.
    assert scenario.unit_mismatch_mask.all(axis=0).any()


@pytest.mark.real_simulation
def test_negligible_drift_does_not_falsely_mark_every_sensor_faulty() -> None:
    """A near-zero drift rate must not trip drift_mask at all -- otherwise
    every scenario, including deliberately clean ones, would appear to have
    a sensor fault on every node forever."""

    network = build_wntr_network()
    generator = WNTRScenarioGenerator()
    scenario = generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=503, network_id="ref", network_family="reference", event_type=EventType.NORMAL,
            frozen_probability=0.0, communication_outage_probability=0.0,
            drift_per_hour=0.0, unit_mismatch_probability=0.0,
        ),
    )
    assert not scenario.drift_mask.any()
    assert not scenario.unit_mismatch_mask.any()


@pytest.mark.real_simulation
def test_sensor_fault_only_does_not_smuggle_finite_values_into_missing_slots() -> None:
    network = build_wntr_network()
    generator = WNTRScenarioGenerator()
    # High missingness so the forced-fault carry-forward logic must contend
    # with NaN gaps.
    scenario = generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=502, network_id="ref", network_family="reference", event_type=EventType.SENSOR_FAULT_ONLY,
            frozen_probability=0.0, communication_outage_probability=0.0, missing_probability=0.4,
        ),
    )
    observed = scenario.observed_concentration
    mask = scenario.observation_mask
    assert np.all(np.isfinite(observed[mask]))
    assert not np.any(np.isfinite(observed[~mask]))


def test_observed_concentration_never_contains_signed_negative_zero() -> None:
    """np.maximum(0.0, x) is IEEE-754 implementation-defined for x == -0.0 --
    some CPU/SIMD backends preserve the sign bit, some don't. A scenario
    generated on one architecture and replayed on another can therefore hash
    differently (artifact_sha256) despite every value comparing numerically
    equal (-0.0 == 0.0), which broke the deterministic_replay corpus gate
    when replaying an x86-generated corpus on an aarch64 host. Directly feed
    _degrade a truth array containing a literal -0.0 (with all randomized
    degradation disabled, so it survives to the clip step untouched) and
    assert the negative-zero sign bit does not survive."""

    truth = np.array([[-0.0, 1.5, 0.0]], dtype=np.float32)
    timestamps = np.array([0.0])
    config = ScenarioGenerationConfig(
        seed=1, network_id="ref", network_family="reference", stage=CurriculumStage.CLEAN,
        sensor_noise_std=0.0, drift_per_hour=0.0, frozen_probability=0.0,
        communication_outage_probability=0.0, quantization_step=0.0, unit_mismatch_probability=0.0,
        missing_probability=0.0,
    )
    rng = np.random.default_rng(1)
    observed, *_ = WNTRScenarioGenerator._degrade(truth, timestamps, config, rng)
    zero_with_sign_bit = (observed == 0.0) & np.signbit(observed)
    assert not zero_with_sign_bit.any(), observed


@pytest.mark.real_simulation
def test_contamination_event_type_is_unaffected_default() -> None:
    network = build_wntr_network()
    generator = WNTRScenarioGenerator()
    scenario = generator.generate(
        network,
        ScenarioGenerationConfig(seed=503, network_id="ref", network_family="reference"),
    )
    assert scenario.manifest.event_type == "contamination"
    assert scenario.truth_concentration.max() > 1.0  # a real, non-negligible injection


@pytest.mark.real_simulation
def test_event_type_is_part_of_the_replay_hash() -> None:
    network = build_wntr_network()
    generator = WNTRScenarioGenerator()
    normal = generator.generate(
        network,
        ScenarioGenerationConfig(seed=504, network_id="ref", network_family="reference", event_type=EventType.NORMAL),
    )
    contamination = generator.generate(
        network,
        ScenarioGenerationConfig(seed=504, network_id="ref", network_family="reference", event_type=EventType.CONTAMINATION),
    )
    assert normal.manifest.replay_sha256 != contamination.manifest.replay_sha256
    assert normal.manifest.scenario_id != contamination.manifest.scenario_id
