import json

import numpy as np
import pytest

from hydroswarm.data.scenarios import (
    CurriculumStage,
    DatasetSplit,
    ScenarioDatasetWriter,
    ScenarioGenerationConfig,
    SplitPlanner,
    WNTRScenarioGenerator,
    load_generated_scenarios,
    validate_split_integrity,
)
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.corpus import fit_signature_library, scenario_to_example


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
