import json

import numpy as np

from hydroswarm.data.scenarios import (
    CurriculumStage,
    DatasetSplit,
    ScenarioDatasetWriter,
    ScenarioGenerationConfig,
    SplitPlanner,
    WNTRScenarioGenerator,
    validate_split_integrity,
)
from hydroswarm.simulation.network import build_wntr_network


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
    validate_split_integrity([first.manifest, second.manifest])


def test_leave_one_network_family_out_is_assigned_before_simulation() -> None:
    planner = SplitPlanner(held_out_network_families=("C-Town",))
    assert planner.assign("C-Town", 1) == DatasetSplit.TEST
    assert planner.assign("Net1", 1234) in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}
