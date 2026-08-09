"""Bundle F Cycle B needs 3 training topologies + 1 development-OOD topology;
only golden-reference and branched-loop existed before this. These two new
independent EPANET networks -- loop-grid.inp (a 2x4 multi-loop grid, a
training topology) and coastal-branch.inp (a longer branch-with-loop
network including both a tank and a spur, held out entirely for
development OOD) -- must load, hydraulically validate, and flow through
the real corpus pipeline exactly like the two topologies Cycle A already
proved out.
"""

from __future__ import annotations

import pytest
import wntr

from hydroswarm.data.scenarios import CurriculumStage, DatasetSplit, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.simulation import HydraulicSimulator
from hydroswarm.training.corpus import build_feature_context, fit_signature_library, scenario_to_example

LOOP_GRID_PATH = "data/topologies/loop-grid.inp"
COASTAL_BRANCH_PATH = "data/topologies/coastal-branch.inp"


def test_loop_grid_is_a_genuinely_different_structure_from_every_existing_topology() -> None:
    model = wntr.network.WaterNetworkModel(LOOP_GRID_PATH)
    assert len(model.junction_name_list) == 8
    assert len(model.reservoir_name_list) == 1
    assert len(model.tank_name_list) == 0
    # Cycles in a connected graph = edges - nodes + 1. 9 nodes (8 junctions
    # + 1 reservoir), 11 pipes -> 3 independent loop faces in the 2x4 grid,
    # more than either existing topology (golden-reference and
    # branched-loop each have exactly 1).
    node_count = len(model.node_name_list)
    pipe_count = len(model.pipe_name_list)
    assert pipe_count - node_count + 1 == 3


def test_coastal_branch_combines_a_tank_and_a_branch_unlike_either_training_topology() -> None:
    model = wntr.network.WaterNetworkModel(COASTAL_BRANCH_PATH)
    assert len(model.junction_name_list) == 6
    assert len(model.tank_name_list) == 1
    assert len(model.reservoir_name_list) == 1


def _validate_and_generate_one_example(path: str, family: str):
    simulator = HydraulicSimulator(path)
    assert simulator.validate() == ()
    simulator.calculate_state(at_time=0)  # must not raise

    model = wntr.network.WaterNetworkModel(path)
    junctions = tuple(sorted(model.junction_name_list))
    generator = WNTRScenarioGenerator()
    scenarios = [
        generator.generate(
            model,
            ScenarioGenerationConfig(
                seed=600_000 + index * 100,
                network_id=family,
                network_family=family,
                split=DatasetSplit.TRAIN,
                stage=CurriculumStage.CLEAN,
                source_node=source,
                sensor_count=min(4, len(junctions)),
            ),
        )
        for index, source in enumerate(junctions)
    ]
    library = fit_signature_library(scenarios, junctions)
    feature_context = build_feature_context(model)
    return scenario_to_example(scenarios[0], model, library, feature_context=feature_context), junctions


#: 18 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
def test_loop_grid_flows_through_the_real_corpus_pipeline() -> None:
    example, junctions = _validate_and_generate_one_example(LOOP_GRID_PATH, "loop-grid")
    assert example.inputs["node_features"].shape == (len(junctions) + 1, 19)
    assert example.targets["source_node"].item() in range(len(junctions) + 1)


#: 14 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
def test_coastal_branch_flows_through_the_real_corpus_pipeline() -> None:
    example, junctions = _validate_and_generate_one_example(COASTAL_BRANCH_PATH, "coastal-branch")
    assert example.inputs["node_features"].shape == (len(junctions) + 2, 19)  # + reservoir + tank
    assert example.targets["source_node"].item() in range(len(junctions) + 2)
