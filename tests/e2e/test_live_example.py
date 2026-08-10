from __future__ import annotations

import pytest

from hydroswarm.evaluation.live_example import CANDIDATES, TRUE_SOURCE, build_live_example_inputs
from hydroswarm.runtime.paths import resolve_frozen_scenario_dir

#: Drives a real bounded WNTR simulation -- see pyproject.toml's
#: real_simulation marker docstring.
pytestmark = pytest.mark.real_simulation


@pytest.fixture(scope="module")
def inputs():
    return build_live_example_inputs(resolve_frozen_scenario_dir())


def test_network_inp_text_is_the_real_frozen_network_file(inputs) -> None:
    assert inputs["network_filename"] == "golden_network.inp"
    real_text = (resolve_frozen_scenario_dir() / "golden_network.inp").read_text(encoding="utf-8")
    assert inputs["network_inp_text"] == real_text


def test_candidate_signatures_cover_every_candidate_node_with_real_simulated_values(inputs) -> None:
    signatures = inputs["candidate_signatures_mg_l"]
    assert set(signatures) == set(CANDIDATES)
    for value in signatures.values():
        assert isinstance(value, float)
        assert value >= 0.0
    # The true source's own node must show measurable contamination by the
    # sample time -- a real physical consequence of simulating a real
    # source profile there, not an arbitrary placeholder.
    assert signatures[TRUE_SOURCE] > 0.0


def test_initial_observation_is_a_real_pre_contamination_reading(inputs) -> None:
    observation = inputs["initial_observation"]
    assert observation["concentration_mg_l"] == 0.0
    assert observation["node_id"]
    assert observation["sensor_id"]


def test_is_deterministic_across_repeated_calls(inputs) -> None:
    second = build_live_example_inputs(resolve_frozen_scenario_dir())
    assert second["candidate_signatures_mg_l"] == inputs["candidate_signatures_mg_l"]
    assert second["network_inp_text"] == inputs["network_inp_text"]
