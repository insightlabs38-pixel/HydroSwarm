from __future__ import annotations

import pytest

from hydroswarm.evaluation import GoldenScenarioRunner


#: 13 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
pytestmark = pytest.mark.real_simulation


@pytest.mark.full_simulation
def test_golden_scenario_executes_complete_human_gated_workflow(tmp_path) -> None:
    result = GoldenScenarioRunner(tmp_path, seed=2026, cache_directory=tmp_path / "cache").run()

    localization = result["localization"]
    assert len(localization["initial_credible_region"]) >= 3
    assert localization["sample_node"] == "J2"
    assert localization["candidate_contraction"] > 0
    assert max(localization["posterior_probabilities"], key=localization["posterior_probabilities"].get) == "J2"

    assert result["plans"]["generated_count"] >= 3
    assert result["plans"]["unsafe"]["verification"]["decision"] == "REJECTED"
    assert "PRESSURE_BELOW_MINIMUM" in result["plans"]["unsafe"]["verification"]["rejection_codes"]
    assert result["plans"]["safe"]["verification"]["decision"] == "VERIFIED"
    assert result["consequences"]["no_response"]["contaminant_mass_consumed_mg"] > 0
    assert result["consequences"]["exposure_reduction_mg"] > 0
    #: important-issues.txt requirement 11: golden/demo evaluation must use
    #: the SAME canonical plan consequence evaluator as training/PlanVerifier/
    #: the live /verify API -- confirmed by checking the evaluator's own
    #: exposure_evaluated marker survives into the fixture's serialized
    #: verification payloads, not a separately hand-merged workaround value.
    assert result["plans"]["no_response"]["verification"]["consequences"]["exposure_evaluated"] is True
    assert result["plans"]["safe"]["verification"]["consequences"]["exposure_evaluated"] is True

    assert result["workflow"]["sampling_pause_state"] == "SAMPLE_SELECTION"
    assert result["workflow"]["approval_pause_state"] == "HUMAN_APPROVAL"
    assert result["workflow"]["pause_replay_state"] == "HUMAN_APPROVAL"
    assert result["workflow"]["completed_replay_state"] == "COMPLETE"
    assert "OPERATOR APPROVAL PENDING" in result["operational_summary"]

