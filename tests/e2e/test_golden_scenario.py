from __future__ import annotations

from hydroswarm.evaluation import GoldenScenarioRunner


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

    assert result["workflow"]["sampling_pause_state"] == "SAMPLE_SELECTION"
    assert result["workflow"]["approval_pause_state"] == "HUMAN_APPROVAL"
    assert result["workflow"]["pause_replay_state"] == "HUMAN_APPROVAL"
    assert result["workflow"]["completed_replay_state"] == "COMPLETE"
    assert "OPERATOR APPROVAL PENDING" in result["operational_summary"]

