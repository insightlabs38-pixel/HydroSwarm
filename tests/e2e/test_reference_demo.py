from __future__ import annotations

import pytest

from hydroswarm.evaluation import GoldenScenarioRunner, build_reference_incident_artifact

#: Drives the real WNTR-backed golden scenario -- see
#: pyproject.toml's real_simulation marker docstring.
pytestmark = pytest.mark.real_simulation


@pytest.fixture(scope="module")
def golden_result(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference-demo-golden")
    return GoldenScenarioRunner(root, seed=2026, cache_directory=root / "cache").run()


@pytest.fixture(scope="module")
def artifact(golden_result):
    return build_reference_incident_artifact(
        golden_result, generator="test", source_commit="deadbeef"
    )


def test_artifact_is_tied_to_the_golden_result_by_hash(artifact, golden_result) -> None:
    assert artifact["golden_result_hash"] == golden_result["result_sha256"]
    assert artifact["final_event_hash"] == golden_result["workflow"]["final_event_hash"]
    assert artifact["event_count"] == golden_result["workflow"]["event_count"]
    for name, entry in golden_result["fixture_manifest"]["artifacts"].items():
        assert artifact["source_artifact_hashes"][name] == entry["sha256"]


def test_generator_is_deterministic_across_repeated_builds(golden_result) -> None:
    first = build_reference_incident_artifact(golden_result, generator="test", source_commit="deadbeef")
    second = build_reference_incident_artifact(golden_result, generator="test", source_commit="deadbeef")
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert first["generated_at"] != second["generated_at"], (
        "generated_at is a real wall-clock timestamp -- it is expected to differ, and is "
        "deliberately excluded from the semantic hash above"
    )


def test_milestones_cover_every_event_contiguously(artifact) -> None:
    milestones = artifact["milestones"]
    assert milestones[0]["event_sequence_start"] == 0
    assert milestones[-1]["event_sequence_end"] == artifact["event_count"] - 1
    for previous, current in zip(milestones, milestones[1:]):
        assert current["event_sequence_start"] == previous["event_sequence_end"] + 1


def test_initial_candidate_set_is_broad_and_uniform(artifact) -> None:
    initial_view = artifact["milestones"][1]["incident_view"]
    assert initial_view["candidates"] is not None
    assert len(initial_view["candidates"]) == 4
    assert len(set(initial_view["candidates"].values())) == 1, "initial probabilities must be uniform"


def test_sample_recommendation_is_j2_with_measured_information_gain(artifact, golden_result) -> None:
    milestone = next(m for m in artifact["milestones"] if m["milestone_id"] == "sample_recommended")
    recommendation = milestone["incident_view"]["recommended_sample"]
    assert recommendation["node_id"] == "J2"
    expected_gain = golden_result["localization"]["information_gain_by_node_bits"]["J2"]
    assert recommendation["expected_information_gain_bits"] == expected_gain


def test_posterior_only_appears_after_sample_receipt(artifact) -> None:
    by_id = {m["milestone_id"]: m for m in artifact["milestones"]}
    initial_candidates = by_id["initial_uncertainty"]["incident_view"]["candidates"]
    sample_received_candidates = by_id["sample_received"]["incident_view"]["candidates"]
    posterior_candidates = by_id["posterior_contracted"]["incident_view"]["candidates"]

    assert sample_received_candidates == initial_candidates, (
        "candidates must still read as pre-sample at the sample_received milestone"
    )
    assert posterior_candidates != initial_candidates
    assert max(posterior_candidates, key=posterior_candidates.get) == "J2"


def test_unsafe_plan_rejected_only_after_verifier_stage(artifact) -> None:
    by_id = {m["milestone_id"]: m for m in artifact["milestones"]}
    plans_generated = by_id["plans_generated"]["incident_view"]["plans"]
    assert all(entry["verification"] is None for entry in plans_generated)

    unsafe_rejected = by_id["unsafe_plan_rejected"]["incident_view"]["plans"]
    unsafe_entry = next(e for e in unsafe_rejected if e["plan"]["name"] == "Close sole reservoir feeder")
    assert unsafe_entry["verification"]["decision"] == "REJECTED"


def test_safe_plan_verified_only_after_verifier_stage(artifact) -> None:
    by_id = {m["milestone_id"]: m for m in artifact["milestones"]}
    unsafe_rejected_plans = by_id["unsafe_plan_rejected"]["incident_view"]["plans"]
    safe_entry_before = next(e for e in unsafe_rejected_plans if e["plan"]["name"] == "Flush downstream J4")
    assert safe_entry_before["verification"] is None

    safe_verified_plans = by_id["safe_plan_verified"]["incident_view"]["plans"]
    safe_entry_after = next(e for e in safe_verified_plans if e["plan"]["name"] == "Flush downstream J4")
    assert safe_entry_after["verification"]["decision"] == "VERIFIED"


def test_approval_not_shown_before_plan_is_verified_and_selected(artifact) -> None:
    by_id = {m["milestone_id"]: m for m in artifact["milestones"]}
    for milestone_id in ("alert", "initial_uncertainty", "plans_generated", "unsafe_plan_rejected"):
        assert by_id[milestone_id]["incident_view"]["approval_pending"] is False
        assert by_id[milestone_id]["incident_view"]["approved_plan_id"] is None

    boundary = by_id["human_approval_boundary"]["incident_view"]
    assert boundary["approval_pending"] is True
    assert boundary["approved_plan_id"] is None
    assert boundary["selected_plan_id"] is not None


def test_final_completion_occurs_only_after_human_approval(artifact) -> None:
    by_id = {m["milestone_id"]: m for m in artifact["milestones"]}
    boundary = by_id["human_approval_boundary"]["incident_view"]
    completed = by_id["completed"]["incident_view"]

    assert boundary["final_event_hash"] is None
    assert completed["approved_plan_id"] == boundary["selected_plan_id"]
    assert completed["final_event_hash"] == artifact["final_event_hash"]


def test_human_approval_boundary_pauses_and_others_auto_advance(artifact) -> None:
    for milestone in artifact["milestones"]:
        if milestone["milestone_id"] == "human_approval_boundary":
            assert milestone["auto_advance"] is False
            assert milestone["pause_reason"]
        else:
            assert milestone["auto_advance"] is True
            assert milestone["pause_reason"] is None


def test_artifact_round_trips_through_json_with_no_extra_computation(artifact) -> None:
    import json

    text = json.dumps(artifact)
    reloaded = json.loads(text)
    assert reloaded["artifact_sha256"] == artifact["artifact_sha256"]
