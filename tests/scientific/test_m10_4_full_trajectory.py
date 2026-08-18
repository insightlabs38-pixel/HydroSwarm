"""Milestone 10.4 focused tests: governed full-trajectory end-to-end
validation.

Frozen protocol: `docs/evaluation/HYDROCORE_V5_M10_4_FULL_TRAJECTORY_PROTOCOL.md`.
Reuses `hydroswarm.evaluation.live_robustness`'s own governed Condition/
scenario/payload machinery -- no new perturbation framework. Real,
small-scale WNTR/EPANET-backed incidents (golden-reference only, small
incident counts) -- marked `real_simulation` like every other M9/M10
integration test that drives a real HydraulicSimulator.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

import m10_4_common as m104  # noqa: E402
import m10_4_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402

from hydroswarm.api import create_app  # noqa: E402
from hydroswarm.evaluation.live_robustness import Condition, locked_test_opened  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

pytestmark = pytest.mark.real_simulation

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def _client_seed0():
    factory = m104.M10_4_PipelineFactory(seed=m10.SEEDS[0], project_root=m10.ROOT_PATH)
    tmp = tempfile.TemporaryDirectory(prefix="hydroswarm-m10-4-test-")
    tmp_path = Path(tmp.name)
    app = create_app(
        pipeline_factory=factory, database_path=tmp_path / "state.sqlite3",
        ledger_path=tmp_path / "audit.sqlite3", network_directory=tmp_path / "networks",
    )
    inp = m104.network_inp_path("golden-reference", tmp_path)
    client = TestClient(app)
    client.__enter__()
    imported = client.post("/api/networks/import", files={"file": (inp.name, inp.read_bytes(), "application/octet-stream")})
    network_id = imported.json()["network_id"]
    yield client, inp, network_id, factory
    client.__exit__(None, None, None)
    tmp.cleanup()


# ---------------------------------------------------------------------------
# Canonical checkpoint / calibration identity.
# ---------------------------------------------------------------------------


def test_canonical_checkpoints_hash_verified():
    verification = m104.verify_canonical_checkpoints()
    assert set(verification) == set(m10.SEEDS)
    for record in verification.values():
        assert record["matches"], record


def test_no_experimental_checkpoint_path_used():
    verification = m104.verify_canonical_checkpoints()
    forbidden = ("m10-2-refit", "m10-3-refit", "level-a", "level-b")
    for record in verification.values():
        lowered = record["path"].lower()
        assert not any(marker in lowered for marker in forbidden)


def test_calibration_identity_reaches_calibrated_true(_client_seed0):
    client, inp, network_id, _factory = _client_seed0
    safety = dict(m104.SAFETY_COUNTERS_TEMPLATE)
    reached = False
    for i in range(4):
        cond = Condition(f"test-cal-{i}", "nominal", "clean_operational", 1_500_010_000 + i, network_id="golden-reference")
        record = m104.run_incident_pair(client=client, network_path=inp, network_id=network_id, condition=cond, maximum_samples=3, safety=safety)
        if record["arms"]["FULL"].get("final_analysis", {}).get("calibrated"):
            reached = True
    assert reached, "frozen calibration identity never reaches calibrated=True through the real pipeline"


# ---------------------------------------------------------------------------
# Governance: learned OOD/Scout/Strategist non-authoritative.
# ---------------------------------------------------------------------------


def test_trained_tasks_excludes_learned_specialists():
    assert m104.M10_4_TRAINED_TASKS == frozenset({"sentinel"})


def test_production_scout_endpoint_is_rank_sample_locations_not_deterministic_fallback():
    """`/api/incidents/{id}/samples/recommend` (`recommend_sample`) reads
    `analysis.sample_result`, computed by `HybridInferencePipeline.analyze()`
    via `rank_sample_locations` -- NOT `HydroScout.deterministic_fallback`.
    `HydroScout` DOES appear elsewhere in this module (`_run_swarm_workflow_
    unlocked`, a SEPARATE, unrelated "swarm replay" endpoint M10.4 never
    calls -- it wraps `HydroScout` as a thin data adapter over the SAME
    already-decided `analysis.recommended_sample`, not as an independent
    decision authority), so this test checks the SPECIFIC recommend_sample
    function body, not the whole module."""

    import importlib
    import inspect

    app_module = importlib.import_module("hydroswarm.api.app")
    source = inspect.getsource(app_module)
    assert "analysis.sample_result" in source
    start = source.index("def recommend_sample")
    end = source.index("\n    @app.", start)
    recommend_sample_body = source[start:end]
    assert "sample_result" in recommend_sample_body
    assert "HydroScout" not in recommend_sample_body
    assert "deterministic_fallback" not in recommend_sample_body


def test_production_strategist_path_is_deterministic_candidate_generation():
    import inspect

    from hydroswarm.inference import pipeline as pipeline_module
    source = inspect.getsource(pipeline_module)
    assert "generate_response_plans" in source


# ---------------------------------------------------------------------------
# Causal evidence revelation / no future leakage.
# ---------------------------------------------------------------------------


def test_future_dated_evidence_is_rejected_causally(_client_seed0):
    """A hard requirement of the M10.4 causal trajectory semantics: no
    future evidence may be admitted at analysis time. Confirmed here
    against the REAL production /analyze endpoint (this exact behavior is
    why `m10_4_common.run_incident_pair`'s origin is deliberately bounded
    to the past)."""

    client, inp, network_id, _factory = _client_seed0
    from datetime import UTC, datetime, timedelta

    from hydroswarm.data.scenarios import WNTRScenarioGenerator
    from hydroswarm.evaluation.live_robustness import _payloads, _scenario_config
    import wntr

    network = wntr.network.WaterNetworkModel(str(inp))
    condition = Condition("test-future", "nominal", "clean_operational", 1_500_020_000, network_id="golden-reference")
    scenario, _randomized = WNTRScenarioGenerator().generate_with_network(network, _scenario_config(condition))
    future_origin = datetime.now(UTC) + timedelta(days=1000)
    observations = _payloads(scenario, condition, future_origin)
    created = client.post("/api/incidents", json={
        "network_id": network_id, "detected_at": future_origin.isoformat(),
        "observations": observations, "maximum_samples": 3,
    })
    assert created.status_code == 201
    incident_id = created.json()["incident_id"]
    analyzed = client.post(f"/api/incidents/{incident_id}/analyze")
    assert analyzed.status_code == 409, "future-dated evidence must be causally rejected, not silently admitted"


def test_no_scout_reselection_and_budget_enforced(_client_seed0):
    client, inp, network_id, _factory = _client_seed0
    safety = dict(m104.SAFETY_COUNTERS_TEMPLATE)
    cond = Condition("test-lowcov", "sensor_coverage", "25%", 1_500_030_000, network_id="golden-reference", coverage=0.25)
    m104.run_incident_pair(client=client, network_path=inp, network_id=network_id, condition=cond, maximum_samples=3, safety=safety)
    assert safety["already_sampled_reselected"] == 0
    assert safety["sampling_budget_exceeded"] == 0
    assert safety["inaccessible_sample_selected"] == 0


def test_sampling_budget_pre_exhausted_fails_closed(_client_seed0):
    client, _inp, network_id, _factory = _client_seed0
    created = client.post("/api/incidents", json={
        "network_id": network_id, "detected_at": "2025-01-01T00:00:00Z",
        "observations": [], "maximum_samples": 0,
    })
    if created.status_code == 201:
        incident_id = created.json()["incident_id"]
        analyzed = client.post(f"/api/incidents/{incident_id}/analyze")
        if analyzed.status_code == 200:
            recommend = client.post(f"/api/incidents/{incident_id}/samples/recommend")
            assert recommend.status_code == 409
        else:
            assert analyzed.status_code in (409, 422)
    else:
        assert created.status_code in (409, 422)


# ---------------------------------------------------------------------------
# WNTR rejection cannot become actionable / human approval required.
# ---------------------------------------------------------------------------


def test_wntr_rejected_plan_cannot_be_approved(_client_seed0):
    client, inp, network_id, _factory = _client_seed0
    safety = dict(m104.SAFETY_COUNTERS_TEMPLATE)
    for i in range(6):
        cond = Condition(f"test-wntr-{i}", "nominal", "clean_operational", 1_500_040_000 + i, network_id="golden-reference")
        m104.run_incident_pair(client=client, network_path=inp, network_id=network_id, condition=cond, maximum_samples=3, safety=safety)
    assert safety["wntr_rejected_plan_surfaced_as_safe"] == 0
    assert safety["human_approval_bypassed"] == 0


def test_paired_arm_initial_state_equal(_client_seed0):
    client, inp, network_id, _factory = _client_seed0
    safety = dict(m104.SAFETY_COUNTERS_TEMPLATE)
    cond = Condition("test-paired", "nominal", "clean_operational", 1_500_050_000, network_id="golden-reference")
    record = m104.run_incident_pair(client=client, network_path=inp, network_id=network_id, condition=cond, maximum_samples=3, safety=safety)
    assert record["paired_initial_state_equal"] is True


# ---------------------------------------------------------------------------
# Locked test guard / seed disjointness / experimental checkpoint forbidden.
# ---------------------------------------------------------------------------


def test_locked_test_guard():
    assert locked_test_opened(ROOT) is False


def test_seed_namespace_disjoint_from_every_prior_milestone():
    disjointness = m104.verify_seed_disjointness()
    assert disjointness["disjoint"], disjointness
    assert all(not overlap for overlap in disjointness["overlaps"].values())


def test_seed_formula_matches_frozen_protocol():
    for family, kind in proto.population_cells()[:3]:
        seed = proto.incident_seed(m10.SEEDS[0], family, kind, 0)
        assert m104.M10_4_RANGE[0] <= seed <= m104.M10_4_RANGE[1]


# ---------------------------------------------------------------------------
# Closure decision logic (pure function, no execution required).
# ---------------------------------------------------------------------------


def test_closure_decision_blocked_on_preflight_failure():
    from run_m10_4_metrics import compute_closure

    closure = compute_closure(
        preflight={"result": "M10_4_PREFLIGHT_BLOCKED"},
        gate={"all_checks_pass": True, "checks": {}},
        safety={"all_zero": True},
    )
    assert closure["closure_state"] == "M10_4_FULL_TRAJECTORY_BLOCKED"
    assert closure["m10_5_authorized"] is False


def test_closure_decision_blocked_on_safety_violation():
    from run_m10_4_metrics import compute_closure

    closure = compute_closure(
        preflight={"result": "M10_4_PREFLIGHT_PASS"},
        gate={"all_checks_pass": True, "checks": {}},
        safety={"all_zero": False},
    )
    assert closure["closure_state"] == "M10_4_FULL_TRAJECTORY_BLOCKED"


def test_closure_decision_utility_not_established():
    from run_m10_4_metrics import compute_closure

    closure = compute_closure(
        preflight={"result": "M10_4_PREFLIGHT_PASS"},
        gate={"all_checks_pass": False, "checks": {"A": True, "B": False}},
        safety={"all_zero": True},
    )
    assert closure["closure_state"] == "M10_4_FULL_TRAJECTORY_UTILITY_NOT_ESTABLISHED"


def test_closure_decision_pass():
    from run_m10_4_metrics import compute_closure

    closure = compute_closure(
        preflight={"result": "M10_4_PREFLIGHT_PASS"},
        gate={"all_checks_pass": True, "checks": {"A": True}},
        safety={"all_zero": True},
    )
    assert closure["closure_state"] == "M10_4_FULL_TRAJECTORY_PASS"
    assert closure["m10_5_authorized"] is False


# ---------------------------------------------------------------------------
# Deterministic trajectory replay.
# ---------------------------------------------------------------------------


def test_deterministic_trajectory_replay(_client_seed0):
    """The same physical seed run twice through separate incidents produces
    the same initial source metrics (no hidden nondeterminism in the
    causal state machine)."""

    client, inp, network_id, _factory = _client_seed0
    safety = dict(m104.SAFETY_COUNTERS_TEMPLATE)
    cond = Condition("test-replay", "nominal", "clean_operational", 1_500_060_000, network_id="golden-reference")
    first = m104.run_incident_pair(client=client, network_path=inp, network_id=network_id, condition=cond, maximum_samples=3, safety=safety)
    second = m104.run_incident_pair(client=client, network_path=inp, network_id=network_id, condition=cond, maximum_samples=3, safety=safety)
    assert first["arms"]["FULL"]["initial"] == second["arms"]["FULL"]["initial"]
    assert first["source_node"] == second["source_node"]
