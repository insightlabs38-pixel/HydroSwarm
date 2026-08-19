"""M11.6A-1 -- locked evaluation DESIGN FREEZE tests.

Every test here uses only tmp directories and NON-LOCKED development/synthetic
fixtures. The ``M11_6A_DESIGN_SMOKE_ONLY`` namespace is used exclusively for
smoke fixtures and is asserted to be forbidden from the real locked splits.
No test materializes a locked population, derives a final locked seed, or
evaluates the frozen finalist on locked data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import m11_6a_design as design  # noqa: E402
import m11_6a_topology as topology  # noqa: E402
import run_m11_6_locked_evaluation as evaluator  # noqa: E402
import run_m11_6a_materialize as materializer  # noqa: E402

TEST_SHA = "0" * 40
SMOKE = design.FORBIDDEN_SMOKE_NAMESPACE


def _final_master() -> str:
    return design.derive_master_seed(TEST_SHA, design.MASTER_DOMAIN_FINAL)


def _topology_master() -> str:
    return design.derive_master_seed(TEST_SHA, design.MASTER_DOMAIN_TOPOLOGY)


# ---------------------------------------------------------------------------
# Seed derivation (task Section 6): determinism + domain separation.
# ---------------------------------------------------------------------------

def test_seed_derivation_is_deterministic():
    a = design.derive_seed(_final_master(), "FINAL_SCENARIO", 0)
    b = design.derive_seed(_final_master(), "FINAL_SCENARIO", 0)
    assert a == b
    assert 0 <= a < design.SEED_MODULUS


def test_seed_derivation_domain_separation():
    final = design.derive_seed(_final_master(), "FINAL_SCENARIO", 0)
    topo = design.derive_seed(_topology_master(), "TOPOLOGY_TEST_SCENARIO", 0)
    assert final != topo
    # Different labels under the same master also separate.
    assert design.derive_seed(_final_master(), "FINAL_SCENARIO", 0) != design.derive_seed(_final_master(), "TOPOLOGY_TEST_SCENARIO", 0)


def test_seed_derivation_counter_changes_seed():
    a = design.derive_seed(_final_master(), "FINAL_SCENARIO", 0, counter=0)
    b = design.derive_seed(_final_master(), "FINAL_SCENARIO", 0, counter=1)
    assert a != b


def test_master_seed_formula_rejects_unknown_domain():
    with pytest.raises(ValueError):
        design.derive_master_seed(TEST_SHA, "BOGUS")


def test_master_seed_formula_is_sha256_of_frozen_material():
    expected = "HYDROSWARM|M11.6|LOCKED_FINAL|v1|" + TEST_SHA
    import hashlib

    assert design.derive_master_seed(TEST_SHA, design.MASTER_DOMAIN_FINAL) == hashlib.sha256(expected.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest schema + validation (task Section 11).
# ---------------------------------------------------------------------------

def _valid_manifest() -> dict:
    return {
        "schema_version": design.MANIFEST_SCHEMA_VERSION,
        "design_freeze_commit_sha": TEST_SHA,
        "design_protocol_sha256": design.design_hash(),
        "generator_source_sha256": {"m11_6a_design.py": "a" * 64},
        "evaluator_source_sha256": {"run_m11_6_locked_evaluation.py": "b" * 64},
        "seed_derivation": design.seed_derivation_spec(),
        "master_seeds": {
            "locked_final_master": {"domain": design.MASTER_DOMAIN_FINAL, "hex": _final_master()},
            "locked_topology_master": {"domain": design.MASTER_DOMAIN_TOPOLOGY, "hex": _topology_master()},
        },
        "splits": {
            design.LOCKED_FINAL_TEST: {"count": design.LOCKED_FINAL_TOTAL},
            design.LOCKED_TOPOLOGY_TEST: {"count": design.LOCKED_TOPOLOGY_TOTAL},
        },
        "topologies": [],
        "scenarios": [],
        "artifact_sha256": {},
        "simulator": {"backend": "WNTR/EPANET", "wntr_version": "1.5.0"},
        "generation_complete": True,
        "overlap_audit": {"result": "PASS"},
        "novelty_audit": {"result": "PASS"},
        "evaluated_by_finalist": False,
        "locked_test_opened": False,
    }


def test_manifest_hash_generation_is_deterministic():
    manifest = _valid_manifest()
    assert evaluator.sha256_file_manifest(manifest) == evaluator.sha256_file_manifest(manifest)


def test_manifest_validates_clean():
    assert design.validate_manifest(_valid_manifest()) == []


def test_manifest_tampering_rejected():
    manifest = _valid_manifest()
    manifest["schema_version"] = "tampered"
    assert "schema_version" in " ".join(design.validate_manifest(manifest))

    manifest = _valid_manifest()
    manifest["evaluated_by_finalist"] = True
    assert any("evaluated_by_finalist" in v for v in design.validate_manifest(manifest))

    manifest = _valid_manifest()
    manifest["metrics"] = {"top1": 0.9}
    assert any("forbidden field" in v for v in design.validate_manifest(manifest))

    manifest = _valid_manifest()
    manifest["locked_topology_test_scenario_0"] = SMOKE
    assert any(SMOKE in v for v in design.validate_manifest(manifest))


def test_manifest_requires_both_locked_splits():
    manifest = _valid_manifest()
    del manifest["splits"][design.LOCKED_TOPOLOGY_TEST]
    assert any("splits missing locked_topology_test" in v for v in design.validate_manifest(manifest))


# ---------------------------------------------------------------------------
# Frozen population counts (task Section 5) chosen independent of outcomes.
# ---------------------------------------------------------------------------

def test_locked_population_counts_frozen():
    assert design.LOCKED_FINAL_TOTAL == 105
    assert design.LOCKED_TOPOLOGY_TOTAL == 20
    assert len(design.locked_final_cells()) == 21
    assert len(design.LOCKED_FINAL_CONDITIONS) == 7
    assert design.LOCKED_TOPOLOGY_INSTANCES == 4


def test_smoke_namespace_forbidden_from_locked_splits():
    assert SMOKE not in design.LOCKED_FINAL_TEST
    assert SMOKE not in design.LOCKED_TOPOLOGY_TEST
    assert SMOKE not in " ".join(design.LOCKED_SPLIT_NAMES)
    assert SMOKE not in " ".join(design.LOCKED_FINAL_FAMILIES)
    assert SMOKE in design.scenario_definition_schema()["forbidden_namespaces"]


# ---------------------------------------------------------------------------
# Topology generator determinism / novelty (task Sections 7/8).
# ---------------------------------------------------------------------------

def test_topology_spec_bounds_frozen():
    spec = topology.topology_spec()
    assert spec["family_grammar"]["junction_count"] == "9 + topology_index (index 0..3 => 9..12 junctions)"
    assert spec["max_candidate_attempts"] == design.MAX_TOPOLOGY_CANDIDATE_ATTEMPTS


def test_prior_topology_inventory_is_complete_and_decisive():
    assert len(design.PRIOR_TOPOLOGY_SIGNATURES) == 6
    junction_counts = {item["junction_count"] for item in design.PRIOR_TOPOLOGY_SIGNATURES}
    assert max(junction_counts) == 8
    assert min(junction_counts) == 4


def test_is_prior_topology_detects_every_prior():
    for prior in design.PRIOR_TOPOLOGY_SIGNATURES:
        sig = design.graph_signature(
            prior["node_count"], prior["junction_count"], prior["link_count"],
            prior["cycle_rank"], prior["degree_profile"],
        )
        assert design.is_prior_topology(sig, prior["network_sha256"]) is True


@pytest.mark.real_simulation
def test_topology_generator_determinism_and_novelty():
    pytest.importorskip("wntr")
    master = _topology_master()
    first = topology.generate_locked_topology(master, 0)
    second = topology.generate_locked_topology(master, 0)
    assert first["network_sha256"] == second["network_sha256"]
    assert first["graph_signature"] == second["graph_signature"]
    assert first["candidate_index"] == second["candidate_index"]
    novel, reasons = topology.is_novel_topology(first["network"])
    assert novel, reasons


@pytest.mark.real_simulation
def test_topology_generator_produces_4_distinct_novel_topologies():
    pytest.importorskip("wntr")
    master = _topology_master()
    hashes = []
    signatures = []
    for index in range(design.LOCKED_TOPOLOGY_INSTANCES):
        result = topology.generate_locked_topology(master, index)
        assert result["junction_count"] == 9 + index
        assert result["cycle_rank"] == 1 + index
        hashes.append(result["network_sha256"])
        signatures.append(result["graph_signature"])
    assert len(set(hashes)) == design.LOCKED_TOPOLOGY_INSTANCES
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            assert not design.signatures_equal(signatures[i], signatures[j])


@pytest.mark.real_simulation
def test_topology_validity_rejection_is_simulator_only():
    """Feasibility is decided by EPANET hydraulics (junction pressures), never a
    model output; an infeasible elevation/head combination is rejected."""
    pytest.importorskip("wntr")
    seed = design.derive_seed(_topology_master(), "TOPOLOGY_TEST_NETWORK", 0, 0)
    model = topology.build_procedural_topology_model(seed, 9, 1)
    assert topology.topology_is_feasible(model) is True
    # A deliberately infeasible network (huge elevations, tiny head) is rejected.
    import wntr as _wntr

    bad = _wntr.network.WaterNetworkModel()
    bad.add_pattern("diurnal", list(topology.DIURNAL_PATTERN))
    bad.add_reservoir("R1", base_head=100.0, coordinates=(0.0, 0.0))
    for i in range(9):
        bad.add_junction(f"J{i + 1}", base_demand=0.05, demand_pattern="diurnal", elevation=200.0, coordinates=(i * 500.0, 0.0))
    for i in range(9):
        start = "R1" if i == 0 else f"J{i}"
        bad.add_pipe(f"P_{start}_{i + 1}", start, f"J{i + 1}", length=1200.0, diameter=0.15, roughness=100.0, minor_loss=0.0, initial_status="OPEN")
    bad.options.time.pattern_timestep = 3600
    bad.options.time.hydraulic_timestep = 3600
    bad.options.time.quality_timestep = 300
    bad.options.time.duration = 86400
    assert topology.topology_is_feasible(bad) is False


# ---------------------------------------------------------------------------
# Scenario definition determinism + non-overlap (task Section 9).
# ---------------------------------------------------------------------------

def test_scenario_definition_hash_is_deterministic():
    definition = {
        "schema_version": design.SCENARIO_SCHEMA_VERSION,
        "split": design.LOCKED_FINAL_TEST,
        "scenario_index": 0,
        "topology_id": "locked-final:golden-reference",
        "network_family": "golden-reference",
        "network_sha256": "x" * 64,
        "seed": 123,
        "seed_domain": "FINAL_SCENARIO",
        "seed_derivation_counter": 0,
        "event_type": "contamination",
        "source_node": "J1",
        "condition_kind": "NOMINAL",
        "condition": {"perturbation_type": "nominal"},
        "generator_config": design.scenario_config_for_condition("NOMINAL"),
    }
    assert design.scenario_definition_hash(definition) == design.scenario_definition_hash(definition)
    other = dict(definition, scenario_index=1)
    assert design.scenario_definition_hash(other) != design.scenario_definition_hash(definition)


@pytest.mark.real_simulation
def test_locked_final_definitions_deterministic_and_counted():
    pytest.importorskip("wntr")
    master = _final_master()
    a = materializer.build_locked_final_definitions(master)
    b = materializer.build_locked_final_definitions(master)
    assert len(a) == design.LOCKED_FINAL_TOTAL
    assert [design.scenario_definition_hash(d) for d in a] == [design.scenario_definition_hash(d) for d in b]
    hashes = [design.scenario_definition_hash(d) for d in a]
    assert len(set(hashes)) == len(hashes)


def test_overlap_audit_detects_collision():
    # Distinct definitions -> distinct canonical hashes -> PASS.
    assert materializer._overlap_audit([
        {"scenario_index": 0, "seed": 0}, {"scenario_index": 1, "seed": 0},
    ])["result"] == "PASS"
    # Identical definitions -> duplicate canonical hash -> FAIL.
    assert materializer._overlap_audit([
        {"scenario_index": 0, "seed": 0}, {"scenario_index": 0, "seed": 0},
    ])["result"] == "FAIL"
    # A seed outside the derived range is always flagged.
    assert materializer._overlap_audit([{"seed": design.SEED_MODULUS + 1}])["result"] == "FAIL"


# ---------------------------------------------------------------------------
# Exactly-once guard (task Sections 13/14).
# ---------------------------------------------------------------------------

def test_opened_record_binds_all_required_fields():
    record = design.opened_record(
        run_id="r", code_under_test_sha="c", design_freeze_sha="d",
        materialization_manifest_sha="m", finalist_checkpoint_sha="f",
        calibration_sha="cal", release_manifest_sha="rel", evaluator_sha="e",
    )
    for field in ("run_id", "code_under_test_sha", "design_freeze_sha",
                  "materialization_manifest_sha", "finalist_checkpoint_sha",
                  "calibration_sha", "release_manifest_sha", "evaluator_sha",
                  "locked_test_opened"):
        assert field in record
    assert record["locked_test_opened"] is True


def test_atomic_opened_creation_and_second_refusal(tmp_path):
    state = design.LockedRunState(tmp_path / "opened.json")
    assert state.exists() is False
    record = design.opened_record(
        run_id="r", code_under_test_sha="c", design_freeze_sha="d",
        materialization_manifest_sha="m", finalist_checkpoint_sha="f",
        calibration_sha="cal", release_manifest_sha="rel", evaluator_sha="e",
    )
    state.acquire(record)
    assert state.exists() is True
    assert state.read()["run_id"] == "r"
    with pytest.raises(design.LockedAlreadyOpened):
        state.acquire(record)


def test_no_force_or_reset_bypass():
    # The one-shot guard exposes no clearing/reset API.
    assert not hasattr(design.LockedRunState, "clear")
    assert not hasattr(design.LockedRunState, "reset")
    assert not hasattr(design.LockedRunState, "delete")
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    assert 'add_argument("--force"' not in source
    assert 'add_argument("--reset"' not in source
    assert 'add_argument("--manifest"' in source
    assert 'add_argument("--authorization"' in source


def test_post_open_crash_preserves_opened_state(tmp_path):
    state = design.LockedRunState(tmp_path / "opened.json")
    record = design.opened_record(
        run_id="r", code_under_test_sha="c", design_freeze_sha="d",
        materialization_manifest_sha="m", finalist_checkpoint_sha="f",
        calibration_sha="cal", release_manifest_sha="rel", evaluator_sha="e",
    )
    state.acquire(record)
    # Simulate a crash after OPENED: the record must persist and still refuse.
    del record
    assert state.exists() is True
    assert state.read()["run_id"] == "r"
    with pytest.raises(design.LockedAlreadyOpened):
        state.acquire(design.opened_record(
            run_id="r2", code_under_test_sha="c", design_freeze_sha="d",
            materialization_manifest_sha="m", finalist_checkpoint_sha="f",
            calibration_sha="cal", release_manifest_sha="rel", evaluator_sha="e",
        ))


# ---------------------------------------------------------------------------
# Evaluator contract: identity / authorization / manifest verification.
# ---------------------------------------------------------------------------

def test_frozen_finalist_identity_verified():
    assert evaluator.verify_finalist_identity() is True


def test_finalist_identity_mismatch_rejected(monkeypatch):
    monkeypatch.setitem(evaluator.FINALIST, "checkpoint", "0" * 64)
    assert evaluator.verify_finalist_identity() is False


def test_calibration_mismatch_rejected(monkeypatch):
    monkeypatch.setitem(evaluator.FINALIST, "calibration", "0" * 64)
    assert evaluator.verify_finalist_identity() is False


def test_authorization_absence_rejected():
    manifest = _valid_manifest()
    violations = evaluator.verify_authorization({}, manifest)
    assert any("authorization_consumed" in v for v in violations)
    assert any("locked_evaluation_authorized" in v for v in violations)


def test_authorization_must_be_fresh_and_match_manifest():
    manifest = _valid_manifest()
    authorization = {
        "authorization_consumed": False,
        "authorized_openings": 0,
        "locked_evaluation_authorized": True,
        "design_freeze_commit_sha": TEST_SHA,
        "manifest_sha256": evaluator.sha256_file_manifest(manifest),
        "finalist_checkpoint_sha256": evaluator.FINALIST["checkpoint"],
    }
    assert evaluator.verify_authorization(authorization, manifest) == []


def test_locked_test_opened_true_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluator, "locked_test_opened", lambda _root: True)
    manifest = _valid_manifest()
    authorization = {
        "authorization_consumed": False, "authorized_openings": 0,
        "locked_evaluation_authorized": True,
        "design_freeze_commit_sha": TEST_SHA,
        "manifest_sha256": evaluator.sha256_file_manifest(manifest),
        "finalist_checkpoint_sha256": evaluator.FINALIST["checkpoint"],
    }
    monkeypatch.setattr(evaluator, "OPENED_RECORD_PATH", tmp_path / "opened.json")
    with pytest.raises(RuntimeError, match="locked_test_opened"):
        evaluator.acquire_locked_open(authorization=authorization, manifest=manifest)


# ---------------------------------------------------------------------------
# Evaluator pure metric/gate/closure + result schema (task Sections 16/20).
# ---------------------------------------------------------------------------

def _synthetic_rows() -> list[dict]:
    return [
        {
            "split": design.LOCKED_FINAL_TEST, "top1_correct": True, "top3_correct": True,
            "reciprocal_rank": 1.0, "conformal_truth_coverage": True, "candidate_set_size": 2,
            "posterior_entropy": 1.0, "calibrated": True, "planning_allowed": True,
            "samples_taken": 1, "rounds": [{"status": "SAMPLE", "true_source_rank_before": 3,
            "true_source_rank_after": 1, "entropy_before": 2.0, "entropy_after": 1.0}],
            "plans_generated": 2, "plans_verified": 1, "plans_rejected": 0,
            "no_safe_plan": False, "human_approved": True, "safety_counters": {}, "invariants": {},
        },
        {
            "split": design.LOCKED_TOPOLOGY_TEST, "top1_correct": False, "top3_correct": True,
            "reciprocal_rank": 0.5, "conformal_truth_coverage": False, "candidate_set_size": 4,
            "posterior_entropy": 1.5, "calibrated": False, "planning_allowed": False,
            "samples_taken": 0, "rounds": [], "plans_generated": 0, "plans_verified": 0,
            "plans_rejected": 0, "no_safe_plan": False, "human_approved": False,
            "safety_counters": {}, "invariants": {},
        },
    ]


def test_development_dry_run_produces_metrics():
    metrics = evaluator.compute_metrics(_synthetic_rows())
    assert metrics["locked_final_test"]["source"]["n"] == 1
    assert metrics["locked_topology_test"]["source"]["n"] == 1
    assert metrics["locked_final_test"]["source"]["coverage"]["rate"] == 1.0
    assert metrics["locked_topology_test"]["topology_shift_predictive"] == "DESCRIPTIVE_NON_GATING"


def test_safety_counter_serialization():
    rows = _synthetic_rows()
    rows[0]["safety_counters"] = {"human_approval_bypassed": 1}
    counters = evaluator.compute_safety_counters(rows, identity_ok=True)
    assert counters["human_approval_bypassed"] == 1
    assert counters["finalist_identity_drift"] == 0
    assert set(counters) == set(design.SAFETY_COUNTERS_TEMPLATE)


def test_gates_and_closure_frozen_semantics():
    metrics = evaluator.compute_metrics(_synthetic_rows())
    safety = evaluator.compute_safety_counters(_synthetic_rows(), identity_ok=True)
    gates = evaluator.compute_gates(metrics=metrics, safety=safety, identity_ok=True, manifest_ok=True, novelty_ok=True)
    assert gates["all_checks_pass"] is True
    assert gates["coverage_floor"] == design.OPERATIONAL_COVERAGE_FLOOR

    closure = evaluator.compute_closure(gates=gates, crashed_after_open=False, opened=True)
    assert closure["closure_state"] == "M11_6_LOCKED_EVALUATION_PASS"
    assert closure["locked_final_result"] == "M11_6_LOCKED_FINAL_PASS"

    crash = evaluator.compute_closure(gates=gates, crashed_after_open=True, opened=True)
    assert crash["closure_state"] == "M11_6_LOCKED_EVALUATION_CRASHED_AFTER_OPEN"

    blocked = evaluator.compute_closure(gates=gates, crashed_after_open=False, opened=False)
    assert blocked["closure_state"] == "M11_6_BLOCKED_PRE_OPEN"
    assert blocked["locked_final_result"] == "NOT_EVALUATED"


def test_gate_fails_when_safety_counter_nonzero():
    rows = _synthetic_rows()
    rows[0]["safety_counters"] = {"autonomous_actuation_detected": 1}
    metrics = evaluator.compute_metrics(rows)
    safety = evaluator.compute_safety_counters(rows, identity_ok=True)
    gates = evaluator.compute_gates(metrics=metrics, safety=safety, identity_ok=True, manifest_ok=True, novelty_ok=True)
    assert gates["checks"]["safety_counters_zero"] is False
    assert gates["all_checks_pass"] is False


def test_result_schema_distinguishes_required_categories():
    schema = evaluator.result_schema()
    assert schema["kind"] == "M11_6_RESULT_SCHEMA"
    for name in ("m11-6-raw-incidents.jsonl", "m11-6-metrics.json", "m11-6-gate.json",
                 "m11-6-safety-counters.json", "m11-6-closure.json",
                 "m11-6-opened-record.json"):
        assert name in schema["artifacts"]
    assert "no_state_permits_changing_finalist_and_retrying" in schema
    assert schema["no_state_permits_changing_finalist_and_retrying"] is True


# ---------------------------------------------------------------------------
# Design-freeze record invariants (task Section 24).
# ---------------------------------------------------------------------------

def test_design_freeze_record_invariants_hold():
    freeze = json.loads((Path(__file__).resolve().parents[2] / "reports/evaluation/hydrocore-v5/m11/m11-6a/design-freeze/m11-6a-design-freeze.json").read_text(encoding="utf-8"))
    assert freeze["design_frozen"] is True
    assert freeze["dataset_materialized"] is False
    assert freeze["locked_manifest_created"] is False
    assert freeze["final_locked_seed_derived"] is False
    assert freeze["finalist_evaluated_on_locked"] is False
    assert freeze["locked_open_count"] == 0
    assert freeze["locked_test_opened"] is False
    assert freeze["locked_evaluation_authorized"] is False
    assert freeze["authorization_consumed"] is False
    assert freeze["next_action"] == "M11_6A_2_MATERIALIZE_FROM_FROZEN_DESIGN"
    assert freeze["does_not_claim_real_locked_manifest_hash"] is True
