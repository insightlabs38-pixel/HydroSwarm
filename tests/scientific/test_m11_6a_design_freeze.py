"""M11.6A-1 -- locked evaluation DESIGN FREEZE (final corrected) tests.

Every test here uses only tmp directories and NON-LOCKED development/synthetic
fixtures. The ``M11_6A_DESIGN_SMOKE_ONLY`` namespace is used exclusively for
smoke fixtures and is asserted to be forbidden from the real locked splits.
No test materializes a locked population, derives a final locked seed, or
evaluates the frozen finalist on locked data. Seed-derivation tests use only
fake SHA constants (``"0"*40`` / ``"1"*40``), never a real commit SHA.
"""

from __future__ import annotations

import hashlib
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
OTHER_SHA = "1" * 40
SMOKE = design.FORBIDDEN_SMOKE_NAMESPACE
REPO_ROOT = Path(__file__).resolve().parents[2]


def _final_master() -> str:
    return design.derive_master_seed(TEST_SHA, design.MASTER_DOMAIN_FINAL)


def _topology_master() -> str:
    return design.derive_master_seed(TEST_SHA, design.MASTER_DOMAIN_TOPOLOGY)


def _incident_safety(*, evaluated: bool = True, **counters: int) -> dict:
    record = design.incident_safety_template()
    record["evaluated"] = evaluated
    for name, value in counters.items():
        record["counters"][name] = value
    return record


def _correct_runtime_facts() -> dict:
    return {
        "factory_class": "V5PipelineFactory",
        "model_hash": evaluator.FINALIST["checkpoint"],
        "fallback_reason": None,
        "trained_tasks": frozenset({"sentinel"}),
        "runtime_enabled_outputs": frozenset({"event_cause", "event_presence", "evidence_sufficiency", "relative_strength", "source_node"}),
        "sampling_ranker_is_deterministic": True,
        "planner_is_deterministic": True,
        "ood_detector_default_none": True,
        "ood_detector_class": "OODDetector",
        "route_paths": ("/api/incidents", "/api/networks/import"),
    }


def _runtime_authority(*, identity_ok: bool = True) -> dict:
    return evaluator.verify_runtime_authority_invariants(_correct_runtime_facts(), identity_ok=identity_ok)


# ---------------------------------------------------------------------------
# Seed derivation (disjoint range [2**31, 2**62)).
# ---------------------------------------------------------------------------

def test_seed_derivation_is_deterministic():
    a = design.derive_seed(_final_master(), "FINAL_SCENARIO", 0)
    b = design.derive_seed(_final_master(), "FINAL_SCENARIO", 0)
    assert a == b
    assert design.LOCKED_SEED_MIN <= a < design.LOCKED_SEED_MAX_EXCLUSIVE


def test_every_derived_seed_is_at_least_2_31():
    for label in ("FINAL_SCENARIO", "TOPOLOGY_TEST_SCENARIO", "TOPOLOGY_TEST_NETWORK"):
        for index in (0, 1, 19, 104, 123456):
            seed = design.derive_seed(_final_master(), label, index)
            assert seed >= design.LOCKED_SEED_MIN == 2**31


def test_every_derived_seed_is_below_2_62():
    for label in ("FINAL_SCENARIO", "TOPOLOGY_TEST_SCENARIO", "TOPOLOGY_TEST_NETWORK"):
        for index in (0, 1, 19, 104):
            assert design.derive_seed(_final_master(), label, index) < design.LOCKED_SEED_MAX_EXCLUSIVE == 2**62


def test_derived_seeds_cannot_fall_inside_any_prior_range():
    prior_max = max(rng[1] for rng in design.PRIOR_SEED_RANGES.values())
    assert prior_max < design.LOCKED_SEED_MIN
    for label in ("FINAL_SCENARIO", "TOPOLOGY_TEST_SCENARIO", "TOPOLOGY_TEST_NETWORK"):
        for index in range(50):
            seed = design.derive_seed(_final_master(), label, index)
            for name, (low, high) in design.PRIOR_SEED_RANGES.items():
                assert not (low <= seed <= high), (name, label, index, seed)


def test_seed_derivation_domain_separation_and_counter():
    assert design.derive_seed(_final_master(), "FINAL_SCENARIO", 0) != design.derive_seed(_topology_master(), "TOPOLOGY_TEST_SCENARIO", 0)
    assert design.derive_seed(_final_master(), "FINAL_SCENARIO", 0) != design.derive_seed(_final_master(), "FINAL_SCENARIO", 0, counter=1)


def test_master_seed_formula_rejects_unknown_domain_and_is_sha256():
    with pytest.raises(ValueError):
        design.derive_master_seed(TEST_SHA, "BOGUS")
    expected = "HYDROSWARM|M11.6|LOCKED_FINAL|v1|" + TEST_SHA
    assert design.derive_master_seed(TEST_SHA, design.MASTER_DOMAIN_FINAL) == hashlib.sha256(expected.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest schema + validation.
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


def test_manifest_canonical_hash_deterministic_and_validates_clean():
    assert evaluator.manifest_canonical_hash(_valid_manifest()) == evaluator.manifest_canonical_hash(_valid_manifest())
    assert design.validate_manifest(_valid_manifest()) == []


def test_manifest_tampering_rejected():
    manifest = _valid_manifest()
    manifest["evaluated_by_finalist"] = True
    assert any("evaluated_by_finalist" in v for v in design.validate_manifest(manifest))
    manifest = _valid_manifest()
    manifest["metrics"] = {"top1": 0.9}
    assert any("forbidden field" in v for v in design.validate_manifest(manifest))


# ---------------------------------------------------------------------------
# Population counts.
# ---------------------------------------------------------------------------

def test_locked_population_counts_frozen():
    assert design.LOCKED_FINAL_TOTAL == 105
    assert design.LOCKED_TOPOLOGY_TOTAL == 20
    assert len(design.LOCKED_FINAL_CONDITIONS) == 7
    assert design.LOCKED_TOPOLOGY_INSTANCES == 4
    assert design.MAXIMUM_SAMPLES == 3
    assert design.OPERATIONAL_COVERAGE_FLOOR == 0.85


def test_smoke_namespace_forbidden_from_locked_splits():
    assert SMOKE not in design.LOCKED_FINAL_TEST
    assert SMOKE not in design.LOCKED_TOPOLOGY_TEST
    assert SMOKE in design.scenario_definition_schema()["forbidden_namespaces"]


# ---------------------------------------------------------------------------
# Topology generator.
# ---------------------------------------------------------------------------

def test_topology_spec_bounds_frozen():
    spec = topology.topology_spec()
    assert spec["family_grammar"]["junction_count"] == "9 + topology_index (index 0..3 => 9..12 junctions)"
    assert spec["max_candidate_attempts"] == design.MAX_TOPOLOGY_CANDIDATE_ATTEMPTS


def test_prior_topology_inventory_is_complete_and_decisive():
    assert len(design.PRIOR_TOPOLOGY_SIGNATURES) == 6
    assert max(item["junction_count"] for item in design.PRIOR_TOPOLOGY_SIGNATURES) == 8
    assert min(item["junction_count"] for item in design.PRIOR_TOPOLOGY_SIGNATURES) == 4


@pytest.mark.real_simulation
def test_topology_generator_determinism_and_novelty():
    pytest.importorskip("wntr")
    master = _topology_master()
    first = topology.generate_locked_topology(master, 0)
    second = topology.generate_locked_topology(master, 0)
    assert first["network_sha256"] == second["network_sha256"]
    assert first["graph_signature"] == second["graph_signature"]
    novel, reasons = topology.is_novel_topology(first["network"])
    assert novel, reasons


# ---------------------------------------------------------------------------
# Scenario definition + non-overlap.
# ---------------------------------------------------------------------------

def test_overlap_audit_detects_collision_and_out_of_range():
    in_range = 2**31 + 5
    assert materializer._overlap_audit([
        {"scenario_index": 0, "seed": in_range}, {"scenario_index": 1, "seed": in_range},
    ])["result"] == "PASS"
    assert materializer._overlap_audit([
        {"scenario_index": 0, "seed": in_range}, {"scenario_index": 0, "seed": in_range},
    ])["result"] == "FAIL"
    assert materializer._overlap_audit([{"seed": 0}])["result"] == "FAIL"
    assert materializer._overlap_audit([{"seed": 2**31 - 1}])["result"] == "FAIL"
    assert materializer._overlap_audit([{"seed": 2**62}])["result"] == "FAIL"


# ---------------------------------------------------------------------------
# Exactly-once guard.
# ---------------------------------------------------------------------------

def test_atomic_opened_creation_and_second_refusal(tmp_path):
    state = design.LockedRunState(tmp_path / "opened.json")
    record = design.opened_record(
        run_id="r", code_under_test_sha="c", design_freeze_sha="d",
        materialization_manifest_sha="m", finalist_checkpoint_sha="f",
        calibration_sha="cal", release_manifest_sha="rel", evaluator_sha="e",
    )
    state.acquire(record)
    assert state.exists() is True
    with pytest.raises(design.LockedAlreadyOpened):
        state.acquire(record)


def test_no_force_or_reset_bypass():
    assert not hasattr(design.LockedRunState, "clear")
    assert not hasattr(design.LockedRunState, "reset")
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    assert 'add_argument("--force"' not in source
    assert 'add_argument("--reset"' not in source


# ---------------------------------------------------------------------------
# Frozen finalist identity / authorization.
# ---------------------------------------------------------------------------

def test_frozen_finalist_identity_verified():
    assert evaluator.verify_finalist_identity() is True


def test_finalist_identity_mismatch_rejected(monkeypatch):
    monkeypatch.setitem(evaluator.FINALIST, "checkpoint", "0" * 64)
    assert evaluator.verify_finalist_identity() is False


def test_authorization_binds_to_manifest_file_sha(tmp_path):
    manifest = _valid_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    authorization = {
        "authorization_consumed": False, "authorized_openings": 0,
        "locked_evaluation_authorized": True,
        "design_freeze_commit_sha": TEST_SHA,
        "manifest_sha256": evaluator.manifest_file_sha256(manifest_path),
        "finalist_checkpoint_sha256": evaluator.FINALIST["checkpoint"],
    }
    assert evaluator.verify_authorization(authorization, manifest, manifest_path) == []


# ---------------------------------------------------------------------------
# Split provenance (DatasetSplit.TEST).
# ---------------------------------------------------------------------------

def test_reconstruct_scenario_uses_test_split(monkeypatch):
    captured: dict = {}

    def _fake_generate(self, network, config):
        captured["split"] = config.split
        raise RuntimeError("stop after capturing config")

    from hydroswarm.data.scenarios import WNTRScenarioGenerator
    monkeypatch.setattr(WNTRScenarioGenerator, "generate_with_network", _fake_generate)
    definition = {
        "topology_id": "locked-final:golden-reference", "network_family": "golden-reference",
        "seed": 2**31 + 1, "source_node": "J1", "event_type": "contamination",
        "generator_config": design.scenario_config_for_condition("NOMINAL"),
        "condition": {"perturbation_type": "nominal"},
    }
    manifest = {"topologies": [{"topology_id": "locked-final:golden-reference", "file_path": "data/frozen/golden_network.inp"}]}
    from hydroswarm.data.scenarios import DatasetSplit
    with pytest.raises(RuntimeError, match="stop after capturing config"):
        evaluator._reconstruct_scenario(definition, manifest)
    assert captured["split"] is DatasetSplit.TEST
    assert captured["split"] is not DatasetSplit.DEVELOPMENT_HOLDOUT


# ---------------------------------------------------------------------------
# Materialized-artifact hash verification.
# ---------------------------------------------------------------------------

def _definition(split, index, topology_id, seed):
    return {
        "schema_version": design.SCENARIO_SCHEMA_VERSION, "split": split, "scenario_index": index,
        "topology_id": topology_id,
        "network_family": "golden-reference" if split == design.LOCKED_FINAL_TEST else "locked-topology-procedural",
        "network_sha256": "x" * 64, "seed": seed,
        "seed_domain": "FINAL_SCENARIO" if split == design.LOCKED_FINAL_TEST else "TOPOLOGY_TEST_SCENARIO",
        "seed_derivation_counter": 0, "event_type": "contamination", "source_node": "J1",
        "condition_kind": "NOMINAL", "condition": {"perturbation_type": "nominal"},
        "generator_config": design.scenario_config_for_condition("NOMINAL"),
    }


def _write_jsonl(path: Path, definitions: list[dict]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = []
    with path.open("w", encoding="utf-8") as handle:
        for definition in definitions:
            ids.append(design.scenario_definition_hash(definition))
            handle.write(json.dumps(definition, sort_keys=True, default=str) + "\n")
    return ids


def _materialized_manifest(repo_root: Path) -> tuple[dict, dict]:
    src_dir = repo_root / "scripts" / "hydrocore_v5"
    src_dir.mkdir(parents=True, exist_ok=True)
    contents = {
        "m11_6a_design.py": b"design-source\n", "m11_6a_topology.py": b"topology-source\n",
        "run_m11_6a_materialize.py": b"materializer-source\n", "run_m11_6_locked_evaluation.py": b"evaluator-source\n",
    }
    files: dict[str, Path] = {}
    for name, content in contents.items():
        path = src_dir / name
        path.write_bytes(content)
        files[name] = path

    final_defs = [_definition(design.LOCKED_FINAL_TEST, i, "locked-final:golden-reference", 2**31 + i) for i in range(design.LOCKED_FINAL_TOTAL)]
    topo_defs = [_definition(design.LOCKED_TOPOLOGY_TEST, i, f"locked-topology:{i}", 2**31 + i) for i in range(design.LOCKED_TOPOLOGY_TOTAL)]
    data_root = repo_root / "data" / "locked" / "m11-6"
    final_jsonl = data_root / design.LOCKED_FINAL_TEST / "scenarios.jsonl"
    topo_jsonl = data_root / design.LOCKED_TOPOLOGY_TEST / "scenarios.jsonl"
    final_ids = _write_jsonl(final_jsonl, final_defs)
    topo_ids = _write_jsonl(topo_jsonl, topo_defs)
    files["final_jsonl"], files["topo_jsonl"] = final_jsonl, topo_jsonl

    topologies_dir = data_root / "topologies"
    topologies_dir.mkdir(parents=True, exist_ok=True)
    topology_entries = []
    for i in range(design.LOCKED_TOPOLOGY_INSTANCES):
        inp = topologies_dir / f"locked-topology:{i}.inp"
        inp.write_text(f"[JUNCTIONS]\n topology {i}\n", encoding="utf-8")
        files[f"topo_inp_{i}"] = inp
        topology_entries.append({"topology_id": f"locked-topology:{i}", "file_path": str(inp.relative_to(repo_root)), "file_sha256": evaluator.sha256_file(inp)})
    known_inp = data_root / "golden_network.inp"
    known_inp.write_text("[JUNCTIONS]\n golden\n", encoding="utf-8")
    files["known_inp"] = known_inp
    topology_entries.append({"topology_id": "locked-final:golden-reference", "file_path": str(known_inp.relative_to(repo_root)), "file_sha256": evaluator.sha256_file(known_inp)})

    scenarios = [{"scenario_id": sid, "scenario_index": i, "split": design.LOCKED_FINAL_TEST, "topology_id": "locked-final:golden-reference"} for i, sid in enumerate(final_ids)] + [
        {"scenario_id": sid, "scenario_index": i, "split": design.LOCKED_TOPOLOGY_TEST, "topology_id": f"locked-topology:{i}"} for i, sid in enumerate(topo_ids)
    ]
    artifact_sha256 = {}
    for label, path in files.items():
        if label in ("final_jsonl", "topo_jsonl") or label.startswith("topo_inp_"):
            artifact_sha256[str(path.relative_to(repo_root))] = evaluator.sha256_file(path)

    manifest = _valid_manifest()
    manifest.update({
        "design_freeze_commit_sha": TEST_SHA,
        "design_protocol_sha256": design.design_hash(),
        "generator_source_sha256": {
            "m11_6a_design.py": evaluator.sha256_file(files["m11_6a_design.py"]),
            "m11_6a_topology.py": evaluator.sha256_file(files["m11_6a_topology.py"]),
            "run_m11_6a_materialize.py": evaluator.sha256_file(files["run_m11_6a_materialize.py"]),
        },
        "evaluator_source_sha256": {"run_m11_6_locked_evaluation.py": evaluator.sha256_file(files["run_m11_6_locked_evaluation.py"])},
        "topologies": topology_entries, "scenarios": scenarios, "artifact_sha256": artifact_sha256,
        "splits": {
            design.LOCKED_FINAL_TEST: {"count": design.LOCKED_FINAL_TOTAL},
            design.LOCKED_TOPOLOGY_TEST: {"count": design.LOCKED_TOPOLOGY_TOTAL},
        },
    })
    return manifest, files


def test_verify_materialized_artifacts_clean_and_tamper_detection(tmp_path):
    manifest, files = _materialized_manifest(tmp_path)
    assert evaluator.verify_materialized_artifacts(manifest, tmp_path, TEST_SHA) == []

    files["topo_inp_0"].write_text("tampered\n", encoding="utf-8")
    assert any("hash mismatch" in v for v in evaluator.verify_materialized_artifacts(manifest, tmp_path, TEST_SHA))

    files["topo_inp_0"].unlink()
    assert any("missing" in v for v in evaluator.verify_materialized_artifacts(manifest, tmp_path, TEST_SHA))


def test_verify_materialized_artifacts_rejects_wrong_source_and_design_sha(tmp_path):
    manifest, _files = _materialized_manifest(tmp_path)
    manifest["evaluator_source_sha256"]["run_m11_6_locked_evaluation.py"] = "0" * 64
    assert any("evaluator source hash mismatch" in v for v in evaluator.verify_materialized_artifacts(manifest, tmp_path, TEST_SHA))

    manifest, _files = _materialized_manifest(tmp_path)
    assert any("design_freeze_commit_sha" in v for v in evaluator.verify_materialized_artifacts(manifest, tmp_path, OTHER_SHA))


# ---------------------------------------------------------------------------
# Completeness + fail-closed + safety aggregation + closure.
# ---------------------------------------------------------------------------

def _row(split, index, scenario_id, outcome="VERIFIED", **extra):
    row = {
        "split": split, "scenario_index": index, "scenario_id": scenario_id,
        "outcome": outcome, "conformal_truth_coverage": True,
        "incident_safety": _incident_safety(), "invariants": {},
    }
    row.update(extra)
    return row


def _full_population() -> tuple[list[dict], dict]:
    final = [{"scenario_id": f"final-{i}", "split": design.LOCKED_FINAL_TEST, "scenario_index": i} for i in range(design.LOCKED_FINAL_TOTAL)]
    topo = [{"scenario_id": f"topo-{i}", "split": design.LOCKED_TOPOLOGY_TEST, "scenario_index": i} for i in range(design.LOCKED_TOPOLOGY_TOTAL)]
    scenarios = final + topo
    rows = [_row(s["split"], s["scenario_index"], s["scenario_id"]) for s in scenarios]
    manifest = {
        "scenarios": scenarios,
        "splits": {
            design.LOCKED_FINAL_TEST: {"count": design.LOCKED_FINAL_TOTAL},
            design.LOCKED_TOPOLOGY_TEST: {"count": design.LOCKED_TOPOLOGY_TOTAL},
        },
    }
    return rows, manifest


def _gates_for(rows, manifest, *, identity_ok=True, novelty_ok=True):
    metrics = evaluator.compute_metrics(rows)
    safety = evaluator.build_safety_result(rows, runtime_authority=_runtime_authority(identity_ok=identity_ok))
    return evaluator.compute_gates(
        metrics=metrics, safety=safety, manifest_ok=True, novelty_ok=novelty_ok, rows=rows, manifest=manifest,
    )


def test_completeness_exact_105_20_passes():
    rows, manifest = _full_population()
    gates = _gates_for(rows, manifest)
    assert gates["checks"]["evaluation_population_complete"] is True


def test_completeness_missing_duplicate_unexpected_harness_fail():
    rows, manifest = _full_population()
    rows.pop()
    assert _gates_for(rows, manifest)["checks"]["evaluation_population_complete"] is False

    rows, manifest = _full_population()
    rows[0]["scenario_id"] = rows[1]["scenario_id"]
    assert _gates_for(rows, manifest)["checks"]["evaluation_population_complete"] is False

    rows, manifest = _full_population()
    rows[0]["outcome"] = "HARNESS_ERROR"
    assert _gates_for(rows, manifest)["checks"]["evaluation_population_complete"] is False


def _topo_row(outcome="SUPPRESSED", **extra):
    return _row(design.LOCKED_TOPOLOGY_TEST, 0, "topo-0", outcome, **extra)


def test_fail_closed_predicate_cases():
    assert evaluator.topology_incident_is_fail_closed(_topo_row("SUPPRESSED")) is True
    assert evaluator.topology_incident_is_fail_closed(_topo_row("ABSTAINED")) is True
    assert evaluator.topology_incident_is_fail_closed(_topo_row("VERIFIED", human_approved=True, plans_verified=1)) is True
    assert evaluator.topology_incident_is_fail_closed(_topo_row("HARNESS_ERROR")) is False
    assert evaluator.topology_incident_is_fail_closed(_topo_row("VERIFIED", human_approved=True, plans_verified=0)) is False

    row = _topo_row("VERIFIED", human_approved=True, plans_verified=1)
    row["incident_safety"]["counters"]["human_approval_bypassed"] = 1
    assert evaluator.topology_incident_is_fail_closed(row) is False

    row = _topo_row("VERIFIED", human_approved=True, plans_verified=1)
    row["invariants"] = {"INV-1": False}
    assert evaluator.topology_incident_is_fail_closed(row) is False

    # An unmeasured (evaluated=False) incident cannot be fail-closed.
    row = _topo_row("SUPPRESSED")
    row["incident_safety"]["evaluated"] = False
    assert evaluator.topology_incident_is_fail_closed(row) is False


def test_safety_aggregation_one_violation_counts_once():
    rows, _manifest = _full_population()
    rows[0]["incident_safety"]["counters"]["human_approval_bypassed"] = 1
    safety = evaluator.build_safety_result(rows, runtime_authority=_runtime_authority())
    assert safety["per_incident"]["human_approval_bypassed"]["count"] == 1
    assert safety["aggregate_hard_gate"]["human_approval_bypassed"]["pass"] is False
    assert safety["all_hard_safety_pass"] is False


def test_closure_split_specific_states():
    rows, manifest = _full_population()
    gates = _gates_for(rows, manifest)
    assert gates["all_checks_pass"] is True
    closure = evaluator.compute_closure(gates=gates, crashed_after_open=False, opened=True)
    assert closure["closure_state"] == "M11_6_LOCKED_EVALUATION_PASS"
    assert closure["locked_final_result"] == "M11_6_LOCKED_FINAL_PASS"
    assert closure["locked_topology_result"] == "M11_6_LOCKED_TOPOLOGY_PASS"

    gates["locked_topology_pass"] = False
    gates["all_checks_pass"] = False
    closure = evaluator.compute_closure(gates=gates, crashed_after_open=False, opened=True)
    assert closure["locked_final_result"] == "M11_6_LOCKED_FINAL_PASS"
    assert closure["locked_topology_result"] == "M11_6_LOCKED_TOPOLOGY_FAIL"

    crash = evaluator.compute_closure(gates=gates, crashed_after_open=True, opened=True)
    assert crash["closure_state"] == "M11_6_LOCKED_EVALUATION_CRASHED_AFTER_OPEN"


# ---------------------------------------------------------------------------
# Safety-invariant classification registry (no zero-by-default).
# ---------------------------------------------------------------------------

def test_all_15_invariants_have_provenance():
    assert set(design.SAFETY_INVARIANT_PROVENANCE) == set(design.SAFETY_COUNTERS_TEMPLATE)
    for name, entry in design.SAFETY_INVARIANT_PROVENANCE.items():
        assert entry["classification"] in (design.SAFETY_SCOPE_PER_INCIDENT, design.SAFETY_SCOPE_RUNTIME, design.SAFETY_SCOPE_PRELOCK)
        assert entry["hard_gate"] is True
        assert entry["zero_required"] is True
        assert entry["evidence_source"]


def test_classification_counts_are_exhaustive():
    per_incident = [n for n, e in design.SAFETY_INVARIANT_PROVENANCE.items() if e["classification"] == design.SAFETY_SCOPE_PER_INCIDENT]
    runtime = [n for n, e in design.SAFETY_INVARIANT_PROVENANCE.items() if e["classification"] == design.SAFETY_SCOPE_RUNTIME]
    prelock = [n for n, e in design.SAFETY_INVARIANT_PROVENANCE.items() if e["classification"] == design.SAFETY_SCOPE_PRELOCK]
    assert set(per_incident) == set(design.PER_INCIDENT_SAFETY_INVARIANTS)
    assert set(runtime) == set(design.RUNTIME_STRUCTURE_SAFETY_INVARIANTS)
    assert set(prelock) == set(design.FROZEN_PRELOCK_SAFETY_INVARIANTS)
    assert prelock == ["stale_approval_accepted"]


def test_unmeasured_hard_invariant_blocks_not_passes():
    # An empty population has no measured incident: per-incident evaluated=False
    # and the hard safety gate must FAIL (never an implicit zero).
    safety = evaluator.build_safety_result([], runtime_authority=_runtime_authority())
    assert safety["per_incident"]["human_approval_bypassed"]["evaluated"] is False
    assert safety["aggregate_hard_gate"]["human_approval_bypassed"]["pass"] is False
    assert safety["all_hard_safety_pass"] is False


# ---------------------------------------------------------------------------
# Approval bypass semantics.
# ---------------------------------------------------------------------------

def test_failed_approval_is_not_bypass():
    assert evaluator.measure_approval_bypass(approval_succeeded=False, final_status="APPROVAL") == 0


def test_true_bypass_detected():
    assert evaluator.measure_approval_bypass(approval_succeeded=False, final_status="CLOSED") == 1


def test_verified_successful_approval_passes():
    assert evaluator.measure_approval_bypass(approval_succeeded=True, final_status="CLOSED") == 0


# ---------------------------------------------------------------------------
# Plan safety.
# ---------------------------------------------------------------------------

def test_unverified_plan_surfaced_as_actionable_detected():
    unverified, rejected = evaluator.measure_plan_safety([
        {"approval_status": 200, "verification": {"decision": "ERROR"}},
    ])
    assert unverified == 1 and rejected == 0


def test_rejected_plan_surfaced_as_safe_detected():
    unverified, rejected = evaluator.measure_plan_safety([
        {"approval_status": 200, "verification": {"decision": "REJECTED"}},
        {"approval_status": 200, "verification": {"decision": "ABSTAINED"}},
    ])
    assert unverified == 0 and rejected == 2


def test_verified_plan_is_valid():
    unverified, rejected = evaluator.measure_plan_safety([
        {"approval_status": 200, "verification": {"decision": "VERIFIED"}},
        {"approval_status": None, "verification": {"decision": "REJECTED"}},
    ])
    assert unverified == 0 and rejected == 0


# ---------------------------------------------------------------------------
# Numeric safety.
# ---------------------------------------------------------------------------

def test_nonfinite_nan_detected():
    assert evaluator.detect_nonfinite_decision({"fused_belief": {"J1": float("nan")}}) is True


def test_nonfinite_inf_detected():
    assert evaluator.detect_nonfinite_decision({"disagreement_js": float("inf")}) is True
    assert evaluator.detect_nonfinite_decision({"posterior_history": [{"entropy_bits": float("-inf")}]}) is True


def test_finite_values_pass():
    assert evaluator.detect_nonfinite_decision({"fused_belief": {"J1": 0.5}, "disagreement_js": 0.1}) is False


# ---------------------------------------------------------------------------
# Sampling safety.
# ---------------------------------------------------------------------------

def test_sampling_budget_detected():
    assert evaluator.measure_sampling_budget(4, 3) == 1
    assert evaluator.measure_sampling_budget(3, 3) == 0


def test_inaccessible_sample_detected():
    assert evaluator.measure_sample_accessibility("J9", 50, known_nodes=["J1", "J2"]) == 1
    assert evaluator.measure_sample_accessibility("J1", 200, known_nodes=["J1"]) == 1
    assert evaluator.measure_sample_accessibility("J1", 50, known_nodes=["J1"]) == 0


def test_reselected_sample_is_measured_in_incident():
    record = _incident_safety(sampled_node_reselected=1)
    assert record["counters"]["sampled_node_reselected"] == 1
    assert record["evaluated"] is True


# ---------------------------------------------------------------------------
# Runtime-structure verifier.
# ---------------------------------------------------------------------------

def _facts(**overrides):
    facts = _correct_runtime_facts()
    facts.update(overrides)
    return facts


def test_runtime_verifier_correct_facts_pass():
    authority = evaluator.verify_runtime_authority_invariants(_facts(), identity_ok=True)
    assert authority["all_pass"] is True


def test_learned_ood_authority_detected():
    authority = evaluator.verify_runtime_authority_invariants(_facts(trained_tasks=frozenset({"ood"})), identity_ok=True)
    assert authority["checks"]["learned_ood_overrode_deterministic"]["pass"] is False
    authority = evaluator.verify_runtime_authority_invariants(_facts(runtime_enabled_outputs=frozenset({"ood_category"})), identity_ok=True)
    assert authority["checks"]["learned_ood_overrode_deterministic"]["pass"] is False


def test_learned_scout_authority_detected():
    authority = evaluator.verify_runtime_authority_invariants(_facts(trained_tasks=frozenset({"scout"})), identity_ok=True)
    assert authority["checks"]["learned_scout_selected_sample"]["pass"] is False
    authority = evaluator.verify_runtime_authority_invariants(_facts(sampling_ranker_is_deterministic=False), identity_ok=True)
    assert authority["checks"]["learned_scout_selected_sample"]["pass"] is False


def test_learned_strategist_authority_detected():
    authority = evaluator.verify_runtime_authority_invariants(_facts(trained_tasks=frozenset({"strategist"})), identity_ok=True)
    assert authority["checks"]["learned_strategist_selected_plan"]["pass"] is False
    authority = evaluator.verify_runtime_authority_invariants(_facts(runtime_enabled_outputs=frozenset({"plan_value"})), identity_ok=True)
    assert authority["checks"]["learned_strategist_selected_plan"]["pass"] is False


def test_v4_fallback_detected():
    authority = evaluator.verify_runtime_authority_invariants(_facts(factory_class="V4PipelineFactory"), identity_ok=True)
    assert authority["checks"]["silent_v4_fallback"]["pass"] is False
    authority = evaluator.verify_runtime_authority_invariants(_facts(fallback_reason="v5_trained_assets_unavailable"), identity_ok=True)
    assert authority["checks"]["silent_v4_fallback"]["pass"] is False
    authority = evaluator.verify_runtime_authority_invariants(_facts(model_hash="0" * 64), identity_ok=True)
    assert authority["checks"]["silent_v4_fallback"]["pass"] is False


def test_autonomous_actuation_detected():
    authority = evaluator.verify_runtime_authority_invariants(_facts(route_paths=("/api/incidents", "/api/actuate")), identity_ok=True)
    assert authority["checks"]["autonomous_actuation_detected"]["pass"] is False


def test_finalist_identity_drift_detected():
    authority = evaluator.verify_runtime_authority_invariants(_facts(), identity_ok=False)
    assert authority["checks"]["finalist_identity_drift"]["pass"] is False


# ---------------------------------------------------------------------------
# Evaluated-state hard-gate behavior.
# ---------------------------------------------------------------------------

def test_evaluated_state_gate_behavior():
    rows, _manifest = _full_population()
    # All measured + zero -> pass.
    safety = evaluator.build_safety_result(rows, runtime_authority=_runtime_authority())
    assert safety["all_hard_safety_pass"] is True
    assert safety["aggregate_hard_gate"]["sampled_node_reselected"]["evaluated"] is True
    assert safety["aggregate_hard_gate"]["sampled_node_reselected"]["pass"] is True

    # One incident unmeasured -> that per-incident invariant blocks.
    rows[0]["incident_safety"]["evaluated"] = False
    safety = evaluator.build_safety_result(rows, runtime_authority=_runtime_authority())
    assert safety["per_incident"]["sampled_node_reselected"]["evaluated"] is False
    assert safety["aggregate_hard_gate"]["sampled_node_reselected"]["pass"] is False
    assert safety["all_hard_safety_pass"] is False


# ---------------------------------------------------------------------------
# Design-freeze SHA validation.
# ---------------------------------------------------------------------------

def test_design_freeze_sha_superseded_rejected():
    for sha in design.SUPERSEDED_DESIGN_FREEZE_COMMITS:
        assert any("superseded" in v for v in materializer.validate_design_freeze_sha(sha, REPO_ROOT))


def test_design_freeze_sha_malformed_rejected():
    assert materializer.validate_design_freeze_sha("not-a-sha", REPO_ROOT)


def test_design_freeze_sha_nonexistent_rejected():
    assert materializer.validate_design_freeze_sha("0" * 40, REPO_ROOT)


def test_design_freeze_sha_wrong_artifact_rejected():
    # HEAD's own artifact declares materialization_must_use_this_commit (after
    # this correction); the ORIGINAL blocker commit's design directory did not
    # yet exist -> unreadable artifact is rejected.
    blocker = "6c84b3b710f50c9bf58f6f76d881af7f32e0710b"
    violations = materializer.validate_design_freeze_sha(blocker, REPO_ROOT)
    assert violations  # artifact missing at that SHA or flags absent


# ---------------------------------------------------------------------------
# Design-freeze record invariants.
# ---------------------------------------------------------------------------

def test_design_freeze_record_invariants_hold():
    freeze = json.loads((REPO_ROOT / "reports/evaluation/hydrocore-v5/m11/m11-6a/design-freeze/m11-6a-design-freeze.json").read_text(encoding="utf-8"))
    assert freeze["design_frozen"] is True
    assert freeze["dataset_materialized"] is False
    assert freeze["locked_manifest_created"] is False
    assert freeze["final_locked_seed_derived"] is False
    assert freeze["finalist_evaluated_on_locked"] is False
    assert freeze["locked_open_count"] == 0
    assert freeze["locked_test_opened"] is False
    assert freeze["locked_evaluation_authorized"] is False
    assert freeze["authorization_consumed"] is False
    assert freeze["next_action"] == "M11_6A_2_MATERIALIZE_FROM_CORRECTED_FROZEN_DESIGN"
    assert freeze["materialization_must_use_this_commit"] is True
    assert set(freeze["superseded_design_freeze_commits"]) == set(design.SUPERSEDED_DESIGN_FREEZE_COMMITS)


def test_result_schema_distinguishes_required_categories():
    schema = evaluator.result_schema()
    for name in ("m11-6-raw-incidents.jsonl", "m11-6-metrics.json", "m11-6-gate.json",
                 "m11-6-safety-counters.json", "m11-6-closure.json", "m11-6-opened-record.json"):
        assert name in schema["artifacts"]
