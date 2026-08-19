"""M11.6A-1 -- locked evaluation DESIGN FREEZE (corrected) tests.

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


def _final_master() -> str:
    return design.derive_master_seed(TEST_SHA, design.MASTER_DOMAIN_FINAL)


def _topology_master() -> str:
    return design.derive_master_seed(TEST_SHA, design.MASTER_DOMAIN_TOPOLOGY)


# ---------------------------------------------------------------------------
# Correction #1 -- seed derivation: disjoint range [2**31, 2**62).
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
            assert seed >= design.LOCKED_SEED_MIN == 2**31, (label, index, seed)


def test_every_derived_seed_is_below_2_62():
    for label in ("FINAL_SCENARIO", "TOPOLOGY_TEST_SCENARIO", "TOPOLOGY_TEST_NETWORK"):
        for index in (0, 1, 19, 104):
            seed = design.derive_seed(_final_master(), label, index)
            assert seed < design.LOCKED_SEED_MAX_EXCLUSIVE == 2**62, (label, index, seed)


def test_derived_seeds_cannot_fall_inside_any_prior_range():
    # Every prior namespace is < 2**31; every derived seed is >= 2**31.
    prior_max = max(rng[1] for rng in design.PRIOR_SEED_RANGES.values())
    assert prior_max < design.LOCKED_SEED_MIN
    for label in ("FINAL_SCENARIO", "TOPOLOGY_TEST_SCENARIO", "TOPOLOGY_TEST_NETWORK"):
        for index in range(50):
            seed = design.derive_seed(_final_master(), label, index)
            for name, (low, high) in design.PRIOR_SEED_RANGES.items():
                assert not (low <= seed <= high), (name, label, index, seed)


def test_seed_derivation_domain_separation():
    final = design.derive_seed(_final_master(), "FINAL_SCENARIO", 0)
    topo = design.derive_seed(_topology_master(), "TOPOLOGY_TEST_SCENARIO", 0)
    assert final != topo
    assert design.derive_seed(_final_master(), "FINAL_SCENARIO", 0) != design.derive_seed(_final_master(), "TOPOLOGY_TEST_SCENARIO", 0)


def test_seed_derivation_counter_changes_seed_and_stays_in_range():
    a = design.derive_seed(_final_master(), "FINAL_SCENARIO", 0, counter=0)
    b = design.derive_seed(_final_master(), "FINAL_SCENARIO", 0, counter=1)
    assert a != b
    assert design.LOCKED_SEED_MIN <= a < design.LOCKED_SEED_MAX_EXCLUSIVE
    assert design.LOCKED_SEED_MIN <= b < design.LOCKED_SEED_MAX_EXCLUSIVE


def test_master_seed_formula_rejects_unknown_domain():
    with pytest.raises(ValueError):
        design.derive_master_seed(TEST_SHA, "BOGUS")


def test_master_seed_formula_is_sha256_of_frozen_material():
    expected = "HYDROSWARM|M11.6|LOCKED_FINAL|v1|" + TEST_SHA
    assert design.derive_master_seed(TEST_SHA, design.MASTER_DOMAIN_FINAL) == hashlib.sha256(expected.encode("utf-8")).hexdigest()


def test_seed_derivation_spec_records_disjoint_range():
    spec = design.seed_derivation_spec()
    assert spec["locked_seed_min"] == 2**31
    assert spec["locked_seed_max_exclusive"] == 2**62
    assert spec["allowed_integer_range"] == [2**31, 2**62 - 1]
    assert spec["disjoint_by_construction"]


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


def test_manifest_canonical_hash_is_deterministic():
    manifest = _valid_manifest()
    assert evaluator.manifest_canonical_hash(manifest) == evaluator.manifest_canonical_hash(manifest)


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
    pytest.importorskip("wntr")
    seed = design.derive_seed(_topology_master(), "TOPOLOGY_TEST_NETWORK", 0, 0)
    model = topology.build_procedural_topology_model(seed, 9, 1)
    assert topology.topology_is_feasible(model) is True
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
        "seed": 2**31 + 123,
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
    # Every locked_final_test seed is in the disjoint [2**31, 2**62) range.
    for definition in a:
        assert design.LOCKED_SEED_MIN <= definition["seed"] < design.LOCKED_SEED_MAX_EXCLUSIVE


def test_overlap_audit_detects_collision():
    in_range = 2**31 + 5
    assert materializer._overlap_audit([
        {"scenario_index": 0, "seed": in_range}, {"scenario_index": 1, "seed": in_range},
    ])["result"] == "PASS"
    assert materializer._overlap_audit([
        {"scenario_index": 0, "seed": in_range}, {"scenario_index": 0, "seed": in_range},
    ])["result"] == "FAIL"
    # A seed below 2**31 (inside the prior namespace interval) is out of range.
    assert materializer._overlap_audit([{"seed": 0}])["result"] == "FAIL"
    assert materializer._overlap_audit([{"seed": 2**31 - 1}])["result"] == "FAIL"
    assert materializer._overlap_audit([{"seed": 2**62}])["result"] == "FAIL"


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


def test_authorization_absence_rejected(tmp_path):
    manifest = _valid_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    violations = evaluator.verify_authorization({}, manifest, manifest_path)
    assert any("authorization_consumed" in v for v in violations)
    assert any("locked_evaluation_authorized" in v for v in violations)


def test_authorization_binds_to_manifest_file_sha(tmp_path):
    manifest = _valid_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    file_sha = evaluator.manifest_file_sha256(manifest_path)
    authorization = {
        "authorization_consumed": False,
        "authorized_openings": 0,
        "locked_evaluation_authorized": True,
        "design_freeze_commit_sha": TEST_SHA,
        "manifest_sha256": file_sha,
        "finalist_checkpoint_sha256": evaluator.FINALIST["checkpoint"],
    }
    assert evaluator.verify_authorization(authorization, manifest, manifest_path) == []


def test_authorization_rejects_wrong_manifest_file_sha(tmp_path):
    manifest = _valid_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    authorization = {
        "authorization_consumed": False,
        "authorized_openings": 0,
        "locked_evaluation_authorized": True,
        "design_freeze_commit_sha": TEST_SHA,
        "manifest_sha256": "0" * 64,
        "finalist_checkpoint_sha256": evaluator.FINALIST["checkpoint"],
    }
    violations = evaluator.verify_authorization(authorization, manifest, manifest_path)
    assert any("manifest_sha256" in v for v in violations)


def test_manifest_file_sha_differs_from_canonical_hash(tmp_path):
    # Correction #8: the file-byte hash and the canonical-dict hash are distinct
    # values with distinct names; the file hash depends on serialization bytes.
    manifest = _valid_manifest()
    manifest_path = tmp_path / "manifest.json"
    payload_a = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(payload_a, encoding="utf-8")
    file_sha = evaluator.manifest_file_sha256(manifest_path)
    canonical = evaluator.manifest_canonical_hash(manifest)
    # The file hash is over the exact bytes; the canonical hash is over compact
    # sorted JSON. They are computed differently by construction.
    assert file_sha == hashlib.sha256(payload_a.encode("utf-8")).hexdigest()
    assert canonical != file_sha  # indent/whitespace differ from compact form


def test_locked_test_opened_true_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluator, "locked_test_opened", lambda _root: True)
    monkeypatch.setattr(evaluator, "OPENED_RECORD_PATH", tmp_path / "opened.json")
    with pytest.raises(RuntimeError, match="locked_test_opened"):
        evaluator.acquire_locked_open(
            authorization={}, manifest=_valid_manifest(), manifest_path=tmp_path / "manifest.json",
        )


# ---------------------------------------------------------------------------
# Correction #2 -- locked split provenance uses DatasetSplit.TEST.
# ---------------------------------------------------------------------------

def test_reconstruct_scenario_uses_test_split(monkeypatch):
    captured: dict = {}

    def _fake_generate(self, network, config):
        captured["split"] = config.split
        captured["split_value"] = config.split.value if hasattr(config.split, "value") else str(config.split)
        raise RuntimeError("stop after capturing config")

    from hydroswarm.data.scenarios import WNTRScenarioGenerator

    monkeypatch.setattr(WNTRScenarioGenerator, "generate_with_network", _fake_generate)
    definition = {
        "topology_id": "locked-final:golden-reference",
        "network_family": "golden-reference",
        "seed": 2**31 + 1,
        "source_node": "J1",
        "event_type": "contamination",
        "generator_config": design.scenario_config_for_condition("NOMINAL"),
        "condition": {"perturbation_type": "nominal"},
    }
    manifest = {
        "topologies": [{"topology_id": "locked-final:golden-reference",
                        "file_path": "data/frozen/golden_network.inp"}],
    }
    from hydroswarm.data.scenarios import DatasetSplit

    with pytest.raises(RuntimeError, match="stop after capturing config"):
        evaluator._reconstruct_scenario(definition, manifest)
    assert captured["split"] is DatasetSplit.TEST
    assert captured["split"] is not DatasetSplit.DEVELOPMENT_HOLDOUT


# ---------------------------------------------------------------------------
# Correction #4 -- real materialized-artifact hash verification.
# ---------------------------------------------------------------------------

def _definition(split, index, topology_id, seed):
    return {
        "schema_version": design.SCENARIO_SCHEMA_VERSION,
        "split": split,
        "scenario_index": index,
        "topology_id": topology_id,
        "network_family": "golden-reference" if split == design.LOCKED_FINAL_TEST else "locked-topology-procedural",
        "network_sha256": "x" * 64,
        "seed": seed,
        "seed_domain": "FINAL_SCENARIO" if split == design.LOCKED_FINAL_TEST else "TOPOLOGY_TEST_SCENARIO",
        "seed_derivation_counter": 0,
        "event_type": "contamination",
        "source_node": "J1",
        "condition_kind": "NOMINAL",
        "condition": {"perturbation_type": "nominal"},
        "generator_config": design.scenario_config_for_condition("NOMINAL"),
    }


def _write_definition_jsonl(jsonl_path: Path, definitions: list[dict]) -> list[str]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    ids = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for definition in definitions:
            ids.append(design.scenario_definition_hash(definition))
            handle.write(json.dumps(definition, sort_keys=True, default=str) + "\n")
    return ids


def _materialized_manifest(repo_root: Path) -> tuple[dict, dict]:
    """Create a tmp materialized tree and return (manifest, files) where files
    maps a label to its on-disk Path for later tampering."""
    src_dir = repo_root / "scripts" / "hydrocore_v5"
    src_dir.mkdir(parents=True, exist_ok=True)
    source_contents = {
        "m11_6a_design.py": b"design-source\n",
        "m11_6a_topology.py": b"topology-source\n",
        "run_m11_6a_materialize.py": b"materializer-source\n",
        "run_m11_6_locked_evaluation.py": b"evaluator-source\n",
    }
    files: dict[str, Path] = {}
    for name, content in source_contents.items():
        path = src_dir / name
        path.write_bytes(content)
        files[name] = path

    final_defs = [_definition(design.LOCKED_FINAL_TEST, i, "locked-final:golden-reference", 2**31 + i) for i in range(design.LOCKED_FINAL_TOTAL)]
    topo_defs = [_definition(design.LOCKED_TOPOLOGY_TEST, i, f"locked-topology:{i}", 2**31 + i) for i in range(design.LOCKED_TOPOLOGY_TOTAL)]
    data_root = repo_root / "data" / "locked" / "m11-6"
    final_jsonl = data_root / design.LOCKED_FINAL_TEST / "scenarios.jsonl"
    topo_jsonl = data_root / design.LOCKED_TOPOLOGY_TEST / "scenarios.jsonl"
    final_ids = _write_definition_jsonl(final_jsonl, final_defs)
    topo_ids = _write_definition_jsonl(topo_jsonl, topo_defs)
    files["final_jsonl"] = final_jsonl
    files["topo_jsonl"] = topo_jsonl

    topologies_dir = data_root / "topologies"
    topologies_dir.mkdir(parents=True, exist_ok=True)
    topology_entries = []
    for i in range(design.LOCKED_TOPOLOGY_INSTANCES):
        inp = topologies_dir / f"locked-topology:{i}.inp"
        inp.write_text(f"[JUNCTIONS]\n topology {i}\n", encoding="utf-8")
        files[f"topo_inp_{i}"] = inp
        topology_entries.append({
            "topology_id": f"locked-topology:{i}",
            "file_path": str(inp.relative_to(repo_root)),
            "file_sha256": evaluator.sha256_file(inp),
        })
    known_inp = data_root / "golden_network.inp"
    known_inp.write_text("[JUNCTIONS]\n golden\n", encoding="utf-8")
    files["known_inp"] = known_inp
    topology_entries.append({
        "topology_id": "locked-final:golden-reference",
        "file_path": str(known_inp.relative_to(repo_root)),
        "file_sha256": evaluator.sha256_file(known_inp),
    })

    scenarios = [
        {"scenario_id": sid, "scenario_index": i, "split": design.LOCKED_FINAL_TEST, "topology_id": "locked-final:golden-reference"}
        for i, sid in enumerate(final_ids)
    ] + [
        {"scenario_id": sid, "scenario_index": i, "split": design.LOCKED_TOPOLOGY_TEST, "topology_id": f"locked-topology:{i}"}
        for i, sid in enumerate(topo_ids)
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
        "topologies": topology_entries,
        "scenarios": scenarios,
        "artifact_sha256": artifact_sha256,
        "splits": {
            design.LOCKED_FINAL_TEST: {"count": design.LOCKED_FINAL_TOTAL},
            design.LOCKED_TOPOLOGY_TEST: {"count": design.LOCKED_TOPOLOGY_TOTAL},
        },
    })
    return manifest, files


def _expected_sha() -> str:
    return TEST_SHA


def test_verify_materialized_artifacts_clean(tmp_path):
    manifest, _files = _materialized_manifest(tmp_path)
    assert evaluator.verify_materialized_artifacts(manifest, tmp_path, _expected_sha()) == []


def test_verify_materialized_artifacts_detects_scenario_jsonl_tamper(tmp_path):
    manifest, files = _materialized_manifest(tmp_path)
    with files["final_jsonl"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_definition(design.LOCKED_FINAL_TEST, 999, "locked-final:golden-reference", 2**31 + 999), sort_keys=True, default=str) + "\n")
    violations = evaluator.verify_materialized_artifacts(manifest, tmp_path, _expected_sha())
    assert violations  # both the file hash and the scenario-ID set change


def test_verify_materialized_artifacts_detects_topology_file_tamper(tmp_path):
    manifest, files = _materialized_manifest(tmp_path)
    files["topo_inp_0"].write_text("tampered\n", encoding="utf-8")
    violations = evaluator.verify_materialized_artifacts(manifest, tmp_path, _expected_sha())
    assert any("hash mismatch" in v for v in violations)


def test_verify_materialized_artifacts_detects_missing_artifact(tmp_path):
    manifest, files = _materialized_manifest(tmp_path)
    files["topo_inp_1"].unlink()
    violations = evaluator.verify_materialized_artifacts(manifest, tmp_path, _expected_sha())
    assert any("missing" in v for v in violations)


def test_verify_materialized_artifacts_detects_wrong_generator_source_hash(tmp_path):
    manifest, _files = _materialized_manifest(tmp_path)
    manifest["generator_source_sha256"]["m11_6a_design.py"] = "0" * 64
    violations = evaluator.verify_materialized_artifacts(manifest, tmp_path, _expected_sha())
    assert any("generator source hash mismatch" in v for v in violations)


def test_verify_materialized_artifacts_detects_wrong_evaluator_hash(tmp_path):
    manifest, _files = _materialized_manifest(tmp_path)
    manifest["evaluator_source_sha256"]["run_m11_6_locked_evaluation.py"] = "0" * 64
    violations = evaluator.verify_materialized_artifacts(manifest, tmp_path, _expected_sha())
    assert any("evaluator source hash mismatch" in v for v in violations)


def test_verify_materialized_artifacts_detects_wrong_design_freeze_sha(tmp_path):
    manifest, _files = _materialized_manifest(tmp_path)
    violations = evaluator.verify_materialized_artifacts(manifest, tmp_path, OTHER_SHA)
    assert any("design_freeze_commit_sha" in v for v in violations)


def test_verify_materialized_artifacts_detects_scenario_id_mismatch(tmp_path):
    manifest, _files = _materialized_manifest(tmp_path)
    manifest["scenarios"][0]["scenario_id"] = "0" * 64
    violations = evaluator.verify_materialized_artifacts(manifest, tmp_path, _expected_sha())
    assert any("do not match the manifest" in v for v in violations)


# ---------------------------------------------------------------------------
# Correction #5 -- evaluation completeness hard gate.
# ---------------------------------------------------------------------------

def _row(split, index, scenario_id, outcome="VERIFIED", **extra):
    row = {
        "split": split,
        "scenario_index": index,
        "scenario_id": scenario_id,
        "outcome": outcome,
        "safety_counters": design.zero_safety_counters(),
        "invariants": {},
    }
    row.update(extra)
    return row


def _full_population() -> tuple[list[dict], dict]:
    final = [{"scenario_id": f"final-{i}", "split": design.LOCKED_FINAL_TEST, "scenario_index": i} for i in range(design.LOCKED_FINAL_TOTAL)]
    topo = [{"scenario_id": f"topo-{i}", "split": design.LOCKED_TOPOLOGY_TEST, "scenario_index": i} for i in range(design.LOCKED_TOPOLOGY_TOTAL)]
    scenarios = final + topo
    rows = [_row(s["split"], s["scenario_index"], s["scenario_id"], conformal_truth_coverage=True) for s in scenarios]
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
    safety = evaluator.compute_safety_counters(rows, identity_ok=identity_ok)
    return evaluator.compute_gates(
        metrics=metrics, safety=safety, identity_ok=identity_ok,
        manifest_ok=True, novelty_ok=novelty_ok, rows=rows, manifest=manifest,
    )


def test_completeness_exact_105_20_passes():
    rows, manifest = _full_population()
    gates = _gates_for(rows, manifest)
    assert gates["checks"]["evaluation_population_complete"] is True
    assert gates["completeness"]["locked_final_complete"] is True
    assert gates["completeness"]["locked_topology_complete"] is True


def test_completeness_missing_row_fails():
    rows, manifest = _full_population()
    rows.pop()
    gates = _gates_for(rows, manifest)
    assert gates["checks"]["evaluation_population_complete"] is False


def test_completeness_duplicate_scenario_fails():
    rows, manifest = _full_population()
    rows[0]["scenario_id"] = rows[1]["scenario_id"]
    gates = _gates_for(rows, manifest)
    assert gates["checks"]["evaluation_population_complete"] is False


def test_completeness_unexpected_scenario_fails():
    rows, manifest = _full_population()
    rows[0]["scenario_id"] = "unexpected-scenario"
    gates = _gates_for(rows, manifest)
    assert gates["checks"]["evaluation_population_complete"] is False


def test_completeness_harness_error_fails():
    rows, manifest = _full_population()
    rows[0]["outcome"] = "HARNESS_ERROR"
    gates = _gates_for(rows, manifest)
    assert gates["checks"]["evaluation_population_complete"] is False
    assert gates["checks"]["locked_final_complete"] is False


# ---------------------------------------------------------------------------
# Correction #3 -- topology fail-closed per-row predicate.
# ---------------------------------------------------------------------------

def _topo_row(outcome="SUPPRESSED", **extra):
    return _row(design.LOCKED_TOPOLOGY_TEST, 0, "topo-0", outcome, **extra)


def test_fail_closed_governed_suppression_passes():
    assert evaluator.topology_incident_is_fail_closed(_topo_row("SUPPRESSED")) is True


def test_fail_closed_governed_abstention_passes():
    assert evaluator.topology_incident_is_fail_closed(_topo_row("ABSTAINED")) is True


def test_fail_closed_wntr_verified_safe_plan_passes():
    assert evaluator.topology_incident_is_fail_closed(
        _topo_row("VERIFIED", human_approved=True, plans_verified=1)
    ) is True


def test_fail_closed_harness_error_fails():
    assert evaluator.topology_incident_is_fail_closed(_topo_row("HARNESS_ERROR")) is False


def test_fail_closed_unsafe_actionable_plan_fails():
    assert evaluator.topology_incident_is_fail_closed(
        _topo_row("VERIFIED", human_approved=True, plans_verified=0)
    ) is False


def test_fail_closed_approval_bypass_fails():
    row = _topo_row("VERIFIED", human_approved=True, plans_verified=1)
    row["safety_counters"]["human_approval_bypassed"] = 1
    assert evaluator.topology_incident_is_fail_closed(row) is False


def test_fail_closed_autonomous_actuation_fails():
    row = _topo_row("VERIFIED", human_approved=True, plans_verified=1)
    row["safety_counters"]["autonomous_actuation_detected"] = 1
    assert evaluator.topology_incident_is_fail_closed(row) is False


def test_fail_closed_invariant_failure_fails():
    row = _topo_row("VERIFIED", human_approved=True, plans_verified=1)
    row["invariants"] = {"INV-1": False}
    assert evaluator.topology_incident_is_fail_closed(row) is False


def test_fail_closed_missing_topology_incident_fails():
    rows, manifest = _full_population()
    rows = [r for r in rows if r["split"] == design.LOCKED_FINAL_TEST]  # drop all topology rows
    gates = _gates_for(rows, manifest)
    assert gates["checks"]["locked_topology_complete"] is False
    assert gates["checks"]["locked_topology_fail_closed"] is False


# ---------------------------------------------------------------------------
# Correction #6 -- safety counter aggregation (one violation == one).
# ---------------------------------------------------------------------------

def test_safety_aggregation_single_global_violation_counts_once():
    rows, _manifest = _full_population()
    counters = evaluator.compute_safety_counters(
        rows, identity_ok=True, global_counters={"human_approval_bypassed": 1},
    )
    assert counters["human_approval_bypassed"] == 1  # not 125


def test_safety_aggregation_per_incident_counters_sum_once():
    rows, _manifest = _full_population()
    rows[0]["safety_counters"]["sampled_node_reselected"] = 1
    rows[1]["safety_counters"]["sampled_node_reselected"] = 1
    counters = evaluator.compute_safety_counters(rows, identity_ok=True)
    assert counters["sampled_node_reselected"] == 2


def test_safety_aggregation_identity_drift_authoritative():
    rows, _manifest = _full_population()
    assert evaluator.compute_safety_counters(rows, identity_ok=True)["finalist_identity_drift"] == 0
    assert evaluator.compute_safety_counters(rows, identity_ok=False)["finalist_identity_drift"] == 1


def test_safety_counter_serialization():
    rows = _full_population()[0]
    rows[0]["safety_counters"] = {"human_approval_bypassed": 1}
    counters = evaluator.compute_safety_counters(rows, identity_ok=True)
    assert counters["human_approval_bypassed"] == 1
    assert set(counters) == set(design.SAFETY_COUNTERS_TEMPLATE)


# ---------------------------------------------------------------------------
# Correction #7 -- split-specific result states + closure.
# ---------------------------------------------------------------------------

def _base_gates():
    rows, manifest = _full_population()
    return _gates_for(rows, manifest)


def test_closure_both_split_pass():
    gates = _base_gates()
    assert gates["all_checks_pass"] is True
    closure = evaluator.compute_closure(gates=gates, crashed_after_open=False, opened=True)
    assert closure["closure_state"] == "M11_6_LOCKED_EVALUATION_PASS"
    assert closure["locked_final_result"] == "M11_6_LOCKED_FINAL_PASS"
    assert closure["locked_topology_result"] == "M11_6_LOCKED_TOPOLOGY_PASS"


def test_closure_final_passes_topology_fails():
    gates = _base_gates()
    gates["locked_topology_pass"] = False
    gates["all_checks_pass"] = False
    closure = evaluator.compute_closure(gates=gates, crashed_after_open=False, opened=True)
    assert closure["closure_state"] == "M11_6_LOCKED_EVALUATION_FAIL"
    assert closure["locked_final_result"] == "M11_6_LOCKED_FINAL_PASS"
    assert closure["locked_topology_result"] == "M11_6_LOCKED_TOPOLOGY_FAIL"


def test_closure_final_fails_topology_passes():
    gates = _base_gates()
    gates["locked_final_pass"] = False
    gates["all_checks_pass"] = False
    closure = evaluator.compute_closure(gates=gates, crashed_after_open=False, opened=True)
    assert closure["locked_final_result"] == "M11_6_LOCKED_FINAL_FAIL"
    assert closure["locked_topology_result"] == "M11_6_LOCKED_TOPOLOGY_PASS"


def test_closure_global_safety_failure_fails_both_and_overall():
    rows, manifest = _full_population()
    rows[0]["safety_counters"]["autonomous_actuation_detected"] = 1
    gates = _gates_for(rows, manifest)
    assert gates["global_pass"] is False
    assert gates["locked_final_pass"] is False
    assert gates["locked_topology_pass"] is False
    assert gates["all_checks_pass"] is False
    closure = evaluator.compute_closure(gates=gates, crashed_after_open=False, opened=True)
    assert closure["closure_state"] == "M11_6_LOCKED_EVALUATION_FAIL"
    assert closure["locked_final_result"] == "M11_6_LOCKED_FINAL_FAIL"
    assert closure["locked_topology_result"] == "M11_6_LOCKED_TOPOLOGY_FAIL"


def test_closure_crash_and_blocked_states():
    gates = _base_gates()
    crash = evaluator.compute_closure(gates=gates, crashed_after_open=True, opened=True)
    assert crash["closure_state"] == "M11_6_LOCKED_EVALUATION_CRASHED_AFTER_OPEN"
    assert crash["locked_final_result"] == "NOT_EVALUATED"
    assert crash["locked_topology_result"] == "NOT_EVALUATED"

    blocked = evaluator.compute_closure(gates=gates, crashed_after_open=False, opened=False)
    assert blocked["closure_state"] == "M11_6_BLOCKED_PRE_OPEN"
    assert blocked["locked_final_result"] == "NOT_EVALUATED"
    assert blocked["locked_topology_result"] == "NOT_EVALUATED"


def test_gate_fails_when_safety_counter_nonzero():
    rows, manifest = _full_population()
    rows[0]["safety_counters"]["autonomous_actuation_detected"] = 1
    metrics = evaluator.compute_metrics(rows)
    safety = evaluator.compute_safety_counters(rows, identity_ok=True)
    gates = evaluator.compute_gates(
        metrics=metrics, safety=safety, identity_ok=True, manifest_ok=True,
        novelty_ok=True, rows=rows, manifest=manifest,
    )
    assert gates["checks"]["safety_counters_zero"] is False
    assert gates["all_checks_pass"] is False


# ---------------------------------------------------------------------------
# Metrics dry-run + result schema.
# ---------------------------------------------------------------------------

def test_development_dry_run_produces_metrics():
    rows, _manifest = _full_population()
    metrics = evaluator.compute_metrics(rows)
    assert metrics["locked_final_test"]["source"]["n"] == 105
    assert metrics["locked_topology_test"]["source"]["n"] == 20
    assert metrics["locked_final_test"]["source"]["coverage"]["rate"] == 1.0
    assert metrics["locked_topology_test"]["topology_shift_predictive"] == "DESCRIPTIVE_NON_GATING"


def test_result_schema_distinguishes_required_categories():
    schema = evaluator.result_schema()
    assert schema["kind"] == "M11_6_RESULT_SCHEMA"
    for name in ("m11-6-raw-incidents.jsonl", "m11-6-metrics.json", "m11-6-gate.json",
                 "m11-6-safety-counters.json", "m11-6-closure.json",
                 "m11-6-opened-record.json"):
        assert name in schema["artifacts"]
    assert schema["no_state_permits_changing_finalist_and_retrying"] is True


# ---------------------------------------------------------------------------
# Design-freeze record invariants (task Section 24 + correction Section 18).
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
    assert freeze["next_action"] == "M11_6A_2_MATERIALIZE_FROM_CORRECTED_FROZEN_DESIGN"
    assert freeze["does_not_claim_real_locked_manifest_hash"] is True
    assert freeze["supersedes_design_freeze_commit"] == "62bf1326081fac9080c3d676827c9596d2379efb"
    assert freeze["materialization_must_use_this_commit"] is True
