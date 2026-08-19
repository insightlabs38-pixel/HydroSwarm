"""M11.6A-2 -- locked population materializer (from the frozen design).

This is the SEPARATE materialization program (task Section 12: "Dataset
materialization and evaluation must be separate commands/programs"). It is
NOT run against the locked namespace in M11.6A-1; a DIFFERENT, fresh session
runs it exactly once after the design freeze is independently reviewed.

Contract:
    python scripts/hydrocore_v5/run_m11_6a_materialize.py \
        --design-freeze-sha <M11_6A_1_COMMIT_SHA> \
        --output-root data/locked/m11-6

The materializer derives the two master seeds from DESIGN_FREEZE_COMMIT_SHA
(there is NO numeric seed chosen here), generates the four novelty-verified
procedural topologies, writes deterministic scenario DEFINITIONS (the full
ground truth is regenerated from these by the evaluator via WNTR/EPANET),
and writes the content-addressed materialization manifest. It never evaluates
the finalist, never sets locked_test_opened, and never sets
locked_evaluation_authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wntr  # noqa: E402

import m11_6a_design as design  # noqa: E402
import m11_6a_topology as topology  # noqa: E402

from hydroswarm.data.scenarios import network_sha256  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402

#: Known-family loaders and their committed .inp paths (the locked final
#: test reuses ONLY these already-governed known families).
KNOWN_FAMILY_LOADERS: dict[str, Any] = {
    "golden-reference": build_wntr_network,
    "branched-loop": lambda: wntr.network.WaterNetworkModel(str(ROOT / "data/topology-transfer/branched-loop.inp")),
    "loop-grid": lambda: wntr.network.WaterNetworkModel(str(ROOT / "data/topologies/loop-grid.inp")),
}
KNOWN_FAMILY_INP_PATHS: dict[str, Path] = {
    "golden-reference": ROOT / "data/frozen/golden_network.inp",
    "branched-loop": ROOT / "data/topology-transfer/branched-loop.inp",
    "loop-grid": ROOT / "data/topologies/loop-grid.inp",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _junctions(network: Any) -> tuple[str, ...]:
    return tuple(sorted(network.junction_name_list))


def _scenario_definition(
    *, split: str, scenario_index: int, topology_id: str, network_family: str,
    network_sha: str, seed: int, seed_domain: str, seed_derivation_counter: int,
    source_node: str, condition_kind: str, condition_fields: dict[str, Any],
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": design.SCENARIO_SCHEMA_VERSION,
        "split": split,
        "scenario_index": scenario_index,
        "topology_id": topology_id,
        "network_family": network_family,
        "network_sha256": network_sha,
        "seed": seed,
        "seed_domain": seed_domain,
        "seed_derivation_counter": seed_derivation_counter,
        "event_type": design.EVENT_TYPE,
        "source_node": source_node,
        "condition_kind": condition_kind,
        "condition": condition_fields,
        "generator_config": generator_config,
    }


def build_locked_final_definitions(locked_final_master: str) -> list[dict[str, Any]]:
    """Deterministically build the 105 locked_final_test scenario definitions."""
    definitions: list[dict[str, Any]] = []
    scenario_index = 0
    for family in design.LOCKED_FINAL_FAMILIES:
        network = KNOWN_FAMILY_LOADERS[family]()
        junctions = _junctions(network)
        sha = network_sha256(network)
        for condition_kind in design.LOCKED_FINAL_CONDITIONS:
            condition_fields = dict(design.LOCKED_FINAL_CONDITION_KWARGS[condition_kind])
            generator_config = design.scenario_config_for_condition(condition_kind)
            for incident_index in range(design.LOCKED_FINAL_INCIDENTS_PER_CELL):
                seed = design.derive_seed(locked_final_master, "FINAL_SCENARIO", scenario_index, 0)
                source_node = junctions[scenario_index % len(junctions)]
                definitions.append(_scenario_definition(
                    split=design.LOCKED_FINAL_TEST, scenario_index=scenario_index,
                    topology_id=f"locked-final:{family}", network_family=family,
                    network_sha=sha, seed=seed, seed_domain="FINAL_SCENARIO",
                    seed_derivation_counter=0, source_node=source_node,
                    condition_kind=condition_kind, condition_fields=condition_fields,
                    generator_config=generator_config,
                ))
                scenario_index += 1
    assert scenario_index == design.LOCKED_FINAL_TOTAL
    return definitions


def build_locked_topology_definitions(
    locked_topology_master: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate the 4 procedural topologies + 20 scenario definitions."""
    topologies: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    seen_network_hashes: list[str] = []
    seen_signatures: list[dict[str, Any]] = []
    scenario_index = 0
    for topology_index in range(design.LOCKED_TOPOLOGY_INSTANCES):
        generated = topology.generate_locked_topology(locked_topology_master, topology_index)
        network = generated["network"]
        signature = generated["graph_signature"]
        sha = generated["network_sha256"]
        # Within-set duplicate rejection (frozen novelty rule).
        novel, reasons = topology.is_novel_topology(
            network, seen_network_hashes=tuple(seen_network_hashes), seen_signatures=tuple(seen_signatures),
        )
        if not novel:
            raise RuntimeError(f"topology index {topology_index} failed within-set novelty: {reasons}")
        seen_network_hashes.append(sha)
        seen_signatures.append(signature)
        junctions = _junctions(network)
        topologies.append({
            "topology_id": f"locked-topology:{topology_index}",
            "topology_index": topology_index,
            "candidate_index": generated["candidate_index"],
            "network_seed": generated["seed"],
            "network_sha256": sha,
            "graph_signature": signature,
            "junction_count": generated["junction_count"],
            "cycle_rank": generated["cycle_rank"],
        })
        for incident_index in range(design.LOCKED_TOPOLOGY_INCIDENTS_PER_TOPOLOGY):
            seed = design.derive_seed(locked_topology_master, "TOPOLOGY_TEST_SCENARIO", scenario_index, 0)
            source_node = junctions[scenario_index % len(junctions)]
            definitions.append(_scenario_definition(
                split=design.LOCKED_TOPOLOGY_TEST, scenario_index=scenario_index,
                topology_id=f"locked-topology:{topology_index}",
                network_family="locked-topology-procedural", network_sha=sha,
                seed=seed, seed_domain="TOPOLOGY_TEST_SCENARIO",
                seed_derivation_counter=0, source_node=source_node,
                condition_kind="NOMINAL",
                condition_fields=dict(design.LOCKED_FINAL_CONDITION_KWARGS["NOMINAL"]),
                generator_config=design.scenario_config_for_condition("NOMINAL"),
            ))
            scenario_index += 1
    assert scenario_index == design.LOCKED_TOPOLOGY_TOTAL
    return topologies, definitions


def _overlap_audit(definitions: list[dict[str, Any]]) -> dict[str, Any]:
    """Frozen non-overlap check (task Section 9), truthfully scoped.

    Prior governed experiments (M0-M11.5) did not materialize a comparable
    canonical scenario-definition hash under SCENARIO_SCHEMA_VERSION, so NO
    direct canonical-hash comparison against historical scenarios is claimed.
    The mechanical guarantee is: derived seeds are disjoint from every prior
    seed namespace BY CONSTRUCTION (all prior namespaces are < 2**31), plus
    within-set canonical-hash uniqueness.
    """
    hashes = [design.scenario_definition_hash(definition) for definition in definitions]
    unique = len(set(hashes))
    seeds = [definition["seed"] for definition in definitions]
    collisions = len(hashes) - unique
    seed_out_of_range = [
        s for s in seeds
        if not (design.LOCKED_SEED_MIN <= s < design.LOCKED_SEED_MAX_EXCLUSIVE)
    ]
    prior_max = max(rng[1] for rng in design.PRIOR_SEED_RANGES.values())
    seed_namespace_disjoint = bool(not seed_out_of_range and prior_max < design.LOCKED_SEED_MIN)
    return {
        "result": (
            "PASS" if (collisions == 0 and not seed_out_of_range and seed_namespace_disjoint) else "FAIL"
        ),
        "n_scenarios": len(definitions),
        "unique_canonical_hashes": unique,
        "collisions": collisions,
        "within_set_unique": collisions == 0,
        "seeds_out_of_range": seed_out_of_range,
        "seed_namespace_disjoint": seed_namespace_disjoint,
        "prior_seed_namespace_max": prior_max,
        "derived_seed_min": design.LOCKED_SEED_MIN,
        "historical_canonical_hash_comparison_performed": False,
        "historical_comparison_note": (
            "No direct canonical-scenario-hash comparison against historical "
            "M0-M11.5 scenarios is performed or claimed: prior experiments used "
            "a different scenario schema with no comparable canonical "
            "definition hash. Non-overlap is guaranteed by seed-namespace "
            "disjointness BY CONSTRUCTION plus within-set uniqueness plus "
            "topology novelty."
        ),
        "seed_namespace_note": (
            "All derived seeds are in [2**31, 2**62); every prior seed "
            "namespace is < 2**31, so derived seeds are disjoint from all "
            "prior namespaces by construction (the interval [0, 2**31) is "
            "entirely excluded)."
        ),
    }


def _novelty_audit(topologies: list[dict[str, Any]]) -> dict[str, Any]:
    """Frozen two-phase topology-novelty audit (task Section 11).

    (A) pre-serialization graph/network novelty: graph_signature + network_sha256
    checked against the frozen prior inventory AND within the generated set;
    (B) post-serialization file-byte novelty: the exact materialized .inp bytes
    (file_sha256) must differ from every frozen prior topology file hash AND
    from every other generated .inp file. ``each_satisfies_frozen_novelty_rule``
    is COMPUTED, never hard-coded true.
    """
    per_topology: list[dict[str, Any]] = []
    seen_network_hashes: list[str] = []
    seen_signatures: list[dict[str, Any]] = []
    seen_file_hashes: list[str] = []
    all_ok = len(topologies) == design.LOCKED_TOPOLOGY_INSTANCES

    for entry in topologies:
        sha = entry.get("network_sha256")
        signature = entry.get("graph_signature") or {}
        file_sha = entry.get("file_sha256")

        graph_signature_novel = not any(
            design.signatures_equal(signature, prior) for prior in design.PRIOR_TOPOLOGY_SIGNATURES
        )
        network_hash_novel = bool(sha) and sha not in design.PRIOR_TOPOLOGY_NETWORK_HASHES
        within_set_network_novel = bool(sha) and sha not in seen_network_hashes
        within_set_signature_novel = not any(
            design.signatures_equal(signature, existing) for existing in seen_signatures
        )
        file_byte_novel = bool(
            file_sha
            and file_sha not in design.PRIOR_TOPOLOGY_FILE_HASHES
            and file_sha not in seen_file_hashes
        )

        entry_ok = bool(
            graph_signature_novel
            and network_hash_novel
            and within_set_network_novel
            and within_set_signature_novel
            and file_byte_novel
        )
        all_ok = all_ok and entry_ok
        per_topology.append({
            "topology_id": entry.get("topology_id"),
            "graph_signature_novel": graph_signature_novel,
            "network_hash_novel": network_hash_novel,
            "within_set_network_novel": within_set_network_novel,
            "within_set_signature_novel": within_set_signature_novel,
            "file_byte_novel": file_byte_novel,
            "file_sha256": file_sha,
        })
        seen_network_hashes.append(sha)
        seen_signatures.append(signature)
        seen_file_hashes.append(file_sha)

    return {
        "result": "PASS" if all_ok else "FAIL",
        "n_topologies": len(topologies),
        "required": design.LOCKED_TOPOLOGY_INSTANCES,
        "each_satisfies_frozen_novelty_rule": bool(all_ok),
        "per_topology": per_topology,
        "prior_topology_junction_range": [4, 8],
        "generated_junction_range": [9, 12],
        "prior_topology_file_hashes_checked": len(design.PRIOR_TOPOLOGY_FILE_HASHES),
    }


def materialize(design_freeze_sha: str, output_root: Path) -> dict[str, Any]:
    """Materialize the locked population (topologies + definitions + manifest).

    Refuses to run if the smoke namespace or an already-materialized marker
    appears anywhere it would write. Does NOT evaluate the finalist.
    """

    if design.FORBIDDEN_SMOKE_NAMESPACE in str(output_root):
        raise ValueError(f"refusing to materialize into a smoke-namespace path: {output_root}")
    if (output_root / "m11-6-materialization-manifest.json").exists():
        raise RuntimeError(f"materialization manifest already exists at {output_root}; M11.6A-2 must run exactly once")

    locked_final_master = design.derive_master_seed(design_freeze_sha, design.MASTER_DOMAIN_FINAL)
    locked_topology_master = design.derive_master_seed(design_freeze_sha, design.MASTER_DOMAIN_TOPOLOGY)

    final_definitions = build_locked_final_definitions(locked_final_master)
    topologies, topology_definitions = build_locked_topology_definitions(locked_topology_master)
    all_definitions = final_definitions + topology_definitions

    output_root.mkdir(parents=True, exist_ok=True)
    topologies_dir = output_root / "topologies"
    topologies_dir.mkdir(parents=True, exist_ok=True)

    # Write the four procedural topology .inp files using the PORTABLE
    # filename mapping (never the colon-bearing logical ID as a filename).
    topologies_doc: list[dict[str, Any]] = []
    for entry in topologies:
        topology_id = entry["topology_id"]
        file_path = topologies_dir / design.topology_filename(topology_id)
        wntr.network.write_inpfile(
            topology.generate_locked_topology(locked_topology_master, entry["topology_index"])["network"],
            str(file_path),
        )
        entry.update({
            "file_path": design.repo_relative_manifest_path(file_path, ROOT),
            "file_sha256": _sha256_file(file_path),
        })
        topologies_doc.append(entry)

    # Known families (locked_final_test): reference the committed .inp files.
    for family in design.LOCKED_FINAL_FAMILIES:
        path = KNOWN_FAMILY_INP_PATHS[family]
        network = KNOWN_FAMILY_LOADERS[family]()
        topologies_doc.append({
            "topology_id": f"locked-final:{family}",
            "split": design.LOCKED_FINAL_TEST,
            "network_family": family,
            "network_sha256": network_sha256(network),
            "file_path": design.repo_relative_manifest_path(path, ROOT),
            "file_sha256": _sha256_file(path),
            "graph_signature": topology.topology_graph_signature(network),
        })

    # Scenario definition JSONL.
    (output_root / design.LOCKED_FINAL_TEST).mkdir(parents=True, exist_ok=True)
    (output_root / design.LOCKED_TOPOLOGY_TEST).mkdir(parents=True, exist_ok=True)
    for split, definitions in ((design.LOCKED_FINAL_TEST, final_definitions), (design.LOCKED_TOPOLOGY_TEST, topology_definitions)):
        with (output_root / split / "scenarios.jsonl").open("w", encoding="utf-8") as handle:
            for definition in definitions:
                handle.write(json.dumps(definition, sort_keys=True, default=str) + "\n")

    # Manifest (content-addressed; NO model performance metrics).
    source_files = {
        "m11_6a_design.py": ROOT / "scripts/hydrocore_v5/m11_6a_design.py",
        "m11_6a_topology.py": ROOT / "scripts/hydrocore_v5/m11_6a_topology.py",
        "run_m11_6a_materialize.py": ROOT / "scripts/hydrocore_v5/run_m11_6a_materialize.py",
    }
    evaluator_file = ROOT / "scripts/hydrocore_v5/run_m11_6_locked_evaluation.py"
    manifest: dict[str, Any] = {
        "schema_version": design.MANIFEST_SCHEMA_VERSION,
        "design_freeze_commit_sha": design_freeze_sha,
        "design_protocol_sha256": design.design_hash(),
        "generator_source_sha256": {name: _sha256_file(path) for name, path in source_files.items()},
        "evaluator_source_sha256": {"run_m11_6_locked_evaluation.py": _sha256_file(evaluator_file)},
        "seed_derivation": design.seed_derivation_spec(),
        "master_seeds": {
            "locked_final_master": {"domain": design.MASTER_DOMAIN_FINAL, "hex": locked_final_master},
            "locked_topology_master": {"domain": design.MASTER_DOMAIN_TOPOLOGY, "hex": locked_topology_master},
        },
        "splits": {
            design.LOCKED_FINAL_TEST: {
                "count": design.LOCKED_FINAL_TOTAL,
                "families": list(design.LOCKED_FINAL_FAMILIES),
                "conditions": list(design.LOCKED_FINAL_CONDITIONS),
                "incidents_per_cell": design.LOCKED_FINAL_INCIDENTS_PER_CELL,
            },
            design.LOCKED_TOPOLOGY_TEST: {
                "count": design.LOCKED_TOPOLOGY_TOTAL,
                "topology_instances": design.LOCKED_TOPOLOGY_INSTANCES,
                "incidents_per_topology": design.LOCKED_TOPOLOGY_INCIDENTS_PER_TOPOLOGY,
            },
        },
        "topologies": topologies_doc,
        "scenarios": [
            {
                "scenario_id": design.scenario_definition_hash(definition),
                "scenario_index": definition["scenario_index"],
                "split": definition["split"],
                "topology_id": definition["topology_id"],
                "network_sha256": definition["network_sha256"],
                "seed": definition["seed"],
                "seed_domain": definition["seed_domain"],
                "source_node": definition["source_node"],
                "condition_kind": definition["condition_kind"],
            }
            for definition in all_definitions
        ],
        "artifact_sha256": {},
        "simulator": {
            "backend": "WNTR/EPANET",
            "wntr_version": getattr(wntr, "__version__", "unavailable"),
            "pattern_timestep_s": 3600,
            "hydraulic_timestep_s": 3600,
            "quality_timestep_s": 300,
            "duration_s": 86400,
        },
        "generation_complete": True,
        "overlap_audit": _overlap_audit(all_definitions),
        "novelty_audit": _novelty_audit(topologies),
        "evaluated_by_finalist": False,
        "locked_test_opened": False,
    }

    # artifact_sha256 for every materialized file (canonical POSIX keys).
    artifact_files = [
        topologies_dir / design.topology_filename(entry["topology_id"]) for entry in topologies
    ]
    artifact_files += [
        output_root / design.LOCKED_FINAL_TEST / "scenarios.jsonl",
        output_root / design.LOCKED_TOPOLOGY_TEST / "scenarios.jsonl",
    ]
    for path in artifact_files:
        manifest["artifact_sha256"][design.repo_relative_manifest_path(path, ROOT)] = _sha256_file(path)

    violations = design.validate_manifest(manifest)
    if violations:
        raise RuntimeError(f"materialized manifest failed validation: {violations}")

    manifest_path = output_root / "m11-6-materialization-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def validate_design_freeze_sha(sha: str, repo_root: Path) -> list[str]:
    """Fail-closed design-freeze SHA validation (final correction).

    Rejects: any superseded freeze SHA, a malformed/nonexistent SHA, a SHA not
    an ancestor of the governed branch HEAD, and a SHA whose frozen design
    artifact does not declare the required materialization authority flags.
    """
    violations: list[str] = []
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        violations.append("design-freeze SHA must be a full 40-hex git commit SHA")
        return violations
    if sha in design.SUPERSEDED_DESIGN_FREEZE_COMMITS:
        violations.append(f"design-freeze SHA {sha} is superseded; materialize only from the final corrected freeze commit")
    try:
        subprocess.check_output(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        violations.append(f"design-freeze SHA {sha} does not exist in git")
        return violations
    try:
        subprocess.check_output(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        violations.append(f"design-freeze SHA {sha} is not an ancestor of HEAD")
    artifact_path = "reports/evaluation/hydrocore-v5/m11/m11-6a/design-freeze/m11-6a-design-freeze.json"
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{sha}:{artifact_path}"], cwd=repo_root, text=True,
        )
        artifact = json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        violations.append(f"cannot read the frozen design artifact at {sha}")
        return violations
    if artifact.get("design_frozen") is not True:
        violations.append("design artifact at SHA does not declare design_frozen=true")
    if artifact.get("materialization_must_use_this_commit") is not True:
        violations.append("design artifact at SHA does not declare materialization_must_use_this_commit=true")
    if artifact.get("dataset_materialized") is not False:
        violations.append("design artifact at SHA declares dataset_materialized != false")
    if artifact.get("locked_open_count") != 0:
        violations.append("design artifact at SHA declares locked_open_count != 0")
    if artifact.get("locked_test_opened") is not False:
        violations.append("design artifact at SHA declares locked_test_opened != false")

    violations.extend(design_identity_violations(artifact, repo_root))
    return violations


def design_identity_violations(artifact: dict[str, Any], repo_root: Path) -> list[str]:
    """Code-identity binding (task Section 6): the freeze SHA's artifact must
    record the SAME design hash as the current governed design code, AND every
    governed design/materializer/evaluator file on disk must be byte-identical
    to the hash frozen in that artifact.

    "SHA is an ancestor" is NOT sufficient: a previously frozen ancestor stays
    syntactically valid even if the governed code at HEAD changed, which would
    silently rebind the seed to different evaluator/materializer code. An
    unrelated later commit may only be accepted if every frozen governed file
    remains byte-identical.
    """
    violations: list[str] = []
    if artifact.get("design_hash") != design.design_hash():
        violations.append(
            "design artifact design_hash does not match the current frozen design code "
            "(design.design_hash())"
        )
    frozen_file_hashes = artifact.get("design_file_hashes") or {}
    for rel_path in design.GOVERNED_DESIGN_FILES:
        expected = frozen_file_hashes.get(rel_path)
        if not expected:
            violations.append(f"design artifact has no design_file_hashes entry for {rel_path}")
            continue
        current_path = repo_root / rel_path
        if not current_path.exists():
            violations.append(f"governed design file missing at HEAD: {rel_path}")
            continue
        current = _sha256_file(current_path)
        if current != expected:
            violations.append(
                f"governed design file changed after freeze commit: {rel_path} "
                f"(artifact={expected} current={current})"
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="M11.6A-2 locked population materializer")
    parser.add_argument("--design-freeze-sha", required=True, help="M11.6A-1 final corrected design-freeze commit SHA (full 40-hex)")
    parser.add_argument("--output-root", default="data/locked/m11-6")
    args = parser.parse_args()
    sha = args.design_freeze_sha.strip()
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        print("error: --design-freeze-sha must be a full 40-hex git commit SHA", file=sys.stderr)
        return 2
    violations = validate_design_freeze_sha(sha, ROOT)
    if violations:
        print("error: design-freeze SHA validation failed:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 2
    manifest = materialize(sha, ROOT / args.output_root)
    print(json.dumps({"materialized": True, "manifest": str((ROOT / args.output_root) / "m11-6-materialization-manifest.json"), "locked_test_opened": manifest["locked_test_opened"], "evaluated_by_finalist": manifest["evaluated_by_finalist"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
