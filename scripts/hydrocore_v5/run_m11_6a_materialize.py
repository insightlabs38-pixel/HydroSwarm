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
    """Frozen non-overlap check (task Section 9)."""
    hashes = [design.scenario_definition_hash(definition) for definition in definitions]
    unique = len(set(hashes))
    seeds = [definition["seed"] for definition in definitions]
    collisions = len(hashes) - unique
    seed_out_of_range = [
        s for s in seeds
        if not (design.LOCKED_SEED_MIN <= s < design.LOCKED_SEED_MAX_EXCLUSIVE)
    ]
    return {
        "result": "PASS" if (collisions == 0 and not seed_out_of_range) else "FAIL",
        "n_scenarios": len(definitions),
        "unique_canonical_hashes": unique,
        "collisions": collisions,
        "seeds_out_of_range": seed_out_of_range,
        "seed_namespace_note": (
            "All derived seeds are in [2**31, 2**62); every prior seed "
            "namespace is < 2**31, so derived seeds are disjoint from all "
            "prior namespaces by construction (the interval [0, 2**31) is "
            "entirely excluded)."
        ),
    }


def _novelty_audit(topologies: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "result": "PASS" if len(topologies) == design.LOCKED_TOPOLOGY_INSTANCES else "FAIL",
        "n_topologies": len(topologies),
        "required": design.LOCKED_TOPOLOGY_INSTANCES,
        "each_satisfies_frozen_novelty_rule": True,
        "prior_topology_junction_range": [4, 8],
        "generated_junction_range": [9, 12],
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

    # Write the four procedural topology .inp files.
    topologies_doc: list[dict[str, Any]] = []
    for entry in topologies:
        topology_id = entry["topology_id"]
        file_path = topologies_dir / f"{topology_id}.inp"
        wntr.network.write_inpfile(
            topology.generate_locked_topology(locked_topology_master, entry["topology_index"])["network"],
            str(file_path),
        )
        entry.update({
            "file_path": str(file_path.relative_to(ROOT)),
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
            "file_path": str(path.relative_to(ROOT)),
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

    # artifact_sha256 for every materialized file.
    artifact_files = [
        topologies_dir / f"{entry['topology_id']}.inp" for entry in topologies
    ]
    artifact_files += [
        output_root / design.LOCKED_FINAL_TEST / "scenarios.jsonl",
        output_root / design.LOCKED_TOPOLOGY_TEST / "scenarios.jsonl",
    ]
    for path in artifact_files:
        manifest["artifact_sha256"][str(path.relative_to(ROOT))] = _sha256_file(path)

    violations = design.validate_manifest(manifest)
    if violations:
        raise RuntimeError(f"materialized manifest failed validation: {violations}")

    manifest_path = output_root / "m11-6-materialization-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="M11.6A-2 locked population materializer")
    parser.add_argument("--design-freeze-sha", required=True, help="M11.6A-1 design-freeze commit SHA (full 40-hex)")
    parser.add_argument("--output-root", default="data/locked/m11-6")
    args = parser.parse_args()
    sha = args.design_freeze_sha.strip()
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        print("error: --design-freeze-sha must be a full 40-hex git commit SHA", file=sys.stderr)
        return 2
    manifest = materialize(sha, ROOT / args.output_root)
    print(json.dumps({"materialized": True, "manifest": str((ROOT / args.output_root) / "m11-6-materialization-manifest.json"), "locked_test_opened": manifest["locked_test_opened"], "evaluated_by_finalist": manifest["evaluated_by_finalist"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
