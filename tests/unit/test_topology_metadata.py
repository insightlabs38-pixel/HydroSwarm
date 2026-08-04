from __future__ import annotations

import json

import pytest
import torch

from hydroswarm.training import (
    CurriculumStage,
    GovernedScenarioDataset,
    ScenarioExample,
    TopologyMetadata,
    load_scenario_examples_jsonl,
    manifest_entry,
    resolve_source_node_id,
)


def _topology(
    *,
    topology_hash: str = "topo-1",
    network_hash: str = "net-1",
    node_ids: tuple[str, ...] = ("J1", "J2", "J3"),
    source_candidate_ids: tuple[str, ...] | None = None,
) -> TopologyMetadata:
    edges = tuple(zip(node_ids, node_ids[1:])) if len(node_ids) >= 2 else ()
    return TopologyMetadata(
        topology_hash=topology_hash,
        network_hash=network_hash,
        node_ids=node_ids,
        edge_ids=edges,
        source_candidate_ids=source_candidate_ids or node_ids,
        hydraulic_state_hash="state-1",
        signature_library_hash="sig-1",
        target_schema_version="targets_v1",
        feature_schema_version="hydroswarm-features-v2",
    )


def _example(scenario_id: str, *, topology: TopologyMetadata, source_local_index: int = 0) -> ScenarioExample:
    return ScenarioExample(
        scenario_id=scenario_id,
        network_id=topology.network_hash,
        split="train",
        seed=1,
        seed_family=f"family-{scenario_id}",
        stage=CurriculumStage.CLEAN,
        inputs={"node_features": torch.zeros(len(topology.node_ids), 3)},
        targets={"source_node": torch.tensor(source_local_index)},
        topology=topology,
    )


def test_topology_metadata_validates_edge_endpoints_and_candidates() -> None:
    with pytest.raises(ValueError, match="edge endpoints"):
        TopologyMetadata(
            topology_hash="t",
            network_hash="n",
            node_ids=("A", "B"),
            edge_ids=(("A", "Z"),),  # Z is not in node_ids
            source_candidate_ids=("A",),
            hydraulic_state_hash="h",
            signature_library_hash="s",
            target_schema_version="v1",
            feature_schema_version="v2",
        )
    with pytest.raises(ValueError, match="source_candidate_ids"):
        TopologyMetadata(
            topology_hash="t",
            network_hash="n",
            node_ids=("A", "B"),
            edge_ids=(),
            source_candidate_ids=("A", "Q"),  # Q is not in node_ids
            hydraulic_state_hash="h",
            signature_library_hash="s",
            target_schema_version="v1",
            feature_schema_version="v2",
        )
    with pytest.raises(ValueError, match="unique"):
        TopologyMetadata(
            topology_hash="t",
            network_hash="n",
            node_ids=("A", "A"),
            edge_ids=(),
            source_candidate_ids=("A",),
            hydraulic_state_hash="h",
            signature_library_hash="s",
            target_schema_version="v1",
            feature_schema_version="v2",
        )


def test_two_networks_with_different_node_ids_serialize_successfully() -> None:
    small = _topology(node_ids=("J1", "J2", "J3"))
    different_ids = _topology(topology_hash="topo-2", network_hash="net-2", node_ids=("X9", "Y7", "Z3"))
    for topology in (small, different_ids):
        payload = topology.to_json()
        json.dumps(payload)  # must be JSON-serializable
        restored = TopologyMetadata.from_json(payload)
        assert restored == topology


def test_two_networks_with_different_node_counts_serialize_successfully() -> None:
    small = _topology(node_ids=("A", "B"), source_candidate_ids=("A", "B"))
    large = _topology(
        topology_hash="topo-big",
        network_hash="net-big",
        node_ids=tuple(f"N{i}" for i in range(50)),
        source_candidate_ids=tuple(f"N{i}" for i in range(50)),
    )
    for topology in (small, large):
        restored = TopologyMetadata.from_json(json.loads(json.dumps(topology.to_json())))
        assert restored.node_count == len(topology.node_ids)
        assert restored == topology
    assert small.node_count == 2
    assert large.node_count == 50


def test_source_node_id_maps_to_correct_local_index_and_back() -> None:
    topology = _topology(node_ids=("J1", "J2", "J3"), source_candidate_ids=("J1", "J2", "J3"))
    assert topology.local_index("J2") == 1
    assert topology.source_node_id_for_local_index(1) == "J2"

    example = _example("s1", topology=topology, source_local_index=1)
    assert resolve_source_node_id(example) == "J2"


def test_source_node_id_uses_full_node_ids_space_not_the_candidate_subset() -> None:
    # source_candidate_ids is a strict subset of node_ids here (J2 is not a
    # candidate). source_node is still a local index into the FULL node_ids
    # space -- matching how HydroCore's source_node_logits cover every node
    # position, with source_candidate_mask separately marking eligibility --
    # so it must map through node_ids, not through the shorter
    # source_candidate_ids list (which would silently misalign).
    topology = _topology(node_ids=("J1", "J2", "J3"), source_candidate_ids=("J1", "J3"))
    assert topology.source_node_id_for_local_index(2) == "J3"

    example = _example("s2", topology=topology, source_local_index=2)
    assert resolve_source_node_id(example) == "J3"


def test_resolve_source_node_id_returns_none_without_topology_metadata() -> None:
    example = ScenarioExample(
        scenario_id="no-topo",
        network_id="net",
        split="train",
        seed=0,
        seed_family="fam",
        stage=CurriculumStage.CLEAN,
        inputs={"node_features": torch.zeros(2, 2)},
        targets={"source_node": torch.tensor(0)},
    )
    assert resolve_source_node_id(example) is None


def test_reordering_node_ids_changes_local_index_but_not_semantic_identity() -> None:
    original = _topology(node_ids=("J1", "J2", "J3"), source_candidate_ids=("J1", "J2", "J3"))
    permuted = _topology(node_ids=("J3", "J1", "J2"), source_candidate_ids=("J3", "J1", "J2"))

    original_index = original.local_index("J2")
    permuted_index = permuted.local_index("J2")
    assert original_index != permuted_index  # local index changed under reordering

    # But both still resolve back to the same real-world node identity.
    assert original.source_node_id_for_local_index(original_index) == "J2"
    assert permuted.source_node_id_for_local_index(permuted_index) == "J2"


def test_manifest_hashes_are_deterministic_across_repeated_generation() -> None:
    topology = _topology()
    examples = [_example("a", topology=topology), _example("b", topology=topology)]
    first = GovernedScenarioDataset(list(examples), expected_split="train").manifest_hash
    second = GovernedScenarioDataset(list(examples), expected_split="train").manifest_hash
    assert first == second


def test_manifest_hash_is_topology_aware() -> None:
    topology_a = _topology(topology_hash="topo-a")
    topology_b = _topology(topology_hash="topo-b")
    examples_a = [_example("a", topology=topology_a)]
    examples_b = [_example("a", topology=topology_b)]
    hash_a = GovernedScenarioDataset(examples_a, expected_split="train").manifest_hash
    hash_b = GovernedScenarioDataset(examples_b, expected_split="train").manifest_hash
    assert hash_a != hash_b


def test_manifest_entry_includes_topology_and_network_hash() -> None:
    topology = _topology()
    example = _example("a", topology=topology)
    entry = manifest_entry(example)
    assert entry["topology_hash"] == "topo-1"
    assert entry["network_hash"] == "net-1"


def test_load_scenario_examples_jsonl_round_trips_topology_metadata(tmp_path) -> None:
    topology = _topology(node_ids=("Q1", "Q2"), source_candidate_ids=("Q1", "Q2"))
    record = {
        "scenario_id": "roundtrip",
        "network_id": topology.network_hash,
        "split": "train",
        "seed": 3,
        "seed_family": "fam-roundtrip",
        "stage": "CLEAN",
        "inputs": {"node_features": [[0.0, 0.0], [1.0, 1.0]]},
        "targets": {"source_node": 1},
        "topology": topology.to_json(),
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    (loaded,) = load_scenario_examples_jsonl(manifest)
    assert loaded.topology == topology
    assert resolve_source_node_id(loaded) == "Q2"


def test_load_scenario_examples_jsonl_without_topology_field_still_works(tmp_path) -> None:
    record = {
        "scenario_id": "legacy",
        "network_id": "net",
        "split": "train",
        "seed": 0,
        "seed_family": "fam-legacy",
        "stage": "CLEAN",
        "inputs": {"node_features": [[0.0]]},
        "targets": {"source_node": 0},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    (loaded,) = load_scenario_examples_jsonl(manifest)
    assert loaded.topology is None
