from __future__ import annotations

import random

import pytest
import torch

from hydroswarm.model import HydroCore
from hydroswarm.training import (
    CurriculumStage,
    ScenarioExample,
    TopologyMetadata,
    collate_variable_topology,
    measure_equivariance,
    permute_example,
    resolve_source_node_id,
)


def _tiny_model() -> HydroCore:
    return HydroCore(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        edge_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        latent_tokens=64,
        adapter_dims=(32, 32, 32),
        dropout=0.0,
    )


def _example(seed: int = 1, nodes: int = 5, source_local_index: int = 2) -> ScenarioExample:
    generator = torch.Generator().manual_seed(seed)
    steps = 2
    edges = [(i, i + 1) for i in range(nodes - 1)] + [(nodes - 1, 0)]  # ring
    edge_index = torch.tensor(edges, dtype=torch.long).T
    node_ids = tuple(f"J{i}" for i in range(nodes))
    topology = TopologyMetadata(
        topology_hash="topo-1",
        network_hash="net-1",
        node_ids=node_ids,
        edge_ids=tuple((node_ids[a], node_ids[b]) for a, b in edges),
        source_candidate_ids=node_ids,
        hydraulic_state_hash="state-1",
        signature_library_hash="sig-1",
        target_schema_version="targets_v1",
        feature_schema_version="hydroswarm-features-v2",
    )
    return ScenarioExample(
        scenario_id=f"scenario-{seed}",
        network_id="net-1",
        split="train",
        seed=seed,
        seed_family=f"family-{seed}",
        stage=CurriculumStage.CLEAN,
        inputs={
            "node_features": torch.randn(nodes, 3, generator=generator),
            "temporal_features": torch.randn(steps, nodes, 2, generator=generator),
            "quality_features": torch.randn(steps, nodes, 2, generator=generator),
            "edge_index": edge_index,
            "edge_features": torch.randn(len(edges), 2, generator=generator),
            "travel_time": torch.rand(nodes, generator=generator),
            "reservoir_reachability": torch.rand(nodes, generator=generator),
            "demand_centrality": torch.rand(nodes, generator=generator),
            "source_candidate_mask": torch.ones(nodes, dtype=torch.bool),
            "node_mask": torch.ones(nodes, dtype=torch.bool),
        },
        targets={
            "source_node": torch.tensor(source_local_index),
            "sensor_fault": torch.rand(nodes, generator=generator) > 0.8,
        },
        topology=topology,
    )


def _random_permutation(nodes: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    permutation = list(range(nodes))
    rng.shuffle(permutation)
    return permutation


def test_permute_example_rejects_invalid_permutation() -> None:
    example = _example(nodes=4)
    with pytest.raises(ValueError, match="rearrangement"):
        permute_example(example, [0, 0, 1, 2])


def test_permutation_remaps_source_node_and_preserves_real_world_identity() -> None:
    example = _example(nodes=5, source_local_index=2)
    original_source_id = resolve_source_node_id(example)
    permutation = _random_permutation(5, seed=42)

    permuted = permute_example(example, permutation)
    permuted_source_id = resolve_source_node_id(permuted)

    assert original_source_id == permuted_source_id  # semantic identity preserved
    # And the local index almost certainly changed under a random permutation
    # of 5 elements (only identity or fixed-point-at-source would keep it the same).
    assert permutation != list(range(5))


def test_permutation_preserves_node_feature_multiset() -> None:
    example = _example(nodes=6)
    permutation = _random_permutation(6, seed=7)
    permuted = permute_example(example, permutation)

    original_rows = {tuple(row.tolist()) for row in example.inputs["node_features"]}
    permuted_rows = {tuple(row.tolist()) for row in permuted.inputs["node_features"]}
    assert original_rows == permuted_rows


def test_permutation_preserves_edge_connectivity_by_real_world_id() -> None:
    example = _example(nodes=5)
    permutation = _random_permutation(5, seed=13)
    permuted = permute_example(example, permutation)

    def edges_as_real_ids(ex: ScenarioExample) -> set[tuple[str, str]]:
        node_ids = ex.topology.node_ids
        edge_index = ex.inputs["edge_index"]
        return {
            (node_ids[int(source)], node_ids[int(target)])
            for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist())
        }

    assert edges_as_real_ids(example) == edges_as_real_ids(permuted)


def test_permutation_preserves_topology_metadata_hashes() -> None:
    example = _example(nodes=4)
    permuted = permute_example(example, _random_permutation(4, seed=1))
    assert permuted.topology.topology_hash == example.topology.topology_hash
    assert permuted.topology.network_hash == example.topology.network_hash
    assert set(permuted.topology.node_ids) == set(example.topology.node_ids)


def test_graph_permutation_equivariance() -> None:
    model = _tiny_model()
    example = _example(nodes=5, source_local_index=2, seed=99)
    permutation = _random_permutation(5, seed=2026)

    report = measure_equivariance(
        model, example, permutation, collate_fn=collate_variable_topology, atol=1e-4
    )

    assert report.non_equivariant_keys == ()
    assert report.predicted_source_agrees
    assert report.max_absolute_source_logit_difference < 1e-4


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_graph_permutation_equivariance_holds_across_several_random_permutations(seed: int) -> None:
    model = _tiny_model()
    example = _example(nodes=6, source_local_index=3, seed=seed)
    permutation = _random_permutation(6, seed=seed * 17)

    report = measure_equivariance(
        model, example, permutation, collate_fn=collate_variable_topology, atol=1e-4
    )
    assert report.non_equivariant_keys == ()
    assert report.predicted_source_agrees


def test_source_region_logits_are_a_fixed_three_way_incident_classification() -> None:
    """core-issues.txt repair item 2: source_region_logits must be exactly 3
    incident-level logits (hydroswarm.training.corpus.SOURCE_REGION_COUNT),
    not one logit per node position -- shape must not scale with node count."""

    model = _tiny_model()
    small = _example(nodes=4, seed=1)
    large = _example(nodes=9, seed=2)

    with torch.no_grad():
        small_output = model(collate_variable_topology([small])[0])
        large_output = model(collate_variable_topology([large])[0])

    assert small_output["source_region_logits"].shape == (1, 3)
    assert large_output["source_region_logits"].shape == (1, 3)


def test_source_region_logits_are_invariant_under_node_permutation() -> None:
    """A genuine incident-level classification must not change at all when
    nodes are relabeled/reordered -- unlike source_node_logits (which is
    node-indexed and must permute equivariantly), source_region_logits has
    no node axis to permute."""

    model = _tiny_model()
    example = _example(nodes=6, source_local_index=3, seed=7)
    permutation = _random_permutation(6, seed=99)
    permuted_example = permute_example(example, permutation)

    model.eval()
    with torch.no_grad():
        original = model(collate_variable_topology([example])[0])["source_region_logits"]
        permuted = model(collate_variable_topology([permuted_example])[0])["source_region_logits"]

    torch.testing.assert_close(original, permuted, atol=1e-4, rtol=1e-4)
