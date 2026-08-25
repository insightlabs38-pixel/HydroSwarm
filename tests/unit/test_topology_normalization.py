"""Permutation/relabeling-invariance tests for the experimental topology-
relative feature augmentation (`hydroswarm.model.topology_normalization`).

See docs/evaluation/experimental/TOPOLOGY_GENERALIZATION_EXPERIMENT_PLAN.md.
Not wired into any production default; these tests exist to verify the
augmentation itself never breaks the equivariance guarantee the rest of
HydroCore already has (`tests/unit/test_permutation.py`), since a
representation change that broke permutation equivariance would silently
reintroduce exactly the "brittle absolute node identity" failure mode this
experiment is trying to move away from.
"""

from __future__ import annotations

import random

import pytest
import torch

from hydroswarm.model import HydroCore
from hydroswarm.model.topology_normalization import (
    EDGE_TOPOLOGY_RELATIVE_COLUMNS,
    NODE_TOPOLOGY_RELATIVE_COLUMNS,
    augment_batch,
    augment_with_relative_scale,
    augmented_width,
)
from hydroswarm.preprocessing.schema import (
    EDGE_FEATURE_NAMES,
    EDGE_FEATURE_SEMANTICS,
    NODE_FEATURE_NAMES,
    NODE_FEATURE_SEMANTICS,
    FeatureScope,
)
from hydroswarm.training import (
    CurriculumStage,
    ScenarioExample,
    TopologyMetadata,
    collate_variable_topology,
    measure_equivariance,
    permute_example,
    resolve_source_node_id,
)


# NODE_TOPOLOGY_RELATIVE_COLUMNS/EDGE_TOPOLOGY_RELATIVE_COLUMNS are indices
# into the real 19-/13-wide schema.py feature vectors -- the toy examples
# below must use those same real widths (not a smaller ad hoc size) or
# indexing into the topology-relative columns would go out of bounds.
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)


def _augmented_model() -> HydroCore:
    return HydroCore(
        node_feature_dim=augmented_width(NODE_FEATURE_DIM, NODE_TOPOLOGY_RELATIVE_COLUMNS),
        edge_feature_dim=augmented_width(EDGE_FEATURE_DIM, EDGE_TOPOLOGY_RELATIVE_COLUMNS),
        temporal_feature_dim=2,
        quality_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        latent_tokens=64,
        adapter_dims=(32, 32, 32),
        dropout=0.0,
    )


def _augmented_collate(examples):
    inputs, targets = collate_variable_topology(examples)
    return augment_batch(inputs), targets


def _example(seed: int = 1, nodes: int = 5, source_local_index: int = 2) -> ScenarioExample:
    """Mirrors tests/unit/test_permutation.py's `_example` helper but with
    node/edge feature widths matching the real schema.py contract
    (NODE_FEATURE_DIM/EDGE_FEATURE_DIM), since NODE_TOPOLOGY_RELATIVE_COLUMNS/
    EDGE_TOPOLOGY_RELATIVE_COLUMNS are schema-derived indices that assume
    those real widths."""

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
            "node_features": torch.randn(nodes, NODE_FEATURE_DIM, generator=generator),
            "temporal_features": torch.randn(steps, nodes, 2, generator=generator),
            "quality_features": torch.randn(steps, nodes, 2, generator=generator),
            "edge_index": edge_index,
            "edge_features": torch.randn(len(edges), EDGE_FEATURE_DIM, generator=generator),
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


# ---------------------------------------------------------------------------
# Column selection stays derived from schema.py, never hand-duplicated.
# ---------------------------------------------------------------------------


def test_topology_relative_columns_match_schema_scope() -> None:
    expected_node = tuple(
        index
        for index, name in enumerate(NODE_FEATURE_NAMES)
        if NODE_FEATURE_SEMANTICS[name].scope == FeatureScope.TOPOLOGY_RELATIVE
    )
    expected_edge = tuple(
        index
        for index, name in enumerate(EDGE_FEATURE_NAMES)
        if EDGE_FEATURE_SEMANTICS[name].scope == FeatureScope.TOPOLOGY_RELATIVE
    )
    assert NODE_TOPOLOGY_RELATIVE_COLUMNS == expected_node
    assert EDGE_TOPOLOGY_RELATIVE_COLUMNS == expected_edge
    # Sanity: every TOPOLOGY_RELATIVE-tagged feature really is one of the two
    # sets above, not silently dropped by an off-by-one.
    assert set(expected_node) <= set(range(len(NODE_FEATURE_NAMES)))
    assert len(expected_node) > 0 and len(expected_edge) > 0


# ---------------------------------------------------------------------------
# augment_with_relative_scale: pure-function invariances.
# ---------------------------------------------------------------------------


def test_augment_returns_features_unchanged_when_none_or_no_columns() -> None:
    assert augment_with_relative_scale(None, None, NODE_TOPOLOGY_RELATIVE_COLUMNS) is None
    features = torch.randn(2, 4, 5)
    assert augment_with_relative_scale(features, None, ()) is features


def test_augment_appends_columns_and_preserves_original() -> None:
    features = torch.randn(2, 4, 6)
    columns = (1, 3)
    out = augment_with_relative_scale(features, None, columns)
    assert out.shape == (2, 4, 6 + len(columns))
    torch.testing.assert_close(out[..., :6], features)


def test_augment_masks_padding_out_of_scale_statistic_and_zeros_padded_rows() -> None:
    features = torch.zeros(1, 3, 2)
    features[0, 0, 0] = 1.0
    features[0, 1, 0] = 1000.0  # padded row -- must not distort the scale
    features[0, 2, 0] = 2.0
    mask = torch.tensor([[True, False, True]])
    out = augment_with_relative_scale(features, mask, (0,))
    scale = 2.0  # max(|1.0|, |2.0|) over valid rows only, floor 1.0
    assert out[0, 0, -1].item() == pytest.approx(1.0 / scale)
    assert out[0, 2, -1].item() == pytest.approx(2.0 / scale)
    assert out[0, 1, -1].item() == 0.0  # padded row zeroed, not left at 1000/scale


def test_augment_is_permutation_equivariant() -> None:
    torch.manual_seed(0)
    features = torch.randn(2, 6, 4)
    mask = torch.rand(2, 6) > 0.2
    permutation = torch.randperm(6)

    direct = augment_with_relative_scale(features, mask, (0, 2))
    permuted_first = augment_with_relative_scale(features[:, permutation], mask[:, permutation], (0, 2))
    torch.testing.assert_close(permuted_first, direct[:, permutation])


# ---------------------------------------------------------------------------
# End-to-end: the augmented HydroCore forward pass stays permutation-
# equivariant, mirroring tests/unit/test_permutation.py's own coverage.
# ---------------------------------------------------------------------------


def test_augmented_model_graph_permutation_equivariance() -> None:
    model = _augmented_model()
    example = _example(nodes=5, source_local_index=2, seed=99)
    permutation = _random_permutation(5, seed=2026)

    report = measure_equivariance(model, example, permutation, collate_fn=_augmented_collate, atol=1e-4)

    assert report.non_equivariant_keys == ()
    assert report.predicted_source_agrees
    assert report.max_absolute_source_logit_difference < 1e-4


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_augmented_model_equivariance_holds_across_several_permutations(seed: int) -> None:
    model = _augmented_model()
    example = _example(nodes=6, source_local_index=3, seed=seed)
    permutation = _random_permutation(6, seed=seed * 17)

    report = measure_equivariance(model, example, permutation, collate_fn=_augmented_collate, atol=1e-4)
    assert report.non_equivariant_keys == ()
    assert report.predicted_source_agrees


def test_augmented_model_predicted_real_world_source_survives_relabeling() -> None:
    """The augmented model's belief about *which real-world node* is the
    source must not change under relabeling, even though the source's local
    array index does change -- the failure mode a brittle absolute-node-
    identity representation would produce. Mirrors
    tests/unit/test_permutation.py::test_permutation_remaps_source_node_and_preserves_real_world_identity
    but exercised through the augmented representation's actual model
    predictions, not just the relabeling utility."""

    example = _example(nodes=6, source_local_index=3, seed=11)
    permutation = _random_permutation(6, seed=4)
    relabeled = permute_example(example, permutation)
    assert permutation != list(range(6))  # otherwise this test proves nothing

    model = _augmented_model()
    model.eval()
    with torch.no_grad():
        original_inputs, _ = _augmented_collate([example])
        relabeled_inputs, _ = _augmented_collate([relabeled])
        original_prediction = int(model(original_inputs)["source_node_logits"][0].argmax())
        relabeled_prediction = int(model(relabeled_inputs)["source_node_logits"][0].argmax())

    original_predicted_id = example.topology.node_ids[original_prediction]
    relabeled_predicted_id = relabeled.topology.node_ids[relabeled_prediction]
    assert original_predicted_id == relabeled_predicted_id
    # And the true source's own real-world identity is unchanged too, per
    # the existing relabeling utility's own contract.
    assert resolve_source_node_id(example) == resolve_source_node_id(relabeled)
