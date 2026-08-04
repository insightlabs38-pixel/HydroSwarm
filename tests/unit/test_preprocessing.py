from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from hydroswarm.preprocessing import (
    DEFAULT_FEATURE_SCHEMA,
    EDGE_FEATURE_NAMES,
    EDGE_FEATURE_SEMANTICS,
    NODE_FEATURE_NAMES,
    NODE_FEATURE_SEMANTICS,
    GraphSample,
    NormalizationStats,
    align_edges,
    align_node_features,
    canonical_node_order,
    pad_graph_batch,
    timestamp_windows,
)


def test_canonical_node_and_edge_alignment_is_order_independent() -> None:
    assert canonical_node_order(["J10", "J2", "J1"]) == ("J1", "J10", "J2")
    aligned, mask, node_ids = align_node_features(
        ["B", "A"], [[2.0], [1.0]], ["A", "B", "C"]
    )
    assert node_ids == ("A", "B", "C")
    assert aligned[:2, 0].tolist() == [1.0, 2.0]
    assert mask.tolist() == [True, True, False]
    assert np.isnan(aligned[2, 0])

    edge_index, features = align_edges(
        [("B", "A"), ("A", "C")], [[2.0], [1.0]], ["A", "B", "C"]
    )
    assert edge_index.tolist() == [[0, 1], [2, 0]]
    assert features[:, 0].tolist() == [1.0, 2.0]


def test_feature_schema_and_normalization_are_versioned_and_nan_safe() -> None:
    assert len(NODE_FEATURE_NAMES) == 19
    assert len(EDGE_FEATURE_NAMES) == 13
    assert len(DEFAULT_FEATURE_SCHEMA.fingerprint) == 64
    nodes = np.zeros((2, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    nodes[1, 1] = np.nan
    DEFAULT_FEATURE_SCHEMA.validate_node_array(nodes)
    stats = NormalizationStats.fit(nodes, NODE_FEATURE_NAMES)
    transformed = stats.transform(nodes)
    assert transformed.shape == nodes.shape
    assert np.isfinite(transformed).all()
    with pytest.raises(ValueError, match="exactly"):
        DEFAULT_FEATURE_SCHEMA.validate_edge_array(np.zeros((2, 2)))


def test_normalization_stats_save_load_round_trip_and_hash(tmp_path) -> None:
    nodes = np.random.default_rng(0).normal(size=(5, len(NODE_FEATURE_NAMES))).astype(np.float32)
    stats = NormalizationStats.fit(nodes, NODE_FEATURE_NAMES)
    output = tmp_path / "node-normalization.json"
    digest = stats.save(output)
    assert digest == stats.fingerprint
    assert output.exists()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    assert sidecar.read_text(encoding="utf-8").strip() == digest

    reloaded = NormalizationStats.load(output)
    assert reloaded.feature_names == stats.feature_names
    assert reloaded.schema_version == stats.schema_version
    np.testing.assert_array_equal(reloaded.mean, stats.mean)
    np.testing.assert_array_equal(reloaded.scale, stats.scale)
    assert reloaded.fingerprint == digest


def test_normalization_fingerprint_changes_when_statistics_differ() -> None:
    a = NormalizationStats.fit(np.zeros((3, len(NODE_FEATURE_NAMES)), dtype=np.float32), NODE_FEATURE_NAMES)
    b = NormalizationStats.fit(np.ones((3, len(NODE_FEATURE_NAMES)), dtype=np.float32), NODE_FEATURE_NAMES)
    assert a.fingerprint != b.fingerprint


def test_every_node_and_edge_feature_has_documented_semantics() -> None:
    assert set(NODE_FEATURE_SEMANTICS) == set(NODE_FEATURE_NAMES)
    assert set(EDGE_FEATURE_SEMANTICS) == set(EDGE_FEATURE_NAMES)
    for semantics in {**NODE_FEATURE_SEMANTICS, **EDGE_FEATURE_SEMANTICS}.values():
        assert semantics.unit
        assert semantics.scope in ("absolute", "topology_relative")


def test_timestamp_windows_require_order_and_preserve_missing_mask() -> None:
    start = datetime(2026, 8, 3, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=index * 5) for index in range(4)]
    values = np.arange(8, dtype=np.float32).reshape(4, 2, 1)
    values[1, 0, 0] = np.nan
    windows = timestamp_windows(timestamps, values, window_size=3, stride=1)
    assert len(windows) == 2
    assert windows[0].timestamps_s.tolist() == [0.0, 300.0, 600.0]
    assert windows[0].mask[:, 0].tolist() == [True, False, True]
    with pytest.raises(ValueError, match="non-decreasing"):
        timestamp_windows(timestamps[::-1], values, window_size=2)


def test_padded_graph_batch_masks_nodes_edges_and_time() -> None:
    first = GraphSample(
        node_features=torch.ones(2, 3),
        temporal_features=torch.ones(2, 2, 2),
        quality_features=torch.ones(2, 2, 1),
        edge_index=torch.tensor([[0], [1]]),
        edge_features=torch.ones(1, 4),
        timestamps=torch.tensor([0.0, 5.0]),
    )
    second = GraphSample(
        node_features=torch.ones(3, 3),
        temporal_features=torch.ones(1, 3, 2),
        quality_features=torch.ones(1, 3, 1),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_features=torch.ones(2, 4),
    )
    batch = pad_graph_batch([first, second])
    assert batch["node_features"].shape == (2, 3, 3)
    assert batch["edge_index"].shape == (2, 2, 2)
    assert batch["node_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["edge_mask"].tolist() == [[True, False], [True, True]]
    assert batch["sensor_mask"][1, 1].sum() == 0

