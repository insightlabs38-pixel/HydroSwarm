"""Canonical network alignment independent of EPANET iteration order."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

NodeId = Hashable


def canonical_node_order(node_ids: Iterable[NodeId]) -> tuple[NodeId, ...]:
    values = tuple(node_ids)
    if len(set(values)) != len(values):
        raise ValueError("node IDs must be unique")
    return tuple(sorted(values, key=lambda value: (type(value).__name__, str(value))))


def align_node_features(
    node_ids: Sequence[NodeId],
    values: ArrayLike,
    canonical_ids: Sequence[NodeId] | None = None,
    *,
    fill_value: float = np.nan,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], tuple[NodeId, ...]]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 2 or array.shape[0] != len(node_ids):
        raise ValueError("node feature first dimension must match node_ids")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node_ids must be unique")
    canonical = canonical_node_order(node_ids) if canonical_ids is None else tuple(canonical_ids)
    if len(set(canonical)) != len(canonical):
        raise ValueError("canonical_ids must be unique")
    locations = {node_id: index for index, node_id in enumerate(canonical)}
    unknown = set(node_ids) - set(canonical)
    if unknown:
        raise ValueError(f"nodes absent from canonical order: {sorted(map(str, unknown))}")
    aligned = np.full((len(canonical), *array.shape[1:]), fill_value, dtype=np.float32)
    mask = np.zeros(len(canonical), dtype=bool)
    for source_index, node_id in enumerate(node_ids):
        target_index = locations[node_id]
        aligned[target_index] = array[source_index]
        mask[target_index] = True
    return aligned, mask, canonical


def align_edges(
    edges: Sequence[tuple[NodeId, NodeId]],
    edge_features: ArrayLike,
    canonical_ids: Sequence[NodeId],
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    features = np.asarray(edge_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(edges):
        raise ValueError("edge feature rows must match edges")
    positions = {node_id: index for index, node_id in enumerate(canonical_ids)}
    indexed: list[tuple[int, int, int]] = []
    for row, (source, target) in enumerate(edges):
        if source not in positions or target not in positions:
            raise ValueError("edge endpoint missing from canonical node order")
        indexed.append((positions[source], positions[target], row))
    indexed.sort(key=lambda item: (item[0], item[1], item[2]))
    edge_index = np.asarray([(source, target) for source, target, _ in indexed], dtype=np.int64).T
    if not indexed:
        edge_index = np.empty((2, 0), dtype=np.int64)
    rows = [row for _, _, row in indexed]
    return edge_index, features[rows]

