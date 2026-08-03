"""Timestamp windowing and padded, masked hydraulic graph batches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TimestampWindow:
    timestamps_s: NDArray[np.float64]
    values: NDArray[np.float32]
    mask: NDArray[np.bool_]


def timestamp_windows(
    timestamps: Sequence[datetime | float],
    values: ArrayLike,
    *,
    window_size: int,
    stride: int = 1,
) -> tuple[TimestampWindow, ...]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] != len(timestamps):
        raise ValueError("timestamp count must match first value dimension")
    seconds = np.asarray([_seconds(value) for value in timestamps], dtype=np.float64)
    if not np.all(np.isfinite(seconds)) or np.any(np.diff(seconds) < 0):
        raise ValueError("timestamps must be finite and non-decreasing")
    if len(seconds) < window_size:
        return ()
    windows = []
    for start in range(0, len(seconds) - window_size + 1, stride):
        stop = start + window_size
        chunk = array[start:stop]
        windows.append(
            TimestampWindow(
                timestamps_s=seconds[start:stop] - seconds[start],
                values=np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0),
                mask=np.isfinite(chunk).all(axis=-1),
            )
        )
    return tuple(windows)


def _seconds(value: datetime | float) -> float:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("datetime timestamps must be timezone-aware")
        return value.timestamp()
    return float(value)


@dataclass(frozen=True, slots=True)
class GraphSample:
    node_features: Tensor
    temporal_features: Tensor
    quality_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    timestamps: Tensor | None = None

    def __post_init__(self) -> None:
        nodes = self.node_features.shape[0]
        if self.node_features.ndim != 2:
            raise ValueError("node_features must be [nodes, features]")
        if self.temporal_features.ndim != 3 or self.temporal_features.shape[1] != nodes:
            raise ValueError("temporal_features must be [time, nodes, features]")
        if self.quality_features.ndim != 3 or self.quality_features.shape[:2] != self.temporal_features.shape[:2]:
            raise ValueError("quality_features must share [time, nodes]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2, edges]")
        if self.edge_features.ndim != 2 or self.edge_features.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_features rows must match edge_index")
        if self.edge_index.numel() and (self.edge_index.min() < 0 or self.edge_index.max() >= nodes):
            raise ValueError("edge_index contains an invalid node index")


def pad_graph_batch(samples: Sequence[GraphSample]) -> dict[str, Tensor]:
    if not samples:
        raise ValueError("at least one graph sample is required")
    node_width = samples[0].node_features.shape[-1]
    temporal_width = samples[0].temporal_features.shape[-1]
    quality_width = samples[0].quality_features.shape[-1]
    edge_width = samples[0].edge_features.shape[-1]
    if any(
        sample.node_features.shape[-1] != node_width
        or sample.temporal_features.shape[-1] != temporal_width
        or sample.quality_features.shape[-1] != quality_width
        or sample.edge_features.shape[-1] != edge_width
        for sample in samples
    ):
        raise ValueError("all samples must use the same feature widths")
    batch = len(samples)
    max_nodes = max(sample.node_features.shape[0] for sample in samples)
    max_steps = max(sample.temporal_features.shape[0] for sample in samples)
    max_edges = max(sample.edge_index.shape[1] for sample in samples)
    device = samples[0].node_features.device
    node = torch.zeros(batch, max_nodes, node_width, device=device)
    temporal = torch.zeros(batch, max_steps, max_nodes, temporal_width, device=device)
    quality = torch.zeros(batch, max_steps, max_nodes, quality_width, device=device)
    edge_index = torch.zeros(batch, 2, max_edges, dtype=torch.long, device=device)
    edge = torch.zeros(batch, max_edges, edge_width, device=device)
    node_mask = torch.zeros(batch, max_nodes, dtype=torch.bool, device=device)
    sensor_mask = torch.zeros(batch, max_steps, max_nodes, dtype=torch.bool, device=device)
    quality_mask = torch.zeros(batch, max_steps, max_nodes, dtype=torch.bool, device=device)
    edge_mask = torch.zeros(batch, max_edges, dtype=torch.bool, device=device)
    timestamps = torch.zeros(batch, max_steps, device=device)
    for index, sample in enumerate(samples):
        nodes = sample.node_features.shape[0]
        steps = sample.temporal_features.shape[0]
        edges = sample.edge_index.shape[1]
        node[index, :nodes] = sample.node_features
        temporal[index, :steps, :nodes] = torch.nan_to_num(sample.temporal_features)
        quality[index, :steps, :nodes] = torch.nan_to_num(sample.quality_features)
        node_mask[index, :nodes] = True
        sensor_mask[index, :steps, :nodes] = torch.isfinite(sample.temporal_features).all(-1)
        quality_mask[index, :steps, :nodes] = torch.isfinite(sample.quality_features).all(-1)
        if edges:
            edge_index[index, :, :edges] = sample.edge_index
            edge[index, :edges] = sample.edge_features
            edge_mask[index, :edges] = True
        if sample.timestamps is None:
            timestamps[index, :steps] = torch.arange(steps, device=device)
        else:
            if sample.timestamps.shape != (steps,):
                raise ValueError("timestamps must have one value per time step")
            timestamps[index, :steps] = sample.timestamps
    return {
        "node_features": node,
        "temporal_features": temporal,
        "quality_features": quality,
        "edge_index": edge_index,
        "edge_features": edge,
        "node_mask": node_mask,
        "sensor_mask": sensor_mask,
        "quality_mask": quality_mask,
        "edge_mask": edge_mask,
        "timestamps": timestamps,
    }
