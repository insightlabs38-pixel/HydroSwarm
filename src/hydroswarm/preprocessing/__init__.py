"""Deterministic simulator-to-model preprocessing."""

from .alignment import align_edges, align_node_features, canonical_node_order
from .batching import GraphSample, TimestampWindow, pad_graph_batch, timestamp_windows
from .schema import (
    DEFAULT_FEATURE_SCHEMA,
    EDGE_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    NODE_FEATURE_NAMES,
    FeatureSchema,
    NormalizationStats,
)
from .builder import BuiltHydroBatch, HydraulicFeatureBuilder, SensorSeries

__all__ = [
    "DEFAULT_FEATURE_SCHEMA",
    "BuiltHydroBatch",
    "HydraulicFeatureBuilder",
    "EDGE_FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "GraphSample",
    "NODE_FEATURE_NAMES",
    "FeatureSchema",
    "NormalizationStats",
    "SensorSeries",
    "TimestampWindow",
    "align_edges",
    "align_node_features",
    "canonical_node_order",
    "pad_graph_batch",
    "timestamp_windows",
]
