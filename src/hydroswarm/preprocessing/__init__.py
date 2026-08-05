"""Deterministic simulator-to-model preprocessing."""

from .alignment import align_edges, align_node_features, canonical_node_order
from .batching import (
    NODE_SCALAR_FEATURE_KEYS,
    GraphSample,
    TimestampWindow,
    pad_graph_batch,
    timestamp_windows,
)
from .schema import (
    DEFAULT_FEATURE_SCHEMA,
    EDGE_FEATURE_NAMES,
    EDGE_FEATURE_SEMANTICS,
    FEATURE_SCHEMA_VERSION,
    NODE_FEATURE_NAMES,
    NODE_FEATURE_SEMANTICS,
    FeatureScope,
    FeatureSchema,
    FeatureSemantics,
    NormalizationStats,
)
from .builder import NO_NORMALIZATION_SENTINEL, BuiltHydroBatch, HydraulicFeatureBuilder, SensorSeries

__all__ = [
    "DEFAULT_FEATURE_SCHEMA",
    "NO_NORMALIZATION_SENTINEL",
    "BuiltHydroBatch",
    "HydraulicFeatureBuilder",
    "EDGE_FEATURE_NAMES",
    "EDGE_FEATURE_SEMANTICS",
    "FEATURE_SCHEMA_VERSION",
    "FeatureScope",
    "FeatureSemantics",
    "GraphSample",
    "NODE_FEATURE_NAMES",
    "NODE_SCALAR_FEATURE_KEYS",
    "NODE_FEATURE_SEMANTICS",
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
