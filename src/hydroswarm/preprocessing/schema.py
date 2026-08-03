"""Versioned numeric feature contracts for simulator-to-model boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FEATURE_SCHEMA_VERSION = "hydroswarm-features-v1"

NODE_FEATURE_NAMES = (
    "node_type",
    "elevation",
    "base_demand",
    "current_demand",
    "pressure",
    "hydraulic_head",
    "estimated_concentration",
    "sensor_presence",
    "sensor_health",
    "measurement_age",
    "missingness",
    "classical_source_prior",
    "arrival_time_residual",
    "positive_sensor_consistency",
    "negative_sensor_consistency",
    "hydraulic_zone",
    "distance_to_reservoir",
    "distance_to_sensor",
    "isolation_region",
)

EDGE_FEATURE_NAMES = (
    "pipe_type",
    "length",
    "diameter",
    "roughness",
    "flow_magnitude",
    "flow_direction",
    "velocity",
    "estimated_travel_time",
    "valve_state",
    "pump_state",
    "flow_reversal",
    "current_operability",
    "pipe_volume",
)


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    version: str = FEATURE_SCHEMA_VERSION
    node_features: tuple[str, ...] = NODE_FEATURE_NAMES
    edge_features: tuple[str, ...] = EDGE_FEATURE_NAMES

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("feature schema version must not be empty")
        if len(set(self.node_features)) != len(self.node_features):
            raise ValueError("node feature names must be unique")
        if len(set(self.edge_features)) != len(self.edge_features):
            raise ValueError("edge feature names must be unique")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "version": self.version,
                "node_features": self.node_features,
                "edge_features": self.edge_features,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def validate_node_array(self, values: ArrayLike) -> NDArray[np.float32]:
        return _validate(values, len(self.node_features), "node")

    def validate_edge_array(self, values: ArrayLike) -> NDArray[np.float32]:
        return _validate(values, len(self.edge_features), "edge")


def _validate(values: ArrayLike, width: int, kind: str) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 2 or array.shape[-1] != width:
        raise ValueError(f"{kind} features must end with exactly {width} columns")
    if np.isinf(array).any():
        raise ValueError(f"{kind} features cannot contain infinity")
    return array


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    feature_names: tuple[str, ...]
    mean: NDArray[np.float32]
    scale: NDArray[np.float32]
    schema_version: str = FEATURE_SCHEMA_VERSION

    @classmethod
    def fit(
        cls,
        values: ArrayLike,
        feature_names: Iterable[str],
        *,
        schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> NormalizationStats:
        names = tuple(feature_names)
        array = np.asarray(values, dtype=np.float32)
        if array.ndim < 2 or array.shape[-1] != len(names):
            raise ValueError("normalization data width must match feature names")
        axes = tuple(range(array.ndim - 1))
        valid = np.isfinite(array)
        count = valid.sum(axis=axes).clip(min=1)
        clean = np.where(valid, array, 0.0)
        mean = (clean.sum(axis=axes) / count).astype(np.float32)
        centered = np.where(valid, array - mean, 0.0)
        scale = np.sqrt((centered * centered).sum(axis=axes) / count).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(names, mean, scale, schema_version)

    def transform(self, values: ArrayLike) -> NDArray[np.float32]:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != len(self.feature_names):
            raise ValueError("normalization input width does not match statistics")
        normalized = (array - self.mean) / self.scale
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )


DEFAULT_FEATURE_SCHEMA = FeatureSchema()
