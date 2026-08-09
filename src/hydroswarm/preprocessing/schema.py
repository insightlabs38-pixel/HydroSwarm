"""Versioned numeric feature contracts for simulator-to-model boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FEATURE_SCHEMA_VERSION = "hydroswarm-features-v2"

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

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @property
    def fingerprint(self) -> str:
        text = json.dumps(self._payload(), indent=2, sort_keys=True) + "\n"
        return hashlib.sha256(text.encode()).hexdigest()

    def save(self, path: str | Path) -> str:
        """Persist as a governed JSON artifact plus a `.sha256` sidecar
        (matching the existing reports/results/*.json / *.json.sha256
        convention). Returns the artifact hash so callers can record it
        alongside a run's provenance (e.g. ExperimentRegistry.manifest_hashes
        or a checkpoint's metadata)."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._payload(), indent=2, sort_keys=True) + "\n"
        # newline="" -- Path.write_text's default newline=None translates
        # every "\n" in `text` to os.linesep (CRLF on Windows) as it
        # writes, but `digest` below is hashed from the pre-translation
        # string. Without this, the sidecar records a hash that only ever
        # matches the file's actual bytes on POSIX; a strict byte-level
        # verifier (e.g. hydroswarm.runtime.v4_normalization, which
        # deliberately hashes path.read_bytes() rather than round-tripping
        # through text mode) fails closed on every native Windows write.
        path.write_text(text, encoding="utf-8", newline="")
        digest = hashlib.sha256(text.encode()).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(
            digest + "\n", encoding="utf-8", newline=""
        )
        return digest

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            schema_version=payload["schema_version"],
        )


class FeatureScope:
    ABSOLUTE = "absolute"
    TOPOLOGY_RELATIVE = "topology_relative"


@dataclass(frozen=True, slots=True)
class FeatureSemantics:
    unit: str
    scope: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.scope not in (FeatureScope.ABSOLUTE, FeatureScope.TOPOLOGY_RELATIVE):
            raise ValueError(f"unknown feature scope: {self.scope}")


# Task 0.6 requires documenting, per feature, whether it is a physically
# absolute quantity (comparable across any topology -- e.g. concentration
# thresholds, typical pipe velocity ranges) or a topology-relative quantity
# whose meaningful scale depends on that network's size/terrain/demand (e.g.
# raw elevation, travel time, graph distance). Fitting normalization on the
# training split only (never on validation/calibration/development/OOD/
# locked test) is enforced procedurally by scripts/fit_normalization.py,
# which only accepts a single manifest labeled split="train"; this table is
# the accompanying documentation of what each fitted statistic actually
# represents.
NODE_FEATURE_SEMANTICS: dict[str, FeatureSemantics] = {
    "node_type": FeatureSemantics("category", FeatureScope.ABSOLUTE, "junction/tank/reservoir enum"),
    "elevation": FeatureSemantics(
        "meters", FeatureScope.TOPOLOGY_RELATIVE, "meaningful only relative to that network's own reference elevation"
    ),
    "base_demand": FeatureSemantics("m3/s", FeatureScope.TOPOLOGY_RELATIVE, "scales with network size"),
    "current_demand": FeatureSemantics("m3/s", FeatureScope.TOPOLOGY_RELATIVE, "scales with network size"),
    "pressure": FeatureSemantics("meters (head)", FeatureScope.ABSOLUTE, "operating pressure ranges are comparable across networks"),
    "hydraulic_head": FeatureSemantics(
        "meters", FeatureScope.TOPOLOGY_RELATIVE, "inherits elevation's topology dependency"
    ),
    "estimated_concentration": FeatureSemantics("mg/L", FeatureScope.ABSOLUTE, "contamination thresholds are absolute health quantities"),
    "sensor_presence": FeatureSemantics("indicator [0,1]", FeatureScope.ABSOLUTE),
    "sensor_health": FeatureSemantics("indicator [0,1]", FeatureScope.ABSOLUTE),
    "measurement_age": FeatureSemantics("seconds", FeatureScope.ABSOLUTE, "sensor staleness has an absolute operational meaning"),
    "missingness": FeatureSemantics("indicator [0,1]", FeatureScope.ABSOLUTE),
    "classical_source_prior": FeatureSemantics("probability [0,1]", FeatureScope.ABSOLUTE),
    "arrival_time_residual": FeatureSemantics("seconds", FeatureScope.TOPOLOGY_RELATIVE, "depends on network scale/travel time"),
    "positive_sensor_consistency": FeatureSemantics("indicator [0,1]", FeatureScope.ABSOLUTE),
    "negative_sensor_consistency": FeatureSemantics("indicator [0,1]", FeatureScope.ABSOLUTE),
    "hydraulic_zone": FeatureSemantics("category", FeatureScope.ABSOLUTE, "structural zone id"),
    "distance_to_reservoir": FeatureSemantics("graph hops or meters", FeatureScope.TOPOLOGY_RELATIVE, "depends on network scale"),
    "distance_to_sensor": FeatureSemantics("graph hops or meters", FeatureScope.TOPOLOGY_RELATIVE, "depends on network scale"),
    "isolation_region": FeatureSemantics("category", FeatureScope.ABSOLUTE, "structural region id"),
}

EDGE_FEATURE_SEMANTICS: dict[str, FeatureSemantics] = {
    "pipe_type": FeatureSemantics("category", FeatureScope.ABSOLUTE),
    "length": FeatureSemantics("meters", FeatureScope.TOPOLOGY_RELATIVE, "scales with network size"),
    "diameter": FeatureSemantics("millimeters", FeatureScope.ABSOLUTE, "standard pipe diameters are comparable across networks"),
    "roughness": FeatureSemantics("Hazen-Williams C", FeatureScope.ABSOLUTE, "standard material property range"),
    "flow_magnitude": FeatureSemantics("m3/s", FeatureScope.TOPOLOGY_RELATIVE, "scales with network demand"),
    "flow_direction": FeatureSemantics("sign {-1,0,1}", FeatureScope.ABSOLUTE, "structural, not a magnitude"),
    "velocity": FeatureSemantics("m/s", FeatureScope.ABSOLUTE, "safe operating velocity ranges are comparable across networks"),
    "estimated_travel_time": FeatureSemantics("seconds", FeatureScope.TOPOLOGY_RELATIVE, "depends on network scale"),
    "valve_state": FeatureSemantics("category", FeatureScope.ABSOLUTE),
    "pump_state": FeatureSemantics("category", FeatureScope.ABSOLUTE),
    "flow_reversal": FeatureSemantics("indicator [0,1]", FeatureScope.ABSOLUTE),
    "current_operability": FeatureSemantics("indicator [0,1]", FeatureScope.ABSOLUTE),
    "pipe_volume": FeatureSemantics("m3", FeatureScope.TOPOLOGY_RELATIVE, "depends on pipe length and diameter"),
}


DEFAULT_FEATURE_SCHEMA = FeatureSchema()
