"""Versioned candidate signatures over complete source hypotheses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

from hydroswarm.classical.prior import BayesianPosterior, SignatureLibrary, bayesian_source_posterior


@dataclass(frozen=True, slots=True, order=True)
class SourceHypothesis:
    source_node: str
    start_time_min: int
    duration_min: int
    relative_strength: float
    demand_regime: str

    def __post_init__(self) -> None:
        if not self.source_node or not self.demand_regime:
            raise ValueError("source node and demand regime are required")
        if self.start_time_min < 0 or self.duration_min <= 0 or self.relative_strength <= 0:
            raise ValueError("source timing, duration, and strength must be positive")

    @property
    def identifier(self) -> str:
        return (
            f"{self.source_node}|{self.start_time_min}|{self.duration_min}|"
            f"{self.relative_strength:g}|{self.demand_regime}"
        )


@dataclass(frozen=True, slots=True)
class SignatureCacheKey:
    network_hash: str
    hydraulic_state_hash: str
    simulator_version: str
    configuration_hash: str
    sensor_layout_hash: str
    schema_version: str = "hydroswarm-signatures-v1"

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SignatureArtifact:
    key: SignatureCacheKey
    library: SignatureLibrary
    hypotheses: tuple[SourceHypothesis, ...]
    sensor_nodes: tuple[str, ...]
    sample_times_seconds: tuple[int, ...]
    cache_hit: bool
    artifact_hash: str


class IncidentSimulator(Protocol):
    simulator_version: str

    def simulate_incident(
        self, source_node_id: str, *, strength_mg_min: float, start_minute: int, duration_minutes: int
    ) -> Any: ...


def _json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class SignatureCache:
    """Compressed `.npz` cache with strict metadata and corruption invalidation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: SignatureCacheKey) -> tuple[Path, Path]:
        return self.root / f"{key.digest}.npz", self.root / f"{key.digest}.json"

    def load(self, key: SignatureCacheKey) -> SignatureArtifact | None:
        arrays_path, metadata_path = self._paths(key)
        if not arrays_path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["key"] != asdict(key):
                return None
            expected_hash = metadata["artifact_hash"]
            if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != expected_hash:
                return None
            with np.load(arrays_path, allow_pickle=False) as archive:
                values = archive["signatures"]
            hypotheses = tuple(SourceHypothesis(**item) for item in metadata["hypotheses"])
            if values.shape[0] != len(hypotheses):
                return None
            library = SignatureLibrary()
            for hypothesis, signature in zip(hypotheses, values, strict=True):
                library.add(hypothesis.identifier, signature)
            return SignatureArtifact(
                key=key,
                library=library,
                hypotheses=hypotheses,
                sensor_nodes=tuple(metadata["sensor_nodes"]),
                sample_times_seconds=tuple(metadata["sample_times_seconds"]),
                cache_hit=True,
                artifact_hash=expected_hash,
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def store(
        self,
        key: SignatureCacheKey,
        library: SignatureLibrary,
        hypotheses: Sequence[SourceHypothesis],
        sensor_nodes: Sequence[str],
        sample_times_seconds: Sequence[int],
    ) -> SignatureArtifact:
        arrays_path, metadata_path = self._paths(key)
        values = np.stack([library.get(item.identifier) for item in hypotheses]).astype(np.float16)
        temporary = arrays_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, signatures=values)
        temporary.replace(arrays_path)
        artifact_hash = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        metadata = {
            "key": asdict(key),
            "hypotheses": [asdict(item) for item in hypotheses],
            "sensor_nodes": list(sensor_nodes),
            "sample_times_seconds": list(sample_times_seconds),
            "artifact_hash": artifact_hash,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return SignatureArtifact(
            key=key,
            library=library,
            hypotheses=tuple(hypotheses),
            sensor_nodes=tuple(sensor_nodes),
            sample_times_seconds=tuple(sample_times_seconds),
            cache_hit=False,
            artifact_hash=artifact_hash,
        )


class SignatureBuilder:
    def __init__(self, simulator: IncidentSimulator, cache: SignatureCache) -> None:
        self.simulator = simulator
        self.cache = cache

    def build_or_load(
        self,
        *,
        key: SignatureCacheKey,
        source_nodes: Sequence[str],
        start_time_bins: Sequence[int],
        duration_bins: Sequence[int],
        strength_bins: Sequence[float],
        demand_regimes: Sequence[str],
        sensor_nodes: Sequence[str],
        sample_times_seconds: Sequence[int],
        base_strength_mg_min: float = 10.0,
    ) -> SignatureArtifact:
        cached = self.cache.load(key)
        if cached is not None:
            return cached
        hypotheses = tuple(
            SourceHypothesis(*values)
            for values in product(
                source_nodes, start_time_bins, duration_bins, strength_bins, demand_regimes
            )
        )
        if not hypotheses or not sensor_nodes or not sample_times_seconds:
            raise ValueError("signature dimensions, sensors, and sample times must be non-empty")
        library = SignatureLibrary()
        for hypothesis in hypotheses:
            result = self.simulator.simulate_incident(
                hypothesis.source_node,
                strength_mg_min=base_strength_mg_min * hypothesis.relative_strength,
                start_minute=hypothesis.start_time_min,
                duration_minutes=hypothesis.duration_min,
            )
            frame = result.concentration_mg_l
            missing_sensors = set(sensor_nodes) - set(frame.columns)
            if missing_sensors:
                raise ValueError(f"simulation omitted sensors: {sorted(missing_sensors)}")
            aligned = frame.loc[:, list(sensor_nodes)].reindex(
                index=sample_times_seconds, method="nearest"
            )
            values = aligned.to_numpy(dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"simulation produced nonfinite signature for {hypothesis.identifier}")
            library.add(hypothesis.identifier, values)
        return self.cache.store(key, library, hypotheses, sensor_nodes, sample_times_seconds)


@dataclass(frozen=True, slots=True)
class CandidateExplanation:
    hypothesis: SourceHypothesis
    probability: float
    residual_rmse: float
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class ClassicalLocalizationResult:
    posterior: BayesianPosterior
    ranked_hypotheses: tuple[CandidateExplanation, ...]
    source_probabilities: Mapping[str, float]
    candidate_regions: tuple[tuple[str, ...], ...]
    start_time_interval_min: tuple[int, int]
    cache_hit: bool
    signature_artifact_hash: str


def classify_trace_residual(
    residual: ArrayLike, *, sensor_axis: int = -1
) -> str:
    values = np.asarray(residual, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return "insufficient_evidence"
    mean = float(values.mean())
    scale = float(values.std())
    if abs(mean) > max(0.05, 2.0 * scale):
        return "sensor_bias_or_strength_mismatch"
    if values.ndim >= 2 and np.mean(np.abs(np.diff(values, axis=0))) > max(0.05, scale * 1.5):
        return "timing_mismatch"
    sensor_means = np.mean(values, axis=tuple(i for i in range(values.ndim) if i != sensor_axis))
    if np.std(sensor_means) > max(0.05, scale):
        return "topology_or_demand_mismatch"
    return "ordinary_noise_or_wrong_source"


def localize_with_signatures(
    observations: ArrayLike,
    artifact: SignatureArtifact,
    *,
    observation_mask: ArrayLike | None = None,
    prior: Mapping[str, float] | None = None,
    feasible_sources: Mapping[str, bool] | None = None,
    noise_scale: float = 0.05,
    hydraulic_graph: nx.Graph | None = None,
    cumulative_mass: float = 0.9,
) -> ClassicalLocalizationResult:
    feasible_hypotheses = None
    if feasible_sources is not None:
        feasible_hypotheses = {
            item.identifier: feasible_sources.get(item.source_node, False) for item in artifact.hypotheses
        }
    posterior = bayesian_source_posterior(
        observations,
        artifact.library,
        prior=prior,
        observation_mask=observation_mask,
        noise_scale=noise_scale,
        feasible_mask=feasible_hypotheses,
    )
    by_id = {item.identifier: item for item in artifact.hypotheses}
    ranked: list[CandidateExplanation] = []
    source_probabilities: dict[str, float] = {}
    for identifier, probability in posterior.probabilities.items():
        hypothesis = by_id[str(identifier)]
        source_probabilities[hypothesis.source_node] = (
            source_probabilities.get(hypothesis.source_node, 0.0) + probability
        )
        rmse = posterior.residual_rmse[identifier]
        feasible = feasible_sources is None or feasible_sources.get(hypothesis.source_node, False)
        ranked.append(CandidateExplanation(
            hypothesis=hypothesis,
            probability=probability,
            residual_rmse=rmse,
            status="accepted" if feasible and probability > 0 else "rejected",
            reason=("trace-compatible posterior" if feasible else "hydraulically infeasible"),
        ))
    ranked.sort(key=lambda item: item.probability, reverse=True)
    selected_sources: list[str] = []
    mass = 0.0
    for source, probability in sorted(source_probabilities.items(), key=lambda item: item[1], reverse=True):
        selected_sources.append(source)
        mass += probability
        if mass >= cumulative_mass:
            break
    if hydraulic_graph is None:
        regions = tuple((source,) for source in selected_sources)
    else:
        induced = hydraulic_graph.to_undirected().subgraph(selected_sources)
        regions = tuple(tuple(sorted(component)) for component in nx.connected_components(induced))
    credible = [item.hypothesis.start_time_min for item in ranked if item.probability > 0.01]
    interval = (min(credible), max(credible)) if credible else (0, 0)
    return ClassicalLocalizationResult(
        posterior=posterior,
        ranked_hypotheses=tuple(ranked),
        source_probabilities=source_probabilities,
        candidate_regions=regions,
        start_time_interval_min=interval,
        cache_hit=artifact.cache_hit,
        signature_artifact_hash=artifact.artifact_hash,
    )
