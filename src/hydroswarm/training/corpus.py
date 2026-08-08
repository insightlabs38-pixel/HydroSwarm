"""Leakage-controlled conversion from WNTR scenarios to HydroCore tensors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch

from hydroswarm.classical import HydraulicStateEstimator, OperationalTelemetry
from hydroswarm.data.scenarios import EventType, GeneratedScenario
from hydroswarm.preprocessing import HydraulicFeatureBuilder, SensorSeries
from hydroswarm.preprocessing.schema import NormalizationStats
from hydroswarm.simulation import build_wntr_network
from hydroswarm.simulation.wrapper import FEATURE_SNAPSHOT_TIME_SECONDS, HydraulicSimulator

from .data import CurriculumStage, ScenarioExample, TopologyMetadata
from .targets_v2 import TARGETS_V2_SCHEMA_VERSION, EventCause

#: Number of source_region buckets (see assign_source_regions).
SOURCE_REGION_COUNT = 3


STAGE_MAP = {
    "clean": CurriculumStage.CLEAN,
    "operational": CurriculumStage.OPERATIONAL,
    "degraded": CurriculumStage.DEGRADED,
    "distribution_shift": CurriculumStage.SHIFT,
    "adversarial": CurriculumStage.ADVERSARIAL,
}


@dataclass(frozen=True, slots=True)
class SignatureLibrary:
    """Training-only concentration templates used by the classical baseline."""

    node_ids: tuple[str, ...]
    signatures: Mapping[str, np.ndarray]
    manifest_hash: str

    def posterior(self, scenario: GeneratedScenario) -> np.ndarray:
        observed, valid = aligned_observations(scenario, self.node_ids)
        return self.posterior_from_observations(observed, valid)

    def posterior_from_observations(self, observed: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """The governed MODEL-INPUT ``classical_prior`` algorithm
        (core-issues5.txt delta item 1): per-source-node log1p-residual
        softmax. ``posterior()`` (training, from a ``GeneratedScenario``'s
        raw arrays) and live serving's ``model_input_classical_prior``
        (from live ``SensorSeries`` evidence via
        ``aligned_observations_from_series``) both call this one
        implementation so the two paths can never independently drift --
        before this fix, live serving instead called
        ``hydroswarm.classical.signatures.localize_with_signatures``, a
        structurally different Bayesian-posterior algorithm over a
        different (hypothesis-grid) signature representation, which is a
        real train/serve input-distribution skew HydroCore's
        ``classical_prior`` feature is directly sensitive to (see
        ``scripts/run_train_serve_parity_gate.py``'s module docstring for
        the original finding)."""

        transformed = np.log1p(np.nan_to_num(observed, nan=0.0))
        residuals = []
        for node_id in self.node_ids:
            signature = self.signatures[node_id]
            comparable = valid & np.isfinite(signature)
            if not comparable.any():
                residuals.append(1e6)
            else:
                residuals.append(float(np.mean((transformed[comparable] - signature[comparable]) ** 2)))
        values = -np.asarray(residuals, dtype=np.float64)
        values -= values.max()
        probability = np.exp(values / max(float(np.std(values)), 0.05))
        return (probability / probability.sum()).astype(np.float32)


@dataclass(frozen=True, slots=True)
class FeatureContext:
    """Reusable canonical hydraulic state and graph for a network family."""

    state: Any
    graph: Any


def assign_source_regions(network: Any, *, num_regions: int = SOURCE_REGION_COUNT) -> dict[str, int]:
    """Deterministic hop-distance-from-reservoir/tank region partition.

    Nodes at similar hydraulic distance from the network's feed points tend
    to share travel-time and contamination-spread characteristics, so
    grouping by hop-distance band is a reasonable, fully-deterministic proxy
    for source_region -- it depends only on static network structure, never
    on any particular incident, so it cannot leak incident-specific
    information and is identical for a given network across every scenario.
    """

    graph = network.to_graph().to_undirected()
    feed_points = [node for node in (*network.reservoir_name_list, *network.tank_name_list) if node in graph]
    if not feed_points:
        feed_points = [sorted(graph.nodes)[0]] if graph.nodes else []

    distances: dict[str, int] = {}
    for feed_point in feed_points:
        for node, distance in nx.single_source_shortest_path_length(graph, feed_point).items():
            if node not in distances or distance < distances[node]:
                distances[node] = distance

    junctions = sorted(network.junction_name_list)
    unique_distances = sorted({distances.get(node, 0) for node in junctions})
    bucket_of_distance = {
        distance: min(num_regions - 1, (index * num_regions) // max(len(unique_distances), 1))
        for index, distance in enumerate(unique_distances)
    }
    return {node: bucket_of_distance[distances.get(node, 0)] for node in junctions}


def build_feature_context(network: Any) -> FeatureContext:
    simulator = HydraulicSimulator(network)
    simulated = simulator.calculate_state(FEATURE_SNAPSHOT_TIME_SECONDS)
    state = HydraulicStateEstimator().estimate(simulated, OperationalTelemetry())
    return FeatureContext(state=state, graph=simulator.build_dynamic_graph(simulated))


def _hydraulic_state_hash(state: Any) -> str:
    """Fingerprints the estimated hydraulic state itself (pressure/demand/
    flow/tank-level/pump/valve values) -- distinct from network_hash, which
    fingerprints the network configuration that produced it
    (HydraulicSimulator.state_hash()). Two scenarios could in principle
    share a network_hash but differ in estimated state (different
    telemetry/estimation inputs), or vice versa; this is core-issues.txt
    repair item 5's hydraulic_state_hash field."""

    payload = {
        "pressure_m": {key: value.estimate for key, value in state.pressure_m.items()},
        "demand_m3s": {key: value.estimate for key, value in state.demand_m3s.items()},
        "flow_m3s": {key: value.estimate for key, value in state.flow_m3s.items()},
        "tank_level_m": {key: value.estimate for key, value in state.tank_level_m.items()},
        "pump_open": dict(state.pump_open),
        "valve_open": dict(state.valve_open),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def aligned_observations(
    scenario: GeneratedScenario, node_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    steps = len(scenario.timestamps_seconds)
    values = np.full((steps, len(node_ids)), np.nan, dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    for source_column, node_id in enumerate(scenario.sensor_nodes):
        target_column = positions[node_id]
        values[:, target_column] = scenario.observed_concentration[:, source_column]
        valid[:, target_column] = scenario.observation_mask[:, source_column]
    return values, valid


def aligned_observations_from_series(
    node_ids: Sequence[str],
    series: Sequence[Any],
    target_timestamps_seconds: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Live-serving equivalent of ``aligned_observations``: builds the same
    ``(steps, node)`` observation grid from live ``SensorSeries`` evidence
    (``hydroswarm.preprocessing.SensorSeries``) instead of a
    ``GeneratedScenario``'s raw arrays, resampled onto
    ``target_timestamps_seconds`` by nearest-time matching -- the same
    resampling convention
    ``HybridInferencePipeline._signature_observations`` already uses to
    align live evidence onto a *different* (hypothesis-grid) target time
    axis. Feeding real telemetry timestamps (which need not coincide with
    any training scenario's own report timesteps) through this function is
    what lets ``model_input_classical_prior`` reuse
    ``SignatureLibrary.posterior_from_observations`` unchanged
    (core-issues5.txt delta item 1)."""

    targets = np.asarray(target_timestamps_seconds, dtype=float)
    if targets.size == 0:
        raise ValueError("target_timestamps_seconds must not be empty")
    values = np.full((targets.size, len(node_ids)), np.nan, dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    by_node = {item.node_id: item for item in series}
    for node_id, column in positions.items():
        item = by_node.get(node_id)
        if item is None:
            continue
        timestamps = np.asarray(item.timestamps_seconds, dtype=float)
        if timestamps.size == 0:
            continue
        for row, target in enumerate(targets):
            source_index = int(np.argmin(np.abs(timestamps - target)))
            value = item.concentration_mg_l[source_index]
            is_missing = item.missing[source_index]
            if value is not None and not is_missing and np.isfinite(value):
                values[row, column] = float(value)
                valid[row, column] = True
    return values, valid


def fit_signature_library(
    scenarios: Sequence[GeneratedScenario], node_ids: Sequence[str]
) -> SignatureLibrary:
    """Fit source signatures from training scenarios only."""

    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for scenario in scenarios:
        if scenario.manifest.split.value != "train":
            raise ValueError("signature fitting accepts training scenarios only")
        source = scenario.manifest.incident.source_nodes[0]
        observed, _ = aligned_observations(scenario, node_ids)
        grouped[source].append(np.log1p(observed))
    missing = set(node_ids) - set(grouped)
    if missing:
        raise ValueError(f"training corpus has no signatures for sources: {sorted(missing)}")
    signatures = {}
    for node_id in node_ids:
        values = np.stack(grouped[node_id])
        counts = np.isfinite(values).sum(axis=0)
        sums = np.nansum(values, axis=0)
        mean = np.full(sums.shape, np.nan, dtype=np.float64)
        np.divide(sums, counts, out=mean, where=counts > 0)
        signatures[node_id] = mean.astype(np.float32)
    payload = {
        node_id: np.nan_to_num(value, nan=-1.0).round(7).tolist()
        for node_id, value in signatures.items()
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SignatureLibrary(tuple(node_ids), signatures, digest)


#: Default location of the committed per-topology-family MODEL-INPUT
#: signature libraries (data/learning-v2/cycle-b2 is a protected artifact;
#: this module only ever reads from it, never writes).
DEFAULT_MODEL_INPUT_SIGNATURE_ROOT = Path("data/learning-v2/cycle-b2")


def load_committed_signature_library(
    cycle_b2_root: str | Path, network_family: str, node_ids: Sequence[str]
) -> SignatureLibrary:
    """Load a topology family's already-fit, already-committed
    ``SignatureLibrary`` straight from
    ``<cycle_b2_root>/signatures/<network_family>.json`` -- the exact real
    training-fit artifact ``fit_signature_library`` produced and Stage-F's
    training tensors were built from -- rather than re-simulating and
    refitting (cross-architecture EPANET floating-point divergence makes
    scenario regeneration untrustworthy on some hosts; see
    ``scripts/generate_ood_extension_corpus.py``'s own original docstring
    for this same reasoning, which this function was factored out of so
    live serving -- ``resolve_model_input_signature_library`` below -- and
    every corpus/OOD-extension caller share one implementation rather than
    two independently-maintained copies (core-issues5.txt delta item 1)).

    Self-consistency check, not a re-derivation from scratch:
    ``fit_signature_library``'s own hashing convention
    (``np.nan_to_num(value, nan=-1.0).round(7).tolist()`` per node,
    JSON-serialized with sorted keys) is applied to the values already
    stored in the file and compared against that same file's own recorded
    ``sha256`` -- this proves the file parses correctly and its hash truly
    describes the values sitting next to it, without requiring any
    simulation at all.
    """

    recorded_path = Path(cycle_b2_root) / "signatures" / f"{network_family}.json"
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    stored_node_ids = tuple(recorded["node_ids"])
    if stored_node_ids != tuple(node_ids):
        raise ValueError(
            f"{network_family!r}: signatures/{network_family}.json node_ids do not match the "
            "supplied junction list -- refusing to use a mismatched signature artifact"
        )
    stored = {node_id: np.asarray(values, dtype=np.float32) for node_id, values in recorded["signatures"].items()}
    hash_payload = {node_id: value.round(7).tolist() for node_id, value in stored.items()}
    recomputed_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recomputed_hash != recorded["sha256"]:
        raise ValueError(
            f"{network_family!r}: signatures/{network_family}.json's stored values do not reproduce its "
            f"own recorded sha256 (recomputed {recomputed_hash}, recorded {recorded['sha256']}) -- the "
            "file is corrupted or was hand-edited; refusing to use it"
        )
    # fit_signature_library's own sentinel: -1.0 stands in for NaN
    # (np.nan_to_num(nan=-1.0)) because JSON has no NaN literal.
    runtime_signatures = {
        node_id: np.where(value == -1.0, np.nan, value).astype(np.float32) for node_id, value in stored.items()
    }
    return SignatureLibrary(node_ids=stored_node_ids, signatures=runtime_signatures, manifest_hash=recorded["sha256"])


def reference_training_timestamps_seconds(network: Any) -> np.ndarray:
    """The canonical scenario observation-window timestamps a topology
    family's committed signature library
    (``data/learning-v2/cycle-b2/signatures/<family>.json``) was fit
    against.

    ``WNTRScenarioGenerator.generate_with_network`` never randomizes
    simulation TIMING (only demand/roughness/tank/pipe state -- see
    ``_randomize_hydraulics``); a scenario's own ``timestamps_seconds`` is
    simply ``simulate_incident(...).concentration_mg_l.index``, which
    depends only on the network's own WNTR time configuration
    (``report_timestep``/``duration``), not on which source/start/duration/
    strength was simulated. Reproduced here by running one real
    ``simulate_incident`` call rather than hand-deriving the report-time
    formula, so this can never silently drift from what
    ``generate_with_network`` actually does."""

    junctions = sorted(network.junction_name_list)
    if not junctions:
        raise ValueError("network has no junction source candidates")
    simulation = HydraulicSimulator(network).simulate_incident(junctions[0])
    return np.asarray(simulation.concentration_mg_l.index, dtype=np.float64)


#: The exact same three topology-family loaders
#: ``scripts/generate_cycle_b_corpus.py``'s own ``TRAIN_TOPOLOGIES`` uses,
#: mirrored here (``training/corpus.py`` cannot import from ``scripts/``)
#: so ``resolve_model_input_signature_library`` can load a FRESH pristine
#: reference network for a ``GOVERNED_KNOWN_NETWORK`` match, rather than
#: trusting the served ``network`` object's own WNTR time configuration --
#: ``network_sha256`` deliberately hashes only node/link topology and
#: link roughness/length/diameter (see its own docstring), NOT
#: ``options.time`` (report_timestep/duration), so a hash-matched served
#: network could in principle carry different simulation timing than the
#: pristine network the committed signature library was actually fit
#: against (this is exactly what several `HybridInferencePipeline` test
#: fixtures do: build the real golden-reference topology but shorten its
#: simulation duration for fast tests). Kept in sync with
#: ``KNOWN_TRAINING_TOPOLOGY_FAMILY_BY_HASH`` by
#: ``tests/unit/test_signature_policy.py``'s own hash-reproduction test.
def _load_pristine_reference_network(family: str) -> Any:
    if family == "golden-reference":
        return build_wntr_network()
    import wntr

    if family == "branched-loop":
        return wntr.network.WaterNetworkModel("data/topology-transfer/branched-loop.inp")
    if family == "loop-grid":
        return wntr.network.WaterNetworkModel("data/topologies/loop-grid.inp")
    raise ValueError(f"unknown known-training-topology family: {family!r}")


def fit_runtime_signature_library(network: Any) -> SignatureLibrary:
    """RUNTIME_GENERATED_IMPORTED_NETWORK equivalent of
    ``fit_signature_library`` for the MODEL-INPUT ``classical_prior``
    algorithm (core-issues5.txt delta item 1) -- one ``simulate_incident``
    per junction source against a network this policy's training corpus
    never saw, mirroring
    ``hydroswarm.classical.signature_policy``'s already-established
    two-mode pattern (``GOVERNED_KNOWN_NETWORK`` vs.
    ``RUNTIME_GENERATED_IMPORTED_NETWORK``) for the *other* (hypothesis-grid)
    signature artifact. Deterministically reproducible, but MUST be labeled
    ``RUNTIME_GENERATED_IMPORTED_NETWORK`` by callers -- never presented as
    equivalent to a training-owned artifact fit from many independent
    scenarios."""

    junctions = tuple(sorted(network.junction_name_list))
    if not junctions:
        raise ValueError("network has no junction source candidates")
    simulator = HydraulicSimulator(network)
    signatures: dict[str, np.ndarray] = {}
    for node_id in junctions:
        simulation = simulator.simulate_incident(node_id)
        frame = simulation.concentration_mg_l.loc[:, list(junctions)]
        signatures[node_id] = np.log1p(frame.to_numpy(dtype=np.float32))
    payload = {
        node_id: np.nan_to_num(value, nan=-1.0).round(7).tolist() for node_id, value in signatures.items()
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SignatureLibrary(junctions, signatures, digest)


#: In-process cache of resolved model-input signature libraries, keyed by
#: topology_hash -- avoids re-simulating (either loading + hashing a
#: committed artifact, or, for imported networks, re-running one
#: simulate_incident per junction) on every single HybridInferencePipeline.
#: analyze() call for what is, for a fixed topology_hash, always the exact
#: same deterministic result. Not persisted to disk: purely a same-process
#: performance cache, safe to lose on restart.
_MODEL_INPUT_SIGNATURE_CACHE: dict[str, tuple[SignatureLibrary, tuple[float, ...], str]] = {}


def resolve_model_input_signature_library(
    topology_hash: str,
    node_ids: Sequence[str],
    network: Any,
    *,
    cycle_b2_root: str | Path = DEFAULT_MODEL_INPUT_SIGNATURE_ROOT,
) -> tuple[SignatureLibrary, tuple[float, ...], str]:
    """Resolve the governed MODEL-INPUT ``classical_prior`` signature
    library for a served network (core-issues5.txt delta item 1).

    Returns ``(library, reference_timestamps_seconds, mode)`` where
    ``mode`` is a ``hydroswarm.classical.signature_policy.SignatureMode``
    string. For ``GOVERNED_KNOWN_NETWORK``, ``library`` is the real
    training-fit ``SignatureLibrary`` loaded from the committed
    ``data/learning-v2/cycle-b2/signatures/<family>.json`` artifact -- the
    exact object ``scenario_to_example`` uses to build Stage-F training
    tensors. For ``RUNTIME_GENERATED_IMPORTED_NETWORK``, no governed
    training-owned artifact exists; ``library`` is a
    ``fit_runtime_signature_library`` best-effort approximation using the
    identical algorithm, and callers must record/surface the mode rather
    than silently presenting it as training-owned provenance."""

    from hydroswarm.classical.signature_policy import (
        KNOWN_TRAINING_TOPOLOGY_FAMILY_BY_HASH,
        resolve_signature_mode,
    )

    cached = _MODEL_INPUT_SIGNATURE_CACHE.get(topology_hash)
    if cached is not None and cached[0].node_ids == tuple(node_ids):
        return cached

    mode = resolve_signature_mode(topology_hash)
    if mode == "GOVERNED_KNOWN_NETWORK":
        family = KNOWN_TRAINING_TOPOLOGY_FAMILY_BY_HASH[topology_hash]
        # Reference timestamps come from a FRESH pristine copy of the
        # training topology, not the served `network` object -- see
        # _load_pristine_reference_network's own docstring for why.
        reference_timestamps = tuple(
            map(float, reference_training_timestamps_seconds(_load_pristine_reference_network(family)))
        )
        library = load_committed_signature_library(cycle_b2_root, family, node_ids)
    else:
        reference_timestamps = tuple(map(float, reference_training_timestamps_seconds(network)))
        library = fit_runtime_signature_library(network)
    result = (library, reference_timestamps, mode)
    _MODEL_INPUT_SIGNATURE_CACHE[topology_hash] = result
    return result


def model_input_classical_prior(
    library: SignatureLibrary,
    node_ids: Sequence[str],
    series: Sequence[Any],
    target_timestamps_seconds: Sequence[float],
) -> dict[str, float]:
    """The MODEL-INPUT ``classical_prior`` HydroCore actually consumes,
    computed with the exact same algorithm Stage-F training tensors were
    built with (``SignatureLibrary.posterior_from_observations``) --
    distinct from
    ``hydroswarm.inference.pipeline.HybridInferencePipeline``'s richer
    ``live_classical_localization`` (``localize_with_signatures``'s
    Bayesian posterior over the runtime hypothesis-grid artifact), which
    remains available for deterministic reasoning/fusion/operator evidence
    but must never be silently substituted here (core-issues5.txt delta
    item 1)."""

    observed, valid = aligned_observations_from_series(node_ids, series, target_timestamps_seconds)
    vector = library.posterior_from_observations(observed, valid)
    return dict(zip(library.node_ids, map(float, vector), strict=True))


#: Canonical integer ordering for the event_cause target.
EVENT_CAUSE_INDEX: dict[EventCause, int] = {cause: index for index, cause in enumerate(EventCause)}


#: core-issues3.txt Phase 6.4 / item K: EventCause.HYDRAULIC_MISMATCH and
#: AMBIGUOUS are defined in the governed taxonomy but have no reproducible
#: generator behind them yet -- see _event_cause's docstring. Runtime/
#: promotion code must treat these as unsupported (never assume the
#: event_cause head was actually trained to distinguish them) until one is
#: implemented and this set is updated to match.
UNSUPPORTED_EVENT_CAUSES: frozenset[EventCause] = frozenset({EventCause.HYDRAULIC_MISMATCH, EventCause.AMBIGUOUS})
SUPPORTED_EVENT_CAUSES: frozenset[EventCause] = frozenset(EventCause) - UNSUPPORTED_EVENT_CAUSES


def _event_cause(scenario: GeneratedScenario) -> EventCause:
    """Deterministic event_cause derivation from the generator's event_type
    -- never a hand label. Only ever returns a SUPPORTED_EVENT_CAUSES
    member.

    core-issues3.txt Phase 6.4 / item K: previously returned
    HYDRAULIC_MISMATCH whenever model_mismatch['valve_telemetry_incorrect']
    or ['missing_topology_element'] was set -- but
    hydroswarm.data.scenarios sets 'valve_telemetry_incorrect' purely from
    `config.stage in {SHIFT, ADVERSARIAL}` (a curriculum-stage *label*)
    with NO corresponding simulated valve/pump/topology perturbation
    behind it (confirmed by inspection of WNTRScenarioGenerator.
    generate_with_network: the flag is set in the `mismatch` dict handed
    to ScenarioManifest, never fed into `_randomize_hydraulics` or any
    other function that would make the actual simulated network disagree
    with its own telemetry). Every such "normal" scenario was therefore a
    genuinely quiet, internally-consistent network mislabeled as
    HYDRAULIC_MISMATCH -- training the event_cause head to associate that
    class with curriculum-stage-correlated spurious features rather than
    any real hydraulic/telemetry discrepancy present in the input. Fixed
    by removing the flag from label derivation entirely: a normal-event
    scenario is always NORMAL until a real, reproducible mismatch
    perturbation exists (Phase 6.4's "implement...or remove" choice --
    remove was taken here). AMBIGUOUS was already, separately, never
    produced (unchanged)."""

    event_type = scenario.manifest.event_type
    if event_type == EventType.CONTAMINATION.value:
        return EventCause.CONTAMINATION
    if event_type == EventType.SENSOR_FAULT_ONLY.value:
        return EventCause.SENSOR_FAULT
    return EventCause.NORMAL


def sensor_health_summary(
    series: Sequence[SensorSeries], *, health_threshold: float = 0.75
) -> tuple[float, int]:
    """(healthy_fraction, sensors_ever_healthy) over `series` -- the raw
    sensor-health signal both `_evidence_sufficiency`'s sensor-health-only
    rule and hydroswarm.training.control_labels' fuller rule are built
    from, factored out so both compute it identically rather than risking
    drift between two independent implementations."""

    all_health = [value for item in series for value in item.health]
    if not all_health:
        return 0.0, 0
    healthy_fraction = sum(1 for value in all_health if value >= health_threshold) / len(all_health)
    sensors_ever_healthy = sum(1 for item in series if any(value >= health_threshold for value in item.health))
    return healthy_fraction, sensors_ever_healthy


def _evidence_sufficiency(series: Sequence[SensorSeries], *, health_threshold: float = 0.75) -> bool:
    """Documented deterministic rule (overnight-plan.txt Task 2.2): evidence
    is sufficient when at least half of all (sensor, timestep) health
    readings clear health_threshold and at least two distinct sensors ever
    clear it. This is the sensor-health-based subset of the plan's full
    rule (which also references calibrated candidate-set size, posterior
    entropy, disagreement, and OOD state). hydroswarm.training.control_labels.
    classify_evidence_sufficiency now extends this with posterior entropy
    and OOD-category calibration validity (core-issues2.txt Phase 5) --
    calibrated candidate-set size specifically still cannot be computed
    here: it requires a CalibrationArtifact already fitted against a
    trained Sentinel checkpoint, which does not exist until after Stage 1
    of Phase 8's staged training sequence. This narrower rule remains the
    one scenario_to_example calls at pure corpus-generation time, before
    any model exists to calibrate against.
    """

    healthy_fraction, sensors_ever_healthy = sensor_health_summary(series, health_threshold=health_threshold)
    return healthy_fraction >= 0.5 and sensors_ever_healthy >= 2


def build_sensor_series(scenario: GeneratedScenario, context: FeatureContext) -> list[SensorSeries]:
    """Real per-sensor observation series for ``scenario``, built against
    ``context``'s hydraulic state -- the same construction
    ``scenario_to_example`` uses to build its stored features, extracted so
    a live-pipeline re-analysis of a governed scenario (e.g. fitting
    calibration against the exact deployed dynamic hybrid predictor rather
    than a post-hoc approximation -- core-issues.txt Phase 3 item 18) can
    reuse it exactly instead of re-deriving it and risking drift."""

    series: list[SensorSeries] = []
    for source_column, node_id in enumerate(scenario.sensor_nodes):
        valid = scenario.observation_mask[:, source_column]
        frozen = scenario.frozen_mask[:, source_column]
        outage = scenario.communication_outage_mask[:, source_column]
        concentration = scenario.observed_concentration[:, source_column]
        pressure = context.state.pressure_m[node_id].estimate
        series.append(
            SensorSeries(
                node_id=node_id,
                timestamps_seconds=tuple(map(float, scenario.timestamps_seconds)),
                concentration_mg_l=tuple(
                    float(value) if is_valid else None
                    for value, is_valid in zip(concentration, valid, strict=True)
                ),
                pressure_m=tuple(pressure if is_valid else None for is_valid in valid),
                health=tuple(
                    0.0 if not is_valid else 0.25 if is_frozen or is_outage else 1.0
                    for is_valid, is_frozen, is_outage in zip(valid, frozen, outage, strict=True)
                ),
                missing=tuple(map(bool, ~valid)),
                drift=tuple(map(bool, frozen)),
                delayed=tuple(map(bool, outage)),
            )
        )
    return series


def scenario_to_example(
    scenario: GeneratedScenario,
    network: Any,
    signature_library: SignatureLibrary,
    *,
    feature_context: FeatureContext | None = None,
    node_normalization: NormalizationStats | None = None,
    edge_normalization: NormalizationStats | None = None,
) -> ScenarioExample:
    junction_ids = tuple(sorted(network.junction_name_list))
    if junction_ids != signature_library.node_ids:
        raise ValueError("scenario network nodes do not match the signature library")
    context = feature_context or build_feature_context(network)
    series = build_sensor_series(scenario, context)
    prior_values = signature_library.posterior(scenario)
    prior = dict(zip(junction_ids, map(float, prior_values), strict=True))
    built = HydraulicFeatureBuilder(
        node_normalization=node_normalization, edge_normalization=edge_normalization
    ).build(
        network,
        context.graph,
        context.state,
        series,
        classical_prior=prior,
        window_steps=len(scenario.timestamps_seconds),
    )
    node_ids = built.node_ids
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    start_bins = (0, 60, 120, 240)
    duration_bins = (30, 60, 120)
    strength_bins = (0.5, 1.0, 2.0)
    split = scenario.manifest.split.value

    event_presence = scenario.manifest.event_type == EventType.CONTAMINATION.value
    cause = _event_cause(scenario)
    regions = assign_source_regions(network)
    if event_presence:
        source_node_id = scenario.manifest.incident.source_nodes[0]
        source = positions[source_node_id]
        source_region = regions[source_node_id]
        start_time = start_bins.index(scenario.manifest.incident.start_minute)
        duration = duration_bins.index(scenario.manifest.incident.duration_minutes)
        relative_strength = strength_bins.index(scenario.manifest.incident.relative_strength)
    else:
        # No real source exists for a normal/sensor-fault-only scenario;
        # these are placeholders masked out by the *_mask companions below,
        # never invented labels a loss could train against.
        source = source_region = start_time = duration = relative_strength = 0

    # core-issues.txt repair item 5: every generated example must carry its
    # own non-null topology provenance, not the None default corpus
    # generation left every example with previously.
    edge_ids = tuple(
        (network.get_link(name).start_node_name, network.get_link(name).end_node_name)
        for name in sorted(network.link_name_list)
    )
    source_candidate_ids = tuple(node_id for node_id in node_ids if node_id in network.junction_name_list)
    topology_metadata = TopologyMetadata(
        # Stable across every scenario sharing this topology family: hashes
        # the pristine network passed into WNTRScenarioGenerator.generate
        # (see ScenarioManifest.network_sha256), not this scenario's own
        # randomized hydraulics.
        topology_hash=scenario.manifest.network_sha256,
        # This scenario's own exact (possibly randomized) network
        # configuration -- demand pattern, roughness, tank levels, pipe
        # status -- as actually used to build its features (item 4).
        network_hash=HydraulicSimulator(network).state_hash(),
        node_ids=node_ids,
        edge_ids=edge_ids,
        source_candidate_ids=source_candidate_ids,
        hydraulic_state_hash=_hydraulic_state_hash(context.state),
        signature_library_hash=signature_library.manifest_hash,
        target_schema_version=TARGETS_V2_SCHEMA_VERSION,
        feature_schema_version=built.feature_schema_version,
    )

    return ScenarioExample(
        scenario_id=str(scenario.manifest.scenario_id),
        network_id=scenario.manifest.network_id,
        split=split,
        seed=scenario.manifest.seed,
        seed_family=f"{scenario.manifest.network_family}:{scenario.manifest.seed_family}",
        stage=STAGE_MAP[scenario.manifest.stage.value],
        inputs={key: value.squeeze(0) for key, value in built.batch.items()},
        targets={
            "source_node": torch.tensor(source),
            "source_node_mask": torch.tensor(event_presence),
            "source_region": torch.tensor(source_region),
            "source_region_mask": torch.tensor(event_presence),
            "start_time": torch.tensor(start_time),
            "start_time_mask": torch.tensor(event_presence),
            "duration": torch.tensor(duration),
            "duration_mask": torch.tensor(event_presence),
            "relative_strength": torch.tensor(relative_strength),
            "relative_strength_mask": torch.tensor(event_presence),
            "event_presence": torch.tensor(event_presence),
            "event_cause": torch.tensor(EVENT_CAUSE_INDEX[cause]),
            "evidence_sufficiency": torch.tensor(_evidence_sufficiency(series)),
            "sensor_fault": torch.tensor([
                float(
                    node_id in scenario.sensor_nodes
                    and (
                        scenario.frozen_mask[:, scenario.sensor_nodes.index(node_id)].any()
                        or scenario.communication_outage_mask[
                            :, scenario.sensor_nodes.index(node_id)
                        ].any()
                        # core-issues.txt repair item 3: drift and unit-mismatch
                        # are real, generated fault modes (see
                        # GeneratedScenario.drift_mask/unit_mismatch_mask) that
                        # this target's own definition already names
                        # ("frozen, drifting, or in communication outage") but
                        # previously never checked.
                        or scenario.drift_mask[:, scenario.sensor_nodes.index(node_id)].any()
                        or scenario.unit_mismatch_mask[:, scenario.sensor_nodes.index(node_id)].any()
                    )
                )
                for node_id in node_ids
            ]),
            # core-issues.txt repair item 3: sensor_fault is only ever
            # meaningful for a node that actually has a sensor -- an
            # unsensored node's "0.0" above is a placeholder, not a real
            # "healthy" observation, and must not be trained against or
            # counted in prevalence/audit statistics as if it were one.
            "sensor_fault_mask": torch.tensor([node_id in scenario.sensor_nodes for node_id in node_ids]),
        },
        topology=topology_metadata,
    )


def write_tensor_manifest(
    path: str | Path,
    examples: Sequence[ScenarioExample],
    *,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            record = {
                "scenario_id": example.scenario_id,
                "network_id": example.network_id,
                "split": example.split,
                "seed": example.seed,
                "seed_family": example.seed_family,
                "stage": example.stage.name,
                "inputs": {key: value.tolist() for key, value in example.inputs.items()},
                "targets": {key: value.tolist() for key, value in example.targets.items()},
                "metadata": dict((metadata or {}).get(example.scenario_id, {})),
            }
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(line)
            hasher.update(line.encode())
    return hasher.hexdigest()


def signature_metadata(library: SignatureLibrary) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "node_ids": list(library.node_ids),
        "sha256": library.manifest_hash,
        "signatures": {
            node_id: np.nan_to_num(value, nan=-1.0).tolist()
            for node_id, value in library.signatures.items()
        },
    }
