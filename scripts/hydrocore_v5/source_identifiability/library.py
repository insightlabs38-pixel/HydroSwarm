"""Per-incident candidate-signature library construction (Phase 2 I/O layer).

This is the only module in the package that calls into
`HydraulicSimulator`/EPANET. Everything else (`signatures.py`,
`centrality.py`, `oracle.py`) is pure computation over the arrays this
module produces, following the repo's own `*_common.py` (I/O) vs.
`*_analysis_lib.py` (pure) split convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import common, signatures

#: Conditions whose stress is fully reproducible by replaying
#: `ScenarioGenerationConfig` (their `generator_config` differs from
#: NOMINAL) -- `scenario.observed_concentration` for these already reflects
#: the exact same degradation the frozen M11.6 evaluation used.
CONFIG_REPRODUCIBLE_CONDITIONS = frozenset({"NOMINAL", "MEASUREMENT_NOISE", "SEVERITY_SHIFT"})

#: Conditions whose `generator_config` is byte-identical to NOMINAL in the
#: locked scenario specs -- their stress is applied by the M11.6 evaluator
#: through a separate multi-round sampling/health mechanism outside
#: `ScenarioGenerationConfig` (see docs/evaluation/
#: SOURCE_IDENTIFIABILITY_ANALYSIS_PROTOCOL.md Section 5/6 note). This
#: module applies a documented, clearly-labeled APPROXIMATION of each,
#: directly from the `condition` metadata already recorded on the row, so
#: the stress-vs-clean comparison is not silently skipped for them -- but
#: results for these four are never presented as byte-identical replication.
APPROXIMATED_CONDITIONS = frozenset(
    {"SENSOR_DROPOUT", "LOW_COVERAGE_ACTIVE_SAMPLING", "SENSOR_HEALTH_DEGRADED", "AMBIGUITY_DISAGREEMENT"}
)


@dataclass(frozen=True, slots=True)
class IncidentBundle:
    row: dict[str, Any]
    incident: common.ReconstructedIncident
    signature_set: signatures.SignatureSet
    full_node_traces: dict[str, np.ndarray]  # candidate -> (n_times, n_all_nodes) mg/L
    all_node_ids: tuple[str, ...]
    observed_sensor_matrix: np.ndarray  # (n_times, n_sensors) log1p(mg/L), condition-stressed
    observed_mask: np.ndarray  # bool, True = usable
    stress_treatment: str  # "config_exact" | "approximated" | "clean_only"
    noise_floor_distance: float


def _noise_floor_distance(row: dict[str, Any]) -> float:
    """A physically motivated absolute log1p-scale distance floor.

    Two candidate signatures separated by less than the sensor's own
    reporting resolution (quantization) plus its measurement-noise standard
    deviation are, in principle, not distinguishable from a single reading
    -- independent of any data-driven percentile choice.
    """

    generator_config = row.get("generator_config") or {}
    noise_std = float(generator_config.get("sensor_noise_std", 0.01))
    quantization = float(generator_config.get("quantization_step", 0.001))
    return float(np.log1p(noise_std) + quantization)


def _condition_rng(row: dict[str, Any]) -> np.random.Generator:
    # Deterministic, disjoint from the generation seed's own rng stream
    # (which is fully consumed inside `reconstruct_incident`) -- offsetting
    # by a fixed odd constant keeps this reproducible without colliding
    # with `WNTRScenarioGenerator`'s own seed space.
    return np.random.default_rng((int(row["seed"]) ^ 0x5F3759DF) & 0xFFFFFFFFFFFFFFFF)


def _approximate_condition_stress(
    observed: np.ndarray, sensor_nodes: tuple[str, ...], row: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Documented approximation for the four conditions not reachable via
    `ScenarioGenerationConfig` (see module docstring). Operates on the
    already-`_degrade`d NOMINAL-level observed matrix."""

    condition_kind = row["condition_kind"]
    condition = row.get("condition") or {}
    mask = np.ones_like(observed, dtype=bool)
    rng = _condition_rng(row)
    result = observed.copy()

    if condition_kind == "SENSOR_DROPOUT":
        missing_rate = float(condition.get("missing_rate", 0.0))
        mask &= rng.random(observed.shape) >= missing_rate
    elif condition_kind == "LOW_COVERAGE_ACTIVE_SAMPLING":
        coverage = float(condition.get("coverage", 1.0))
        keep = max(1, round(coverage * len(sensor_nodes)))
        keep_columns = np.sort(rng.choice(len(sensor_nodes), size=keep, replace=False))
        column_mask = np.zeros(len(sensor_nodes), dtype=bool)
        column_mask[keep_columns] = True
        mask &= column_mask[np.newaxis, :]
    elif condition_kind == "SENSOR_HEALTH_DEGRADED":
        health_fraction = float(condition.get("health_fraction", 0.0))
        n_frozen = max(0, round(health_fraction * len(sensor_nodes)))
        frozen_columns = rng.choice(len(sensor_nodes), size=n_frozen, replace=False)
        for column in frozen_columns:
            result[:, column] = result[0, column]
    # AMBIGUITY_DISAGREEMENT: no known generation-time mechanism: this
    # incident was selected by the original evaluator because it produced
    # classical/neural disagreement, not because of a distinct injected
    # perturbation -- analyzed at nominal-level observation stress.
    return result, mask


def build_incident_bundle(row: dict[str, Any]) -> IncidentBundle:
    incident = common.reconstruct_incident(row)
    all_node_ids = tuple(sorted(incident.randomized_network.node_name_list))
    candidates = incident.junctions

    full_traces: dict[str, np.ndarray] = {}
    sensor_only: dict[str, np.ndarray] = {}
    timestamps: tuple[int, ...] | None = None
    for candidate in candidates:
        result = common.simulate_candidate(incident, candidate)
        frame = result.concentration_mg_l.reindex(columns=list(all_node_ids))
        if timestamps is None:
            timestamps = tuple(int(t) for t in frame.index)
        full_traces[candidate] = frame.to_numpy(dtype=np.float64)
        sensor_only[candidate] = frame.loc[:, list(incident.sensor_nodes)].to_numpy(dtype=np.float64)
    assert timestamps is not None

    signature_set = signatures.build_signature_set(
        sensor_only, sensor_nodes=incident.sensor_nodes, timestamps_seconds=timestamps
    )

    observed = np.log1p(np.asarray(incident.scenario.observed_concentration, dtype=np.float64))
    observed_mask = np.asarray(incident.scenario.observation_mask, dtype=bool)
    condition_kind = row["condition_kind"]
    if condition_kind in CONFIG_REPRODUCIBLE_CONDITIONS:
        stress_treatment = "config_exact"
    elif condition_kind in APPROXIMATED_CONDITIONS:
        observed, extra_mask = _approximate_condition_stress(observed, incident.sensor_nodes, row)
        observed_mask = observed_mask & extra_mask
        stress_treatment = "approximated"
    else:
        stress_treatment = "clean_only"

    return IncidentBundle(
        row=row,
        incident=incident,
        signature_set=signature_set,
        full_node_traces=full_traces,
        all_node_ids=all_node_ids,
        observed_sensor_matrix=observed,
        observed_mask=observed_mask,
        stress_treatment=stress_treatment,
        noise_floor_distance=_noise_floor_distance(row),
    )


__all__ = [
    "CONFIG_REPRODUCIBLE_CONDITIONS",
    "APPROXIMATED_CONDITIONS",
    "IncidentBundle",
    "build_incident_bundle",
]
