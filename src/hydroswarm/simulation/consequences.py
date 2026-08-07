"""Transparent consequence calculations and multi-objective plan comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from hydroswarm.domain import ConsequenceMetrics


def _interval_seconds(index: Sequence[int | float]) -> np.ndarray:
    values = np.asarray(index, dtype=np.float64)
    if values.size == 0:
        raise ValueError("simulation timeline cannot be empty")
    if values.size == 1:
        return np.asarray([0.0])
    differences = np.diff(values)
    if np.any(differences <= 0):
        raise ValueError("simulation timestamps must increase")
    return np.concatenate((differences, differences[-1:]))


def calculate_exposure_consequences(
    concentration_mg_l: pd.DataFrame,
    demand_m3s: pd.DataFrame,
    *,
    threshold_mg_l: float,
    population_by_node: Mapping[str, float] | None = None,
    pipe_endpoints: Mapping[str, tuple[str, str, float]] | None = None,
    pressure_m: pd.DataFrame | None = None,
    minimum_pressure_m: float = 10.0,
    requested_demand_m3s: pd.DataFrame | None = None,
    operation_count: int = 0,
) -> ConsequenceMetrics:
    """Integrate exposure, service, and pressure directly over simulator frames."""
    if threshold_mg_l < 0:
        raise ValueError("concentration threshold cannot be negative")
    nodes = [name for name in concentration_mg_l.columns if name in demand_m3s.columns]
    if not nodes:
        raise ValueError("concentration and demand frames have no common nodes")
    concentration = concentration_mg_l[nodes].reindex(demand_m3s.index, method="nearest")
    demand = demand_m3s[nodes].clip(lower=0.0)
    seconds = _interval_seconds(list(demand.index))[:, None]
    volume_l = demand.to_numpy(dtype=float) * seconds * 1000.0
    concentration_values = np.maximum(0.0, concentration.to_numpy(dtype=float))
    above = concentration_values >= threshold_mg_l
    mass_consumed_mg = float(np.sum(concentration_values * volume_l))
    volume_above_l = float(np.sum(volume_l * above))

    exposed_nodes = {
        node for index, node in enumerate(nodes) if bool(np.any(above[:, index]))
    }
    if population_by_node:
        population = sum(max(0.0, population_by_node.get(node, 0.0)) for node in exposed_nodes)
    else:
        total_volume = volume_l.sum(axis=0)
        population = float(sum(total_volume[index] > 0 for index, node in enumerate(nodes) if node in exposed_nodes))

    extent = 0.0
    for _pipe, (start, end, length_m) in (pipe_endpoints or {}).items():
        if start in concentration.columns and end in concentration.columns:
            contaminated = (
                concentration[start].ge(threshold_mg_l) | concentration[end].ge(threshold_mg_l)
            ).any()
            extent += max(0.0, length_m) if contaminated else 0.0

    minimum_pressure = float(pressure_m.min().min()) if pressure_m is not None else 0.0
    pressure_violation_minutes = 0.0
    if pressure_m is not None:
        pressure_seconds = _interval_seconds(list(pressure_m.index))
        flags = pressure_m.lt(minimum_pressure_m).any(axis=1).to_numpy()
        pressure_violation_minutes = float(pressure_seconds[flags].sum() / 60.0)

    requested = requested_demand_m3s[nodes].clip(lower=0.0) if requested_demand_m3s is not None else demand
    requested = requested.reindex(demand.index, method="nearest")
    requested_volume = requested.to_numpy(dtype=float) * seconds * 1000.0
    delivered_volume = volume_l
    unserved_l = float(np.maximum(0.0, requested_volume - delivered_volume).sum())
    total_requested = float(requested_volume.sum())
    availability = 1.0 - unserved_l / total_requested if total_requested else 1.0

    contaminated_by_time = above.any(axis=1)
    containment_time: float | None = None
    if contaminated_by_time.any():
        last_position = int(np.flatnonzero(contaminated_by_time)[-1])
        if not contaminated_by_time[last_position:].any() and last_position + 1 < len(demand.index):
            containment_time = float(demand.index[last_position + 1] - demand.index[0]) / 60.0
        elif last_position + 1 < len(demand.index):
            containment_time = float(demand.index[last_position + 1] - demand.index[0]) / 60.0

    return ConsequenceMetrics(
        population_impacted=float(population),
        contaminant_mass_consumed_mg=max(0.0, mass_consumed_mg),
        volume_above_threshold_l=max(0.0, volume_above_l),
        contaminated_pipe_extent_m=max(0.0, extent),
        minimum_pressure_m=minimum_pressure,
        pressure_violation_minutes=max(0.0, pressure_violation_minutes),
        unserved_demand_l=max(0.0, unserved_l),
        service_availability=max(0.0, min(1.0, availability)),
        operation_count=operation_count,
        containment_time_minutes=containment_time,
        exposure_evaluated=True,
    )


@dataclass(frozen=True, slots=True)
class PlanOutcome:
    plan_id: str
    metrics: ConsequenceMetrics
    valid: bool = True


@dataclass(frozen=True, slots=True)
class RankedPlanOutcome:
    plan_id: str
    metrics: ConsequenceMetrics
    exposure_reduction: float
    service_change: float
    dominated: bool
    regret: float
    recommended: bool


def _objectives(metrics: ConsequenceMetrics) -> tuple[float, ...]:
    return (
        metrics.contaminant_mass_consumed_mg,
        metrics.volume_above_threshold_l,
        metrics.population_impacted,
        metrics.pressure_violation_minutes,
        metrics.unserved_demand_l,
        float(metrics.operation_count),
        -metrics.service_availability,
    )


def _dominates(left: ConsequenceMetrics, right: ConsequenceMetrics) -> bool:
    a, b = _objectives(left), _objectives(right)
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(
        x < y for x, y in zip(a, b, strict=True)
    )


def rank_plan_outcomes(
    no_response: ConsequenceMetrics, outcomes: Sequence[PlanOutcome]
) -> tuple[RankedPlanOutcome, ...]:
    valid = [item for item in outcomes if item.valid]
    if not valid:
        return ()
    scale = max(no_response.contaminant_mass_consumed_mg, 1e-9)
    scores = {
        item.plan_id: (
            item.metrics.contaminant_mass_consumed_mg / scale
            + item.metrics.pressure_violation_minutes / 60.0
            + item.metrics.unserved_demand_l / max(no_response.unserved_demand_l, 1.0)
            + 0.02 * item.metrics.operation_count
        )
        for item in valid
    }
    best_score = min(scores.values())
    nondominated = [
        item for item in valid if not any(
            other.plan_id != item.plan_id and _dominates(other.metrics, item.metrics)
            for other in valid
        )
    ]
    recommended_id = min(nondominated, key=lambda item: scores[item.plan_id]).plan_id
    ranked = [
        RankedPlanOutcome(
            plan_id=item.plan_id,
            metrics=item.metrics,
            exposure_reduction=(
                no_response.contaminant_mass_consumed_mg - item.metrics.contaminant_mass_consumed_mg
            ),
            service_change=item.metrics.service_availability - no_response.service_availability,
            dominated=item not in nondominated,
            regret=max(0.0, scores[item.plan_id] - best_score),
            recommended=item.plan_id == recommended_id,
        )
        for item in valid
    ]
    return tuple(sorted(ranked, key=lambda item: (not item.recommended, item.dominated, item.regret)))

