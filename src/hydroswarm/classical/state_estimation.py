"""Hydraulic telemetry reconciliation and uncertainty-aware state estimation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Mapping

from hydroswarm.simulation.wrapper import HydraulicState


@dataclass(frozen=True, slots=True)
class ValueRange:
    estimate: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.estimate, self.lower, self.upper)):
            raise ValueError("estimated value range must be finite")
        if self.lower > self.estimate or self.estimate > self.upper:
            raise ValueError("estimate must lie inside its uncertainty range")


@dataclass(frozen=True, slots=True)
class OperationalTelemetry:
    pressure_m: Mapping[str, float | None] = field(default_factory=dict)
    demand_m3s: Mapping[str, float | None] = field(default_factory=dict)
    flow_m3s: Mapping[str, float | None] = field(default_factory=dict)
    tank_level_m: Mapping[str, float | None] = field(default_factory=dict)
    pump_open: Mapping[str, bool | None] = field(default_factory=dict)
    valve_open: Mapping[str, bool | None] = field(default_factory=dict)
    operator_overrides: Mapping[str, float | bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StateResidualReport:
    pressure_rmse_m: float
    demand_rmse_m3s: float
    flow_rmse_m3s: float
    missing_values_imputed: int
    reconciled_pumps: tuple[str, ...]
    reconciled_valves: tuple[str, ...]
    mismatch_scores: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class EstimatedHydraulicState:
    timestamp_seconds: int
    pressure_m: Mapping[str, ValueRange]
    demand_m3s: Mapping[str, ValueRange]
    flow_m3s: Mapping[str, ValueRange]
    velocity_mps: Mapping[str, float]
    tank_level_m: Mapping[str, ValueRange]
    pump_open: Mapping[str, bool]
    valve_open: Mapping[str, bool]
    zone_demand_multipliers: Mapping[str, float]
    residuals: StateResidualReport

    @property
    def normalized_uncertainty(self) -> float:
        ranges = list(self.pressure_m.values()) + list(self.demand_m3s.values())
        if not ranges:
            return 1.0
        relative_widths = [
            (item.upper - item.lower) / max(abs(item.estimate), 1e-6) for item in ranges
        ]
        return min(1.0, sum(relative_widths) / (2.0 * len(relative_widths)))

    def as_hydraulic_state(self) -> HydraulicState:
        return HydraulicState(
            timestamp_seconds=self.timestamp_seconds,
            pressure_m={name: item.estimate for name, item in self.pressure_m.items()},
            demand_m3s={name: item.estimate for name, item in self.demand_m3s.items()},
            flow_m3s={name: item.estimate for name, item in self.flow_m3s.items()},
            velocity_mps=dict(self.velocity_mps),
        )


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _rmse(model: Mapping[str, float], observed: Mapping[str, float | None]) -> float:
    residuals = [
        float(observed[name]) - value
        for name, value in model.items()
        if name in observed and _finite(observed[name])
    ]
    return math.sqrt(sum(value * value for value in residuals) / len(residuals)) if residuals else 0.0


def _range(estimate: float, absolute_error: float, relative_error: float) -> ValueRange:
    radius = max(absolute_error, abs(estimate) * relative_error)
    return ValueRange(estimate=estimate, lower=estimate - radius, upper=estimate + radius)


class HydraulicStateEstimator:
    """Reconcile a simulated state with incomplete operator telemetry."""

    def __init__(
        self,
        *,
        node_to_zone: Mapping[str, str] | None = None,
        pressure_sensor_error_m: float = 0.5,
        relative_hydraulic_error: float = 0.1,
    ) -> None:
        self.node_to_zone = dict(node_to_zone or {})
        self.pressure_sensor_error_m = pressure_sensor_error_m
        self.relative_hydraulic_error = relative_hydraulic_error

    def estimate(
        self,
        simulated: HydraulicState,
        telemetry: OperationalTelemetry,
        *,
        model_tank_levels_m: Mapping[str, float] | None = None,
        model_pump_open: Mapping[str, bool] | None = None,
        model_valve_open: Mapping[str, bool] | None = None,
    ) -> EstimatedHydraulicState:
        zone_ratios: dict[str, list[float]] = {}
        for node, simulated_demand in simulated.demand_m3s.items():
            observed = telemetry.demand_m3s.get(node)
            if _finite(observed) and abs(simulated_demand) > 1e-9:
                zone = self.node_to_zone.get(node, "default")
                zone_ratios.setdefault(zone, []).append(float(observed) / simulated_demand)
        multipliers = {
            zone: max(0.25, min(4.0, median(values))) for zone, values in zone_ratios.items()
        }
        for zone in set(self.node_to_zone.values()) | {"default"}:
            multipliers.setdefault(zone, 1.0)

        missing = 0
        pressure: dict[str, ValueRange] = {}
        for node, model_value in simulated.pressure_m.items():
            value = telemetry.pressure_m.get(node)
            if _finite(value):
                estimate, relative = float(value), self.relative_hydraulic_error * 0.5
            else:
                estimate, relative = model_value, self.relative_hydraulic_error
                missing += int(node in telemetry.pressure_m)
            pressure[node] = _range(estimate, self.pressure_sensor_error_m, relative)

        demand: dict[str, ValueRange] = {}
        for node, model_value in simulated.demand_m3s.items():
            value = telemetry.demand_m3s.get(node)
            if _finite(value):
                estimate, relative = float(value), self.relative_hydraulic_error * 0.5
            else:
                zone = self.node_to_zone.get(node, "default")
                estimate = model_value * multipliers[zone]
                relative = self.relative_hydraulic_error * 1.5
                missing += int(node in telemetry.demand_m3s)
            demand[node] = _range(estimate, 1e-7, relative)

        flow: dict[str, ValueRange] = {}
        for link, model_value in simulated.flow_m3s.items():
            value = telemetry.flow_m3s.get(link)
            if _finite(value):
                estimate, relative = float(value), self.relative_hydraulic_error * 0.5
            else:
                estimate, relative = model_value, self.relative_hydraulic_error * 1.5
                missing += int(link in telemetry.flow_m3s)
            flow[link] = _range(estimate, 1e-7, relative)

        tank_levels: dict[str, ValueRange] = {}
        for tank, model_level in (model_tank_levels_m or {}).items():
            observed = telemetry.tank_level_m.get(tank)
            estimate = float(observed) if _finite(observed) else model_level
            missing += int(tank in telemetry.tank_level_m and not _finite(observed))
            tank_levels[tank] = _range(estimate, 0.25, self.relative_hydraulic_error)

        pumps = dict(model_pump_open or {})
        valves = dict(model_valve_open or {})
        reconciled_pumps: list[str] = []
        reconciled_valves: list[str] = []
        for name, state in telemetry.pump_open.items():
            if state is not None and pumps.get(name) != state:
                pumps[name] = state
                reconciled_pumps.append(name)
        for name, state in telemetry.valve_open.items():
            if state is not None and valves.get(name) != state:
                valves[name] = state
                reconciled_valves.append(name)

        pressure_rmse = _rmse(simulated.pressure_m, telemetry.pressure_m)
        demand_rmse = _rmse(simulated.demand_m3s, telemetry.demand_m3s)
        flow_rmse = _rmse(simulated.flow_m3s, telemetry.flow_m3s)
        mean_pressure = max(1.0, sum(map(abs, simulated.pressure_m.values())) / max(1, len(simulated.pressure_m)))
        mean_demand = max(1e-8, sum(map(abs, simulated.demand_m3s.values())) / max(1, len(simulated.demand_m3s)))
        mismatch = {
            "pipe_roughness": min(1.0, pressure_rmse / mean_pressure),
            "demand": min(1.0, demand_rmse / mean_demand),
            "valve_telemetry": len(reconciled_valves) / max(1, len(valves)),
            "tank_levels": min(1.0, sum(
                abs(item.estimate - (model_tank_levels_m or {}).get(name, item.estimate))
                for name, item in tank_levels.items()
            ) / max(1.0, len(tank_levels) * 5.0)),
            "pump_schedules": len(reconciled_pumps) / max(1, len(pumps)),
        }
        report = StateResidualReport(
            pressure_rmse_m=pressure_rmse,
            demand_rmse_m3s=demand_rmse,
            flow_rmse_m3s=flow_rmse,
            missing_values_imputed=missing,
            reconciled_pumps=tuple(sorted(reconciled_pumps)),
            reconciled_valves=tuple(sorted(reconciled_valves)),
            mismatch_scores=mismatch,
        )
        return EstimatedHydraulicState(
            timestamp_seconds=simulated.timestamp_seconds,
            pressure_m=pressure,
            demand_m3s=demand,
            flow_m3s=flow,
            velocity_mps=dict(simulated.velocity_mps),
            tank_level_m=tank_levels,
            pump_open=pumps,
            valve_open=valves,
            zone_demand_multipliers=multipliers,
            residuals=report,
        )

    def apply_operator_state(self, network: Any, telemetry: OperationalTelemetry) -> Any:
        model = copy.deepcopy(network)
        for tank, level in telemetry.tank_level_m.items():
            if _finite(level) and tank in model.tank_name_list:
                model.get_node(tank).init_level = float(level)
        for names, states in (
            (set(model.pump_name_list), telemetry.pump_open),
            (set(model.valve_name_list), telemetry.valve_open),
        ):
            for name, is_open in states.items():
                if name in names and is_open is not None:
                    status_type = type(model.get_link(name).initial_status)
                    model.get_link(name).initial_status = status_type.Open if is_open else status_type.Closed
        return model

