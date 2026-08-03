"""Authoritative WNTR hydraulic and water-quality simulation wrapper."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from hydroswarm.domain import ActionType, ConsequenceMetrics, OperationalPlan

try:
    import wntr
except ImportError as exc:  # pragma: no cover - optional dependency path
    wntr = None
    _WNTR_IMPORT_ERROR: ImportError | None = exc
else:
    _WNTR_IMPORT_ERROR = None


@dataclass(frozen=True)
class HydraulicState:
    """Hydraulic values at one simulation timestamp, expressed in SI units."""

    timestamp_seconds: int
    pressure_m: dict[str, float]
    demand_m3s: dict[str, float]
    flow_m3s: dict[str, float]
    velocity_mps: dict[str, float]


@dataclass(frozen=True)
class IncidentSimulation:
    """Time-aligned concentration output from EPANET via WNTR."""

    concentration_mg_l: pd.DataFrame
    source_node_id: str
    state_hash: str
    simulator: str
    simulator_version: str


@dataclass(frozen=True)
class HydraulicEvaluation:
    """Exact plan simulation plus threshold findings."""

    consequences: ConsequenceMetrics
    rejection_codes: tuple[str, ...]
    state_hash: str


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class HydraulicSimulator:
    """Load, validate and simulate water networks using WNTR as authority."""

    simulator_name = "WNTRSimulator"

    def __init__(
        self,
        network: Any | str | Path | None = None,
        *,
        minimum_pressure_m: float = 10.0,
        minimum_service_availability: float = 0.90,
        pressure_required_m: float = 10.0,
    ) -> None:
        if minimum_service_availability < 0.0 or minimum_service_availability > 1.0:
            raise ValueError("minimum_service_availability must be between zero and one")
        self.minimum_pressure_m = float(minimum_pressure_m)
        self.minimum_service_availability = float(minimum_service_availability)
        self.pressure_required_m = max(float(pressure_required_m), self.minimum_pressure_m)
        self._network: Any | None = None
        if network is not None:
            self.load(network)

    @property
    def simulator_version(self) -> str:
        return getattr(wntr, "__version__", "unavailable")

    @property
    def network(self) -> Any:
        if self._network is None:
            raise RuntimeError("no network is loaded")
        return self._network

    def load(self, network: Any | str | Path) -> HydraulicSimulator:
        """Load an INP file or take an isolated copy of a WNTR model."""

        if wntr is None:
            raise ImportError("WNTR is required for authoritative hydraulic simulation") from _WNTR_IMPORT_ERROR
        if isinstance(network, (str, Path)):
            loaded = wntr.network.WaterNetworkModel(str(network))
        elif hasattr(network, "node_name_list") and hasattr(network, "link_name_list"):
            loaded = copy.deepcopy(network)
        else:
            raise TypeError("network must be an EPANET INP path or WNTR WaterNetworkModel")
        self._network = loaded
        return self

    def validate(self) -> tuple[str, ...]:
        """Return deterministic structural validation codes without running a plan."""

        model = self.network
        issues: list[str] = []
        if not model.reservoir_name_list:
            issues.append("NO_RESERVOIR")
        if not model.junction_name_list:
            issues.append("NO_JUNCTIONS")
        graph = model.to_graph()
        if graph.number_of_nodes() and not nx.is_weakly_connected(nx.MultiDiGraph(graph)):
            issues.append("DISCONNECTED_NETWORK")
        for pipe_name in model.pipe_name_list:
            pipe = model.get_link(pipe_name)
            if pipe.length <= 0 or pipe.diameter <= 0 or pipe.roughness <= 0:
                issues.append(f"INVALID_PIPE:{pipe_name}")
        return tuple(sorted(set(issues)))

    def _prepared_network(self) -> Any:
        model = copy.deepcopy(self.network)
        model.options.hydraulic.demand_model = "PDD"
        model.options.hydraulic.minimum_pressure = 0.0
        model.options.hydraulic.required_pressure = self.pressure_required_m
        return model

    def _run_hydraulics(self, model: Any) -> Any:
        return wntr.sim.WNTRSimulator(model).run_sim()

    def calculate_state(self, at_time: int | None = None) -> HydraulicState:
        """Calculate a hydraulic snapshot; defaults to the last simulated time."""

        results = self._run_hydraulics(self._prepared_network())
        pressure = results.node["pressure"]
        timestamp = int(pressure.index[-1] if at_time is None else at_time)
        if timestamp not in pressure.index:
            timestamp = int(min(pressure.index, key=lambda value: abs(int(value) - timestamp)))
        return HydraulicState(
            timestamp_seconds=timestamp,
            pressure_m={key: float(value) for key, value in pressure.loc[timestamp].items()},
            demand_m3s={key: float(value) for key, value in results.node["demand"].loc[timestamp].items()},
            flow_m3s={key: float(value) for key, value in results.link["flowrate"].loc[timestamp].items()},
            velocity_mps={key: float(value) for key, value in results.link["velocity"].loc[timestamp].items()},
        )

    def build_dynamic_graph(self, state: HydraulicState | None = None) -> nx.MultiDiGraph:
        """Orient links by current flow and attach travel-time structural features."""

        state = state or self.calculate_state()
        model = self.network
        graph = nx.MultiDiGraph(timestamp_seconds=state.timestamp_seconds)
        reservoirs = set(model.reservoir_name_list)
        for node_name in model.node_name_list:
            node = model.get_node(node_name)
            graph.add_node(
                node_name,
                pressure_m=state.pressure_m.get(node_name, 0.0),
                demand_m3s=state.demand_m3s.get(node_name, 0.0),
                reservoir=node_name in reservoirs,
            )
        for link_name in model.link_name_list:
            link = model.get_link(link_name)
            flow = state.flow_m3s.get(link_name, 0.0)
            velocity = abs(state.velocity_mps.get(link_name, 0.0))
            start, end = link.start_node_name, link.end_node_name
            if flow < 0:
                start, end = end, start
            length = float(getattr(link, "length", 0.0))
            graph.add_edge(
                start,
                end,
                key=link_name,
                link_id=link_name,
                flow_m3s=abs(flow),
                velocity_mps=velocity,
                travel_time_seconds=length / velocity if velocity > 1e-9 else float("inf"),
                operable=self._is_link_operable(link),
            )
        reachable: set[str] = set()
        for reservoir in reservoirs:
            reachable.add(reservoir)
            reachable.update(nx.descendants(graph, reservoir))
        nx.set_node_attributes(graph, {name: name in reachable for name in graph}, "reservoir_reachable")
        demand_total = sum(max(0.0, state.demand_m3s.get(name, 0.0)) for name in graph)
        nx.set_node_attributes(
            graph,
            {
                name: max(0.0, state.demand_m3s.get(name, 0.0)) / demand_total if demand_total else 0.0
                for name in graph
            },
            "demand_centrality",
        )
        return graph

    @staticmethod
    def _is_link_operable(link: Any) -> bool:
        return not bool(getattr(link, "check_valve", False)) and bool(getattr(link, "operable", True))

    def simulate_incident(
        self,
        source_node_id: str,
        *,
        strength_mg_min: float = 10.0,
        start_minute: int = 0,
        duration_minutes: int = 60,
    ) -> IncidentSimulation:
        """Run EPANET chemical transport through WNTR for a bounded mass source."""

        if source_node_id not in self.network.node_name_list:
            raise ValueError(f"unknown incident source node: {source_node_id}")
        if strength_mg_min <= 0 or duration_minutes <= 0 or start_minute < 0:
            raise ValueError("incident strength/duration must be positive and start must be non-negative")
        model = copy.deepcopy(self.network)
        model.options.quality.parameter = "CHEMICAL"
        model.options.quality.inpfile_units = "mg/L"
        timestep = int(model.options.time.pattern_timestep)
        periods = max(1, int(model.options.time.duration // timestep) + 1)
        start = int(start_minute * 60 // timestep)
        stop = max(start + 1, int((start_minute + duration_minutes) * 60 / timestep + 0.999))
        multipliers = [1.0 if start <= index < stop else 0.0 for index in range(periods)]
        model.add_pattern("hydroswarm_incident", multipliers)
        model.add_source(
            "hydroswarm_incident",
            source_node_id,
            "MASS",
            float(strength_mg_min),
            "hydroswarm_incident",
        )
        with tempfile.TemporaryDirectory(prefix="hydroswarm-epanet-") as temp_directory:
            prefix = str(Path(temp_directory) / "incident")
            results = wntr.sim.EpanetSimulator(model).run_sim(file_prefix=prefix)
            quality = results.node["quality"].copy()
        state_hash = _digest(
            {
                "network": self._network_fingerprint(),
                "source": source_node_id,
                "strength": strength_mg_min,
                "start": start_minute,
                "duration": duration_minutes,
                "quality": quality.round(9).to_dict(),
            }
        )
        return IncidentSimulation(
            concentration_mg_l=quality,
            source_node_id=source_node_id,
            state_hash=state_hash,
            simulator="EpanetSimulator via WNTR",
            simulator_version=self.simulator_version,
        )

    def evaluate_plan(self, plan: OperationalPlan) -> HydraulicEvaluation:
        """Apply a prescreened plan to a copy and evaluate exact hydraulic thresholds."""

        model = self._prepared_network()
        self._apply_actions(model, plan)
        baseline = self._run_hydraulics(self._prepared_network())
        results = self._run_hydraulics(model)
        junctions = model.junction_name_list
        pressure = results.node["pressure"][junctions]
        delivered = results.node["demand"][junctions].clip(lower=0.0)
        requested = baseline.node["demand"][junctions].clip(lower=0.0)
        denominator = float(requested.to_numpy().sum())
        availability = min(1.0, float(delivered.to_numpy().sum()) / denominator) if denominator else 1.0

        index = list(pressure.index)
        if len(index) > 1:
            intervals = [max(0, int(right) - int(left)) for left, right in zip(index, index[1:])]
            intervals.append(intervals[-1])
        else:
            intervals = [int(model.options.time.hydraulic_timestep)]
        violating = pressure.lt(self.minimum_pressure_m).any(axis=1)
        violation_minutes = sum(seconds for seconds, flag in zip(intervals, violating, strict=True) if flag) / 60.0
        missing_m3s = (requested - delivered).clip(lower=0.0).sum(axis=1)
        unserved_l = sum(float(rate) * seconds * 1000.0 for rate, seconds in zip(missing_m3s, intervals, strict=True))
        minimum_pressure = float(pressure.min().min())

        codes: list[str] = []
        if minimum_pressure < self.minimum_pressure_m:
            codes.append("PRESSURE_BELOW_MINIMUM")
        if availability < self.minimum_service_availability:
            codes.append("SERVICE_BELOW_MINIMUM")
        metrics = ConsequenceMetrics(
            minimum_pressure_m=minimum_pressure,
            pressure_violation_minutes=violation_minutes,
            unserved_demand_l=max(0.0, unserved_l),
            service_availability=max(0.0, min(1.0, availability)),
            operation_count=sum(action.action_type != ActionType.END_PLAN for action in plan.actions),
        )
        return HydraulicEvaluation(
            consequences=metrics,
            rejection_codes=tuple(codes),
            state_hash=self.state_hash(plan),
        )

    def _apply_actions(self, model: Any, plan: OperationalPlan) -> None:
        for index, action in enumerate(plan.actions):
            if action.action_type == ActionType.FLUSH_NODE:
                timestep = int(model.options.time.pattern_timestep)
                periods = max(1, int(model.options.time.duration // timestep) + 1)
                start = int(action.start_minute * 60 // timestep)
                if action.duration_minutes:
                    stop = max(
                        start + 1,
                        int((action.start_minute + action.duration_minutes) * 60 / timestep + 0.999),
                    )
                else:
                    stop = periods
                pattern_name = f"hydroswarm_flush_{index}"
                model.add_pattern(
                    pattern_name,
                    [1.0 if start <= period < stop else 0.0 for period in range(periods)],
                )
                model.get_node(action.target_id).add_demand(
                    float(action.flow_rate_lps) / 1000.0,
                    pattern_name,
                    category="hydroswarm_flush",
                )
                continue
            if action.action_type not in {ActionType.CLOSE_PIPE, ActionType.OPEN_PIPE}:
                continue
            link = model.get_link(action.target_id)
            status = (
                wntr.network.LinkStatus.Closed
                if action.action_type == ActionType.CLOSE_PIPE
                else wntr.network.LinkStatus.Open
            )
            start_seconds = action.start_minute * 60
            if start_seconds == 0:
                link.initial_status = status
            else:
                condition = wntr.network.controls.SimTimeCondition(model, "=", start_seconds)
                control_action = wntr.network.controls.ControlAction(link, "status", status)
                model.add_control(f"hydroswarm_{index}_start", wntr.network.controls.Control(condition, control_action))
            if action.duration_minutes > 0:
                inverse = (
                    wntr.network.LinkStatus.Open
                    if status == wntr.network.LinkStatus.Closed
                    else wntr.network.LinkStatus.Closed
                )
                condition = wntr.network.controls.SimTimeCondition(
                    model, "=", (action.start_minute + action.duration_minutes) * 60
                )
                control_action = wntr.network.controls.ControlAction(link, "status", inverse)
                model.add_control(f"hydroswarm_{index}_stop", wntr.network.controls.Control(condition, control_action))

    def _network_fingerprint(self) -> dict[str, object]:
        model = self.network
        return {
            "nodes": sorted(model.node_name_list),
            "links": [
                (
                    name,
                    model.get_link(name).start_node_name,
                    model.get_link(name).end_node_name,
                    str(model.get_link(name).initial_status),
                )
                for name in sorted(model.link_name_list)
            ],
            "duration": int(model.options.time.duration),
        }

    def state_hash(self, plan: OperationalPlan | None = None) -> str:
        payload: dict[str, object] = {
            "network": self._network_fingerprint(),
            "thresholds": {
                "pressure": self.minimum_pressure_m,
                "service": self.minimum_service_availability,
            },
        }
        if plan is not None:
            payload["plan"] = plan.model_dump(mode="json", exclude={"created_at"})
        return _digest(payload)
