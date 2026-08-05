"""Authoritative WNTR hydraulic and water-quality simulation wrapper."""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import queue
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from hydroswarm.domain import ActionType, ConsequenceMetrics, OperationalPlan
from hydroswarm.storage.cache import SimulationResultCache

from .consequences import calculate_exposure_consequences

try:
    import wntr
except ImportError as exc:  # pragma: no cover - optional dependency path
    wntr = None
    _WNTR_IMPORT_ERROR: ImportError | None = exc
else:
    _WNTR_IMPORT_ERROR = None


class SimulationError(RuntimeError):
    """Base class for fail-closed simulator runtime errors."""


class SimulationTimeoutError(SimulationError):
    pass


class SimulationIncompleteError(SimulationError):
    pass


class SimulationUnstableError(SimulationError):
    pass


class SimulationBudgetExceeded(SimulationError):
    pass


@dataclass(frozen=True, slots=True)
class IncidentSource:
    node_id: str
    strength_mg_min: float = 10.0
    start_minute: int = 0
    duration_minutes: int = 60

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("source node_id is required")
        if self.strength_mg_min <= 0 or self.duration_minutes <= 0 or self.start_minute < 0:
            raise ValueError("source strength/duration must be positive and start non-negative")


@dataclass(frozen=True, slots=True)
class IncidentSourceProfile:
    """Structured source hypothesis, optionally containing stress-test sources."""

    source_node_id: str
    strength_mg_min: float = 10.0
    start_minute: int = 0
    duration_minutes: int = 60
    additional_sources: tuple[IncidentSource, ...] = ()
    stress_mode: bool = False
    label: str = "incident-hypothesis"

    def __post_init__(self) -> None:
        IncidentSource(
            self.source_node_id,
            self.strength_mg_min,
            self.start_minute,
            self.duration_minutes,
        )
        if self.additional_sources and not self.stress_mode:
            raise ValueError("multiple sources require stress_mode=True")
        if len(self.additional_sources) > 15:
            raise ValueError("stress mode supports at most 16 sources")

    @property
    def sources(self) -> tuple[IncidentSource, ...]:
        return (
            IncidentSource(
                self.source_node_id,
                self.strength_mg_min,
                self.start_minute,
                self.duration_minutes,
            ),
            *self.additional_sources,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "strength_mg_min": self.strength_mg_min,
            "start_minute": self.start_minute,
            "duration_minutes": self.duration_minutes,
            "additional_sources": [
                {
                    "node_id": source.node_id,
                    "strength_mg_min": source.strength_mg_min,
                    "start_minute": source.start_minute,
                    "duration_minutes": source.duration_minutes,
                }
                for source in self.additional_sources
            ],
            "stress_mode": self.stress_mode,
            "label": self.label,
        }


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
    source_node_ids: tuple[str, ...] = ()
    profile: IncidentSourceProfile | None = None
    demand_m3s: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    pressure_m: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    water_age_hours: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    tracer_percent: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    pump_energy_kwh: float | None = None
    pump_cost: float | None = None
    complete: bool = True
    unstable: bool = False
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class SimulatedSample:
    node_id: str
    timestamp_seconds: int
    concentration_mg_l: float
    water_age_hours: float | None
    tracer_percent: float | None
    state_hash: str


@dataclass(frozen=True)
class HydraulicEvaluation:
    """Exact plan simulation plus threshold findings."""

    consequences: ConsequenceMetrics
    rejection_codes: tuple[str, ...]
    state_hash: str
    pump_energy_kwh: float | None = None
    pump_cost: float | None = None
    cache_hit: bool = False


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
        timeout_seconds: float = 60.0,
        cache: SimulationResultCache | str | Path | None = None,
        exact_simulation_budget: int | None = None,
        budget_hook: Callable[[str, int, int | None], None] | None = None,
        energy_price_per_kwh: float = 0.12,
    ) -> None:
        if minimum_service_availability < 0.0 or minimum_service_availability > 1.0:
            raise ValueError("minimum_service_availability must be between zero and one")
        self.minimum_pressure_m = float(minimum_pressure_m)
        self.minimum_service_availability = float(minimum_service_availability)
        self.pressure_required_m = max(float(pressure_required_m), self.minimum_pressure_m)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if exact_simulation_budget is not None and exact_simulation_budget < 0:
            raise ValueError("exact_simulation_budget must be non-negative")
        if energy_price_per_kwh < 0:
            raise ValueError("energy_price_per_kwh must be non-negative")
        self.timeout_seconds = float(timeout_seconds)
        self.cache = (
            cache
            if isinstance(cache, SimulationResultCache) or cache is None
            else SimulationResultCache(cache)
        )
        self.exact_simulation_budget = exact_simulation_budget
        self.budget_hook = budget_hook
        self.energy_price_per_kwh = float(energy_price_per_kwh)
        self.exact_runs = 0
        self._network: Any | None = None
        if network is not None:
            self.load(network)

    @property
    def remaining_exact_runs(self) -> int | None:
        if self.exact_simulation_budget is None:
            return None
        return max(0, self.exact_simulation_budget - self.exact_runs)

    def _consume_budget(self, operation: str) -> None:
        if self.exact_simulation_budget is not None and self.exact_runs >= self.exact_simulation_budget:
            raise SimulationBudgetExceeded(f"exact simulation budget exhausted before {operation}")
        self.exact_runs += 1
        if self.budget_hook is not None:
            self.budget_hook(operation, self.exact_runs, self.remaining_exact_runs)

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

    def _run_with_timeout(self, operation: str, function: Callable[[], Any]) -> Any:
        """Run `function` with a hard wall-clock deadline in a real, killable
        OS subprocess -- not a daemon thread (core-issues.txt: a thread that
        exceeds its timeout cannot be forcibly stopped in Python and simply
        keeps running in the background, consuming CPU for however long the
        underlying EPANET/WNTR call takes, or forever if it is genuinely
        hung).

        Uses the "fork" start method deliberately, not "spawn": this call
        sits on the live incident-analysis latency path (real profiling
        shows single-digit milliseconds today), and spawn's from-scratch
        interpreter/import cost (numpy/pandas/wntr/torch) turns that into
        multiple seconds per call -- an unacceptable regression for a
        function invoked on every hydraulic simulation. Forking a
        multi-threaded parent (e.g. a live server) carries a narrow,
        well-documented risk of a child inheriting a lock another thread
        held at fork time; unlike the daemon-thread version this replaces,
        that failure mode is still fully contained -- a wedged child is a
        real OS process, so the timeout below still forcibly terminates it
        and this call still raises SimulationTimeoutError rather than
        hanging or leaking the process."""

        context = multiprocessing.get_context("fork")
        result_queue: multiprocessing.Queue = context.Queue(maxsize=1)

        def _worker() -> None:
            try:
                result_queue.put((True, function()))
            except BaseException as exc:  # propagate simulator failures to caller
                result_queue.put((False, exc))

        process = context.Process(target=_worker, name=f"hydroswarm-{operation}", daemon=True)
        process.start()
        process.join(self.timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join()
            raise SimulationTimeoutError(
                f"{operation} exceeded the {self.timeout_seconds:g}-second timeout"
            )
        try:
            succeeded, value = result_queue.get_nowait()
        except queue.Empty:
            raise SimulationError(
                f"{operation} worker exited without a result (exit code {process.exitcode})"
            ) from None
        finally:
            process.join()
        if not succeeded:
            raise value
        return value

    def _validate_results(self, results: Any, *, require_quality: bool = False) -> None:
        error_code = getattr(results, "error_code", None)
        if error_code not in (None, 0):
            raise SimulationUnstableError(f"simulator returned error code {error_code}")
        required = ("pressure", "demand")
        for name in required:
            frame = results.node.get(name) if hasattr(results, "node") else None
            if frame is None or frame.empty:
                raise SimulationIncompleteError(f"simulation omitted node {name}")
            values = frame.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise SimulationUnstableError(f"simulation produced non-finite node {name}")
        if require_quality:
            quality = results.node.get("quality") if hasattr(results, "node") else None
            if quality is None or quality.empty:
                raise SimulationIncompleteError("simulation omitted water-quality results")
            if set(self.network.node_name_list) - set(quality.columns):
                raise SimulationIncompleteError("water-quality results omit network nodes")
            if not np.isfinite(quality.to_numpy(dtype=float)).all():
                raise SimulationUnstableError("simulation produced non-finite water quality")
        pressure = results.node["pressure"]
        expected_end = int(self.network.options.time.duration)
        if int(pressure.index[-1]) < expected_end:
            raise SimulationIncompleteError("simulation ended before the configured duration")
        if float(pressure.min().min()) < -1_000.0:
            raise SimulationUnstableError("simulation pressure is numerically unstable")

    def _run_hydraulics(self, model: Any) -> Any:
        results = self._run_with_timeout(
            "hydraulics", lambda: wntr.sim.WNTRSimulator(model).run_sim()
        )
        self._validate_results(results)
        return results

    def _run_epanet(self, model: Any, operation: str) -> Any:
        def execute() -> Any:
            with tempfile.TemporaryDirectory(prefix="hydroswarm-epanet-") as directory:
                prefix = str(Path(directory) / operation)
                return wntr.sim.EpanetSimulator(model).run_sim(file_prefix=prefix)

        results = self._run_with_timeout(operation, execute)
        self._validate_results(results, require_quality=True)
        return results

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
        profile = IncidentSourceProfile(
            source_node_id=source_node_id,
            strength_mg_min=strength_mg_min,
            start_minute=start_minute,
            duration_minutes=duration_minutes,
        )
        return self.simulate_hypothesis(profile, include_diagnostics=False)

    def simulate_hypothesis(
        self,
        profile: IncidentSourceProfile,
        *,
        include_diagnostics: bool = True,
    ) -> IncidentSimulation:
        """Run one structured source hypothesis or a bounded multi-source stress case."""

        unknown = sorted({source.node_id for source in profile.sources} - set(self.network.node_name_list))
        if unknown:
            raise ValueError(f"unknown incident source nodes: {', '.join(unknown)}")
        cache_key = self._cache_key(
            profile=profile.as_dict(),
            plan=None,
            operation=f"hypothesis-diagnostics-{include_diagnostics}",
        )
        cached = self.cache.get(cache_key) if self.cache else None
        if cached is not None:
            try:
                return self._incident_from_payload(cached, profile, cache_hit=True)
            except (KeyError, TypeError, ValueError, SimulationIncompleteError):
                self.cache.invalidate(cache_key)
        self._consume_budget("hypothesis")
        model = copy.deepcopy(self.network)
        model.options.quality.parameter = "CHEMICAL"
        model.options.quality.inpfile_units = "mg/L"
        timestep = int(model.options.time.pattern_timestep)
        periods = max(1, int(model.options.time.duration // timestep) + 1)
        for index, source in enumerate(profile.sources):
            start = int(source.start_minute * 60 // timestep)
            stop = max(
                start + 1,
                int((source.start_minute + source.duration_minutes) * 60 / timestep + 0.999),
            )
            pattern_name = f"hydroswarm_incident_pattern_{index}"
            source_name = f"hydroswarm_incident_source_{index}"
            model.add_pattern(
                pattern_name,
                [1.0 if start <= period < stop else 0.0 for period in range(periods)],
            )
            model.add_source(
                source_name,
                source.node_id,
                "MASS",
                float(source.strength_mg_min),
                pattern_name,
            )
        results = self._run_epanet(model, "incident")
        quality = results.node["quality"].copy()
        demand = results.node["demand"].copy()
        pressure = results.node["pressure"].copy()
        water_age = self._quality_diagnostic("AGE") if include_diagnostics else None
        tracer = (
            self._quality_diagnostic("TRACE", trace_node=profile.source_node_id)
            if include_diagnostics
            else None
        )
        energy, cost = self._pump_energy(results, model)
        state_hash = _digest(
            {
                "network": self._network_fingerprint(),
                "profile": profile.as_dict(),
                "quality": quality.round(9).to_dict(),
            }
        )
        simulation = IncidentSimulation(
            concentration_mg_l=quality,
            source_node_id=profile.source_node_id,
            state_hash=state_hash,
            simulator="EpanetSimulator via WNTR",
            simulator_version=self.simulator_version,
            source_node_ids=tuple(source.node_id for source in profile.sources),
            profile=profile,
            demand_m3s=demand,
            pressure_m=pressure,
            water_age_hours=water_age,
            tracer_percent=tracer,
            pump_energy_kwh=energy,
            pump_cost=cost,
        )
        if self.cache:
            self.cache.put(cache_key, self._incident_payload(simulation))
        return simulation

    def simulate_stress(self, sources: tuple[IncidentSource, ...]) -> IncidentSimulation:
        """Explicit convenience API for multiple-source stress testing."""

        if not sources:
            raise ValueError("stress simulation requires at least one source")
        first, *additional = sources
        return self.simulate_hypothesis(
            IncidentSourceProfile(
                source_node_id=first.node_id,
                strength_mg_min=first.strength_mg_min,
                start_minute=first.start_minute,
                duration_minutes=first.duration_minutes,
                additional_sources=tuple(additional),
                stress_mode=True,
                label="multiple-source-stress",
            )
        )

    def _quality_diagnostic(self, mode: str, *, trace_node: str | None = None) -> pd.DataFrame:
        model = copy.deepcopy(self.network)
        model.options.quality.parameter = mode
        if trace_node is not None:
            model.options.quality.trace_node = trace_node
        results = self._run_epanet(model, f"quality-{mode.lower()}")
        return results.node["quality"].copy()

    def simulate_water_age(self) -> pd.DataFrame:
        self._consume_budget("water-age")
        return self._quality_diagnostic("AGE")

    def simulate_tracer(self, source_node_id: str) -> pd.DataFrame:
        if source_node_id not in self.network.node_name_list:
            raise ValueError(f"unknown tracer node: {source_node_id}")
        self._consume_budget("tracer")
        return self._quality_diagnostic("TRACE", trace_node=source_node_id)

    def simulate_sample(
        self,
        node_id: str,
        *,
        at_minute: int,
        profile: IncidentSourceProfile | None = None,
        simulation: IncidentSimulation | None = None,
    ) -> SimulatedSample:
        if node_id not in self.network.node_name_list:
            raise ValueError(f"unknown sample node: {node_id}")
        if at_minute < 0:
            raise ValueError("sample minute must be non-negative")
        if simulation is None:
            if profile is None:
                raise ValueError("profile or simulation is required")
            simulation = self.simulate_hypothesis(profile)
        timestamp = at_minute * 60
        index = simulation.concentration_mg_l.index
        nearest = int(min(index, key=lambda value: abs(int(value) - timestamp)))
        age = (
            float(simulation.water_age_hours.loc[nearest, node_id])
            if simulation.water_age_hours is not None and nearest in simulation.water_age_hours.index
            else None
        )
        tracer = (
            float(simulation.tracer_percent.loc[nearest, node_id])
            if simulation.tracer_percent is not None and nearest in simulation.tracer_percent.index
            else None
        )
        return SimulatedSample(
            node_id=node_id,
            timestamp_seconds=nearest,
            concentration_mg_l=float(simulation.concentration_mg_l.loc[nearest, node_id]),
            water_age_hours=age,
            tracer_percent=tracer,
            state_hash=simulation.state_hash,
        )

    def calculate_consequences(
        self,
        simulation: IncidentSimulation,
        *,
        threshold_mg_l: float,
        population_by_node: Mapping[str, float] | None = None,
        operation_count: int = 0,
    ) -> ConsequenceMetrics:
        if not simulation.complete or simulation.unstable:
            raise SimulationIncompleteError("cannot calculate consequences from incomplete results")
        if simulation.demand_m3s is None or simulation.pressure_m is None:
            raise SimulationIncompleteError("simulation lacks hydraulic consequence frames")
        pipe_endpoints = {
            name: (
                self.network.get_link(name).start_node_name,
                self.network.get_link(name).end_node_name,
                float(getattr(self.network.get_link(name), "length", 0.0)),
            )
            for name in self.network.pipe_name_list
        }
        return calculate_exposure_consequences(
            simulation.concentration_mg_l,
            simulation.demand_m3s,
            threshold_mg_l=threshold_mg_l,
            population_by_node=population_by_node,
            pipe_endpoints=pipe_endpoints,
            pressure_m=simulation.pressure_m,
            minimum_pressure_m=self.minimum_pressure_m,
            operation_count=operation_count,
        )

    @staticmethod
    def _frame_payload(frame: pd.DataFrame | None) -> dict[str, Any] | None:
        if frame is None:
            return None
        return {
            "index": [int(value) for value in frame.index],
            "columns": [str(value) for value in frame.columns],
            "data": frame.to_numpy(dtype=float).tolist(),
        }

    @staticmethod
    def _frame_from_payload(payload: dict[str, Any] | None) -> pd.DataFrame | None:
        if payload is None:
            return None
        return pd.DataFrame(payload["data"], index=payload["index"], columns=payload["columns"])

    def _incident_payload(self, simulation: IncidentSimulation) -> dict[str, Any]:
        return {
            "concentration": self._frame_payload(simulation.concentration_mg_l),
            "demand": self._frame_payload(simulation.demand_m3s),
            "pressure": self._frame_payload(simulation.pressure_m),
            "water_age": self._frame_payload(simulation.water_age_hours),
            "tracer": self._frame_payload(simulation.tracer_percent),
            "source_node_id": simulation.source_node_id,
            "source_node_ids": list(simulation.source_node_ids),
            "state_hash": simulation.state_hash,
            "simulator": simulation.simulator,
            "simulator_version": simulation.simulator_version,
            "pump_energy_kwh": simulation.pump_energy_kwh,
            "pump_cost": simulation.pump_cost,
            "complete": simulation.complete,
            "unstable": simulation.unstable,
        }

    def _incident_from_payload(
        self,
        payload: dict[str, Any],
        profile: IncidentSourceProfile,
        *,
        cache_hit: bool,
    ) -> IncidentSimulation:
        concentration = self._frame_from_payload(payload.get("concentration"))
        if concentration is None or concentration.empty:
            raise SimulationIncompleteError("cached simulation has no concentration data")
        return IncidentSimulation(
            concentration_mg_l=concentration,
            source_node_id=payload["source_node_id"],
            state_hash=payload["state_hash"],
            simulator=payload["simulator"],
            simulator_version=payload["simulator_version"],
            source_node_ids=tuple(payload.get("source_node_ids", ())),
            profile=profile,
            demand_m3s=self._frame_from_payload(payload.get("demand")),
            pressure_m=self._frame_from_payload(payload.get("pressure")),
            water_age_hours=self._frame_from_payload(payload.get("water_age")),
            tracer_percent=self._frame_from_payload(payload.get("tracer")),
            pump_energy_kwh=payload.get("pump_energy_kwh"),
            pump_cost=payload.get("pump_cost"),
            complete=bool(payload.get("complete", False)),
            unstable=bool(payload.get("unstable", True)),
            cache_hit=cache_hit,
        )

    def _cache_key(self, *, profile: Any, plan: Any, operation: str) -> str:
        if self.cache is None:
            return ""
        return self.cache.key(
            network=self._network_fingerprint(),
            state=self.state_hash(),
            profile=profile,
            plan=plan,
            simulator={"name": self.simulator_name, "version": self.simulator_version},
            config={
                "operation": operation,
                "minimum_pressure_m": self.minimum_pressure_m,
                "minimum_service_availability": self.minimum_service_availability,
                "pressure_required_m": self.pressure_required_m,
                "energy_price_per_kwh": self.energy_price_per_kwh,
            },
        )

    def _pump_energy(self, results: Any, model: Any) -> tuple[float | None, float | None]:
        pump_names = list(getattr(model, "pump_name_list", ()))
        if not pump_names:
            return 0.0, 0.0
        head = results.node.get("head")
        flow = results.link.get("flowrate") if hasattr(results, "link") else None
        if head is None or flow is None:
            return None, None
        index = np.asarray(head.index, dtype=float)
        intervals = np.diff(index)
        if not len(intervals):
            intervals = np.asarray([float(model.options.time.hydraulic_timestep)])
        else:
            intervals = np.concatenate((intervals, intervals[-1:]))
        energy_joules = 0.0
        for name in pump_names:
            pump = model.get_link(name)
            head_gain = (
                head[pump.end_node_name].to_numpy(dtype=float)
                - head[pump.start_node_name].to_numpy(dtype=float)
            )
            rate = np.abs(flow[name].to_numpy(dtype=float))
            energy_joules += float(
                np.sum(np.maximum(0.0, head_gain) * rate * intervals * 1000.0 * 9.80665 / 0.75)
            )
        energy_kwh = energy_joules / 3_600_000.0
        return energy_kwh, energy_kwh * self.energy_price_per_kwh

    def evaluate_plan(self, plan: OperationalPlan) -> HydraulicEvaluation:
        """Apply a prescreened plan to a copy and evaluate exact hydraulic thresholds."""

        plan_payload = plan.model_dump(mode="json", exclude={"created_at"})
        cache_key = self._cache_key(profile=None, plan=plan_payload, operation="plan-evaluation")
        cached = self.cache.get(cache_key) if self.cache else None
        if cached is not None:
            try:
                return HydraulicEvaluation(
                    consequences=ConsequenceMetrics.model_validate(cached["consequences"]),
                    rejection_codes=tuple(cached["rejection_codes"]),
                    state_hash=cached["state_hash"],
                    pump_energy_kwh=cached.get("pump_energy_kwh"),
                    pump_cost=cached.get("pump_cost"),
                    cache_hit=True,
                )
            except (KeyError, TypeError, ValueError):
                self.cache.invalidate(cache_key)
        self._consume_budget("plan-evaluation")
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
        energy, cost = self._pump_energy(results, model)

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
        evaluation = HydraulicEvaluation(
            consequences=metrics,
            rejection_codes=tuple(codes),
            state_hash=self.state_hash(plan),
            pump_energy_kwh=energy,
            pump_cost=cost,
        )
        if self.cache:
            self.cache.put(
                cache_key,
                {
                    "consequences": metrics.model_dump(mode="json"),
                    "rejection_codes": list(codes),
                    "state_hash": evaluation.state_hash,
                    "pump_energy_kwh": energy,
                    "pump_cost": cost,
                },
            )
        return evaluation

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
            "nodes": [
                (
                    name,
                    type(model.get_node(name)).__name__,
                    float(getattr(model.get_node(name), "elevation", 0.0)),
                    float(getattr(model.get_node(name), "base_head", 0.0)),
                    float(getattr(model.get_node(name), "init_level", 0.0)),
                    tuple(
                        float(item.base_value)
                        for item in getattr(model.get_node(name), "demand_timeseries_list", ())
                    ),
                )
                for name in sorted(model.node_name_list)
            ],
            "links": [
                (
                    name,
                    type(model.get_link(name)).__name__,
                    model.get_link(name).start_node_name,
                    model.get_link(name).end_node_name,
                    str(model.get_link(name).initial_status),
                    float(getattr(model.get_link(name), "length", 0.0)),
                    float(getattr(model.get_link(name), "diameter", 0.0)),
                    float(getattr(model.get_link(name), "roughness", 0.0)),
                )
                for name in sorted(model.link_name_list)
            ],
            "duration": int(model.options.time.duration),
            "hydraulic_timestep": int(model.options.time.hydraulic_timestep),
            "quality_timestep": int(model.options.time.quality_timestep),
            "patterns": {
                name: [float(value) for value in model.get_pattern(name).multipliers]
                for name in sorted(model.pattern_name_list)
            },
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


def calculate_consequences(
    simulator: HydraulicSimulator,
    simulation: IncidentSimulation,
    *,
    threshold_mg_l: float,
    population_by_node: Mapping[str, float] | None = None,
    operation_count: int = 0,
) -> ConsequenceMetrics:
    """Public integration helper for exact simulation consequence calculation."""

    return simulator.calculate_consequences(
        simulation,
        threshold_mg_l=threshold_mg_l,
        population_by_node=population_by_node,
        operation_count=operation_count,
    )
