"""Authoritative WNTR hydraulic and water-quality simulation wrapper."""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import queue
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
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


@dataclass(frozen=True, slots=True)
class WeightedSourceHypothesis:
    """One bounded credible source hypothesis with its posterior
    probability -- used for runtime plan-consequence evaluation when no
    ground-truth source profile is available (important-issues.txt
    requirement 7). Not presented as ground truth: `probability` reflects
    calibrated localization evidence, never a certainty."""

    profile: IncidentSourceProfile
    probability: float

    def __post_init__(self) -> None:
        if not (0.0 < self.probability <= 1.0):
            raise ValueError("hypothesis probability must be in (0, 1]")


#: important-issues.txt requirement 7: "Prefer a maximum of 3 credible
#: SourceHypothesis values."
MAXIMUM_EVALUATION_HYPOTHESES = 3


@dataclass(frozen=True, slots=True)
class PlanEvaluationContext:
    """Explicit incident-evaluation assumptions for exposure-aware plan
    verification (important-issues.txt requirement 5) -- carried
    alongside the plan rather than hidden inside evaluate_plan()'s own
    defaults.

    Exactly one of `source_profile` (training: the scenario's exact
    ground-truth source, per requirement 6) or `hypotheses` (runtime: a
    bounded credible set built from calibrated localization evidence, per
    requirement 7) must be supplied.
    """

    contamination_threshold_mg_l: float
    source_profile: IncidentSourceProfile | None = None
    hypotheses: tuple[WeightedSourceHypothesis, ...] = ()
    population_by_node: Mapping[str, float] | None = None
    #: "posterior_weighted" (default; the expected-consequence headline
    #: number) or "worst_case" (the conservative headline number) -- either
    #: way, both are always computed and reported when hypotheses are used
    #: (requirement 7: "report both"). Ignored in single-profile mode.
    aggregation_policy: str = "posterior_weighted"
    #: Bumped whenever calculate_exposure_consequences' formula changes.
    consequence_policy_version: str = "exposure-consequences-v1"

    def __post_init__(self) -> None:
        if self.contamination_threshold_mg_l < 0:
            raise ValueError("contamination threshold cannot be negative")
        has_profile = self.source_profile is not None
        has_hypotheses = bool(self.hypotheses)
        if has_profile == has_hypotheses:
            raise ValueError(
                "PlanEvaluationContext requires exactly one of source_profile "
                "(training/known ground truth) or hypotheses (runtime/bounded credible set)"
            )
        if len(self.hypotheses) > MAXIMUM_EVALUATION_HYPOTHESES:
            raise ValueError(
                f"bounded hypothesis evaluation supports at most {MAXIMUM_EVALUATION_HYPOTHESES} "
                "credible source hypotheses"
            )
        if self.aggregation_policy not in ("posterior_weighted", "worst_case"):
            raise ValueError(f"unknown aggregation_policy: {self.aggregation_policy!r}")

    @property
    def profiles(self) -> tuple[tuple[IncidentSourceProfile, float], ...]:
        """Every (profile, probability) pair to evaluate; probability is
        1.0 (a certainty) only in ground-truth single-profile mode."""

        if self.source_profile is not None:
            return ((self.source_profile, 1.0),)
        return tuple((h.profile, h.probability) for h in self.hypotheses)

    @property
    def population_map_identity(self) -> str:
        if not self.population_by_node:
            return "none"
        return _digest({str(k): float(v) for k, v in sorted(self.population_by_node.items())})

    @property
    def hypothesis_identity(self) -> dict[str, Any]:
        """Deterministic identity payload for cache keys and provenance
        (requirement 9: "source profile/hypothesis-set identity")."""

        if self.source_profile is not None:
            return {"mode": "single", "profile": self.source_profile.as_dict()}
        return {
            "mode": "hypotheses",
            "hypotheses": [
                {"profile": h.profile.as_dict(), "probability": h.probability} for h in self.hypotheses
            ],
        }


@dataclass(frozen=True, slots=True)
class HypothesisConsequence:
    """One evaluated hypothesis' exact plan-modified consequence."""

    profile: IncidentSourceProfile
    probability: float
    consequences: ConsequenceMetrics
    rejection_codes: tuple[str, ...]


@dataclass(frozen=True)
class PlanExposureEvaluation:
    """Exposure-aware plan verification result (important-issues.txt
    requirements 3/4/7): the canonical output of `evaluate_plan_consequences`.

    `consequences` is the headline ConsequenceMetrics per
    `evaluation_context.aggregation_policy` -- the exact result in
    single-profile (training) mode, or the requested aggregate (posterior-
    weighted expected or worst-case) across the bounded hypothesis set in
    runtime mode. `worst_case_consequences` is always populated alongside
    it in hypothesis mode (never presented as the only number, per
    requirement 7's "report both"). `rejection_codes` is the safety-
    conservative UNION of PRESSURE_BELOW_MINIMUM/SERVICE_BELOW_MINIMUM
    across every evaluated hypothesis (requirement 4: hydraulic safety
    authority must not be averaged away by a benign hypothesis).
    """

    consequences: ConsequenceMetrics
    worst_case_consequences: ConsequenceMetrics | None
    rejection_codes: tuple[str, ...]
    state_hash: str
    per_hypothesis: tuple[HypothesisConsequence, ...]
    #: Real number of new EPANET simulations this evaluation call itself
    #: consumed (0 when every underlying simulate_incident_plan call hit
    #: cache) -- requirement 10: "must not silently count as one simulator
    #: call."
    exact_simulation_count: int
    pump_energy_kwh: float | None = None
    pump_cost: float | None = None
    cache_hit: bool = False


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


#: Sentinel used only when weighting a "never contained within the
#: simulation window" hypothesis (containment_time_minutes=None) against
#: hypotheses that WERE contained, for the posterior-weighted headline
#: aggregate -- not a claim about actual containment time. Matches
#: hydroswarm.planning.plan_value_policy's own DEFAULT_CONTAINMENT_SCALE_
#: MINUTES treatment of "not contained" as maximally-bad-but-bounded rather
#: than unboundedly bad; kept as an independent local constant rather than
#: importing planning/plan_value_policy here to avoid inverting this
#: module's (simulation/) lower architectural layering under planning/.
_UNCONTAINED_AGGREGATION_SENTINEL_MINUTES = 240.0


def _aggregate_hypothesis_consequences(
    items: Sequence["HypothesisConsequence"],
    *,
    operation_count: int,
    mode: str,
) -> ConsequenceMetrics:
    """Combine per-hypothesis exact ConsequenceMetrics into one headline
    metric (important-issues.txt requirement 7: "report both
    posterior-weighted expected consequence and worst-case consequence").

    mode="posterior_weighted": probability-weighted expectation, weights
    renormalized over the evaluated hypotheses (a bounded credible set need
    not itself sum to exactly 1). mode="worst_case": the maximally-adverse
    value per field independently (minimum_pressure_m takes the minimum,
    every other field takes the maximum) -- a conservative composite used
    for safety authority, not necessarily any single hypothesis' own
    outcome.
    """

    if not items:
        raise ValueError("cannot aggregate zero hypothesis consequences")
    if len(items) == 1:
        return items[0].consequences.model_copy(update={"operation_count": operation_count})

    if mode == "worst_case":
        containment_values = [item.consequences.containment_time_minutes for item in items]
        return ConsequenceMetrics(
            population_impacted=max(item.consequences.population_impacted for item in items),
            contaminant_mass_consumed_mg=max(item.consequences.contaminant_mass_consumed_mg for item in items),
            volume_above_threshold_l=max(item.consequences.volume_above_threshold_l for item in items),
            contaminated_pipe_extent_m=max(item.consequences.contaminated_pipe_extent_m for item in items),
            minimum_pressure_m=min(item.consequences.minimum_pressure_m for item in items),
            pressure_violation_minutes=max(item.consequences.pressure_violation_minutes for item in items),
            unserved_demand_l=max(item.consequences.unserved_demand_l for item in items),
            service_availability=min(item.consequences.service_availability for item in items),
            operation_count=operation_count,
            containment_time_minutes=(
                None
                if all(value is None for value in containment_values)
                else max(
                    value if value is not None else _UNCONTAINED_AGGREGATION_SENTINEL_MINUTES
                    for value in containment_values
                )
            ),
            exposure_evaluated=True,
        )
    if mode != "posterior_weighted":
        raise ValueError(f"unknown aggregation mode: {mode!r}")

    total_probability = sum(item.probability for item in items) or 1.0
    weights = [item.probability / total_probability for item in items]

    def _weighted(getter: Callable[[ConsequenceMetrics], float]) -> float:
        return sum(w * getter(item.consequences) for w, item in zip(weights, items, strict=True))

    containment_values = [item.consequences.containment_time_minutes for item in items]
    containment_weighted = (
        None
        if all(value is None for value in containment_values)
        else sum(
            w * (value if value is not None else _UNCONTAINED_AGGREGATION_SENTINEL_MINUTES)
            for w, value in zip(weights, containment_values, strict=True)
        )
    )
    return ConsequenceMetrics(
        population_impacted=_weighted(lambda c: c.population_impacted),
        contaminant_mass_consumed_mg=_weighted(lambda c: c.contaminant_mass_consumed_mg),
        volume_above_threshold_l=_weighted(lambda c: c.volume_above_threshold_l),
        contaminated_pipe_extent_m=_weighted(lambda c: c.contaminated_pipe_extent_m),
        minimum_pressure_m=_weighted(lambda c: c.minimum_pressure_m),
        pressure_violation_minutes=_weighted(lambda c: c.pressure_violation_minutes),
        unserved_demand_l=_weighted(lambda c: c.unserved_demand_l),
        service_availability=_weighted(lambda c: c.service_availability),
        operation_count=operation_count,
        containment_time_minutes=containment_weighted,
        exposure_evaluated=True,
    )


#: The hydraulic snapshot time (seconds into the simulation) that governed
#: feature construction has always used for computing a representative
#: static-ish node/edge feature snapshot -- hydroswarm.training.corpus.
#: build_feature_context (training) and hydroswarm.cli's fixed-inference
#: verification both call calculate_state(3_600) explicitly. Discovered
#: via core-issues.txt Phase 3 item 18 (fitting calibration against the
#: real, exact deployed HybridInferencePipeline): HybridInferencePipeline
#: .analyze() called calculate_state() with no argument, which defaults to
#: the network's LAST simulated timestamp -- for a typical 24-hour
#: simulation (86,400s) that is nowhere near the 3,600s snapshot the model
#: was actually trained on, and measurably degraded the real trained
#: model's neural-branch accuracy at live inference (confirmed by an A/B
#: comparison against a real trained checkpoint: the true source's belief
#: went from an undifferentiated ~0.30 among candidates to a correctly-
#: ranked ~0.49 once the snapshot time matched training). Use this
#: constant at every call site that builds model input features from a
#: HydraulicSimulator snapshot, so train and serve can never silently
#: diverge on it again.
FEATURE_SNAPSHOT_TIME_SECONDS = 3_600

#: core-issues5.txt delta item 5: the governed default safety thresholds
#: HydraulicSimulator/PlanVerifier actually verify plans against when no
#: per-incident override is supplied (every current call site, including
#: hydroswarm.api.app.verify_plan, constructs a HydraulicSimulator without
#: overriding either). Named module-level constants (not bare literal
#: constructor defaults) so a verification-context identity can reference
#: the exact same values a change to either constant is required to also
#: change, rather than duplicating "10.0"/"0.90" as a second, independently
#: -maintained literal.
DEFAULT_MINIMUM_PRESSURE_M = 10.0
DEFAULT_MINIMUM_SERVICE_AVAILABILITY = 0.90


def _multiprocessing_worker_entrypoint(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    result_queue: "multiprocessing.Queue[tuple[bool, Any]]",
) -> None:
    """The actual `multiprocessing.Process` target for
    HydraulicSimulator._run_with_timeout.

    Must be a real module-level function, not a nested closure: under the
    "spawn" start method (mandatory on Windows; POSIX still uses "fork",
    which has no such restriction) the child interpreter imports the
    target by its qualified name rather than inheriting the parent's
    memory, so the target -- and everything reachable from `args` -- must
    be picklable-by-reference. A lambda or a function defined inside
    another function is not."""

    try:
        result_queue.put((True, function(*args)))
    except BaseException as exc:  # propagate simulator failures to caller
        result_queue.put((False, exc))


def _invoke_wntr_simulator(model: Any) -> Any:
    """Picklable stand-in for the ``lambda: wntr.sim.WNTRSimulator(model).
    run_sim()`` closure _run_hydraulics used to pass directly -- see
    _multiprocessing_worker_entrypoint's docstring for why a closure can't
    cross a "spawn" process boundary. `model` itself must also be
    picklable for the same reason; verified empirically (not merely
    assumed) in test_simulator_extended.py's spawn-context regression
    test."""
    return wntr.sim.WNTRSimulator(model).run_sim()


def _invoke_epanet_simulator(model: Any, operation: str) -> Any:
    """Picklable stand-in for _run_epanet's former nested `execute()`
    closure -- same "spawn"-cannot-pickle-a-closure reason as
    _invoke_wntr_simulator above."""
    with tempfile.TemporaryDirectory(prefix="hydroswarm-epanet-") as directory:
        prefix = str(Path(directory) / operation)
        return wntr.sim.EpanetSimulator(model).run_sim(file_prefix=prefix)


class HydraulicSimulator:
    """Load, validate and simulate water networks using WNTR as authority."""

    simulator_name = "WNTRSimulator"

    def __init__(
        self,
        network: Any | str | Path | None = None,
        *,
        minimum_pressure_m: float = DEFAULT_MINIMUM_PRESSURE_M,
        minimum_service_availability: float = DEFAULT_MINIMUM_SERVICE_AVAILABILITY,
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

    def _run_with_timeout(
        self, operation: str, function: Callable[..., Any], args: tuple[Any, ...] = ()
    ) -> Any:
        """Run `function(*args)` with a hard wall-clock deadline in a real,
        killable OS subprocess -- not a daemon thread (core-issues.txt: a
        thread that exceeds its timeout cannot be forcibly stopped in
        Python and simply keeps running in the background, consuming CPU
        for however long the underlying EPANET/WNTR call takes, or forever
        if it is genuinely hung).

        `function` must be a module-level function (not a lambda or a
        nested closure) and every element of `args` must be picklable --
        see _multiprocessing_worker_entrypoint's docstring for why: under
        "spawn" (mandatory on Windows) the child interpreter imports the
        target by reference rather than inheriting the parent's memory.

        Uses the "fork" start method deliberately on POSIX, not "spawn":
        this call sits on the live incident-analysis latency path (real
        profiling shows single-digit milliseconds today), and spawn's
        from-scratch interpreter/import cost (numpy/pandas/wntr/torch)
        turns that into multiple seconds per call -- an unacceptable
        regression for a function invoked on every hydraulic simulation.
        Forking a multi-threaded parent (e.g. a live server) carries a
        narrow, well-documented risk of a child inheriting a lock another
        thread held at fork time; unlike the daemon-thread version this
        replaces, that failure mode is still fully contained -- a wedged
        child is a real OS process, so the timeout below still forcibly
        terminates it and this call still raises SimulationTimeoutError
        rather than hanging or leaking the process.

        Windows has no fork() syscall at all -- multiprocessing.get_context
        ("fork") raises ValueError there unconditionally, not a latency
        tradeoff to make but a hard platform constraint. "spawn" is the
        only start method Windows supports, so this falls back to it there
        (accepting its slower per-call startup cost, and the module-level-
        function/picklable-args requirement above, on that platform only;
        POSIX production/CI behavior is unchanged)."""

        context = multiprocessing.get_context("spawn" if sys.platform == "win32" else "fork")
        result_queue: multiprocessing.Queue = context.Queue(maxsize=1)

        process = context.Process(
            target=_multiprocessing_worker_entrypoint,
            args=(function, args, result_queue),
            name=f"hydroswarm-{operation}",
            daemon=True,
        )
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
        results = self._run_with_timeout("hydraulics", _invoke_wntr_simulator, (model,))
        self._validate_results(results)
        return results

    def _run_epanet(self, model: Any, operation: str) -> Any:
        results = self._run_with_timeout(operation, _invoke_epanet_simulator, (model, operation))
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

    @staticmethod
    def _inject_incident_sources(model: Any, profile: IncidentSourceProfile) -> None:
        """Add the profile's mass-source patterns to `model` in place.

        Shared by `simulate_hypothesis` (pristine network) and
        `simulate_incident_plan` (plan-modified network) so both source-
        injection paths stay byte-for-byte identical
        (important-issues.txt requirement 2: "Refactor simulate_hypothesis()
        and/or the new path to share the source-injection/EPANET
        implementation rather than duplicating it").
        """

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

    def _build_incident_simulation(
        self,
        model: Any,
        results: Any,
        profile: IncidentSourceProfile,
        *,
        include_diagnostics: bool,
        extra_state_payload: Mapping[str, Any] | None = None,
    ) -> IncidentSimulation:
        """Shared EPANET-result -> IncidentSimulation assembly for both
        `simulate_hypothesis` and `simulate_incident_plan`."""

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
        state_payload: dict[str, Any] = {
            "network": self._network_fingerprint(),
            "profile": profile.as_dict(),
            "quality": quality.round(9).to_dict(),
        }
        if extra_state_payload:
            state_payload.update(extra_state_payload)
        state_hash = _digest(state_payload)
        return IncidentSimulation(
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
        self._inject_incident_sources(model, profile)
        results = self._run_epanet(model, "incident")
        simulation = self._build_incident_simulation(
            model, results, profile, include_diagnostics=include_diagnostics
        )
        if self.cache:
            self.cache.put(cache_key, self._incident_payload(simulation))
        return simulation

    def simulate_incident_plan(
        self,
        plan: OperationalPlan,
        incident_profile: IncidentSourceProfile,
        *,
        include_diagnostics: bool = False,
    ) -> IncidentSimulation:
        """Canonical plan-aware incident simulation path
        (important-issues.txt requirement 2).

        Copies the exact scenario/runtime network state, applies the
        `OperationalPlan`'s actions (pipe closures, flushing, etc.), injects
        the incident source profile into that SAME modified network, then
        runs ONE EPANET chemical-transport pass -- returning concentration,
        pressure, demand, flow and energy together (requirement 8: never a
        separate, unrelated hydraulic-only pass followed by an independent
        chemical pass). Shares `_inject_incident_sources`/
        `_build_incident_simulation` with `simulate_hypothesis` so the two
        paths cannot silently diverge (requirement 2).
        """

        unknown = sorted(
            {source.node_id for source in incident_profile.sources} - set(self.network.node_name_list)
        )
        if unknown:
            raise ValueError(f"unknown incident source nodes: {', '.join(unknown)}")
        plan_payload = plan.model_dump(mode="json", exclude={"created_at"})
        cache_key = self._cache_key(
            profile=incident_profile.as_dict(),
            plan=plan_payload,
            operation=f"incident-plan-diagnostics-{include_diagnostics}",
        )
        cached = self.cache.get(cache_key) if self.cache else None
        if cached is not None:
            try:
                return self._incident_from_payload(cached, incident_profile, cache_hit=True)
            except (KeyError, TypeError, ValueError, SimulationIncompleteError):
                self.cache.invalidate(cache_key)
        self._consume_budget("incident-plan")
        model = self._prepared_network()
        self._apply_actions(model, plan)
        self._inject_incident_sources(model, incident_profile)
        results = self._run_epanet(model, "incident-plan")
        simulation = self._build_incident_simulation(
            model,
            results,
            incident_profile,
            include_diagnostics=include_diagnostics,
            extra_state_payload={"plan": plan_payload},
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
        requested_demand_m3s: pd.DataFrame | None = None,
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
        #: Reservoirs/tanks are boundary nodes, not real service points --
        #: WNTR/EPANET report a reservoir's "pressure" as its fixed head
        #: relative to elevation, which is frequently ~0 m by construction
        #: and NOT a real low-pressure violation. evaluate_plan()'s legacy
        #: hydraulic-only path has always filtered to junction_name_list
        #: before computing minimum_pressure_m; evaluation/golden.py's
        #: hand-merged exposure workaround independently discovered and
        #: applied the same filter. Apply it here too, once, centrally, so
        #: every canonical-evaluator caller (training labels, golden
        #: fixture, live API) agrees -- found by this pass's own real
        #: execution: an unfiltered reservoir pressure of ~0 m falsely
        #: triggered PRESSURE_BELOW_MINIMUM for every plan, including
        #: NO_ACTION, on the very first real run of this new code path.
        junctions = [name for name in self.network.junction_name_list if name in simulation.pressure_m.columns]
        concentration = simulation.concentration_mg_l[
            [name for name in junctions if name in simulation.concentration_mg_l.columns]
        ]
        demand = simulation.demand_m3s[[name for name in junctions if name in simulation.demand_m3s.columns]]
        pressure = simulation.pressure_m[junctions]
        requested = (
            requested_demand_m3s[[name for name in junctions if name in requested_demand_m3s.columns]]
            if requested_demand_m3s is not None
            else None
        )
        return calculate_exposure_consequences(
            concentration,
            demand,
            threshold_mg_l=threshold_mg_l,
            population_by_node=population_by_node,
            pipe_endpoints=pipe_endpoints,
            pressure_m=pressure,
            minimum_pressure_m=self.minimum_pressure_m,
            #: important-issues.txt requirement 3/8: pass the plan-free
            #: baseline demand so service_availability/unserved_demand_l
            #: reflect real delivery shortfall relative to what the network
            #: would have demanded with NO plan applied, not relative to
            #: the plan's own (possibly reduced) demand -- without this, a
            #: plan that closes a pipe and thereby suppresses demand at the
            #: nodes behind it would otherwise show no unserved-demand
            #: penalty at all.
            requested_demand_m3s=requested,
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

    def _cache_key(
        self,
        *,
        profile: Any,
        plan: Any,
        operation: str,
        extra_config: Mapping[str, Any] | None = None,
    ) -> str:
        if self.cache is None:
            return ""
        config: dict[str, Any] = {
            "operation": operation,
            "minimum_pressure_m": self.minimum_pressure_m,
            "minimum_service_availability": self.minimum_service_availability,
            "pressure_required_m": self.pressure_required_m,
            "energy_price_per_kwh": self.energy_price_per_kwh,
        }
        if extra_config:
            config.update(extra_config)
        return self.cache.key(
            network=self._network_fingerprint(),
            state=self.state_hash(),
            profile=profile,
            plan=plan,
            simulator={"name": self.simulator_name, "version": self.simulator_version},
            config=config,
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
        """Apply a prescreened plan to a copy and evaluate exact hydraulic
        thresholds ONLY.

        Legacy hydraulic-only path (important-issues.txt requirement 12):
        the returned ConsequenceMetrics has `exposure_evaluated=False` and
        its contamination-exposure fields (`contaminant_mass_consumed_mg`,
        `volume_above_threshold_l`, `population_impacted`,
        `contaminated_pipe_extent_m`, `containment_time_minutes`) are
        Pydantic defaults, NOT measured. Callers must not present them as
        real exposure. Use `evaluate_plan_consequences` with an explicit
        `PlanEvaluationContext` whenever incident evidence is available and
        a real contamination-exposure consequence is required -- this
        method remains only for plans with no associated incident/source
        context at all (e.g. generic pre-incident network safety checks)."""

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

    def _baseline_requested_demand(self) -> pd.DataFrame:
        """No-plan, no-incident baseline demand -- cached independently of
        plan/profile since it depends only on network state
        (important-issues.txt requirement 8: "a separately cached
        no-action/baseline demand state is acceptable")."""

        cache_key = self._cache_key(profile=None, plan=None, operation="baseline-demand")
        cached = self.cache.get(cache_key) if self.cache else None
        if cached is not None:
            frame = self._frame_from_payload(cached.get("demand"))
            if frame is not None:
                return frame
            if self.cache:
                self.cache.invalidate(cache_key)
        self._consume_budget("baseline-demand")
        results = self._run_hydraulics(self._prepared_network())
        demand = results.node["demand"].copy()
        if self.cache:
            self.cache.put(cache_key, {"demand": self._frame_payload(demand)})
        return demand

    def evaluate_plan_consequences(
        self,
        plan: OperationalPlan,
        evaluation_context: PlanEvaluationContext,
    ) -> PlanExposureEvaluation:
        """Canonical exposure-aware plan verification path
        (important-issues.txt requirements 3/4/7/8/9/10).

        For every hypothesis in `evaluation_context` (a single ground-truth
        profile in training, or a bounded credible set at runtime), applies
        the plan and injects that source into the SAME modified network via
        `simulate_incident_plan`, then builds the complete ConsequenceMetrics
        (contamination exposure AND hydraulic pressure/service, together,
        from the SAME EPANET chemical-transport run -- never an independent
        hydraulic-only pass followed by an unrelated chemical pass) via
        `calculate_consequences`. PRESSURE_BELOW_MINIMUM/
        SERVICE_BELOW_MINIMUM are derived per-hypothesis from exact
        simulator results and UNIONED across hypotheses, so hydraulic
        safety authority can never be averaged away by a benign hypothesis
        (requirement 4).
        """

        profiles = evaluation_context.profiles
        threshold = evaluation_context.contamination_threshold_mg_l
        population = evaluation_context.population_by_node
        operation_count = sum(action.action_type != ActionType.END_PLAN for action in plan.actions)
        plan_payload = plan.model_dump(mode="json", exclude={"created_at"})

        cache_key = self._cache_key(
            profile=evaluation_context.hypothesis_identity,
            plan=plan_payload,
            operation="plan-exposure-evaluation",
            extra_config={
                "contamination_threshold_mg_l": threshold,
                "population_map_identity": evaluation_context.population_map_identity,
                "consequence_policy_version": evaluation_context.consequence_policy_version,
                "aggregation_policy": evaluation_context.aggregation_policy,
            },
        )
        cached = self.cache.get(cache_key) if self.cache else None
        if cached is not None:
            try:
                return PlanExposureEvaluation(
                    consequences=ConsequenceMetrics.model_validate(cached["consequences"]),
                    worst_case_consequences=(
                        ConsequenceMetrics.model_validate(cached["worst_case_consequences"])
                        if cached.get("worst_case_consequences") is not None
                        else None
                    ),
                    rejection_codes=tuple(cached["rejection_codes"]),
                    state_hash=cached["state_hash"],
                    per_hypothesis=(),
                    exact_simulation_count=0,
                    pump_energy_kwh=cached.get("pump_energy_kwh"),
                    pump_cost=cached.get("pump_cost"),
                    cache_hit=True,
                )
            except (KeyError, TypeError, ValueError):
                self.cache.invalidate(cache_key)

        baseline_demand = self._baseline_requested_demand()
        runs_before = self.exact_runs
        per_hypothesis: list[HypothesisConsequence] = []
        energy_total = 0.0
        cost_total = 0.0
        energy_known = True
        for profile, probability in profiles:
            simulation = self.simulate_incident_plan(plan, profile, include_diagnostics=False)
            metrics = self.calculate_consequences(
                simulation,
                threshold_mg_l=threshold,
                population_by_node=population,
                operation_count=operation_count,
                requested_demand_m3s=baseline_demand,
            )
            codes: list[str] = []
            if metrics.minimum_pressure_m < self.minimum_pressure_m:
                codes.append("PRESSURE_BELOW_MINIMUM")
            if metrics.service_availability < self.minimum_service_availability:
                codes.append("SERVICE_BELOW_MINIMUM")
            per_hypothesis.append(
                HypothesisConsequence(
                    profile=profile,
                    probability=probability,
                    consequences=metrics,
                    rejection_codes=tuple(codes),
                )
            )
            if simulation.pump_energy_kwh is None:
                energy_known = False
            else:
                energy_total += simulation.pump_energy_kwh
                cost_total += simulation.pump_cost or 0.0

        #: requirement 10: the real number of NEW exact EPANET runs this one
        #: verification consumed -- may exceed 1 for a multi-hypothesis
        #: evaluation, and is 0 when every simulate_incident_plan call hit
        #: cache. Never silently collapsed to "one simulator call."
        exact_simulation_count = self.exact_runs - runs_before

        rejection_codes = tuple(sorted({code for item in per_hypothesis for code in item.rejection_codes}))
        aggregated = _aggregate_hypothesis_consequences(
            per_hypothesis, operation_count=operation_count, mode="posterior_weighted"
        )
        worst_case = (
            _aggregate_hypothesis_consequences(per_hypothesis, operation_count=operation_count, mode="worst_case")
            if len(per_hypothesis) > 1
            else None
        )
        consequences = (
            worst_case
            if evaluation_context.aggregation_policy == "worst_case" and worst_case is not None
            else aggregated
        )

        state_hash = _digest(
            {
                "network": self._network_fingerprint(),
                "plan": plan_payload,
                "hypotheses": evaluation_context.hypothesis_identity,
                "threshold": threshold,
                "population_map_identity": evaluation_context.population_map_identity,
            }
        )
        evaluation = PlanExposureEvaluation(
            consequences=consequences,
            worst_case_consequences=worst_case,
            rejection_codes=rejection_codes,
            state_hash=state_hash,
            per_hypothesis=tuple(per_hypothesis),
            exact_simulation_count=exact_simulation_count,
            pump_energy_kwh=energy_total if energy_known else None,
            pump_cost=cost_total if energy_known else None,
        )
        if self.cache:
            self.cache.put(
                cache_key,
                {
                    "consequences": consequences.model_dump(mode="json"),
                    "worst_case_consequences": (
                        worst_case.model_dump(mode="json") if worst_case is not None else None
                    ),
                    "rejection_codes": list(rejection_codes),
                    "state_hash": state_hash,
                    "pump_energy_kwh": evaluation.pump_energy_kwh,
                    "pump_cost": evaluation.pump_cost,
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
    requested_demand_m3s: pd.DataFrame | None = None,
) -> ConsequenceMetrics:
    """Public integration helper for exact simulation consequence calculation."""

    return simulator.calculate_consequences(
        simulation,
        threshold_mg_l=threshold_mg_l,
        population_by_node=population_by_node,
        operation_count=operation_count,
        requested_demand_m3s=requested_demand_m3s,
    )
