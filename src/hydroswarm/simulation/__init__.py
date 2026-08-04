"""Deterministic water-network construction helpers."""

from .network import NetworkDefinition, build_networkx_network, build_wntr_network
from .context_cache import HydraulicContextCache, ScenarioHydraulicContextKey
from .consequences import (
    PlanOutcome,
    RankedPlanOutcome,
    calculate_exposure_consequences,
    rank_plan_outcomes,
)
from .verifier import PlanVerifier
from .wrapper import (
    HydraulicEvaluation,
    HydraulicSimulator,
    HydraulicState,
    IncidentSimulation,
    IncidentSource,
    IncidentSourceProfile,
    SimulatedSample,
    SimulationBudgetExceeded,
    SimulationError,
    SimulationIncompleteError,
    SimulationTimeoutError,
    SimulationUnstableError,
    calculate_consequences,
)

__all__ = [
    "HydraulicContextCache",
    "HydraulicEvaluation",
    "HydraulicSimulator",
    "HydraulicState",
    "ScenarioHydraulicContextKey",
    "IncidentSimulation",
    "IncidentSource",
    "IncidentSourceProfile",
    "NetworkDefinition",
    "PlanOutcome",
    "PlanVerifier",
    "RankedPlanOutcome",
    "SimulatedSample",
    "SimulationBudgetExceeded",
    "SimulationError",
    "SimulationIncompleteError",
    "SimulationTimeoutError",
    "SimulationUnstableError",
    "build_networkx_network",
    "build_wntr_network",
    "calculate_consequences",
    "calculate_exposure_consequences",
    "rank_plan_outcomes",
]
