"""Deterministic HydroSwarm specialist agents and finite-state controller."""

from .base import AgentPermissionError, AgentTimeoutError
from .controller import ALLOWED_TRANSITIONS, SwarmController, SwarmLimits
from .schemas import (
    CandidateScore,
    FSMState,
    FallbackMode,
    PlanCandidate,
    ReplayResult,
    ScoutAction,
    ScoutOutput,
    SentinelOutput,
    StrategistOutput,
    SwarmEvent,
    SwarmResult,
    ToolPermission,
)
from .scout import HydroScout
from .sentinel import HydroSentinel
from .strategist import HydroStrategist

__all__ = [
    "ALLOWED_TRANSITIONS",
    "AgentPermissionError",
    "AgentTimeoutError",
    "CandidateScore",
    "FSMState",
    "FallbackMode",
    "HydroScout",
    "HydroSentinel",
    "HydroStrategist",
    "PlanCandidate",
    "ReplayResult",
    "ScoutAction",
    "ScoutOutput",
    "SentinelOutput",
    "StrategistOutput",
    "SwarmController",
    "SwarmEvent",
    "SwarmLimits",
    "SwarmResult",
    "ToolPermission",
]
