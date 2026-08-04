"""Multi-step incident trajectory serialization (overnight-plan.txt Task 2.6).

HydroSwarm is a sequential decision system, but the existing
AgentTrajectory/TrajectoryStep (hydroswarm.training.data) are a lightweight
hash-chained audit record -- state_hash, action, verifier_decision -- built
for replay/audit verification, not for reconstructing the full state a
Scout/Strategist/next-step model would need to train on. This module adds a
richer, versioned snapshot type alongside it (not a replacement) so:

- supervised Scout training can see the actual candidate set and observation
  history a sample recommendation was conditioned on;
- supervised Strategist training can see the actual verifier feedback that
  preceded a plan revision;
- next-step supervision can see the actual remaining budgets a controller
  used to decide COLLECT_SAMPLE vs GENERATE_PLANS vs ABSTAIN;
- full-loop evaluation can replay a trajectory exactly; and
- a future DAgger-style expansion has real state to imitate against,
  without redesigning storage again.

Reuses the existing validated domain schemas (IncidentState, SensorObservation
already inside it, OperationalAction, PlanVerification) rather than
duplicating their fields.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from hydroswarm.domain import IncidentState, OperationalAction, PlanVerification

TRAJECTORY_SCHEMA_VERSION = "hydroswarm-trajectory-v2"


@dataclass(frozen=True, slots=True)
class RemainingBudgets:
    samples: int
    exact_simulations: int
    actions: int

    def __post_init__(self) -> None:
        if self.samples < 0 or self.exact_simulations < 0 or self.actions < 0:
            raise ValueError("remaining budgets cannot be negative")

    def to_json(self) -> dict[str, int]:
        return {"samples": self.samples, "exact_simulations": self.exact_simulations, "actions": self.actions}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "RemainingBudgets":
        return cls(
            samples=int(payload["samples"]),
            exact_simulations=int(payload["exact_simulations"]),
            actions=int(payload["actions"]),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryState:
    """One full step of a trajectory: the state the controller observed,
    what led into it, what budget remained, and what it decided next."""

    step_index: int
    incident_state: IncidentState
    remaining_budgets: RemainingBudgets
    previous_action: OperationalAction | None = None
    verifier_feedback: PlanVerification | None = None
    selected_next_action: OperationalAction | None = None
    resulting_next_state_hash: str | None = None
    schema_version: str = TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "step_index": self.step_index,
            "incident_state": self.incident_state.model_dump(mode="json"),
            "remaining_budgets": self.remaining_budgets.to_json(),
            "previous_action": (
                self.previous_action.model_dump(mode="json") if self.previous_action is not None else None
            ),
            "verifier_feedback": (
                self.verifier_feedback.model_dump(mode="json") if self.verifier_feedback is not None else None
            ),
            "selected_next_action": (
                self.selected_next_action.model_dump(mode="json")
                if self.selected_next_action is not None
                else None
            ),
            "resulting_next_state_hash": self.resulting_next_state_hash,
        }

    @property
    def state_hash(self) -> str:
        """Deterministic hash of this step's own content, for chain
        verification against the previous step's resulting_next_state_hash."""

        payload = self.to_json()
        payload.pop("resulting_next_state_hash")  # a step does not hash its own forward pointer
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TrajectoryState":
        version = payload.get("schema_version")
        if version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                f"incompatible trajectory schema version {version!r}; expected "
                f"{TRAJECTORY_SCHEMA_VERSION!r}"
            )
        return cls(
            step_index=int(payload["step_index"]),
            incident_state=IncidentState.model_validate(payload["incident_state"]),
            remaining_budgets=RemainingBudgets.from_json(payload["remaining_budgets"]),
            previous_action=(
                OperationalAction.model_validate(payload["previous_action"])
                if payload.get("previous_action") is not None
                else None
            ),
            verifier_feedback=(
                PlanVerification.model_validate(payload["verifier_feedback"])
                if payload.get("verifier_feedback") is not None
                else None
            ),
            selected_next_action=(
                OperationalAction.model_validate(payload["selected_next_action"])
                if payload.get("selected_next_action") is not None
                else None
            ),
            resulting_next_state_hash=payload.get("resulting_next_state_hash"),
        )


class TrajectoryIntegrityError(Exception):
    """Raised when a FullTrajectory's steps are not contiguous or the hash
    chain between consecutive steps does not match."""


@dataclass(frozen=True, slots=True)
class FullTrajectory:
    trajectory_id: str
    scenario_id: str
    steps: tuple[TrajectoryState, ...]

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.scenario_id:
            raise ValueError("trajectory_id and scenario_id are required")
        if not self.steps:
            raise ValueError("a trajectory must contain at least one step")
        indices = [step.step_index for step in self.steps]
        if indices != list(range(len(indices))):
            raise TrajectoryIntegrityError("trajectory steps must be contiguous and zero-based")
        for current, following in zip(self.steps, self.steps[1:]):
            if current.resulting_next_state_hash is not None and current.resulting_next_state_hash != following.state_hash:
                raise TrajectoryIntegrityError(
                    f"step {current.step_index}'s resulting_next_state_hash does not match "
                    f"step {following.step_index}'s actual state_hash"
                )

    def to_json(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "scenario_id": self.scenario_id,
            "steps": [step.to_json() for step in self.steps],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "FullTrajectory":
        return cls(
            trajectory_id=payload["trajectory_id"],
            scenario_id=payload["scenario_id"],
            steps=tuple(TrajectoryState.from_json(step) for step in payload["steps"]),
        )
