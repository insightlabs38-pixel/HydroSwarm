"""Scenario-specific hydraulic context caching (overnight-plan.txt Task 1.4).

HydraulicSimulator.state_hash() already fingerprints the live WNTR network
object -- node elevation/base_head/init_level (tank levels), demand
timeseries (demand pattern), link initial_status (pump/valve state) as
currently applied to the model, roughness/length/diameter, and the
simulation duration/timestep configuration. Because it re-reads the model
object at call time, it already changes whenever _apply_actions() mutates
valve or pump status, so a scenario builder that calls it per-scenario (not
once per network family) already gets scenario-specific state rather than a
single shared context.

What was missing is (a) an explicit key type that also captures the two
things state_hash() cannot see -- the simulation timestamp being queried and
the state-estimator configuration used to reconstruct that state from
telemetry -- so two scenarios that share a network configuration but differ
in either of those still get distinct cache entries, and (b) a cache
wrapper that records hit/miss counts explicitly and fails closed (a
corrupted or key-mismatched entry is always treated as a miss, never
silently returned).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .wrapper import HydraulicSimulator


@dataclass(frozen=True, slots=True)
class ScenarioHydraulicContextKey:
    network_state_hash: str
    simulator_version: str
    simulation_timestamp_seconds: int
    state_estimator_config_hash: str
    schema_version: str = "hydroswarm-hydraulic-context-v1"

    def __post_init__(self) -> None:
        if not self.network_state_hash or not self.simulator_version or not self.state_estimator_config_hash:
            raise ValueError(
                "network_state_hash, simulator_version, and state_estimator_config_hash are required"
            )
        if self.simulation_timestamp_seconds < 0:
            raise ValueError("simulation_timestamp_seconds cannot be negative")

    @classmethod
    def from_simulator(
        cls,
        simulator: HydraulicSimulator,
        *,
        simulation_timestamp_seconds: int,
        state_estimator_config_hash: str,
    ) -> "ScenarioHydraulicContextKey":
        """Build the key from a simulator's *current* state. Call this after
        any scenario-specific mutation (e.g. _apply_actions for a valve/pump
        change) so network_state_hash reflects that scenario, not a
        pre-mutation shared baseline."""

        return cls(
            network_state_hash=simulator.state_hash(),
            simulator_version=simulator.simulator_version,
            simulation_timestamp_seconds=simulation_timestamp_seconds,
            state_estimator_config_hash=state_estimator_config_hash,
        )

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "network_state_hash": self.network_state_hash,
            "simulator_version": self.simulator_version,
            "simulation_timestamp_seconds": self.simulation_timestamp_seconds,
            "state_estimator_config_hash": self.state_estimator_config_hash,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class HydraulicContextCache:
    """Explicit-hit/miss cache of dynamic hydraulic graph contexts, keyed by
    ScenarioHydraulicContextKey.digest.

    Stores whatever JSON-safe context payload the caller provides (e.g. a
    serialized DirectedHydraulicGraph plus travel times), so it stays
    decoupled from any one graph representation.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: ScenarioHydraulicContextKey) -> Path:
        return self.root / f"{key.digest}.json"

    def get(self, key: ScenarioHydraulicContextKey) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("key_digest") != key.digest:
                self.misses += 1
                return None
        except (OSError, json.JSONDecodeError, KeyError):
            self.misses += 1
            return None
        self.hits += 1
        return payload["context"]

    def put(self, key: ScenarioHydraulicContextKey, context: dict[str, Any]) -> None:
        path = self._path(key)
        temporary = path.with_suffix(".tmp.json")
        temporary.write_text(
            json.dumps({"key_digest": key.digest, "context": context}, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    def get_or_build(
        self, key: ScenarioHydraulicContextKey, builder: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        cached = self.get(key)
        if cached is not None:
            return cached
        context = builder()
        self.put(key, context)
        return context
