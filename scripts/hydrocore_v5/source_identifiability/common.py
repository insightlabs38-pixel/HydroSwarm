"""Shared, read-only infrastructure for the source-identifiability analysis.

Reproduces the exact randomized-network/incident state a frozen locked
scenario spec (`data/locked/m11-6/{locked_final_test,locked_topology_test}/
scenarios.jsonl`) was originally evaluated against, by calling the SAME
`WNTRScenarioGenerator.generate_with_network` code path the M11.6 locked
evaluator itself uses (mirrors `_reconstruct_scenario` in
`scripts/hydrocore_v5/run_m11_6_locked_evaluation.py`, reimplemented here
rather than imported so this module has no side effects from importing a
runnable script). Never opens, mutates, or re-scores the locked evaluation
itself -- this only replays already-frozen scenario *specs* to obtain the
physical signals HydroSwarm's own sensors could see, for a signature
comparison that never touches the frozen model or its predictions.
"""

from __future__ import annotations

import copy
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import wntr

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
SCRIPTS_HYDROCORE_V5 = ROOT / "scripts" / "hydrocore_v5"
for path in (SRC, SCRIPTS_HYDROCORE_V5):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    GeneratedScenario,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402

import m9_1_common as m91  # noqa: E402  (canonical per_row_metrics/paired_bootstrap)

LOCKED_ROOT = ROOT / "data" / "locked" / "m11-6"
LOCKED_FINAL_TEST = LOCKED_ROOT / "locked_final_test" / "scenarios.jsonl"
LOCKED_TOPOLOGY_TEST = LOCKED_ROOT / "locked_topology_test" / "scenarios.jsonl"
LOCKED_TOPOLOGIES_DIR = LOCKED_ROOT / "topologies"
M11_6_RAW_INCIDENTS = (
    ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m11" / "m11-6-final" / "m11-6-raw-incidents.jsonl"
)

OUTPUT_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "source-identifiability"

#: golden-reference / branched-loop / loop-grid are the three network
#: families the frozen v5 finalist was trained on (see m9-0/m9-0a summaries
#: and m11_6a_design.py's LOCKED_FINAL_FAMILIES) -- the "known topology" set.
KNOWN_FAMILY_LOADERS = {
    "golden-reference": lambda: build_wntr_network(),
    "branched-loop": lambda: wntr.network.WaterNetworkModel(
        str(ROOT / "data" / "topology-transfer" / "branched-loop.inp")
    ),
    "loop-grid": lambda: wntr.network.WaterNetworkModel(str(ROOT / "data" / "topologies" / "loop-grid.inp")),
}
KNOWN_NETWORK_FAMILIES = frozenset(KNOWN_FAMILY_LOADERS)
#: locked-topology-procedural is the one held-out family used for M11.6's
#: unseen-topology locked test -- genuinely never seen in training data.
UNSEEN_NETWORK_FAMILIES = frozenset({"locked-topology-procedural"})

_NETWORK_CACHE: dict[str, Any] = {}


def load_base_network(row: dict[str, Any]) -> Any:
    """Load (and cache) the exact base WNTR model a locked scenario row references.

    Verifies the recorded ``network_sha256`` before returning, so any drift
    between this analysis and the frozen scenario spec fails loudly instead
    of silently comparing signatures from the wrong network.
    """

    topology_id = row["topology_id"]
    cache_key = topology_id
    if cache_key not in _NETWORK_CACHE:
        if topology_id.startswith("locked-topology:"):
            index = topology_id.split(":", 1)[1]
            network = wntr.network.WaterNetworkModel(
                str(LOCKED_TOPOLOGIES_DIR / f"locked-topology-{index}.inp")
            )
        else:
            loader = KNOWN_FAMILY_LOADERS.get(row["network_family"])
            if loader is None:
                raise KeyError(f"no base-network loader registered for {row['network_family']!r}")
            network = loader()
        _NETWORK_CACHE[cache_key] = network
    network = _NETWORK_CACHE[cache_key]
    computed = network_sha256(network)
    if computed != row["network_sha256"]:
        raise ValueError(
            f"network identity mismatch for {topology_id}: "
            f"recorded {row['network_sha256']} != recomputed {computed}"
        )
    return network


@dataclass(frozen=True, slots=True)
class ReconstructedIncident:
    row: dict[str, Any]
    scenario: GeneratedScenario
    randomized_network: Any
    junctions: tuple[str, ...]
    sensor_nodes: tuple[str, ...]
    start_minute: int
    duration_minutes: int
    relative_strength: float
    injection_strength_mg_min: float
    base_strength_mg_min: float


def reconstruct_incident(row: dict[str, Any]) -> ReconstructedIncident:
    """Deterministically replay one locked scenario spec row.

    Faithful to `WNTRScenarioGenerator.generate_with_network`'s single
    `np.random.default_rng(seed)` draw order (source is fixed here so its
    draw is skipped, exactly as the original evaluation run did). Returns
    the post-randomization, pre-injection WNTR model so callers can run
    additional counterfactual `simulate_incident` calls against the
    identical hydraulic state without re-deriving any RNG draws.
    """

    network = load_base_network(row)
    generator_config = dict(row.get("generator_config") or {})
    stage = CurriculumStage(generator_config.pop("stage", "operational"))
    config = ScenarioGenerationConfig(
        seed=row["seed"],
        network_id=row["network_family"],
        network_family=row["network_family"],
        split=DatasetSplit.TEST,
        stage=stage,
        event_type=EventType(row.get("event_type", "contamination")),
        source_node=row["source_node"],
        **generator_config,
    )
    scenario, randomized = WNTRScenarioGenerator().generate_with_network(network, config)
    incident = scenario.manifest.incident
    is_contamination = config.event_type == EventType.CONTAMINATION
    injection_strength = (
        incident.relative_strength if is_contamination else 1e-9 / config.base_strength_mg_min
    )
    return ReconstructedIncident(
        row=row,
        scenario=scenario,
        randomized_network=randomized,
        junctions=tuple(sorted(randomized.junction_name_list)),
        sensor_nodes=scenario.sensor_nodes,
        start_minute=incident.start_minute,
        duration_minutes=incident.duration_minutes,
        relative_strength=incident.relative_strength,
        injection_strength_mg_min=config.base_strength_mg_min * injection_strength,
        base_strength_mg_min=config.base_strength_mg_min,
    )


def simulate_candidate(incident: ReconstructedIncident, candidate_node: str):
    """Run one counterfactual `simulate_incident` for `candidate_node`.

    Reuses `incident.randomized_network` (never mutated -- `HydraulicSimulator`
    deep-copies on construction and per run) so every candidate in the same
    incident is simulated under bit-for-bit identical hydraulics, timing,
    strength, and demand -- the "identical/controlled incident conditions"
    the pairwise-distance analysis requires.
    """

    simulator = HydraulicSimulator(incident.randomized_network)
    return simulator.simulate_incident(
        candidate_node,
        strength_mg_min=incident.injection_strength_mg_min,
        start_minute=incident.start_minute,
        duration_minutes=incident.duration_minutes,
    )


def undirected_graph(network: Any) -> nx.Graph:
    """Generic WNTR-model -> plain undirected `networkx.Graph`.

    Uses `WaterNetworkModel.to_graph()` (the same export the rest of the
    repo uses for structural queries, e.g. `HydraulicSimulator.validate()`)
    rather than the golden-reference-only `build_networkx_network()` helper,
    since candidates here span every locked network family.
    """

    return nx.Graph(network.to_graph())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deep_copy_network(network: Any) -> Any:
    return copy.deepcopy(network)


__all__ = [
    "ROOT",
    "LOCKED_ROOT",
    "LOCKED_FINAL_TEST",
    "LOCKED_TOPOLOGY_TEST",
    "LOCKED_TOPOLOGIES_DIR",
    "M11_6_RAW_INCIDENTS",
    "OUTPUT_ROOT",
    "KNOWN_FAMILY_LOADERS",
    "KNOWN_NETWORK_FAMILIES",
    "UNSEEN_NETWORK_FAMILIES",
    "load_base_network",
    "ReconstructedIncident",
    "reconstruct_incident",
    "simulate_candidate",
    "undirected_graph",
    "sha256_bytes",
    "deep_copy_network",
    "m91",
]
