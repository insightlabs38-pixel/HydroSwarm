"""M11.6A-1 -- deterministic procedural topology generator for
`locked_topology_test` (task Section 7), plus the formal topology-novelty
check (task Section 8).

This module contains ONLY the frozen, deterministic procedural generator and
novelty rule. It is constrained to produce topologies with junction counts in
[9, 12] -- outside every prior topology's [4, 8] range -- and never inspects
any HydroCore prediction or model output. Rejection sampling is permitted
ONLY for the predeclared structural/physical validity criteria listed in
`m11_6a_design`, never for a learned/model result.

It materializes NOTHING: it returns WNTR models for a caller (the M11.6A-2
materializer) to serialize. In M11.6A-1 it is exercised only against
non-locked smoke fixtures and its own deterministic self-tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

import m11_6a_design as design  # noqa: E402

from hydroswarm.data.scenarios import network_sha256  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402

try:
    import wntr  # noqa: E402
except ImportError:  # pragma: no cover - optional dependency path
    wntr = None

#: Frozen structural bounds (task Section 7). Junction count is kept OUTSIDE
#: every prior topology's [4, 8] range so the novelty rule is decisive by
#: construction; the full signature/hash check remains the frozen rule.
JUNCTION_COUNT_BASE = 9          # junction_count = 9 + topology_index
CYCLE_RANK_BASE = 1              # cycle_rank = 1 + topology_index
RESERVOIR_HEAD_MIN_M = 130.0
RESERVOIR_HEAD_MAX_M = 140.0
JUNCTION_ELEVATION_MIN_M = 95.0
JUNCTION_ELEVATION_MAX_M = 108.0
JUNCTION_DEMAND_MIN_M3S = 0.003
JUNCTION_DEMAND_MAX_M3S = 0.007
PIPE_LENGTH_MIN_M = 300.0
PIPE_LENGTH_MAX_M = 700.0
PIPE_DIAMETER_MIN_M = 0.25
PIPE_DIAMETER_MAX_M = 0.40
PIPE_ROUGHNESS_MIN = 100.0
PIPE_ROUGHNESS_MAX = 130.0
#: Feasibility floor evaluated over JUNCTION nodes only (reservoirs/tanks
#: report boundary head as 0.0 pressure in WNTR's PDD output and must not be
#: part of the physical feasibility check).
PRESSURE_FLOOR_M = 5.0

#: Same 24-step diurnal pattern the project already governs
#: (run_m7_topology.DIURNAL_PATTERN), reused verbatim, not invented here.
DIURNAL_PATTERN = (
    0.62, 0.56, 0.52, 0.50, 0.55, 0.72,
    1.00, 1.22, 1.18, 1.05, 0.98, 1.02,
    1.08, 1.04, 0.98, 0.96, 1.08, 1.28,
    1.38, 1.24, 1.08, 0.94, 0.80, 0.70,
)


def topology_spec() -> dict[str, Any]:
    return {
        "family_grammar": {
            "reservoirs": "one R1 (base_head uniform [120, 135] m)",
            "junction_count": "9 + topology_index (index 0..3 => 9..12 junctions)",
            "cycle_rank": "1 + topology_index (index 0..3 => 1..4)",
            "connectivity": "spanning path backbone R1->J1->...->JN (guarantees connectivity), plus `cycle_rank` deterministic extra edges (no multi-edges) to create loops",
            "pipe_length_m": [PIPE_LENGTH_MIN_M, PIPE_LENGTH_MAX_M],
            "pipe_diameter_m": [PIPE_DIAMETER_MIN_M, PIPE_DIAMETER_MAX_M],
            "pipe_roughness": [PIPE_ROUGHNESS_MIN, PIPE_ROUGHNESS_MAX],
            "junction_demand_m3s": [JUNCTION_DEMAND_MIN_M3S, JUNCTION_DEMAND_MAX_M3S],
            "junction_elevation_m": [JUNCTION_ELEVATION_MIN_M, JUNCTION_ELEVATION_MAX_M],
            "hydraulic_feasibility": (
                "must run cleanly under WNTR/EPANET hydraulics: finite pressures "
                f"and min pressure >= {PRESSURE_FLOOR_M} m under the preregistered baseline"
            ),
            "timesteps": {"pattern_timestep_s": 3600, "hydraulic_timestep_s": 3600, "quality_timestep_s": 300, "duration_s": 86400},
        },
        "rejection_criteria_permitted": [
            "disconnected graph",
            "invalid EPANET model",
            "simulator failure",
            "impossible pressure/hydraulic state under the preregistered baseline",
            "duplicate topology hash",
            "structural identity/isomorphism to a previously exposed topology",
        ],
        "rejection_criteria_forbidden": [
            "bad HydroCore prediction", "low accuracy", "high disagreement",
            "poor calibration", "undesirable planning result", "any learned/model output",
        ],
        "candidate_ordering": "candidate counter 0, 1, 2, ... (deterministic next-candidate rule)",
        "max_candidate_attempts": design.MAX_TOPOLOGY_CANDIDATE_ATTEMPTS,
        "exhaustion_behavior": "IT MUST BLOCK (never loosen constraints or choose a new seed)",
    }


def build_procedural_topology_model(seed: int, junction_count: int, cycle_rank: int) -> Any:
    """Deterministically build one procedural WNTR model.

    ``seed`` fully determines the geometry; ``junction_count``/``cycle_rank``
    are the frozen structural targets (never tuned to HydroCore). A spanning
    path backbone guarantees connectivity; ``cycle_rank`` extra edges create
    the requested loop structure.
    """

    if wntr is None:
        raise ImportError("WNTR is required for the procedural topology generator")
    if not (9 <= junction_count <= 12):
        raise ValueError(f"junction_count must be in [9, 12], got {junction_count}")
    if not (1 <= cycle_rank <= 4):
        raise ValueError(f"cycle_rank must be in [1, 4], got {cycle_rank}")

    rng = np.random.default_rng(seed)
    model = wntr.network.WaterNetworkModel()
    model.name = f"m11-6-locked-topology-j{junction_count}-r{cycle_rank}"
    model.add_pattern("diurnal", list(DIURNAL_PATTERN))
    model.add_reservoir("R1", base_head=float(rng.uniform(RESERVOIR_HEAD_MIN_M, RESERVOIR_HEAD_MAX_M)), coordinates=(0.0, 0.0))
    for index in range(junction_count):
        name = f"J{index + 1}"
        x = 800.0 + index * 600.0 + float(rng.uniform(-150.0, 150.0))
        y = float(rng.uniform(-400.0, 400.0))
        elevation = float(rng.uniform(JUNCTION_ELEVATION_MIN_M, JUNCTION_ELEVATION_MAX_M))
        demand = float(rng.uniform(JUNCTION_DEMAND_MIN_M3S, JUNCTION_DEMAND_MAX_M3S))
        model.add_junction(name, base_demand=demand, demand_pattern="diurnal", elevation=elevation, coordinates=(x, y))

    # Spanning path backbone R1 -> J1 -> J2 -> ... -> JN.
    for index in range(junction_count):
        start = "R1" if index == 0 else f"J{index}"
        end = f"J{index + 1}"
        model.add_pipe(
            f"P_{start}_{end}", start, end,
            length=float(rng.uniform(PIPE_LENGTH_MIN_M, PIPE_LENGTH_MAX_M)),
            diameter=float(rng.uniform(PIPE_DIAMETER_MIN_M, PIPE_DIAMETER_MAX_M)),
            roughness=float(rng.uniform(PIPE_ROUGHNESS_MIN, PIPE_ROUGHNESS_MAX)),
            minor_loss=0.0, initial_status="OPEN",
        )

    # Deterministic extra loop edges (no backbone-adjacent pairs, no multi-edges).
    pairs = [
        (f"J{a}", f"J{b}")
        for a in range(1, junction_count + 1)
        for b in range(a + 1, junction_count + 1)
        if b != a + 1
    ]
    rng.shuffle(pairs)
    added = 0
    for start, end in pairs:
        if added >= cycle_rank:
            break
        model.add_pipe(
            f"P_{start}_{end}_loop{added}", start, end,
            length=float(rng.uniform(PIPE_LENGTH_MIN_M, PIPE_LENGTH_MAX_M)),
            diameter=float(rng.uniform(PIPE_DIAMETER_MIN_M, PIPE_DIAMETER_MAX_M)),
            roughness=float(rng.uniform(PIPE_ROUGHNESS_MIN, PIPE_ROUGHNESS_MAX)),
            minor_loss=0.0, initial_status="OPEN",
        )
        added += 1

    model.options.time.pattern_timestep = 3_600
    model.options.time.hydraulic_timestep = 3_600
    model.options.time.quality_timestep = 300
    model.options.time.duration = 24 * 3_600
    return model


def topology_graph_signature(network: Any) -> dict[str, Any]:
    """The frozen canonical graph-structural signature of a topology."""
    graph = nx.Graph()
    graph.add_nodes_from(network.node_name_list)
    graph.add_edges_from(
        (network.get_link(name).start_node_name, network.get_link(name).end_node_name)
        for name in network.link_name_list
    )
    degree_profile = sorted(dict(graph.degree()).values())
    return design.graph_signature(
        node_count=len(network.node_name_list),
        junction_count=len(network.junction_name_list),
        link_count=len(network.link_name_list),
        cycle_rank=(
            len(network.link_name_list)
            - len(network.node_name_list)
            + nx.number_connected_components(graph)
        ),
        degree_profile=degree_profile,
    )


def topology_is_feasible(network: Any) -> bool:
    """Predeclared structural/physical validity check.

    Runs WNTR/EPANET hydraulics under the preregistered baseline and requires
    finite pressures with min pressure >= PRESSURE_FLOOR_M. This is the ONLY
    rejection criterion that inspects the simulator -- never a model output.
    """

    if wntr is None:
        return False
    try:
        simulator = HydraulicSimulator(network)
        results = simulator._run_hydraulics(simulator._prepared_network())
        junction_names = list(network.junction_name_list)
        if not junction_names:
            return False
        pressure = results.node["pressure"][junction_names].to_numpy(dtype=float)
        if not np.isfinite(pressure).all():
            return False
        return bool(float(pressure.min()) >= PRESSURE_FLOOR_M)
    except Exception:
        return False


def is_novel_topology(
    network: Any,
    *,
    seen_network_hashes: tuple[str, ...] = (),
    seen_signatures: tuple[dict[str, Any], ...] = (),
) -> tuple[bool, list[str]]:
    """Frozen novelty rule (task Section 8).

    Returns (novel, reasons). A topology is unseen iff its canonical
    ``network_sha256`` is not in the frozen prior-hash set, its
    graph-structural signature matches no prior signature, and it is not a
    duplicate of anything already generated in the locked_topology_test set.
    """

    signature = topology_graph_signature(network)
    sha = network_sha256(network)
    reasons: list[str] = []
    if design.is_prior_topology(signature, sha):
        reasons.append("matches a prior topology (network_sha256 or graph signature)")
    if sha in seen_network_hashes:
        reasons.append("duplicate network_sha256 within generated set")
    if any(design.signatures_equal(signature, existing) for existing in seen_signatures):
        reasons.append("duplicate graph signature within generated set")
    return (not reasons), reasons


def generate_locked_topology(master_hex: str, topology_index: int) -> dict[str, Any]:
    """Deterministically generate one novel locked_topology_test topology.

    Uses the frozen rejection-sampling rule: candidate counter 0, 1, 2, ...
    each derived via ``derive_seed(master, "TOPOLOGY_TEST_NETWORK", index,
    counter)``. If the predeclared attempt budget is exhausted, it BLOCKS
    (raises) -- it never loosens constraints or picks a new seed.
    """

    junction_count = JUNCTION_COUNT_BASE + topology_index
    cycle_rank = CYCLE_RANK_BASE + topology_index
    for candidate in range(design.MAX_TOPOLOGY_CANDIDATE_ATTEMPTS):
        seed = design.derive_seed(master_hex, "TOPOLOGY_TEST_NETWORK", topology_index, candidate)
        model = build_procedural_topology_model(seed, junction_count, cycle_rank)
        if not topology_is_feasible(model):
            continue
        novel, reasons = is_novel_topology(model)
        if not novel:
            continue
        return {
            "topology_index": topology_index,
            "candidate_index": candidate,
            "seed": seed,
            "network": model,
            "network_sha256": network_sha256(model),
            "graph_signature": topology_graph_signature(model),
            "junction_count": junction_count,
            "cycle_rank": cycle_rank,
        }
    raise RuntimeError(
        f"locked_topology_test topology index {topology_index} exhausted the "
        f"predeclared attempt budget ({design.MAX_TOPOLOGY_CANDIDATE_ATTEMPTS}); "
        "materialization must BLOCK (constraints are not loosened, no new seed is chosen)"
    )
