"""SUB-12.1 P1 #4: known reference inputs for the LIVE example vertical
slice.

The LIVE example judge path must use the real production API/pipeline
(real network import, real incident creation, real analysis, real WNTR
verification) -- but its *inputs* are a frozen, known scenario, so a judge
doesn't need to source their own EPANET network or telemetry. Deliberately
separate from `hydroswarm.evaluation.golden` (not a modification of it):
GoldenScenarioRunner drives its own fixed, deterministic classical-only
demo workflow through a hand-built controller against its own
golden_network.inp; this module computes and exposes real, WNTR-simulated
reference *inputs* against a different, calibration-validated network for
the LIVE example to submit through the actual `/api/networks/import`,
`/api/incidents`, and `/api/incidents/{id}/samples` endpoints, where the
real production pipeline decides what happens next.
"""

from __future__ import annotations

import json
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hydroswarm.simulation import HydraulicSimulator, IncidentSourceProfile

# `data/frozen/live_example_network.inp` is a byte-for-byte copy of
# `data/topologies/loop-grid.inp` -- NOT the golden/reference network. The
# frozen production calibration artifact
# (models/hydrocore-v4-release/calibration.json: validated_topology_hashes)
# only recognizes three specific network topologies; a live smoke run
# confirmed golden_network.inp is not one of them, so driving the real
# pipeline against it always yields `calibrated=False` and
# PLANNING_SUPPRESSED (CALIBRATION_INVALID_OR_MISSING) at plan generation --
# a genuine, correctly-governed refusal, not a bug to route around.
# loop-grid.inp's topology hash IS one of the three validated hashes (see
# tests/integration/test_production_runtime_wiring.py, which already proves
# this exact network reaches real FULL_HYBRID), so the LIVE example must use
# it instead, kept as its own copy so this module never has to reach outside
# `data/frozen` (the only directory Docker/the release bundle ship).
CANDIDATES = ("J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8")
TRUE_SOURCE = "J6"
# The real production signature-localization pipeline's governed policy
# (models/hydrocore-v4-release/signature-policy-manifest.json:
# sensor_layout_policy = "all_junctions_as_sensor_candidates") only
# recognizes junctions as valid sensor nodes -- a live smoke run against the
# golden network's reservoir-carried initial sensor confirmed the real
# pipeline's `_signature_observations` fails with "requires at least one
# valid concentration observation" for a non-junction sensor. J1 is
# upstream of TRUE_SOURCE (R3 -> J1 -> J5 -> J6), so it also gives a real,
# physically honest pre-contamination (0.0) initial reading.
INITIAL_SENSOR = "J1"


def build_live_example_inputs(frozen_scenario_dir: str | Path) -> dict[str, Any]:
    """Real, WNTR-simulated reference inputs for the LIVE example.

    `frozen_scenario_dir` should come from
    `hydroswarm.runtime.paths.resolve_frozen_scenario_dir()` -- the same
    env-var-first / project-root-relative / cwd-relative priority every
    other frozen-asset resolver in this codebase uses, so a packaged/
    container deployment can never silently disagree with a source
    checkout about which fixture files are being read.

    Returns the frozen live-example network's own `.inp` text (so the
    frontend imports the *exact same* real, calibration-validated network
    file through the real import endpoint -- not a second, parallel
    representation of it), the real initial sensor observation, and a real
    simulated concentration for EVERY node in the network under the true
    source's profile -- so the LIVE example can submit a physically
    accurate observation for whichever node the real production pipeline
    recommends, not just the eight CANDIDATES junctions (a live smoke run
    against a different, four-junction network previously surfaced the
    real sampling recommendation naming a node outside its own classical
    CANDIDATES set, which GoldenScenarioRunner's separate deterministic-
    classical demo path never has to consider).
    """
    frozen = Path(frozen_scenario_dir).resolve()
    network_path = frozen / "live_example_network.inp"
    scenario_path = frozen / "live_example_scenario.json"
    scenario_bytes = scenario_path.read_bytes()
    network_bytes = network_path.read_bytes()
    scenario = json.loads(scenario_bytes)

    profile_kwargs = scenario["source_profile"]
    simulator = HydraulicSimulator(network_path)
    simulation = simulator.simulate_hypothesis(
        IncidentSourceProfile(source_node_id=TRUE_SOURCE, **profile_kwargs),
        include_diagnostics=False,
    )
    sample_time = scenario["sample_time_seconds"]
    all_nodes = list(simulation.concentration_mg_l.columns)
    if sample_time not in simulation.concentration_mg_l.index:
        raise RuntimeError(f"frozen LIVE sample time {sample_time!r} is not a simulated timestep")
    # Every value is an observation from the same declared instant.  Do not
    # combine each node's peak from a different timestep with one shared
    # sample timestamp: that would be physically impossible evidence.
    node_signatures = {
        node: float(simulation.concentration_mg_l.loc[sample_time, node]) for node in all_nodes
    }
    # A real, WNTR-computed hydraulic state (pre-incident, time zero) for
    # the initial observation's pressure reading -- not a guessed constant.
    initial_state = simulator.calculate_state(0)
    initial_pressure_m = float(initial_state.pressure_m[INITIAL_SENSOR])

    return {
        "network_filename": "live_example_network.inp",
        "network_inp_text": network_bytes.decode("utf-8"),
        "true_source": TRUE_SOURCE,
        "candidate_nodes": list(CANDIDATES),
        "initial_observation": {
            "sensor_id": f"S-{INITIAL_SENSOR}",
            "node_id": INITIAL_SENSOR,
            "concentration_mg_l": 0.0,
            "pressure_m": initial_pressure_m,
        },
        # Real WNTR-simulated concentration at EVERY network node (not
        # just the eight classical candidates), under the true source's
        # real profile, at this scenario's real sample time -- not
        # fabricated. Keyed by node id so the frontend can look up whichever
        # node the live pipeline recommends.
        "candidate_signatures_mg_l": node_signatures,
        "sample_time_seconds": sample_time,
        "contamination_threshold_mg_l": scenario["contamination_threshold_mg_l"],
        "execution_mode": "LIVE",
        "input_source": "FROZEN_REFERENCE_SCENARIO",
        "computed_at": datetime.now(UTC).isoformat(),
        "network_sha256": hashlib.sha256(network_bytes).hexdigest(),
        "scenario_sha256": hashlib.sha256(scenario_bytes).hexdigest(),
        "input_sha256": hashlib.sha256(network_bytes + scenario_bytes).hexdigest(),
    }
