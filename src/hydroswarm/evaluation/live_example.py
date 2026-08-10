"""SUB-12.1 P1 #4: known reference inputs for the LIVE example vertical
slice.

The LIVE example judge path must use the real production API/pipeline
(real network import, real incident creation, real analysis, real WNTR
verification) -- but its *inputs* are the frozen golden scenario, so a
judge doesn't need to source their own EPANET network or telemetry.
Deliberately separate from `hydroswarm.evaluation.golden` (not a
modification of it): GoldenScenarioRunner drives its own fixed,
deterministic classical-only demo workflow through a hand-built
controller; this module only computes and exposes real, WNTR-simulated
reference *inputs* for the LIVE example to submit through the actual
`/api/networks/import`, `/api/incidents`, and `/api/incidents/{id}/samples`
endpoints, where the real production pipeline decides what happens next.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hydroswarm.simulation import HydraulicSimulator, IncidentSourceProfile

CANDIDATES = ("J1", "J2", "J3", "J4")
TRUE_SOURCE = "J2"
INITIAL_SENSOR = "R1"


def build_live_example_inputs(frozen_scenario_dir: str | Path) -> dict[str, Any]:
    """Real, WNTR-simulated reference inputs for the LIVE example.

    `frozen_scenario_dir` should come from
    `hydroswarm.runtime.paths.resolve_frozen_scenario_dir()` -- the same
    env-var-first / project-root-relative / cwd-relative priority every
    other frozen-asset resolver in this codebase uses, so a packaged/
    container deployment can never silently disagree with a source
    checkout about which fixture files are being read.

    Returns the frozen golden network's own `.inp` text (so the frontend
    imports the *exact same* real network file through the real import
    endpoint -- not a second, parallel representation of it), the real
    initial sensor observation, and a real simulated concentration for
    EVERY candidate node under the true source's profile -- so the LIVE
    example can submit a physically accurate observation for whichever
    node the real production pipeline recommends, which is not guaranteed
    to match GoldenScenarioRunner's separate deterministic-classical demo
    path's own recommendation.
    """
    frozen = Path(frozen_scenario_dir).resolve()
    network_path = frozen / "golden_network.inp"
    scenario = json.loads((frozen / "golden_scenario.json").read_text())

    profile_kwargs = scenario["source_profile"]
    simulator = HydraulicSimulator(network_path)
    simulation = simulator.simulate_hypothesis(
        IncidentSourceProfile(source_node_id=TRUE_SOURCE, **profile_kwargs),
        include_diagnostics=False,
    )
    sample_time = scenario["sample_time_seconds"]
    candidate_signatures = {
        node: float(simulation.concentration_mg_l.loc[sample_time, node]) for node in CANDIDATES
    }

    return {
        "network_filename": "golden_network.inp",
        "network_inp_text": network_path.read_text(encoding="utf-8"),
        "true_source": TRUE_SOURCE,
        "candidate_nodes": list(CANDIDATES),
        "initial_observation": {
            "sensor_id": f"S-{INITIAL_SENSOR}",
            "node_id": INITIAL_SENSOR,
            "concentration_mg_l": 0.0,
            "pressure_m": 132.0,
        },
        # Real WNTR-simulated concentration at each candidate node, under
        # the true source's real profile, at the golden scenario's real
        # sample time -- not fabricated, not the golden demo's own
        # (noise-injected) observed value. Keyed by node id so the
        # frontend can look up whichever node the live pipeline
        # recommends, not just J2.
        "candidate_signatures_mg_l": candidate_signatures,
        "sample_time_seconds": sample_time,
        "contamination_threshold_mg_l": scenario["contamination_threshold_mg_l"],
    }
