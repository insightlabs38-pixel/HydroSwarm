"""Build or validate a checksummed source-signature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydroswarm.classical.signatures import SignatureBuilder, SignatureCache, SignatureCacheKey
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.simulation.wrapper import HydraulicSimulator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/signatures"))
    args = parser.parse_args()
    simulator = HydraulicSimulator(build_wntr_network())
    sensors = ("J1", "J2", "J3", "J4")
    times = tuple(range(0, 6 * 3600 + 1, 1800))
    key = SignatureCacheKey(
        network_hash=simulator.state_hash(),
        hydraulic_state_hash=simulator.state_hash(),
        simulator_version=simulator.simulator_version,
        configuration_hash="reference-v1",
        sensor_layout_hash="J1-J2-J3-J4",
    )
    artifact = SignatureBuilder(simulator, SignatureCache(args.output)).build_or_load(
        key=key,
        source_nodes=("J1", "J2", "J3", "J4"),
        start_time_bins=(0, 60),
        duration_bins=(30, 60),
        strength_bins=(0.5, 1.0),
        demand_regimes=("nominal",),
        sensor_nodes=sensors,
        sample_times_seconds=times,
    )
    print(json.dumps({
        "artifact_sha256": artifact.artifact_hash,
        "cache_hit": artifact.cache_hit,
        "hypotheses": len(artifact.hypotheses),
        "sensors": list(artifact.sensor_nodes),
        "cache_key": artifact.key.digest,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
