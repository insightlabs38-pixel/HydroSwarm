"""Import-safe real WNTR smoke used by the CLI self-test on Windows."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import wntr

from .network import build_wntr_network


def main() -> int:
    model = build_wntr_network()
    results = wntr.sim.WNTRSimulator(model).run_sim()
    pressure = results.node["pressure"]
    if pressure.empty or not np.isfinite(pressure.to_numpy(dtype=float)).all():
        raise RuntimeError("WNTR self-test produced invalid pressure output")
    print(json.dumps({"simulation_sha256": hashlib.sha256(pressure.to_numpy().tobytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
