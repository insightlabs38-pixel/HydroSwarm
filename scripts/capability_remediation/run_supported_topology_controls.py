#!/usr/bin/env python3
"""Run clean API-path controls for every governed supported topology.

The frozen 264-row LIVE matrix intentionally focuses on the golden reference,
stress conditions, the coastal development topology, and a loop-grid scale
slice.  This small supplementary control records equivalent nominal runs for
the two other governed families without altering that frozen matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

from hydroswarm.evaluation.live_robustness import (
    Condition,
    load_protocol,
    locked_test_opened,
    run_condition,
)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    if locked_test_opened(root):
        raise SystemExit("locked_test_opened is true; refusing supported-topology controls")
    protocol = load_protocol(root / "reports/evaluation/live-robustness/protocol.json")
    conditions = [
        Condition(
            name=f"nominal-{network_id}-{seed}",
            perturbation_type="nominal_supported_control",
            perturbation_level="clean_operational",
            seed=seed,
            network_id=network_id,
            topology_class="governed_topology",
        )
        for network_id, seed in (
            # ``run_condition`` derives its analysis-time origin from the
            # seed.  Keep this supplementary development slice before the
            # current wall clock so receipt-time causality is exercised rather
            # than intentionally filtering every report as future evidence.
            ("branched-loop", 7195), ("branched-loop", 7196), ("branched-loop", 7197),
            ("loop-grid", 7198), ("loop-grid", 7199), ("loop-grid", 7200),
        )
    ]
    rows = [run_condition(root, condition, protocol=protocol) for condition in conditions]
    if locked_test_opened(root):
        raise SystemExit("locked_test_opened changed during supported-topology controls")
    output = root / "reports/evaluation/capability-remediation/supported-topology-controls.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} supported-topology control rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
