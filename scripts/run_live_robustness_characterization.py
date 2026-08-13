#!/usr/bin/env python3
"""Run the frozen API-driven LIVE robustness protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydroswarm.evaluation.live_robustness import (
    locked_test_opened,
    load_protocol,
    predeclared_conditions,
    run_condition,
    write_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="development smoke only; recorded as incomplete if used")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(root / "reports/evaluation/live-robustness/protocol.json")
    if locked_test_opened(root):
        raise SystemExit("locked_test_opened is true; refusing LIVE study")
    conditions = predeclared_conditions(repetitions=args.repetitions)
    if args.limit is not None:
        conditions = conditions[:args.limit]
    rows = [run_condition(root, condition, protocol=protocol) for condition in conditions]
    if locked_test_opened(root):
        raise SystemExit("locked_test_opened changed during LIVE study")
    summary = write_artifacts(root / "reports/evaluation/live-robustness", rows, locked_opened_after=False)
    print(f"wrote {len(rows)} LIVE rows; invariant failures: {len(summary['invariant_failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
