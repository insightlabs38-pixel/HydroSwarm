#!/usr/bin/env python3
"""Run the frozen API-driven LIVE robustness protocol."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--start", type=int, default=0, help="zero-based deterministic condition offset")
    parser.add_argument("--limit", type=int, default=None, help="number of deterministic conditions to run")
    parser.add_argument("--resume", action="store_true", help="append a disjoint deterministic slice to existing raw rows")
    parser.add_argument("--replace", action="store_true", help="supersede the same deterministic run IDs after a documented harness correction")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(root / "reports/evaluation/live-robustness/protocol.json")
    if locked_test_opened(root):
        raise SystemExit("locked_test_opened is true; refusing LIVE study")
    conditions = predeclared_conditions(repetitions=args.repetitions)
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    conditions = conditions[args.start:]
    if args.limit is not None:
        conditions = conditions[:args.limit]
    output_dir = root / "reports/evaluation/live-robustness"
    rows = []
    results_path = output_dir / "results.json"
    if (args.resume or args.replace) and results_path.exists():
        rows = json.loads(results_path.read_text(encoding="utf-8"))
    new_rows = [run_condition(root, condition, protocol=protocol) for condition in conditions]
    old_ids = {row["run_id"] for row in rows}
    new_ids = {row["run_id"] for row in new_rows}
    if args.replace:
        if not old_ids.issuperset(new_ids):
            raise SystemExit("--replace requires existing deterministic run IDs")
        rows = [row for row in rows if row["run_id"] not in new_ids]
    elif old_ids.intersection(new_ids):
        raise SystemExit("refusing duplicate run IDs while resuming LIVE study")
    rows.extend(new_rows)
    if locked_test_opened(root):
        raise SystemExit("locked_test_opened changed during LIVE study")
    summary = write_artifacts(output_dir, rows, locked_opened_after=False)
    print(f"wrote {len(new_rows)} LIVE rows ({len(rows)} total); invariant failures: {len(summary['invariant_failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
