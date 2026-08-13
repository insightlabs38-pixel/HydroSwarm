"""Run the frozen, locked-test-excluding robustness/scale characterization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydroswarm.evaluation.robustness_scale import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--protocol", type=Path, default=Path("reports/evaluation/robustness-scale/protocol.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation/robustness-scale"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    result = run(root, protocol_path=root / args.protocol, output_dir=root / args.output_dir, verify_only=args.verify_only)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
