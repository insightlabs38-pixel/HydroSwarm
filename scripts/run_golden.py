from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation import GoldenScenarioRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen WNTR-backed golden scenario")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "results" / "golden_result.json")
    args = parser.parse_args()
    result = GoldenScenarioRunner(ROOT, seed=args.seed).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Golden scenario: {result['workflow']['approval_pause_state']} -> {result['workflow']['completed_state']}")
    print(f"Result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

