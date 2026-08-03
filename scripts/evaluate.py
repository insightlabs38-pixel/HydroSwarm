from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation import BenchmarkRunner  # noqa: E402


def _summary(result: dict) -> str:
    aggregate = result["aggregate"]
    gate = result["promotion_gate"]
    lines = [
        "# HydroSwarm measured evaluation",
        "",
        f"Promotion gate: **{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "All values below were produced by the frozen WNTR fixture. Missing neural checkpoints are reported as not run.",
        "",
        "| Metric | Mean | 95% normal CI |",
        "| --- | ---: | ---: |",
    ]
    for name, values in aggregate.items():
        lines.append(
            f"| {name} | {values['mean']:.6g} | [{values['ci95_normal_low']:.6g}, {values['ci95_normal_high']:.6g}] |"
        )
    lines.extend(["", "## Gate checks", ""])
    for name, passed in gate["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    lines.extend(
        [
            "",
            "## Measurement limitations",
            "",
            f"- RAM: {result['measurement_notes']['ram']}",
            f"- Confidence intervals: {result['measurement_notes']['confidence_intervals']}",
            "- Small, medium, and large neural variants were not run because no trained checkpoint was supplied.",
            "- This fixture is a regression benchmark, not evidence of field performance.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated-seed HydroSwarm evaluation")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "evaluation.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "results" / "evaluation_results.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports" / "results" / "summary.md")
    args = parser.parse_args()
    result = BenchmarkRunner(ROOT, args.config).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary.write_text(_summary(result), encoding="utf-8")
    print(f"Promotion gate: {'PASS' if result['promotion_gate']['passed'] else 'FAIL'}")
    print(f"Results: {args.output}")
    return 0 if result["promotion_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

