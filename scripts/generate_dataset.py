"""Generate governed WNTR scenarios for the canonical reference network."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hydroswarm.data.scenarios import (
    CurriculumStage,
    ScenarioDatasetWriter,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.simulation.network import build_wntr_network


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stage", choices=[item.value for item in CurriculumStage], default="operational")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    generator = WNTRScenarioGenerator()
    writer = ScenarioDatasetWriter(args.output)
    artifacts: list[dict[str, object]] = []
    for offset in range(args.count):
        seed = args.seed + offset
        scenario = generator.generate(
            build_wntr_network(),
            ScenarioGenerationConfig(
                seed=seed,
                network_id="hydroswarm-reference",
                network_family="HydroSwarm-Reference",
                stage=CurriculumStage(args.stage),
            ),
        )
        path = writer.write(scenario)
        artifacts.append({
            "scenario_id": str(scenario.manifest.scenario_id),
            "split": scenario.manifest.split.value,
            "seed": seed,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    summary = {"count": len(artifacts), "output": str(args.output), "artifacts": artifacts}
    summary_path = args.output / "generation-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
