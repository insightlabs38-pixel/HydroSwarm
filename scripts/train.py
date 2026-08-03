"""Run governed HydroCore training from JSONL tensor examples."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

from hydroswarm.model import HydroCore
from hydroswarm.training import (
    CurriculumStage,
    GovernedScenarioDataset,
    ScenarioExample,
    Trainer,
    TrainingConfig,
)


def load_dataset(path: Path, *, split: str) -> GovernedScenarioDataset:
    examples = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                examples.append(
                    ScenarioExample(
                        scenario_id=record["scenario_id"],
                        network_id=record["network_id"],
                        split=record["split"],
                        seed=int(record["seed"]),
                        seed_family=record["seed_family"],
                        stage=CurriculumStage[record["stage"]],
                        inputs={key: torch.tensor(value) for key, value in record["inputs"].items()},
                        targets={key: torch.tensor(value) for key, value in record["targets"].items()},
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid manifest line {line_number}: {error}") from error
    return GovernedScenarioDataset(examples, expected_split=split)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--variant", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--resume-from", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainingConfig.from_yaml(args.config)
    train = load_dataset(args.train_manifest, split="train")
    validation = (
        load_dataset(args.validation_manifest, split="validation")
        if args.validation_manifest
        else None
    )
    model = HydroCore.from_variant(args.variant)
    trainer = Trainer(
        model,
        train,
        validation_dataset=validation,
        config=config,
        run_root=args.run_root,
        workdir=Path.cwd(),
    )
    summary = trainer.fit(resume_from=args.resume_from)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
