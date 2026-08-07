"""core-issues3.txt Phase 11.2: report real per-split class prevalence for
the imbalanced classification tasks this pass has real committed corpus
data for, and derive train-owned class weights from each task's TRAIN
split only.

Deliberately scoped to what real, already-committed corpora actually
contain -- not every governed classification task has real label data yet
(e.g. ood_class only has real non-NONE examples in
data/learning-v2/cycle-b2-ood-extension, whose shard `split` metadata does
not line up with a single conventional split name the way cycle-b2's does;
left for a follow-up rather than forcing a mismatched load path here):

- event_cause: data/learning-v2/cycle-b2 (train/validation/calibration/
  development_holdout)
- next_step: data/learning-v2/cycle-b2-control-v2 (train/validation)
- plan_validity: data/learning-v2/cycle-b2-trajectories-v3/
  strategist-tensors-normalized (train/validation)

Usage:
    PYTHONPATH=src python scripts/report_class_prevalence.py \
        --output reports/results/v4/class-prevalence.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

from hydroswarm.training import ShardedScenarioDataset
from hydroswarm.training.class_balance import (
    CLASS_WEIGHT_POLICY_VERSION,
    class_prevalence,
    merge_prevalence,
    train_owned_class_weights,
)


class TaskSource(NamedTuple):
    task: str
    corpus_root: Path
    splits: tuple[str, ...]


TASK_SOURCES: tuple[TaskSource, ...] = (
    TaskSource(
        "event_cause",
        Path("data/learning-v2/cycle-b2/tensors-normalized"),
        ("train", "validation", "calibration", "development_holdout"),
    ),
    TaskSource(
        "next_step",
        Path("data/learning-v2/cycle-b2-control-v2/tensors-normalized"),
        ("train", "validation"),
    ),
    TaskSource(
        "plan_validity",
        Path("data/learning-v2/cycle-b2-trajectories-v3/strategist-tensors-normalized"),
        ("train", "validation"),
    ),
)


def _split_prevalence(corpus_root: Path, split: str, task: str) -> dict[int, int] | None:
    split_dir = corpus_root / split
    if not split_dir.exists():
        return None
    dataset = ShardedScenarioDataset(split_dir, expected_split=split)
    per_example = []
    for index in range(len(dataset)):
        example = dataset[index]
        per_example.append(
            class_prevalence(example.targets[task], mask=example.targets.get(f"{task}_mask"))
        )
    return merge_prevalence(*per_example) if per_example else {}


def build_report() -> dict[str, object]:
    report: dict[str, object] = {"class_weight_policy_version": CLASS_WEIGHT_POLICY_VERSION, "tasks": {}}
    for source in TASK_SOURCES:
        by_split: dict[str, dict[str, int]] = {}
        for split in source.splits:
            prevalence = _split_prevalence(source.corpus_root, split, source.task)
            if prevalence is None:
                continue
            by_split[split] = {str(key): value for key, value in prevalence.items()}
        if "train" not in by_split:
            continue  # train-owned weights require a real train split
        train_prevalence = {int(key): value for key, value in by_split["train"].items()}
        weights = train_owned_class_weights(train_prevalence)
        report["tasks"][source.task] = {  # type: ignore[index]
            "corpus_root": str(source.corpus_root),
            "prevalence_by_split": by_split,
            "train_owned_class_weights": {str(key): value for key, value in weights.items()},
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/class-prevalence.json"))
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for task, detail in report["tasks"].items():  # type: ignore[union-attr]
        print(f"{task}: {detail['prevalence_by_split']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
