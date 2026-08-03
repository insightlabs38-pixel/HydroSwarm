"""Freeze test artifacts and audit profile supervision without reading test labels."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PROFILE_CLASSES = {"start_time": 4, "duration": 3, "relative_strength": 3}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_split(path: Path) -> dict[str, object]:
    counts = {name: Counter() for name in PROFILE_CLASSES}
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows += 1
        for name, class_count in PROFILE_CLASSES.items():
            label = int(record["targets"][name])
            if not 0 <= label < class_count:
                raise ValueError(f"{path}: invalid {name} label {label}")
            counts[name][label] += 1
    return {
        "rows": rows,
        "class_counts": {
            name: {str(index): values[index] for index in range(PROFILE_CLASSES[name])}
            for name, values in counts.items()
        },
        "imbalance_ratio": {
            name: max(values.values()) / min(values.values()) for name, values in counts.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-directory", type=Path, required=True)
    parser.add_argument("--scenario-test-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tensor_test = args.tensor_directory / "test.jsonl"
    report = {
        "schema_version": 1,
        "decision_data": ["train", "validation"],
        "test_labels_inspected": False,
        "test_split_lock": {
            "scenario_manifest_sha256": sha256(args.scenario_test_manifest),
            "tensor_manifest_sha256": sha256(tensor_test),
        },
        "profile_label_space": PROFILE_CLASSES,
        "audits": {
            split: audit_split(args.tensor_directory / f"{split}.jsonl")
            for split in ("train", "validation")
        },
        "training_budget": {
            "maximum_epochs": 30,
            "maximum_runtime_seconds": 2400,
            "early_stopping_patience": 6,
            "minimum_delta": 0.001,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
