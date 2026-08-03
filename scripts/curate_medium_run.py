"""Publish the budget-complete M candidate without optimizer or resume state."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "metadata.json", "metrics.jsonl", "status.json", "summary.json"):
        shutil.copyfile(args.run / filename, args.output / filename)
    candidate = args.output / "candidate.safetensors"
    shutil.copyfile(args.run / "model-export.safetensors", candidate)

    epochs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in (args.run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        epochs[int(record["epoch"])].append(record)
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    curves = []
    for epoch, records in sorted(epochs.items()):
        if epoch > int(summary["best_epoch"]):
            continue
        tasks = sorted(records[0]["task_losses"])
        curves.append({
            "epoch": epoch,
            "batches": len(records),
            "mean_training_loss": sum(item["loss"] for item in records) / len(records),
            "mean_task_losses": {
                task: sum(item["task_losses"][task] for item in records) / len(records)
                for task in tasks
            },
            "final_learning_rate": records[-1]["learning_rate"],
        })
    curve_path = args.output / "training-curves.json"
    curve_path.write_text(json.dumps(curves, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    registry = {
        "schema_version": 1,
        "training_status": "budget_complete_not_promoted",
        "stop_reason": summary["stop_reason"],
        "epochs_completed": summary["epochs_completed"],
        "global_steps": summary["global_steps"],
        "best_epoch": summary["best_epoch"],
        "best_validation_loss": summary["best_validation_loss"],
        "dataset_manifest_sha256": metadata["dataset_manifest_hash"],
        "candidate_sha256": sha256(candidate),
        "candidate_bytes": candidate.stat().st_size,
        "training_curve_sha256": sha256(curve_path),
        "metrics_sha256": sha256(args.output / "metrics.jsonl"),
        "optimizer_state_published": False,
        "runtime_default": False,
        "promotion_decision": "rejected_by_locked_operational_gate",
    }
    (args.output / "registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
