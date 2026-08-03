"""Publish concise experiment curves and provenance without optimizer state."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


RUNS = {
    "hydrocore_s": ("experiments/runs/learning-v4-scaled", "trained"),
    "hydromono_s": ("experiments/runs/learning-v4-hydromono", "trained"),
    "hydrocore_m_partial": ("experiments/runs/learning-v4-medium-partial", "partial"),
}


def _latest(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise ValueError(f"no experiment runs beneath {root}")
    return candidates[-1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "experiments" / "learning-v1"
    output.mkdir(parents=True, exist_ok=True)
    registry: dict[str, Any] = {"schema_version": 1, "experiments": {}}
    for name, (relative, status) in RUNS.items():
        run = _latest(root / relative)
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "metadata.json", "metrics.jsonl", "status.json", "summary.json"):
            shutil.copyfile(run / filename, destination / filename)
        epochs: dict[int, list[dict[str, Any]]] = defaultdict(list)
        with (run / "metrics.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                epochs[int(record["epoch"])].append(record)
        curves = []
        for epoch, records in sorted(epochs.items()):
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
        curve_path = destination / "training-curves.json"
        curve_path.write_text(json.dumps(curves, indent=2, sort_keys=True), encoding="utf-8")
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        registry["experiments"][name] = {
            "training_status": status,
            "epochs_completed": summary["epochs_completed"],
            "global_steps": summary["global_steps"],
            "best_validation_loss": summary["best_validation_loss"],
            "dataset_manifest_sha256": metadata["dataset_manifest_hash"],
            "training_curve_sha256": _hash(curve_path),
            "metrics_sha256": _hash(destination / "metrics.jsonl"),
            "resume_supported": True,
            "optimizer_state_published": False,
            "limitations": (
                ["Only two curriculum epochs; not a converged medium-model result."]
                if status == "partial"
                else []
            ),
        }
    registry_path = output / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
