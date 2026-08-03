"""Finalize a safely preserved best-model snapshot after a legacy timeout exit."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

from safetensors import safe_open


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    best = args.run / "best-model.safetensors"
    if not best.is_file():
        raise SystemExit("best-model.safetensors is missing")
    epoch_summary = json.loads((args.run / "epoch_summary.json").read_text(encoding="utf-8"))
    with safe_open(best, framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata() or {}
    best_epoch = int(metadata["epoch"])
    best_loss = float(metadata["validation_loss"])
    if best_epoch != int(epoch_summary["epoch"]):
        raise SystemExit("latest complete epoch is not the preserved best; manual audit required")
    shutil.copyfile(best, args.run / "model-export.safetensors")
    metrics = [
        json.loads(line)
        for line in (args.run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    best_metrics = [item for item in metrics if int(item["epoch"]) == best_epoch]
    if not best_metrics:
        raise SystemExit("best epoch has no metrics")
    summary = {
        "run_directory": str(args.run),
        "epochs_completed": best_epoch + 1,
        "global_steps": int(best_metrics[-1]["global_step"]),
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "stopped_early": False,
        "stop_reason": "runtime_budget",
        "final_checkpoint": "",
        "export_path": str(args.run / "model-export.safetensors"),
    }
    (args.run / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = {
        "state": "COMPLETED",
        "stop_reason": "runtime_budget",
        "updated_at": datetime.now(UTC).isoformat(),
        "recovered_from_legacy_timeout": True,
    }
    (args.run / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
