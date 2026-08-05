"""Promote safe model weights and provenance without optimizer pickle state."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

from safetensors import safe_open

from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.tasks import validate_tasks


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_set(value: str) -> frozenset[str]:
    return frozenset(token.strip() for token in value.split(",") if token.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--corpus-report", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--training-seconds", type=float, required=True)
    parser.add_argument("--status", choices=("trained", "partial"), required=True)
    parser.add_argument(
        "--trained-tasks",
        required=True,
        help=(
            "comma-separated runtime tasks genuinely optimized against real labels "
            "for this checkpoint (see hydroswarm.tasks.RUNTIME_TASKS); e.g. 'sentinel'. "
            "core-issues.txt repair item 8: HybridInferencePipeline ignores any task's "
            "neural outputs that is not declared here."
        ),
    )
    parser.add_argument(
        "--validated-tasks",
        required=True,
        help=(
            "comma-separated subset of --trained-tasks that has also passed "
            "audit/calibration validation (e.g. label_audit, corpus gates)."
        ),
    )
    args = parser.parse_args()
    trained_tasks = _task_set(args.trained_tasks)
    validated_tasks = _task_set(args.validated_tasks)
    validate_tasks(trained_tasks, label="--trained-tasks")
    validate_tasks(validated_tasks, label="--validated-tasks")
    if not validated_tasks <= trained_tasks:
        raise ValueError(
            f"--validated-tasks {sorted(validated_tasks)} must be a subset of "
            f"--trained-tasks {sorted(trained_tasks)}"
        )
    if args.checkpoint.is_dir():
        source = args.checkpoint / "model.safetensors"
        trainer_state = json.loads(
            (args.checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
    else:
        source = args.checkpoint
        summary = json.loads((source.parent / "summary.json").read_text(encoding="utf-8"))
        trainer_state = {
            key: summary[key]
            for key in ("best_epoch", "best_validation_loss", "epochs_completed", "global_steps")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, args.output)
    with safe_open(args.output, framework="pt", device="cpu") as artifact:
        tensor_count = len(artifact.keys())
        parameter_count = sum(artifact.get_tensor(key).numel() for key in artifact.keys())
    metadata = {
        "schema_version": 1,
        "architecture": args.architecture,
        "variant": args.variant,
        "training_status": args.status,
        "sha256": _hash(args.output),
        "bytes": args.output.stat().st_size,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "corpus_report_sha256": _hash(args.corpus_report),
        "calibration_manifest_sha256": _hash(args.calibration_manifest),
        "feature_schema_version": DEFAULT_FEATURE_SCHEMA.version,
        "feature_schema_sha256": DEFAULT_FEATURE_SCHEMA.fingerprint,
        "training_seconds": args.training_seconds,
        "trainer_state": trainer_state,
        "promoted_at": datetime.now(UTC).isoformat(),
        "optimizer_state_included": False,
        "trained_tasks": sorted(trained_tasks),
        "validated_tasks": sorted(validated_tasks),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
