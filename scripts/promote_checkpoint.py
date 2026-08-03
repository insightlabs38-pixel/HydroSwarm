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


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    args = parser.parse_args()
    source = args.checkpoint / "model.safetensors"
    trainer_state = json.loads(
        (args.checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
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
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
