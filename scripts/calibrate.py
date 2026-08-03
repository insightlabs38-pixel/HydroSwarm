"""Fit a checksummed split/Mondrian conformal artifact from prediction JSONL."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/calibration/conformal.json"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--feature-schema-hash", required=True)
    args = parser.parse_args()
    raw = args.predictions.read_bytes()
    examples = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            examples.append(CalibrationExample(
                probabilities=tuple(float(value) for value in item["probabilities"]),
                true_index=int(item["true_index"]),
                condition=str(item["condition"]),
                network_id=str(item["network_id"]),
            ))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid prediction on line {number}: {exc}") from exc
    calibrator = SplitConformalCalibrator.fit(
        examples,
        alpha=args.alpha,
        model_hash=args.model_hash,
        feature_schema_hash=args.feature_schema_hash,
        dataset_manifest_hash=hashlib.sha256(raw).hexdigest(),
    )
    calibrator.save(args.output)
    print(json.dumps({
        "artifact": str(args.output),
        "artifact_sha256": calibrator.artifact.artifact_hash,
        "report": asdict(calibrator.artifact.report),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
