"""core-issues4.txt Section F: stream SecondPassControlLabel rows to a
versioned, checksummed JSONL artifact usable as real training data.

scripts/run_second_pass_control_labels.py (core-issues3.txt Phase 8 steps
4/9) already runs second-pass label generation against a real checkpoint,
but only ever materialized the full list of labels in memory to compute
aggregate summary statistics -- it never persisted the per-example rows
themselves, so its output could not be used to train anything. This script
is the streaming, per-row counterpart: it writes ONE JSONL row at a time as
hydroswarm.training.second_pass_control_labels.generate_second_pass_control_labels
yields each label, without ever holding the whole split resident (Section
F's explicit "do not materialize the entire 9,000-example split in memory
merely to write the file").

Usage (mirrors run_second_pass_control_labels.py's arguments):

    python scripts/persist_second_pass_control_labels.py \
        --checkpoint experiments/runs/v4-stage-a-sentinel/E1-seed20260810/20260807T020714Z-12fe7f02/checkpoints/checkpoint-0016/model.safetensors \
        --calibration experiments/runs/v4-stage-a-sentinel/E1-seed20260810/calibration.json \
        --corpus-root data/learning-v2/cycle-b2 --tensors-dirname tensors-normalized \
        --split train --prior-mode feature_only \
        --output-dir data/learning-v2/cycle-b2-control-v2/second-pass-labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from safetensors.torch import load_file

from hydroswarm.calibration.conformal import SplitConformalCalibrator
from hydroswarm.model import HydroCore
from hydroswarm.training import ShardedScenarioDataset
from hydroswarm.training.second_pass_control_labels import (
    SECOND_PASS_CONTROL_POLICY_VERSION,
    generate_second_pass_control_labels,
    second_pass_control_policy_hash,
)


def _row_dict(label, *, split: str) -> dict:
    row = asdict(label)
    row["next_step"] = label.next_step.value
    row["source_split"] = split
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, default=Path("data/learning-v2/cycle-b2"))
    parser.add_argument("--tensors-dirname", default="tensors-normalized")
    parser.add_argument("--split", required=True, choices=("train", "validation"))
    parser.add_argument("--variant", default="small")
    parser.add_argument("--prior-mode", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    overrides = {"prior_mode": args.prior_mode} if args.prior_mode else {}
    model = HydroCore.from_variant(args.variant, **overrides)
    model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=True)
    model.eval()

    calibrator = SplitConformalCalibrator.load(args.calibration)
    teacher_checkpoint_hash = calibrator.artifact.model_hash
    validated_topology_hashes = frozenset(calibrator.artifact.validated_topology_hashes)

    dataset = ShardedScenarioDataset(
        args.corpus_root / args.tensors_dirname / args.split, expected_split=args.split
    )
    dataset.verify_shard_checksums()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.split}.jsonl"
    hasher = hashlib.sha256()
    row_count = 0
    next_step_counts: dict[str, int] = {}
    # Streamed: exactly one label materialized at a time via the
    # generator, written and discarded immediately -- never a `list(...)`
    # of the whole split.
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for label in generate_second_pass_control_labels(
            model, dataset, calibrator,
            teacher_checkpoint_hash=teacher_checkpoint_hash,
            validated_topology_hashes=validated_topology_hashes,
            batch_size=args.batch_size,
        ):
            line = json.dumps(_row_dict(label, split=args.split), sort_keys=True) + "\n"
            stream.write(line)
            hasher.update(line.encode("utf-8"))
            row_count += 1
            next_step_counts[label.next_step.value] = next_step_counts.get(label.next_step.value, 0) + 1

    manifest = {
        "schema_version": 1,
        "split": args.split,
        "row_count": row_count,
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": hasher.hexdigest(),
        "checkpoint": str(args.checkpoint),
        "calibration": str(args.calibration),
        "teacher_checkpoint_hash": teacher_checkpoint_hash,
        "calibration_hash": calibrator.artifact.artifact_hash,
        "control_policy_version": SECOND_PASS_CONTROL_POLICY_VERSION,
        "control_policy_hash": second_pass_control_policy_hash(),
        "corpus_root": str(args.corpus_root),
        "tensors_dirname": args.tensors_dirname,
        "next_step_distribution": next_step_counts,
    }
    manifest_path = args.output_dir / f"{args.split}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
