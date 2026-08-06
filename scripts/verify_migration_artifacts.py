"""Verify Cycle B2 migration artifacts (Arm-migration prep and clean-clone check).

Checks, in order, aborting (nonzero exit) on the first failure category found
but reporting every failure within it:

1. shard_manifests   -- every .safetensors shard under a corpus tensors dir
                        matches its manifest.json-recorded sha256; the split
                        set and per-split example counts are identical
                        between tensors/ and tensors-normalized/.
2. checkpoints        -- a provided mapping of {label: (path, expected_sha256)}
                        all match.
3. calibration        -- same, for calibration artifacts.
4. normalization      -- same, for node/edge normalization artifacts.

Used both to gate the migration commit (source repo) and to verify a clean
clone + `git lfs pull` reproduced everything bit-for-bit (Arm migration
check, core-issues.txt-style Phase 3 follow-up).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_dirs(tensors_root: Path) -> dict[str, dict[str, Any]]:
    manifests = {}
    for candidate in sorted(tensors_root.iterdir()):
        manifest_path = candidate / "manifest.json"
        if candidate.is_dir() and manifest_path.exists():
            manifests[candidate.name] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifests


def verify_shard_manifests(corpus_dir: Path) -> list[str]:
    problems: list[str] = []
    raw_root = corpus_dir / "tensors"
    normalized_root = corpus_dir / "tensors-normalized"
    raw_manifests = _split_dirs(raw_root)
    normalized_manifests = _split_dirs(normalized_root)

    if set(raw_manifests) != set(normalized_manifests):
        problems.append(
            f"split set mismatch between tensors/ ({sorted(raw_manifests)}) and "
            f"tensors-normalized/ ({sorted(normalized_manifests)})"
        )

    for root, manifests in ((raw_root, raw_manifests), (normalized_root, normalized_manifests)):
        for split, manifest in manifests.items():
            split_dir = root / split
            for shard in manifest["shards"]:
                shard_path = split_dir / shard["file"]
                if not shard_path.exists():
                    problems.append(f"missing shard: {shard_path}")
                    continue
                actual = _sha256(shard_path)
                if actual != shard["sha256"]:
                    problems.append(
                        f"checksum mismatch: {shard_path} expected={shard['sha256']} actual={actual}"
                    )
            actual_shard_files = sorted(p.name for p in split_dir.glob("*.safetensors"))
            expected_shard_files = sorted(s["file"] for s in manifest["shards"])
            if actual_shard_files != expected_shard_files:
                problems.append(
                    f"shard file set mismatch in {split_dir}: "
                    f"expected {expected_shard_files}, found {actual_shard_files}"
                )

    for split in sorted(set(raw_manifests) & set(normalized_manifests)):
        raw_count = raw_manifests[split]["total_examples"]
        normalized_count = normalized_manifests[split]["total_examples"]
        if raw_count != normalized_count:
            problems.append(
                f"split {split!r}: raw total_examples={raw_count} != "
                f"normalized total_examples={normalized_count}"
            )
    return problems


def verify_file_hashes(mapping: dict[str, tuple[str, str]]) -> list[str]:
    problems: list[str] = []
    for label, (path_str, expected) in mapping.items():
        path = Path(path_str)
        if not path.exists():
            problems.append(f"{label}: missing file {path}")
            continue
        actual = _sha256(path)
        if actual != expected:
            problems.append(f"{label}: {path} expected={expected} actual={actual}")
    return problems


def verify_calibration_artifact_hashes(mapping: dict[str, tuple[str, str]]) -> list[str]:
    """CalibrationArtifact.artifact_hash is a semantic property computed over
    a canonical subset of fields (see hydroswarm.calibration.conformal) --
    NOT sha256 of the saved JSON's bytes, which vary with formatting. Load
    and ask the artifact for its own hash, the same way runtime code does."""

    from hydroswarm.calibration.conformal import SplitConformalCalibrator

    problems: list[str] = []
    for label, (path_str, expected) in mapping.items():
        path = Path(path_str)
        if not path.exists():
            problems.append(f"{label}: missing file {path}")
            continue
        actual = SplitConformalCalibrator.load(path).artifact.artifact_hash
        if actual != expected:
            problems.append(f"{label}: {path} expected={expected} actual={actual}")
    return problems


CHECKPOINT_HASHES: dict[str, tuple[str, str]] = {
    "E1-seed20260810": (
        "models/cycle-b2-candidates/E1-seed20260810/model.safetensors",
        "051cfd94dec4a7ec61e559a1268b66acaada2d6248bda8c976846f9064ef3a23",
    ),
    "E1-seed20260811": (
        "models/cycle-b2-candidates/E1-seed20260811/model.safetensors",
        "4ae71f3b31c3e7d4e10667126aad5343d64dad513aa48c626c4e3fa42a5dd63a",
    ),
    "E0-seed20260810": (
        "models/cycle-b2-candidates/E0-seed20260810/model.safetensors",
        "04ada898f994c8cd54e12d65a7997256d80e5d6fb4c96a003f56e3492ad43580",
    ),
    "E0-seed20260811": (
        "models/cycle-b2-candidates/E0-seed20260811/model.safetensors",
        "c8f6a5e62a09264f653eec90854ca4934581348e05aa3c86cefc65cb5eee65df",
    ),
}

#: hydroswarm.calibration.conformal.CalibrationArtifact.artifact_hash -- a
#: semantic property, NOT sha256 of the saved JSON file's bytes (which
#: varies with formatting). See verify_calibration_artifact_hashes.
CALIBRATION_HASHES: dict[str, tuple[str, str]] = {
    "E1-seed20260810": (
        "models/cycle-b2-candidates/E1-seed20260810/calibration-dynamic-fusion.json",
        "639384e86ce3c6ad30fb73914b08b8aa302337d77feb3472481e00c6d6cf040d",
    ),
    "E0-seed20260811": (
        "models/cycle-b2-candidates/E0-seed20260811/calibration-dynamic-fusion.json",
        "548009981c74a1d1c66c28936c1e66d65eca670638329c737101dad3d22a922f",
    ),
}

CONTROL_HASHES: dict[str, tuple[str, str]] = {
    "no-adapter-seed20260810": (
        "models/cycle-b2-controls/no-adapter-seed20260810/model.safetensors",
        "fe2bd18b6849d680beae3c4274481797d873f752388ca412ee0a9965b4bb0e3b",
    ),
    "no-adapter-seed20260811": (
        "models/cycle-b2-controls/no-adapter-seed20260811/model.safetensors",
        "f9fa30883cfe33c7ac1f272daaed62d9e263f743760fc7717e00e354703f5d2e",
    ),
}

NORMALIZATION_HASHES: dict[str, tuple[str, str]] = {
    "node": (
        "data/learning-v2/cycle-b2/normalization/node-normalization.json",
        "4dcf22a68839a8630e83b0e90f47ac3400b176b576e76d8bee5662221d238691",
    ),
    "edge": (
        "data/learning-v2/cycle-b2/normalization/edge-normalization.json",
        "3e715d707475d81eba90de6609246f51bb0eee8a94c58eab4958f4370fca514d",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/learning-v2/cycle-b2"))
    parser.add_argument("--skip-shards", action="store_true", help="skip the (slow) shard checksum pass")
    args = parser.parse_args(argv)

    results: dict[str, list[str]] = {}
    if not args.skip_shards:
        results["shard_manifests"] = verify_shard_manifests(args.corpus_dir)
    results["checkpoints"] = verify_file_hashes(CHECKPOINT_HASHES)
    results["calibration"] = verify_calibration_artifact_hashes(CALIBRATION_HASHES)
    results["controls"] = verify_file_hashes(CONTROL_HASHES)
    results["normalization"] = verify_file_hashes(NORMALIZATION_HASHES)

    failed = {name: problems for name, problems in results.items() if problems}
    print(json.dumps(results, indent=2, sort_keys=True))
    if failed:
        print(f"\nMIGRATION ARTIFACT VERIFICATION FAILED: {sorted(failed)}", file=sys.stderr)
        return 1
    print("\nall migration artifacts verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
