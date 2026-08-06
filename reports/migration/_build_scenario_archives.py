"""One-off migration helper: deterministic tar+zstd archives of Cycle B2's
raw per-scenario .npz/.parquet arrays, one archive per split. OOD scenarios
live physically inside scenarios/development_holdout/ (see
generate_cycle_b_corpus.py) so they are included in that split's archive,
not separately."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/workspace/HydroSwarm")
SCENARIOS_ROOT = ROOT / "data/learning-v2/cycle-b2/scenarios"
OUTPUT_ROOT = ROOT / "artifacts/migration"
SPLITS = ["train", "validation", "calibration", "development_holdout"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive(split: str) -> dict:
    split_dir = SCENARIOS_ROOT / split
    files = sorted(p for p in split_dir.iterdir() if p.is_file())
    uncompressed_bytes = sum(p.stat().st_size for p in files)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tar_path = OUTPUT_ROOT / f"cycle-b2-scenarios-{split.replace('_', '-')}.tar"
    archive_path = OUTPUT_ROOT / f"cycle-b2-scenarios-{split.replace('_', '-')}.tar.zst"

    # Deterministic tar: fixed mtime/owner/group/mode, sorted member order,
    # archive member paths relative to the split (e.g. "train/<id>.npz") so
    # extraction at the repo root reproduces the exact original layout.
    with tarfile.open(tar_path, "w") as tar:
        for path in files:
            info = tar.gettarinfo(str(path), arcname=f"{split}/{path.name}")
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            with path.open("rb") as fh:
                tar.addfile(info, fh)

    subprocess.run(
        ["zstd", "-19", "--force", "-o", str(archive_path), str(tar_path)],
        check=True, capture_output=True,
    )
    tar_path.unlink()

    return {
        "split": split,
        "archive": str(archive_path.relative_to(ROOT)),
        "archive_sha256": sha256(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "file_count": len(files),
        "uncompressed_bytes": uncompressed_bytes,
        "extraction_command": f"tar --use-compress-program=unzstd -xf {archive_path.relative_to(ROOT)} -C data/learning-v2/cycle-b2/scenarios",
        "expected_destination": f"data/learning-v2/cycle-b2/scenarios/{split}/",
        "includes_ood_scenarios": split == "development_holdout",
    }


def main() -> None:
    manifest_entries = [build_archive(split) for split in SPLITS]
    manifest = {
        "schema_version": 1,
        "source_git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "archives": manifest_entries,
    }
    manifest_path = OUTPUT_ROOT / "cycle-b2-scenarios-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
