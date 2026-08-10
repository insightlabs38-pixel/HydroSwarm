"""SUB-3 (submission.txt SS21): produce the source/runtime release archive
`HydroSwarm-v{version}-hackathon-runtime.zip`, in addition to the Docker
release path.

Contents (per SS21): source/, frontend/dist/, models/hydrocore-v4-release/,
configs/, examples/ (if present), artifacts/reference-demo/ (if present),
the setup/start scripts, SHA256SUMS, RELEASE_MANIFEST.json, LICENSE,
README.md.

Deliberately excludes training corpus shards, model checkpoints outside the
one frozen release bundle, and any experiment scratch data -- this is a
judge-facing runtime artifact, not a research mirror of the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SETUP_SCRIPTS = [
    "setup_hydroswarm_linux.sh",
    "setup_hydroswarm_macos.sh",
    "setup_hydroswarm_windows.ps1",
    "start_hydroswarm_linux.sh",
    "start_hydroswarm_macos.sh",
    "start_hydroswarm_windows.ps1",
    "start_hydroswarm.sh",
    "start_hydroswarm.bat",
]

TOP_LEVEL_FILES = [
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "RELEASE_MANIFEST.json",
]

INCLUDED_DIRS = [
    "src",
    "frontend/dist",
    "models/hydrocore-v4-release",
    "configs",
    "examples",
    "artifacts/reference-demo",
]


def _iter_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    ]


def build_bundle(output_path: Path, *, release_version: str) -> Path:
    entries: list[tuple[Path, str]] = []

    for relative in TOP_LEVEL_FILES:
        source = PROJECT_ROOT / relative
        if source.is_file():
            entries.append((source, relative))

    for relative in SETUP_SCRIPTS:
        source = PROJECT_ROOT / relative
        if source.is_file():
            entries.append((source, relative))

    for relative_dir in INCLUDED_DIRS:
        source_dir = PROJECT_ROOT / relative_dir
        for file_path in _iter_files(source_dir):
            entries.append((file_path, str(file_path.relative_to(PROJECT_ROOT))))

    if not entries:
        raise RuntimeError("no files resolved for the release bundle -- refusing to write an empty archive")

    checksum_lines = []
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, arcname in entries:
            archive.write(source, arcname)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            checksum_lines.append(f"{digest}  {arcname}")

        sha256sums = "\n".join(sorted(checksum_lines)) + "\n"
        archive.writestr("SHA256SUMS", sha256sums)

    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-version", required=True, help='e.g. "v0.1.0-hackathon"')
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "dist")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / f"HydroSwarm-{args.release_version}-runtime.zip"
    build_bundle(output_path, release_version=args.release_version)
    print(f"wrote {output_path} ({output_path.stat().st_size / (1024 * 1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
