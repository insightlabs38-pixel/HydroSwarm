"""SUB-3 (submission.txt SS21): produce the source/runtime release archive
`HydroSwarm-{version}-runtime.zip` (e.g. `HydroSwarm-v0.2.0-runtime.zip`),
in addition to the Docker release path.

Contents (per SS21): source/, frontend/dist/, models/hydrocore-v5-release/,
configs/, examples/ (if present), artifacts/reference-demo/ (if present),
the setup/start scripts, SHA256SUMS, RELEASE_MANIFEST.json, LICENSE,
README.md.

Deliberately excludes training corpus shards, model checkpoints outside the
one frozen release bundle, and any experiment scratch data -- this is a
judge-facing runtime artifact, not a research mirror of the repository. The
historical `models/hydrocore-v4-release/` bundle is intentionally excluded
too: no current runtime path (serving app, setup verification, self-test,
Docker image) depends on it, so the frozen no-V4-fallback release identity
never ships it as a live runtime asset -- it remains in the git history/
checkout as historical evidence, not in the judge-facing release archive.
"""

from __future__ import annotations

import argparse
import hashlib
import re
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

#: SUB-12.1 P0 fix: every setup script above invokes
#: `scripts/setup_common.py` (check-python / verify-bundle /
#: frontend-status / self-test subcommands) -- a runtime dependency of the
#: setup path itself, not just a repo-dev convenience script. Omitting it
#: from the release zip broke every included setup script the moment a
#: judge actually ran one from the extracted archive. Audited against
#: every `scripts/*` reference in SETUP_SCRIPTS' own files (see
#: tests/unit/test_release_packaging.py's audit test, which fails loudly
#: if a setup script starts referencing a scripts/*.py or scripts/*.sh
#: file not listed here) -- this is the complete set, not a guess.
#:
#: `scripts/build_epanet_arm64.sh`: setup_hydroswarm_linux.sh now invokes
#: this on linux-arm64 to build the architecture-native EPANET
#: water-quality library (wntr ships no linux-arm64 EPANET binary
#: upstream; see docs/ARM_MIGRATION.md) -- without it in the release zip,
#: a judge extracting the archive on native Linux ARM64 would hit the same
#: "file not found" setup_common.py used to hit before that P0 fix.
REQUIRED_HELPER_SCRIPTS = [
    "scripts/setup_common.py",
    "scripts/build_epanet_arm64.sh",
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
    "models/hydrocore-v5-release",
    "configs",
    "examples",
    "artifacts/reference-demo",
    # SUB-12.1 P1 #4: the frozen golden network/scenario fixture the LIVE
    # example's real reference inputs (GET /api/live-example-inputs) need
    # -- without it, the extracted release zip's LIVE example judge path
    # 404s even though the REFERENCE INCIDENT path works fine.
    "data/frozen",
    # v0.2.0 native linux-arm64 EPANET fix: scripts/build_epanet_arm64.sh's
    # own real-simulation self-verification (run automatically by
    # setup_hydroswarm_linux.sh on arm64) needs this exact fixture --
    # without it, ARM64 setup from an extracted release zip would fail
    # even though the actual EPANET library build succeeded. The same
    # directory the Dockerfile already COPYs in for the live pipeline's
    # classical signature priors (see Dockerfile's own comment).
    "data/topologies",
]


#: Matches a `scripts/<name>.py` / `scripts/<name>.sh` or their `\`-separated
#: equivalent inside a setup/start script's own source text (bash and
#: PowerShell both write it this way -- see e.g. setup_hydroswarm_windows.ps1's
#: `"$ProjectRoot\scripts\setup_common.py"` and setup_hydroswarm_linux.sh's
#: `"$PROJECT_ROOT/scripts/build_epanet_arm64.sh"`).
_SCRIPT_REFERENCE = re.compile(r"scripts[/\\]([A-Za-z0-9_]+\.(?:py|sh))")


def referenced_helper_scripts() -> set[str]:
    """Every `scripts/*.py`/`scripts/*.sh` file the committed setup/start
    scripts actually reference, found by scanning their own source text --
    not a hand-maintained guess. `REQUIRED_HELPER_SCRIPTS` must be a
    superset of this (checked in `build_bundle` and in
    tests/unit/test_release_packaging.py) so a future setup script that
    starts calling a new helper script cannot silently ship a release zip
    missing it, the same way `scripts/setup_common.py` was missing before
    this fix."""
    found: set[str] = set()
    for relative in SETUP_SCRIPTS:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8")
        for match in _SCRIPT_REFERENCE.finditer(text):
            found.add(f"scripts/{match.group(1)}")
    return found


def _iter_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        # release.yml's release-artifacts job runs `pip install -e .`
        # (needed to build the frontend/run the manifest generator) before
        # calling this function -- an editable setuptools install creates
        # `src/hydroswarm.egg-info/` (SOURCES.txt, PKG-INFO, etc.), a local
        # build-scratch artifact, not a runtime source file. Without this
        # exclusion it would silently ship inside every real release ZIP.
        and not any(part.endswith((".egg-info", ".dist-info")) for part in path.parts)
        and path.suffix not in {".pyc", ".pyo"}
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

    missing_from_manifest = referenced_helper_scripts() - set(REQUIRED_HELPER_SCRIPTS)
    if missing_from_manifest:
        raise RuntimeError(
            "setup/start scripts reference helper script(s) not listed in "
            f"REQUIRED_HELPER_SCRIPTS: {sorted(missing_from_manifest)} -- add them there "
            "before building the release bundle, or it will ship a broken setup path"
        )

    for relative in REQUIRED_HELPER_SCRIPTS:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            raise RuntimeError(
                f"required helper script missing from the repository: {relative} -- "
                "the release bundle would ship a setup script that cannot run"
            )
        entries.append((source, relative))

    for relative_dir in INCLUDED_DIRS:
        source_dir = PROJECT_ROOT / relative_dir
        for file_path in _iter_files(source_dir):
            # .as_posix(), not str(): on Windows, str(WindowsPath) uses
            # backslashes, which would both violate the ZIP spec's
            # mandatory forward-slash archive-member names and produce a
            # SHA256SUMS file whose paths don't match the real
            # (forward-slash) paths `sha256sum --check` sees after
            # extraction -- a real Windows CI run caught this exact bug
            # ("FAILED open or read" for every entry).
            entries.append((file_path, file_path.relative_to(PROJECT_ROOT).as_posix()))

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
    parser.add_argument("--release-version", required=True, help='e.g. "v0.2.0"')
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "dist")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / f"HydroSwarm-{args.release_version}-runtime.zip"
    build_bundle(output_path, release_version=args.release_version)
    print(f"wrote {output_path} ({output_path.stat().st_size / (1024 * 1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
