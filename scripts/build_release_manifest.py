"""SUB-3 (submission.txt SS22): produce `RELEASE_MANIFEST.json`.

Sources every scientific/build hash from the real, already-generated
artifacts it describes -- the frozen V4 bundle's own `runtime_manifest.json`
and the architecture-freeze declaration -- rather than typing any hash by
hand, per submission.txt's explicit "Do not manually type scientific hashes
where generation can source them from the real manifests" instruction.

Container image/digest fields are optional inputs: this script runs both
locally (no image yet, before a release build) and in `release.yml` after
`docker buildx build --push` produces a real digest to pass in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = PROJECT_ROOT / "models" / "hydrocore-v4-release"
ARCHITECTURE_FREEZE_PATH = PROJECT_ROOT / "reports" / "results" / "v4" / "architecture-freeze.json"
REFERENCE_DEMO_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "reference-demo" / "manifest.json"


def _git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _pyproject_version() -> str:
    import re

    text = (PROJECT_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("could not read project version from pyproject.toml")
    return match.group(1)


def _sha256_dir(directory: Path) -> str:
    """A stable content hash over every file in `directory`, used for the
    frontend build hash (no single canonical frontend/dist file exists)."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_manifest(
    *,
    release_version: str | None = None,
    container_image: str | None = None,
    container_digest: str | None = None,
    supported_platforms: list[str] | None = None,
) -> dict[str, Any]:
    runtime_manifest = json.loads((BUNDLE_DIR / "runtime_manifest.json").read_text())
    architecture_freeze = (
        json.loads(ARCHITECTURE_FREEZE_PATH.read_text()) if ARCHITECTURE_FREEZE_PATH.is_file() else {}
    )

    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    frontend_build_hash = _sha256_dir(frontend_dist) if frontend_dist.is_dir() else None

    reference_demo_hash = None
    if REFERENCE_DEMO_MANIFEST_PATH.is_file():
        reference_demo_manifest = json.loads(REFERENCE_DEMO_MANIFEST_PATH.read_text())
        reference_demo_hash = reference_demo_manifest.get("artifact_sha256")

    node_version_result = subprocess.run(["node", "--version"], capture_output=True, text=True, check=False)
    node_version = node_version_result.stdout.strip().lstrip("v") if node_version_result.returncode == 0 else None

    return {
        "release_version": release_version or _pyproject_version(),
        "git_commit": _git_commit_hash(),
        "created_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "node_version_used_for_build": node_version,
        "frontend_build_hash": frontend_build_hash,
        "model_hash": runtime_manifest.get("model_sha256"),
        "calibration_hash": runtime_manifest.get("calibration_artifact_hash"),
        "normalization_hash": runtime_manifest.get("normalization_hash"),
        "feature_schema_hash": runtime_manifest.get("feature_schema_hash"),
        "signature_policy_hash": runtime_manifest.get("signature_policy_hash"),
        "reference_demo_hash": reference_demo_hash,
        "container_image": container_image,
        "container_digest": container_digest,
        "supported_platforms": supported_platforms or ["linux/amd64", "linux/arm64"],
        "locked_test_status": {
            "locked_test_opened": architecture_freeze.get("locked_test_opened"),
            "locked_evaluation_status": architecture_freeze.get("locked_evaluation_status"),
        },
        "schema_version": 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-version", default=None)
    parser.add_argument("--container-image", default=None)
    parser.add_argument("--container-digest", default=None)
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "RELEASE_MANIFEST.json", help="Output path for the manifest."
    )
    args = parser.parse_args(argv)

    manifest = build_manifest(
        release_version=args.release_version,
        container_image=args.container_image,
        container_digest=args.container_digest,
    )
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
