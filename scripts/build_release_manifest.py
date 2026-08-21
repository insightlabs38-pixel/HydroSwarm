"""SUB-3 (submission.txt SS22): produce `RELEASE_MANIFEST.json`.

Sources every scientific/build hash from the real, already-generated
artifacts it describes -- the frozen V5 bundle's own `runtime_manifest.json`,
the M11.2 finalist-identity record, and the M11.6 locked-evaluation current-
status index -- rather than typing any hash by hand, per submission.txt's
explicit "Do not manually type scientific hashes where generation can source
them from the real manifests" instruction.

Schema v2 (this rebase) replaces the superseded v1 schema, which sourced its
model/calibration hashes from the historical `models/hydrocore-v4-release/
runtime_manifest.json` and its locked-evaluation status from the pre-M11.6
`reports/results/v4/architecture-freeze.json` declaration -- both stale now
that HydroCore-v5 is the frozen release and M11.6 has actually executed and
passed. v2 also drops `normalization_hash`/`signature_policy_hash`: those are
V4-runtime-specific concepts with no V5 equivalent (see
`models/hydrocore-v5-release/runtime_manifest.json`, which carries no such
fields), so v1's fields for them were always `null` for a V5 release and are
removed rather than kept as dead placeholders.

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
BUNDLE_DIR = PROJECT_ROOT / "models" / "hydrocore-v5-release"
FINALIST_IDENTITY_PATH = PROJECT_ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m11" / "m11-2" / "m11-2-finalist-identity.json"
M11_CURRENT_STATUS_PATH = PROJECT_ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m11" / "m11-current-status.json"
M11_6_CLOSURE_PATH = PROJECT_ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m11" / "m11-6-final" / "m11-6-closure.json"
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    *,
    release_version: str | None = None,
    container_image: str | None = None,
    container_digest: str | None = None,
    supported_platforms: list[str] | None = None,
) -> dict[str, Any]:
    runtime_manifest = json.loads((BUNDLE_DIR / "runtime_manifest.json").read_text())
    finalist_identity = (
        json.loads(FINALIST_IDENTITY_PATH.read_text()) if FINALIST_IDENTITY_PATH.is_file() else {}
    )
    # Sourced from the real M11.6 evidence trail, not the superseded V4
    # architecture-freeze declaration: reports/evaluation/hydrocore-v5/m11/
    # m11-current-status.json is the running index M11.1-M11.6a already
    # maintain, and m11-6-final/m11-6-closure.json is the terminal,
    # authorization-consumed closure record M11.6 itself produced. Neither
    # is regenerated or reinterpreted here -- their fields are copied as-is,
    # truthfully reporting PASS now that the locked evaluation has actually
    # executed exactly once.
    m11_status = json.loads(M11_CURRENT_STATUS_PATH.read_text()) if M11_CURRENT_STATUS_PATH.is_file() else {}
    m11_6_closure = json.loads(M11_6_CLOSURE_PATH.read_text()) if M11_6_CLOSURE_PATH.is_file() else {}

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
        "model_system": finalist_identity.get("system"),
        "model_variant": finalist_identity.get("model_variant"),
        "model_parameter_count": finalist_identity.get("parameter_count"),
        "model_selected_seed": runtime_manifest.get("selected_seed"),
        "model_hash": runtime_manifest.get("model_sha256"),
        "calibration_file_sha256": runtime_manifest.get("calibration_file_sha256"),
        "calibration_artifact_hash": runtime_manifest.get("calibration_artifact_hash"),
        "feature_schema_hash": runtime_manifest.get("feature_schema_hash"),
        "fusion_config_hash": runtime_manifest.get("fusion_config_hash"),
        "runtime_manifest_sha256": _sha256_file(BUNDLE_DIR / "runtime_manifest.json"),
        "runtime_enabled_outputs": sorted(runtime_manifest.get("runtime_enabled_outputs", ())),
        "reference_demo_hash": reference_demo_hash,
        "container_image": container_image,
        "container_digest": container_digest,
        "supported_platforms": supported_platforms or ["linux/amd64", "linux/arm64"],
        "locked_evaluation_status": {
            "milestone": "M11.6",
            "state": m11_status.get("m11_6_state"),
            "locked_test_opened": m11_status.get("locked_test_opened"),
            "authorization_consumed": m11_status.get("authorization_consumed"),
            "authorized_openings": m11_status.get("authorized_openings"),
            "locked_open_count": m11_status.get("locked_open_count"),
            "locked_rerun": m11_status.get("locked_rerun"),
            "post_locked_tuning": m11_status.get("post_locked_tuning"),
            "locked_final_result": m11_6_closure.get("locked_final_result"),
            "locked_topology_result": m11_6_closure.get("locked_topology_result"),
        },
        "schema_version": 2,
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
