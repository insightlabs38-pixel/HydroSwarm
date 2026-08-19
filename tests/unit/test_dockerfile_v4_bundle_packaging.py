"""Submission-readiness SUB-1 (P0): no Docker daemon is available in this
development sandbox, so these tests cannot actually build or run the image.
They instead statically verify the two things that would otherwise make a
built image silently fall back to the classical-safe path while reporting
healthy:

1. `.dockerignore` does not swallow the one `models/` subtree the
   Dockerfile actually needs (it excludes the rest of `models/`, which
   holds unpromoted training checkpoints that must not ship).
2. The Dockerfile copies the release bundle to the exact directory
   `HYDROSWARM_V4_BUNDLE_DIR` is set to, and every file
   `hydroswarm.runtime.v4_inference_bundle.REQUIRED_BUNDLE_FILES` (plus a
   calibration artifact) needs is actually present in the committed bundle
   -- i.e. the COPY has something real to copy.

Building/running the image itself remains a manual/CI verification step
(tracked in the SUB-0 handoff report), not something this test suite
claims to prove.
"""

from __future__ import annotations

import re
from pathlib import Path

from hydroswarm.runtime.v4_inference_bundle import (
    CALIBRATION_ARTIFACT_FILENAME,
    CALIBRATION_STATUS_FILENAME,
    REQUIRED_BUNDLE_FILES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_dockerignore_reincludes_the_v4_release_bundle() -> None:
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text()
    lines = [line.strip() for line in dockerignore.splitlines()]

    assert "models/*" in lines, ".dockerignore must exclude models/* (not bare `models`) so the re-include below can work"
    assert "!models/hydrocore-v4-release" in lines
    assert "!models/hydrocore-v4-release/**" in lines
    assert "!models/hydrocore-v5-release" in lines
    assert "!models/hydrocore-v5-release/**" in lines


def test_dockerfile_copies_bundle_to_the_env_var_it_sets() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()

    env_match = re.search(r"HYDROSWARM_V4_BUNDLE_DIR=(\S+)", dockerfile)
    assert env_match is not None, "Dockerfile must set HYDROSWARM_V4_BUNDLE_DIR"
    bundle_env_value = env_match.group(1)

    assert "COPY models/hydrocore-v4-release/ models/hydrocore-v4-release/" in dockerfile
    # WORKDIR is /app, so the copied relative destination must match the
    # absolute path the env var points the runtime at.
    assert bundle_env_value == "/app/models/hydrocore-v4-release"

    v5_env_match = re.search(r"HYDROSWARM_V5_BUNDLE_DIR=(\S+)", dockerfile)
    assert v5_env_match is not None, "Dockerfile must set HYDROSWARM_V5_BUNDLE_DIR"
    assert "COPY models/hydrocore-v5-release/ models/hydrocore-v5-release/" in dockerfile
    assert v5_env_match.group(1) == "/app/models/hydrocore-v5-release"


def test_dockerfile_copies_the_reference_demo_artifact_to_the_env_var_it_sets() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()

    env_match = re.search(r"HYDROSWARM_REFERENCE_DEMO_PATH=(\S+)", dockerfile)
    assert env_match is not None, "Dockerfile must set HYDROSWARM_REFERENCE_DEMO_PATH"

    assert "COPY artifacts/reference-demo/ artifacts/reference-demo/" in dockerfile
    assert env_match.group(1) == "/app/artifacts/reference-demo/reference-incident-v1.json"

    artifact = PROJECT_ROOT / "artifacts" / "reference-demo" / "reference-incident-v1.json"
    assert artifact.is_file(), "COPY has nothing real to copy without the generated artifact"


def test_dockerfile_runs_a_build_time_self_test_gate() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()

    # SUB-12.1 #21: the build-time gate uses strict=True, which already
    # requires trained_assets.ready, a FITTED calibration, the
    # reference-demo artifact, and a built frontend -- result['ok'] is the
    # single real verdict, not a hand-picked subset of fields re-checked here.
    assert "run_self_test" in dockerfile
    assert "strict=True" in dockerfile
    assert "result['ok']" in dockerfile


def test_committed_release_bundle_has_every_file_the_loader_requires() -> None:
    bundle_dir = PROJECT_ROOT / "models" / "hydrocore-v4-release"

    for name in REQUIRED_BUNDLE_FILES:
        assert (bundle_dir / name).is_file(), f"missing required bundle file: {name}"

    has_calibration = (bundle_dir / CALIBRATION_ARTIFACT_FILENAME).is_file()
    has_calibration_status = (bundle_dir / CALIBRATION_STATUS_FILENAME).is_file()
    assert has_calibration or has_calibration_status, (
        "bundle must have either a calibration artifact or an explicit calibration-status"
    )
