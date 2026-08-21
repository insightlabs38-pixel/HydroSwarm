"""Submission-readiness SUB-1 (P0): no Docker daemon is available in this
development sandbox, so these tests cannot actually build or run the image.
They instead statically verify the two things that would otherwise make a
built image silently fall back to the classical-safe path while reporting
healthy:

1. `.dockerignore` does not swallow the one `models/` subtree the
   Dockerfile actually needs (it excludes the rest of `models/`, which
   holds unpromoted training checkpoints -- and the historical V4 release
   bundle -- that must not ship in the current release image).
2. The Dockerfile copies the frozen V5 release bundle to the exact
   directory `HYDROSWARM_V5_BUNDLE_DIR` is set to, and every file the V5
   runtime manifest's own `files` mapping declares is actually present in
   the committed bundle -- i.e. the COPY has something real to copy.

Building/running the image itself remains a manual/CI verification step
(`.github/workflows/docker-verify.yml`, `.github/workflows/release.yml`),
not something this static unit suite claims to prove.

No current runtime path depends on the historical
`models/hydrocore-v4-release/` bundle (the production app, `hydroswarm.cli
run_self_test`, and this Dockerfile's own build-time strict-self-test gate
all resolve exclusively through `V5PipelineFactory`/
`resolve_v5_bundle_dir`), so per the frozen no-V4-fallback release policy
that bundle is deliberately excluded from both the build context and the
image -- see `test_dockerignore_and_dockerfile_exclude_the_historical_v4_bundle`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_BUNDLE_DIR = PROJECT_ROOT / "models" / "hydrocore-v5-release"


def test_dockerignore_reincludes_the_v5_release_bundle() -> None:
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text()
    lines = [line.strip() for line in dockerignore.splitlines()]

    assert "models/*" in lines, ".dockerignore must exclude models/* (not bare `models`) so the re-include below can work"
    assert "!models/hydrocore-v5-release" in lines
    assert "!models/hydrocore-v5-release/**" in lines


def test_dockerignore_and_dockerfile_exclude_the_historical_v4_bundle() -> None:
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text()
    lines = [line.strip() for line in dockerignore.splitlines()]
    assert "!models/hydrocore-v4-release" not in lines
    assert "!models/hydrocore-v4-release/**" not in lines

    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
    assert "COPY models/hydrocore-v4-release" not in dockerfile
    assert "HYDROSWARM_V4_BUNDLE_DIR=" not in dockerfile


def test_dockerfile_copies_v5_bundle_to_the_env_var_it_sets() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()

    v5_env_match = re.search(r"HYDROSWARM_V5_BUNDLE_DIR=(\S+)", dockerfile)
    assert v5_env_match is not None, "Dockerfile must set HYDROSWARM_V5_BUNDLE_DIR"
    assert "COPY models/hydrocore-v5-release/ models/hydrocore-v5-release/" in dockerfile
    # WORKDIR is /app, so the copied relative destination must match the
    # absolute path the env var points the runtime at.
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


def test_committed_v5_release_bundle_has_every_file_the_loader_requires() -> None:
    manifest = json.loads((V5_BUNDLE_DIR / "runtime_manifest.json").read_text())
    for name in manifest["files"]:
        assert (V5_BUNDLE_DIR / name).is_file(), f"missing required V5 bundle file: {name}"
