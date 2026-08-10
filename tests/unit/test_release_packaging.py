"""SUB-3: verify the multiarch release packaging surface -- the release
compose file, the release GitHub Actions workflow, and the two generator
scripts (`build_release_manifest.py`, `build_release_bundle.py`).

An actual `docker build`/`docker run` could not be executed in this
sandbox -- see reports/submission-readiness/sub3-docker-sandbox-limitation.md
for the confirmed root cause (CAP_SYS_ADMIN stripped; `unshare` blocked
even for an unprivileged user namespace). These tests verify everything
that does not require a working container runtime: YAML/script structure,
manifest field sourcing from the real frozen-bundle manifests (no
hand-typed scientific hashes), and zip-archive contents.
"""

from __future__ import annotations

import importlib
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

build_release_manifest = importlib.import_module("build_release_manifest")
build_release_bundle = importlib.import_module("build_release_bundle")


# --- docker-compose.release.yml --------------------------------------------


def test_release_compose_exists_and_parses() -> None:
    with (PROJECT_ROOT / "docker-compose.release.yml").open() as handle:
        config = yaml.safe_load(handle)
    assert "hydroswarm" in config["services"]


def test_release_compose_uses_published_image_with_no_local_build() -> None:
    with (PROJECT_ROOT / "docker-compose.release.yml").open() as handle:
        config = yaml.safe_load(handle)
    service = config["services"]["hydroswarm"]
    assert service["image"].startswith("ghcr.io/")
    assert "build" not in service, "release compose must not require a local build"
    assert ":v" in service["image"] or "@sha256:" in service["image"], "must pin a versioned tag, not just :latest"


def test_release_compose_preserves_security_hardening() -> None:
    with (PROJECT_ROOT / "docker-compose.release.yml").open() as handle:
        config = yaml.safe_load(handle)
    service = config["services"]["hydroswarm"]
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert {"no-new-privileges:true"} <= set(service["security_opt"])
    assert service["ports"][0].startswith("127.0.0.1:")


# --- .github/workflows/release.yml ------------------------------------------


def test_release_workflow_parses() -> None:
    with (PROJECT_ROOT / ".github" / "workflows" / "release.yml").open() as handle:
        workflow = yaml.safe_load(handle)
    assert "jobs" in workflow


def test_release_workflow_does_not_publish_on_arbitrary_branch_push() -> None:
    with (PROJECT_ROOT / ".github" / "workflows" / "release.yml").open() as handle:
        workflow = yaml.safe_load(handle)
    # YAML parses the bare `on:` key as boolean True.
    triggers = workflow[True]
    assert "push" in triggers
    assert triggers["push"].get("tags"), "release workflow must gate push triggers on version tags, not branches"
    assert "branches" not in triggers["push"]
    assert "pull_request" not in triggers, "must not publish from PRs against arbitrary branches"


def test_release_workflow_targets_both_platforms() -> None:
    text = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "linux/amd64,linux/arm64" in text
    assert "setup-qemu-action" in text
    assert "setup-buildx-action" in text


def test_release_workflow_has_a_container_self_test_gate_for_each_platform() -> None:
    with (PROJECT_ROOT / ".github" / "workflows" / "release.yml").open() as handle:
        workflow = yaml.safe_load(handle)
    job = workflow["jobs"]["container-self-test"]
    assert job["strategy"]["matrix"]["platform"] == ["linux/amd64", "linux/arm64"]
    text = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "trained_assets']['ready'] is True" in text
    assert "frontend_assets'] == 'built'" in text


def test_release_workflow_verifies_frozen_bundle_hashes_before_building() -> None:
    with (PROJECT_ROOT / ".github" / "workflows" / "release.yml").open() as handle:
        workflow = yaml.safe_load(handle)
    job_order = list(workflow["jobs"])
    assert job_order.index("verify-frozen-bundle") < job_order.index("build-and-push")
    assert "needs" in workflow["jobs"]["build-and-push"]


# --- scripts/build_release_manifest.py --------------------------------------


def test_release_manifest_sources_hashes_from_the_real_runtime_manifest() -> None:
    import json

    runtime_manifest = json.loads(
        (PROJECT_ROOT / "models" / "hydrocore-v4-release" / "runtime_manifest.json").read_text()
    )
    manifest = build_release_manifest.build_manifest()

    assert manifest["model_hash"] == runtime_manifest["model_sha256"]
    assert manifest["normalization_hash"] == runtime_manifest["normalization_hash"]
    assert manifest["calibration_hash"] == runtime_manifest["calibration_artifact_hash"]
    assert manifest["feature_schema_hash"] == runtime_manifest["feature_schema_hash"]
    assert manifest["signature_policy_hash"] == runtime_manifest["signature_policy_hash"]
    assert manifest["schema_version"] == 1
    assert manifest["git_commit"] != "unavailable"


def test_release_manifest_reports_locked_test_status_from_freeze_declaration() -> None:
    manifest = build_release_manifest.build_manifest()
    assert manifest["locked_test_status"]["locked_test_opened"] is False


def test_release_manifest_accepts_container_identity_overrides() -> None:
    manifest = build_release_manifest.build_manifest(
        container_image="ghcr.io/example/hydroswarm:v0.1.0-hackathon",
        container_digest="sha256:deadbeef",
    )
    assert manifest["container_image"] == "ghcr.io/example/hydroswarm:v0.1.0-hackathon"
    assert manifest["container_digest"] == "sha256:deadbeef"


def test_release_manifest_reference_demo_hash_is_null_when_artifact_absent() -> None:
    # SUB-4 (the reference-demo artifact) has not landed yet as of SUB-3;
    # the manifest must degrade to null, not fabricate/guess a hash.
    if (PROJECT_ROOT / "artifacts" / "reference-demo" / "manifest.json").exists():
        pytest.skip("reference-demo artifact already exists in this checkout")
    manifest = build_release_manifest.build_manifest()
    assert manifest["reference_demo_hash"] is None


# --- scripts/build_release_bundle.py ----------------------------------------


def test_release_bundle_contains_required_top_level_files(tmp_path: Path) -> None:
    output = tmp_path / "test-runtime.zip"
    build_release_bundle.build_bundle(output, release_version="v-test")

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())

    assert "SHA256SUMS" in names
    assert "LICENSE" in names
    assert "README.md" in names
    assert "setup_hydroswarm_linux.sh" in names
    assert "start_hydroswarm_linux.sh" in names
    assert any(name.startswith("models/hydrocore-v4-release/") for name in names)
    assert any(name.startswith("src/hydroswarm/") for name in names)


def test_release_bundle_excludes_pycache(tmp_path: Path) -> None:
    output = tmp_path / "test-runtime.zip"
    build_release_bundle.build_bundle(output, release_version="v-test")

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()

    assert not any("__pycache__" in name for name in names)
    assert not any(name.endswith((".pyc", ".pyo")) for name in names)


def test_release_bundle_sha256sums_matches_every_entry(tmp_path: Path) -> None:
    import hashlib

    output = tmp_path / "test-runtime.zip"
    build_release_bundle.build_bundle(output, release_version="v-test")

    with zipfile.ZipFile(output) as archive:
        sha256sums = archive.read("SHA256SUMS").decode("utf-8")
        expected = {}
        for line in sha256sums.strip().splitlines():
            digest, _, name = line.partition("  ")
            expected[name] = digest

        for name in archive.namelist():
            if name == "SHA256SUMS":
                continue
            assert name in expected, f"{name} missing from SHA256SUMS"
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            assert actual == expected[name], f"checksum mismatch for {name}"
