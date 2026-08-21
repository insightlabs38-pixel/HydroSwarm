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
    # SUB-12.1 #21: the release gate uses `--strict`, which already
    # requires trained_assets.ready, a FITTED calibration, the
    # reference-demo artifact, and a built frontend -- report['ok'] is the
    # single real verdict, not a hand-picked subset of fields re-checked here.
    assert "self-test --strict" in text
    assert "report['ok'] is True" in text


def test_release_workflow_verifies_frozen_bundle_hashes_before_building() -> None:
    with (PROJECT_ROOT / ".github" / "workflows" / "release.yml").open() as handle:
        workflow = yaml.safe_load(handle)
    job_order = list(workflow["jobs"])
    assert job_order.index("verify-frozen-bundle") < job_order.index("build-candidate")
    assert "needs" in workflow["jobs"]["build-candidate"]


# --- P0 fix: publication order (v0.2.0 preflight) ---------------------------
#
# The previous revision of this workflow pushed the final version tag and
# `latest` in the same job that built the image, before the post-push
# amd64/arm64 strict container-self-test ran -- a broken image could exist
# under a final public tag for as long as it took that job to fail. These
# tests prove the fixed publication order is structural (in the workflow
# graph itself), not just a comment/intention.


def test_release_workflow_never_pushes_a_final_tag_before_testing() -> None:
    """The pre-validation build/push step (`build-candidate`) must publish
    ONLY a disposable, clearly non-final candidate tag -- never the
    requested release version and never `latest`. If a future edit
    reintroduces `${{ needs.verify-frozen-bundle.outputs.release_version }}`
    or `:latest` into this job's own tags, this must fail."""
    with (PROJECT_ROOT / ".github" / "workflows" / "release.yml").open() as handle:
        workflow = yaml.safe_load(handle)
    build_job = workflow["jobs"]["build-candidate"]
    build_step = next(step for step in build_job["steps"] if step.get("id") == "build")
    tags = build_step["with"]["tags"]
    assert "release-candidate-" in tags
    assert "release_version" not in tags
    assert ":latest" not in tags


def test_release_workflow_container_self_test_pulls_by_exact_digest() -> None:
    """container-self-test must pull the exact digest build-candidate
    produced, not a mutable tag -- otherwise "the tested image" and "the
    image later promoted" could silently diverge."""
    text = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "needs.build-candidate.outputs.digest" in text
    assert "docker pull --platform ${{ matrix.platform }} \"$image_ref\"" in text


def test_release_workflow_final_promotion_depends_on_container_self_test() -> None:
    """Final-tag promotion must be gated on the container-self-test job
    actually succeeding on both platforms, and must reuse the SAME tested
    digest (via `docker buildx imagetools create`, a registry-native
    retag) rather than rebuilding a second, unvalidated image."""
    with (PROJECT_ROOT / ".github" / "workflows" / "release.yml").open() as handle:
        workflow = yaml.safe_load(handle)
    promote_job = workflow["jobs"]["promote-release"]
    assert "container-self-test" in promote_job["needs"]
    assert "build-candidate" in promote_job["needs"]

    text = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "docker buildx imagetools create" in text
    promote_step = next(step for step in promote_job["steps"] if "imagetools create" in step.get("run", ""))
    assert "needs.build-candidate.outputs.digest" in promote_step["run"]


def test_release_workflow_promotion_and_github_release_are_publish_only() -> None:
    """`promote-release` (which writes the final version tag and `latest`)
    and the "Create GitHub Release" step must both be gated to a real
    `push` (tag) event -- never a manual `workflow_dispatch` run, and
    never merely because `GITHUB_REF_NAME` happens to be `main`."""
    with (PROJECT_ROOT / ".github" / "workflows" / "release.yml").open() as handle:
        workflow = yaml.safe_load(handle)

    promote_job = workflow["jobs"]["promote-release"]
    assert promote_job["if"] == "needs.verify-frozen-bundle.outputs.is_release == 'true'"

    release_job = workflow["jobs"]["release-artifacts"]
    gh_release_step = next(
        step for step in release_job["steps"] if step.get("uses", "").startswith("softprops/action-gh-release")
    )
    assert gh_release_step["if"] == "needs.verify-frozen-bundle.outputs.is_release == 'true'"


def test_release_workflow_dispatch_has_no_publishable_version_input() -> None:
    """OPTION A (v0.2.0 preflight task SS4): a tag push is the ONLY
    publishing trigger. `workflow_dispatch` must not accept a
    `release_version`-style input that could be mistaken for -- or misused
    to request -- a real publish; it always resolves to a non-version
    `preflight-<sha>` label that is never promoted."""
    with (PROJECT_ROOT / ".github" / "workflows" / "release.yml").open() as handle:
        workflow = yaml.safe_load(handle)
    triggers = workflow[True]
    dispatch = triggers["workflow_dispatch"]
    assert not dispatch or not dispatch.get("inputs")

    text = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "GITHUB_REF_NAME" not in text or "is_release" in text


# --- scripts/build_release_manifest.py --------------------------------------


def test_release_manifest_sources_hashes_from_the_real_v5_runtime_manifest() -> None:
    import json

    runtime_manifest = json.loads(
        (PROJECT_ROOT / "models" / "hydrocore-v5-release" / "runtime_manifest.json").read_text()
    )
    manifest = build_release_manifest.build_manifest()

    assert manifest["model_hash"] == runtime_manifest["model_sha256"]
    assert manifest["calibration_artifact_hash"] == runtime_manifest["calibration_artifact_hash"]
    assert manifest["calibration_file_sha256"] == runtime_manifest["calibration_file_sha256"]
    assert manifest["feature_schema_hash"] == runtime_manifest["feature_schema_hash"]
    assert manifest["fusion_config_hash"] == runtime_manifest["fusion_config_hash"]
    assert manifest["model_selected_seed"] == runtime_manifest["selected_seed"]
    assert set(manifest["runtime_enabled_outputs"]) == set(runtime_manifest["runtime_enabled_outputs"])
    assert manifest["schema_version"] == 2
    assert manifest["git_commit"] != "unavailable"


def test_release_manifest_runtime_manifest_sha256_matches_the_real_bundle_file() -> None:
    import hashlib

    manifest = build_release_manifest.build_manifest()
    bundle_manifest_path = PROJECT_ROOT / "models" / "hydrocore-v5-release" / "runtime_manifest.json"
    assert manifest["runtime_manifest_sha256"] == hashlib.sha256(bundle_manifest_path.read_bytes()).hexdigest()


def test_release_manifest_sources_frozen_identity_from_m11_2_finalist_identity() -> None:
    import json

    finalist_identity = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "evaluation"
            / "hydrocore-v5"
            / "m11"
            / "m11-2"
            / "m11-2-finalist-identity.json"
        ).read_text()
    )
    manifest = build_release_manifest.build_manifest()
    assert manifest["model_system"] == finalist_identity["system"]
    assert manifest["model_variant"] == finalist_identity["model_variant"]
    assert manifest["model_parameter_count"] == finalist_identity["parameter_count"]


def test_release_manifest_reports_the_real_completed_m11_6_locked_evaluation() -> None:
    """Do NOT source the current release's final-lock status from the
    superseded V4 architecture-freeze record: M11.6 actually executed
    exactly once, after finalist freeze and locked-population
    materialization, and passed both locked-final and locked-topology
    gates. The manifest must report that truthfully, not a stale
    'not yet opened' status left over from before M11.6 ran."""
    manifest = build_release_manifest.build_manifest()
    status = manifest["locked_evaluation_status"]
    assert status["milestone"] == "M11.6"
    assert status["state"] == "M11_6_LOCKED_EVALUATION_PASS"
    assert status["locked_test_opened"] is True
    assert status["authorization_consumed"] is True
    assert status["authorized_openings"] == 1
    assert status["locked_open_count"] == 1
    assert status["locked_rerun"] is False
    assert status["post_locked_tuning"] is False
    assert status["locked_final_result"] == "M11_6_LOCKED_FINAL_PASS"
    assert status["locked_topology_result"] == "M11_6_LOCKED_TOPOLOGY_PASS"


def test_release_manifest_accepts_container_identity_overrides() -> None:
    manifest = build_release_manifest.build_manifest(
        container_image="ghcr.io/example/hydroswarm:v0.1.0-hackathon",
        container_digest="sha256:deadbeef",
    )
    assert manifest["container_image"] == "ghcr.io/example/hydroswarm:v0.1.0-hackathon"
    assert manifest["container_digest"] == "sha256:deadbeef"


def test_release_manifest_reference_demo_hash_is_null_when_artifact_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The manifest must degrade to null rather than fabricate/guess a hash
    # when no reference-demo artifact exists yet. Verified against a real
    # absent-manifest path, independent of whether SUB-4's artifact
    # currently exists in this checkout.
    monkeypatch.setattr(
        build_release_manifest,
        "REFERENCE_DEMO_MANIFEST_PATH",
        PROJECT_ROOT / "artifacts" / "reference-demo" / "does-not-exist.json",
    )
    result = build_release_manifest.build_manifest()
    assert result["reference_demo_hash"] is None


def test_release_manifest_reference_demo_hash_matches_the_real_artifact_when_present() -> None:
    if not (PROJECT_ROOT / "artifacts" / "reference-demo" / "manifest.json").exists():
        pytest.skip("reference-demo artifact not generated in this checkout")
    import json

    reference_manifest = json.loads(
        (PROJECT_ROOT / "artifacts" / "reference-demo" / "manifest.json").read_text()
    )
    manifest = build_release_manifest.build_manifest()
    assert manifest["reference_demo_hash"] == reference_manifest["artifact_sha256"]


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
    assert any(name.startswith("models/hydrocore-v5-release/") for name in names)
    assert any(name.startswith("src/hydroswarm/") for name in names)


def test_release_bundle_excludes_the_historical_v4_release_bundle(tmp_path: Path) -> None:
    """No current runtime path (serving app, setup verify-bundle, strict
    self-test, Docker image) depends on models/hydrocore-v4-release/ -- the
    frozen no-V4-fallback release identity must not ship it as a live
    runtime asset in the judge-facing archive. The historical bundle
    remains in the repository/git history; it is simply not packaged."""
    output = tmp_path / "test-runtime.zip"
    build_release_bundle.build_bundle(output, release_version="v-test")

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())

    assert not any(name.startswith("models/hydrocore-v4-release/") for name in names)


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


def test_release_bundle_includes_every_helper_script_the_setup_scripts_reference(
    tmp_path: Path,
) -> None:
    """SUB-12.1 P0: scripts/setup_common.py was missing from the release
    zip even though every setup_hydroswarm_*.{sh,ps1} script requires it
    to run at all -- a judge extracting the archive would have hit an
    immediate "file not found" the moment they ran setup. This test scans
    the real setup scripts for scripts/*.py references (the same audit
    build_bundle() itself now runs) rather than special-casing
    setup_common.py by name, so a *future* new helper-script dependency
    fails this test too, not just the one already found."""
    referenced = build_release_bundle.referenced_helper_scripts()
    assert referenced, "sanity check: expected at least one scripts/*.py reference"

    output = tmp_path / "test-runtime.zip"
    build_release_bundle.build_bundle(output, release_version="v-test")

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())

    missing = referenced - names
    assert not missing, f"release zip is missing helper script(s) its own setup scripts require: {missing}"


def test_build_bundle_fails_loudly_if_a_setup_script_references_an_unlisted_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the audit in build_bundle() is load-bearing, not dead code:
    if REQUIRED_HELPER_SCRIPTS ever falls out of sync with what the setup
    scripts actually reference, the build must fail, not silently ship a
    broken zip again."""
    monkeypatch.setattr(build_release_bundle, "REQUIRED_HELPER_SCRIPTS", [])
    with pytest.raises(RuntimeError, match="not listed in REQUIRED_HELPER_SCRIPTS"):
        build_release_bundle.build_bundle(tmp_path / "test-runtime.zip", release_version="v-test")


def test_extracted_release_bundle_has_every_file_its_own_setup_scripts_need(tmp_path: Path) -> None:
    """Structural extract-and-run check (fast tier -- no venv/network):
    extracts the real built zip to a clean directory and verifies every
    file `scripts/setup_common.py`'s own consumers, and pip's own editable
    install, actually need is present at the paths the setup scripts
    expect them at. The full setup-script-execution smoke (creates a real
    venv, installs dependencies, launches the server) is a separate,
    slower CI job -- see .github/workflows for the release-zip-smoke job
    -- deliberately not run in the default fast unit-test tier."""
    output = tmp_path / "test-runtime.zip"
    build_release_bundle.build_bundle(output, release_version="v-test")

    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(output) as archive:
        archive.extractall(extract_dir)

    for relative in build_release_bundle.SETUP_SCRIPTS:
        assert (extract_dir / relative).is_file(), f"missing setup/start script: {relative}"
    for relative in build_release_bundle.REQUIRED_HELPER_SCRIPTS:
        assert (extract_dir / relative).is_file(), f"missing helper script: {relative}"

    assert (extract_dir / "pyproject.toml").is_file()
    assert (extract_dir / "LICENSE").is_file()
    assert (extract_dir / "README.md").is_file()
    assert (extract_dir / "src" / "hydroswarm" / "__init__.py").is_file()
    assert (extract_dir / "models" / "hydrocore-v5-release" / "model.safetensors").is_file()

    import shutil
    import subprocess

    if shutil.which("sha256sum") is None:
        pytest.skip("sha256sum binary not available on this platform")

    # Feed the manifest bytes directly.  With text=True, Windows converts
    # LF to CRLF when writing subprocess stdin and GNU sha256sum then treats
    # the trailing CR as part of every archive member path.
    sha256sums = (extract_dir / "SHA256SUMS").read_bytes()
    verify = subprocess.run(
        ["sha256sum", "--check", "--strict"],
        input=sha256sums,
        cwd=extract_dir,
        capture_output=True,
    )
    assert verify.returncode == 0, (
        "sha256sum --check failed on the extracted tree:\n"
        f"{verify.stdout.decode(errors='replace')}\n{verify.stderr.decode(errors='replace')}"
    )
