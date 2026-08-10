"""Submission-readiness SUB-1: hydroswarm.runtime.paths is the single
resolver hydroswarm.api.app and hydroswarm.cli.run_self_test both use for
the frozen V4 release bundle directory and the writable runtime data
directory. These tests exercise the resolver in isolation (no torch/WNTR
import needed), independent of a real bundle being present.
"""

from __future__ import annotations

from pathlib import Path

from hydroswarm.runtime.paths import (
    DATA_DIR_ENV_VAR,
    FROZEN_SCENARIO_DIR_ENV_VAR,
    REFERENCE_DEMO_PATH_ENV_VAR,
    V4_BUNDLE_DIR_ENV_VAR,
    resolve_data_dir,
    resolve_frozen_scenario_dir,
    resolve_reference_demo_path,
    resolve_v4_bundle_dir,
)


def test_env_var_override_takes_priority_over_project_root(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "custom-bundle-location"
    override.mkdir()
    monkeypatch.setenv(V4_BUNDLE_DIR_ENV_VAR, str(override))

    resolved = resolve_v4_bundle_dir(project_root=tmp_path / "unrelated-project-root")

    assert resolved == override.resolve()


def test_env_var_override_need_not_exist_yet(tmp_path: Path, monkeypatch) -> None:
    """The override is trusted verbatim (fail-closed happens downstream in
    V4PipelineFactory, not here) so a container can set it before the
    bundle is mounted/verified."""
    override = tmp_path / "not-created-yet"
    monkeypatch.setenv(V4_BUNDLE_DIR_ENV_VAR, str(override))

    resolved = resolve_v4_bundle_dir()

    assert resolved == override.resolve()


def test_project_root_relative_default_when_no_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(V4_BUNDLE_DIR_ENV_VAR, raising=False)
    bundle = tmp_path / "models" / "hydrocore-v4-release"
    bundle.mkdir(parents=True)

    resolved = resolve_v4_bundle_dir(project_root=tmp_path)

    assert resolved == bundle.resolve()


def test_falls_back_to_cwd_relative_bundle_when_project_root_candidate_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv(V4_BUNDLE_DIR_ENV_VAR, raising=False)
    fake_project_root = tmp_path / "project-root-without-a-bundle"
    fake_project_root.mkdir()

    cwd = tmp_path / "cwd-with-a-bundle"
    (cwd / "models" / "hydrocore-v4-release").mkdir(parents=True)
    monkeypatch.chdir(cwd)

    resolved = resolve_v4_bundle_dir(project_root=fake_project_root)

    assert resolved == (cwd / "models" / "hydrocore-v4-release").resolve()


def test_cli_and_api_resolve_identically_with_no_override(monkeypatch) -> None:
    """hydroswarm.cli.run_self_test calls resolve_v4_bundle_dir() with no
    arguments; hydroswarm.api.app passes its own module-file-derived
    project root. Both anchor to this module's own file location
    (independent of the caller's __file__ or cwd), so they must agree."""
    monkeypatch.delenv(V4_BUNDLE_DIR_ENV_VAR, raising=False)

    from hydroswarm.api.app import _PROJECT_ROOT

    assert resolve_v4_bundle_dir() == resolve_v4_bundle_dir(project_root=_PROJECT_ROOT)


def test_data_dir_env_var_override(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "writable-volume"
    monkeypatch.setenv(DATA_DIR_ENV_VAR, str(override))

    resolved = resolve_data_dir(project_root=tmp_path / "unrelated")

    assert resolved == override.resolve()


def test_data_dir_falls_back_to_project_root_generated_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(DATA_DIR_ENV_VAR, raising=False)

    resolved = resolve_data_dir(project_root=tmp_path)

    assert resolved == (tmp_path / "data" / "generated").resolve()


def test_reference_demo_env_var_override_takes_priority(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "custom-reference.json"
    monkeypatch.setenv(REFERENCE_DEMO_PATH_ENV_VAR, str(override))

    resolved = resolve_reference_demo_path(project_root=tmp_path / "unrelated-project-root")

    assert resolved == override.resolve()


def test_reference_demo_project_root_relative_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(REFERENCE_DEMO_PATH_ENV_VAR, raising=False)
    artifact = tmp_path / "artifacts" / "reference-demo" / "reference-incident-v1.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}")

    resolved = resolve_reference_demo_path(project_root=tmp_path)

    assert resolved == artifact.resolve()


def test_reference_demo_falls_back_to_cwd_relative_when_project_root_candidate_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv(REFERENCE_DEMO_PATH_ENV_VAR, raising=False)
    fake_project_root = tmp_path / "project-root-without-artifact"
    fake_project_root.mkdir()

    cwd = tmp_path / "cwd-with-artifact"
    artifact = cwd / "artifacts" / "reference-demo" / "reference-incident-v1.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}")
    monkeypatch.chdir(cwd)

    resolved = resolve_reference_demo_path(project_root=fake_project_root)

    assert resolved == artifact.resolve()


def test_frozen_scenario_dir_env_var_override_takes_priority(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "custom-frozen-scenario"
    monkeypatch.setenv(FROZEN_SCENARIO_DIR_ENV_VAR, str(override))

    resolved = resolve_frozen_scenario_dir(project_root=tmp_path / "unrelated-project-root")

    assert resolved == override.resolve()


def test_frozen_scenario_dir_project_root_relative_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(FROZEN_SCENARIO_DIR_ENV_VAR, raising=False)
    frozen = tmp_path / "data" / "frozen"
    frozen.mkdir(parents=True)

    resolved = resolve_frozen_scenario_dir(project_root=tmp_path)

    assert resolved == frozen.resolve()
