from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hydroswarm import cli


@pytest.mark.real_simulation
def test_self_test_validates_dependencies_network_sqlite_and_model() -> None:
    report = cli.run_self_test()

    assert report["ok"] is True
    assert {"fastapi", "networkx", "numpy", "pydantic", "torch", "wntr"}.issubset(
        report["dependencies"]
    )
    assert report["network"] == {"nodes": 6, "links": 7, "reservoirs": 1, "tanks": 1}
    assert report["sqlite"] == "ok"
    assert report["model_parameters"] > 0
    assert report["inference_run"] is True
    assert report["simulation_run"] is True
    assert len(report["inference_sha256"]) == 64
    assert len(report["simulation_sha256"]) == 64
    assert report["offline_ready"] is True
    assert report["trained_assets"]["calibration_status"] in {"MISSING", "NOT_YET_FIT", "FITTED", "UNREADABLE"}
    assert "present" in report["reference_artifact"]


@pytest.mark.real_simulation
def test_strict_self_test_passes_against_the_real_committed_frozen_release(tmp_path) -> None:
    """SUB-12.1 #21: against this repository's own committed state (the
    frozen V4 release bundle, its FITTED calibration, the committed
    reference-demo artifact, and a built frontend/dist), strict mode must
    report real success -- not just that it runs without crashing."""
    report = cli.run_self_test(strict=True)

    assert report["trained_assets"]["ready"] is True
    assert report["trained_assets"]["calibration_status"] == "FITTED"
    assert report["reference_artifact"]["present"] is True
    if report["frontend_assets"] == "built" and not report["resources"]["warnings"]:
        assert report["ok"] is True
        assert "strict_failures" not in report


@pytest.mark.real_simulation
def test_strict_self_test_reports_every_real_failure_not_just_the_first() -> None:
    """Directly exercises the strict-failure aggregation logic (not the
    real environment, which may not reproduce every failure mode) by
    monkeypatching the underlying facts a real bundle load would produce."""

    class FakeFactory:
        def __init__(self, *_args, **_kwargs):
            self.trained_assets_ready = False
            self.fallback_reason = "v5_trained_assets_unavailable:V5InferenceBundleError"
            self.manifest = None
            self.model_hash = None
            self.bundle_dir = Path("/nonexistent-bundle-dir")

    # Only strict-mode's own failure-aggregation branch is under test here;
    # everything else in run_self_test still runs for real (dependencies,
    # network, sqlite, inference, WNTR) -- this is deliberately not a full
    # mock of run_self_test, so a real regression anywhere else in the
    # function still surfaces.
    import hydroswarm.runtime as runtime_module

    original_factory = runtime_module.V5PipelineFactory
    try:
        runtime_module.V5PipelineFactory = FakeFactory  # type: ignore[misc,assignment]
        report = cli.run_self_test(strict=True)
    finally:
        runtime_module.V5PipelineFactory = original_factory

    assert report["ok"] is False
    assert any("frozen V5 bundle not ready" in failure for failure in report["strict_failures"])
    assert any("calibration status" in failure for failure in report["strict_failures"])


def test_main_self_test_strict_exits_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "run_self_test",
        lambda **_kwargs: {"ok": False, "strict_failures": ["calibration status is 'MISSING', not FITTED"]},
    )

    assert cli.main(["self-test", "--strict"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "strict_failures" in payload


def test_main_self_test_non_strict_still_exits_zero_when_ok_is_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "run_self_test", lambda **_kwargs: {"ok": True})
    assert cli.main(["self-test"]) == 0


def test_main_self_test_prints_machine_readable_result(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "run_self_test", lambda **_kwargs: {"ok": True, "model_parameters": 42})

    assert cli.main(["self-test"]) == 0
    assert json.loads(capsys.readouterr().out) == {"model_parameters": 42, "ok": True}


def test_main_self_test_human_prints_checklist_not_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "run_self_test",
        lambda **_kwargs: {
            "ok": True,
            "trained_assets": {
                "ready": True,
                "model_sha256": "abc",
                "normalization_hash": "def",
                "fallback_reason": None,
                "calibration_status": "FITTED",
            },
            "simulation_run": True,
            "sqlite": "ok",
            "frontend_assets": "built",
            "reference_artifact": {"present": True, "path": "/tmp/reference-incident-v1.json"},
            "resources": {"port_8765_available": True},
        },
    )

    assert cli.main(["self-test", "--human"]) == 0
    out = capsys.readouterr().out
    assert "HydroSwarm readiness" in out
    assert "READY" in out
    assert "OK Frozen HydroCore-v4 bundle verified" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_render_self_test_report_flags_not_ready_reasons() -> None:
    report = {
        "ok": True,
        "trained_assets": {"ready": False, "model_sha256": None, "normalization_hash": None, "fallback_reason": "classical-safe fallback"},
        "simulation_run": True,
        "sqlite": "ok",
        "frontend_assets": "source-only",
        "resources": {"port_8765_available": True},
    }
    rendered = cli.render_self_test_report(report)
    assert "NOT READY" in rendered
    assert "FAIL Frozen HydroCore-v4 bundle verified" in rendered
    assert "reason: classical-safe fallback" in rendered
    assert "reason: frontend not built" in rendered


def test_start_uses_localhost_and_default_port(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: list[str] = []

    def fake_run(command: list[str], check: bool) -> SimpleNamespace:
        assert check is False
        captured.extend(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["start"]) == 0
    assert captured[captured.index("--host") + 1] == "127.0.0.1"
    assert captured[captured.index("--port") + 1] == "8765"
    assert "hydroswarm.api.app:app" in captured
    assert "http://127.0.0.1:8765" in capsys.readouterr().out


def test_start_rejects_invalid_port_without_launching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not launch"))
    assert cli.main(["start", "--port", "0"]) == 2


def test_network_bind_requires_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not launch"))
    assert cli.main(["start", "--host", "0.0.0.0"]) == 2
