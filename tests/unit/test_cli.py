from __future__ import annotations

import json
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


def test_main_self_test_prints_machine_readable_result(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "run_self_test", lambda: {"ok": True, "model_parameters": 42})

    assert cli.main(["self-test"]) == 0
    assert json.loads(capsys.readouterr().out) == {"model_parameters": 42, "ok": True}


def test_main_self_test_human_prints_checklist_not_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "run_self_test",
        lambda: {
            "ok": True,
            "trained_assets": {"ready": True, "model_sha256": "abc", "normalization_hash": "def", "fallback_reason": None},
            "simulation_run": True,
            "sqlite": "ok",
            "frontend_assets": "built",
            "resources": {"port_8765_available": True},
        },
    )

    assert cli.main(["self-test", "--human"]) == 0
    out = capsys.readouterr().out
    assert "HydroSwarm readiness" in out
    assert "READY" in out
    assert "✓ Frozen HydroCore-v4 bundle verified" in out
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
    assert "✗ Frozen HydroCore-v4 bundle verified" in rendered
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
