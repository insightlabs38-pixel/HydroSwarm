from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hydroswarm import cli


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


def test_start_uses_localhost_and_default_port(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: list[str] = []

    def fake_run(command: list[str], check: bool) -> SimpleNamespace:
        assert check is False
        captured.extend(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["start"]) == 0
    assert captured[captured.index("--server.address") + 1] == "127.0.0.1"
    assert captured[captured.index("--server.port") + 1] == "8765"
    assert "http://127.0.0.1:8765" in capsys.readouterr().out


def test_start_rejects_invalid_port_without_launching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not launch"))
    assert cli.main(["start", "--port", "0"]) == 2
