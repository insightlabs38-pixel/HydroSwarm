"""V5 release-engineering rebase: `scripts/setup_common.py verify-bundle`
must validate the SAME frozen bundle the running application actually
serves (`hydroswarm.runtime.v5_defaults.V5PipelineFactory` against
`models/hydrocore-v5-release`), and must never pass merely because the
historical `models/hydrocore-v4-release` bundle happens to also exist in
the checkout. Mirrors the real-bundle-copy-and-corrupt pattern already
used in tests/scientific/test_m10_5_v5_serving_freeze.py.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_BUNDLE = PROJECT_ROOT / "models" / "hydrocore-v5-release"
V4_BUNDLE = PROJECT_ROOT / "models" / "hydrocore-v4-release"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
setup_common = importlib.import_module("setup_common")


def _run_verify_bundle() -> int:
    return setup_common.cmd_verify_bundle(argparse.Namespace())


def test_a_valid_v5_bundle_passes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("HYDROSWARM_V5_BUNDLE_DIR", str(V5_BUNDLE))
    assert _run_verify_bundle() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["model_sha256"] == "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5"


def test_b_missing_v5_bundle_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HYDROSWARM_V5_BUNDLE_DIR", str(tmp_path / "does-not-exist"))
    assert _run_verify_bundle() == 1


def test_c_corrupt_v5_model_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    broken = tmp_path / "v5-corrupt-model"
    shutil.copytree(V5_BUNDLE, broken)
    with (broken / "model.safetensors").open("ab") as handle:
        handle.write(b"corrupt")
    monkeypatch.setenv("HYDROSWARM_V5_BUNDLE_DIR", str(broken))
    assert _run_verify_bundle() == 1


def test_d_corrupt_v5_calibration_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    broken = tmp_path / "v5-corrupt-calibration"
    shutil.copytree(V5_BUNDLE, broken)
    with (broken / "calibration.json").open("a") as handle:
        handle.write("corrupt")
    monkeypatch.setenv("HYDROSWARM_V5_BUNDLE_DIR", str(broken))
    assert _run_verify_bundle() == 1


def test_d_corrupt_v5_manifest_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    broken = tmp_path / "v5-corrupt-manifest"
    shutil.copytree(V5_BUNDLE, broken)
    (broken / "runtime_manifest.json").write_text("not-json")
    monkeypatch.setenv("HYDROSWARM_V5_BUNDLE_DIR", str(broken))
    assert _run_verify_bundle() == 1


def test_e_v4_present_v5_missing_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """V4 being present and hash-verifiable must never be sufficient --
    verify-bundle resolves and checks only the V5 bundle."""
    assert V4_BUNDLE.is_dir(), "sanity check: the historical V4 bundle must exist in this checkout"
    monkeypatch.setenv("HYDROSWARM_V5_BUNDLE_DIR", str(tmp_path / "no-v5-here"))
    assert _run_verify_bundle() == 1


def test_f_v5_valid_v4_absent_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Copy the real V5 bundle to a location where no V4 bundle is
    reachable at all (a bare tmp_path project root) and confirm
    verify-bundle still passes on the V5 bundle alone."""
    isolated_v5 = tmp_path / "isolated-v5-only" / "models" / "hydrocore-v5-release"
    isolated_v5.parent.mkdir(parents=True)
    shutil.copytree(V5_BUNDLE, isolated_v5)
    monkeypatch.setenv("HYDROSWARM_V5_BUNDLE_DIR", str(isolated_v5))
    assert not (isolated_v5.parent / "hydrocore-v4-release").exists()
    assert _run_verify_bundle() == 0


def test_verify_bundle_never_imports_or_uses_the_v4_factory() -> None:
    source = (PROJECT_ROOT / "scripts" / "setup_common.py").read_text()
    import re

    cmd_source = re.search(r"def cmd_verify_bundle.*?(?=\ndef |\Z)", source, re.DOTALL)
    assert cmd_source is not None
    body = cmd_source.group(0)
    assert "V4PipelineFactory" not in body
    assert "resolve_v4_bundle_dir" not in body
    assert "V5PipelineFactory" in body
    assert "resolve_v5_bundle_dir" in body
