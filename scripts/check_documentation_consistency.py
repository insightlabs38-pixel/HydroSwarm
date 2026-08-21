"""Focused guard for high-value current-system metadata in public docs."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path


def check(root: Path) -> list[str]:
    errors: list[str] = []
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = pyproject["project"]["version"]
    init_text = (root / "src/hydroswarm/__init__.py").read_text(encoding="utf-8")
    frontend = json.loads((root / "frontend/package.json").read_text(encoding="utf-8"))
    if f'__version__ = "{package_version}"' not in init_text:
        errors.append("package version differs from hydroswarm.__version__")
    if frontend["version"] != package_version:
        errors.append("frontend/package.json version differs from project version")

    # FINAL_SYSTEM.md documents the frozen HydroCore-v5 finalist (the
    # current serving identity, per its own "Current authority" banner),
    # so this must check the v5 release manifest, not the superseded v4
    # one. The v5 manifest also has no single `normalization_hash` field
    # (that concept is v4-runtime-specific -- see runtime/v4_normalization.py);
    # v5's checked identity fields are model_sha256, feature_schema_hash,
    # and calibration_artifact_hash.
    manifest = json.loads((root / "models/hydrocore-v5-release/runtime_manifest.json").read_text(encoding="utf-8"))
    final_system = (root / "docs/FINAL_SYSTEM.md").read_text(encoding="utf-8")
    for key in ("model_sha256", "feature_schema_hash"):
        if manifest[key] not in final_system:
            errors.append(f"FINAL_SYSTEM.md is missing current {key}")
    if manifest["calibration_artifact_hash"] not in final_system:
        errors.append("FINAL_SYSTEM.md is missing current calibration hash")

    freeze = json.loads((root / "reports/results/v4/architecture-freeze.json").read_text(encoding="utf-8"))
    if freeze["locked_test_opened"] is not False:
        errors.append("locked_test_opened must remain false")
    devpost = (root / "docs/DEVPOST.md").read_text(encoding="utf-8").lower()
    checklist = (root / "docs/SUBMISSION_CHECKLIST.md").read_text(encoding="utf-8").lower()
    if "pending final recording" in devpost or "not started" in checklist or "no video exists" in checklist:
        errors.append("public submission docs contain stale video status")
    model_card = (root / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
    if "HydroCore-v4" not in model_card or "4.18M" not in model_card:
        errors.append("MODEL_CARD.md does not clearly identify current HydroCore-v4")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = check(root)
    if errors:
        raise SystemExit("documentation consistency failed:\n- " + "\n- ".join(errors))
    print("documentation consistency: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
