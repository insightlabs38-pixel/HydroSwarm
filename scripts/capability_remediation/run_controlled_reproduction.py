"""Reproduce controlled HydroCore-v4 validation into remediation evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

SPEC = importlib.util.spec_from_file_location("phase13", ROOT / "scripts/run_phase13_sentinel_metrics.py")
assert SPEC is not None and SPEC.loader is not None
PHASE13 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE13)

MODEL = ROOT / "models/hydrocore-v4-release/model.safetensors"
RANGES = {"top1": (0.7205, 0.7331), "top3": (0.8680, 0.8756), "mrr": (0.8113, 0.8172)}


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    model_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    model = PHASE13.load_model(MODEL, use_adapters=False, strategist_fields_available=True)
    dataset = PHASE13._load_split("validation")
    dataset.verify_shard_checksums()
    localization = PHASE13.evaluate_split(model, dataset)["localization"]
    values = {"top1": localization["source_top1"], "top3": localization["source_top3"], "mrr": localization["mrr"]}
    in_range = {key: RANGES[key][0] - 0.00005 <= value <= RANGES[key][1] + 0.00005 for key, value in values.items()}
    factory_status = json.loads((ROOT / "models/hydrocore-v4-release/calibration-status.json").read_text())
    manifest = json.loads((ROOT / "models/hydrocore-v4-release/runtime_manifest.json").read_text())
    report = {
        "schema_version": 1, "code_under_test_commit": code_commit, "model_sha": model_sha,
        "calibration_sha": factory_status["calibration_artifact_hash"],
        "feature_schema_sha": manifest["feature_schema_hash"], "normalization_sha": manifest["normalization_hash"],
        "signature_policy_hash": manifest["signature_policy_hash"], "locked_test_opened": False,
        "split": "validation", "examples": localization["examples"], **values,
        "historical_ranges": RANGES, "historical_range_reproduced": all(in_range.values()),
        "in_range": in_range,
    }
    target = ROOT / "reports/evaluation/capability-remediation/controlled-reproduction.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["historical_range_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
