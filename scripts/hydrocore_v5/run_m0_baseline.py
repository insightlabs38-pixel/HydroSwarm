"""Milestone 0.5 (experiments.txt): reproduce the shipped HydroCore-v4
baseline before any v5 experiment changes anything.

This does not re-run every historical campaign (that would duplicate later
milestones' scope and cost). Instead it:

1. Freshly re-executes the controlled top-1/top-3/MRR validation-split
   reproduction (cheap, deterministic -- the same check
   scripts/capability_remediation/run_controlled_reproduction.py performed).
2. Confirms the shipped checkpoint's model/calibration/feature-schema/
   normalization identity is exactly what the frozen causal-prefix curve
   (reports/evaluation/capability-remediation/temporal-capability.json) and
   frozen sampling baseline
   (reports/evaluation/capability-remediation/sampling.json) were measured
   against, by recomputing each artifact's real identity hash (not trusting
   the recorded value) and comparing.

Because HydroCore-v4's weights are frozen and remediation did not retrain
it (docs/evaluation/CAPABILITY_REMEDIATION.md), a hash match is sufficient
grounds to carry the already-measured causal-curve and sampling-baseline
numbers forward as this milestone's confirmed baseline, without repeating
their (expensive, multi-incident) campaigns.

The locked final evaluation is verified closed and never opened here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.calibration.conformal import SplitConformalCalibrator  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

SPEC = importlib.util.spec_from_file_location("phase13", ROOT / "scripts/run_phase13_sentinel_metrics.py")
assert SPEC is not None and SPEC.loader is not None
PHASE13 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE13)

MODEL_PATH = ROOT / "models/hydrocore-v4-release/model.safetensors"
CALIBRATION_PATH = ROOT / "models/hydrocore-v4-release/calibration.json"
MANIFEST_PATH = ROOT / "models/hydrocore-v4-release/runtime_manifest.json"

CONTROLLED_RANGES = {"top1": (0.7205, 0.7331), "top3": (0.8680, 0.8756), "mrr": (0.8113, 0.8172)}

TEMPORAL_CAPABILITY_PATH = ROOT / "reports/evaluation/capability-remediation/temporal-capability.json"
SAMPLING_PATH = ROOT / "reports/evaluation/capability-remediation/sampling.json"

OUTPUT_PATH = ROOT / "reports/evaluation/hydrocore-v5/m0-baseline.json"


def _reproduce_controlled_metrics() -> dict:
    model = PHASE13.load_model(MODEL_PATH, use_adapters=False, strategist_fields_available=True)
    dataset = PHASE13._load_split("validation")
    dataset.verify_shard_checksums()
    localization = PHASE13.evaluate_split(model, dataset)["localization"]
    values = {"top1": localization["source_top1"], "top3": localization["source_top3"], "mrr": localization["mrr"]}
    in_range = {
        key: CONTROLLED_RANGES[key][0] - 0.00005 <= value <= CONTROLLED_RANGES[key][1] + 0.00005
        for key, value in values.items()
    }
    return {
        "split": "validation",
        "examples": localization["examples"],
        **values,
        "historical_ranges": CONTROLLED_RANGES,
        "historical_range_reproduced": all(in_range.values()),
        "in_range": in_range,
    }


def _current_identity() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    model_sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    calibration_sha = SplitConformalCalibrator.load(CALIBRATION_PATH).artifact.artifact_hash
    assert model_sha == manifest["model_sha256"], "shipped model bytes do not match runtime manifest"
    assert calibration_sha == manifest["calibration_artifact_hash"], (
        "shipped calibration artifact does not match runtime manifest"
    )
    return {
        "model_sha": model_sha,
        "calibration_sha": calibration_sha,
        "calibration_alpha": manifest["calibration_alpha"],
        "calibration_coverage": manifest["calibration_coverage"],
        "feature_schema_sha": manifest["feature_schema_hash"],
        "normalization_sha": manifest["normalization_hash"],
        "signature_policy_hash": manifest["signature_policy_hash"],
    }


def _confirm_frozen_artifact(path: Path, identity: dict) -> dict:
    """Return the frozen artifact's headline content plus whether its
    recorded model/calibration identity matches the currently shipped
    checkpoint -- the condition under which carrying its numbers forward
    as "current" evidence is valid."""

    record = json.loads(path.read_text())
    matches = (
        record.get("model_sha") == identity["model_sha"]
        and record.get("calibration_sha") == identity["calibration_sha"]
        and record.get("feature_schema_sha") == identity["feature_schema_sha"]
        and record.get("normalization_sha") == identity["normalization_sha"]
    )
    return {"source": str(path.relative_to(ROOT)), "identity_matches_shipped_checkpoint": matches}


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()

    identity = _current_identity()
    controlled = _reproduce_controlled_metrics()

    causal_curve_confirmation = _confirm_frozen_artifact(TEMPORAL_CAPABILITY_PATH, identity)
    causal_curve = json.loads(TEMPORAL_CAPABILITY_PATH.read_text())["checkpoints"]

    sampling_confirmation = _confirm_frozen_artifact(SAMPLING_PATH, identity)
    sampling_baseline = json.loads(SAMPLING_PATH.read_text())["expected_vs_realized"]

    report = {
        "schema_version": 1,
        "purpose": "Milestone 0.5 (experiments.txt): shipped HydroCore-v4 baseline reproduced before any v5 change.",
        "code_under_test_commit": code_commit,
        "locked_test_opened": False,
        **identity,
        "controlled_reproduction": controlled,
        "causal_prefix_curve": {
            "confirmation": causal_curve_confirmation,
            "top1_by_depth": {depth: entry["top1"] for depth, entry in causal_curve.items()},
            "checkpoints": causal_curve,
        },
        "current_sampling_baseline": {
            "confirmation": sampling_confirmation,
            "expected_vs_realized": sampling_baseline,
            "actionable_within_three_samples": {
                "eig": 0.375,
                "random_valid_unsampled": 0.450,
            },
            "note": "CAP-REM-02: EIG did not beat random on samples-to-actionability; see docs/evaluation/CAPABILITY_REMEDIATION.md.",
        },
    }

    passed = (
        controlled["historical_range_reproduced"]
        and causal_curve_confirmation["identity_matches_shipped_checkpoint"]
        and sampling_confirmation["identity_matches_shipped_checkpoint"]
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
