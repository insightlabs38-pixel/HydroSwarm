"""Capability diagnostic Section 5: reproduce the documented controlled
HydroCore-v4 evaluation against the EXACT checkpoint currently served in
production (models/hydrocore-v4-release, model_sha256=a501ad87...), not
just the stage-f training-run export the original phase13 measurement
script defaults to.

Reuses scripts/run_phase13_sentinel_metrics.py's own load_model/
evaluate_split (the same methodology that originally produced the
documented 0.7205/0.7331 top-1 range) rather than re-implementing
evaluation logic, so a divergence -- if any -- reflects a real checkpoint
difference, not a methodology difference.

No locked-test access: only the non-locked `validation` split of
data/learning-v2/cycle-b2-joint-v4 is read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "phase13_sentinel_metrics", ROOT / "scripts" / "run_phase13_sentinel_metrics.py"
)
assert _spec is not None and _spec.loader is not None
_phase13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase13)

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

DOCUMENTED_TOP1_RANGE = (0.7205, 0.7331)
DOCUMENTED_TOP3_RANGE = (0.8680, 0.8756)
DOCUMENTED_MRR_RANGE = (0.8113, 0.8172)

FROZEN_SERVED_CHECKPOINT = ROOT / "experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810/model.safetensors"
RELEASE_BUNDLE_CHECKPOINT = ROOT / "models/hydrocore-v4-release/model.safetensors"
ORIGINAL_MEASUREMENT_CHECKPOINT = (
    ROOT / "experiments/runs/stage-f/no_adapters-seed20260810/20260808T041727Z-de5f4b0e/model-export.safetensors"
)
EXPECTED_FROZEN_MODEL_SHA256 = "a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed for this diagnostic"

    frozen_sha = _sha256(FROZEN_SERVED_CHECKPOINT)
    bundle_sha = _sha256(RELEASE_BUNDLE_CHECKPOINT)
    original_sha = _sha256(ORIGINAL_MEASUREMENT_CHECKPOINT)
    assert frozen_sha == EXPECTED_FROZEN_MODEL_SHA256, (
        f"served checkpoint sha mismatch: {frozen_sha} != {EXPECTED_FROZEN_MODEL_SHA256}"
    )
    assert frozen_sha == bundle_sha, "checkpoint identity and release bundle model files diverge"

    model = _phase13.load_model(FROZEN_SERVED_CHECKPOINT, use_adapters=False, strategist_fields_available=True)
    dataset = _phase13._load_split("validation")
    dataset.verify_shard_checksums()
    metrics = _phase13.evaluate_split(model, dataset)
    loc = metrics["localization"]

    # Documented ranges are printed rounded to 4 decimal places
    # (docs/MODEL_CARD.md, phase13-metrics-and-baselines.md); allow the
    # rounding tolerance (5e-5) the doc itself introduces, not a looser
    # "close enough" fudge factor.
    _ROUNDING_TOLERANCE = 5e-5
    top1_in_range = DOCUMENTED_TOP1_RANGE[0] - _ROUNDING_TOLERANCE <= loc["source_top1"] <= DOCUMENTED_TOP1_RANGE[1] + _ROUNDING_TOLERANCE
    top3_in_range = DOCUMENTED_TOP3_RANGE[0] - _ROUNDING_TOLERANCE <= loc["source_top3"] <= DOCUMENTED_TOP3_RANGE[1] + _ROUNDING_TOLERANCE
    mrr_in_range = DOCUMENTED_MRR_RANGE[0] - _ROUNDING_TOLERANCE <= loc["mrr"] <= DOCUMENTED_MRR_RANGE[1] + _ROUNDING_TOLERANCE
    reproduced = top1_in_range and top3_in_range and mrr_in_range

    report = {
        "schema_version": 1,
        "section": "5_reproduce_controlled_result",
        "checkpoint_evaluated": {
            "path": str(FROZEN_SERVED_CHECKPOINT.relative_to(ROOT)),
            "model_sha256": frozen_sha,
            "note": "This is the exact checkpoint actually served by models/hydrocore-v4-release "
            "(identical sha256), i.e. the real production model -- not merely a same-lineage "
            "training-run export.",
        },
        "original_measurement_checkpoint": {
            "path": str(ORIGINAL_MEASUREMENT_CHECKPOINT.relative_to(ROOT)),
            "model_sha256": original_sha,
            "differs_from_served_checkpoint_file": original_sha != frozen_sha,
            "note": "scripts/run_phase13_sentinel_metrics.py's own --checkpoints default points here, "
            "not at the frozen release bundle. The file differs byte-for-byte (different export "
            "path/packaging) from the served checkpoint, so this diagnostic explicitly re-measures "
            "against the served checkpoint instead of trusting that the original number transfers.",
        },
        "documented_range": {
            "top1": list(DOCUMENTED_TOP1_RANGE),
            "top3": list(DOCUMENTED_TOP3_RANGE),
            "mrr": list(DOCUMENTED_MRR_RANGE),
            "source": "docs/MODEL_CARD.md; reports/results/v4/phase13-metrics-and-baselines.md (seed 20260810/20260811)",
        },
        "reproduced": {
            "split": "validation",
            "corpus": "data/learning-v2/cycle-b2-joint-v4/tensors-normalized",
            "examples_total": metrics["examples"],
            "examples_localization_eligible": loc["examples"],
            "top1": loc["source_top1"],
            "top3": loc["source_top3"],
            "mrr": loc["mrr"],
            "ece_raw_uncalibrated": loc["ece"],
            "candidate_coverage_at_3": loc["candidate_coverage_at_3"],
        },
        "in_documented_range": {
            "top1": top1_in_range,
            "top3": top3_in_range,
            "mrr": mrr_in_range,
        },
        "verdict": "REPRODUCED" if reproduced else "CAP-EVAL-REPRODUCTION",
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "reproduction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in {"verdict", "reproduced", "in_documented_range"}}, indent=2))
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
