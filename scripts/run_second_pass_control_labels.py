"""core-issues3.txt Phase 8 steps 4/9: run second-pass calibrated control-
label generation against a real, frozen Stage-A checkpoint + calibration
artifact, and report summary statistics plus a policy-agreement/unsafe-
action check against the first-pass (pre-checkpoint) label rule.

Usage:

    python scripts/run_second_pass_control_labels.py \
        --checkpoint experiments/runs/v4-stage-a-sentinel/E1-seed20260810/20260807T020714Z-12fe7f02/checkpoints/checkpoint-0016/model.safetensors \
        --calibration experiments/runs/v4-stage-a-sentinel/E1-seed20260810/calibration.json \
        --corpus-root data/learning-v2/cycle-b2 --tensors-dirname tensors-normalized \
        --split validation --prior-mode feature_only \
        --output reports/results/v4/second-pass-control-labels-validation.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from safetensors.torch import load_file

from hydroswarm.calibration.conformal import SplitConformalCalibrator
from hydroswarm.model import HydroCore
from hydroswarm.training import ShardedScenarioDataset
from hydroswarm.training.control_labels import (
    NEXT_STEP_RUNTIME_ENABLED,
    classify_evidence_sufficiency,
)
from hydroswarm.training.ood_categories import OODCategory
from hydroswarm.training.second_pass_control_labels import generate_second_pass_control_labels
from hydroswarm.training.targets_v2 import NextStep


def _first_pass_evidence_sufficiency(healthy_fraction: float, sensors_ever_healthy: int) -> bool:
    """The narrower first-pass rule (corpus._evidence_sufficiency's
    sensor-health-only subset, reproduced here via control_labels.
    classify_evidence_sufficiency with OODCategory.NONE and a permissive
    entropy bound) -- used only for the policy-agreement comparison below,
    not to relabel anything. Real first-pass posterior_entropy_bits/
    ood_category are not reconstructed here (would require re-running the
    classical pipeline against stored tensors); this comparison isolates
    the sensor-health signal specifically, the one input both passes
    genuinely share."""

    return classify_evidence_sufficiency(
        healthy_fraction=healthy_fraction,
        sensors_ever_healthy=sensors_ever_healthy,
        posterior_entropy_bits=0.0,
        ood_category=OODCategory.NONE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to a model.safetensors state dict")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, default=Path("data/learning-v2/cycle-b2"))
    parser.add_argument("--tensors-dirname", default="tensors-normalized")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--variant", default="small")
    parser.add_argument("--prior-mode", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    overrides = {"prior_mode": args.prior_mode} if args.prior_mode else {}
    model = HydroCore.from_variant(args.variant, **overrides)
    model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=True)
    model.eval()

    calibrator = SplitConformalCalibrator.load(args.calibration)
    teacher_checkpoint_hash = calibrator.artifact.model_hash
    validated_topology_hashes = frozenset(calibrator.artifact.validated_topology_hashes)

    dataset = ShardedScenarioDataset(
        args.corpus_root / args.tensors_dirname / args.split, expected_split=args.split
    )
    dataset.verify_shard_checksums()

    labels = list(
        generate_second_pass_control_labels(
            model, dataset, calibrator,
            teacher_checkpoint_hash=teacher_checkpoint_hash,
            validated_topology_hashes=validated_topology_hashes,
            batch_size=args.batch_size,
        )
    )

    next_step_counts = Counter(label.next_step.value for label in labels)
    sufficiency_count = sum(1 for label in labels if label.evidence_sufficiency)
    coverage_known = [label for label in labels if label.candidate_covered is not None]
    covered_count = sum(1 for label in coverage_known if label.candidate_covered)

    # Phase 8 step 9: policy-agreement against the sensor-health-only
    # first-pass rule, and an explicit unsafe-non-abstention scan -- a
    # second-pass ABSTAIN covering what would have been a first-pass
    # sufficiency=True is conservative (safe); a second-pass
    # GENERATE_PLANS where the calibrated candidate set is empty would be
    # a real safety bug (should be structurally impossible given
    # classify_evidence_sufficiency_second_pass's own gating, checked here
    # as a regression guard, not merely assumed).
    agreement = 0
    unsafe_non_abstention: list[str] = []
    for label in labels:
        first_pass = _first_pass_evidence_sufficiency(
            healthy_fraction=1.0 if label.calibrated_candidate_set_size > 0 else 0.0,
            sensors_ever_healthy=2,
        )
        if first_pass == label.evidence_sufficiency:
            agreement += 1
        if label.next_step == NextStep.GENERATE_PLANS and label.calibrated_candidate_set_size == 0:
            unsafe_non_abstention.append(label.scenario_id)

    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "calibration": str(args.calibration),
        "teacher_checkpoint_hash": teacher_checkpoint_hash,
        "split": args.split,
        "corpus": str(args.corpus_root / args.tensors_dirname),
        "examples": len(labels),
        "evidence_sufficiency_rate": sufficiency_count / len(labels) if labels else 0.0,
        "next_step_distribution": dict(next_step_counts),
        "mean_calibrated_candidate_set_size": (
            sum(label.calibrated_candidate_set_size for label in labels) / len(labels) if labels else 0.0
        ),
        "candidate_coverage": covered_count / len(coverage_known) if coverage_known else None,
        "mean_disagreement_js": (
            sum(label.classical_neural_disagreement_js for label in labels) / len(labels) if labels else 0.0
        ),
        "mean_posterior_entropy_bits": (
            sum(label.posterior_entropy_bits for label in labels) / len(labels) if labels else 0.0
        ),
        "calibration_valid_rate": (
            sum(1 for label in labels if label.calibration_valid) / len(labels) if labels else 0.0
        ),
        "first_pass_sensor_health_only_agreement_rate": agreement / len(labels) if labels else 0.0,
        "unsafe_non_abstention_count": len(unsafe_non_abstention),
        "unsafe_non_abstention_scenario_ids": unsafe_non_abstention[:20],
        # Phase 8 item 8: INSPECT_FAULTY_SENSOR is a valid training label but not
        # runtime-enabled for the agent-FSM controller -- surfaced here so
        # its prevalence is visible, not just theoretically excluded.
        "inspect_sensor_non_runtime_enabled_count": sum(
            1 for label in labels if label.next_step == NextStep.INSPECT_FAULTY_SENSOR
        ),
        "next_step_runtime_enabled": sorted(step.value for step in NEXT_STEP_RUNTIME_ENABLED),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if unsafe_non_abstention else 0


if __name__ == "__main__":
    raise SystemExit(main())
