"""Milestone 8.7: train one representation arm/seed under the frozen M8.7
protocol (docs/evaluation/HYDROCORE_V5_M8_7_PROTOCOL.md).

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m8_7_arm.py --arm CURRENT_CONTROL --seed 31874
    .venv/bin/python scripts/hydrocore_v5/run_m8_7_arm.py --arm AGE_FIX_ONLY --seed 31874
    .venv/bin/python scripts/hydrocore_v5/run_m8_7_arm.py --arm AGE_FIX_PLUS_RELATIVE_TIME --seed 31874

Byte-for-byte the same training entry point M1 uses (`run_m1_arm.py`:
same `configs/training-v5-causal.yaml`, same `Trainer`, same train/
validation scenario pools, same Arm-A full-history depth policy for ALL
THREE M8.7 arms -- see the frozen protocol document, Section 6, for why
M8.7 does not re-run the depth-POLICY ablation), differing only in:
  - which `HydraulicFeatureBuilder.build` keyword arguments
    (`unobserved_age_sentinel`, `include_relative_gap_feature`) the
    dataset view applies, and
  - which `HydroCore` architecture keyword arguments
    (`temporal_feature_dim`, `quality_feature_dim`,
    `elapsed_time_normalization`) the model is constructed with.

Checkpoints are written under `experiments/runs/hydrocore-v5-causal-m8-7/`
(a separate path from Milestone 1's `hydrocore-v5-causal/` runs -- never
overwrites or is confused with a historical v4/M1 checkpoint).

Writes:
  reports/evaluation/hydrocore-v5/m8-7-runs/<ARM>-seed<SEED>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    ARM_POLICIES,
    CausalPrefixDatasetView,
    build_scenario_pool,
    fit_pool_signature_library,
)
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.trainer import Trainer, set_deterministic_seed  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "training-v5-causal.yaml"
RUN_ROOT = ROOT / "experiments" / "runs" / "hydrocore-v5-causal-m8-7"
SUMMARY_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-7-runs"

#: Identical to run_m1_arm.py's own SHARED_MODEL_CONFIG -- the frozen v5
#: architecture baseline every M8.7 arm builds from.
SHARED_MODEL_CONFIG = dict(
    prior_mode="feature_only",
    event_control_heads=True,
    scout_control_heads=True,
    strategist_mode="candidate_conditioned",
    action_vocabulary_size=ACTION_TEMPLATE_COUNT,
    consequence_prescreening_heads=True,
    ood_category_head=True,
)

GRADNORM_LOG_EVERY_N_BATCHES = 5  # identical compute-cost knob to run_m1_arm.py.

#: The three required M8.7 arms (Section 2 of the milestone). Each entry:
#: (feature-builder kwargs, HydroCore architecture kwargs).
ARM_DEFINITIONS: dict[str, dict[str, object]] = {
    "CURRENT_CONTROL": {
        "feature_kwargs": {"unobserved_age_sentinel": "incident_elapsed", "include_relative_gap_feature": False},
        "model_kwargs": {"temporal_feature_dim": 6, "quality_feature_dim": 4, "elapsed_time_normalization": "window_relative"},
    },
    "AGE_FIX_ONLY": {
        "feature_kwargs": {"unobserved_age_sentinel": "fixed", "include_relative_gap_feature": False},
        "model_kwargs": {"temporal_feature_dim": 6, "quality_feature_dim": 4, "elapsed_time_normalization": "window_relative"},
    },
    "AGE_FIX_PLUS_RELATIVE_TIME": {
        "feature_kwargs": {"unobserved_age_sentinel": "fixed", "include_relative_gap_feature": True},
        "model_kwargs": {"temporal_feature_dim": 7, "quality_feature_dim": 5, "elapsed_time_normalization": "fixed_scale"},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_DEFINITIONS))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    assert not locked_test_opened(ROOT), "locked test must remain closed"

    definition = ARM_DEFINITIONS[args.arm]
    feature_kwargs = definition["feature_kwargs"]
    model_kwargs = definition["model_kwargs"]

    config = TrainingConfig.from_yaml(str(CONFIG_PATH), require_complete_task_weights=True)
    config = replace(config, seed=args.seed, gradnorm_log_every_n_batches=GRADNORM_LOG_EVERY_N_BATCHES)

    set_deterministic_seed(config.seed, deterministic=config.deterministic)

    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    validation_records = build_scenario_pool("validation", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    # All three M8.7 arms use Arm-A's full-history-control depth policy
    # (frozen protocol Section 6) -- M8.7 is a representation study, not a
    # re-run of Milestone 1's depth-POLICY ablation.
    depth_policy = ARM_POLICIES["A"]

    train_view = CausalPrefixDatasetView(
        train_records, expected_split="train", signature_library=library, depth_policy=depth_policy,
        base_seed=config.seed, batch_size=config.batch_size, **feature_kwargs,
    )
    validation_view = CausalPrefixDatasetView(
        validation_records, expected_split="validation", signature_library=library, depth_policy=depth_policy,
        base_seed=config.seed, batch_size=config.batch_size, **feature_kwargs,
    )

    model = HydroCore.from_variant("small", use_adapters=False, **model_kwargs, **SHARED_MODEL_CONFIG)
    param_count = sum(p.numel() for p in model.parameters())

    run_root = RUN_ROOT / f"{args.arm}-seed{args.seed}"
    started = time.time()
    trainer = Trainer(model, train_view, config=config, run_root=run_root, validation_dataset=validation_view)
    summary = trainer.fit()
    wall_seconds = time.time() - started

    export_path = summary.export_path
    checkpoint_sha256 = summary.export_sha256

    record = {
        "schema_version": 1,
        "purpose": "Milestone 8.7: one representation-arm/seed training run under the frozen M8.7 protocol.",
        "arm": args.arm,
        "seed": args.seed,
        "feature_kwargs": feature_kwargs,
        "model_kwargs": model_kwargs,
        "model_architecture": {
            "variant": "small", "use_adapters": False, **model_kwargs, **SHARED_MODEL_CONFIG, "param_count": param_count,
        },
        "training_config": asdict(config),
        "training_config_source": str(CONFIG_PATH.relative_to(ROOT)),
        "train_manifest_hash": train_view.manifest_hash,
        "validation_manifest_hash": validation_view.manifest_hash,
        "signature_library_manifest_hash": library.manifest_hash,
        "train_scenario_count": len(train_records),
        "validation_scenario_count": len(validation_records),
        "wall_seconds": wall_seconds,
        "export_path": export_path,
        "checkpoint_sha256": checkpoint_sha256,
        "training_summary": asdict(summary),
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARY_ROOT / f"{args.arm}-seed{args.seed}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "arm": args.arm, "seed": args.seed, "wall_seconds": wall_seconds, "param_count": param_count,
        "checkpoint_sha256": checkpoint_sha256, "summary": record["training_summary"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
