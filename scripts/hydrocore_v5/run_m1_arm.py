"""Milestone 1.3/1.4 (experiments.txt): train one causal-prefix arm/seed.

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m1_arm.py --arm A --seed 31874

Arm A: corrected full-history control (~4.18M HydroCore, correct v5
training infrastructure, historical/full-history regime).
Arm B: uniform causal prefix (depths sampled ~uniformly over
CAUSAL_PREFIX_DEPTHS).
Arm C: early-weighted causal prefix (1-3 high probability, 4-6 medium,
12-25 lower but still substantial; mature history never eliminated).

All three arms use IDENTICAL architecture (Milestone 1.3's "use identical
architectures initially") and load
configs/training-v5-causal.yaml via
``TrainingConfig.from_yaml(..., require_complete_task_weights=True)``
(Milestone 0.3's frozen v5 training configuration). The only in-memory
override applied here is ``seed`` (per-arm/seed identity) and
``gradnorm_log_every_n_batches`` (a documented compute-cost knob, see
below) -- task_weights and every other field come from the committed
config unchanged.

Checkpoint-selection / early-stopping validation is always evaluated at
FULL HISTORY (depth=25) for every arm, so the three arms are compared
against one consistent selection criterion; the actual causal-depth curve
comparison happens later in evaluate_m1_depths.py against
development_holdout at every depth.
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
    full_history_policy,
)
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.trainer import Trainer, set_deterministic_seed  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "training-v5-causal.yaml"
RUN_ROOT = ROOT / "experiments" / "runs" / "hydrocore-v5-causal"
SUMMARY_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-runs"

#: Milestone 0.3's SHARED_MODEL_CONFIG, identical to the shipped v4
#: architecture (verified: HydroCore.from_variant("small", use_adapters=False,
#: **SHARED_MODEL_CONFIG) has exactly 4,182,612 parameters, matching
#: docs/FINAL_SYSTEM.md's shipped v4 parameter count) and to
#: scripts/run_stage_f_training.py's SHARED_MODEL_CONFIG (read as reference
#: only, per Milestone 0.3's "do not import or extend run_stage_f_training.py").
SHARED_MODEL_CONFIG = dict(
    prior_mode="feature_only",
    event_control_heads=True,
    scout_control_heads=True,
    strategist_mode="candidate_conditioned",
    action_vocabulary_size=ACTION_TEMPLATE_COUNT,
    consequence_prescreening_heads=True,
    ood_category_head=True,
)

#: Compute-cost bound, NOT a scientific override: gradient_conflict_logging
#: stays True (config.py's own default v5 posture, needed for Milestone 2),
#: but computing it on every single batch costs one extra
#: torch.autograd.grad call per present task every step (config.py's own
#: documented cost). Every 5th batch still gives a usable per-epoch trend
#: (config.py's own stated rationale for this field), at roughly 1/5 the
#: added cost -- verified end-to-end in a smoke run before any real arm was
#: trained. Applied here via dataclasses.replace, not by editing the
#: committed configs/training-v5-causal.yaml (Milestone 0's frozen v5
#: config), so the frozen file is untouched and this override is visible in
#: every run's recorded config.json.
GRADNORM_LOG_EVERY_N_BATCHES = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_POLICIES))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    assert not locked_test_opened(ROOT), "locked test must remain closed"

    config = TrainingConfig.from_yaml(str(CONFIG_PATH), require_complete_task_weights=True)
    config = replace(config, seed=args.seed, gradnorm_log_every_n_batches=GRADNORM_LOG_EVERY_N_BATCHES)

    set_deterministic_seed(config.seed, deterministic=config.deterministic)

    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    validation_records = build_scenario_pool("validation", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    train_view = CausalPrefixDatasetView(
        train_records,
        expected_split="train",
        signature_library=library,
        depth_policy=ARM_POLICIES[args.arm],
        base_seed=config.seed,
        batch_size=config.batch_size,
    )
    validation_view = CausalPrefixDatasetView(
        validation_records,
        expected_split="validation",
        signature_library=library,
        depth_policy=full_history_policy,
        base_seed=config.seed,
        batch_size=config.batch_size,
    )

    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    param_count = sum(p.numel() for p in model.parameters())

    run_root = RUN_ROOT / f"{args.arm}-seed{args.seed}"
    started = time.time()
    trainer = Trainer(model, train_view, config=config, run_root=run_root, validation_dataset=validation_view)
    summary = trainer.fit()
    wall_seconds = time.time() - started

    record = {
        "schema_version": 1,
        "purpose": "Milestone 1.3/1.4 (experiments.txt): one causal-prefix arm/seed training run.",
        "arm": args.arm,
        "seed": args.seed,
        "arm_description": {
            "A": "corrected full-history control",
            "B": "uniform causal prefix",
            "C": "early-weighted causal prefix",
        }[args.arm],
        "model_architecture": {"variant": "small", "use_adapters": False, **SHARED_MODEL_CONFIG, "param_count": param_count},
        "training_config": asdict(config),
        "training_config_source": str(CONFIG_PATH.relative_to(ROOT)),
        "train_manifest_hash": train_view.manifest_hash,
        "validation_manifest_hash": validation_view.manifest_hash,
        "signature_library_manifest_hash": library.manifest_hash,
        "train_scenario_count": len(train_records),
        "validation_scenario_count": len(validation_records),
        "wall_seconds": wall_seconds,
        "training_summary": asdict(summary),
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARY_ROOT / f"{args.arm}-seed{args.seed}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"arm": args.arm, "seed": args.seed, "wall_seconds": wall_seconds, "summary": record["training_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
