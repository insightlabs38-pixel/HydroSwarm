"""Milestone 2.2 (experiments.txt): multitask objective-design ablation
arms, trained on the SAME causal-prefix data distribution Milestone 1
selected as its provisional winner (M2 studies the loss/architecture
design, holding the evidence-depth distribution fixed -- Milestone 1's own
variable).

Arm A (full governed multitask) is NOT retrained here: it is Milestone 1's
selected winning arm/seed run itself (see run_m2_conflict.py's
_select_base_run), reused directly by evaluate_m2.py to avoid a redundant,
compute-costly duplicate run of an already-trained configuration.

Arm B (primary-runtime focused): halves the weight of any task that
run_m2_conflict.py's m2-conflict.json found in frequent negative conflict
(>=50% of measured batches) with source_node, EXCLUDING safety/control
tasks (event_presence, evidence_sufficiency, sensor_fault) which
experiments.txt Milestone 2.2 explicitly says to preserve. Auxiliary tasks
are reweighted, never deleted ("Do not simply delete all auxiliary
tasks.").

Arm C (role-isolated/adapters): identical task_weights to Arm A, with
use_adapters=True (HydroCore's existing Scout/Strategist/Sentinel role
adapters), so Scout/Strategist gradients (where present) interfere less
with the shared Sentinel/localization representation.
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
from run_m1_arm import CONFIG_PATH, GRADNORM_LOG_EVERY_N_BATCHES, SHARED_MODEL_CONFIG  # noqa: E402
from run_m2_conflict import M1_RESULTS_PATH, OUTPUT_PATH as M2_CONFLICT_PATH, _select_base_run  # noqa: E402

RUN_ROOT = ROOT / "experiments" / "runs" / "hydrocore-v5-m2"
SUMMARY_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m2-runs"

SAFETY_TASKS = frozenset({"event_presence", "evidence_sufficiency", "sensor_fault"})


def _reweighted_task_weights(base_weights: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    conflict = json.loads(M2_CONFLICT_PATH.read_text())
    downweighted = []
    negative_pairs: set[str] = set()
    for bucket in conflict["per_depth_bucket"].values():
        negative_pairs.update(bucket["frequent_negative_conflict_pairs"])
    weights = dict(base_weights)
    for pair in sorted(negative_pairs):
        primary, other = pair.split("|", 1)
        if primary != "source_node" or other in SAFETY_TASKS or other not in weights:
            continue
        weights[other] = weights[other] * 0.5
        downweighted.append(other)
    return weights, downweighted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("B", "C"))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    assert not locked_test_opened(ROOT), "locked test must remain closed"

    base_run = _select_base_run()
    winning_arm_letter = base_run["_selected_as"]
    depth_policy = ARM_POLICIES[winning_arm_letter]

    config = TrainingConfig.from_yaml(str(CONFIG_PATH), require_complete_task_weights=True)
    config = replace(config, seed=args.seed, gradnorm_log_every_n_batches=GRADNORM_LOG_EVERY_N_BATCHES)

    use_adapters = False
    downweighted: list[str] = []
    if args.arm == "B":
        weights, downweighted = _reweighted_task_weights(config.task_weights)
        config = replace(config, task_weights=weights)
    elif args.arm == "C":
        use_adapters = True

    set_deterministic_seed(config.seed, deterministic=config.deterministic)

    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    validation_records = build_scenario_pool("validation", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    train_view = CausalPrefixDatasetView(
        train_records, expected_split="train", signature_library=library,
        depth_policy=depth_policy, base_seed=config.seed, batch_size=config.batch_size,
    )
    validation_view = CausalPrefixDatasetView(
        validation_records, expected_split="validation", signature_library=library,
        depth_policy=full_history_policy, base_seed=config.seed, batch_size=config.batch_size,
    )

    model = HydroCore.from_variant("small", use_adapters=use_adapters, **SHARED_MODEL_CONFIG)
    param_count = sum(p.numel() for p in model.parameters())

    run_root = RUN_ROOT / f"{args.arm}-seed{args.seed}"
    started = time.time()
    trainer = Trainer(model, train_view, config=config, run_root=run_root, validation_dataset=validation_view)
    summary = trainer.fit()
    wall_seconds = time.time() - started

    record = {
        "schema_version": 1,
        "purpose": "Milestone 2.2 (experiments.txt): multitask objective-design ablation arm.",
        "arm": args.arm,
        "seed": args.seed,
        "arm_description": {
            "B": "primary-runtime focused (reweighted, safety tasks preserved)",
            "C": "role-isolated / adapters",
        }[args.arm],
        "based_on_m1_winning_depth_policy": winning_arm_letter,
        "use_adapters": use_adapters,
        "downweighted_tasks": downweighted,
        "model_architecture": {"variant": "small", "use_adapters": use_adapters, **SHARED_MODEL_CONFIG, "param_count": param_count},
        "training_config": asdict(config),
        "train_manifest_hash": train_view.manifest_hash,
        "validation_manifest_hash": validation_view.manifest_hash,
        "wall_seconds": wall_seconds,
        "training_summary": asdict(summary),
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    (SUMMARY_ROOT / f"{args.arm}-seed{args.seed}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"arm": args.arm, "seed": args.seed, "downweighted": downweighted, "wall_seconds": wall_seconds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
