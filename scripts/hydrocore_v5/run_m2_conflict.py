"""Milestone 2.1 (experiments.txt): multitask gradient-conflict measurement
on the best Milestone-1 causal model/config, grouped by causal depth.

Scope note (see hydroswarm.training.causal_prefix module docstring and
reports/evaluation/hydrocore-v5/m1-prefix-dataset.json): the Milestone-1
corpus supervises the Sentinel task family only (source_node, source_region,
start_time, duration, relative_strength, event_presence, event_cause,
evidence_sufficiency, sensor_fault). Scout/Strategist/OOD tasks
(sample_node, information_gain, candidate_reduction, plan_validity,
ood_class, ...) never receive a real target in this corpus, so their
measured "conflict" here is a real, graph-connected zero (per
MultiTaskLoss's own contract) -- reported honestly as
DEGENERATE_NO_SUPERVISION rather than a genuine interference finding. A
real Scout/Strategist/OOD interference study needs a jointly-supervised
corpus, which is out of this session's scope and is recorded as a deferred
negative/limitation, not silently omitted.

Writes:
  reports/evaluation/hydrocore-v5/m2-conflict.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    SUPERVISED_TASKS,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
)
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.data import collate_scenarios  # noqa: E402
from hydroswarm.training.losses import (  # noqa: E402
    PRIMARY_TASKS,
    compute_multitask_loss,
    task_gradient_conflict,
    task_gradient_norms,
)
from safetensors.torch import load_file  # noqa: E402
from run_m1_arm import CONFIG_PATH, SHARED_MODEL_CONFIG  # noqa: E402

M1_RESULTS_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-training-results.json"
M1_RUNS_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-runs"
OUTPUT_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m2-conflict.json"

DEPTH_BUCKETS: dict[str, tuple[int, ...]] = {
    "EARLY": (1, 2, 3),
    "MID": (4, 6),
    "MATURE": (12, 25),
}
BATCH_SIZE = 4
SCREENING_SEED = 31874


def _select_base_run() -> dict[str, Any]:
    if not M1_RESULTS_PATH.exists():
        raise SystemExit("Milestone 1 results not found; run evaluate_m1_depths.py first")
    m1_results = json.loads(M1_RESULTS_PATH.read_text())
    decision = m1_results["promotion_result"]["decision"]
    best_arm = decision.split("PROVISIONAL_WINNER_")[-1] if decision.startswith("PROVISIONAL_WINNER_") else "A"
    run_path = M1_RUNS_ROOT / f"{best_arm}-seed{SCREENING_SEED}.json"
    if not run_path.exists():
        raise SystemExit(f"expected Milestone-1 run record not found: {run_path}")
    run = json.loads(run_path.read_text())
    run["_m1_decision"] = decision
    run["_selected_as"] = best_arm
    return run


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"

    base_run = _select_base_run()
    export_path = base_run["training_summary"]["export_path"]
    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()

    config = TrainingConfig.from_yaml(str(CONFIG_PATH), require_complete_task_weights=True)
    task_weights = config.task_weights

    dev_records = build_scenario_pool("development_holdout", network_loader=build_wntr_network)
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    per_depth: dict[int, dict[str, Any]] = {}
    for depth in CAUSAL_PREFIX_DEPTHS:
        examples = [
            scenario_to_prefix_example(r.scenario, r.network, library, depth, feature_context=r.feature_context)
            for r in dev_records
        ]
        raw = defaultdict(list)
        weighted = defaultdict(list)
        valid_counts = defaultdict(list)
        grad_norms = defaultdict(list)
        conflict = defaultdict(list)
        for start in range(0, len(examples), BATCH_SIZE):
            batch = examples[start : start + BATCH_SIZE]
            inputs, targets = collate_scenarios(batch)
            output = model(inputs)
            result = compute_multitask_loss(output, targets, task_weights=task_weights)
            for name, loss in result.tasks.items():
                raw[name].append(float(loss.detach()))
                weighted[name].append(float(result.weighted[name].detach()))
                valid_counts[name].append(int(result.valid_counts[name]))
            for name, norm in task_gradient_norms(result.tasks, model).items():
                grad_norms[name].append(norm)
            for pair, cosine in task_gradient_conflict(result.tasks, model).items():
                conflict[pair].append(cosine)

        per_depth[depth] = {
            "n_scenarios": len(examples),
            "per_task": {
                name: {
                    "mean_raw_loss": statistics.fmean(raw[name]),
                    "mean_weighted_contribution": statistics.fmean(weighted[name]),
                    "total_valid_targets": sum(valid_counts[name]),
                    "mean_gradient_norm": statistics.fmean(grad_norms[name]) if name in grad_norms else None,
                    "supervised_in_this_corpus": name in SUPERVISED_TASKS,
                }
                for name in raw
            },
            "primary_vs_other_cosine": {
                pair: {
                    "mean": statistics.fmean(values),
                    "min": min(values),
                    "negative_fraction": statistics.fmean(1.0 if v < 0 else 0.0 for v in values),
                }
                for pair, values in conflict.items()
            },
        }

    bucketed: dict[str, Any] = {}
    for bucket_name, depths in DEPTH_BUCKETS.items():
        pairs_at_depths = [per_depth[d]["primary_vs_other_cosine"] for d in depths if d in per_depth]
        all_pair_names = sorted({pair for entry in pairs_at_depths for pair in entry})
        bucketed[bucket_name] = {
            "depths": list(depths),
            "primary_vs_other_cosine": {
                pair: statistics.fmean(entry[pair]["mean"] for entry in pairs_at_depths if pair in entry)
                for pair in all_pair_names
            },
            "frequent_negative_conflict_pairs": sorted(
                pair
                for pair in all_pair_names
                if statistics.fmean(entry[pair]["negative_fraction"] for entry in pairs_at_depths if pair in entry) >= 0.5
            ),
        }

    supervised_primary = sorted(PRIMARY_TASKS & SUPERVISED_TASKS)
    unsupervised_primary = sorted(PRIMARY_TASKS - SUPERVISED_TASKS)

    report = {
        "schema_version": 1,
        "purpose": "Milestone 2.1 (experiments.txt): per-task loss/gradient/conflict diagnostics grouped by causal depth.",
        "base_model": {
            "arm": base_run["_selected_as"],
            "seed": base_run["seed"],
            "m1_decision": base_run["_m1_decision"],
            "export_path": export_path,
        },
        "task_weights_source": str(CONFIG_PATH.relative_to(ROOT)),
        "supervised_tasks_in_this_corpus": sorted(SUPERVISED_TASKS),
        "primary_tasks_with_real_supervision": supervised_primary,
        "primary_tasks_without_supervision_this_session": unsupervised_primary,
        "scope_limitation": (
            "Scout/Strategist/OOD primary tasks ({}) never receive a real target in this corpus and show "
            "DEGENERATE_NO_SUPERVISION (graph-connected zero loss/gradient) below, not a genuine conflict "
            "measurement -- a real interference study for those tasks needs a jointly-supervised corpus, "
            "deferred beyond this session, reported as a negative/limitation finding rather than omitted."
        ).format(", ".join(unsupervised_primary) or "none"),
        "per_depth": {str(k): v for k, v in per_depth.items()},
        "per_depth_bucket": bucketed,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(bucketed, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
