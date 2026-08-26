"""Graph-structural-encoder-v2 (EXPERIMENTAL, NON-RELEASE): controlled,
paired-example ablation of GraphStructuralEncoder's structural/observability
features and edge_index aggregation.

See docs/evaluation/experimental/GRAPH_STRUCTURAL_ENCODER_V2_PLAN.md for the
full plan this implements (bottleneck, arms, splits, leakage controls,
success criteria). Harness structure (stratified family sampling,
OODDetector/SplitConformalCalibrator reuse, proxy actionable/abstention
metrics, per-row logging) follows `exp/failure-mode-diagnostics`'s
`run_pilot.py`/`rerun_topology_pilot_with_logging.py` convention (read-only
reference on that branch, reimplemented here rather than imported, since
this branch does not merge that one -- see plan doc Section 0). Does not
open `data/locked/`, does not load/fine-tune `models/hydrocore-v5-release`,
trains fresh small-variant models only. No governance module
(`hydroswarm.inference.ood`, `hydroswarm.calibration.conformal`, any
actionability gate) is modified -- called exactly as `run_pilot.py` already
calls them.

Usage: python3 scripts/hydrocore_v5_experimental/graph_structural_encoder_v2/run_experiment.py [--arms A_CONTROL,B_CENTRALITY,...]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental"))

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator  # noqa: E402
from hydroswarm.classical.metrics import entropy as shannon_entropy  # noqa: E402
from hydroswarm.classical.metrics import _ranked  # noqa: E402
from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.inference.ood import OODDetector, OODReference  # noqa: E402
from hydroswarm.model.core import HydroCore  # noqa: E402
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.data import ScenarioExample  # noqa: E402
from hydroswarm.training.sharded_data import ShardedScenarioDataset  # noqa: E402
from hydroswarm.training.trainer import Trainer  # noqa: E402
from hydroswarm.training.variable_collate import collate_variable_topology  # noqa: E402

from graph_structural_encoder_v2 import observability_features as obs_feat  # noqa: E402
from graph_structural_encoder_v2 import structural_features as struct_feat  # noqa: E402

CORPUS_ROOT = ROOT / "data" / "learning-v2" / "cycle-b2" / "tensors-normalized"
TRAINED_FAMILIES = ("golden-reference", "branched-loop", "loop-grid")
UNSEEN_FAMILY = "coastal-branch"  # dataset-report.json: development_ood_topology

#: Matches the shipped v0.2.1 finalist seed (docs/MODEL_CARD.md) and
#: `exp/failure-mode-diagnostics`'s own pilot re-run -- reused verbatim so
#: every arm here trains on the exact same physical examples as that
#: branch's own paired analysis, differing only in architecture/features.
SEED = 20260814
TRAIN_PER_FAMILY = 200
PILOT_EPOCHS = 6
EVAL_VALIDATION_LIMIT = 300
EVAL_DEV_HOLDOUT_LIMIT = 300
ALPHA = 0.1
MINIMUM_GROUP_SIZE = 10

RUN_ROOT = ROOT / "experiments" / "graph-structural-encoder-v2" / "runs"
RESULTS_DIR = ROOT / "reports" / "evaluation" / "graph-structural-encoder-v2"

STRUCT_WIDTH = len(struct_feat.NODE_STRUCTURAL_COLUMNS)
OBS_WIDTH = len(obs_feat.NODE_OBSERVABILITY_COLUMNS)

#: Arm definitions (plan doc Section 3). `feature_mode` selects which batch
#: augmentation `augment_batch` applies; `model_kwargs` are passed straight
#: through to HydroCore.from_variant on top of the shared base config.
ARMS: dict[str, dict[str, Any]] = {
    "A_CONTROL": {"feature_mode": "none", "model_kwargs": {}},
    "B_CENTRALITY": {
        "feature_mode": "centrality",
        "model_kwargs": {"graph_structural_feature_dim": STRUCT_WIDTH},
    },
    "C_OBSERVABILITY": {
        "feature_mode": "observability",
        "model_kwargs": {"graph_structural_feature_dim": OBS_WIDTH},
    },
    "D_STRUCTURAL_AGG": {
        "feature_mode": "none",
        "model_kwargs": {"graph_structural_edge_aggregation": True},
    },
    "D_CAPACITY_CONTROL": {
        "feature_mode": "none",
        "model_kwargs": {
            "graph_structural_edge_aggregation": True,
            "graph_structural_edge_aggregation_source": "self_only",
        },
    },
    "E_COMBINED": {
        "feature_mode": "combined",
        "model_kwargs": {
            "graph_structural_feature_dim": STRUCT_WIDTH + OBS_WIDTH,
            "graph_structural_edge_aggregation": True,
        },
    },
}
#: Compute-conscious fallback order (plan doc Section 3) if wall-clock forces
#: a cut -- not applied unless main() is invoked with --arms explicitly
#: naming a subset.
PRIORITY_ORDER = ("A_CONTROL", "B_CENTRALITY", "C_OBSERVABILITY", "E_COMBINED", "D_STRUCTURAL_AGG", "D_CAPACITY_CONTROL")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def has_real_source(dataset: ShardedScenarioDataset, index: int) -> bool:
    """Same real-source filter as `exp/failure-mode-diagnostics`'s
    `run_pilot.py::has_real_source` -- NORMAL/SENSOR_FAULT_ONLY scenarios
    write `source_node=0`/`source_node_mask=False` as an explicit
    placeholder (`hydroswarm.training.losses`' own documented convention),
    never a real localization label."""

    example = dataset[index]
    mask = example.targets.get("source_node_mask")
    return bool(mask is not None and bool(mask.any()))


def stratified_indices(dataset: ShardedScenarioDataset, *, per_family: int, families: tuple[str, ...], seed: int) -> list[int]:
    by_family: dict[str, list[int]] = {family: [] for family in families}
    for index, entry in enumerate(dataset._entries):
        if entry.network_id in by_family:
            by_family[entry.network_id].append(index)
    rng = random.Random(seed)
    selected: list[int] = []
    for family in families:
        pool = list(by_family[family])
        rng.shuffle(pool)
        chosen = [index for index in pool if has_real_source(dataset, index)][:per_family]
        if len(chosen) < per_family:
            raise ValueError(f"family {family!r} has only {len(chosen)} real-source examples, need {per_family}")
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def capped_indices(dataset: ShardedScenarioDataset, *, limit: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)
    selected: list[int] = []
    for index in order:
        if has_real_source(dataset, index):
            selected.append(index)
        if len(selected) >= limit:
            break
    return sorted(selected)


def augment_batch(inputs: dict[str, torch.Tensor], *, feature_mode: str) -> dict[str, torch.Tensor]:
    """Attach `structural_features` per `feature_mode`. `node_mask`/
    `edge_index`/`edge_mask`/`source_candidate_mask`/`sensor_mask` are all
    already present on `inputs` from `collate_variable_topology` -- nothing
    here reads a target/label tensor (plan doc Section 5)."""

    if feature_mode == "none":
        return inputs
    augmented = dict(inputs)
    node_mask = inputs["node_mask"]
    edge_index = inputs.get("edge_index")
    edge_mask = inputs.get("edge_mask")
    if feature_mode == "centrality":
        augmented["structural_features"] = struct_feat.compute_structural_features(
            node_mask, edge_index, edge_mask, inputs.get("source_candidate_mask")
        )
    elif feature_mode == "observability":
        augmented["structural_features"] = obs_feat.compute_observability_features(
            node_mask, edge_index, edge_mask, inputs.get("sensor_mask")
        )
    elif feature_mode == "combined":
        centrality = struct_feat.compute_structural_features(
            node_mask, edge_index, edge_mask, inputs.get("source_candidate_mask")
        )
        observability = obs_feat.compute_observability_features(
            node_mask, edge_index, edge_mask, inputs.get("sensor_mask")
        )
        augmented["structural_features"] = torch.cat((centrality, observability), dim=-1)
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode!r}")
    return augmented


def make_collate_fn(*, feature_mode: str):
    def _collate(examples):
        inputs, targets = collate_variable_topology(examples)
        inputs = augment_batch(inputs, feature_mode=feature_mode)
        return inputs, targets

    return _collate


def build_model(*, arm_name: str, seed: int) -> HydroCore:
    _set_seed(seed)
    model_kwargs = dict(ARMS[arm_name]["model_kwargs"])
    # event_control_heads=True matches the frozen release's own model_config
    # so this experiment can report event/evidence-head diagnostics, mirroring
    # run_pilot.py's own build_model().
    return HydroCore.from_variant(
        "small", node_feature_dim=19, edge_feature_dim=13, event_control_heads=True, **model_kwargs
    )


def train_arm(
    *, arm_name: str, train_dataset: ShardedScenarioDataset, validation_dataset: ShardedScenarioDataset
) -> tuple[HydroCore, dict[str, Any]]:
    cfg_dict = yaml.safe_load((ROOT / "configs" / "training-v5-causal.yaml").read_text())["training"]
    config = TrainingConfig(
        seed=SEED,
        epochs=PILOT_EPOCHS,
        batch_size=cfg_dict["batch_size"],
        gradient_accumulation_steps=cfg_dict["gradient_accumulation_steps"],
        learning_rate=cfg_dict["learning_rate"],
        weight_decay=cfg_dict["weight_decay"],
        gradient_clip_norm=cfg_dict["gradient_clip_norm"],
        warmup_steps=cfg_dict["warmup_steps"],
        scheduler=cfg_dict["scheduler"],
        checkpoint_every_epochs=PILOT_EPOCHS,
        early_stopping_patience=0,
        maximum_runtime_seconds=3600.0,
        device="cpu",
        fp32=True,
        deterministic=True,
        task_weights=cfg_dict["task_weights"],
    )
    model = build_model(arm_name=arm_name, seed=SEED)
    run_root = RUN_ROOT / arm_name
    started = time.monotonic()
    trainer = Trainer(
        model,
        train_dataset,
        config=config,
        run_root=run_root,
        validation_dataset=validation_dataset,
        collate_fn=make_collate_fn(feature_mode=ARMS[arm_name]["feature_mode"]),
    )
    summary = trainer.fit()
    elapsed = time.monotonic() - started
    return model, {
        "arm": arm_name,
        "elapsed_seconds": elapsed,
        "epochs_completed": summary.epochs_completed,
        "stopped_early": summary.stopped_early,
        "stop_reason": summary.stop_reason,
        "global_steps": summary.global_steps,
        "best_validation_loss": summary.best_validation_loss,
    }


def _row_metrics(model: HydroCore, example: ScenarioExample, *, feature_mode: str, ood_detector: OODDetector) -> dict[str, Any]:
    inputs, targets = collate_variable_topology([example])
    inputs = augment_batch(inputs, feature_mode=feature_mode)
    model.eval()
    with torch.inference_mode():
        output = model(inputs)
    mask = inputs["source_candidate_mask"].bool()[0].numpy()
    probabilities = torch.softmax(output["source_node_logits"][0].float(), dim=-1).numpy()
    probabilities = probabilities * mask
    total = probabilities.sum()
    probabilities = probabilities / total if total > 0 else np.ones_like(probabilities) / max(1, mask.sum())
    has_source = bool(targets["source_node_mask"][0]) if "source_node_mask" in targets else False
    true_index = int(targets["source_node"][0].item()) if has_source else None
    prob_map = {int(index): float(value) for index, value in enumerate(probabilities) if mask[index]}
    node_count = int(inputs["node_mask"][0].sum().item())
    topology = example.topology
    ood_level = ood_detector.topology_level(node_count=node_count, network_hash=topology.topology_hash if topology else None)

    # Diagnostic-only structural/observability features (plan doc Section 6
    # subgroup requirements): computed for EVERY row regardless of which
    # features this arm's model actually consumes, purely for subgroup
    # labeling/reporting -- never fed back into the model here.
    struct_row = struct_feat.compute_structural_features(
        inputs["node_mask"], inputs.get("edge_index"), inputs.get("edge_mask"), inputs.get("source_candidate_mask")
    )[0]
    obs_row = obs_feat.compute_observability_features(
        inputs["node_mask"], inputs.get("edge_index"), inputs.get("edge_mask"), inputs.get("sensor_mask")
    )[0]
    struct_columns = {name: index for index, name in enumerate(struct_feat.NODE_STRUCTURAL_COLUMNS)}
    obs_columns = {name: index for index, name in enumerate(obs_feat.NODE_OBSERVABILITY_COLUMNS)}

    row: dict[str, Any] = {
        "scenario_id": example.scenario_id,
        "network_id": example.network_id,
        "stage": example.stage.name,
        "topology_hash": topology.topology_hash if topology else None,
        "node_count": node_count,
        "has_source": has_source,
        "true_index": true_index,
        "probabilities": prob_map,
        "ood_level": ood_level.name,
    }
    if has_source and true_index is not None and prob_map:
        row["top1"] = localization_top_k(prob_map, true_index, k=1)
        row["top3"] = localization_top_k(prob_map, true_index, k=min(3, len(prob_map)))
        row["reciprocal_rank"] = mean_reciprocal_rank([prob_map], [true_index])
        ranked = _ranked(prob_map)
        row["true_source_rank"] = (ranked.index(true_index) + 1) if true_index in ranked else None
        values = sorted(prob_map.values(), reverse=True)
        row["top1_probability"] = values[0] if values else None
        row["margin_top1_top2"] = (values[0] - values[1]) if len(values) >= 2 else None
        row["true_source_probability"] = prob_map.get(true_index)
        row["posterior_entropy_bits"] = shannon_entropy(list(prob_map.values()))
        row["n_candidates"] = len(prob_map)
        if true_index < struct_row.shape[0]:
            row["source_degree_normalized"] = float(struct_row[true_index, struct_columns["degree_normalized"]])
            row["source_betweenness_centrality"] = float(struct_row[true_index, struct_columns["betweenness_centrality"]])
            row["source_closeness_centrality"] = float(struct_row[true_index, struct_columns["closeness_centrality"]])
            row["source_normalized_graph_position"] = float(struct_row[true_index, struct_columns["normalized_graph_position"]])
            row["source_hop_to_dead_end_normalized"] = float(struct_row[true_index, struct_columns["hop_to_dead_end_normalized"]])
            row["source_hop_to_nearest_sensor_normalized"] = float(obs_row[true_index, obs_columns["hop_to_nearest_sensor_normalized"]])
            row["source_local_sensor_coverage_density"] = float(obs_row[true_index, obs_columns["local_sensor_coverage_density"]])
    if "event_presence_logits" in output and "event_presence" in targets:
        pooled_logit = output["event_presence_logits"][0]
        predicted = int(torch.argmax(pooled_logit).item()) if pooled_logit.numel() > 1 else int((pooled_logit > 0).item())
        row["event_presence_true"] = int(targets["event_presence"][0].item())
        row["event_presence_correct"] = predicted == row["event_presence_true"]
    return row


def evaluate_arm(
    model: HydroCore, *, arm_name: str, datasets: dict[str, tuple[ShardedScenarioDataset, list[int]]]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    feature_mode = ARMS[arm_name]["feature_mode"]
    ood_detector = OODDetector(OODReference(validated_network_hashes=()))
    rows_by_population: dict[str, list[dict[str, Any]]] = {}
    train_topology_hashes: set[str] = set()
    calibration_examples: list[CalibrationExample] = []
    for population, (dataset, indices) in datasets.items():
        rows = []
        for index in indices:
            example = dataset[index]
            row = _row_metrics(model, example, feature_mode=feature_mode, ood_detector=ood_detector)
            rows.append(row)
            if population == "train" and row["topology_hash"]:
                train_topology_hashes.add(row["topology_hash"])
        rows_by_population[population] = rows

    ood_detector = OODDetector(OODReference(validated_network_hashes=tuple(sorted(train_topology_hashes))))
    for rows in rows_by_population.values():
        for row in rows:
            row["ood_level"] = ood_detector.topology_level(
                node_count=row["node_count"], network_hash=row["topology_hash"]
            ).name
            row["topology_known"] = row["topology_hash"] in train_topology_hashes

    for row in rows_by_population.get("calibration", []):
        if row.get("has_source") and row.get("true_index") is not None and row.get("probabilities"):
            ordered_keys = sorted(row["probabilities"])
            calibration_examples.append(
                CalibrationExample(
                    probabilities=tuple(row["probabilities"][key] for key in ordered_keys),
                    true_index=ordered_keys.index(row["true_index"]),
                    condition=row["stage"],
                    network_id=row["network_id"],
                )
            )

    calibrator = None
    calibration_diagnostics: dict[str, Any] = {"n": len(calibration_examples)}
    if len(calibration_examples) >= MINIMUM_GROUP_SIZE:
        calibrator = SplitConformalCalibrator.fit(
            calibration_examples,
            alpha=ALPHA,
            model_hash=f"gse-v2-{arm_name}",
            feature_schema_hash="gse-v2",
            dataset_manifest_hash="gse-v2",
            minimum_group_size=MINIMUM_GROUP_SIZE,
            topology_hashes=tuple(sorted(train_topology_hashes)),
        )
        calibration_diagnostics.update(
            coverage=calibrator.artifact.report.coverage,
            mean_set_size=calibrator.artifact.report.mean_set_size,
            expected_calibration_error=calibrator.artifact.report.expected_calibration_error,
        )

    summary: dict[str, Any] = {
        "arm": arm_name,
        "feature_mode": feature_mode,
        "model_kwargs": ARMS[arm_name]["model_kwargs"],
        "train_topology_hashes": sorted(train_topology_hashes),
        "calibration": calibration_diagnostics,
        "hard_safety_counters": {
            name: 0
            for name in (
                "human_approval_bypassed",
                "invariant_failures",
                "nonfinite_value_reached_decision",
                "unverified_plan_surfaced_as_actionable",
                "rejected_plan_surfaced_as_safe",
                "sampled_node_reselected",
                "sampling_budget_exceeded",
                "inaccessible_sample_selected",
            )
        },
        "hard_safety_counters_note": (
            "This pilot-scale localization-only harness does not exercise the "
            "sampling/planning/execution control loop that produces these "
            "counters in the M11.6 evaluation tier; all are reported as 0 "
            "because the corresponding code paths are never invoked here, not "
            "because they were independently re-verified at that tier. No "
            "governance module is modified by this branch."
        ),
        "populations": {},
    }
    for population in ("validation", "development_holdout", "ood-UNSEEN_TOPOLOGY"):
        rows = rows_by_population.get(population, [])
        localized = [row for row in rows if row.get("has_source") and row.get("true_index") is not None]
        nonfinite_reached_decision = any(
            not all(np.isfinite(value) for value in row["probabilities"].values()) for row in localized
        )
        summary["hard_safety_counters"]["nonfinite_value_reached_decision"] += int(nonfinite_reached_decision)
        top1 = statistics.fmean(row["top1"] for row in localized) if localized else None
        top3 = statistics.fmean(row["top3"] for row in localized) if localized else None
        mrr = statistics.fmean(row["reciprocal_rank"] for row in localized) if localized else None
        proxy_actionable = 0
        proxy_abstained = 0
        candidate_sizes: list[int] = []
        coverage_hits: list[bool] = []
        for row in localized:
            if calibrator is None or not row["topology_known"]:
                proxy_abstained += 1
                continue
            ordered_keys = sorted(row["probabilities"])
            probs = [row["probabilities"][key] for key in ordered_keys]
            candidate_positions = calibrator.candidate_set(
                probs,
                condition=row["stage"],
                network_id=row["network_id"],
                ood_level="OUTSIDE_VALIDATED_RANGE" if row["ood_level"] == "OUTSIDE_VALIDATED_RANGE" else "NORMAL",
            )
            candidates = {ordered_keys[position] for position in candidate_positions}
            candidate_sizes.append(len(candidates))
            coverage_hits.append(row["true_index"] in candidates)
            if candidates:
                proxy_actionable += 1
            else:
                proxy_abstained += 1
        event_rows = [row for row in rows if "event_presence_correct" in row]
        event_accuracy = statistics.fmean(row["event_presence_correct"] for row in event_rows) if event_rows else None
        n_localized = len(localized) or 1
        summary["populations"][population] = {
            "n": len(rows),
            "n_localized": len(localized),
            "top1": top1,
            "top3": top3,
            "mrr": mrr,
            "event_presence_accuracy": event_accuracy,
            "known_family_fraction": sum(1 for row in localized if row["topology_known"]) / n_localized,
            "proxy_actionable_rate": proxy_actionable / n_localized,
            "proxy_abstention_rate": proxy_abstained / n_localized,
            "proxy_candidate_set_size": statistics.fmean(candidate_sizes) if candidate_sizes else None,
            "proxy_calibrated_coverage": statistics.fmean(coverage_hits) if coverage_hits else None,
            "ood_caution_or_outside_rate": statistics.fmean(row["ood_level"] != "NORMAL" for row in rows) if rows else None,
        }
    return summary, rows_by_population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=str, default=None, help="comma-separated arm names; default all")
    args = parser.parse_args()
    arm_names = args.arms.split(",") if args.arms else list(PRIORITY_ORDER)
    for name in arm_names:
        if name not in ARMS:
            raise SystemExit(f"unknown arm {name!r}; choices: {sorted(ARMS)}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets (already-generated, leakage-checked cycle-b2 corpus)...")
    train_full = ShardedScenarioDataset(CORPUS_ROOT / "train", expected_split="train")
    validation_full = ShardedScenarioDataset(CORPUS_ROOT / "validation", expected_split="validation")
    calibration_full = ShardedScenarioDataset(CORPUS_ROOT / "calibration", expected_split="calibration")
    dev_holdout_full = ShardedScenarioDataset(CORPUS_ROOT / "development_holdout", expected_split="development_holdout")
    ood_full = ShardedScenarioDataset(CORPUS_ROOT / "ood-UNSEEN_TOPOLOGY", expected_split="development_holdout")

    train_indices = stratified_indices(train_full, per_family=TRAIN_PER_FAMILY, families=TRAINED_FAMILIES, seed=SEED)
    train_ds = ShardedScenarioDataset(CORPUS_ROOT / "train", expected_split="train", indices=train_indices)
    validation_indices = capped_indices(validation_full, limit=EVAL_VALIDATION_LIMIT, seed=SEED)
    validation_ds = ShardedScenarioDataset(CORPUS_ROOT / "validation", expected_split="validation", indices=validation_indices)
    dev_holdout_indices = capped_indices(dev_holdout_full, limit=EVAL_DEV_HOLDOUT_LIMIT, seed=SEED)

    train_families_present = {train_full[index].network_id for index in train_indices}
    print(f"Train subsample: {len(train_indices)} examples across families {sorted(train_families_present)}")

    manifest = {
        "seed": SEED,
        "pilot_epochs": PILOT_EPOCHS,
        "train_per_family": TRAIN_PER_FAMILY,
        "train_indices_count": len(train_indices),
        "validation_indices_count": len(validation_indices),
        "development_holdout_indices_count": len(dev_holdout_indices),
        "ood_unseen_topology_count": len(ood_full),
        "calibration_count": len(calibration_full),
        "arms_run": arm_names,
    }
    (RESULTS_DIR / "run-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for arm_name in arm_names:
        print(f"\n=== Training arm {arm_name} ===")
        model, train_summary = train_arm(arm_name=arm_name, train_dataset=train_ds, validation_dataset=validation_ds)
        print(f"  trained in {train_summary['elapsed_seconds']:.1f}s, epochs={train_summary['epochs_completed']}, stop={train_summary['stop_reason']}")

        parameter_report = model.parameter_report_dict()
        print(f"  parameters: {parameter_report}")

        print(f"  Evaluating arm {arm_name}...")
        eval_datasets = {
            "train": (train_ds, list(range(len(train_ds)))),
            "validation": (validation_ds, list(range(len(validation_ds)))),
            "calibration": (calibration_full, list(range(len(calibration_full)))),
            "development_holdout": (dev_holdout_full, dev_holdout_indices),
            "ood-UNSEEN_TOPOLOGY": (ood_full, list(range(len(ood_full)))),
        }
        eval_summary, rows_by_population = evaluate_arm(model, arm_name=arm_name, datasets=eval_datasets)
        eval_summary["training"] = train_summary
        eval_summary["parameter_report"] = parameter_report
        (RESULTS_DIR / f"{arm_name.lower()}-evaluation.json").write_text(
            json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for population in ("validation", "development_holdout", "ood-UNSEEN_TOPOLOGY"):
            rows = rows_by_population.get(population, [])
            path = RESULTS_DIR / f"{arm_name.lower()}-{population}-rows.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"  wrote {arm_name.lower()}-evaluation.json and per-population row logs")

    print(f"\nAll requested arms complete. Results under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
