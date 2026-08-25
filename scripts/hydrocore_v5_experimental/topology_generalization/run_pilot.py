"""Controlled pilot: does topology-relative feature augmentation
(`hydroswarm.model.topology_normalization`) improve HydroCore-v5's
generalization to unseen hydraulic-network topologies?

EXPERIMENTAL / NON-RELEASE. See
docs/evaluation/experimental/TOPOLOGY_GENERALIZATION_EXPERIMENT_PLAN.md for
the hypothesis, prior art already tried in this repo, and why this script's
scope is deliberately smaller than the historical M9.6 campaign (a
same-session, same-compute-budget, paired CONTROL vs. EXPERIMENTAL
comparison, not a reproduction of the full 3-seed/20-epoch campaign). Does
not open, read, or write anything under data/locked/ or any m9-*/m10-*/
m11-* report path. Trains fresh models; never loads or fine-tunes the
frozen models/hydrocore-v5-release checkpoint.

Usage: .venv/bin/python scripts/hydrocore_v5_experimental/topology_generalization/run_pilot.py
"""

from __future__ import annotations

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

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator  # noqa: E402
from hydroswarm.inference.ood import OODDetector, OODReference  # noqa: E402
from hydroswarm.model.core import HydroCore  # noqa: E402
from hydroswarm.model.topology_normalization import (  # noqa: E402
    EDGE_TOPOLOGY_RELATIVE_COLUMNS,
    NODE_TOPOLOGY_RELATIVE_COLUMNS,
    augment_batch,
    augmented_width,
)
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.data import ScenarioExample  # noqa: E402
from hydroswarm.training.sharded_data import ShardedScenarioDataset  # noqa: E402
from hydroswarm.training.trainer import Trainer  # noqa: E402
from hydroswarm.training.variable_collate import collate_variable_topology  # noqa: E402
from hydroswarm.classical.metrics import (  # noqa: E402
    localization_top_k,
    mean_reciprocal_rank,
)

CORPUS_ROOT = ROOT / "data" / "learning-v2" / "cycle-b2" / "tensors-normalized"
TRAINED_FAMILIES = ("golden-reference", "branched-loop", "loop-grid")
UNSEEN_FAMILY = "coastal-branch"  # dataset-report.json: development_ood_topology

SEED = 20260814  # matches the shipped v0.2.1 finalist seed (docs/MODEL_CARD.md)
TRAIN_PER_FAMILY = 200  # matches M9.6's own per-family scenario count
PILOT_EPOCHS = 6  # reduced from the production recipe's 20 epochs -- see plan doc Section 4
EVAL_VALIDATION_LIMIT = 300
EVAL_DEV_HOLDOUT_LIMIT = 300
ALPHA = 0.1  # matches the frozen release's calibration alpha (docs/MODEL_CARD.md)
MINIMUM_GROUP_SIZE = 10

RUN_ROOT = ROOT / "experiments" / "topology-generalization" / "runs"
RESULTS_DIR = ROOT / "reports" / "evaluation" / "topology-generalization"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def has_real_source(dataset: ShardedScenarioDataset, index: int) -> bool:
    """True if this example's `source_node` target is a real, unmasked
    localization label (targets_v2's own convention: NORMAL/SENSOR_FAULT_ONLY
    scenarios write `source_node=0` with `source_node_mask=False` as an
    explicit placeholder, per hydroswarm.training.losses' own documentation
    -- never a real label to localize against).

    This pilot's scope (Section 5 of the experiment plan) is source
    localization; restricting every split to examples that actually carry a
    real source keeps CONTROL and EXPERIMENTAL comparable and, incidentally,
    avoids a pre-existing, reproduces-with-or-without-this-experiment's-own
    changes numerical edge case in `hydroswarm.training.losses._cross_entropy`'s
    all-invalid-in-batch fallback (`logits.sum() * 0.0` overflows to NaN when
    every candidate logit in the batch is the model's own large masked-out
    sentinel value) -- noted in the final report as an incidental finding,
    not fixed here since it is unrelated to this experiment's own change and
    out of this branch's scope.
    """

    example = dataset[index]
    mask = example.targets.get("source_node_mask")
    return bool(mask is not None and bool(mask.any()))


def stratified_indices(dataset: ShardedScenarioDataset, *, per_family: int, families: tuple[str, ...], seed: int) -> list[int]:
    """Deterministic, seeded, family-balanced subsample restricted to
    real-source examples (see `has_real_source`). Reused identically for
    both arms so CONTROL and EXPERIMENTAL train on the exact same physical
    examples, differing only in model representation."""

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
    """Deterministic, seeded subsample restricted to real-source examples
    (see `has_real_source`), capped at `limit`."""

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


def build_model(*, augmented: bool, seed: int) -> HydroCore:
    _set_seed(seed)
    node_dim = augmented_width(19, NODE_TOPOLOGY_RELATIVE_COLUMNS) if augmented else 19
    edge_dim = augmented_width(13, EDGE_TOPOLOGY_RELATIVE_COLUMNS) if augmented else 13
    # event_control_heads=True matches the frozen release's own model_config
    # (models/hydrocore-v5-release/runtime_manifest.json) so this pilot can
    # report event/evidence-head metrics per the task's evaluation
    # checklist; HydroCore.from_variant defaults it False otherwise.
    return HydroCore.from_variant(
        "small", node_feature_dim=node_dim, edge_feature_dim=edge_dim, event_control_heads=True
    )


def make_collate_fn(*, augmented: bool):
    def _collate(examples):
        inputs, targets = collate_variable_topology(examples)
        if augmented:
            inputs = augment_batch(inputs)
        return inputs, targets

    return _collate


def train_arm(*, name: str, augmented: bool, train_dataset: ShardedScenarioDataset, validation_dataset: ShardedScenarioDataset) -> tuple[HydroCore, dict[str, Any]]:
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
        checkpoint_every_epochs=PILOT_EPOCHS,  # only need the final checkpoint for this pilot
        early_stopping_patience=0,  # match M9.6: always complete every epoch, no best-val reload
        maximum_runtime_seconds=3600.0,
        device="cpu",
        fp32=True,
        deterministic=True,
        task_weights=cfg_dict["task_weights"],
    )
    model = build_model(augmented=augmented, seed=SEED)
    run_root = RUN_ROOT / name
    started = time.monotonic()
    trainer = Trainer(
        model,
        train_dataset,
        config=config,
        run_root=run_root,
        validation_dataset=validation_dataset,
        collate_fn=make_collate_fn(augmented=augmented),
    )
    summary = trainer.fit()
    elapsed = time.monotonic() - started
    return model, {
        "arm": name,
        "augmented": augmented,
        "elapsed_seconds": elapsed,
        "epochs_completed": summary.epochs_completed,
        "stopped_early": summary.stopped_early,
        "stop_reason": summary.stop_reason,
        "global_steps": summary.global_steps,
        "best_validation_loss": summary.best_validation_loss,
    }


def _row_metrics(model: HydroCore, example: ScenarioExample, *, augmented: bool, ood_detector: OODDetector) -> dict[str, Any]:
    inputs, targets = collate_variable_topology([example])
    if augmented:
        inputs = augment_batch(inputs)
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
    if "event_presence_logits" in output and "event_presence" in targets:
        pooled_logit = output["event_presence_logits"][0]
        predicted = int(torch.argmax(pooled_logit).item()) if pooled_logit.numel() > 1 else int((pooled_logit > 0).item())
        row["event_presence_true"] = int(targets["event_presence"][0].item())
        row["event_presence_correct"] = predicted == row["event_presence_true"]
    return row


def evaluate_arm(model: HydroCore, *, name: str, augmented: bool, datasets: dict[str, tuple[ShardedScenarioDataset, list[int]]]) -> dict[str, Any]:
    ood_detector = OODDetector(OODReference(validated_network_hashes=()))  # filled in per-population below via network_id
    # Real topology-hash validity is per-example randomized-hydraulics
    # (network_hash) but topology FAMILY validity (what calibration.json's
    # validated_topology_hashes actually governs) is the structural
    # topology_hash -- computed once per family from this arm's own train
    # rows below, exactly mirroring how the real calibration artifact
    # declares validated_topology_hashes from the topologies present in its
    # own training/calibration data (conformal.py's `topology_hashes` fit
    # argument).
    rows_by_population: dict[str, list[dict[str, Any]]] = {}
    train_topology_hashes: set[str] = set()
    calibration_examples: list[CalibrationExample] = []
    for population, (dataset, indices) in datasets.items():
        rows = []
        for index in indices:
            example = dataset[index]
            row = _row_metrics(model, example, augmented=augmented, ood_detector=ood_detector)
            rows.append(row)
            if population == "train" and row["topology_hash"]:
                train_topology_hashes.add(row["topology_hash"])
        rows_by_population[population] = rows

    ood_detector = OODDetector(OODReference(validated_network_hashes=tuple(sorted(train_topology_hashes))))
    for population, rows in rows_by_population.items():
        for row in rows:
            row["ood_level"] = ood_detector.topology_level(
                node_count=row["node_count"], network_hash=row["topology_hash"]
            ).name

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
            model_hash=f"pilot-{name}",
            feature_schema_hash="pilot",
            dataset_manifest_hash="pilot",
            minimum_group_size=MINIMUM_GROUP_SIZE,
            topology_hashes=tuple(sorted(train_topology_hashes)),
        )
        calibration_diagnostics.update(
            coverage=calibrator.artifact.report.coverage,
            mean_set_size=calibrator.artifact.report.mean_set_size,
            expected_calibration_error=calibrator.artifact.report.expected_calibration_error,
        )

    summary: dict[str, Any] = {
        "arm": name,
        "augmented": augmented,
        "train_topology_hashes": sorted(train_topology_hashes),
        "calibration": calibration_diagnostics,
        "populations": {},
    }
    for population in ("validation", "development_holdout", "ood-UNSEEN_TOPOLOGY"):
        rows = rows_by_population.get(population, [])
        localized = [row for row in rows if row.get("has_source") and row.get("true_index") is not None]
        top1 = statistics.fmean(row["top1"] for row in localized) if localized else None
        top3 = statistics.fmean(row["top3"] for row in localized) if localized else None
        mrr = statistics.fmean(row["reciprocal_rank"] for row in localized) if localized else None
        by_family: dict[str, dict[str, Any]] = {}
        for network_id in sorted({row["network_id"] for row in rows}):
            family_localized = [row for row in localized if row["network_id"] == network_id]
            by_family[network_id] = {
                "n": len(family_localized),
                "top1": statistics.fmean(row["top1"] for row in family_localized) if family_localized else None,
                "top3": statistics.fmean(row["top3"] for row in family_localized) if family_localized else None,
                "mrr": statistics.fmean(row["reciprocal_rank"] for row in family_localized) if family_localized else None,
            }
        proxy_actionable = 0
        proxy_abstained = 0
        candidate_sizes: list[int] = []
        coverage_hits: list[bool] = []
        for row in localized:
            topology_known = row["topology_hash"] in train_topology_hashes
            ood_level = row["ood_level"]
            if calibrator is None or not topology_known:
                proxy_abstained += 1
                continue
            ordered_keys = sorted(row["probabilities"])
            probs = [row["probabilities"][key] for key in ordered_keys]
            candidate_positions = calibrator.candidate_set(
                probs,
                condition=row["stage"],
                network_id=row["network_id"],
                ood_level="OUTSIDE_VALIDATED_RANGE" if ood_level == "OUTSIDE_VALIDATED_RANGE" else "NORMAL",
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
            "by_family": by_family,
            "known_family_fraction": sum(1 for row in localized if row["topology_hash"] in train_topology_hashes) / n_localized,
            "proxy_actionable_rate": proxy_actionable / n_localized,
            "proxy_abstention_rate": proxy_abstained / n_localized,
            "proxy_candidate_set_size": statistics.fmean(candidate_sizes) if candidate_sizes else None,
            "proxy_calibrated_coverage": statistics.fmean(coverage_hits) if coverage_hits else None,
            "ood_caution_or_outside_rate": statistics.fmean(row["ood_level"] != "NORMAL" for row in rows) if rows else None,
        }
    return summary


def main() -> None:
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

    results: dict[str, Any] = {"seed": SEED, "pilot_epochs": PILOT_EPOCHS, "train_per_family": TRAIN_PER_FAMILY, "arms": {}}

    for name, augmented in (("CONTROL", False), ("EXPERIMENTAL_TOPOLOGY_RELATIVE", True)):
        print(f"\n=== Training arm {name} (augmented={augmented}) ===")
        model, train_summary = train_arm(name=name, augmented=augmented, train_dataset=train_ds, validation_dataset=validation_ds)
        print(f"  trained in {train_summary['elapsed_seconds']:.1f}s, epochs={train_summary['epochs_completed']}, stop={train_summary['stop_reason']}")

        print(f"  Evaluating arm {name}...")
        eval_datasets = {
            "train": (train_ds, list(range(len(train_ds)))),
            "validation": (validation_ds, list(range(len(validation_ds)))),
            "calibration": (calibration_full, list(range(len(calibration_full)))),
            "development_holdout": (dev_holdout_full, dev_holdout_indices),
            "ood-UNSEEN_TOPOLOGY": (ood_full, list(range(len(ood_full)))),
        }
        eval_summary = evaluate_arm(model, name=name, augmented=augmented, datasets=eval_datasets)
        results["arms"][name] = {"training": train_summary, "evaluation": eval_summary}
        (RESULTS_DIR / f"{name.lower()}-evaluation.json").write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    results_path = RESULTS_DIR / "pilot-results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
