"""physics-informed-localizer-validation (EXPERIMENTAL, NON-RELEASE):
confirmation-and-ablation follow-up to `exp/candidate-conditioned-
localizer-v1`. That pilot found Arm C_PHYSICS_INFORMED significantly
improved unseen-topology Top-1 over A_CONTROL (+6.4pp, 90% CI
[+2.5, +10.0]pp) at a single seed, with an uncontrolled +4.6% parameter
delta and no physics-feature ablation. This script:

  1. reproduces that pilot's A_CONTROL / B_CANDIDATE_CONDITIONED /
     C_PHYSICS_INFORMED (here: C_FULL) result at the SAME seed (20260814),
     using the exact same corpus/splits/epochs/optimizer/config;
  2. adds A_CAPACITY_MATCHED, a generic-capacity control with ~the same
     parameter delta as B/C but no candidate conditioning, structure, or
     physics information (see `hydroswarm.model.core`'s
     `localizer_capacity_hidden_dim`);
  3. runs every arm across multiple pre-declared seeds;
  4. ablates C_FULL's three physics-compatibility columns individually
     (C1/C2/C3) and, budget permitting, pairwise (C1_C2/C1_C3/C2_C3) --
     by masking `physics_features.compute_physics_features`'s output
     columns, NOT by changing `CandidateConditionedLocalizer`'s
     architecture or parameter count (Phase 5's requirement: same model,
     same parameterization, only the physics-feature INPUT differs).

Reuses `candidate_conditioned_localizer_v1`'s `candidate_sensor_features.py`
and `physics_features.py` unmodified (imported, not reimplemented -- that
package is part of this branch's own history, unlike the pilot's own
reimplementation-not-import stance toward `exp/graph-structural-encoder-v2`,
since this experiment explicitly "follows" and extends that pilot rather
than being a parallel, independently-evolved branch). Harness structure
(stratified family sampling, OODDetector/SplitConformalCalibrator reuse,
proxy actionable/abstention metrics, per-row logging) is the pilot's own
`run_experiment.py`, generalized to take an explicit `--seed` and a wider
arm registry instead of a single fixed `SEED` constant and 3-arm dict.

Does not open `data/locked/`, does not load/fine-tune
`models/hydrocore-v5-release`, trains fresh small-variant models only. No
governance module is modified.

Usage:
  python3 scripts/hydrocore_v5_experimental/physics_informed_localizer_validation/run_experiment.py \
      --seed 20260814 --arms A_CONTROL,A_CAPACITY_MATCHED,B_CANDIDATE_CONDITIONED,C_FULL,C1,C2,C3
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

from candidate_conditioned_localizer_v1 import candidate_sensor_features as csf  # noqa: E402
from candidate_conditioned_localizer_v1 import physics_features as physf  # noqa: E402

CORPUS_ROOT = ROOT / "data" / "learning-v2" / "cycle-b2" / "tensors-normalized"
TRAINED_FAMILIES = ("golden-reference", "branched-loop", "loop-grid")
UNSEEN_FAMILY = "coastal-branch"

#: Phase 3 pre-registration: fixed seed list, declared before any training
#: in this run. 20260814 is the original candidate-conditioned-localizer-v1
#: pilot seed (continuity / the direct Phase 1 reproduction check); the
#: other two are disjoint dates chosen with no dependence on any run's
#: outcome. Three seeds (not five): each of the 7 priority arms x 3 seeds
#: is already 21 full training+evaluation runs at ~10-13 minutes each on
#: CPU (~4 hours total) -- see FINAL_REPORT.md Section "Compute budget" for
#: why 5 seeds and the optional pairwise ablation arms were treated as
#: budget-permitting rather than mandatory.
SEEDS: tuple[int, ...] = (20260814, 20260901, 20260915)

TRAIN_PER_FAMILY = 200
PILOT_EPOCHS = 6
EVAL_VALIDATION_LIMIT = 300
EVAL_DEV_HOLDOUT_LIMIT = 300
ALPHA = 0.1
MINIMUM_GROUP_SIZE = 10

#: Phase 2: A_CAPACITY_MATCHED's generic-capacity hidden width. Selected by
#: direct enumeration (see FINAL_REPORT.md's parameter-count table) as the
#: value whose resulting total parameter count is closest to Arm B/C's own
#: +4.6% delta over A_CONTROL without adding any structural/physics/
#: candidate information -- 4,231,223 total vs B's 4,231,129 and C's
#: 4,231,897 (within 700 parameters of both, an order of magnitude smaller
#: than the ~187,000-parameter delta itself).
CAPACITY_MATCHED_HIDDEN_DIM = 482

EXPERIMENT_NAME = "physics-informed-localizer-validation"
RUN_ROOT = ROOT / "experiments" / EXPERIMENT_NAME / "runs"
RESULTS_ROOT = ROOT / "reports" / "evaluation" / EXPERIMENT_NAME

STRUCT_WIDTH = len(csf.NODE_STRUCTURAL_COLUMNS)
PHYS_WIDTH = len(physf.PHYSICS_FEATURE_COLUMNS)
PHYSICS_COLUMNS = physf.PHYSICS_FEATURE_COLUMNS  # ("nearest_sensor_log_concentration", "hop_magnitude_compatibility", "hop_arrival_time_compatibility")

_CANDIDATE_CONDITIONED_BASE_KWARGS = {
    "localizer_mode": "candidate_conditioned",
    "localizer_structural_feature_dim": STRUCT_WIDTH,
}


def _physics_kwargs(dim: int) -> dict[str, Any]:
    return {**_CANDIDATE_CONDITIONED_BASE_KWARGS, "localizer_physics_feature_dim": dim}


#: Every C-family arm (C_FULL, C1, C2, C3, and the pairwise combinations)
#: shares IDENTICAL model_kwargs (physics_feature_dim=PHYS_WIDTH=3, same
#: CandidateConditionedLocalizer, same structural features) -- Phase 5's
#: requirement that ablation changes only the physics-feature INPUT, never
#: the architecture or parameter count. `physics_columns` selects which of
#: PHYSICS_COLUMNS stay nonzero; the rest are masked to exactly zero before
#: the (unchanged) physics_projection layer sees them.
ARMS: dict[str, dict[str, Any]] = {
    "A_CONTROL": {
        "localizer_mode": "default",
        "model_kwargs": {},
        "physics_columns": None,
    },
    "A_CAPACITY_MATCHED": {
        "localizer_mode": "default",
        "model_kwargs": {"localizer_capacity_hidden_dim": CAPACITY_MATCHED_HIDDEN_DIM},
        "physics_columns": None,
    },
    "B_CANDIDATE_CONDITIONED": {
        "localizer_mode": "candidate_conditioned",
        "model_kwargs": _physics_kwargs(0),
        "physics_columns": None,
    },
    "C_FULL": {
        "localizer_mode": "candidate_conditioned",
        "model_kwargs": _physics_kwargs(PHYS_WIDTH),
        "physics_columns": PHYSICS_COLUMNS,
    },
    "C1": {
        "localizer_mode": "candidate_conditioned",
        "model_kwargs": _physics_kwargs(PHYS_WIDTH),
        "physics_columns": ("nearest_sensor_log_concentration",),
    },
    "C2": {
        "localizer_mode": "candidate_conditioned",
        "model_kwargs": _physics_kwargs(PHYS_WIDTH),
        "physics_columns": ("hop_magnitude_compatibility",),
    },
    "C3": {
        "localizer_mode": "candidate_conditioned",
        "model_kwargs": _physics_kwargs(PHYS_WIDTH),
        "physics_columns": ("hop_arrival_time_compatibility",),
    },
    "C1_C2": {
        "localizer_mode": "candidate_conditioned",
        "model_kwargs": _physics_kwargs(PHYS_WIDTH),
        "physics_columns": ("nearest_sensor_log_concentration", "hop_magnitude_compatibility"),
    },
    "C1_C3": {
        "localizer_mode": "candidate_conditioned",
        "model_kwargs": _physics_kwargs(PHYS_WIDTH),
        "physics_columns": ("nearest_sensor_log_concentration", "hop_arrival_time_compatibility"),
    },
    "C2_C3": {
        "localizer_mode": "candidate_conditioned",
        "model_kwargs": _physics_kwargs(PHYS_WIDTH),
        "physics_columns": ("hop_magnitude_compatibility", "hop_arrival_time_compatibility"),
    },
}

#: Task's own stated priority order; pairwise arms are budget-permitting
#: (Phase 4) and deliberately excluded from PRIORITY_ORDER.
PRIORITY_ORDER: tuple[str, ...] = (
    "A_CONTROL",
    "A_CAPACITY_MATCHED",
    "B_CANDIDATE_CONDITIONED",
    "C_FULL",
    "C1",
    "C2",
    "C3",
)
PAIRWISE_ORDER: tuple[str, ...] = ("C1_C2", "C1_C3", "C2_C3")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def has_real_source(dataset: ShardedScenarioDataset, index: int) -> bool:
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


def _mask_physics_columns(features: torch.Tensor, physics_columns: tuple[str, ...] | None) -> torch.Tensor:
    """Zeroes every physics-feature column NOT in `physics_columns` (ablation:
    changes only the model's INPUT, never `physics_feature_dim` or any
    other architecture/parameter-count-affecting quantity -- Phase 5).
    `physics_columns=None` or the full column set is a no-op."""

    if physics_columns is None or set(physics_columns) == set(PHYSICS_COLUMNS):
        return features
    keep = torch.zeros(len(PHYSICS_COLUMNS), dtype=torch.bool)
    for name in physics_columns:
        keep[PHYSICS_COLUMNS.index(name)] = True
    return features * keep.to(dtype=features.dtype, device=features.device)


def augment_batch(
    inputs: dict[str, torch.Tensor], *, localizer_mode: str, with_physics: bool, physics_columns: tuple[str, ...] | None
) -> dict[str, torch.Tensor]:
    """Attaches the new HydroBatch fields CandidateConditionedLocalizer
    needs. `node_mask`/`edge_index`/`edge_mask`/`source_candidate_mask`/
    `sensor_mask`/`temporal_features`/`timestamps` are all already present
    on `inputs` from `collate_variable_topology` -- nothing here reads a
    target/label tensor."""

    if localizer_mode == "default":
        return inputs
    augmented = dict(inputs)
    node_mask = inputs["node_mask"]
    edge_index = inputs.get("edge_index")
    edge_mask = inputs.get("edge_mask")
    active_sensor = csf.active_sensor_mask_from_temporal(inputs.get("sensor_mask"), node_mask)
    hop_distance = csf.compute_hop_distance(node_mask, edge_index, edge_mask)
    augmented["active_sensor_mask_nodes"] = active_sensor
    augmented["candidate_hop_distance"] = hop_distance
    augmented["candidate_structural_features"] = csf.compute_structural_features(
        node_mask, edge_index, edge_mask, active_sensor, hop_distance
    )
    if with_physics:
        physics = physf.compute_physics_features(
            inputs["temporal_features"], hop_distance, active_sensor, node_mask, inputs.get("timestamps")
        )
        augmented["candidate_physics_features"] = _mask_physics_columns(physics, physics_columns)
    return augmented


def make_collate_fn(*, localizer_mode: str, with_physics: bool, physics_columns: tuple[str, ...] | None):
    def _collate(examples):
        inputs, targets = collate_variable_topology(examples)
        inputs = augment_batch(inputs, localizer_mode=localizer_mode, with_physics=with_physics, physics_columns=physics_columns)
        return inputs, targets

    return _collate


def build_model(*, arm_name: str, seed: int) -> HydroCore:
    _set_seed(seed)
    model_kwargs = dict(ARMS[arm_name]["model_kwargs"])
    return HydroCore.from_variant(
        "small", node_feature_dim=19, edge_feature_dim=13, event_control_heads=True, **model_kwargs
    )


def train_arm(
    *, arm_name: str, seed: int, train_dataset: ShardedScenarioDataset, validation_dataset: ShardedScenarioDataset
) -> tuple[HydroCore, dict[str, Any]]:
    cfg_dict = yaml.safe_load((ROOT / "configs" / "training-v5-causal.yaml").read_text())["training"]
    config = TrainingConfig(
        seed=seed,
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
    model = build_model(arm_name=arm_name, seed=seed)
    run_root = RUN_ROOT / f"seed-{seed}" / arm_name
    started = time.monotonic()
    localizer_mode = ARMS[arm_name]["localizer_mode"]
    with_physics = ARMS[arm_name]["model_kwargs"].get("localizer_physics_feature_dim", 0) > 0
    physics_columns = ARMS[arm_name]["physics_columns"]
    trainer = Trainer(
        model,
        train_dataset,
        config=config,
        run_root=run_root,
        validation_dataset=validation_dataset,
        collate_fn=make_collate_fn(localizer_mode=localizer_mode, with_physics=with_physics, physics_columns=physics_columns),
    )
    summary = trainer.fit()
    elapsed = time.monotonic() - started
    return model, {
        "arm": arm_name,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "epochs_completed": summary.epochs_completed,
        "stopped_early": summary.stopped_early,
        "stop_reason": summary.stop_reason,
        "global_steps": summary.global_steps,
        "best_validation_loss": summary.best_validation_loss,
    }


def _row_metrics(model: HydroCore, example: ScenarioExample, *, arm_name: str, ood_detector: OODDetector) -> dict[str, Any]:
    localizer_mode = ARMS[arm_name]["localizer_mode"]
    with_physics = ARMS[arm_name]["model_kwargs"].get("localizer_physics_feature_dim", 0) > 0
    physics_columns = ARMS[arm_name]["physics_columns"]
    inputs, targets = collate_variable_topology([example])
    inputs = augment_batch(inputs, localizer_mode=localizer_mode, with_physics=with_physics, physics_columns=physics_columns)
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

    active_sensor = csf.active_sensor_mask_from_temporal(inputs.get("sensor_mask"), inputs["node_mask"])
    hop = csf.compute_hop_distance(inputs["node_mask"], inputs.get("edge_index"), inputs.get("edge_mask"))
    struct_row = csf.compute_structural_features(inputs["node_mask"], inputs.get("edge_index"), inputs.get("edge_mask"), active_sensor, hop)[0]
    struct_columns = {name: index for index, name in enumerate(csf.NODE_STRUCTURAL_COLUMNS)}

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
            row["source_hop_to_nearest_sensor_normalized"] = float(struct_row[true_index, struct_columns["hop_to_nearest_sensor_normalized"]])
            row["source_mean_hop_to_sensors_normalized"] = float(struct_row[true_index, struct_columns["mean_hop_to_sensors_normalized"]])
    if "event_presence_logits" in output and "event_presence" in targets:
        pooled_logit = output["event_presence_logits"][0]
        predicted = int(torch.argmax(pooled_logit).item()) if pooled_logit.numel() > 1 else int((pooled_logit > 0).item())
        row["event_presence_true"] = int(targets["event_presence"][0].item())
        row["event_presence_correct"] = predicted == row["event_presence_true"]
    return row


def evaluate_arm(
    model: HydroCore, *, arm_name: str, datasets: dict[str, tuple[ShardedScenarioDataset, list[int]]]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    ood_detector = OODDetector(OODReference(validated_network_hashes=()))
    rows_by_population: dict[str, list[dict[str, Any]]] = {}
    train_topology_hashes: set[str] = set()
    calibration_examples: list[CalibrationExample] = []
    for population, (dataset, indices) in datasets.items():
        rows = []
        for index in indices:
            example = dataset[index]
            row = _row_metrics(model, example, arm_name=arm_name, ood_detector=ood_detector)
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
            model_hash=f"pilv-{arm_name}",
            feature_schema_hash="pilv-1",
            dataset_manifest_hash="pilv-1",
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
        "localizer_mode": ARMS[arm_name]["localizer_mode"],
        "model_kwargs": ARMS[arm_name]["model_kwargs"],
        "physics_columns": list(ARMS[arm_name]["physics_columns"]) if ARMS[arm_name]["physics_columns"] else None,
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


def run_seed(seed: int, arm_names: list[str]) -> None:
    results_dir = RESULTS_ROOT / f"seed-{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[seed {seed}] Loading datasets (already-generated, leakage-checked cycle-b2 corpus)...")
    train_full = ShardedScenarioDataset(CORPUS_ROOT / "train", expected_split="train")
    validation_full = ShardedScenarioDataset(CORPUS_ROOT / "validation", expected_split="validation")
    calibration_full = ShardedScenarioDataset(CORPUS_ROOT / "calibration", expected_split="calibration")
    dev_holdout_full = ShardedScenarioDataset(CORPUS_ROOT / "development_holdout", expected_split="development_holdout")
    ood_full = ShardedScenarioDataset(CORPUS_ROOT / "ood-UNSEEN_TOPOLOGY", expected_split="development_holdout")

    train_indices = stratified_indices(train_full, per_family=TRAIN_PER_FAMILY, families=TRAINED_FAMILIES, seed=seed)
    train_ds = ShardedScenarioDataset(CORPUS_ROOT / "train", expected_split="train", indices=train_indices)
    validation_indices = capped_indices(validation_full, limit=EVAL_VALIDATION_LIMIT, seed=seed)
    validation_ds = ShardedScenarioDataset(CORPUS_ROOT / "validation", expected_split="validation", indices=validation_indices)
    dev_holdout_indices = capped_indices(dev_holdout_full, limit=EVAL_DEV_HOLDOUT_LIMIT, seed=seed)

    train_families_present = {train_full[index].network_id for index in train_indices}
    print(f"[seed {seed}] Train subsample: {len(train_indices)} examples across families {sorted(train_families_present)}")

    manifest_path = results_dir / "run-manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(
        {
            "seed": seed,
            "all_predeclared_seeds": list(SEEDS),
            "pilot_epochs": PILOT_EPOCHS,
            "train_per_family": TRAIN_PER_FAMILY,
            "train_indices_count": len(train_indices),
            "validation_indices_count": len(validation_indices),
            "development_holdout_indices_count": len(dev_holdout_indices),
            "ood_unseen_topology_count": len(ood_full),
            "calibration_count": len(calibration_full),
            "capacity_matched_hidden_dim": CAPACITY_MATCHED_HIDDEN_DIM,
        }
    )
    manifest.setdefault("arms_run", [])
    for name in arm_names:
        if name not in manifest["arms_run"]:
            manifest["arms_run"].append(name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for arm_name in arm_names:
        eval_path = results_dir / f"{arm_name.lower()}-evaluation.json"
        if eval_path.exists():
            print(f"[seed {seed}] arm {arm_name} already evaluated, skipping (delete {eval_path} to rerun)")
            continue
        print(f"\n=== [seed {seed}] Training arm {arm_name} ===")
        model, train_summary = train_arm(arm_name=arm_name, seed=seed, train_dataset=train_ds, validation_dataset=validation_ds)
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
        eval_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for population in ("validation", "development_holdout", "ood-UNSEEN_TOPOLOGY"):
            rows = rows_by_population.get(population, [])
            path = results_dir / f"{arm_name.lower()}-{population}-rows.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"  wrote {arm_name.lower()}-evaluation.json and per-population row logs")

    print(f"\n[seed {seed}] All requested arms complete. Results under {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="single seed to run; default: all of SEEDS")
    parser.add_argument("--arms", type=str, default=None, help="comma-separated arm names; default: PRIORITY_ORDER")
    args = parser.parse_args()
    arm_names = args.arms.split(",") if args.arms else list(PRIORITY_ORDER)
    for name in arm_names:
        if name not in ARMS:
            raise SystemExit(f"unknown arm {name!r}; choices: {sorted(ARMS)}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    for seed in seeds:
        if seed not in SEEDS:
            raise SystemExit(f"seed {seed} is not in the pre-declared SEEDS list {SEEDS}")
        run_seed(seed, arm_names)


if __name__ == "__main__":
    main()
