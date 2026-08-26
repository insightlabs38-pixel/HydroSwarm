"""Re-run the exp/topology-generalization pilot's exact CONTROL vs
EXPERIMENTAL_TOPOLOGY_RELATIVE protocol, this time persisting per-example
prediction rows (branch exp/failure-mode-diagnostics, Phase 4 input).

Why this script exists: the original pilot (scripts/hydrocore_v5_experimental/
topology_generalization/run_pilot.py) computed per-example rows internally
(`evaluate_arm`'s `rows_by_population`) but only ever wrote AGGREGATE
summaries to disk (reports/evaluation/topology-generalization/*.json); its
trained checkpoints were also never committed (see .gitignore:
experiments/topology-generalization/runs/). A paired per-example analysis
(2x2 top-1 transition table, per-example rank/margin deltas -- required by
this branch's Phase 4) needs the actual per-example predictions, which do
not exist as a committed artifact. This script reproduces them by re-running
the *identical* protocol (same seed 20260814, same stratified/capped
indices, same architecture/training config, imported directly from
run_pilot.py rather than reimplemented, so there is exactly one source of
truth for "what the pilot did") and additionally dumps per-example rows.

This is a fresh, non-deterministic-across-machines re-run (CPU float
arithmetic, library versions), so its aggregate numbers are EXPECTED to
differ slightly from the committed exp/topology-generalization aggregate
JSONs -- this script diffs against them and reports the discrepancy
explicitly (reproduction-check.json) rather than asserting bit-identity.
The original pilot's own committed conclusions are treated as the
confirmatory record; this re-run is exploratory instrumentation to obtain
per-example structure for Phase 4, not a re-litigation of H1/H2.

Never opens data/locked/. Trains fresh models; never loads or fine-tunes
models/hydrocore-v5-release.

Usage: python3 scripts/hydrocore_v5_experimental/failure_mode_diagnostics/rerun_topology_pilot_with_logging.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental" / "topology_generalization"))
sys.path.insert(0, str(ROOT))

import run_pilot as rp  # noqa: E402
from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator  # noqa: E402
from hydroswarm.classical.metrics import entropy as shannon_entropy  # noqa: E402
from hydroswarm.classical.metrics import _ranked  # noqa: E402
from hydroswarm.training.sharded_data import ShardedScenarioDataset  # noqa: E402
from scripts.hydrocore_v5_experimental.failure_mode_diagnostics import graph_features as gf  # noqa: E402

OUTPUT_DIR = ROOT / "reports" / "evaluation" / "failure-mode-diagnostics" / "pilot-rerun"
COMMITTED_EVAL_DIR = ROOT / "reports" / "evaluation" / "topology-generalization"


def enrich_row(row: dict[str, Any], example) -> dict[str, Any]:
    """Add rank/margin/graph-structural diagnostics to a run_pilot._row_metrics row."""

    probabilities = row["probabilities"]
    ranked = _ranked(probabilities) if probabilities else []
    true_index = row.get("true_index")
    row["true_source_rank"] = (ranked.index(true_index) + 1) if (true_index in ranked) else None
    values = sorted(probabilities.values(), reverse=True) if probabilities else []
    row["top1_probability"] = values[0] if values else None
    row["margin_top1_top2"] = (values[0] - values[1]) if len(values) >= 2 else None
    row["true_source_probability"] = probabilities.get(true_index) if true_index is not None else None
    row["posterior_entropy_bits"] = shannon_entropy(list(probabilities.values())) if probabilities else None
    row["n_candidates"] = len(probabilities)

    topology = example.topology
    if topology is not None:
        graph = gf.build_graph(list(topology.node_ids), [tuple(pair) for pair in topology.edge_ids])
        # coastal-branch's reservoir node ids follow the same "R<n>" naming
        # visible in data/topologies/coastal-branch.inp; recovered generically
        # by node_type-agnostic degree only if named, else fall back to none.
        reservoir_ids = [node for node in topology.node_ids if node.startswith("R")]
        level = gf.graph_level_features(graph, reservoir_ids=reservoir_ids)
        source_id = topology.source_node_id_for_local_index(true_index) if true_index is not None else None
        source = gf.source_node_features(graph, source_id, reservoir_ids=reservoir_ids) if source_id else None
        row["graph_node_count"] = level.node_count
        row["graph_edge_count"] = level.edge_count
        row["graph_density"] = level.density
        row["graph_diameter"] = level.diameter
        row["source_degree"] = source.degree if source else None
        row["source_betweenness_centrality"] = source.betweenness_centrality if source else None
        row["source_closeness_centrality"] = source.closeness_centrality if source else None
        row["source_hops_to_reservoir"] = source.hops_to_reservoir if source else None
        row["source_is_boundary_node"] = source.is_boundary_node if source else None
    return row


def evaluate_arm_with_rows(model, *, name: str, augmented: bool, datasets: dict[str, tuple]) -> tuple[dict, dict[str, list[dict]]]:
    """Reproduction of run_pilot.evaluate_arm's exact logic, extended to also
    return every population's enriched per-example rows (not just the
    aggregate summary). Kept as a near-verbatim copy rather than a monkeypatch
    so run_pilot.py itself (frozen prior-pilot evidence) stays untouched."""

    ood_detector = rp.OODDetector(rp.OODReference(validated_network_hashes=()))
    rows_by_population: dict[str, list[dict[str, Any]]] = {}
    examples_by_population: dict[str, list[Any]] = {}
    train_topology_hashes: set[str] = set()
    calibration_examples: list[CalibrationExample] = []

    for population, (dataset, indices) in datasets.items():
        rows = []
        examples = []
        for index in indices:
            example = dataset[index]
            row = rp._row_metrics(model, example, augmented=augmented, ood_detector=ood_detector)
            rows.append(row)
            examples.append(example)
            if population == "train" and row["topology_hash"]:
                train_topology_hashes.add(row["topology_hash"])
        rows_by_population[population] = rows
        examples_by_population[population] = examples

    ood_detector = rp.OODDetector(rp.OODReference(validated_network_hashes=tuple(sorted(train_topology_hashes))))
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
    if len(calibration_examples) >= rp.MINIMUM_GROUP_SIZE:
        calibrator = SplitConformalCalibrator.fit(
            calibration_examples,
            alpha=rp.ALPHA,
            model_hash=f"pilot-rerun-{name}",
            feature_schema_hash="pilot-rerun",
            dataset_manifest_hash="pilot-rerun",
            minimum_group_size=rp.MINIMUM_GROUP_SIZE,
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
    enriched_rows: dict[str, list[dict[str, Any]]] = {}
    for population in ("validation", "development_holdout", "ood-UNSEEN_TOPOLOGY"):
        rows = rows_by_population.get(population, [])
        examples = examples_by_population.get(population, [])
        localized = [(row, example) for row, example in zip(rows, examples) if row.get("has_source") and row.get("true_index") is not None]
        top1 = statistics.fmean(row["top1"] for row, _ in localized) if localized else None
        top3 = statistics.fmean(row["top3"] for row, _ in localized) if localized else None
        mrr = statistics.fmean(row["reciprocal_rank"] for row, _ in localized) if localized else None
        by_family: dict[str, dict[str, Any]] = {}
        for network_id in sorted({row["network_id"] for row in rows}):
            family_localized = [row for row, _ in localized if row["network_id"] == network_id]
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
        for row, _ in localized:
            topology_known = row["topology_hash"] in train_topology_hashes
            ood_level = row["ood_level"]
            if calibrator is None or not topology_known:
                proxy_abstained += 1
                row["candidate_set_size"] = None
                row["proxy_actionable"] = False
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
            row["candidate_set_size"] = len(candidates)
            row["conformal_truth_coverage"] = row["true_index"] in candidates
            if candidates:
                proxy_actionable += 1
                row["proxy_actionable"] = True
            else:
                proxy_abstained += 1
                row["proxy_actionable"] = False
        event_rows = [row for row, _ in localized if "event_presence_correct" in row]
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
            "known_family_fraction": sum(1 for row, _ in localized if row["topology_hash"] in train_topology_hashes) / n_localized,
            "proxy_actionable_rate": proxy_actionable / n_localized,
            "proxy_abstention_rate": proxy_abstained / n_localized,
            "proxy_candidate_set_size": statistics.fmean(candidate_sizes) if candidate_sizes else None,
            "proxy_calibrated_coverage": statistics.fmean(coverage_hits) if coverage_hits else None,
            "ood_caution_or_outside_rate": statistics.fmean(row["ood_level"] != "NORMAL" for row in rows) if rows else None,
        }
        enriched_rows[population] = [enrich_row(row, example) for row, example in localized]
    return summary, enriched_rows


def diff_against_committed(name: str, summary: dict) -> dict:
    committed_path = COMMITTED_EVAL_DIR / f"{name.lower()}-evaluation.json"
    if not committed_path.exists():
        return {"committed_artifact_found": False}
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    diffs = {}
    for population in ("validation", "development_holdout", "ood-UNSEEN_TOPOLOGY"):
        for metric in ("top1", "top3", "mrr"):
            rerun_value = summary["populations"][population][metric]
            committed_value = committed["populations"][population][metric]
            diffs[f"{population}.{metric}"] = {
                "rerun": rerun_value,
                "committed": committed_value,
                "delta": None if rerun_value is None or committed_value is None else rerun_value - committed_value,
            }
    return {"committed_artifact_found": True, "metric_diffs": diffs}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets (identical to run_pilot.py)...")
    train_full = ShardedScenarioDataset(rp.CORPUS_ROOT / "train", expected_split="train")
    validation_full = ShardedScenarioDataset(rp.CORPUS_ROOT / "validation", expected_split="validation")
    calibration_full = ShardedScenarioDataset(rp.CORPUS_ROOT / "calibration", expected_split="calibration")
    dev_holdout_full = ShardedScenarioDataset(rp.CORPUS_ROOT / "development_holdout", expected_split="development_holdout")
    ood_full = ShardedScenarioDataset(rp.CORPUS_ROOT / "ood-UNSEEN_TOPOLOGY", expected_split="development_holdout")

    train_indices = rp.stratified_indices(train_full, per_family=rp.TRAIN_PER_FAMILY, families=rp.TRAINED_FAMILIES, seed=rp.SEED)
    train_ds = ShardedScenarioDataset(rp.CORPUS_ROOT / "train", expected_split="train", indices=train_indices)
    validation_indices = rp.capped_indices(validation_full, limit=rp.EVAL_VALIDATION_LIMIT, seed=rp.SEED)
    validation_ds = ShardedScenarioDataset(rp.CORPUS_ROOT / "validation", expected_split="validation", indices=validation_indices)
    dev_holdout_indices = rp.capped_indices(dev_holdout_full, limit=rp.EVAL_DEV_HOLDOUT_LIMIT, seed=rp.SEED)

    reproduction_report: dict[str, Any] = {"seed": rp.SEED, "pilot_epochs": rp.PILOT_EPOCHS, "arms": {}}

    for name, augmented in (("CONTROL", False), ("EXPERIMENTAL_TOPOLOGY_RELATIVE", True)):
        print(f"\n=== Training arm {name} (augmented={augmented}) ===")
        started = time.monotonic()
        model, train_summary = rp.train_arm(name=name, augmented=augmented, train_dataset=train_ds, validation_dataset=validation_ds)
        print(f"  trained in {time.monotonic() - started:.1f}s")

        print(f"  Evaluating arm {name} with per-example logging...")
        eval_datasets = {
            "train": (train_ds, list(range(len(train_ds)))),
            "validation": (validation_ds, list(range(len(validation_ds)))),
            "calibration": (calibration_full, list(range(len(calibration_full)))),
            "development_holdout": (dev_holdout_full, dev_holdout_indices),
            "ood-UNSEEN_TOPOLOGY": (ood_full, list(range(len(ood_full)))),
        }
        summary, rows = evaluate_arm_with_rows(model, name=name, augmented=augmented, datasets=eval_datasets)
        reproduction_report["arms"][name] = {
            "training": train_summary,
            "reproduction_check": diff_against_committed(name, summary),
        }
        for population, population_rows in rows.items():
            out_path = OUTPUT_DIR / f"{name.lower()}-{population}-rows.jsonl"
            with out_path.open("w", encoding="utf-8") as stream:
                for row in population_rows:
                    stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            print(f"    wrote {len(population_rows)} rows to {out_path}")

    report_path = OUTPUT_DIR / "reproduction-check.json"
    report_path.write_text(json.dumps(reproduction_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
