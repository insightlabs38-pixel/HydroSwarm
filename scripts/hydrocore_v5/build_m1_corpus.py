"""Milestone 1.1 + 1.2 (experiments.txt): build and validate the v5
causal-prefix corpus, then run the identifiability baseline diagnostic.

Scope (see hydroswarm.training.causal_prefix module docstring for the full
rationale): a single canonical topology (golden-reference), freshly
generated on this host rather than replaying the historical committed
cycle-b2 corpus (WNTR regeneration on this host does not reproduce
cycle-b2 bit-for-bit -- verified and rejected, see that module). Splits are
assigned via disjoint seed ranges BEFORE any causal-prefix depth is
generated; every depth-truncated example is a view over an
already-split-assigned scenario, so prefixing cannot move a scenario
between splits by construction.

Writes:
  reports/evaluation/hydrocore-v5/m1-prefix-dataset.json
  reports/evaluation/hydrocore-v5/m1-identifiability.json

The locked final evaluation is verified closed and never opened here.
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.data.scenarios import EventType  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    SPLIT_SEED_RANGES,
    ScenarioRecord,
    _prefix_classical_prior,
    build_scenario_pool,
    fit_pool_signature_library,
)
from hydroswarm.training.corpus import build_sensor_series  # noqa: E402

OUTPUT_DATASET = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-prefix-dataset.json"
OUTPUT_IDENTIFIABILITY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-identifiability.json"

SPLITS = ("train", "validation", "development_holdout", "calibration")


def _split_integrity_report(pools: dict[str, list[ScenarioRecord]]) -> dict[str, Any]:
    all_ids: dict[str, str] = {}
    collisions = []
    for split, records in pools.items():
        for record in records:
            sid = str(record.scenario.manifest.scenario_id)
            if sid in all_ids:
                collisions.append({"scenario_id": sid, "splits": [all_ids[sid], split]})
            all_ids[sid] = split
    return {
        "total_scenarios": sum(len(records) for records in pools.values()),
        "counts_by_split": {split: len(records) for split, records in pools.items()},
        "cross_split_scenario_id_collisions": collisions,
        "disjoint": not collisions,
    }


def _prefix_causality_spotcheck(pools: dict[str, list[ScenarioRecord]]) -> dict[str, Any]:
    """Confirms, for a real regenerated scenario, that each depth's
    truncated evidence contains exactly `min(depth, n)` timestamps and that
    later depths strictly extend (never replace) earlier ones -- i.e. a
    real causal prefix, not an arbitrary window."""

    record = pools["development_holdout"][0]
    context = record.feature_context
    series = build_sensor_series(record.scenario, context)
    n = len(series[0].timestamps_seconds)
    checks = {}
    for depth in CAUSAL_PREFIX_DEPTHS:
        k = min(depth, n)
        from hydroswarm.training.causal_prefix import truncate_causal_prefix

        truncated = truncate_causal_prefix(series[0], depth)
        checks[str(depth)] = {
            "requested_depth": depth,
            "actual_timesteps": len(truncated.timestamps_seconds),
            "matches_min_depth_n": len(truncated.timestamps_seconds) == k,
            "is_prefix_of_full": tuple(truncated.timestamps_seconds) == tuple(series[0].timestamps_seconds[:k]),
        }
    return {
        "scenario_id": str(record.scenario.manifest.scenario_id),
        "full_history_timesteps": n,
        "per_depth": checks,
        "all_pass": all(c["matches_min_depth_n"] and c["is_prefix_of_full"] for c in checks.values()),
    }


def _identifiability_diagnostic(
    pools: dict[str, list[ScenarioRecord]], library
) -> dict[str, Any]:
    node_ids = library.node_ids
    per_depth: dict[str, Any] = {}
    for depth in CAUSAL_PREFIX_DEPTHS:
        beliefs = []
        truths = []
        ambiguous_counts = []
        true_probabilities = []
        for record in pools["development_holdout"]:
            scenario = record.scenario
            if scenario.manifest.event_type != EventType.CONTAMINATION.value:
                continue
            truth = scenario.manifest.incident.source_nodes[0]
            series = build_sensor_series(scenario, record.feature_context)
            prior = _prefix_classical_prior(library, node_ids, series, depth)
            beliefs.append(prior)
            truths.append(truth)
            true_probabilities.append(prior.get(truth, 0.0))
            sorted_probs = sorted(prior.values(), reverse=True)
            best = sorted_probs[0] if sorted_probs else 0.0
            ambiguous_counts.append(sum(1 for p in sorted_probs if best > 0 and p >= 0.8 * best))

        top1 = [localization_top_k(belief, truth, k=1) for belief, truth in zip(beliefs, truths, strict=True)]
        top3 = [localization_top_k(belief, truth, k=3) for belief, truth in zip(beliefs, truths, strict=True)]
        mrr = mean_reciprocal_rank(beliefs, truths) if beliefs else None

        # Pairwise physics-signature similarity restricted to this depth's
        # timesteps -- a scenario-independent diagnostic of whether the
        # underlying physics (not any particular incident's noise) actually
        # separates candidate sources at this evidence depth.
        pairwise = []
        for a, b in itertools.combinations(node_ids, 2):
            sig_a = np.asarray(library.signatures[a][:depth], dtype=np.float64)
            sig_b = np.asarray(library.signatures[b][:depth], dtype=np.float64)
            valid = np.isfinite(sig_a) & np.isfinite(sig_b)
            if valid.sum() < 2 or np.std(sig_a[valid]) < 1e-9 or np.std(sig_b[valid]) < 1e-9:
                continue
            correlation = float(np.corrcoef(sig_a[valid], sig_b[valid])[0, 1])
            if np.isfinite(correlation):
                pairwise.append(correlation)

        per_depth[str(depth)] = {
            "n": len(beliefs),
            "classical_top1": statistics.fmean(top1) if top1 else None,
            "classical_top3": statistics.fmean(top3) if top3 else None,
            "classical_mrr": mrr,
            "mean_true_source_probability": statistics.fmean(true_probabilities) if true_probabilities else None,
            "mean_source_ambiguity_count": statistics.fmean(ambiguous_counts) if ambiguous_counts else None,
            "distinguishable_source_fraction": (
                statistics.fmean(1.0 if count <= 1 else 0.0 for count in ambiguous_counts) if ambiguous_counts else None
            ),
            "mean_pairwise_signature_correlation": statistics.fmean(pairwise) if pairwise else None,
        }
    return {
        "purpose": (
            "Milestone 1.2: deterministic/classical separability diagnostic -- prevents treating "
            "physically ambiguous low-depth observations as a neural failure. Computed on "
            "development_holdout only, never train/calibration/locked."
        ),
        "node_ids": list(node_ids),
        "per_depth": per_depth,
    }


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"

    pools: dict[str, list[ScenarioRecord]] = {
        split: build_scenario_pool(split, network_loader=build_wntr_network) for split in SPLITS
    }
    library = fit_pool_signature_library(pools["train"])

    dataset_report = {
        "schema_version": 1,
        "purpose": "Milestone 1.1 (experiments.txt): v5 causal-prefix corpus construction and split-safety validation.",
        "network_family": "golden-reference",
        "causal_prefix_depths": list(CAUSAL_PREFIX_DEPTHS),
        "split_seed_ranges": {split: {"start_seed": start, "count": count} for split, (start, count) in SPLIT_SEED_RANGES.items()},
        "split_policy": (
            "Splits assigned via disjoint seed ranges BEFORE any prefix depth is generated; every "
            "depth-truncated example is a lazy view (hydroswarm.training.causal_prefix."
            "CausalPrefixDatasetView) over an already-split-assigned GeneratedScenario, so prefixing "
            "cannot move a scenario between splits."
        ),
        "corpus_regeneration_note": (
            "Fresh host-native generation, not a replay of the historical committed cycle-b2 corpus: "
            "replaying scripts/generate_cycle_b_corpus.py's own _generate_topology_scenarios exactly "
            "reproduced 0/3000 golden-reference train scenarios' scenario_id/sensor_nodes/source_nodes "
            "on this host (cross-environment WNTR/numpy RNG divergence, consistent with but more severe "
            "than the cross-architecture non-reproducibility already documented in "
            "scripts/restore_cycle_b2_train_scenario_cache.py). This corpus therefore does not share "
            "scenario_ids with cycle-b2/joint-v4 and is not used to compare against the frozen v4 "
            "baseline curve directly -- only the causal-prefix ARMS trained on it are compared to each "
            "other (Milestone 1's actual scientific question)."
        ),
        "supervised_task_scope": (
            "Sentinel task family only (source_node, source_region, start_time, duration, "
            "relative_strength, event_presence, event_cause, evidence_sufficiency, sensor_fault). "
            "Scout/Strategist/OOD/auxiliary heads remain declared in task_weights "
            "(configs/training-v5-causal.yaml, require_complete_task_weights=True) but receive no "
            "target in this corpus -- an explicit scope limit, not a silent omission; see "
            "hydroswarm.training.causal_prefix module docstring."
        ),
        "event_scope": "CONTAMINATION only (event_presence=True for every scenario).",
        "signature_library_manifest_hash": library.manifest_hash,
        "split_integrity": _split_integrity_report(pools),
        "prefix_causality_spotcheck": _prefix_causality_spotcheck(pools),
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    identifiability_report = {
        "schema_version": 1,
        **_identifiability_diagnostic(pools, library),
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATASET.write_text(json.dumps(dataset_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    OUTPUT_IDENTIFIABILITY.write_text(
        json.dumps(identifiability_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    print(json.dumps(dataset_report["split_integrity"], indent=2))
    print(json.dumps({k: v["classical_top1"] for k, v in identifiability_report["per_depth"].items()}, indent=2))

    passed = dataset_report["split_integrity"]["disjoint"] and dataset_report["prefix_causality_spotcheck"]["all_pass"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
