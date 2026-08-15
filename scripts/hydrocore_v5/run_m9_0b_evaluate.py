"""Milestone 9.0b Sections 2-13: fit and evaluate the four frozen calibration
grouping/fallback schemes (`m9_0b_calibration_schemes.py`) against the
FROZEN M9.0a `ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY` predictor
checkpoints (docs/evaluation/HYDROCORE_V5_M9_0B_PROTOCOL.md).

No retraining, no gradient computation anywhere in this script (`_infer`
and the calibration-example forward passes below both use
`torch.no_grad()`, matching `run_m9_0a_evaluate.py`'s own convention
exactly). Checkpoint provenance is cross-verified against both
`m9-0a-runs/*.json` and `m9-0a-results.json` before any inference.

Writes:
  reports/evaluation/hydrocore-v5/m9-0b-results.json
  reports/evaluation/hydrocore-v5/m9-0b-calibration-by-seed.json
  reports/evaluation/hydrocore-v5/m9-0b-group-support.json
  reports/evaluation/hydrocore-v5/m9-0b-unseen-transfer.json
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.calibration.conformal import classify_runtime_condition  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    fit_pool_signature_library,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.corpus import build_sensor_series  # noqa: E402

from m9_0b_calibration_schemes import (  # noqa: E402
    ALPHA,
    MINIMUM_GROUP_SIZE,
    SCHEME_NAMES,
    SchemeRow,
    _quantile,
    candidate_set_for_scheme,
    fit_hierarchical,
    fit_scheme,
    wilson_interval,
)
from run_m3_calibration import DEPTH_BUCKET_OF  # noqa: E402
from run_m7_topology import (  # noqa: E402
    CALIBRATION_PER_FAMILY,
    SEED_BASES,
    TRAIN_PER_FAMILY,
    TRAINED_FAMILIES,
    _family_scenario_pool,
    _generate_eval_scenarios,
    _infer,
)
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m9_0_arm_b import FEATURE_KWARGS  # noqa: E402

RUNS_M9_0A = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-runs"
RESULTS_M9_0A_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-results.json"
TOPOLOGY_M9_0A_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-topology-generalization.json"

OUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0b-results.json"
OUT_CAL_BY_SEED = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0b-calibration-by-seed.json"
OUT_GROUP_SUPPORT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0b-group-support.json"
OUT_UNSEEN_TRANSFER = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0b-unseen-transfer.json"

SEEDS = (20260814, 31874, 20260815)
KNOWN_FAMILIES = tuple(name for name, _ in TRAINED_FAMILIES)
UNSEEN_FAMILIES_NAMES = ("coastal-branch", "tree-branch", "dense-loop")
MATURITY_BUCKETS = ("EARLY", "MID", "MATURE")
NORMALIZED_SET_SIZE_BAR = 0.5  # protocol Section 10, equivalent to M9.0a's frozen 0.5x-node-count bar.


# ---------------------------------------------------------------------------
# Frozen predictor loading, provenance-verified.
# ---------------------------------------------------------------------------


def _verify_and_load_arm_b2_model(seed: int) -> tuple[HydroCore, str]:
    run_record = json.loads((RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
    results_record = json.loads(RESULTS_M9_0A_PATH.read_text())
    recorded_hash = run_record["training_summary"]["export_sha256"]
    cross_check_hash = results_record["arms"]["ARM_B2"]["per_seed"][str(seed)]["checkpoint_sha256"]
    if recorded_hash != cross_check_hash:
        raise ValueError(f"seed {seed}: checkpoint provenance cross-check failed (m9-0a-runs vs m9-0a-results.json)")
    export_path = Path(run_record["training_summary"]["export_path"])
    on_disk_hash = hashlib.sha256(export_path.read_bytes()).hexdigest()
    if on_disk_hash != recorded_hash:
        raise ValueError(f"seed {seed}: on-disk checkpoint hash drifted from recorded provenance -- refusing to evaluate")
    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(str(export_path), device="cpu"), strict=True)
    model.eval()
    return model, recorded_hash


# ---------------------------------------------------------------------------
# Physics-only signature libraries, built once, shared across all 3 seeds
# (model-independent -- matches ARM_B2's own training libraries exactly).
# ---------------------------------------------------------------------------


def _build_known_family_libraries() -> dict[str, Any]:
    libraries: dict[str, Any] = {}
    for family, loader in TRAINED_FAMILIES:
        pool = _family_scenario_pool(
            "train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")], count=TRAIN_PER_FAMILY,
        )
        libraries[family] = fit_pool_signature_library(pool)
    return libraries


# ---------------------------------------------------------------------------
# Row generation -- reusing M9.0a's own methodology unmodified (protocol
# Section 2). Model-forward-only, torch.no_grad() throughout, no gradients.
# ---------------------------------------------------------------------------


def _calibration_pool_for(family: str, loader) -> list[Any]:
    return _family_scenario_pool(
        "calibration", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "calibration")], count=CALIBRATION_PER_FAMILY,
    )


def _scheme_calibration_rows(model: HydroCore, family: str, library: Any, calibration_pool: list[Any]) -> list[SchemeRow]:
    rows: list[SchemeRow] = []
    with torch.no_grad():
        for depth in CAUSAL_PREFIX_DEPTHS:
            for record in calibration_pool:
                example = scenario_to_prefix_example(
                    record.scenario, record.network, library, depth, feature_context=record.feature_context, **FEATURE_KWARGS,
                )
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
                truth = int(example.targets["source_node"].item())
                full_series = build_sensor_series(record.scenario, record.feature_context)
                truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
                condition = classify_runtime_condition(truncated_series)
                bucket = DEPTH_BUCKET_OF[depth]
                rows.append(SchemeRow(probabilities=tuple(probs), true_index=truth, condition=condition, family=family, depth_bucket=bucket))
    return rows


def _scheme_development_rows(model: HydroCore, family: str, library: Any, eval_scenarios: list[tuple[Any, Any, Any]]) -> list[SchemeRow]:
    rows: list[SchemeRow] = []
    for scenario, network, context in eval_scenarios:
        truth_node = scenario.manifest.incident.source_nodes[0]
        full_series = build_sensor_series(scenario, context)
        reference_timestamps = full_series[0].timestamps_seconds if full_series else ()
        for depth in CAUSAL_PREFIX_DEPTHS:
            series = [truncate_causal_prefix(item, depth) for item in full_series]
            result = _infer(model, network, context, series, library, reference_timestamps)
            node_ids = result["node_ids"]
            if truth_node not in node_ids:
                continue
            truth_index = node_ids.index(truth_node)
            bucket = DEPTH_BUCKET_OF[depth]
            rows.append(SchemeRow(
                probabilities=tuple(result["neural_probs"]), true_index=truth_index,
                condition=result["condition"], family=family, depth_bucket=bucket,
            ))
    return rows


def _unseen_rows_from_m9_0a(topology: dict[str, Any], seed: int) -> dict[str, list[SchemeRow]]:
    out: dict[str, list[SchemeRow]] = {}
    for family in UNSEEN_FAMILIES_NAMES:
        raw_rows = topology["arms"]["ARM_B2"]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"][str(seed)]
        out[family] = [
            SchemeRow(
                probabilities=tuple(row["neural_probs"]), true_index=row["truth_index"],
                condition=row["condition"], family=row["family"], depth_bucket=row["depth_bucket"],
            )
            for row in raw_rows
        ]
    return out


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def _candidate_and_source(scheme: str, calibrator: Any, row: SchemeRow) -> tuple[tuple[int, ...], str]:
    if scheme == "HIERARCHICAL_CONSERVATIVE":
        candidate, _q, source, _group = calibrator.candidate_set(row)
        return candidate, source
    candidate, source, _group = candidate_set_for_scheme(scheme, calibrator, row)
    return candidate, source


def _row_metrics(scheme: str, calibrator: Any, rows: list[SchemeRow]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        candidate, source = _candidate_and_source(scheme, calibrator, row)
        covered = row.true_index in candidate
        out.append({
            "family": row.family, "depth_bucket": row.depth_bucket, "condition": row.condition,
            "covered": covered, "set_size": len(candidate),
            "normalized_set_size": len(candidate) / len(row.probabilities),
            "singleton": len(candidate) == 1,
            "eligible_source_node_count": len(row.probabilities),
            "selection_source": source,
        })
    return out


def _summary_over(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_rows:
        return {"n": 0}
    covered = [row["covered"] for row in metric_rows]
    n = len(metric_rows)
    n_covered = sum(covered)
    lower, upper = wilson_interval(n_covered, n)
    return {
        "n": n,
        "marginal_coverage": n_covered / n,
        "wilson_95_ci": [lower, upper],
        "mean_set_size": statistics.fmean(row["set_size"] for row in metric_rows),
        "median_set_size": statistics.median(row["set_size"] for row in metric_rows),
        "p90_set_size": float(np.percentile([row["set_size"] for row in metric_rows], 90)),
        "mean_normalized_set_size": statistics.fmean(row["normalized_set_size"] for row in metric_rows),
        "singleton_rate": statistics.fmean(row["singleton"] for row in metric_rows),
        "source_inclusion_rate": n_covered / n,  # identical concept to marginal_coverage in this single-source-localization task.
    }


def _by(metric_rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    values = sorted({row[key] for row in metric_rows})
    return {value: _summary_over([row for row in metric_rows if row[key] == value]) for value in values}


# ---------------------------------------------------------------------------
# Group-support audit (protocol Section 7).
# ---------------------------------------------------------------------------


def _group_support_for_scheme(scheme: str, calibrator: Any) -> dict[str, Any]:
    if scheme == "HIERARCHICAL_CONSERVATIVE":
        return {
            "family_depth": _group_support_for_scheme("CURRENT_FAMILY_DEPTH", calibrator.family_depth),
            "pooled_depth": _group_support_for_scheme("POOLED_DEPTH_AWARE", calibrator.pooled_depth),
        }
    groups = {}
    for key, scores in calibrator.artifact.network_scores.items():
        n = len(scores)
        rank = min(n, max(1, int(np.ceil((n + 1) * (1 - ALPHA)))))
        groups[str(key)] = {
            "n": n, "min_score": min(scores), "median_score": float(np.median(scores)), "max_score": max(scores),
            "quantile": _quantile(scores, ALPHA), "quantile_rank": rank,
            "meets_minimum_group_size": n >= MINIMUM_GROUP_SIZE,
        }
    condition_groups = {}
    for key, scores in calibrator.artifact.mondrian_scores.items():
        n = len(scores)
        condition_groups[str(key)] = {"n": n, "quantile": _quantile(scores, ALPHA), "meets_minimum_group_size": n >= MINIMUM_GROUP_SIZE}
    return {
        "network_groups": groups,
        "condition_groups": condition_groups,
        "global_n": len(calibrator.artifact.global_scores),
        "global_quantile": _quantile(calibrator.artifact.global_scores, ALPHA) if calibrator.artifact.global_scores else None,
    }


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    print("building known-family signature libraries...", flush=True)
    libraries = _build_known_family_libraries()

    print("building calibration + development-holdout scenario pools (shared across seeds)...", flush=True)
    calibration_pools = {family: _calibration_pool_for(family, loader) for family, loader in TRAINED_FAMILIES}
    eval_scenarios = {family: _generate_eval_scenarios(family, loader, SEED_BASES[(family, "eval")]) for family, loader in TRAINED_FAMILIES}

    topology_m9_0a = json.loads(TOPOLOGY_M9_0A_PATH.read_text())

    results: dict[str, Any] = {"schemes": {name: {"per_seed": {}} for name in SCHEME_NAMES}}
    calibration_by_seed: dict[str, Any] = {"schemes": {name: {"per_seed": {}} for name in SCHEME_NAMES}}
    group_support: dict[str, Any] = {"schemes": {name: {"per_seed": {}} for name in SCHEME_NAMES}}
    unseen_transfer: dict[str, Any] = {"schemes": {name: {"per_seed": {}} for name in SCHEME_NAMES}}
    checkpoint_provenance: dict[str, str] = {}

    for seed in SEEDS:
        print(f"seed {seed}: loading frozen ARM_B2 checkpoint (provenance-verified)...", flush=True)
        model, checkpoint_sha256 = _verify_and_load_arm_b2_model(seed)
        checkpoint_provenance[str(seed)] = checkpoint_sha256

        print(f"seed {seed}: regenerating calibration-split rows...", flush=True)
        cal_rows: list[SchemeRow] = []
        for family, _loader in TRAINED_FAMILIES:
            cal_rows.extend(_scheme_calibration_rows(model, family, libraries[family], calibration_pools[family]))

        print(f"seed {seed}: regenerating development-holdout rows...", flush=True)
        dev_rows: list[SchemeRow] = []
        for family, _loader in TRAINED_FAMILIES:
            dev_rows.extend(_scheme_development_rows(model, family, libraries[family], eval_scenarios[family]))

        unseen_rows_by_family = _unseen_rows_from_m9_0a(topology_m9_0a, seed)

        for scheme in SCHEME_NAMES:
            print(f"seed {seed}: fitting+evaluating scheme {scheme}...", flush=True)
            calibrator = fit_hierarchical(cal_rows, model_hash=f"m9-0b-arm_b2-seed{seed}") if scheme == "HIERARCHICAL_CONSERVATIVE" else fit_scheme(
                scheme, cal_rows, model_hash=f"m9-0b-arm_b2-seed{seed}",
            )

            dev_metric_rows = _row_metrics(scheme, calibrator, dev_rows)
            overall = _summary_over(dev_metric_rows)
            by_maturity = _by(dev_metric_rows, "depth_bucket")
            by_family = _by(dev_metric_rows, "family")
            by_family_maturity: dict[str, dict[str, Any]] = {}
            for family in KNOWN_FAMILIES:
                for bucket in MATURITY_BUCKETS:
                    subset = [row for row in dev_metric_rows if row["family"] == family and row["depth_bucket"] == bucket]
                    by_family_maturity[f"{family}:{bucket}"] = _summary_over(subset)
            results["schemes"][scheme]["per_seed"][str(seed)] = {
                "overall": overall, "by_maturity": by_maturity, "by_family": by_family, "by_family_maturity": by_family_maturity,
            }

            group_support["schemes"][scheme]["per_seed"][str(seed)] = _group_support_for_scheme(scheme, calibrator)

            unseen_scheme_result: dict[str, Any] = {}
            for family in UNSEEN_FAMILIES_NAMES:
                unseen_metric_rows = _row_metrics(scheme, calibrator, unseen_rows_by_family[family])
                unseen_summary = _summary_over(unseen_metric_rows)
                unseen_summary["by_maturity"] = _by(unseen_metric_rows, "depth_bucket")
                unseen_summary["fallback_sources"] = sorted({row["selection_source"] for row in unseen_metric_rows})
                unseen_summary["applicability_rate"] = statistics.fmean(row["selection_source"] == "NETWORK_SPECIFIC" for row in unseen_metric_rows) if unseen_metric_rows else None
                unseen_scheme_result[family] = unseen_summary
            unseen_transfer["schemes"][scheme]["per_seed"][str(seed)] = unseen_scheme_result

            calibration_by_seed["schemes"][scheme]["per_seed"][str(seed)] = {
                "alpha": ALPHA, "checkpoint_sha256": checkpoint_sha256,
                "n_calibration_rows": len(cal_rows), "n_development_rows": len(dev_rows),
            }

    for artifact_name, payload in (
        ("results", results), ("calibration_by_seed", calibration_by_seed),
        ("group_support", group_support), ("unseen_transfer", unseen_transfer),
    ):
        payload["checkpoint_provenance"] = checkpoint_provenance
        payload["alpha"] = ALPHA

    locked_after = locked_test_opened(ROOT)
    for payload in (results, calibration_by_seed, group_support, unseen_transfer):
        payload["locked_test_opened_before"] = locked_before
        payload["locked_test_opened_after"] = locked_after

    OUT_RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    OUT_CAL_BY_SEED.write_text(json.dumps(calibration_by_seed, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    OUT_GROUP_SUPPORT.write_text(json.dumps(group_support, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    OUT_UNSEEN_TRANSFER.write_text(json.dumps(unseen_transfer, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("wrote", OUT_RESULTS, OUT_CAL_BY_SEED, OUT_GROUP_SUPPORT, OUT_UNSEEN_TRANSFER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
