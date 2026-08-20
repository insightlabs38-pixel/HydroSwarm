"""Milestone 9.0a Sections 9-11: evaluate SINGLE_FAMILY_CONTROL (Arm A --
the existing Milestone-8.7 AGE_FIX_ONLY checkpoints, reused verbatim) and
STEP_MATCHED_INTERLEAVED_MULTI_FAMILY (Arm B2 -- trained fresh by
run_m9_0a_arm_b2.py) under the frozen M9.0a protocol
(docs/evaluation/HYDROCORE_V5_M9_0A_PROTOCOL.md).

Byte-for-byte the same known-network/trained-family/unseen-topology
evaluation methodology `run_m9_0_evaluate.py` used (Sections 11-14 of that
document; same shared eval-scenario pools so B2 is paired against Arm A on
IDENTICAL held-out incidents, and historically comparable to M9.0 Arm B).
The ONLY methodological change from `run_m9_0_evaluate.py` is Section 10 of
the M9.0a protocol: calibration is fit and evaluated SEPARATELY for each of
the three predictor seeds (for BOTH arms), instead of once on a single
"representative" seed reused for every seed's rows -- this milestone's own
named uncertainty is whether the M9.0 calibration rejection (0.833 known-
family coverage) was representative-seed-specific.

Reuses unmodified: run_m7_topology's TRAINED_FAMILIES/UNSEEN_FAMILIES
network loaders, SEED_BASES, _family_scenario_pool, _generate_eval_scenarios,
_classical_fit_library, _fit_model_calibrator (unused directly here --
calibration pooling is inlined identically to run_m9_0_evaluate.py's own
per-arm construction), _rank_metrics, _entropy_normalized;
hydroswarm.inference.fusion's real fusion.

Writes:
  reports/evaluation/hydrocore-v5/m9-0a-results.json
  reports/evaluation/hydrocore-v5/m9-0a-topology-generalization.json
  reports/evaluation/hydrocore-v5/m9-0a-calibration.json
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator, classify_runtime_condition  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.inference.fusion import TrustFeatures, fuse_source_probabilities  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.corpus import build_sensor_series  # noqa: E402

from run_m3_calibration import DEPTH_BUCKET_OF  # noqa: E402
from run_m7_topology import (  # noqa: E402
    CALIBRATION_PER_FAMILY,
    SEED_BASES,
    TRAIN_PER_FAMILY,
    TRAINED_FAMILIES,
    UNSEEN_FAMILIES,
    _classical_belief,
    _classical_fit_library,
    _entropy_normalized,
    _family_scenario_pool,
    _generate_eval_scenarios,
    _infer,
    _rank_metrics,
)
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m9_0_arm_b import FAMILY_NAMES, FEATURE_KWARGS  # noqa: E402

RUNS_M8_7 = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-7-runs"
RUNS_M9_0A = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-runs"
OUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-results.json"
OUT_TOPOLOGY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-topology-generalization.json"
OUT_CALIBRATION = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-calibration.json"

ALPHA = 0.1
SEEDS = (20260814, 31874, 20260815)
EARLY_DEPTHS = (1, 2, 3)
MID_DEPTHS = (4, 6)
MATURE_DEPTHS = (12, 25)
EPS = 1e-9

ARM_A_KNOWN_FAMILIES = frozenset({"golden-reference"})
ARM_B2_KNOWN_FAMILIES = frozenset(FAMILY_NAMES)


# ---------------------------------------------------------------------------
# Checkpoint loading.
# ---------------------------------------------------------------------------


def _load_model_from_export(export_path: str) -> HydroCore:
    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()
    return model


def _arm_a_model(seed: int) -> tuple[HydroCore, str]:
    record = json.loads((RUNS_M8_7 / f"AGE_FIX_ONLY-seed{seed}.json").read_text())
    return _load_model_from_export(record["export_path"]), record["checkpoint_sha256"]


def _arm_b2_model(seed: int) -> tuple[HydroCore, str]:
    record = json.loads((RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
    return _load_model_from_export(record["training_summary"]["export_path"]), record["training_summary"]["export_sha256"]


# ---------------------------------------------------------------------------
# Signature libraries -- physics-only, model-independent, built once per
# family and shared across every arm/seed evaluated against that family.
# ---------------------------------------------------------------------------


def _build_libraries() -> dict[str, Any]:
    libraries: dict[str, Any] = {}
    libraries["golden-reference:arm_a"] = fit_pool_signature_library(
        build_scenario_pool("train", network_loader=build_wntr_network)
    )
    for family, loader in TRAINED_FAMILIES:
        pool = _family_scenario_pool(
            "train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")], count=TRAIN_PER_FAMILY,
        )
        libraries[f"{family}:arm_b2"] = fit_pool_signature_library(pool)
    for family, loader in UNSEEN_FAMILIES:
        library, _records = _classical_fit_library(family, loader)
        libraries[family] = library
    return libraries


def _library_for(libraries: dict[str, Any], family: str, arm: str) -> Any:
    if family == "golden-reference":
        return libraries["golden-reference:arm_a"] if arm == "ARM_A" else libraries["golden-reference:arm_b2"]
    if family in {"branched-loop", "loop-grid"}:
        return libraries[f"{family}:arm_b2"]
    return libraries[family]


# ---------------------------------------------------------------------------
# Full-depth-grid per-family evaluation -- byte-for-byte the same as
# run_m9_0_evaluate.py's own _evaluate_on_family/_postprocess_rows.
# ---------------------------------------------------------------------------


def _evaluate_on_family(
    model: HydroCore, family: str, library: Any, eval_scenarios: list[tuple[Any, Any, Any]], *, known: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario, network, context in eval_scenarios:
        truth_node = scenario.manifest.incident.source_nodes[0]
        full_series = build_sensor_series(scenario, context)
        reference_timestamps = full_series[0].timestamps_seconds if full_series else ()
        for depth in CAUSAL_PREFIX_DEPTHS:
            series = [truncate_causal_prefix(item, depth) for item in full_series]
            result = _infer(model, network, context, series, library, reference_timestamps)
            classical_belief, residual_raw = _classical_belief(library, result["node_ids"], series, reference_timestamps)
            node_ids = result["node_ids"]
            if truth_node not in node_ids:
                continue
            truth_index = node_ids.index(truth_node)
            healthy = statistics.fmean(item.health[-1] for item in series) if series else 1.0
            missing = statistics.fmean(float(item.missing[-1]) for item in series) if series else 0.0
            rows.append({
                "family": family, "known": known, "depth": depth, "depth_bucket": DEPTH_BUCKET_OF[depth],
                "seed": scenario.manifest.seed, "truth_node": truth_node, "truth_index": truth_index,
                "node_ids": node_ids, "neural_probs": result["neural_probs"], "neural_logits": result["neural_logits"],
                "classical_belief": classical_belief, "condition": result["condition"],
                "evidence_sufficiency": result["evidence_sufficiency"], "residual_raw": residual_raw,
                "healthy_sensor_fraction": healthy, "missing_rate": missing,
            })
    return rows


def _postprocess_rows(rows: list[dict[str, Any]], calibrator: SplitConformalCalibrator | None) -> None:
    if not rows:
        return
    residuals = np.asarray([row["residual_raw"] for row in rows], dtype=np.float64)
    lo, hi = float(residuals.min()), float(residuals.max())
    span = hi - lo
    for row, residual_raw in zip(rows, residuals, strict=True):
        normalized_residual = 0.0 if span <= EPS else float((residual_raw - lo) / span)
        neural_probs = row["neural_probs"]
        classical_belief = row["classical_belief"]
        topology_novelty = 0.0 if row["known"] else 1.0
        features = TrustFeatures(
            healthy_sensor_fraction=float(np.clip(row["healthy_sensor_fraction"], 0.0, 1.0)),
            missing_rate=float(np.clip(row["missing_rate"], 0.0, 1.0)),
            normalized_residual=normalized_residual,
            hydraulic_uncertainty=float(np.clip(1.0 - row["evidence_sufficiency"], 0.0, 1.0)),
            neural_entropy=_entropy_normalized(neural_probs),
            classical_entropy=_entropy_normalized(classical_belief),
            ood_score=topology_novelty,
        )
        node_ids = row["node_ids"]
        mask = np.ones(len(node_ids), dtype=bool)
        classical_array = np.asarray(classical_belief, dtype=np.float64)
        classical_array = classical_array / max(float(classical_array.sum()), EPS)
        fused, diag = fuse_source_probabilities(row["neural_logits"], classical_array, mask, features)
        row["normalized_residual"] = normalized_residual
        row["trust_features"] = asdict(features)
        row["hybrid_belief"] = fused.tolist()
        row["hybrid_trust"] = diag.classical_trust
        truth_index = row["truth_index"]
        row["metrics_neural"] = _rank_metrics(neural_probs, truth_index)
        row["metrics_classical"] = _rank_metrics(classical_belief, truth_index)
        row["metrics_hybrid"] = _rank_metrics(fused, truth_index)
        p_truth_neural = float(neural_probs[truth_index])
        row["nll_neural"] = -math.log(p_truth_neural + EPS)
        row["posterior_entropy_neural"] = _entropy_normalized(neural_probs) * max(1.0, math.log2(len(neural_probs)))
        row["all_finite"] = bool(np.isfinite(neural_probs).all())
        if calibrator is not None:
            bucket = row["depth_bucket"]
            network_id = f"{row['family']}:{bucket}"
            source, group_key, _scores = calibrator.selection(condition=row["condition"], network_id=network_id)
            candidate = calibrator.candidate_set(neural_probs, condition=row["condition"], network_id=network_id)
            row["calibration_source"] = source
            row["calibration_group_key"] = group_key
            row["calibration_applicable"] = source == "NETWORK_SPECIFIC"
            row["candidate_set_size"] = len(candidate)
            row["candidate_set_includes_truth"] = truth_index in candidate
            row["candidate_covered"] = truth_index in candidate


def _mean(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values = []
    for row in rows:
        node: Any = row
        for key in keys:
            node = node[key]
        if node is not None:
            values.append(float(node))
    return statistics.fmean(values) if values else None


def _median(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values = []
    for row in rows:
        node: Any = row
        for key in keys:
            node = node[key]
        if node is not None:
            values.append(float(node))
    return statistics.median(values) if values else None


def _bucket_summary(rows: list[dict[str, Any]], bucket: str) -> dict[str, Any]:
    subset = [row for row in rows if row["depth_bucket"] == bucket]
    if not subset:
        return {}
    return {
        "n": len(subset),
        "neural": {m: _mean(subset, "metrics_neural", m) for m in ("top1", "top3", "mrr")},
        "hybrid": {m: _mean(subset, "metrics_hybrid", m) for m in ("top1", "top3", "mrr")},
        "classical": {m: _mean(subset, "metrics_classical", m) for m in ("top1", "top3", "mrr")},
        "nll_neural": _mean(subset, "nll_neural"),
        "posterior_entropy_neural": _mean(subset, "posterior_entropy_neural"),
        "all_finite": all(row["all_finite"] for row in subset),
    }


def _calibration_examples(model: HydroCore, family: str, library: Any, calibration_pool: list[Any]) -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []
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
            examples.append(CalibrationExample(
                probabilities=tuple(probs), true_index=truth, condition=condition, network_id=f"{family}:{bucket}",
            ))
    return examples


def _fit_calibrator_for_seed(arm: str, seed: int, libraries: dict[str, Any], loader_fn) -> SplitConformalCalibrator:
    known_families = ARM_A_KNOWN_FAMILIES if arm == "ARM_A" else ARM_B2_KNOWN_FAMILIES
    all_eval_families = dict(TRAINED_FAMILIES) | dict(UNSEEN_FAMILIES)
    model, _sha = loader_fn(seed)
    examples: list[CalibrationExample] = []
    with torch.no_grad():
        for family in sorted(known_families):
            loader = all_eval_families[family]
            library = _library_for(libraries, family, arm)
            calibration_pool = (
                build_scenario_pool("calibration", network_loader=build_wntr_network)
                if family == "golden-reference" and arm == "ARM_A"
                else _family_scenario_pool(
                    "calibration", network_loader=loader, family=family,
                    seed_base=SEED_BASES[(family, "calibration")], count=CALIBRATION_PER_FAMILY,
                )
            )
            examples.extend(_calibration_examples(model, family, library, calibration_pool))
    return SplitConformalCalibrator.fit(
        examples, alpha=ALPHA, model_hash=f"m9-0a-{arm}-seed{seed}", feature_schema_hash="n/a",
        dataset_manifest_hash=f"m9-0a-{arm}-seed{seed}-calibration-pool",
    )


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    print("building signature libraries...", flush=True)
    libraries = _build_libraries()

    print("generating shared eval scenario pools (identical across arms/seeds, matching M9.0)...", flush=True)
    all_eval_families = list(TRAINED_FAMILIES) + list(UNSEEN_FAMILIES)
    eval_scenarios_by_family = {
        family: _generate_eval_scenarios(family, loader, SEED_BASES[(family, "eval")])
        for family, loader in all_eval_families
    }

    results: dict[str, Any] = {"arms": {"ARM_A": {"per_seed": {}}, "ARM_B2": {"per_seed": {}}}}
    topology: dict[str, Any] = {"arms": {"ARM_A": {}, "ARM_B2": {}}}
    all_rows: dict[str, dict[int, list[dict[str, Any]]]] = {"ARM_A": {}, "ARM_B2": {}}

    for arm, known_families, loader_fn in (
        ("ARM_A", ARM_A_KNOWN_FAMILIES, _arm_a_model), ("ARM_B2", ARM_B2_KNOWN_FAMILIES, _arm_b2_model),
    ):
        for seed in SEEDS:
            print(f"evaluating {arm} seed {seed}...", flush=True)
            model, checkpoint_sha256 = loader_fn(seed)
            seed_rows: list[dict[str, Any]] = []
            for family, _loader in all_eval_families:
                if family in {"branched-loop", "loop-grid"} and arm == "ARM_A":
                    continue
                known = family in known_families
                library = _library_for(libraries, family, arm)
                rows = _evaluate_on_family(model, family, library, eval_scenarios_by_family[family], known=known)
                seed_rows.extend(rows)
            all_rows[arm][seed] = seed_rows
            results["arms"][arm]["per_seed"][str(seed)] = {"checkpoint_sha256": checkpoint_sha256, "n_rows": len(seed_rows)}

    # Section 10: calibration -- fit SEPARATELY for EACH of the 3 predictor
    # seeds, for BOTH arms (the M9.0a-specific change from run_m9_0_evaluate.py).
    calibration_report: dict[str, Any] = {"arms": {"ARM_A": {"per_seed": {}}, "ARM_B2": {"per_seed": {}}}}
    calibrators: dict[str, dict[int, SplitConformalCalibrator]] = {"ARM_A": {}, "ARM_B2": {}}
    for arm, loader_fn in (("ARM_A", _arm_a_model), ("ARM_B2", _arm_b2_model)):
        for seed in SEEDS:
            print(f"fitting {arm} calibrator, seed {seed}...", flush=True)
            calibrators[arm][seed] = _fit_calibrator_for_seed(arm, seed, libraries, loader_fn)

    # Post-process every arm/seed's rows with THAT SAME SEED's OWN calibrator
    # (no representative-seed reuse -- protocol Section 10).
    for arm in ("ARM_A", "ARM_B2"):
        for seed in SEEDS:
            _postprocess_rows(all_rows[arm][seed], calibrators[arm][seed])

    # Section 9: known-network (golden-reference) localization.
    for arm in ("ARM_A", "ARM_B2"):
        per_seed_localization = {}
        for seed in SEEDS:
            golden_rows = [row for row in all_rows[arm][seed] if row["family"] == "golden-reference"]
            per_seed_localization[str(seed)] = {
                "EARLY": _bucket_summary(golden_rows, "EARLY"), "MID": _bucket_summary(golden_rows, "MID"), "MATURE": _bucket_summary(golden_rows, "MATURE"),
            }
        results["arms"][arm]["known_network_localization"] = per_seed_localization

    # Trained-family retention (Arm B2 only).
    for family in ("branched-loop", "loop-grid"):
        per_seed = {}
        for seed in SEEDS:
            rows = [row for row in all_rows["ARM_B2"][seed] if row["family"] == family]
            per_seed[str(seed)] = {"EARLY": _bucket_summary(rows, "EARLY"), "MID": _bucket_summary(rows, "MID"), "MATURE": _bucket_summary(rows, "MATURE")}
        topology["arms"]["ARM_B2"][f"TRAINED_FAMILY_GENERALIZATION:{family}"] = per_seed

    # Primary unseen-topology generalization, both arms, per-incident preserved.
    unseen_families = [name for name, _ in UNSEEN_FAMILIES]
    for arm in ("ARM_A", "ARM_B2"):
        per_family: dict[str, Any] = {}
        for family in unseen_families:
            per_seed_rows = {str(seed): [row for row in all_rows[arm][seed] if row["family"] == family] for seed in SEEDS}
            per_seed_summary = {
                seed: {"EARLY": _bucket_summary(rows, "EARLY"), "MID": _bucket_summary(rows, "MID"), "MATURE": _bucket_summary(rows, "MATURE")}
                for seed, rows in per_seed_rows.items()
            }
            per_family[family] = {"per_seed_summary": per_seed_summary, "per_incident_rows": {
                seed: [
                    {k: v for k, v in row.items() if k not in ("neural_logits",)}
                    for row in rows
                ]
                for seed, rows in per_seed_rows.items()
            }}
        topology["arms"][arm]["UNSEEN_TOPOLOGY"] = per_family

    # Section 10/11: calibration report, per seed, known-family + unseen transfer.
    def _cal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        return {
            "marginal_coverage": _mean(rows, "candidate_covered"),
            "mean_candidate_set_size": _mean(rows, "candidate_set_size"),
            "median_candidate_set_size": _median(rows, "candidate_set_size"),
            "singleton_rate": statistics.fmean(row["candidate_set_size"] == 1 for row in rows),
            "calibration_applicability_rate": _mean(rows, "calibration_applicable"),
            "by_maturity": {
                bucket: {
                    "coverage": _mean([r for r in rows if r["depth_bucket"] == bucket], "candidate_covered"),
                    "mean_set_size": _mean([r for r in rows if r["depth_bucket"] == bucket], "candidate_set_size"),
                }
                for bucket in ("EARLY", "MID", "MATURE")
            },
        }

    for arm in ("ARM_A", "ARM_B2"):
        for seed in SEEDS:
            rows_this_seed = all_rows[arm][seed]
            known_rows = [row for row in rows_this_seed if row["known"]]
            unseen_rows = [row for row in rows_this_seed if not row["known"]]
            calibration_report["arms"][arm]["per_seed"][str(seed)] = {
                "alpha": ALPHA,
                "known_family": _cal_summary(known_rows),
                "unseen_topology_calibration_transfer": _cal_summary(unseen_rows),
            }
        # Aggregate mean/min/max known-family marginal coverage across the 3 seeds (protocol Section 10/11).
        coverages = [
            calibration_report["arms"][arm]["per_seed"][str(seed)]["known_family"].get("marginal_coverage")
            for seed in SEEDS
        ]
        coverages = [c for c in coverages if c is not None]
        calibration_report["arms"][arm]["aggregate"] = {
            "mean_marginal_coverage": statistics.fmean(coverages) if coverages else None,
            "min_marginal_coverage": min(coverages) if coverages else None,
            "max_marginal_coverage": max(coverages) if coverages else None,
            "all_3_seeds_pass_0_85": all(c >= 0.85 for c in coverages) if len(coverages) == 3 else False,
            "n_seeds_passing_0_85": sum(1 for c in coverages if c >= 0.85),
        }

    locked_after = locked_test_opened(ROOT)
    for payload in (results, topology, calibration_report):
        payload["locked_test_opened_before"] = locked_before
        payload["locked_test_opened_after"] = locked_after

    OUT_RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    OUT_TOPOLOGY.write_text(json.dumps(topology, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    OUT_CALIBRATION.write_text(json.dumps(calibration_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("wrote", OUT_RESULTS, OUT_TOPOLOGY, OUT_CALIBRATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
