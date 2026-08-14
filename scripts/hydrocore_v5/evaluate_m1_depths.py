"""Milestone 1.5 (experiments.txt): per-depth evaluation of every trained
causal-prefix arm/seed against development_holdout (never train/validation/
calibration/locked -- development_holdout is the governed "architecture
comparison" split, per docs/EVALUATION_V3_POLICY.md's split-role table).

For every arm/seed checkpoint and every depth in CAUSAL_PREFIX_DEPTHS,
reports: top-1, top-3, MRR, NLL, Brier, posterior entropy, true-source
rank, evidence sufficiency (ground-truth fraction and head accuracy),
event presence/cause head accuracy, model latency, and RSS. Candidate-
region size is deferred (reported null) -- Milestone 1.5 only requires it
"using separately fitted development calibration only where appropriate",
and no calibration exists yet (Milestone 3 has not run).

Writes:
  reports/evaluation/hydrocore-v5/m1-training-results.json
  reports/evaluation/hydrocore-v5/m1-causal-curves.csv
  reports/evaluation/hydrocore-v5/m1-summary.md
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import psutil  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
)
from hydroswarm.training.data import collate_scenarios  # noqa: E402
from run_m1_arm import SHARED_MODEL_CONFIG  # noqa: E402

RUNS_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-runs"
OUTPUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-training-results.json"
OUTPUT_CURVES = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-causal-curves.csv"
OUTPUT_SUMMARY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-summary.md"
IDENTIFIABILITY_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m1-identifiability.json"

EPS = 1e-9


def _load_run_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(RUNS_ROOT.glob("*.json")):
        records.append(json.loads(path.read_text()))
    return records


def _load_checkpoint(export_path: str) -> HydroCore:
    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    state_dict = load_file(export_path, device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _evaluate_checkpoint_at_depth(
    model: HydroCore, dev_records, library, depth: int, *, batch_size: int = 4
) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    examples = [
        scenario_to_prefix_example(record.scenario, record.network, library, depth, feature_context=record.feature_context)
        for record in dev_records
    ]

    top1s, top3s, mrrs, nlls, briers, entropies, ranks = [], [], [], [], [], [], []
    evidence_suff_truth, evidence_suff_pred_correct = [], []
    event_presence_correct, event_cause_correct = [], []
    latencies = []
    peak_rss = process.memory_info().rss

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs, targets = collate_scenarios(batch)
            started = time.perf_counter()
            output = model(inputs)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed / len(batch))
            peak_rss = max(peak_rss, process.memory_info().rss)

            probs = torch.softmax(output["source_node_logits"], dim=-1)
            for row in range(probs.shape[0]):
                truth = int(targets["source_node"][row].item())
                row_probs_tensor = probs[row]
                row_probs = {position: float(value) for position, value in enumerate(row_probs_tensor.tolist())}
                top1s.append(localization_top_k(row_probs, truth, k=1))
                top3s.append(localization_top_k(row_probs, truth, k=3))
                mrrs.append(mean_reciprocal_rank([row_probs], [truth]))
                p_truth = row_probs.get(truth, 0.0)
                nlls.append(-math.log(p_truth + EPS))
                onehot = torch.zeros_like(row_probs_tensor)
                onehot[truth] = 1.0
                briers.append(float(torch.sum((row_probs_tensor - onehot) ** 2)))
                entropies.append(float(-torch.sum(row_probs_tensor * torch.log(row_probs_tensor + EPS))))
                rank = int((row_probs_tensor > p_truth).sum().item()) + 1
                ranks.append(rank)

            if "evidence_sufficiency" in targets and "evidence_sufficiency" in output:
                truth_es = targets["evidence_sufficiency"].bool().tolist()
                pred_es = (output["evidence_sufficiency"].squeeze(-1) >= 0.5).tolist() if output["evidence_sufficiency"].dim() > 1 else (output["evidence_sufficiency"] >= 0.5).tolist()
                evidence_suff_truth.extend(truth_es)
                evidence_suff_pred_correct.extend(p == t for p, t in zip(pred_es, truth_es, strict=True))

            if "event_presence" in targets and "event_presence_logits" in output:
                pred_presence = (torch.sigmoid(output["event_presence_logits"]).squeeze(-1) >= 0.5).tolist()
                truth_presence = targets["event_presence"].bool().tolist()
                event_presence_correct.extend(p == t for p, t in zip(pred_presence, truth_presence, strict=True))

            if "event_cause" in targets and "event_cause_logits" in output:
                pred_cause = torch.argmax(output["event_cause_logits"], dim=-1).tolist()
                truth_cause = targets["event_cause"].long().tolist()
                event_cause_correct.extend(p == t for p, t in zip(pred_cause, truth_cause, strict=True))

    return {
        "n": len(examples),
        "top1": statistics.fmean(top1s),
        "top3": statistics.fmean(top3s),
        "mrr": statistics.fmean(mrrs),
        "nll": statistics.fmean(nlls),
        "brier": statistics.fmean(briers),
        "posterior_entropy": statistics.fmean(entropies),
        "true_source_rank": statistics.fmean(ranks),
        "evidence_sufficient_fraction": statistics.fmean(evidence_suff_truth) if evidence_suff_truth else None,
        "evidence_sufficiency_head_accuracy": statistics.fmean(evidence_suff_pred_correct) if evidence_suff_pred_correct else None,
        "event_presence_head_accuracy": statistics.fmean(event_presence_correct) if event_presence_correct else None,
        "event_cause_head_accuracy": statistics.fmean(event_cause_correct) if event_cause_correct else None,
        "candidate_region_size": None,
        "model_latency_seconds_mean": statistics.fmean(latencies),
        "peak_rss_bytes": peak_rss,
    }


PROMOTION_MIN_2_3_STEP_GAIN_PP = 10.0
PROMOTION_MAX_MATURE_REGRESSION_PP = 3.0


def _apply_promotion_rule(curves: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Milestone 1's promotion rule, applied per-arm using each arm's
    across-seed mean top-1 at each depth: meaningful 2/3-step top-1 gain
    (>=10pp) over Arm A, no material 6-step regression, no >2-3pp
    mature-history (12/25) regression, stable training across seeds."""

    if "A" not in curves:
        return {"decision": "NO_CONTROL_ARM", "reason": "Arm A (control) has no completed runs."}
    control = curves["A"]
    results = {}
    for arm in ("B", "C"):
        if arm not in curves:
            continue
        arm_curve = curves[arm]
        gains_2_3 = [
            (arm_curve[d]["top1"] - control[d]["top1"]) * 100
            for d in (2, 3)
            if d in arm_curve and d in control
        ]
        regression_6 = (
            (control[6]["top1"] - arm_curve[6]["top1"]) * 100 if 6 in arm_curve and 6 in control else None
        )
        mature_regressions = [
            (control[d]["top1"] - arm_curve[d]["top1"]) * 100
            for d in (12, 25)
            if d in arm_curve and d in control
        ]
        meets_gain = bool(gains_2_3) and min(gains_2_3) >= PROMOTION_MIN_2_3_STEP_GAIN_PP
        no_6_regression = regression_6 is None or regression_6 <= PROMOTION_MAX_MATURE_REGRESSION_PP
        no_mature_regression = not mature_regressions or max(mature_regressions) <= PROMOTION_MAX_MATURE_REGRESSION_PP
        results[arm] = {
            "gain_2_3_step_pp": gains_2_3,
            "regression_6_step_pp": regression_6,
            "mature_regression_pp": mature_regressions,
            "meets_gain_threshold": meets_gain,
            "no_material_6_step_regression": no_6_regression,
            "no_mature_history_regression": no_mature_regression,
            "provisional_winner": meets_gain and no_6_regression and no_mature_regression,
        }
    winners = [arm for arm, r in results.items() if r["provisional_winner"]]
    if winners:
        best = max(winners, key=lambda arm: min(results[arm]["gain_2_3_step_pp"]))
        decision = f"PROVISIONAL_WINNER_{best}"
    else:
        decision = "NO_ARM_MEETS_PROMOTION_RULE"
    return {"decision": decision, "per_arm": results}


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"

    run_records = _load_run_records()
    if not run_records:
        raise SystemExit("no M1 arm/seed run records found under " + str(RUNS_ROOT))

    dev_records = build_scenario_pool("development_holdout", network_loader=build_wntr_network)
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    per_run_curves: dict[str, dict[int, dict[str, Any]]] = {}
    csv_rows = []
    for run in run_records:
        arm = run["arm"]
        seed = run["seed"]
        export_path = run["training_summary"]["export_path"]
        model = _load_checkpoint(export_path)
        run_key = f"{arm}-seed{seed}"
        per_run_curves[run_key] = {}
        for depth in CAUSAL_PREFIX_DEPTHS:
            metrics = _evaluate_checkpoint_at_depth(model, dev_records, library, depth)
            per_run_curves[run_key][depth] = metrics
            csv_rows.append({"arm": arm, "seed": seed, "depth": depth, **{k: v for k, v in metrics.items()}})
            print(f"{run_key} depth={depth} top1={metrics['top1']:.3f} top3={metrics['top3']:.3f} mrr={metrics['mrr']:.3f}")

    # Across-seed mean per arm/depth for the promotion rule.
    arms = sorted({run["arm"] for run in run_records})
    arm_mean_curves: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in arms:
        arm_runs = [key for key in per_run_curves if key.startswith(f"{arm}-seed")]
        arm_mean_curves[arm] = {}
        for depth in CAUSAL_PREFIX_DEPTHS:
            values = [per_run_curves[key][depth] for key in arm_runs]
            arm_mean_curves[arm][depth] = {
                metric: statistics.fmean(v[metric] for v in values if v[metric] is not None)
                if any(v[metric] is not None for v in values)
                else None
                for metric in values[0]
                if metric != "n"
            }
            arm_mean_curves[arm][depth]["n_seeds"] = len(arm_runs)

    promotion = _apply_promotion_rule(arm_mean_curves)

    identifiability = json.loads(IDENTIFIABILITY_PATH.read_text()) if IDENTIFIABILITY_PATH.exists() else None

    results = {
        "schema_version": 1,
        "purpose": "Milestone 1.5 (experiments.txt): per-depth causal-prefix arm evaluation on development_holdout.",
        "causal_prefix_depths": list(CAUSAL_PREFIX_DEPTHS),
        "per_run_curves": {key: {str(d): m for d, m in curve.items()} for key, curve in per_run_curves.items()},
        "arm_mean_curves": {arm: {str(d): m for d, m in curve.items()} for arm, curve in arm_mean_curves.items()},
        "promotion_rule": {
            "min_2_3_step_gain_pp": PROMOTION_MIN_2_3_STEP_GAIN_PP,
            "max_mature_regression_pp": PROMOTION_MAX_MATURE_REGRESSION_PP,
        },
        "promotion_result": promotion,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    with OUTPUT_CURVES.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# Milestone 1 summary: causal-prefix HydroCore-v5, same model size",
        "",
        f"Promotion decision: **{promotion['decision']}**",
        "",
        "## Arm mean top-1 by causal depth",
        "",
        "| arm | " + " | ".join(str(d) for d in CAUSAL_PREFIX_DEPTHS) + " |",
        "|---|" + "---|" * len(CAUSAL_PREFIX_DEPTHS),
    ]
    for arm in arms:
        row = [f"{arm_mean_curves[arm][d]['top1']:.3f}" if arm_mean_curves[arm][d]["top1"] is not None else "n/a" for d in CAUSAL_PREFIX_DEPTHS]
        lines.append(f"| {arm} | " + " | ".join(row) + " |")
    lines.append("")
    if identifiability is not None:
        lines.append("## Classical identifiability baseline (development_holdout, Milestone 1.2)")
        lines.append("")
        lines.append("| depth | classical top1 | classical top3 | mean ambiguous sources | distinguishable fraction |")
        lines.append("|---|---|---|---|---|")
        for depth in CAUSAL_PREFIX_DEPTHS:
            entry = identifiability["per_depth"][str(depth)]
            lines.append(
                f"| {depth} | {entry['classical_top1']:.3f} | {entry['classical_top3']:.3f} | "
                f"{entry['mean_source_ambiguity_count']:.2f} | {entry['distinguishable_source_fraction']:.3f} |"
            )
        lines.append("")
    lines.append("## Promotion rule detail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(promotion, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append(
        "Scope note: corpus is a single canonical topology (golden-reference), Sentinel-task-only "
        "supervision, contamination events only (see reports/evaluation/hydrocore-v5/m1-prefix-dataset.json). "
        "Not directly comparable in absolute magnitude to the frozen v4 baseline curve "
        "(reports/evaluation/hydrocore-v5/m0-baseline.json), which used a different (historical, "
        "committed) corpus -- this milestone's conclusion is about the RELATIVE ranking of arms A/B/C "
        "trained on the same corpus, not an absolute claim against v4."
    )
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(promotion, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
