"""core-issues3.txt Phase 13: required Strategist physical-unit metrics not yet
computed anywhere.

Gap this closes: reports/results/v4/stage-e-strategist-comparison-v4corpus-corrected.json
already compares 4 policies on plan_value/regret/simulator-calls, but never
converts to physical units (exposure mg, pressure-violation minutes, service
availability, containment time) or computes an NDCG-style ranking-quality
metric or proxy-head error in physical units -- only normalized-scale MSEs
exist (reports/results/v4/strategist-heads-training-v4corpus-corrected.json).

Per targets_v2.py's own governed unit declarations, exposure_proxy/
pressure_risk_proxy/service_loss_proxy/containment_time_proxy/
plan_regret_proxy are ALREADY stored (and the model already predicts them)
in physical units (mg, pressure-violation minutes, fraction, seconds,
value-scale) -- no inverse transform is needed; this script just aggregates
what run_stage_e_strategist_comparison.py's own per-candidate ScenarioCandidates
view leaves on the table.

Reuses scripts/run_stage_e_strategist_comparison.py's own established data
access pattern (real WNTR-verified targets already stored in
data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized-corrected,
Phase 3.1's full-candidate-set verification -- no re-simulation, no locked
test, no new corpus generation) and its own already-trained, on-disk
v4-strategist-heads-v4corpus-corrected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage_e_strategist_comparison import (  # noqa: E402
    ACTION_TEMPLATES,
    ELIGIBILITY_THRESHOLD,
    SHORTLIST_K,
    build_heuristic_template_scores,
    default_strategist_checkpoint,
    load_strategist_model,
)

from hydroswarm.training import ShardedScenarioDataset, collate_variable_topology  # noqa: E402

PROXY_FIELDS = ("exposure_proxy", "pressure_risk_proxy", "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy")
POLICIES = ("exact_all", "deterministic_heuristic", "learned_ordering", "learned_prescreen")


class Candidates:
    def __init__(self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], row: int) -> None:
        mask = inputs["plan_mask"][row].bool()
        self.positions = mask.nonzero(as_tuple=True)[0].tolist()
        template_ids = inputs["plan_template_ids"][row]
        self.template_names = [ACTION_TEMPLATES[int(template_ids[p])] for p in self.positions]
        self.valid = [bool(targets["plan_validity"][row][p]) for p in self.positions]
        self.value = [float(targets["plan_value"][row][p]) for p in self.positions]
        self.value_mask = [bool(targets["plan_value_mask"][row][p]) for p in self.positions]
        self.count = len(self.positions)

        self.proxy_target: dict[str, list[float]] = {}
        self.proxy_mask: dict[str, list[bool]] = {}
        self.proxy_predicted: dict[str, list[float]] = {}
        for field in PROXY_FIELDS:
            self.proxy_target[field] = [float(targets[field][row][p]) for p in self.positions]
            self.proxy_mask[field] = [bool(targets[f"{field}_mask"][row][p]) for p in self.positions]

    def ground_truth_value(self, position: int) -> float:
        return self.value[position] if self.value_mask[position] else 0.0

    def no_action_index(self) -> int | None:
        for position, name in enumerate(self.template_names):
            if name == "NO_ACTION":
                return position
        return None


def _select(candidates: Candidates, order: list[int]) -> int | None:
    """Same selection rule as run_stage_e_strategist_comparison.py's
    _select_from_shortlist: first-encountered-or-better VALID candidate by
    ground-truth value; None if none valid (caller falls back to
    NO_ACTION)."""

    best_position: int | None = None
    best_value = float("-inf")
    for position in order:
        if candidates.valid[position]:
            value = candidates.ground_truth_value(position)
            if value > best_value:
                best_value = value
                best_position = position
    return best_position


def _policy_order(policy: str, candidates: Candidates, heuristic_scores: dict[str, tuple[float, float]], predicted_value: list[float], predicted_valid_probability: list[float]) -> list[int]:
    if policy == "exact_all":
        return sorted(range(candidates.count), key=lambda p: -candidates.ground_truth_value(p))
    if policy == "deterministic_heuristic":
        eligible = [p for p in range(candidates.count) if heuristic_scores[candidates.template_names[p]][1] >= ELIGIBILITY_THRESHOLD]
        eligible.sort(key=lambda p: (candidates.template_names[p] == "NO_ACTION", -(heuristic_scores[candidates.template_names[p]][0] * heuristic_scores[candidates.template_names[p]][1])))
        return eligible[:SHORTLIST_K]
    if policy == "learned_ordering":
        return sorted(range(candidates.count), key=lambda p: -predicted_value[p])[:1]
    if policy == "learned_prescreen":
        eligible = [p for p in range(candidates.count) if predicted_valid_probability[p] >= ELIGIBILITY_THRESHOLD]
        eligible.sort(key=lambda p: -(predicted_value[p] * predicted_valid_probability[p]))
        return eligible[:SHORTLIST_K]
    raise ValueError(policy)


def _ndcg_at_k(order: list[int], candidates: Candidates, *, k: int) -> float:
    """NDCG over the policy's own ranked shortlist against ground-truth
    plan_value relevance (shifted non-negative per scenario, since raw
    plan_value can be negative and DCG gain requires relevance >= 0)."""

    values = [candidates.ground_truth_value(p) for p in range(candidates.count)]
    if not values:
        return 0.0
    floor = min(values)
    relevance = {p: values[p] - floor for p in range(candidates.count)}
    ranked = order[:k] if order else []
    dcg = sum(relevance[p] / np.log2(index + 2) for index, p in enumerate(ranked))
    ideal_order = sorted(range(candidates.count), key=lambda p: -relevance[p])[:k]
    idcg = sum(relevance[p] / np.log2(index + 2) for index, p in enumerate(ideal_order))
    return float(dcg / idcg) if idcg > 0 else 1.0


@torch.no_grad()
def run(*, corpus_root: Path, split: str, checkpoint: Path, limit: int, batch_size: int) -> dict[str, Any]:
    import psutil

    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss

    heuristic_scores = build_heuristic_template_scores()
    model = load_strategist_model(checkpoint)

    dataset = ShardedScenarioDataset(corpus_root / split, expected_split=split)
    dataset.verify_shard_checksums()
    total = len(dataset)
    evaluate_count = min(limit, total) if limit else total
    indices = list(range(total))
    if evaluate_count < total:
        stride = max(1, total // evaluate_count)
        indices = indices[::stride][:evaluate_count]

    per_policy: dict[str, dict[str, list[float]]] = {
        policy: {"exposure_reduction_vs_no_action": [], "pressure_violation_minutes": [], "service_availability": [], "containment_time_minutes": [], "ndcg": []}
        for policy in POLICIES
    }
    proxy_errors: dict[str, list[float]] = {field: [] for field in PROXY_FIELDS}
    latencies: list[float] = []
    skipped = 0

    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        examples = [dataset[i] for i in batch_indices]
        inputs, targets = collate_variable_topology(examples)
        started_forward = time.perf_counter()
        output = model(inputs)
        latencies.append((time.perf_counter() - started_forward) / len(examples))
        peak_rss = max(peak_rss, process.memory_info().rss)
        batch_value = output["plan_value"].float()
        batch_valid_probability = torch.softmax(output["plan_validity_logits"].float(), dim=-1)[..., 1]

        for row in range(len(examples)):
            candidates = Candidates(inputs, targets, row)
            if candidates.count == 0:
                skipped += 1
                continue
            no_action_index = candidates.no_action_index()
            predicted_value = batch_value[row, : candidates.count].tolist()
            predicted_valid_probability = batch_valid_probability[row, : candidates.count].tolist()

            for field in PROXY_FIELDS:
                predicted = output[field][row, : candidates.count].tolist()
                for position, mask_ok in enumerate(candidates.proxy_mask[field]):
                    if mask_ok:
                        proxy_errors[field].append(abs(predicted[position] - candidates.proxy_target[field][position]))

            for policy in POLICIES:
                order = _policy_order(policy, candidates, heuristic_scores, predicted_value, predicted_valid_probability)
                selected = _select(candidates, order)
                if selected is None:
                    selected = no_action_index
                if selected is None:
                    continue

                metrics = per_policy[policy]
                if no_action_index is not None and candidates.proxy_mask["exposure_proxy"][no_action_index] and candidates.proxy_mask["exposure_proxy"][selected]:
                    metrics["exposure_reduction_vs_no_action"].append(
                        candidates.proxy_target["exposure_proxy"][no_action_index] - candidates.proxy_target["exposure_proxy"][selected]
                    )
                if candidates.proxy_mask["pressure_risk_proxy"][selected]:
                    metrics["pressure_violation_minutes"].append(candidates.proxy_target["pressure_risk_proxy"][selected])
                if candidates.proxy_mask["service_loss_proxy"][selected]:
                    metrics["service_availability"].append(1.0 - candidates.proxy_target["service_loss_proxy"][selected])
                if candidates.proxy_mask["containment_time_proxy"][selected]:
                    metrics["containment_time_minutes"].append(candidates.proxy_target["containment_time_proxy"][selected] / 60.0)
                metrics["ndcg"].append(_ndcg_at_k(order, candidates, k=SHORTLIST_K))

        if (batch_start // batch_size + 1) % 10 == 0:
            print(f"  ... {min(batch_start + batch_size, len(indices))}/{len(indices)} scenarios ({time.perf_counter() - started:.0f}s)")

    summary: dict[str, Any] = {}
    for policy in POLICIES:
        metrics = per_policy[policy]
        summary[policy] = {name: (float(np.mean(values)) if values else None) for name, values in metrics.items()}
        summary[policy]["scenarios_with_data"] = {name: len(values) for name, values in metrics.items()}

    proxy_summary: dict[str, Any] = {}
    for field, errors in proxy_errors.items():
        if errors:
            proxy_summary[field] = {
                "mean_absolute_error": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "valid_count": len(errors),
                "unit": {
                    "exposure_proxy": "mg",
                    "pressure_risk_proxy": "pressure-violation minutes",
                    "service_loss_proxy": "fraction [0,1]",
                    "containment_time_proxy": "seconds",
                    "plan_regret_proxy": "plan_value scale",
                }[field],
            }

    return {
        "corpus": str(corpus_root),
        "split": split,
        "checkpoint": str(checkpoint),
        "scenarios_requested": len(indices),
        "scenarios_evaluated": len(indices) - skipped,
        "skipped_no_real_candidates": skipped,
        "policies": summary,
        "proxy_error_physical_units": proxy_summary,
        "mean_latency_seconds": float(np.mean(latencies)) if latencies else 0.0,
        "peak_rss_bytes": int(peak_rss),
        "wall_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-root", type=Path, default=Path("data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized-corrected"))
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/phase13-strategist-physical-metrics.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint or default_strategist_checkpoint()
    result = run(corpus_root=args.corpus_root, split=args.split, checkpoint=checkpoint, limit=args.limit, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["policies"], indent=2))
    print(json.dumps(result["proxy_error_physical_units"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
