"""Capability diagnostic Section 20: tensor-level (logit-granularity)
reproduction spot-check.

Complementary to `reproduce_controlled_eval.py` (full 1000-example
validation-split reproduction, already committed as `reproduction.json`)
and `train_serve_parity_full.py` (which already isolated the FIRST tensor
divergence point between the training construction path and a
production-style construction path, CAP-PARITY-01). This script does NOT
re-derive raw scenario objects behind stored shards (a separate, heavier
undertaking, explicitly out of scope per the task brief) -- instead it:

1. Takes 10 real STORED validation examples directly from the committed
   sharded tensors (index i = 0, 100, 200, ..., 900 -- a stride sample
   across the full 1000-example shard, not just the first 10 rows).
2. Runs each stored tensor through the exact frozen served checkpoint
   THREE separate times and diffs the resulting `source_node_logits`
   bit-for-bit -- this rules out nondeterminism (e.g. dropout left in
   train mode, a stateful buffer, non-deterministic op) in the supposedly
   frozen eval path.
3. Reports, for each of the 10 examples: top-1 correct (y/n), the margin
   between the top-1 and top-2 softmax probability, and the predicted
   distribution's Shannon entropy -- as an independent small spot-check
   against the full 1000-example `reproduction.json` top1=0.7205 figure.

No locked-test access: only the non-locked `validation` split of
data/learning-v2/cycle-b2-joint-v4 is read (same split/corpus
`reproduce_controlled_eval.py` already used).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "phase13_sentinel_metrics", ROOT / "scripts" / "run_phase13_sentinel_metrics.py"
)
assert _spec is not None and _spec.loader is not None
_phase13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase13)

from hydroswarm.classical.metrics import localization_top_k  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.training import ShardedScenarioDataset, collate_variable_topology  # noqa: E402

FROZEN_SERVED_CHECKPOINT = ROOT / "experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810/model.safetensors"
CORPUS_ROOT = ROOT / "data" / "learning-v2" / "cycle-b2-joint-v4" / "tensors-normalized" / "validation"
STRIDE_INDICES = list(range(0, 1000, 100))  # 0, 100, ..., 900 -- 10 examples


def _entropy_bits(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log2(p)).sum())


def _forward_once(model: Any, inputs: dict[str, Any]) -> torch.Tensor:
    with torch.no_grad():
        model.eval()
        output = model(inputs)
    return output["source_node_logits"].detach().clone()


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed for this diagnostic"

    model = _phase13.load_model(FROZEN_SERVED_CHECKPOINT, use_adapters=False, strategist_fields_available=True)
    dataset = ShardedScenarioDataset(CORPUS_ROOT, expected_split="validation")
    dataset.verify_shard_checksums()

    per_example: list[dict[str, Any]] = []
    all_bit_identical = True
    max_cross_pass_abs_diff_overall = 0.0

    for index in STRIDE_INDICES:
        example = dataset[index]
        inputs, targets = collate_variable_topology([example])

        logits_runs = [_forward_once(model, inputs) for _ in range(3)]

        # --- determinism check: diff run 2 and run 3 against run 1 ---
        run_diffs = []
        for run_index in (1, 2):
            a = logits_runs[0].numpy().astype(np.float64)
            b = logits_runs[run_index].numpy().astype(np.float64)
            max_abs_diff = float(np.abs(a - b).max())
            bit_identical = bool(np.array_equal(logits_runs[0].numpy(), logits_runs[run_index].numpy()))
            run_diffs.append({"run_pair": f"1_vs_{run_index + 1}", "max_abs_diff": max_abs_diff, "bit_identical": bit_identical})
            max_cross_pass_abs_diff_overall = max(max_cross_pass_abs_diff_overall, max_abs_diff)
            if not bit_identical:
                all_bit_identical = False

        # --- logit/probability spot-check metrics (using run 1's output) ---
        source_mask = targets.get("source_node_mask")
        localization_eligible = source_mask is None or bool(source_mask[0])
        result: dict[str, Any] = {
            "dataset_index": index,
            "scenario_id": str(example.scenario_id) if hasattr(example, "scenario_id") else None,
            "determinism_check": run_diffs,
            "localization_eligible": localization_eligible,
        }
        if localization_eligible and "source_node" in targets:
            probs = torch.softmax(logits_runs[0][0], dim=-1).numpy().astype(np.float64)
            truth = int(targets["source_node"][0].item())
            sorted_probs = np.sort(probs)[::-1]
            top1_prob = float(sorted_probs[0])
            top2_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
            belief = {position: float(value) for position, value in enumerate(probs) if value > 0}
            top1_correct = bool(truth in belief and localization_top_k(belief, truth, k=1))
            top3_correct = bool(truth in belief and localization_top_k(belief, truth, k=3))
            result.update({
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
                "top1_probability": top1_prob,
                "top2_probability": top2_prob,
                "top1_top2_margin": top1_prob - top2_prob,
                "entropy_bits": _entropy_bits(probs),
                "true_source_probability": float(probs[truth]) if truth < len(probs) else None,
            })
        else:
            result["note"] = "not localization-eligible (source_node_mask False or target absent) for this stored example"
        per_example.append(result)

    eligible = [r for r in per_example if r.get("top1_correct") is not None]
    spot_check_top1 = (sum(1 for r in eligible if r["top1_correct"]) / len(eligible)) if eligible else None
    spot_check_top3 = (sum(1 for r in eligible if r["top3_correct"]) / len(eligible)) if eligible else None
    spot_check_mean_margin = (sum(r["top1_top2_margin"] for r in eligible) / len(eligible)) if eligible else None
    spot_check_mean_entropy = (sum(r["entropy_bits"] for r in eligible) / len(eligible)) if eligible else None

    full_reproduction_path = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "reproduction.json"
    full_reproduction_top1 = None
    if full_reproduction_path.exists():
        full_reproduction_top1 = json.loads(full_reproduction_path.read_text()).get("reproduced", {}).get("top1")

    locked_after = locked_test_opened(ROOT)
    report = {
        "schema_version": 1,
        "section": "20_tensor_level_reproduction",
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "checkpoint_evaluated": str(FROZEN_SERVED_CHECKPOINT.relative_to(ROOT)),
        "corpus": str(CORPUS_ROOT.relative_to(ROOT)),
        "stride_indices_sampled": STRIDE_INDICES,
        "n_examples": len(per_example),
        "determinism": {
            "all_three_repeated_forward_passes_bit_identical_for_every_example": all_bit_identical,
            "max_cross_pass_abs_logit_diff_across_all_examples": max_cross_pass_abs_diff_overall,
            "verdict": (
                "DETERMINISTIC -- no nondeterminism detected in the frozen eval path across 3 repeated forward "
                "passes on 10 stored examples."
                if all_bit_identical
                else "NONDETERMINISM DETECTED -- escalation-worthy: repeated forward passes on the same stored "
                "tensor batch produced different logits on the supposedly frozen eval path."
            ),
        },
        "per_example": per_example,
        "spot_check_summary": {
            "n_eligible": len(eligible),
            "top1": spot_check_top1,
            "top3": spot_check_top3,
            "mean_top1_top2_margin": spot_check_mean_margin,
            "mean_entropy_bits": spot_check_mean_entropy,
        },
        "comparison_to_full_1000_example_reproduction": {
            "full_reproduction_top1": full_reproduction_top1,
            "spot_check_top1_n10": spot_check_top1,
            "consistent": (
                abs(full_reproduction_top1 - spot_check_top1) <= 0.30
                if (full_reproduction_top1 is not None and spot_check_top1 is not None)
                else None
            ),
            "caveat": (
                "N=10 stride-sampled examples is far too small to precisely match a 1000-example top1 rate "
                "(binomial noise at n=10 is roughly +-15 points at this base rate); this is a sanity spot-check "
                "for gross inconsistency (e.g. tensor corruption, wrong split), not an independent precision "
                "estimate."
            ),
        },
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "tensor-level-reproduction.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "determinism_verdict": report["determinism"]["verdict"],
        "spot_check_summary": report["spot_check_summary"],
        "comparison_to_full_1000_example_reproduction": report["comparison_to_full_1000_example_reproduction"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
