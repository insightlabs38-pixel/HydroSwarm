"""Paired bootstrap follow-up for the topology-generalization pilot
(run_pilot.py). Loads the two already-trained arm checkpoints (no
retraining) and recomputes raw per-example top1/top3/reciprocal-rank rows
on validation/development_holdout/ood-UNSEEN_TOPOLOGY, then reports a
paired bootstrap CI on the CONTROL-vs-EXPERIMENTAL delta, matching the
2000-resample/90%-interval convention already used by
reports/evaluation/hydrocore-v5/m9-0-summary.md and m9-6-summary.md.

EXPERIMENTAL / NON-RELEASE. See run_pilot.py's own module docstring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pilot as rp  # noqa: E402
from hydroswarm.training.sharded_data import ShardedScenarioDataset  # noqa: E402

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_INTERVAL = 0.90

RUN_DIRS = {
    "CONTROL": ROOT / "experiments" / "topology-generalization" / "runs" / "CONTROL" / "20260825T213148Z-7efa6835",
    "EXPERIMENTAL_TOPOLOGY_RELATIVE": ROOT
    / "experiments"
    / "topology-generalization"
    / "runs"
    / "EXPERIMENTAL_TOPOLOGY_RELATIVE"
    / "20260825T214003Z-3b88f361",
}


def load_arm_model(name: str, *, augmented: bool):
    model = rp.build_model(augmented=augmented, seed=rp.SEED)
    state = load_file(RUN_DIRS[name] / "model-export.safetensors")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def collect_rows(model, *, augmented: bool, dataset: ShardedScenarioDataset, indices: list[int]) -> list[dict]:
    ood_detector = rp.OODDetector(rp.OODReference())
    rows = []
    for index in indices:
        example = dataset[index]
        row = rp._row_metrics(model, example, augmented=augmented, ood_detector=ood_detector)
        rows.append(row)
    return rows


def paired_bootstrap(control: list[float], experimental: list[float], *, seed: int, resamples: int, interval: float) -> dict:
    assert len(control) == len(experimental)
    control_arr = np.asarray(control)
    experimental_arr = np.asarray(experimental)
    observed = float(np.mean(experimental_arr) - np.mean(control_arr))
    rng = np.random.default_rng(seed)
    n = len(control_arr)
    deltas = np.empty(resamples)
    for index in range(resamples):
        sample = rng.integers(0, n, size=n)
        deltas[index] = experimental_arr[sample].mean() - control_arr[sample].mean()
    alpha = (1 - interval) / 2
    lower, upper = np.quantile(deltas, [alpha, 1 - alpha])
    return {"observed": observed, "ci_low": float(lower), "ci_high": float(upper), "n": n, "resamples": resamples}


def main() -> None:
    corpus = rp.CORPUS_ROOT
    validation_full = ShardedScenarioDataset(corpus / "validation", expected_split="validation")
    dev_holdout_full = ShardedScenarioDataset(corpus / "development_holdout", expected_split="development_holdout")
    ood_full = ShardedScenarioDataset(corpus / "ood-UNSEEN_TOPOLOGY", expected_split="development_holdout")

    validation_indices = rp.capped_indices(validation_full, limit=rp.EVAL_VALIDATION_LIMIT, seed=rp.SEED)
    dev_holdout_indices = rp.capped_indices(dev_holdout_full, limit=rp.EVAL_DEV_HOLDOUT_LIMIT, seed=rp.SEED)
    # ood-UNSEEN_TOPOLOGY: same real-source filter as run_pilot.main(), but
    # unlike run_pilot.py we use ALL of it here (not just a cap) for the
    # tightest CI this pilot's compute budget affords.
    ood_indices = [index for index in range(len(ood_full)) if rp.has_real_source(ood_full, index)]

    populations = {
        "validation": (validation_full, validation_indices),
        "development_holdout": (dev_holdout_full, dev_holdout_indices),
        "ood-UNSEEN_TOPOLOGY": (ood_full, ood_indices),
    }

    arm_rows: dict[str, dict[str, list[dict]]] = {}
    for name, augmented in (("CONTROL", False), ("EXPERIMENTAL_TOPOLOGY_RELATIVE", True)):
        print(f"Loading {name} checkpoint and re-evaluating (augmented={augmented})...")
        model = load_arm_model(name, augmented=augmented)
        arm_rows[name] = {population: collect_rows(model, augmented=augmented, dataset=dataset, indices=indices) for population, (dataset, indices) in populations.items()}

    results: dict[str, dict] = {}
    for population in populations:
        control_rows = arm_rows["CONTROL"][population]
        experimental_rows = arm_rows["EXPERIMENTAL_TOPOLOGY_RELATIVE"][population]
        assert [row["scenario_id"] for row in control_rows] == [row["scenario_id"] for row in experimental_rows], (
            "paired bootstrap requires identical example order between arms"
        )
        top1_control = [row["top1"] for row in control_rows]
        top1_experimental = [row["top1"] for row in experimental_rows]
        top3_control = [row["top3"] for row in control_rows]
        top3_experimental = [row["top3"] for row in experimental_rows]
        rr_control = [row["reciprocal_rank"] for row in control_rows]
        rr_experimental = [row["reciprocal_rank"] for row in experimental_rows]
        results[population] = {
            "n": len(control_rows),
            "control_top1_mean": float(np.mean(top1_control)),
            "experimental_top1_mean": float(np.mean(top1_experimental)),
            "top1_delta": paired_bootstrap(top1_control, top1_experimental, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, interval=BOOTSTRAP_INTERVAL),
            "top3_delta": paired_bootstrap(top3_control, top3_experimental, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, interval=BOOTSTRAP_INTERVAL),
            "mrr_delta": paired_bootstrap(rr_control, rr_experimental, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, interval=BOOTSTRAP_INTERVAL),
        }
        print(f"{population}: top1 CONTROL={results[population]['control_top1_mean']:.4f} EXPERIMENTAL={results[population]['experimental_top1_mean']:.4f} "
              f"delta={results[population]['top1_delta']['observed']:+.4f} 90% CI [{results[population]['top1_delta']['ci_low']:+.4f}, {results[population]['top1_delta']['ci_high']:+.4f}]")

    output_path = ROOT / "reports" / "evaluation" / "topology-generalization" / "paired-bootstrap.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
