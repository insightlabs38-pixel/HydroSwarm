"""Milestone 8.7 third-seed closure, Section 12: paired incident bootstrap,
AGE_FIX_ONLY (seed 20260815, the preregistered third seed) vs CURRENT_CONTROL
(representative seed 31874, consistent with this milestone's existing
convention of using the SCREENING_REPRESENTATIVE_SEED for cross-arm
comparisons elsewhere in the pipeline -- calibration, cadence/irregular-
telemetry diagnostics), per the frozen protocol
(`docs/evaluation/HYDROCORE_V5_M8_7_PROTOCOL.md`, Section 10):
"a paired bootstrap over incidents (2,000 resamples, seed=20260815...) is
used to report a 90% interval on the difference" for a promotion-relevant
A-vs-B metric difference reported at the 3-seed stage (here: mean top1,
paired per development_holdout incident, averaged over the standard depth
grid, matching `run_m8_7_evaluate.py`'s own EARLY/MID/MATURE aggregation
depths).

Does NOT retrain or re-decide anything; reads existing checkpoints only.

Writes:
  reports/evaluation/hydrocore-v5/m8-7-closure.json (bootstrap key merged in)
"""

from __future__ import annotations

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

from hydroswarm.classical.metrics import localization_top_k  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
)
from run_m8_7_arm import ARM_DEFINITIONS  # noqa: E402
from run_m8_7_evaluate import RUNS_ROOT, _load_checkpoint  # noqa: E402

CLOSURE_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-7-closure.json"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260815
INTERVAL = 0.90
THIRD_SEED_ARM, THIRD_SEED = "AGE_FIX_ONLY", 20260815
CONTROL_ARM, CONTROL_SEED = "CURRENT_CONTROL", 31874


def _per_incident_mean_top1(model, records, library, feature_kwargs: dict[str, Any]) -> list[float]:
    per_incident = []
    with torch.no_grad():
        for record in records:
            depth_top1s = []
            for depth in CAUSAL_PREFIX_DEPTHS:
                example = scenario_to_prefix_example(
                    record.scenario, record.network, library, depth, feature_context=record.feature_context, **feature_kwargs,
                )
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                probs = torch.softmax(output["source_node_logits"][0], dim=-1)
                row_probs = {position: float(value) for position, value in enumerate(probs.tolist())}
                truth = int(example.targets["source_node"].item())
                depth_top1s.append(localization_top_k(row_probs, truth, k=1))
            per_incident.append(statistics.fmean(depth_top1s))
    return per_incident


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    third_record = json.loads((RUNS_ROOT / f"{THIRD_SEED_ARM}-seed{THIRD_SEED}.json").read_text())
    control_record = json.loads((RUNS_ROOT / f"{CONTROL_ARM}-seed{CONTROL_SEED}.json").read_text())
    third_model = _load_checkpoint(third_record)
    control_model = _load_checkpoint(control_record)

    dev_records = build_scenario_pool("development_holdout", network_loader=build_wntr_network)
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    third_feature_kwargs = ARM_DEFINITIONS[THIRD_SEED_ARM]["feature_kwargs"]
    control_feature_kwargs = ARM_DEFINITIONS[CONTROL_ARM]["feature_kwargs"]

    third_per_incident = _per_incident_mean_top1(third_model, dev_records, library, third_feature_kwargs)
    control_per_incident = _per_incident_mean_top1(control_model, dev_records, library, control_feature_kwargs)
    assert len(third_per_incident) == len(control_per_incident) == len(dev_records)

    paired_diff = np.array(third_per_incident) - np.array(control_per_incident)
    observed_mean_diff = float(paired_diff.mean())

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(paired_diff)
    resample_means = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        resample_means[i] = paired_diff[idx].mean()
    lower_pct = (1 - INTERVAL) / 2 * 100
    upper_pct = (1 - (1 - INTERVAL) / 2) * 100
    ci_lower = float(np.percentile(resample_means, lower_pct))
    ci_upper = float(np.percentile(resample_means, upper_pct))

    locked_after = locked_test_opened(ROOT)

    bootstrap_report = {
        "purpose": (
            f"Paired incident bootstrap, {THIRD_SEED_ARM} (seed {THIRD_SEED}) minus {CONTROL_ARM} "
            f"(representative seed {CONTROL_SEED}), mean top1 averaged over the standard depth grid "
            "per development_holdout incident (frozen protocol Section 10)."
        ),
        "arm_a": THIRD_SEED_ARM, "seed_a": THIRD_SEED,
        "arm_b": CONTROL_ARM, "seed_b": CONTROL_SEED,
        "metric": "mean_top1_over_depths_per_incident",
        "n_incidents": n,
        "resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interval": INTERVAL,
        "observed_mean_diff": observed_mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_excludes_zero": bool(ci_lower > 0 or ci_upper < 0),
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }

    existing = json.loads(CLOSURE_PATH.read_text()) if CLOSURE_PATH.exists() else {}
    existing["paired_incident_bootstrap"] = bootstrap_report
    CLOSURE_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(bootstrap_report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
