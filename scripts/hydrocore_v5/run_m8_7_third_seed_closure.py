"""Milestone 8.7 third-seed closure (seeds 20260814/31874 already screened;
this script covers the preregistered third seed, 20260815, for the
provisionally selected arm, AGE_FIX_ONLY -- see the M8.7 milestone closure
instructions, Sections 7 and 10).

`run_m8_7_evaluate.py` already re-evaluates every available AGE_FIX_ONLY
seed (including 20260815) for standard localization (Section 6) and
golden-reference origin invariance (Section 10), and folds seed 20260815
into `reports/evaluation/hydrocore-v5/m8-7-results.json` automatically
(unmodified script, unmodified logic). What it does NOT do is repeat, for
every seed, the more expensive checks that only ever run once per arm
against the SCREENING_REPRESENTATIVE_SEED (31874), to keep the two-seed
screening pass tractable (see that script's own "Compute-scoping"
docstring): the dev-grid-25 origin-invariance check, and B_DEPTH_AWARE
calibration.

The closure instructions explicitly require BOTH of those to be repeated
specifically for seed 20260815 (not merely inherited from seed 31874's
screening-time result). This script reuses `run_m8_7_evaluate.py`'s own
`run_origin_invariance_and_counterfactual`, `_load_checkpoint`, and
calibration helpers UNMODIFIED (imported, not reimplemented) to produce
those two additional data points, and writes them alongside the other
closure facts collected here.

Writes:
  reports/evaluation/hydrocore-v5/m8-7-closure.json
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

import torch  # noqa: E402

from hydroswarm.calibration.conformal import classify_runtime_condition  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.classical.state_estimation import HydraulicStateEstimator, OperationalTelemetry  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.corpus import build_sensor_series  # noqa: E402
from run_m6_temporal import _fit_frozen_calibrator  # noqa: E402
from run_m8_7_arm import ARM_DEFINITIONS  # noqa: E402
from run_m8_7_evaluate import (  # noqa: E402
    ALPHA,
    EARLY_DEPTHS,
    MATURE_DEPTHS,
    MID_DEPTHS,
    ORIGIN_TOLERANCE_ARMS_BC,
    RUNS_ROOT,
    _load_checkpoint,
    build_grid_network,
    run_origin_invariance_and_counterfactual,
)

THIRD_SEED = 20260815
SELECTED_ARM = "AGE_FIX_ONLY"
OUTPUT_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-7-closure.json"


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    run_record_path = RUNS_ROOT / f"{SELECTED_ARM}-seed{THIRD_SEED}.json"
    record = json.loads(run_record_path.read_text())
    assert record["arm"] == SELECTED_ARM and record["seed"] == THIRD_SEED

    feature_kwargs = ARM_DEFINITIONS[SELECTED_ARM]["feature_kwargs"]
    model = _load_checkpoint(record)

    dev_grid_network, _dev_grid_names = build_grid_network(25)

    def _dev_grid_context(network=dev_grid_network):
        simulator = HydraulicSimulator(network)
        raw = simulator.calculate_state(3600)
        estimated = HydraulicStateEstimator().estimate(raw, OperationalTelemetry())
        graph = simulator.build_dynamic_graph(estimated.as_hydraulic_state())
        return network, graph, estimated

    dev_grid_25_check = run_origin_invariance_and_counterfactual(
        model, feature_kwargs, tolerance=ORIGIN_TOLERANCE_ARMS_BC, network_factory=_dev_grid_context, label="dev-grid-25",
    )

    # Section 10: B_DEPTH_AWARE, alpha=0.1 calibration for the seed-20260815
    # predictor specifically (m8-7-calibration.json only carries the
    # SCREENING_REPRESENTATIVE_SEED, 31874, per run_m8_7_evaluate.py's
    # compute-scoping) -- same governed calibration split, same frozen
    # calibrator-fitting/candidate-set logic, reused unmodified.
    dev_records = build_scenario_pool("development_holdout", network_loader=build_wntr_network)
    calibration_records = build_scenario_pool("calibration", network_loader=build_wntr_network)
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    calibrator = _fit_frozen_calibrator(model, library, calibration_records, feature_kwargs=feature_kwargs)
    calibration_examples = []
    with torch.no_grad():
        for depth in CAUSAL_PREFIX_DEPTHS:
            for dev_record in dev_records:
                scenario = dev_record.scenario
                example = scenario_to_prefix_example(scenario, dev_record.network, library, depth, feature_context=dev_record.feature_context, **feature_kwargs)
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
                truth = int(example.targets["source_node"].item())
                full_series = build_sensor_series(scenario, dev_record.feature_context)
                truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
                condition = classify_runtime_condition(truncated_series)
                candidate_indices = calibrator.candidate_set(probs, condition=condition, network_id=f"m8-7-closure:{depth}")
                calibration_examples.append({"depth": depth, "covered": truth in candidate_indices, "set_size": len(candidate_indices)})
    marginal_coverage = statistics.fmean(row["covered"] for row in calibration_examples)
    mean_set_size = statistics.fmean(row["set_size"] for row in calibration_examples)
    singleton_rate = statistics.fmean(row["set_size"] == 1 for row in calibration_examples)
    by_maturity = {}
    for label, depths in (("early", EARLY_DEPTHS), ("mid", MID_DEPTHS), ("mature", MATURE_DEPTHS)):
        subset = [row for row in calibration_examples if row["depth"] in depths]
        by_maturity[label] = {
            "coverage": statistics.fmean(row["covered"] for row in subset) if subset else None,
            "mean_set_size": statistics.fmean(row["set_size"] for row in subset) if subset else None,
        }
    calibration_report = {
        "seed": THIRD_SEED, "alpha": ALPHA, "marginal_coverage": marginal_coverage,
        "mean_candidate_set_size": mean_set_size, "singleton_rate": singleton_rate, "by_maturity": by_maturity,
    }

    locked_after = locked_test_opened(ROOT)

    closure: dict[str, Any] = {
        "schema_version": 1,
        "purpose": (
            "Milestone 8.7 third-seed closure: seed 20260815 dev-grid-25 origin-invariance check and "
            "seed-specific B_DEPTH_AWARE calibration for the provisionally selected arm (AGE_FIX_ONLY). "
            "golden-reference origin invariance and standard localization for this seed are already "
            "recorded per-seed in m8-7-results.json (run_m8_7_evaluate.py, unmodified)."
        ),
        "arm": SELECTED_ARM,
        "seed": THIRD_SEED,
        "checkpoint_sha256": record["checkpoint_sha256"],
        "dev_grid_25_origin_invariance_and_counterfactual": dev_grid_25_check,
        "calibration_seed_20260815": calibration_report,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    OUTPUT_PATH.write_text(json.dumps(closure, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(closure, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
