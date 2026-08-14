"""Milestone 2.4 (experiments.txt): evaluate the multitask ablation arms
(A = Milestone-1 winner, reused; B = reweighted; C = role-isolated
adapters) on development_holdout at every causal depth, and record the
exit decision.

Primary metrics: 3-step top-1, 6-step top-1, mature-history (12/25)
retention, evidence-sufficiency quality. PCGrad is only flagged as
JUSTIFIED (never auto-enabled -- experiments.txt: "Do not enable it merely
because the implementation exists") when m2-conflict.json shows frequent
negative conflict between two PRIMARY (not merely primary-vs-auxiliary)
tasks with real supervision in this corpus.

Writes:
  reports/evaluation/hydrocore-v5/m2-results.json
  reports/evaluation/hydrocore-v5/m2-summary.md
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safetensors.torch import load_file  # noqa: E402

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    SUPERVISED_TASKS,
    build_scenario_pool,
    fit_pool_signature_library,
)
from evaluate_m1_depths import _evaluate_checkpoint_at_depth  # noqa: E402
from run_m1_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m2_arm import SUMMARY_ROOT as M2_RUNS_ROOT  # noqa: E402
from run_m2_conflict import M1_RUNS_ROOT, OUTPUT_PATH as M2_CONFLICT_PATH, _select_base_run  # noqa: E402
from hydroswarm.training.losses import PRIMARY_TASKS  # noqa: E402

OUTPUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m2-results.json"
OUTPUT_SUMMARY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m2-summary.md"

GAIN_THRESHOLD_PP = 3.0
REGRESSION_THRESHOLD_PP = 3.0


def _load_checkpoint_with_adapters(export_path: str, use_adapters: bool) -> HydroCore:
    model = HydroCore.from_variant("small", use_adapters=use_adapters, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()
    return model


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"

    base_run = _select_base_run()
    arm_export_paths: dict[str, tuple[str, bool]] = {
        "A": (base_run["training_summary"]["export_path"], False),
    }
    for arm, use_adapters in (("B", False), ("C", True)):
        run_path = M2_RUNS_ROOT / f"{arm}-seed{base_run['seed']}.json"
        if run_path.exists():
            record = json.loads(run_path.read_text())
            arm_export_paths[arm] = (record["training_summary"]["export_path"], use_adapters)

    dev_records = build_scenario_pool("development_holdout", network_loader=build_wntr_network)
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    curves: dict[str, dict[int, dict[str, Any]]] = {}
    for arm, (export_path, use_adapters) in arm_export_paths.items():
        model = _load_checkpoint_with_adapters(export_path, use_adapters)
        curves[arm] = {}
        for depth in CAUSAL_PREFIX_DEPTHS:
            metrics = _evaluate_checkpoint_at_depth(model, dev_records, library, depth)
            curves[arm][depth] = metrics
            print(f"M2 arm={arm} depth={depth} top1={metrics['top1']:.3f}")

    def _delta_pp(arm: str, depth: int) -> float | None:
        if "A" not in curves or arm not in curves:
            return None
        if depth not in curves["A"] or depth not in curves[arm]:
            return None
        return (curves[arm][depth]["top1"] - curves["A"][depth]["top1"]) * 100

    comparisons = {}
    for arm in ("B", "C"):
        if arm not in curves:
            continue
        gain_3 = _delta_pp(arm, 3)
        gain_6 = _delta_pp(arm, 6)
        mature = [d for d in (_delta_pp(arm, 12), _delta_pp(arm, 25)) if d is not None]
        comparisons[arm] = {
            "gain_3_step_pp": gain_3,
            "gain_6_step_pp": gain_6,
            "mature_delta_pp": mature,
            "meets_gain": bool(gain_3 is not None and gain_3 >= GAIN_THRESHOLD_PP),
            "no_mature_regression": not mature or min(mature) >= -REGRESSION_THRESHOLD_PP,
        }

    winners = [arm for arm, c in comparisons.items() if c["meets_gain"] and c["no_mature_regression"]]

    # PCGrad justification check (M2.3): frequent negative conflict between
    # two PRIMARY tasks that BOTH have real supervision in this corpus.
    pcgrad_justified = False
    pcgrad_evidence: list[str] = []
    if M2_CONFLICT_PATH.exists():
        conflict = json.loads(M2_CONFLICT_PATH.read_text())
        supervised_primary = sorted(PRIMARY_TASKS & SUPERVISED_TASKS)
        for bucket in conflict["per_depth_bucket"].values():
            for pair in bucket["frequent_negative_conflict_pairs"]:
                primary, other = pair.split("|", 1)
                if primary in supervised_primary and other in supervised_primary:
                    pcgrad_justified = True
                    pcgrad_evidence.append(pair)

    if winners:
        best = max(winners, key=lambda arm: comparisons[arm]["gain_3_step_pp"])
        decision = {"B": "REWEIGHT_TASKS", "C": "KEEP_ROLE_ADAPTERS_ISOLATION"}[best]
    elif pcgrad_justified:
        decision = "PCGRAD_JUSTIFIED"
    else:
        decision = "KEEP_FULL_MULTITASK"

    results = {
        "schema_version": 1,
        "purpose": "Milestone 2.4 (experiments.txt): multitask ablation-arm comparison and exit decision.",
        "base_arm_a": {"source": "Milestone 1 winning arm/seed run (reused, not retrained)", "arm_letter": base_run["_selected_as"], "seed": base_run["seed"]},
        "arms_evaluated": sorted(arm_export_paths),
        "curves": {arm: {str(d): m for d, m in curve.items()} for arm, curve in curves.items()},
        "comparisons_vs_arm_a": comparisons,
        "pcgrad_justified": pcgrad_justified,
        "pcgrad_evidence_pairs": sorted(set(pcgrad_evidence)),
        "exit_decision": decision,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Milestone 2 summary: multitask interference and objective design",
        "",
        f"Exit decision: **{decision}**",
        "",
        "## Arm comparison vs Arm A (full governed multitask)",
        "",
        "| arm | 3-step gain (pp) | 6-step gain (pp) | mature delta (pp) | meets gain | no mature regression |",
        "|---|---|---|---|---|---|",
    ]
    for arm, c in comparisons.items():
        lines.append(
            f"| {arm} | {c['gain_3_step_pp']:.2f} | {c['gain_6_step_pp']:.2f} | "
            f"{min(c['mature_delta_pp']) if c['mature_delta_pp'] else float('nan'):.2f} | "
            f"{c['meets_gain']} | {c['no_mature_regression']} |"
        )
    lines += [
        "",
        f"PCGrad justified: **{pcgrad_justified}** (evidence pairs: {sorted(set(pcgrad_evidence)) or 'none'})",
        "",
        "Scope note: only Sentinel-family tasks have real supervision in this corpus "
        "(see reports/evaluation/hydrocore-v5/m2-conflict.json's scope_limitation); Scout/Strategist/OOD "
        "interference is not measured this session and is not part of this exit decision.",
    ]
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "comparisons": comparisons, "pcgrad_justified": pcgrad_justified}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
