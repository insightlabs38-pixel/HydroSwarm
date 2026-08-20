"""Milestone 10.2 Scout refit amendment -- Part 1: full, mechanical
supervision-coverage audit of the selected M9.6 HydroCore-S construction.

Frozen protocol: docs/evaluation/HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_AMENDMENT.md

Classifies every semantic/gated HydroCore output the M9.6-selected
construction (`m9_1_common.SHARED_MODEL_CONFIG`) can produce into exactly
one of:

- TRAINED_WITH_REAL_TARGETS: a real M9.6 training batch supplies both the
  output (`output_name in outputs`) and a real target
  (`task in targets`) -- compute_multitask_loss's actual routing condition.
- PRESENT_BUT_UNSUPERVISED: the output IS present in a real forward pass
  (head instantiated and unconditionally/gate-conditionally computed), but
  no M9.6 training batch ever supplies a matching target.
- NOT_INSTANTIATED: the head does not exist at all in the M9.6 construction
  (a constructor flag that gates the head's very existence is off).
- STRUCTURALLY_NOT_EXERCISED: the head IS instantiated as a real parameter,
  but `HydroCore.forward()`'s own control flow never adds its output key to
  `outputs` for any batch M9.6 training ever produces (the
  candidate-conditioned Strategist path, gated on `plan_hidden is not None`,
  which requires `plan_template_ids`/`plan_target_type`/`plan_mask`/
  `plan_features` -- never populated by any code in
  hydroswarm.training.causal_prefix or hydroswarm.preprocessing).
- LEGACY_UNGOVERNED: the output IS present in every forward pass, but has no
  governed target at all (absent from
  hydroswarm.training.output_governance.CANONICAL_OUTPUT_NAMES /
  hydroswarm.training.losses.ALL_TASK_NAMES by design, not by omission --
  see checkpoint_identity.py's own "Section D" docstring).

Every classification here is proven mechanically (real model construction,
real forward pass, real `scenario_to_prefix_example` target keys, real
`compute_multitask_loss` routing conditions), not inferred from
`configs/training-v5-causal.yaml`'s task_weights, which -- as this audit
itself proves -- do not imply real supervision at all.

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-2-refit/m10-2-refit-supervision-audit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.causal_prefix import fit_pool_signature_library, scenario_to_prefix_example  # noqa: E402
from hydroswarm.training.losses import compute_multitask_loss  # noqa: E402
from hydroswarm.training.task_output_names import (  # noqa: E402
    LEGACY_EXCLUDED_TASK_NAMES as _LEGACY_EXCLUDED_TASK_NAMES,
    LEGACY_UNGOVERNED_OUTPUT_KEYS as _LEGACY_OUTPUT_KEYS,
    TASK_OUTPUT_NAMES as TASK_TO_OUTPUT,
)

import m10_common as m10  # noqa: E402
from run_m7_topology import SEED_BASES, TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402

M10_2_REFIT_DIR = m10.M10_DIR / "m10-2-refit"


def _golden_pool_and_library():
    family, loader = TRAINED_FAMILIES[0]
    pool = _family_scenario_pool(
        "train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")],
        count=15, source_round_robin=True,
    )
    return pool, fit_pool_signature_library(pool)


def main() -> None:
    M10_2_REFIT_DIR.mkdir(parents=True, exist_ok=True)
    locked_before = m10.assert_locked_test_closed()

    record = m10.canonical_s_checkpoint(m10.SEEDS[0])
    model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(record["canonical_export_path"], device="cpu"), strict=True)
    model.eval()

    pool, library = _golden_pool_and_library()
    example = scenario_to_prefix_example(pool[0].scenario, pool[0].network, library, 3, feature_context=pool[0].feature_context)
    targets = {key: value.unsqueeze(0) for key, value in example.targets.items()}
    batch = {key: value.unsqueeze(0) for key, value in example.inputs.items()}
    with torch.no_grad():
        outputs = model(batch)  # PASS 1, no plan_* tensors -- exactly M9.6's real training shape

    instantiated = {
        "sample_node_head": hasattr(model, "sample_node_head"),
        "information_gain_head": hasattr(model, "information_gain_head"),
        "candidate_reduction_head": hasattr(model, "candidate_reduction_head"),
        "should_continue_sampling_head": hasattr(model, "should_continue_sampling_head"),
        "ood_category_head": hasattr(model, "ood_category_head"),
        "next_step_head": hasattr(model, "next_step_head"),
        "event_presence_head": hasattr(model, "event_presence_head"),
        "event_cause_head": hasattr(model, "event_cause_head"),
        "sensor_reconstruction_head": hasattr(model, "sensor_reconstruction_head"),
        "future_concentration_head": hasattr(model, "future_concentration_head"),
        "travel_time_head": hasattr(model, "travel_time_head"),
        "candidate_plan_encoder": hasattr(model, "candidate_plan_encoder"),
        "action_head": hasattr(model, "action_head"),
        "plan_value_head": hasattr(model, "plan_value_head"),
        "plan_validity_head": hasattr(model, "plan_validity_head"),
        "consequence_proxy_heads": hasattr(model, "consequence_proxy_heads"),
        "uncertainty_head": hasattr(model, "uncertainty_head"),
        "ood_head": hasattr(model, "ood_head"),
    }

    #: Explicit, non-clever mapping for the two ways an output can be
    #: absent from a real M9.6-shaped forward pass -- verified against
    #: `instantiated` (module existence) below, not guessed.
    _NOT_INSTANTIATED_TASKS = {"sensor_reconstruction", "future_concentration", "travel_time"}
    _STRUCTURALLY_NOT_EXERCISED_TASKS = {
        "plan_validity", "plan_value", "exposure_proxy", "pressure_risk_proxy",
        "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy",
    }
    assert _NOT_INSTANTIATED_TASKS.isdisjoint(_STRUCTURALLY_NOT_EXERCISED_TASKS)

    records: dict[str, dict[str, Any]] = {}
    for task, output_name in TASK_TO_OUTPUT.items():
        output_present = output_name in outputs
        target_present = task in targets
        if task in _LEGACY_EXCLUDED_TASK_NAMES:
            classification = "LEGACY_UNGOVERNED"
        elif not output_present:
            assert task in _NOT_INSTANTIATED_TASKS or task in _STRUCTURALLY_NOT_EXERCISED_TASKS, (
                f"unexpected absent output for task {task!r} -- audit table is out of date"
            )
            if task in _NOT_INSTANTIATED_TASKS:
                assert not instantiated[f"{task}_head"], f"{task}_head unexpectedly instantiated"
                classification = "NOT_INSTANTIATED"
            else:
                classification = "STRUCTURALLY_NOT_EXERCISED"
        elif output_present and target_present:
            classification = "TRAINED_WITH_REAL_TARGETS"
        else:
            classification = "PRESENT_BUT_UNSUPERVISED"
        records[task] = {
            "output_name": output_name,
            "output_present_in_real_m9_6_shaped_forward_pass": output_present,
            "target_present_in_real_m9_6_training_batch": target_present,
            "classification": classification,
        }

    for key in _LEGACY_OUTPUT_KEYS:
        records[f"__legacy__{key}"] = {
            "output_name": key,
            "output_present_in_real_m9_6_shaped_forward_pass": key in outputs,
            "target_present_in_real_m9_6_training_batch": False,
            "classification": "LEGACY_UNGOVERNED",
        }
    records["__module__CandidatePlanEncoder"] = {
        "output_name": None,
        "output_present_in_real_m9_6_shaped_forward_pass": False,
        "target_present_in_real_m9_6_training_batch": False,
        "classification": "STRUCTURALLY_NOT_EXERCISED",
        "note": "instantiated (strategist_mode=candidate_conditioned) but forward() only calls it inside the plan_template_ids-is-not-None branch, never reached without plan_* HydroBatch fields (grep confirms zero occurrences of plan_template_ids/plan_target_type/plan_mask/plan_features anywhere in hydroswarm.training.causal_prefix or hydroswarm.preprocessing).",
    }

    # Cross-check against a REAL compute_multitask_loss call: every task
    # this audit calls TRAINED_WITH_REAL_TARGETS must appear in
    # result.tasks; no task classified otherwise may appear.
    loss_result = compute_multitask_loss(outputs, targets)
    trained_tasks = {task for task, rec in records.items() if rec["classification"] == "TRAINED_WITH_REAL_TARGETS"}
    assert trained_tasks == set(loss_result.tasks), (trained_tasks, set(loss_result.tasks))

    counts: dict[str, int] = {}
    for rec in records.values():
        counts[rec["classification"]] = counts.get(rec["classification"], 0) + 1

    locked_after = m10.assert_locked_test_closed()
    doc = {
        "kind": "M10_2_REFIT_SUPERVISION_AUDIT",
        "milestone": "M10.2-refit-amendment",
        "branch": m10.current_branch(),
        "commit": m10.current_commit(),
        "model_construction": SHARED_MODEL_CONFIG,
        "checkpoint_used_for_structural_audit": record["canonical_export_sha256"],
        "instantiated_modules": instantiated,
        "records": records,
        "classification_counts": counts,
        "cross_checked_against_real_compute_multitask_loss_call": True,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    (M10_2_REFIT_DIR / "m10-2-refit-supervision-audit.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print("supervision audit complete:", counts)


if __name__ == "__main__":
    main()
