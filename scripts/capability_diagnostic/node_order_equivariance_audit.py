"""Capability diagnostic Section 19: node-order/source-label alignment audit.

Runs `hydroswarm.training.permutation.measure_equivariance` against the REAL
frozen checkpoint (`experiments/runs/v4-checkpoint-identity/
no_adapters-seed20260810/model.safetensors`, sha256 a501ad87... -- the same
checkpoint verified byte-identical to `models/hydrocore-v4-release` in
Section 5's reproduction) on real `ScenarioExample`s built via
`hydroswarm.training.corpus.scenario_to_example`, for every governed
topology family (golden-reference, branched-loop, loop-grid) -- exhaustive
per diagnostic.txt Section 19's wording, not spot-checked on just one.

*** IMPORTANT: what this script found and then resolved this session ***
The RAW `measure_equivariance` call (as shipped) reports large, real
non-equivariance (max abs source-logit diff up to ~11.7, predicted-source
disagreement in up to 9/10 tests for a family) -- which would ordinarily be
an immediate Section 47 escalation ("permutation bug"). Direct root-cause
investigation this session (see `_fixed_permute_example` below and its
paired `_debug_confirm_root_cause` sanity check) found the TRUE cause is a
confirmed, reproducible defect in `src/hydroswarm/training/permutation.py`
itself, NOT the trained model: its `NODE_INDEXED_INPUT_KEYS` and
`TIME_NODE_INDEXED_INPUT_KEYS` tuples (module-level constants used by both
`permute_example` and `measure_equivariance`'s own consistency check) omit
two real, node-and-time-indexed batch keys -- `sensor_mask` and
`quality_mask` (both shape `[time, node]`, exactly the same category as
`temporal_features`/`quality_features`, which ARE handled correctly). This
means `permute_example` permutes node_features/temporal_features/etc. to
the new node order but leaves `sensor_mask`/`quality_mask` in the OLD node
order -- a genuine mask/data misalignment for any non-identity permutation.
Because `measure_equivariance`'s own `non_equivariant_keys` check only
inspects `NODE_INDEXED_INPUT_KEYS` (not the mismatched masks), it does not
catch its own inconsistency, so the large forward-pass difference gets
misattributed to the model rather than the mismatched masks feeding it.
This is the SAME recurring "hand-maintained key tuple silently omits a
newer field" drift class this module's own docstring already documents
being fixed once before for `NODE_INDEXED_TARGET_KEYS`/
`NODE_INDEX_TARGET_KEYS` (see permutation.py's Phase-10.2 comment) -- this
is a second, independent instance of the same defect class, this time on
the INPUT side and previously undetected because `permute_example` is not
actually invoked anywhere in the real training loop (confirmed by grep:
only this diagnostic and permutation.py's own test call it), so it has
never been exercised against real training data before.

Given the production freeze, `permutation.py` is NOT modified here. Instead
this script (a) runs the RAW measure_equivariance for the full record, (b)
locally builds a corrected permuted example (adding sensor_mask/
quality_mask to the permutation) INSIDE this diagnostic script only, and
re-measures, to answer the actual question Section 19 asks: is the MODEL
itself equivariant? (c) reports both, plus the CAP finding for the real
permutation.py defect.

Reuses `scripts/run_phase13_sentinel_metrics.py`'s own `load_model`
(imported via `importlib`, the same pattern
`scripts/capability_diagnostic/reproduce_controlled_eval.py` already
established). Freshly-built `ScenarioExample`s from `scenario_to_example`
never carry Strategist plan_* fields, so the model is loaded with
`strategist_fields_available=False`, matching `run_phase13_sentinel_metrics
.py`'s own documented precedent for evaluating populations without those
fields.

5 scenarios x 2 independent random permutations per family = 10 equivariance
tests per family, 30 total (run twice: raw and corrected) -- fixed seeds
throughout for reproducibility.

No locked-test access: only fresh WNTRScenarioGenerator-generated scenarios
(seed family 20260813, reusing the same seed base as Sections 6/11/13).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "phase13_sentinel_metrics", ROOT / "scripts" / "run_phase13_sentinel_metrics.py"
)
assert _spec is not None and _spec.loader is not None
_phase13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase13)

from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training import collate_variable_topology  # noqa: E402
from hydroswarm.training.corpus import resolve_model_input_signature_library, scenario_to_example  # noqa: E402
from hydroswarm.training.permutation import measure_equivariance, permute_example  # noqa: E402

from generate_cycle_b_corpus import TRAIN_TOPOLOGIES  # noqa: E402

FROZEN_SERVED_CHECKPOINT = ROOT / "experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810/model.safetensors"
EXPECTED_FROZEN_MODEL_SHA256 = "a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7"

FAMILY_LOADERS: dict[str, Any] = {
    "golden-reference": build_wntr_network,
    "branched-loop": dict(TRAIN_TOPOLOGIES)["branched-loop"],
    "loop-grid": dict(TRAIN_TOPOLOGIES)["loop-grid"],
}
SCENARIOS_PER_FAMILY = 5
PERMUTATIONS_PER_SCENARIO = 2
BASE_SEED = 20260813_00

#: Real, confirmed omission in src/hydroswarm/training/permutation.py's
#: TIME_NODE_INDEXED_INPUT_KEYS -- both are [time, node]-shaped exactly like
#: temporal_features/quality_features (which ARE in that tuple).
MISSING_TIME_NODE_INDEXED_KEYS = ("sensor_mask", "quality_mask")


def _corrected_permute_example(example: Any, permutation: list[int]) -> Any:
    """Same as permutation.permute_example, PLUS the two real, node-and-
    time-indexed keys it omits. Local to this diagnostic script only --
    permutation.py itself is not modified (production freeze)."""

    permuted = permute_example(example, permutation)
    perm_index = torch.tensor(permutation, dtype=torch.long)
    new_inputs = dict(permuted.inputs)
    for key in MISSING_TIME_NODE_INDEXED_KEYS:
        if key in example.inputs:
            new_inputs[key] = example.inputs[key][:, perm_index]
    return dc_replace(permuted, inputs=new_inputs)


def _measure_equivariance_corrected(model: Any, example: Any, permutation: list[int]) -> dict[str, Any]:
    node_count = example.inputs["node_features"].shape[0]
    corrected_permuted = _corrected_permute_example(example, permutation)
    model.eval()
    with torch.no_grad():
        original_inputs, _ = collate_variable_topology([example])
        original_output = model(original_inputs)
        permuted_inputs, _ = collate_variable_topology([corrected_permuted])
        permuted_output = model(permuted_inputs)
    perm_index = torch.tensor(permutation, dtype=torch.long)
    original_logits = original_output["source_node_logits"][0, :node_count]
    permuted_logits = permuted_output["source_node_logits"][0, :node_count]
    remapped_logits = torch.empty_like(original_logits)
    remapped_logits[perm_index] = permuted_logits
    difference = (original_logits - remapped_logits).abs()
    return {
        "max_absolute_source_logit_difference": float(difference.max()),
        "predicted_source_agrees": int(original_logits.argmax()) == int(remapped_logits.argmax()),
    }


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed for this diagnostic"

    measured_sha = hashlib.sha256(FROZEN_SERVED_CHECKPOINT.read_bytes()).hexdigest()
    assert measured_sha == EXPECTED_FROZEN_MODEL_SHA256, (
        f"frozen checkpoint sha mismatch: {measured_sha} != {EXPECTED_FROZEN_MODEL_SHA256}"
    )

    model = _phase13.load_model(FROZEN_SERVED_CHECKPOINT, use_adapters=False, strategist_fields_available=False)
    generator = WNTRScenarioGenerator()

    # One-time sanity confirmation of the root cause (kept in the report,
    # not just console output): identity permutation must be a true no-op
    # (model determinism), which it is, and the corrected permutation must
    # collapse the raw non-equivariance to floating-point noise.
    root_cause_check: dict[str, Any] = {}

    per_family: dict[str, list[dict[str, Any]]] = {}

    for family, loader in FAMILY_LOADERS.items():
        network = loader()
        junctions = tuple(sorted(network.junction_name_list))
        topology_hash = network_sha256(network)
        library, _ref_ts, mode = resolve_model_input_signature_library(topology_hash, junctions, network)

        records: list[dict[str, Any]] = []
        for scenario_index in range(SCENARIOS_PER_FAMILY):
            seed = BASE_SEED + scenario_index
            config = ScenarioGenerationConfig(
                seed=seed, network_id=family, network_family=family,
                split=DatasetSplit.VALIDATION, stage=CurriculumStage.OPERATIONAL,
                event_type=EventType.CONTAMINATION, pipe_outage_probability=0.0,
            )
            try:
                scenario = generator.generate(network, config)
                example = scenario_to_example(scenario, network, library)
                node_count = example.inputs["node_features"].shape[0]

                for perm_index in range(PERMUTATIONS_PER_SCENARIO):
                    rng = np.random.default_rng(seed * 100 + perm_index)
                    permutation = rng.permutation(node_count).tolist()

                    raw_result = measure_equivariance(model, example, permutation, collate_fn=collate_variable_topology, atol=1e-4)
                    corrected_result = _measure_equivariance_corrected(model, example, permutation)

                    if family == "golden-reference" and scenario_index == 0 and perm_index == 0:
                        identity_result = measure_equivariance(
                            model, example, list(range(node_count)), collate_fn=collate_variable_topology, atol=1e-4
                        )
                        root_cause_check = {
                            "identity_permutation_max_diff": identity_result.max_absolute_source_logit_difference,
                            "identity_permutation_confirms_model_determinism": identity_result.max_absolute_source_logit_difference == 0.0,
                            "example_permutation": permutation,
                            "raw_measure_equivariance_max_diff": raw_result.max_absolute_source_logit_difference,
                            "corrected_measure_equivariance_max_diff": corrected_result["max_absolute_source_logit_difference"],
                            "correction_collapses_to_floating_point_noise": corrected_result["max_absolute_source_logit_difference"] < 1e-4,
                        }

                    records.append({
                        "seed": seed,
                        "permutation_index": perm_index,
                        "node_count": node_count,
                        "permutation": permutation,
                        "signature_mode": mode,
                        "raw": {
                            "max_absolute_source_logit_difference": raw_result.max_absolute_source_logit_difference,
                            "predicted_source_agrees": raw_result.predicted_source_agrees,
                            "non_equivariant_keys": list(raw_result.non_equivariant_keys),
                        },
                        "corrected_for_permutation_py_mask_omission": corrected_result,
                    })
            except Exception as exc:  # noqa: BLE001
                records.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})

        per_family[family] = records

    def _family_summary(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
        ok = [r for r in records if "error" not in r]
        errored = [r for r in records if "error" in r]
        if arm == "raw":
            agrees = [r for r in ok if r["raw"]["predicted_source_agrees"]]
            max_diffs = [r["raw"]["max_absolute_source_logit_difference"] for r in ok]
        else:
            agrees = [r for r in ok if r["corrected_for_permutation_py_mask_omission"]["predicted_source_agrees"]]
            max_diffs = [r["corrected_for_permutation_py_mask_omission"]["max_absolute_source_logit_difference"] for r in ok]
        return {
            "n_tests": len(records),
            "n_ok": len(ok),
            "n_errors": len(errored),
            "n_predicted_source_agrees": len(agrees),
            "all_predicted_source_agree": len(agrees) == len(ok) and len(ok) > 0,
            "max_abs_source_logit_difference_over_all_tests": max(max_diffs) if max_diffs else None,
            "mean_abs_source_logit_difference": (sum(max_diffs) / len(max_diffs)) if max_diffs else None,
            "errors": errored,
        }

    family_summaries = {
        family: {"raw": _family_summary(records, "raw"), "corrected": _family_summary(records, "corrected")}
        for family, records in per_family.items()
    }
    true_model_equivariance_holds = all(
        s["corrected"]["all_predicted_source_agree"] and s["corrected"]["max_abs_source_logit_difference_over_all_tests"] < 1e-3
        for s in family_summaries.values()
    )

    cap_findings = {
        "CAP-DATA-02": {
            "title": "src/hydroswarm/training/permutation.py's NODE_INDEXED_INPUT_KEYS / TIME_NODE_INDEXED_INPUT_KEYS "
            "tuples omit sensor_mask and quality_mask, causing permute_example() to produce internally-inconsistent "
            "(mask misaligned with data) permuted examples -- and, before this session's root-causing, made this "
            "diagnostic's initial raw measure_equivariance() run falsely appear to find a real model permutation bug.",
            "root_cause": (
                "sensor_mask and quality_mask are both real [time, node]-shaped batch keys (same shape category as "
                "temporal_features/quality_features) produced by HydraulicFeatureBuilder.build and consumed by "
                "HydroCore per-node-per-timestep, but are absent from both NODE_INDEXED_INPUT_KEYS and "
                "TIME_NODE_INDEXED_INPUT_KEYS in src/hydroswarm/training/permutation.py. permute_example() only "
                "permutes keys in those two tuples, so it silently leaves sensor_mask/quality_mask in the ORIGINAL "
                "node order while every other node-indexed input moves to the new order -- a genuine data/mask "
                "misalignment for any non-identity permutation. This is the same 'hand-maintained key tuple falls "
                "out of sync with the real schema' drift class this module's own docstring documents already being "
                "fixed once for NODE_INDEXED_TARGET_KEYS/NODE_INDEX_TARGET_KEYS (Phase 10.2) -- a second, "
                "independent instance, this time on the input side. Confirmed by grep: permute_example is not "
                "invoked anywhere in the real training loop (only by this diagnostic and permutation.py's own "
                "unit-style call site), so this defect has never been exercised against real data before."
            ),
            "measured_effect": (
                "Root-cause check (golden-reference, seed 20260813_00, permutation [4,3,5,1,2,0]): RAW "
                f"measure_equivariance max abs source-logit diff = {root_cause_check.get('raw_measure_equivariance_max_diff')}; "
                "after correcting ONLY the mask permutation (locally, inside this diagnostic script -- "
                "permutation.py itself is unmodified per the production freeze), max abs diff collapses to "
                f"{root_cause_check.get('corrected_measure_equivariance_max_diff')} (floating-point noise level). "
                "Across all 30 tests (3 families x 5 scenarios x 2 permutations), the RAW arm shows "
                f"{sum(s['raw']['n_predicted_source_agrees'] for s in family_summaries.values())}/30 "
                "predicted-source agreements; the CORRECTED arm shows "
                f"{sum(s['corrected']['n_predicted_source_agrees'] for s in family_summaries.values())}/30."
            ),
            "downstream_impact": (
                "NONE on current production serving (permute_example is never invoked by the real training loop or "
                "serving path -- production always uses a single fixed canonical node order per network, so this "
                "defect is currently latent/dormant). It WOULD corrupt any future use of this module for genuine "
                "permutation-augmentation training or for a trustworthy automated equivariance regression gate -- "
                "right now this diagnostic's own initial raw run is a live demonstration of exactly that failure "
                "mode: a real code defect producing a false-positive 'model is broken' signal."
            ),
            "taxonomy": "CAP-DATA",
            "severity": "LOW for current production behavior (dormant/unused code path); MEDIUM for diagnostic "
            "trustworthiness and any future permutation-invariance work -- left unfixed on this branch per the "
            "production freeze, reported as a minimized reproducer instead.",
        }
    }

    report = {
        "schema_version": 1,
        "section": "19_node_order_source_label_alignment_audit",
        "locked_test_opened_before": locked_before,
        "checkpoint": str(FROZEN_SERVED_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": measured_sha,
        "families_tested": list(FAMILY_LOADERS),
        "scenarios_per_family": SCENARIOS_PER_FAMILY,
        "permutations_per_scenario": PERMUTATIONS_PER_SCENARIO,
        "total_tests": SCENARIOS_PER_FAMILY * PERMUTATIONS_PER_SCENARIO * len(FAMILY_LOADERS),
        "root_cause_check": root_cause_check,
        "per_family_records": per_family,
        "per_family_summary": family_summaries,
        "cap_findings": cap_findings,
        "overall_verdict": {
            "raw_measure_equivariance_exhaustive_pass_all_families": all(
                s["raw"]["all_predicted_source_agree"] for s in family_summaries.values()
            ),
            "true_model_equivariance_holds_after_correcting_test_tooling_defect": true_model_equivariance_holds,
            "interpretation": (
                "The RAW measure_equivariance() call (as shipped) reports apparent large non-equivariance across "
                "all 3 families -- but this session's root-cause investigation confirmed it is a false signal "
                "caused by a real, reproducible defect in permutation.py's own key lists (CAP-DATA-02), NOT a "
                "trained-model permutation bug: correcting only the mask permutation (locally, in this script) "
                "collapses the difference to floating-point noise and restores 100% predicted-source agreement. "
                "TRUE VERDICT: the frozen HydroCore-v4 checkpoint IS node-order equivariant across every governed "
                "topology family tested (no Section-47 'permutation bug' escalation warranted for the MODEL "
                "itself) -- but permutation.py's own tooling defect (CAP-DATA-02) is real, reproducible, and "
                "worth fixing before this module is ever used for real permutation-augmentation training."
                if true_model_equivariance_holds else
                "Even after correcting the known permutation.py mask-omission defect, real non-equivariance "
                "remains -- this DOES warrant a Section 47 escalation; see per_family_records for specifics."
            ),
        },
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "node-order-audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(family_summaries, indent=2, default=str))
    print(json.dumps(report["overall_verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
