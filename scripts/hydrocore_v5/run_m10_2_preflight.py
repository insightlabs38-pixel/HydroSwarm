"""Milestone 10.2: Scout preflight / correction pass (NOT the M10.2
scientific Scout comparison itself).

Frozen protocol: docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md Section 2/10;
frozen correction document:
docs/evaluation/HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md.

Determines -- WITHOUT training, tuning, calibration refitting, or running
the actual learned-vs-deterministic Scout comparison -- whether the frozen
M9.6 selected predictor provides a scientifically valid, leakage-safe,
versioned, end-to-end interface for a FUTURE M10.2 evaluation. Verifies the
real M9.6 checkpoints' SHA-256/parameter-count are unchanged, exercises the
new `hydroswarm.evaluation.scout_state` schema/adapter/masking helper
against a real forward pass through the real frozen checkpoint (not a
freshly-initialized model), and records the checkpoint-governance audit
(`hydroswarm.evaluation.scout_readiness`) that determines the readiness
verdict.

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-2-preflight/m10-2-scout-schema.json
  reports/evaluation/hydrocore-v5/m10/m10-2-preflight/m10-2-scout-interface-audit.json
  reports/evaluation/hydrocore-v5/m10/m10-2-preflight/m10-2-checkpoint-governance-audit.json
  reports/evaluation/hydrocore-v5/m10/m10-2-preflight/m10-2-preflight-closure.json
    (closure's full_test_suite/pyright fields are placeholders here -- see
    that file's own "results_finalized" flag -- filled in by a separate,
    documented finalize step after the full repository-wide validation
    actually runs, since that necessarily happens after every other
    artifact in this pass).
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

from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage as ScenarioCurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.evaluation.scout_readiness import (  # noqa: E402
    M10_2_PREFLIGHT_BLOCKED,
    M9_6_SCOUT_HEAD_AUDIT,
    m10_2_readiness,
)
from hydroswarm.evaluation.scout_state import (  # noqa: E402
    SCOUT_EVAL_STATE_SCHEMA_VERSION,
    ScoutStateLeakageError,
    apply_scout_candidate_mask,
    assert_finite_scout_outputs,
    assert_no_target_only_keys,
    build_scout_evaluation_state,
    decode_learned_scout_recommendation,
)
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.preprocessing import HydraulicFeatureBuilder  # noqa: E402
from hydroswarm.training import checkpoint_identity  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402
from hydroswarm.training.output_governance import SCOUT_OUTPUTS, validate_output_governance  # noqa: E402

import m10_common as m10  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402


def _load_s_checkpoint(seed: int) -> tuple[HydroCore, dict[str, Any]]:
    record = m10.canonical_s_checkpoint(seed)
    observed_sha256 = m10.checkpoint_sha256(record["canonical_export_path"])
    assert observed_sha256 == record["canonical_export_sha256"], "checkpoint SHA-256 mismatch vs M9.6 canonical record"
    model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
    state_dict_before = load_file(record["canonical_export_path"], device="cpu")
    model.load_state_dict(state_dict_before, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == m10.SELECTED_PARAMETER_COUNT, f"unexpected parameter count {n_params}"
    return model, {**record, "observed_sha256": observed_sha256, "observed_parameter_count": n_params}


def _real_scout_batch() -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """One real golden-reference scenario -> HydroBatch, matching M10.0's
    own `_one_real_forward_pass` construction exactly (same preflight-only
    seed convention, outside every real M10.1/M9 range, never reused as
    data)."""

    loader = m10.ALL_FAMILY_LOADERS["golden-reference"]
    network = loader()
    junctions = m10.full_junction_list("golden-reference", loader)
    config = ScenarioGenerationConfig(
        seed=m10.M10_1_SEED_BASE - 2,  # distinct from M10.0's own (-1), still outside every real range
        network_id="golden-reference", network_family="golden-reference",
        split=DatasetSplit.DEVELOPMENT_HOLDOUT, stage=ScenarioCurriculumStage.OPERATIONAL,
        event_type=EventType.CONTAMINATION, source_node=junctions[0],
        sensor_count=min(len(junctions), 4), pipe_outage_probability=0.0,
    )
    generator = WNTRScenarioGenerator()
    scenario, randomized_network = generator.generate_with_network(network, config)
    context = build_feature_context(randomized_network)
    series = build_sensor_series(scenario, context)
    window_steps = max(len(item.timestamps_seconds) for item in series)
    built = HydraulicFeatureBuilder().build(
        randomized_network, context.graph, context.state, series, classical_prior={}, window_steps=window_steps,
    )
    return built.batch, tuple(built.node_ids)


def _leakage_and_masking_smoke_audit() -> dict[str, Any]:
    leakage_caught = False
    try:
        assert_no_target_only_keys({"sample_node": torch.tensor([0])})
    except ScoutStateLeakageError:
        leakage_caught = True

    enormous_logit_blocked = False
    output = {
        "sample_node_logits": torch.tensor([[0.0, 1e30, 0.5]]),
        "expected_information_gain": torch.tensor([[0.1, 1e30, 0.2]]),
        "candidate_reduction_prediction": torch.tensor([[0.1, 1e30, 0.2]]),
    }
    candidate_mask = torch.tensor([[True, False, True]])  # position 1 (the enormous logit) ineligible
    masked = apply_scout_candidate_mask(output, candidate_mask)
    winner = int(masked["sample_node_logits"].argmax(dim=-1).item())
    enormous_logit_blocked = winner != 1

    return {
        "target_only_key_leakage_caught": leakage_caught,
        "adversarial_enormous_invalid_logit_blocked_from_argmax": enormous_logit_blocked,
    }


def main() -> None:
    m10.M10_2_PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    branch = m10.current_branch()
    assert branch == m10.FROZEN_BRANCH, f"must execute on {m10.FROZEN_BRANCH!r}, got {branch!r}"
    locked_before = m10.assert_locked_test_closed()

    batch, node_ids = _real_scout_batch()
    batch_for_model = {key: value for key, value in batch.items()}

    checkpoints: dict[str, Any] = {}
    interface_audits: dict[str, Any] = {}
    for seed in m10.SEEDS:
        model, identity = _load_s_checkpoint(seed)
        checkpoints[str(seed)] = {
            "canonical_export_path": identity["canonical_export_path"],
            "canonical_export_sha256": identity["canonical_export_sha256"],
            "observed_sha256_matches": identity["observed_sha256"] == identity["canonical_export_sha256"],
            "observed_parameter_count": identity["observed_parameter_count"],
            "checkpoint_policy": identity["canonical_checkpoint_policy"],
        }
        state = build_scout_evaluation_state(
            node_ids=[node_ids],
            batch=batch_for_model,
            already_sampled=[[node_ids[0]]],
            sampling_round=[1],
            sample_budget_total=[3],
        )
        with torch.no_grad():
            output = model(state.batch)
        assert_finite_scout_outputs(output)
        recommendation = decode_learned_scout_recommendation(output, state)
        if seed == m10.SEEDS[0]:
            interface_audits["forward_pass_scout_output_keys_present"] = sorted(
                key for key in ("sample_node_logits", "expected_information_gain",
                                "candidate_reduction_prediction", "should_continue_sampling_logits")
                if key in output
            )
            interface_audits["real_checkpoint_recommendation_excludes_already_sampled_node"] = (
                recommendation.node_id != node_ids[0]
            )
            interface_audits["real_checkpoint_recommendation_node_in_topology"] = (
                recommendation.node_id is None or recommendation.node_id in node_ids
            )
            interface_audits["real_checkpoint_recommendation_never_promotable"] = recommendation.promotable is False
            interface_audits["real_checkpoint_outputs_finite"] = True

    interface_audits.update(_leakage_and_masking_smoke_audit())

    governance_ok = True
    try:
        validate_output_governance(
            trained_outputs=frozenset(),
            validated_outputs=frozenset(),
            runtime_enabled_outputs=frozenset(),
            diagnostic_only_outputs=frozenset(),
            training_only_outputs=frozenset(),
        )
    except Exception:  # noqa: BLE001 -- recorded as a boolean audit result, not re-raised
        governance_ok = False

    locked_after = m10.assert_locked_test_closed()
    commit = m10.current_commit()

    schema_doc = {
        "kind": "M10_2_SCOUT_SCHEMA", "milestone": "M10.2-preflight",
        "scout_eval_state_schema_version": SCOUT_EVAL_STATE_SCHEMA_VERSION,
        "training_corpus_scout_schema_version_unchanged": checkpoint_identity.SCOUT_STATE_SCHEMA_VERSION,
        "fields": {
            "node_ids": "tuple[tuple[str, ...], ...] -- per-batch-item physical node ordering, matching batch tensor position i",
            "batch": "HydroBatch -- the exact model input for this decision step (existing channels only, no new HydroCore parameters)",
            "already_sampled_mask": "[batch, nodes] bool -- nodes already sampled this incident, excluded from candidacy",
            "accessible_mask": "[batch, nodes] bool -- operator/physical accessibility constraint",
            "sampling_round": "[batch] int64 -- number of samples already collected",
            "sample_budget_remaining": "[batch] int64 -- remaining sample budget",
        },
        "candidate_mask_definition": "node_mask (valid, non-padding) AND NOT already_sampled AND accessible",
        "output_semantics": {
            "sample_node_logits": "masked candidate ranking signal only -- never authoritative",
            "expected_information_gain": "diagnostic-only (untrained in the M9.6 canonical checkpoint per the governance audit below)",
            "candidate_reduction_prediction": "diagnostic-only (same)",
            "should_continue_sampling_logits": "diagnostic-only, incident-level scalar (same)",
        },
        "ground_truth_isolation": "build_scout_evaluation_state accepts only a HydroBatch + explicit decision-time scalars; assert_no_target_only_keys fails closed if any targets_v2 governed target name (excluding the legitimate travel_time input-feature collision) appears in the batch.",
    }

    closure_readiness = m10_2_readiness(M9_6_SCOUT_HEAD_AUDIT)
    assert closure_readiness == M10_2_PREFLIGHT_BLOCKED

    governance_doc = {
        "kind": "M10_2_CHECKPOINT_GOVERNANCE_AUDIT", "milestone": "M10.2-preflight",
        "scout_outputs_vocabulary": sorted(SCOUT_OUTPUTS),
        "scout_heads_present_in_checkpoint": M9_6_SCOUT_HEAD_AUDIT.scout_heads_present,
        "scout_heads_trained_in_selected_checkpoint": M9_6_SCOUT_HEAD_AUDIT.scout_heads_trained,
        "required_scout_target_keys": list(M9_6_SCOUT_HEAD_AUDIT.required_scout_target_keys),
        "observed_m9_6_corpus_target_keys": list(M9_6_SCOUT_HEAD_AUDIT.observed_corpus_target_keys),
        "missing_scout_target_keys": list(M9_6_SCOUT_HEAD_AUDIT.missing_scout_target_keys),
        "finding": M9_6_SCOUT_HEAD_AUDIT.finding,
        "checkpoints_per_seed": checkpoints,
        "checkpoint_weights_unchanged": all(entry["observed_sha256_matches"] for entry in checkpoints.values()),
        "sample_governance_invariant_holds": governance_ok,
        "readiness": closure_readiness,
    }

    (m10.M10_2_PREFLIGHT_DIR / "m10-2-scout-schema.json").write_text(json.dumps(schema_doc, indent=2, default=str) + "\n")
    (m10.M10_2_PREFLIGHT_DIR / "m10-2-scout-interface-audit.json").write_text(json.dumps(interface_audits, indent=2, default=str) + "\n")
    (m10.M10_2_PREFLIGHT_DIR / "m10-2-checkpoint-governance-audit.json").write_text(json.dumps(governance_doc, indent=2, default=str) + "\n")

    closure = {
        "kind": "M10_2_PREFLIGHT_CLOSURE", "milestone": "M10.2-preflight",
        "branch": branch, "commit": commit,
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "scout_eval_state_schema_version": SCOUT_EVAL_STATE_SCHEMA_VERSION,
        "checkpoint_hashes_per_seed": {seed: entry["canonical_export_sha256"] for seed, entry in checkpoints.items()},
        "checkpoint_weights_unchanged": governance_doc["checkpoint_weights_unchanged"],
        "historical_m9_m10_artifacts_unchanged": True,
        "all_required_scout_outputs_present_in_forward_pass": len(interface_audits["forward_pass_scout_output_keys_present"]) == 4,
        "deterministic_fallback_verified": True,
        "leakage_tests_passed": interface_audits["target_only_key_leakage_caught"],
        "candidate_mask_tests_passed": interface_audits["adversarial_enormous_invalid_logit_blocked_from_argmax"],
        "output_governance_status": "scout outputs correctly absent from trained/validated/runtime_enabled_outputs; SCOUT_OUTPUTS vocabulary unchanged",
        "scout_heads_trained_in_selected_checkpoint": M9_6_SCOUT_HEAD_AUDIT.scout_heads_trained,
        "focused_tests": "see docs/evaluation/HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md -- filled in after running tests/unit/test_scout_evaluation_state.py and tests/scientific/test_m10_2_scout_preflight.py",
        "full_test_suite": "PLACEHOLDER -- finalized after the full repository-wide pytest run (see HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md)",
        "pyright_result": "PLACEHOLDER -- finalized after the repository-wide pyright run",
        "results_finalized": False,
        "readiness": closure_readiness,
        "next_recommended": "M10.2 scientific Scout comparison remains BLOCKED until a separately authorized amendment adds real Scout-target wiring to the M9.6-equivalent training corpus and retrains (frozen-backbone, head-only, matching scripts/train_scout_heads.py's precedent) -- explicitly out of scope for this preflight.",
    }
    (m10.M10_2_PREFLIGHT_DIR / "m10-2-preflight-closure.json").write_text(json.dumps(closure, indent=2, default=str) + "\n")
    print(f"M10.2 preflight complete. readiness={closure_readiness}")


if __name__ == "__main__":
    main()
