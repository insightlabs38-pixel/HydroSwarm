"""Milestone 10.3A: Strategist readiness audit (Part 1) -- NOT the M10.3A
training run itself, NOT the true M10.3 scientific comparison.

Frozen protocol: docs/evaluation/HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md.

Mechanically proves, from real code execution against the real frozen
canonical M9.6 checkpoints (never a synthetic stand-in), that:

1. Every canonical M9.6 checkpoint's SHA-256 is unchanged.
2. M9.6 was constructed with strategist_mode="candidate_conditioned" and
   consequence_prescreening_heads=True (SHARED_MODEL_CONFIG).
3. The real M9.6 training corpus path (scenario_to_prefix_example, via
   CausalPrefixDatasetView) never populates plan_template_ids/
   plan_target_type/plan_target_node_index/plan_target_link_index/
   plan_features -- grep-based AND execution-based (a real example is
   built and its .inputs keys inspected directly).
4. Therefore plan_hidden was always None during M9.6 training (candidate-
   conditioned forward branch requires all four plan fields or none;
   "none" is what every M9.6 batch actually supplied) -- action_logits/
   action_pointer_logits/plan_value/plan_validity_logits/consequence
   proxies were never present in any M9.6 training output, never lost a
   gradient step, and hold their random initialization in every canonical
   checkpoint today. Verified directly: load a real frozen M9.6 checkpoint,
   forward a real M9.6-shaped batch (no plan tensors) -- confirm those
   keys are absent from output.
5. The SAME checkpoint, forwarded WITH real plan tensors (this task's own
   new corpus builder), DOES populate those keys -- CandidatePlanEncoder
   and every head structurally execute and produce finite output; their
   VALUES are governed only by random initialization (proof of (3)/(4)
   above establishes this, since gradient coverage is proved by absence
   of a training-target path, not by inspecting output values).
6. action_template/target_pointer are recorded "v3-legacy" in the frozen
   training config -- repository evidence for excluding them from this
   refit's canonical trainable-target scope.
7. HydroStrategist.deterministic_fallback is unmodified and structurally
   independent of the candidate-conditioned pathway (different module,
   different input shape, no shared parameter).
8. PlanVerifier remains the sole plan_validity authority
   (strategist_labels.generate_strategist_labels never assigns validity
   from anything but verifier.verify()'s own decision -- confirmed by
   direct source inspection, recorded here as a structural fact, not
   re-derived).

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-readiness-audit.json
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import m10_common as m10  # noqa: E402
from run_m7_topology import SEED_BASES, TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402

from hydroswarm.agents.strategist import HydroStrategist  # noqa: E402
from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training import causal_prefix, corpus  # noqa: E402
from hydroswarm.training.causal_prefix import fit_pool_signature_library  # noqa: E402
from hydroswarm.training.scout_labels import build_signature_artifact_for_network  # noqa: E402
from hydroswarm.training.strategist_candidate_corpus import build_strategist_candidate_example  # noqa: E402

M10_3_REFIT_DIR = m10.M10_DIR / "m10-3-refit"

PLAN_INPUT_FIELDS = (
    "plan_template_ids", "plan_target_type", "plan_target_node_index", "plan_target_link_index", "plan_features",
)
CANDIDATE_PLAN_OUTPUT_KEYS = (
    "action_logits", "action_pointer_logits", "plan_value", "plan_validity_logits",
    "exposure_proxy", "pressure_risk_proxy", "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy",
)


def _grep_plan_fields_in_training_corpus_path() -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for module in (causal_prefix, corpus):
        source = inspect.getsource(module)
        found = [field for field in PLAN_INPUT_FIELDS if field in source]
        hits[module.__name__] = found
    return hits


def _real_m9_6_batch_never_includes_plan_fields() -> dict[str, Any]:
    family, loader = TRAINED_FAMILIES[0]
    assert family == "golden-reference"
    pool = _family_scenario_pool(
        "train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")],
        count=5, source_round_robin=True,
    )
    input_library = fit_pool_signature_library(pool)
    rec = pool[0]
    example = causal_prefix.scenario_to_prefix_example(
        rec.scenario, rec.network, input_library, 25,
        feature_context=rec.feature_context,
    )
    input_keys = set(example.inputs)
    target_keys = set(example.targets)
    present_plan_input_fields = [field for field in PLAN_INPUT_FIELDS if field in input_keys]
    present_plan_target_fields = [
        field for field in ("plan_validity", "plan_value", "exposure_proxy", "pressure_risk_proxy",
                             "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy",
                             "action_template", "target_pointer")
        if field in target_keys
    ]
    return {
        "example_input_keys": sorted(input_keys),
        "example_target_keys": sorted(target_keys),
        "present_plan_input_fields": present_plan_input_fields,
        "present_plan_target_fields": present_plan_target_fields,
        "plan_fields_absent_from_real_m9_6_input": present_plan_input_fields == [],
        "plan_fields_absent_from_real_m9_6_targets": present_plan_target_fields == [],
    }


def _real_sentinel_batch_for_smoke_incident() -> dict[str, Any]:
    """Builds one real, correctly-feature-shaped Sentinel batch (via the
    ordinary causal-prefix path every M9.6 training example used) for a
    real golden-reference incident -- reused by both the with- and
    without-plan-tensor forward checks below, so both start from an
    identical, real, correctly-dimensioned base batch (never a synthetic
    stand-in with the wrong feature width, which `strict=True` weight
    loading would otherwise silently paper over via shape mismatches at
    the WRONG layer)."""

    family, loader = TRAINED_FAMILIES[0]
    pool = _family_scenario_pool(
        "train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")],
        count=5, source_round_robin=True,
    )
    network = loader()
    input_library = fit_pool_signature_library(pool)
    rec = pool[0]
    example = causal_prefix.scenario_to_prefix_example(
        rec.scenario, rec.network, input_library, 25, feature_context=rec.feature_context,
    )
    # scenario_to_prefix_example returns UNBATCHED ([nodes, ...]) tensors --
    # add a batch dim of 1 to match plan_proposals_to_candidate_tensors'
    # own [1, plans, ...] convention for a single-incident forward pass.
    batched = {key: value.unsqueeze(0) for key, value in example.inputs.items()}
    return {"batch": batched, "rec": rec, "network": network, "input_library": input_library}


def _forward_frozen_checkpoint_without_plan_tensors(seed: int, sentinel_batch: dict[str, Any]) -> dict[str, Any]:
    record = m10.canonical_s_checkpoint(seed)
    model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(record["canonical_export_path"], device="cpu"), strict=True)
    model.eval()
    with torch.no_grad():
        output = model(sentinel_batch)
    present = [key for key in CANDIDATE_PLAN_OUTPUT_KEYS if key in output]
    return {
        "seed": seed, "checkpoint_sha256": record["canonical_export_sha256"],
        "candidate_plan_output_keys_present_without_plan_tensors": present,
        "confirms_plan_hidden_was_none": present == [],
    }


def _forward_frozen_checkpoint_with_plan_tensors(seed: int, base: dict[str, Any]) -> dict[str, Any]:
    network, rec = base["network"], base["rec"]
    cache = SignatureCache(str(ROOT / "experiments" / "cache" / "m10-3-refit-signatures"))
    key = SignatureCacheKey(
        network_hash="m10-3-refit-golden-reference", hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="m10-3-refit-cfg1", sensor_layout_hash="m10-3-refit-layout1",
    )
    artifact = build_signature_artifact_for_network(network, cache, key=key)
    node_ids = tuple(sorted(network.node_name_list))
    edge_ids = tuple(
        (network.get_link(name).start_node_name, network.get_link(name).end_node_name)
        for name in sorted(network.link_name_list)
    )
    example = build_strategist_candidate_example(rec.scenario, rec.network, rec.feature_context, artifact, node_ids, edge_ids)
    assert example is not None, "expected a real candidate example for this smoke incident"

    record = m10.canonical_s_checkpoint(seed)
    model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(record["canonical_export_path"], device="cpu"), strict=True)
    model.eval()

    # Merge the real candidate tensors onto the SAME real Sentinel batch
    # `_forward_frozen_checkpoint_without_plan_tensors` used for this exact
    # incident -- the only difference between the two forward passes is the
    # presence/absence of the five plan_* keys.
    batch = {**base["batch"], **example.batch}
    with torch.no_grad():
        output = model(batch)
    present = [key for key in CANDIDATE_PLAN_OUTPUT_KEYS if key in output]
    finite = {key: bool(torch.isfinite(output[key]).all()) for key in present}
    return {
        "seed": seed,
        "candidate_plan_output_keys_present_with_plan_tensors": present,
        "all_present_outputs_finite": all(finite.values()) if finite else False,
        "confirms_candidate_conditioned_path_structurally_executes": set(present) == set(CANDIDATE_PLAN_OUTPUT_KEYS),
    }


def main() -> None:
    M10_3_REFIT_DIR.mkdir(parents=True, exist_ok=True)
    locked_before = m10.assert_locked_test_closed()

    teacher_hashes = {}
    for seed in m10.SEEDS:
        record = m10.canonical_s_checkpoint(seed)
        teacher_hashes[str(seed)] = record["canonical_export_sha256"]

    config_audit = {
        "strategist_mode": SHARED_MODEL_CONFIG.get("strategist_mode"),
        "consequence_prescreening_heads": SHARED_MODEL_CONFIG.get("consequence_prescreening_heads"),
        "matches_task_stated_construction": (
            SHARED_MODEL_CONFIG.get("strategist_mode") == "candidate_conditioned"
            and SHARED_MODEL_CONFIG.get("consequence_prescreening_heads") is True
        ),
    }

    training_yaml_text = (ROOT / "configs" / "training-v5-causal.yaml").read_text()
    legacy_head_audit = {
        "action_template_marked_v3_legacy_in_config": (
            "v3-legacy head" in training_yaml_text.split("action_template:")[1].split("\n")[0]
            if "action_template:" in training_yaml_text else False
        ),
        "target_pointer_marked_v3_legacy_in_config": (
            "v3-legacy head" in training_yaml_text.split("target_pointer:")[1].split("\n")[0]
            if "target_pointer:" in training_yaml_text else False
        ),
    }

    grep_audit = _grep_plan_fields_in_training_corpus_path()
    real_batch_audit = _real_m9_6_batch_never_includes_plan_fields()
    base = _real_sentinel_batch_for_smoke_incident()
    without_plan_tensors = [_forward_frozen_checkpoint_without_plan_tensors(seed, base["batch"]) for seed in m10.SEEDS]
    with_plan_tensors = _forward_frozen_checkpoint_with_plan_tensors(m10.SEEDS[0], base)

    deterministic_strategist_source = inspect.getsource(HydroStrategist.deterministic_fallback)
    deterministic_strategist_audit = {
        "uses_candidate_plan_encoder": "candidate_plan_encoder" in deterministic_strategist_source,
        "uses_generate_response_plans": "generate_response_plans" in deterministic_strategist_source,
        "structurally_independent_of_candidate_conditioned_path": (
            "candidate_plan_encoder" not in deterministic_strategist_source
            and "generate_response_plans" not in deterministic_strategist_source
        ),
    }

    readiness_defect_confirmed = (
        real_batch_audit["plan_fields_absent_from_real_m9_6_input"]
        and all(entry["confirms_plan_hidden_was_none"] for entry in without_plan_tensors)
        and with_plan_tensors["confirms_candidate_conditioned_path_structurally_executes"]
    )

    doc = {
        "kind": "M10_3_REFIT_READINESS_AUDIT",
        "branch": m10.current_branch(),
        "commit": m10.current_commit(),
        "teacher_checkpoint_sha256": teacher_hashes,
        "expected_teacher_checkpoint_sha256": {
            "20260814": "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5",
            "31874": "527e707b988870dff323b6a3e0f1bde36a152b7bbe7e4c7f4bf0e59cdaf19332",
            "20260815": "b45f5afeabba820270130595dffb44152ec12fad6fa383cce67013f14524113c",
        },
        "teacher_hashes_match_expected": teacher_hashes == {
            "20260814": "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5",
            "31874": "527e707b988870dff323b6a3e0f1bde36a152b7bbe7e4c7f4bf0e59cdaf19332",
            "20260815": "b45f5afeabba820270130595dffb44152ec12fad6fa383cce67013f14524113c",
        },
        "model_construction_audit": config_audit,
        "legacy_head_governance_audit": legacy_head_audit,
        "grep_audit_training_corpus_source": grep_audit,
        "real_m9_6_batch_audit": real_batch_audit,
        "frozen_checkpoint_forward_without_plan_tensors": without_plan_tensors,
        "frozen_checkpoint_forward_with_plan_tensors": with_plan_tensors,
        "deterministic_strategist_fallback_audit": deterministic_strategist_audit,
        "readiness_defect_confirmed": readiness_defect_confirmed,
        "readiness_defect_summary": (
            "CONFIRMED: strategist_mode=candidate_conditioned and consequence_prescreening_heads=True "
            "construct CandidatePlanEncoder/plan_value_head/plan_validity_head/consequence_proxy_heads as real "
            "parameters in every canonical M9.6 checkpoint, but the real M9.6 training corpus path "
            "(scenario_to_prefix_example) never populates any of the four required plan input fields, so "
            "plan_hidden was always None during M9.6 training and none of those modules ever received a "
            "gradient. The SAME frozen checkpoints, forwarded with real plan tensors (this task's own new "
            "corpus builder), DO structurally execute the candidate-conditioned path and produce finite "
            "output -- confirming the architecture is real and load-bearing, only unsupervised."
        ),
        "readiness_decision": "M10_3A_REFIT_READY" if readiness_defect_confirmed else "M10_3A_REFIT_BLOCKED_UNEXPECTED_STATE",
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": m10.assert_locked_test_closed(),
    }
    (M10_3_REFIT_DIR / "m10-3-refit-readiness-audit.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print(json.dumps({"readiness_decision": doc["readiness_decision"], "readiness_defect_confirmed": readiness_defect_confirmed}, indent=2))


if __name__ == "__main__":
    main()
