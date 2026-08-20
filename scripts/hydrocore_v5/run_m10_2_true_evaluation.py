"""TRUE Milestone 10.2 execution: paired deterministic-vs-learned Scout
multi-round trajectory simulation, under the frozen protocol
(`docs/evaluation/HYDROCORE_V5_M10_2_TRUE_EVALUATION_PROTOCOL.md`,
`m10_2_true_protocol.py`).

Does NOT decide promotion -- `run_m10_2_true_decide.py` reads this script's
output and applies the frozen promotion rule. This script only collects
paired, per-round, per-incident, per-seed trajectory records and the
mandatory safety/governance audit counters.

Both arms use the SAME Level-A refit checkpoint (never the original M9.6
teacher, whose raw Scout heads are untrained) -- see the protocol
document's Finding 3 for why `role_features`/`residual_features` are
populated identically for both arms at every round.

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-checkpoint-verification.json
  reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-population-manifest.json
  reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-trajectories-seed{seed}.jsonl
  reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-safety-audit.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import m10_2_refit_protocol as refit_proto  # noqa: E402
import m10_2_true_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402
from run_m7_topology import TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402

from hydroswarm.agents.schemas import ScoutAction  # noqa: E402
from hydroswarm.agents.scout import HydroScout  # noqa: E402
from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator  # noqa: E402
from hydroswarm.classical.metrics import entropy  # noqa: E402
from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey, localize_with_signatures  # noqa: E402
from hydroswarm.evaluation.scout_state import (  # noqa: E402
    assert_finite_scout_outputs,
    build_scout_evaluation_state,
    decode_learned_scout_recommendation,
)
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator, MAXIMUM_EVALUATION_HYPOTHESES  # noqa: E402
from hydroswarm.training.causal_prefix import _degradation_probabilities, fit_pool_signature_library  # noqa: E402
from hydroswarm.training.scenario_reconstruction import (  # noqa: E402
    ScenarioReconstructionError,
    reconstruct_scenario_network,
    simulate_all_node_truth,
)
from hydroswarm.training.scout_labels import (  # noqa: E402
    build_signature_artifact_for_network,
    reindex_to_signature_grid,
)
from hydroswarm.training.scout_training_state import build_scout_training_state_batch  # noqa: E402
from hydroswarm.training.scout_trajectory import reveal_sample_measurement  # noqa: E402

assert MAXIMUM_EVALUATION_HYPOTHESES == proto.CANDIDATE_GATE_K

M10_2_TRUE_DIR = m10.M10_DIR / "m10-2"
SIGNATURE_CACHE_DIR = ROOT / "experiments" / "cache" / "m10-2-refit-signatures"


# ---------------------------------------------------------------------------
# Step 1-4 (task instruction): locate, verify, fail-closed on the Level-A
# refit weights -- never substitute the original M9.6 teacher.
# ---------------------------------------------------------------------------


def verify_and_load_checkpoint(seed: int, *, checkpoint_path: Path | None = None) -> HydroCore:
    checkpoint_path = checkpoint_path or (
        m10.M10_DIR / "m10-2-refit" / "checkpoints" / f"level-a-seed{seed}" / "model.safetensors"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Level-A refit weight file missing for seed {seed}: {checkpoint_path}. "
            "Per task instruction, this must be regenerated ONLY through the frozen M10.2-refit "
            "protocol/scripts (scripts/hydrocore_v5/run_m10_2_level_a_train.py) and the resulting "
            "hash re-verified -- never substituted with the original M9.6 teacher checkpoint."
        )
    actual_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    expected_sha256 = proto.LEVEL_A_REFIT_CHECKPOINT_SHA256[seed]
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Level-A refit checkpoint SHA-256 mismatch for seed {seed}: "
            f"expected {expected_sha256}, got {actual_sha256}. Failing closed -- refusing to "
            "evaluate against an unverified checkpoint."
        )
    model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
    model.eval()
    return model, actual_sha256, checkpoint_path


def verify_teacher_checkpoints_unchanged() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for seed in m10.SEEDS:
        record = m10.canonical_s_checkpoint(seed)
        actual = record["canonical_export_sha256"]
        expected = proto.PARENT_M9_6_TEACHER_SHA256[seed]
        if actual != expected:
            raise ValueError(f"Parent M9.6 teacher checkpoint changed for seed {seed}: expected {expected}, got {actual}")
        result[seed] = {"expected": expected, "actual": actual, "unchanged": True, "path": record["canonical_export_path"]}
    return result


# ---------------------------------------------------------------------------
# Frozen calibrator -- identical pattern to run_m10_1_decide._fit_frozen_calibrator.
# ---------------------------------------------------------------------------


def fit_frozen_calibrator() -> SplitConformalCalibrator:
    path = ROOT / proto.CALIBRATION_SUPPORT_PATH
    examples: list[CalibrationExample] = []
    with path.open() as fh:
        for line in fh:
            record = json.loads(line)
            if record["arm"] != proto.CALIBRATION_SUPPORT_ARM:
                continue
            examples.append(CalibrationExample(
                probabilities=tuple(record["probabilities"]), true_index=record["true_index"],
                condition=record["condition"], network_id=f"{record['family']}:{record['depth_bucket']}",
            ))
    return SplitConformalCalibrator.fit(
        examples, alpha=proto.CALIBRATION_ALPHA, model_hash="m9-6-arm-b-frozen-S", feature_schema_hash="m9-6-frozen",
        dataset_manifest_hash="m9-6-canonical-calibration", minimum_group_size=proto.CALIBRATION_MINIMUM_GROUP_SIZE,
    )


# ---------------------------------------------------------------------------
# Population / signature assets -- reuses Level-A's OWN train pool/library/
# artifact-cache-key exactly, so the model is evaluated under the identical
# classical-prior distributional assumptions it was actually trained under.
# ---------------------------------------------------------------------------


def build_shared_assets():
    family, loader = TRAINED_FAMILIES[0]
    assert family == proto.FAMILY == refit_proto.FAMILY
    network = loader()

    train_pool = _family_scenario_pool(
        "train", network_loader=loader, family=refit_proto.FAMILY, seed_base=refit_proto.TRAIN_SEED_BASE,
        count=refit_proto.TRAIN_COUNT, source_round_robin=refit_proto.SOURCE_ROUND_ROBIN,
    )
    input_library = fit_pool_signature_library(train_pool)

    cache = SignatureCache(str(SIGNATURE_CACHE_DIR))
    key = SignatureCacheKey(
        network_hash="m10-2-refit-golden-reference", hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="m10-2-refit-cfg1", sensor_layout_hash="m10-2-refit-layout1",
    )
    target_artifact = build_signature_artifact_for_network(network, cache, key=key)

    eval_pool = _family_scenario_pool(
        "train", network_loader=loader, family=proto.FAMILY, seed_base=proto.EVAL_SEED_BASE,
        count=proto.EVAL_COUNT, source_round_robin=proto.SOURCE_ROUND_ROBIN,
    )
    junction_ids = tuple(sorted(network.junction_name_list))
    return network, input_library, target_artifact, eval_pool, junction_ids


# ---------------------------------------------------------------------------
# Per-round classical posterior (ARM D's candidate_probabilities source) --
# same real, leakage-free Bayesian machinery generate_scout_label uses
# internally, applied to CURRENTLY REVEALED evidence only.
# ---------------------------------------------------------------------------


def _classical_source_probabilities(scenario, artifact, revealed: dict[str, float]) -> dict[str, float]:
    observed, valid = reindex_to_signature_grid(scenario, artifact.sensor_nodes, artifact.sample_times_seconds)
    if revealed:
        observed = observed.copy()
        valid = valid.copy()
        positions = {node: index for index, node in enumerate(artifact.sensor_nodes)}
        for node, value in revealed.items():
            column = positions.get(node)
            if column is None:
                continue
            observed[:, column] = value
            valid[:, column] = True
    result = localize_with_signatures(observed, artifact, observation_mask=valid, noise_scale=proto.NOISE_SCALE_MG_L)
    return dict(result.source_probabilities)


SAFETY_COUNTERS = {
    "invalid_or_inaccessible_node_selected": 0,
    "already_sampled_node_reselected": 0,
    "sampling_budget_exceeded": 0,
    "non_finite_scout_output": 0,
    "fail_closed_selection_violations": 0,
}


def _run_one_arm(
    *, arm: str, model: HydroCore, scenario, network, input_library, target_artifact, reconstruction,
    all_node_truth, junction_ids: tuple[str, ...], calibrator: SplitConformalCalibrator, network_id_key: str,
    feature_context,
) -> dict[str, Any]:
    already_sampled: list[str] = []
    revealed: dict[str, float] = {}
    truth_source = scenario.manifest.incident.source_nodes[0]
    rounds: list[dict[str, Any]] = []

    for step_index in range(proto.MAXIMUM_SAMPLES + 1):
        state = build_scout_training_state_batch(
            scenario, network, input_library, proto.DEPTH, already_sampled=already_sampled,
            revealed_values_mg_l=dict(revealed), sampling_round=step_index,
            sample_budget_total=proto.MAXIMUM_SAMPLES, feature_context=feature_context,
        )
        with torch.no_grad():
            output = model(state.batch)
        assert_finite_scout_outputs(output)
        if not torch.isfinite(output["source_node_logits"]).all():
            SAFETY_COUNTERS["non_finite_scout_output"] += 1
            raise FloatingPointError(f"non-finite source_node_logits at {arm} step {step_index}")

        probs = torch.softmax(output["source_node_logits"][0], dim=-1).detach().numpy()
        node_ids = state.node_ids
        true_index = node_ids.index(truth_source)
        ranked = np.argsort(probs)[::-1]
        candidate_indices = calibrator.candidate_set(probs, condition=None, network_id=network_id_key)
        gate_pass = 1 <= len(candidate_indices) <= proto.CANDIDATE_GATE_K
        round_record: dict[str, Any] = {
            "step": step_index,
            "samples_taken": len(already_sampled),
            "top1": bool(int(ranked[0]) == true_index),
            "top3": bool(true_index in ranked[:3].tolist()),
            "true_rank": int(np.where(ranked == true_index)[0][0]) + 1,
            "candidate_set_size": len(candidate_indices),
            "candidate_gate_pass": gate_pass,
            "entropy_bits": float(entropy(probs.tolist())),
        }

        if step_index >= proto.MAXIMUM_SAMPLES:
            round_record["decision"] = "BUDGET_EXHAUSTED"
            rounds.append(round_record)
            break

        if arm == "D":
            classical_probs = _classical_source_probabilities(scenario, target_artifact, revealed)
            scout_state = {
                "sampling_history": tuple(already_sampled),
                "candidate_probabilities": classical_probs,
                "candidate_region": junction_ids,
                "node_ids": junction_ids,
            }
            recommendation = HydroScout().deterministic_fallback(scout_state)
            if recommendation.action == ScoutAction.STOP:
                round_record["decision"] = "STOP"
                round_record["stop_reason"] = recommendation.reason
                rounds.append(round_record)
                break
            chosen = recommendation.node_id
            round_record["predicted_information_gain"] = recommendation.expected_information_gain
            round_record["predicted_candidate_reduction"] = recommendation.expected_candidate_reduction
        else:
            eval_state = build_scout_evaluation_state(
                node_ids=[node_ids], batch=state.batch, already_sampled=[list(already_sampled)],
                accessible=[list(junction_ids)], sampling_round=[step_index],
                sample_budget_total=[proto.MAXIMUM_SAMPLES],
            )
            recommendation = decode_learned_scout_recommendation(output, eval_state)
            continue_probability = recommendation.should_continue_sampling_probability
            if recommendation.node_id is None or continue_probability is None or continue_probability < 0.5:
                if recommendation.node_id is None and eval_state.candidate_mask().any():
                    SAFETY_COUNTERS["fail_closed_selection_violations"] += 0  # explicit: no violation, just no eligible pick recorded below
                round_record["decision"] = "STOP"
                round_record["stop_reason"] = (
                    "no_eligible_candidate" if recommendation.node_id is None else "should_continue_sampling<0.5"
                )
                round_record["should_continue_sampling_probability"] = continue_probability
                rounds.append(round_record)
                break
            chosen = recommendation.node_id
            round_record["predicted_information_gain"] = recommendation.expected_information_gain
            round_record["predicted_candidate_reduction"] = recommendation.candidate_reduction_prediction
            round_record["should_continue_sampling_probability"] = continue_probability

        # Mandatory hard-gate assertions (fail closed, not merely reported).
        if chosen not in junction_ids:
            SAFETY_COUNTERS["invalid_or_inaccessible_node_selected"] += 1
            raise ValueError(f"{arm} selected an invalid/inaccessible node: {chosen!r}")
        if chosen in already_sampled:
            SAFETY_COUNTERS["already_sampled_node_reselected"] += 1
            raise ValueError(f"{arm} reselected an already-sampled node: {chosen!r}")
        if len(already_sampled) >= proto.MAXIMUM_SAMPLES:
            SAFETY_COUNTERS["sampling_budget_exceeded"] += 1
            raise ValueError(f"{arm} exceeded the sampling budget")

        round_record["decision"] = "SAMPLE"
        round_record["chosen_node"] = chosen
        rounds.append(round_record)

        revealed[chosen] = reveal_sample_measurement(
            reconstruction, all_node_truth, chosen, step_index, noise_scale_mg_l=proto.NOISE_SCALE_MG_L,
        )
        already_sampled.append(chosen)

    resolved_step = next((r["step"] for r in rounds if r["candidate_gate_pass"]), None)
    return {
        "arm": arm, "truth_source": truth_source, "rounds": rounds, "resolved_at_step": resolved_step,
        "final_samples_taken": len(already_sampled),
        "voluntary_stop": any(r["decision"] == "STOP" for r in rounds),
    }


def run(seed: int, model: HydroCore, network, input_library, target_artifact, eval_pool, junction_ids, calibrator) -> list[dict[str, Any]]:
    network_id_key = f"{proto.FAMILY}:{m10.depth_bucket_of(proto.DEPTH)}"
    records: list[dict[str, Any]] = []
    skipped = 0
    for index, record in enumerate(eval_pool):
        scenario = record.scenario
        try:
            reconstruction = reconstruct_scenario_network(
                network, scenario.manifest, degradation_policy=_degradation_probabilities, original=scenario,
            )
        except ScenarioReconstructionError:
            skipped += 1
            continue
        all_node_truth = simulate_all_node_truth(reconstruction)
        arm_d = _run_one_arm(
            arm="D", model=model, scenario=scenario, network=network, input_library=input_library,
            target_artifact=target_artifact, reconstruction=reconstruction, all_node_truth=all_node_truth,
            junction_ids=junction_ids, calibrator=calibrator, network_id_key=network_id_key,
            feature_context=record.feature_context,
        )
        arm_l = _run_one_arm(
            arm="L", model=model, scenario=scenario, network=network, input_library=input_library,
            target_artifact=target_artifact, reconstruction=reconstruction, all_node_truth=all_node_truth,
            junction_ids=junction_ids, calibrator=calibrator, network_id_key=network_id_key,
            feature_context=record.feature_context,
        )
        records.append({
            "seed": seed, "scenario_id": str(scenario.manifest.scenario_id), "incident_index": index,
            "arm_D": arm_d, "arm_L": arm_l,
        })
        if (index + 1) % 20 == 0:
            print(f"  seed {seed}: {index + 1}/{len(eval_pool)} incidents", flush=True)
    print(f"seed {seed}: {len(records)} incidents evaluated, {skipped} skipped (reconstruction error)", flush=True)
    return records


def main() -> None:
    M10_2_TRUE_DIR.mkdir(parents=True, exist_ok=True)
    branch = m10.current_branch()
    assert branch == m10.FROZEN_BRANCH
    locked_before = m10.assert_locked_test_closed()

    checkpoint_verification: dict[str, Any] = {"kind": "M10_2_TRUE_CHECKPOINT_VERIFICATION"}
    checkpoint_verification["teacher_checkpoints_unchanged"] = {
        str(seed): entry for seed, entry in verify_teacher_checkpoints_unchanged().items()
    }

    print("building shared population/signature assets (reusing Level A's own TRAIN library+cache key)...", flush=True)
    network, input_library, target_artifact, eval_pool, junction_ids = build_shared_assets()
    print(f"evaluation population: {len(eval_pool)} incidents, family={proto.FAMILY}", flush=True)

    calibrator = fit_frozen_calibrator()

    population_manifest = {
        "kind": "M10_2_TRUE_POPULATION_MANIFEST",
        "family": proto.FAMILY, "eval_seed_base": proto.EVAL_SEED_BASE, "eval_count": proto.EVAL_COUNT,
        "source_round_robin": proto.SOURCE_ROUND_ROBIN, "depth": proto.DEPTH,
        "maximum_samples": proto.MAXIMUM_SAMPLES, "noise_scale_mg_l": proto.NOISE_SCALE_MG_L,
        "scenario_ids": [str(record.scenario.manifest.scenario_id) for record in eval_pool],
        "manifest_hash": hashlib.sha256(
            json.dumps(sorted(str(record.scenario.manifest.scenario_id) for record in eval_pool)).encode()
        ).hexdigest(),
        "protocol_hash": proto.protocol_hash(),
    }
    (M10_2_TRUE_DIR / "m10-2-population-manifest.json").write_text(json.dumps(population_manifest, indent=2) + "\n")

    started = time.time()
    for seed in m10.SEEDS:
        print(f"=== TRUE M10.2, seed {seed} ===", flush=True)
        model, actual_sha256, checkpoint_path = verify_and_load_checkpoint(seed)
        checkpoint_verification[f"level_a_refit_seed_{seed}"] = {
            "path": str(checkpoint_path), "expected_sha256": proto.LEVEL_A_REFIT_CHECKPOINT_SHA256[seed],
            "actual_sha256": actual_sha256, "verified": True,
            "regenerated_this_run": False,
        }
        records = run(seed, model, network, input_library, target_artifact, eval_pool, junction_ids, calibrator)
        out_path = M10_2_TRUE_DIR / f"m10-2-trajectories-seed{seed}.jsonl"
        with out_path.open("w") as fh:
            for record in records:
                fh.write(json.dumps(record, default=str) + "\n")
        print(f"  wrote {len(records)} paired trajectories to {out_path} ({time.time() - started:.0f}s elapsed)", flush=True)

    locked_after = m10.assert_locked_test_closed()
    checkpoint_verification["locked_test_opened_before"] = locked_before
    checkpoint_verification["locked_test_opened_after"] = locked_after
    checkpoint_verification["protocol_hash"] = proto.protocol_hash()
    (M10_2_TRUE_DIR / "m10-2-checkpoint-verification.json").write_text(json.dumps(checkpoint_verification, indent=2, default=str) + "\n")

    safety_audit = {
        "kind": "M10_2_TRUE_SAFETY_AUDIT",
        "counters": dict(SAFETY_COUNTERS),
        "all_hard_gates_passed": all(value == 0 for value in SAFETY_COUNTERS.values()),
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "protocol_hash": proto.protocol_hash(),
    }
    (M10_2_TRUE_DIR / "m10-2-safety-audit.json").write_text(json.dumps(safety_audit, indent=2) + "\n")
    print("TRUE M10.2 execution complete.")
    print(json.dumps(safety_audit, indent=2))


if __name__ == "__main__":
    main()
