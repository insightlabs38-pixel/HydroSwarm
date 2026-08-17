"""Milestone 10.2 Scout refit -- Level-A representation-sufficiency gate
evaluation, per `docs/evaluation/HYDROCORE_V5_M10_2_SCOUT_REFIT_PROTOCOL.md`
Section 8 (criteria fixed BEFORE any competence number here was computed).

Uses `m10_2_refit_protocol.VALIDATION_COUNT_AMENDMENT_1` (300, not the
original 100) -- see that module's own dated amendment docstring: the
support shortfall (17 < GATE_MIN_SUPPORT=20) was discovered from corpus SIZE
statistics alone, before any competence metric existed, so enlarging N here
is a resourcing correction, not a change to the gate's own criteria.

Does not train anything. Loads the 3 already-trained Level-A checkpoints
(`reports/evaluation/hydrocore-v5/m10/m10-2-refit/checkpoints/level-a-seed{seed}/`)
and evaluates them against a freshly built (larger) validation population,
disjoint from every other split (same `VALIDATION_SEED_BASE`, so the
original 100 validation scenarios are an exact PREFIX of these 300 -- no
scenario is swapped or dropped, only extended).

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-2-refit/m10-2-refit-level-a-gate.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import m10_2_refit_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402
from run_m7_topology import TRAINED_FAMILIES  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey  # noqa: E402
from hydroswarm.evaluation.scout_state import (  # noqa: E402
    build_scout_evaluation_state,
    decode_learned_scout_recommendation,
)
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import _degradation_probabilities, fit_pool_signature_library  # noqa: E402
from hydroswarm.training.scenario_reconstruction import ScenarioReconstructionError, reconstruct_scenario_network  # noqa: E402
from hydroswarm.training.scout_labels import build_signature_artifact_for_network  # noqa: E402
from hydroswarm.training.scout_targets import ScoutTrainingExample, build_scout_training_examples  # noqa: E402

from run_m10_2_level_a_train import _family_scenario_pool  # noqa: E402

M10_2_REFIT_DIR = m10.M10_DIR / "m10-2-refit"
SIGNATURE_CACHE_DIR = ROOT / "experiments" / "cache" / "m10-2-refit-signatures"


def _bootstrap_ci(values: np.ndarray, *, statistic: Callable[[np.ndarray], float], resamples: int, ci: float, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        stats[i] = statistic(values[idx])
    tail = (1.0 - ci) / 2.0
    return float(np.percentile(stats, tail * 100)), float(np.percentile(stats, (1.0 - tail) * 100))


def _paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, *, resamples: int, ci: float, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(a[idx]) - np.mean(b[idx]))
    tail = (1.0 - ci) / 2.0
    return float(np.percentile(diffs, tail * 100)), float(np.percentile(diffs, (1.0 - tail) * 100))


def _spearman_ci(pred: np.ndarray, target: np.ndarray, *, resamples: int, ci: float, seed: int) -> tuple[float, float, float]:
    point = float(spearmanr(pred, target).statistic)
    rng = np.random.default_rng(seed)
    n = len(pred)
    stats = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        result = spearmanr(pred[idx], target[idx]).statistic
        stats[i] = 0.0 if np.isnan(result) else float(result)
    tail = (1.0 - ci) / 2.0
    return point, float(np.percentile(stats, tail * 100)), float(np.percentile(stats, (1.0 - tail) * 100))


def _build_examples(*, seed_base: int, count: int, network, loader) -> tuple[list[ScoutTrainingExample], Any]:
    pool = _family_scenario_pool(
        "train", network_loader=loader, family=proto.FAMILY, seed_base=seed_base, count=count,
        source_round_robin=proto.SOURCE_ROUND_ROBIN,
    )
    input_library = fit_pool_signature_library(pool)
    cache = SignatureCache(str(SIGNATURE_CACHE_DIR))
    key = SignatureCacheKey(
        network_hash="m10-2-refit-golden-reference", hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="m10-2-refit-cfg1", sensor_layout_hash="m10-2-refit-layout1",
    )
    artifact = build_signature_artifact_for_network(network, cache, key=key)
    examples: list[ScoutTrainingExample] = []
    for record in pool:
        try:
            reconstruction = reconstruct_scenario_network(
                network, record.scenario.manifest, degradation_policy=_degradation_probabilities, original=record.scenario,
            )
        except ScenarioReconstructionError:
            continue
        node_ids = tuple(sorted(network.node_name_list))
        examples.extend(
            build_scout_training_examples(
                record.scenario, record.network, input_library, artifact, node_ids, proto.DEPTH,
                reconstruction=reconstruction, maximum_samples=proto.MAXIMUM_SAMPLES,
                noise_scale_mg_l=proto.NOISE_SCALE_MG_L, feature_context=record.feature_context,
            )
        )
    return examples, input_library


def _predict(model: HydroCore, example: ScoutTrainingExample) -> dict[str, Any]:
    with torch.no_grad():
        output = model(example.state.batch)
    eval_state = build_scout_evaluation_state(
        node_ids=[example.state.node_ids], batch=example.state.batch,
        already_sampled=[list(example.already_sampled)], sampling_round=[example.step_index],
        sample_budget_total=[proto.MAXIMUM_SAMPLES],
    )
    recommendation = decode_learned_scout_recommendation(output, eval_state)
    return {
        "predicted_node_id": recommendation.node_id,
        "expected_information_gain": output["expected_information_gain"][0].numpy(),
        "candidate_reduction_prediction": output["candidate_reduction_prediction"][0].numpy(),
        "should_continue_sampling_probability": float(torch.sigmoid(output["should_continue_sampling_logits"][0])),
    }


def _evaluate_seed(model: HydroCore, examples: list[ScoutTrainingExample], train_stats: dict[str, float]) -> dict[str, Any]:
    sample_node_model_correct: list[int] = []
    sample_node_naive_expected: list[float] = []
    ig_pred: list[float] = []
    ig_target: list[float] = []
    cr_pred: list[float] = []
    cr_target: list[float] = []
    stop_pred: list[int] = []
    stop_target: list[int] = []

    for example in examples:
        prediction = _predict(model, example)
        if bool(example.targets["sample_node_mask"]):
            target_node_id = example.state.node_ids[int(example.targets["sample_node"])]
            sample_node_model_correct.append(int(prediction["predicted_node_id"] == target_node_id))
            n_eligible = int(
                sum(
                    1 for node_id in example.state.node_ids
                    if node_id not in example.already_sampled and node_id in example.state.junction_ids
                )
            )
            sample_node_naive_expected.append(1.0 / max(1, n_eligible))

        mask = example.targets["information_gain_mask"].numpy().astype(bool)
        ig_pred.extend(prediction["expected_information_gain"][mask].tolist())
        ig_target.extend(example.targets["information_gain"].numpy()[mask].tolist())

        mask = example.targets["candidate_reduction_mask"].numpy().astype(bool)
        cr_pred.extend(prediction["candidate_reduction_prediction"][mask].tolist())
        cr_target.extend(example.targets["candidate_reduction"].numpy()[mask].tolist())

        stop_pred.append(int(prediction["should_continue_sampling_probability"] >= 0.5))
        stop_target.append(int(bool(example.targets["should_continue_sampling"])))

    resamples, ci, seed = proto.BOOTSTRAP_RESAMPLES, proto.BOOTSTRAP_CI, proto.BOOTSTRAP_SEED
    result: dict[str, Any] = {}

    n_sn = len(sample_node_model_correct)
    result["sample_node"] = {"n": n_sn}
    if n_sn > 0:
        model_arr = np.array(sample_node_model_correct, dtype=np.float64)
        naive_arr = np.array(sample_node_naive_expected, dtype=np.float64)
        diff_lower, diff_upper = _paired_bootstrap_ci(model_arr, naive_arr, resamples=resamples, ci=ci, seed=seed)
        result["sample_node"].update({
            "model_top1_accuracy": float(model_arr.mean()),
            "naive_baseline_top1_accuracy": float(naive_arr.mean()),
            "diff_ci_lower": diff_lower, "diff_ci_upper": diff_upper,
            "criterion_3_passed": diff_lower > 0.0,
        })
    else:
        result["sample_node"]["criterion_3_passed"] = False

    for name, pred_list, target_list, baseline_key in (
        ("information_gain", ig_pred, ig_target, "information_gain_train_mean"),
        ("candidate_reduction", cr_pred, cr_target, "candidate_reduction_train_mean"),
    ):
        pred_arr = np.array(pred_list, dtype=np.float64)
        target_arr = np.array(target_list, dtype=np.float64)
        n = len(pred_arr)
        entry: dict[str, Any] = {"n": n}
        if n > 1:
            model_mse = float(np.mean((pred_arr - target_arr) ** 2))
            baseline_value = train_stats[baseline_key]
            baseline_mse = float(np.mean((baseline_value - target_arr) ** 2))
            point, lower, upper = _spearman_ci(pred_arr, target_arr, resamples=resamples, ci=ci, seed=seed)
            entry.update({
                "model_mse": model_mse, "baseline_mse": baseline_mse, "baseline_value": baseline_value,
                "spearman": point, "spearman_ci_lower": lower, "spearman_ci_upper": upper,
                "criterion_passed": bool(model_mse < baseline_mse and lower > 0.0),
            })
        else:
            entry["criterion_passed"] = False
        result[name] = entry

    stop_pred_arr = np.array(stop_pred)
    stop_target_arr = np.array(stop_target)
    n_stop = len(stop_target_arr)
    majority_class = train_stats["should_continue_sampling_train_majority_class"]
    correct = (stop_pred_arr == stop_target_arr).astype(np.float64)
    majority_correct = (stop_target_arr == majority_class).astype(np.float64)
    lower, upper = (
        _bootstrap_ci(correct, statistic=np.mean, resamples=resamples, ci=ci, seed=seed) if n_stop > 0 else (0.0, 0.0)
    )
    result["should_continue_sampling"] = {
        "n": n_stop,
        "model_accuracy": float(correct.mean()) if n_stop else None,
        "accuracy_ci_lower": lower, "accuracy_ci_upper": upper,
        "majority_baseline_accuracy": float(majority_correct.mean()) if n_stop else None,
        "both_classes_present": bool(len(set(stop_target)) > 1) if n_stop else False,
        "criterion_6_passed": bool(n_stop > 0 and lower > float(majority_correct.mean())),
    }
    return result


def main() -> None:
    locked_before = m10.assert_locked_test_closed()
    family, loader = TRAINED_FAMILIES[0]
    network = loader()

    print("rebuilding TRAIN examples for baseline statistics...", flush=True)
    train_examples, _ = _build_examples(seed_base=proto.TRAIN_SEED_BASE, count=proto.TRAIN_COUNT, network=network, loader=loader)
    train_ig = np.array([
        value for ex in train_examples for value, valid in zip(ex.targets["information_gain"].numpy(), ex.targets["information_gain_mask"].numpy()) if valid
    ])
    train_cr = np.array([
        value for ex in train_examples for value, valid in zip(ex.targets["candidate_reduction"].numpy(), ex.targets["candidate_reduction_mask"].numpy()) if valid
    ])
    train_stop = np.array([int(bool(ex.targets["should_continue_sampling"])) for ex in train_examples])
    train_stats = {
        "information_gain_train_mean": float(train_ig.mean()),
        "candidate_reduction_train_mean": float(train_cr.mean()),
        "should_continue_sampling_train_majority_class": int(round(float(train_stop.mean()))),
    }

    print(f"building ENLARGED VALIDATION examples (count={proto.VALIDATION_COUNT_AMENDMENT_1})...", flush=True)
    validation_examples, _ = _build_examples(
        seed_base=proto.VALIDATION_SEED_BASE, count=proto.VALIDATION_COUNT_AMENDMENT_1, network=network, loader=loader,
    )
    n_with_recommendation = sum(1 for ex in validation_examples if bool(ex.targets["sample_node_mask"]))
    print(f"validation examples: {len(validation_examples)}, with real recommendation: {n_with_recommendation}", flush=True)

    per_seed: dict[str, Any] = {}
    for seed in m10.SEEDS:
        print(f"=== evaluating Level-A gate, seed {seed} ===", flush=True)
        model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
        checkpoint_dir = M10_2_REFIT_DIR / "checkpoints" / f"level-a-seed{seed}"
        model.load_state_dict(load_file(checkpoint_dir / "model.safetensors", device="cpu"), strict=True)
        model.eval()
        per_seed[str(seed)] = _evaluate_seed(model, validation_examples, train_stats)

    gate_per_seed: dict[str, bool] = {}
    for seed_key, metrics in per_seed.items():
        passed = (
            metrics["sample_node"].get("criterion_3_passed", False)
            and metrics["information_gain"].get("criterion_passed", False)
            and metrics["candidate_reduction"].get("criterion_passed", False)
            and metrics["should_continue_sampling"].get("criterion_6_passed", False)
        )
        gate_per_seed[seed_key] = passed

    support_ok = n_with_recommendation >= proto.GATE_MIN_SUPPORT and len(validation_examples) >= proto.GATE_MIN_SUPPORT
    locked_after = m10.assert_locked_test_closed()

    doc = {
        "kind": "M10_2_REFIT_LEVEL_A_GATE",
        "protocol_hash": proto.protocol_hash(),
        "validation_count_used": proto.VALIDATION_COUNT_AMENDMENT_1,
        "validation_count_amendment_reason": proto.VALIDATION_COUNT_AMENDMENT_1_REASON,
        "n_validation_examples": len(validation_examples),
        "n_validation_examples_with_real_recommendation": n_with_recommendation,
        "gate_min_support": proto.GATE_MIN_SUPPORT,
        "support_criterion_2_passed": support_ok,
        "train_baseline_stats": train_stats,
        "per_seed_metrics": per_seed,
        "per_seed_scout_competence_gate_passed": gate_per_seed,
        "all_seeds_pass_competence_gate": all(gate_per_seed.values()),
        "any_seed_passes_competence_gate": any(gate_per_seed.values()),
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    (M10_2_REFIT_DIR / "m10-2-refit-level-a-gate.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print(json.dumps({"support_ok": support_ok, "gate_per_seed": gate_per_seed}, indent=2))


if __name__ == "__main__":
    main()
