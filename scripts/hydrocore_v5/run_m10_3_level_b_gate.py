"""M10.3A Strategist refit -- Level-B competence gate evaluation, per
`docs/evaluation/HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md`
(criteria fixed BEFORE any competence number here was computed). Only run
because the Level-A gate mechanically triggered escalation.

Does not train anything. Loads the 3 already-trained Level-B checkpoints
(`reports/evaluation/hydrocore-v5/m10/m10-3-refit/checkpoints/level-b-seed{seed}/`)
and evaluates them against the SAME validation population the Level-A gate
used (same seed base/count -- rebuilt here, deterministically). Uses the
SAME Section-8 criteria (3)-(6) as Level A, PLUS the frozen Level-B
promotion criterion A: material improvement over Level A's own validation
numbers under the same paired-bootstrap procedure (CI lower bound of
Level B's point estimate exceeds Level A's own point estimate, for the
criteria Level A failed).

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-level-b-gate.json
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

import m10_3_refit_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402
from run_m7_topology import TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402
from run_m10_3_level_a_train import CorpusExample, _build_corpus  # noqa: E402

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import fit_pool_signature_library  # noqa: E402
from hydroswarm.training.scout_labels import build_signature_artifact_for_network  # noqa: E402

M10_3_REFIT_DIR = m10.M10_DIR / "m10-3-refit"
SIGNATURE_CACHE_DIR = ROOT / "experiments" / "cache" / "m10-3-refit-signatures"


def _bootstrap_ci(values: np.ndarray, *, statistic: Callable[[np.ndarray], float], resamples: int, ci: float, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        stats[i] = statistic(values[idx])
    tail = (1.0 - ci) / 2.0
    return float(np.percentile(stats, tail * 100)), float(np.percentile(stats, (1.0 - tail) * 100))


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(ranks[labels.astype(bool)].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


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


def _predict(model: HydroCore, example: CorpusExample) -> dict[str, np.ndarray]:
    with torch.no_grad():
        output = model(example.inputs)
    result: dict[str, np.ndarray] = {
        "plan_validity_probability": torch.softmax(output["plan_validity_logits"][0], dim=-1)[:, 1].numpy(),
    }
    for name in ("plan_value", "exposure_proxy", "pressure_risk_proxy", "service_loss_proxy",
                 "containment_time_proxy", "plan_regret_proxy"):
        result[name] = output[name][0].numpy()
    return result


def _pairwise_ranking_accuracy_per_incident(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Returns (correct_pairs, total_pairs) for one incident's real,
    non-tied, both-valid candidate pairs -- predicted relative order
    (sign of pred[i]-pred[j]) matches true relative order."""

    valid_indices = np.flatnonzero(mask)
    correct = 0
    total = 0
    for a in range(len(valid_indices)):
        for b in range(a + 1, len(valid_indices)):
            i, j = valid_indices[a], valid_indices[b]
            if target[i] == target[j]:
                continue
            total += 1
            true_order = target[i] > target[j]
            pred_order = pred[i] > pred[j]
            if true_order == pred_order:
                correct += 1
    return correct, total


def main() -> None:
    locked_before = m10.assert_locked_test_closed()
    family, loader = TRAINED_FAMILIES[0]
    network = loader()

    print("rebuilding TRAIN examples for baseline statistics...", flush=True)
    train_pool = _family_scenario_pool(
        "train", network_loader=loader, family=family, seed_base=proto.TRAIN_SEED_BASE, count=proto.TRAIN_COUNT,
        source_round_robin=proto.SOURCE_ROUND_ROBIN,
    )
    input_library = fit_pool_signature_library(train_pool)
    cache = SignatureCache(str(SIGNATURE_CACHE_DIR))
    key = SignatureCacheKey(
        network_hash="m10-3-refit-golden-reference", hydraulic_state_hash=HydraulicSimulator(network).state_hash(),
        simulator_version="v1", configuration_hash="m10-3-refit-cfg1", sensor_layout_hash="m10-3-refit-layout1",
    )
    artifact = build_signature_artifact_for_network(network, cache, key=key)

    train_examples = _build_corpus(
        seed_base=proto.TRAIN_SEED_BASE, count=proto.TRAIN_COUNT, network=network, loader=loader,
        input_library=input_library, artifact=artifact,
    )
    train_stats: dict[str, float] = {}
    train_validity = np.concatenate([ex.targets["plan_validity"][ex.targets["plan_validity_mask"]].numpy() for ex in train_examples])
    train_stats["plan_validity_train_majority_class"] = int(round(float(train_validity.mean())))
    for name in ("plan_value", "exposure_proxy", "pressure_risk_proxy", "service_loss_proxy",
                 "containment_time_proxy", "plan_regret_proxy"):
        values = np.concatenate([ex.targets[name][ex.targets[f"{name}_mask"]].numpy() for ex in train_examples])
        train_stats[f"{name}_train_mean"] = float(values.mean())

    print(f"building VALIDATION examples (count={proto.VALIDATION_COUNT})...", flush=True)
    validation_examples = _build_corpus(
        seed_base=proto.VALIDATION_SEED_BASE, count=proto.VALIDATION_COUNT, network=network, loader=loader,
        input_library=input_library, artifact=artifact,
    )
    print(f"validation examples: {len(validation_examples)}", flush=True)

    level_a_gate = json.loads((M10_3_REFIT_DIR / "m10-3-refit-level-a-gate.json").read_text())

    resamples, ci, seed = proto.BOOTSTRAP_RESAMPLES, proto.BOOTSTRAP_CI, proto.BOOTSTRAP_SEED
    per_seed: dict[str, Any] = {}
    for model_seed in m10.SEEDS:
        print(f"=== evaluating Level-B gate, seed {model_seed} ===", flush=True)
        model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
        checkpoint_dir = M10_3_REFIT_DIR / "checkpoints" / f"level-b-seed{model_seed}"
        model.load_state_dict(load_file(checkpoint_dir / "model.safetensors", device="cpu"), strict=True)
        model.eval()

        predictions = [_predict(model, ex) for ex in validation_examples]
        metrics: dict[str, Any] = {}

        validity_target = np.concatenate([ex.targets["plan_validity"].numpy() for ex in validation_examples])
        validity_mask = np.concatenate([ex.targets["plan_validity_mask"].numpy() for ex in validation_examples])
        validity_pred = np.concatenate([p["plan_validity_probability"] for p in predictions])
        v_target, v_pred = validity_target[validity_mask].astype(int), validity_pred[validity_mask]
        auroc_point = _auroc(v_pred, v_target)
        auroc_lower, auroc_upper = (
            _bootstrap_ci(np.arange(len(v_pred)), statistic=lambda idx: _auroc(v_pred[idx], v_target[idx]),
                          resamples=resamples, ci=ci, seed=seed)
            if not np.isnan(auroc_point) else (float("nan"), float("nan"))
        )
        majority_class = train_stats["plan_validity_train_majority_class"]
        majority_accuracy = float((v_target == majority_class).mean())
        metrics["plan_validity"] = {
            "n": int(validity_mask.sum()), "auroc": auroc_point, "auroc_ci_lower": auroc_lower,
            "auroc_ci_upper": auroc_upper, "majority_baseline_accuracy": majority_accuracy,
            "criterion_passed": bool(not np.isnan(auroc_lower) and auroc_lower > 0.5),
        }

        for name in ("plan_value", "exposure_proxy", "pressure_risk_proxy", "service_loss_proxy",
                     "containment_time_proxy", "plan_regret_proxy"):
            target = np.concatenate([ex.targets[name].numpy() for ex in validation_examples])
            mask = np.concatenate([ex.targets[f"{name}_mask"].numpy() for ex in validation_examples])
            pred = np.concatenate([p[name] for p in predictions])
            t, p = target[mask], pred[mask]
            n = len(t)
            entry: dict[str, Any] = {"n": n}
            if n > 1:
                model_mse = float(np.mean((p - t) ** 2))
                baseline_value = train_stats[f"{name}_train_mean"]
                baseline_mse = float(np.mean((baseline_value - t) ** 2))
                point, lower, upper = _spearman_ci(p, t, resamples=resamples, ci=ci, seed=seed)
                entry.update({
                    "model_mse": model_mse, "baseline_mse": baseline_mse, "baseline_value": baseline_value,
                    "spearman": point, "spearman_ci_lower": lower, "spearman_ci_upper": upper,
                    "criterion_passed": bool(model_mse < baseline_mse and lower > 0.0),
                })
            else:
                entry["criterion_passed"] = False
            metrics[name] = entry

        correct_total = 0
        pairs_total = 0
        per_incident_accuracy: list[float] = []
        for ex, pred in zip(validation_examples, predictions):
            mask = ex.targets["plan_value_mask"].numpy()
            target = ex.targets["plan_value"].numpy()
            correct, total = _pairwise_ranking_accuracy_per_incident(pred["plan_value"], target, mask)
            if total > 0:
                correct_total += correct
                pairs_total += total
                per_incident_accuracy.append(correct / total)
        per_incident_arr = np.array(per_incident_accuracy)
        ranking_lower, ranking_upper = (
            _bootstrap_ci(per_incident_arr, statistic=np.mean, resamples=resamples, ci=ci, seed=seed)
            if len(per_incident_arr) > 0 else (float("nan"), float("nan"))
        )
        metrics["pairwise_ranking"] = {
            "n_incidents_with_a_real_pair": len(per_incident_accuracy),
            "n_pairs_total": pairs_total,
            "pooled_accuracy": (correct_total / pairs_total) if pairs_total else None,
            "mean_per_incident_accuracy": float(per_incident_arr.mean()) if len(per_incident_arr) else None,
            "ci_lower": ranking_lower, "ci_upper": ranking_upper,
            "criterion_passed": bool(not np.isnan(ranking_lower) and ranking_lower > 0.5),
        }

        all_finite = all(
            np.isfinite(p[name]).all()
            for p in predictions
            for name in ("plan_validity_probability", "plan_value", "exposure_proxy", "pressure_risk_proxy",
                         "service_loss_proxy", "containment_time_proxy", "plan_regret_proxy")
        )
        support_ok = (
            metrics["plan_validity"]["n"] >= proto.GATE_MIN_SUPPORT
            and all(metrics[name]["n"] >= proto.GATE_MIN_SUPPORT for name in (
                "plan_value", "exposure_proxy", "pressure_risk_proxy", "service_loss_proxy",
                "containment_time_proxy", "plan_regret_proxy"))
            and metrics["pairwise_ranking"]["n_pairs_total"] >= proto.GATE_MIN_SUPPORT
        )

        gate_criteria_passed = (
            support_ok and all_finite
            and metrics["plan_validity"]["criterion_passed"]
            and all(metrics[name]["criterion_passed"] for name in (
                "plan_value", "exposure_proxy", "pressure_risk_proxy", "service_loss_proxy",
                "containment_time_proxy", "plan_regret_proxy"))
            and metrics["pairwise_ranking"]["criterion_passed"]
        )
        # Frozen Level-B promotion criterion A: material improvement over
        # Level A's own point estimate, for every criterion Level A failed
        # (Section 9's "CI lower bound exceeds Level A's own point estimate").
        level_a_metrics = level_a_gate["per_seed"][str(model_seed)]["metrics"]
        material_improvement: dict[str, Any] = {}
        scalar_metric_keys = {
            "plan_validity": ("auroc", "auroc_ci_lower"),
            "plan_value": ("spearman", "spearman_ci_lower"),
            "exposure_proxy": ("spearman", "spearman_ci_lower"),
            "pressure_risk_proxy": ("spearman", "spearman_ci_lower"),
            "service_loss_proxy": ("spearman", "spearman_ci_lower"),
            "containment_time_proxy": ("spearman", "spearman_ci_lower"),
            "plan_regret_proxy": ("spearman", "spearman_ci_lower"),
            "pairwise_ranking": ("pooled_accuracy", "ci_lower"),
        }
        for name, (point_key, ci_lower_key) in scalar_metric_keys.items():
            level_a_failed = not level_a_metrics[name].get("criterion_passed", False)
            if not level_a_failed:
                continue  # only compare metrics Level A itself failed.
            level_a_point = level_a_metrics[name].get(point_key)
            level_b_ci_lower = metrics[name].get(ci_lower_key)
            improved = (
                level_a_point is not None and level_b_ci_lower is not None
                and not (isinstance(level_a_point, float) and level_a_point != level_a_point)  # not NaN
                and not (isinstance(level_b_ci_lower, float) and level_b_ci_lower != level_b_ci_lower)
                and level_b_ci_lower > level_a_point
            )
            material_improvement[name] = {
                "level_a_point_estimate": level_a_point, "level_b_ci_lower": level_b_ci_lower,
                "materially_improved": bool(improved),
            }
        materially_improves_over_a = (
            bool(material_improvement) and all(entry["materially_improved"] for entry in material_improvement.values())
        )

        per_seed[str(model_seed)] = {
            "metrics": metrics, "support_ok": support_ok, "all_finite": all_finite,
            "gate_criteria_passed": gate_criteria_passed,
            "material_improvement_over_level_a": material_improvement,
            "materially_improves_over_level_a": materially_improves_over_a,
        }

    locked_after = m10.assert_locked_test_closed()
    doc = {
        "kind": "M10_3_REFIT_LEVEL_B_GATE",
        "protocol_hash": proto.protocol_hash(),
        "n_train_examples": len(train_examples),
        "n_validation_examples": len(validation_examples),
        "train_baseline_stats": train_stats,
        "per_seed": per_seed,
        "all_seeds_pass_competence_gate": all(entry["gate_criteria_passed"] for entry in per_seed.values()),
        "any_seed_passes_competence_gate": any(entry["gate_criteria_passed"] for entry in per_seed.values()),
        "all_seeds_materially_improve_over_level_a": all(entry["materially_improves_over_level_a"] for entry in per_seed.values()),
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    (M10_3_REFIT_DIR / "m10-3-refit-level-b-gate.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print(json.dumps({seed: entry["gate_criteria_passed"] for seed, entry in per_seed.items()}, indent=2))


if __name__ == "__main__":
    main()
