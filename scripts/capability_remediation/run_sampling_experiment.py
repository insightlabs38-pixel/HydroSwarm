"""Final matched causal EIG versus random sampling experiment.

This is diagnostic instrumentation, not an alternate decision path.  It uses
the production pipeline for every initial analysis and reanalysis, and records
the authority gates which remain after each physically matched measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import wntr  # noqa: E402

from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.preprocessing.builder import SensorSeries  # noqa: E402
from hydroswarm.runtime import V4PipelineFactory  # noqa: E402
from hydroswarm.simulation import HydraulicSimulator  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402


SEEDS = tuple(range(2026081801, 2026081841))
NOISE_STD = 0.05
INITIAL_STEPS = 3
MAX_SAMPLES = 3
OUT = ROOT / "reports/evaluation/capability-remediation"


def _commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()


def _entropy(result: Any) -> float:
    return result.posterior_history[-1].entropy_bits


def _rank(result: Any, truth: str) -> int:
    nodes = sorted(result.fused_belief, key=lambda node: (-result.fused_belief[node], node))
    return nodes.index(truth) + 1


def _state(result: Any, truth: str, *, sample_budget_remaining: int, threshold: float) -> dict[str, Any]:
    reasons = list(result.planning_suppression_reasons)
    candidate_size = len(result.conformal_candidate_nodes)
    learned = result.semantic_predictions.evidence_sufficiency
    deterministic = bool(
        result.calibrated
        and result.ood_level.value == "NORMAL"
        and 0 < candidate_size <= 3
    )
    return {
        "entropy": _entropy(result),
        "true_source_rank": _rank(result, truth),
        "true_source_probability": result.fused_belief.get(truth, 0.0),
        "top1_correct": _rank(result, truth) == 1,
        "candidate_set_size": candidate_size,
        "candidate_contains_true_source": truth in result.conformal_candidate_nodes,
        "planning_maximum_candidate_count": 3,
        "calibrated": result.calibrated,
        "calibration_source": result.calibration_source,
        "calibration_group_identifier": result.calibration_group_identifier,
        "ood_level": result.ood_level.value,
        "neural_classical_disagreement": (
            result.fusion_diagnostics.disagreement_js if result.fusion_diagnostics else 0.0
        ),
        "learned_evidence_sufficiency": learned,
        "learned_evidence_threshold": threshold,
        "learned_evidence_sufficient": None if learned is None else learned >= threshold,
        "deterministic_evidence_sufficient": deterministic,
        "planning_allowed": result.planning_allowed,
        "control_action": result.control_action.value,
        "suppression_reasons": reasons,
        "sample_budget_remaining": sample_budget_remaining,
    }


def _measurement(
    network: Any,
    scenario: Any,
    node: str,
    *,
    decision_seconds: float,
    delay_minutes: float,
    seed: int,
) -> tuple[SensorSeries, dict[str, float]]:
    exact = HydraulicSimulator(network).simulate_incident(
        scenario.manifest.incident.source_nodes[0],
        strength_mg_min=10.0 * scenario.manifest.incident.relative_strength,
        start_minute=scenario.manifest.incident.start_minute,
        duration_minutes=scenario.manifest.incident.duration_minutes,
    )
    acquisition_seconds = decision_seconds + delay_minutes * 60.0
    index = int(np.argmin(np.abs(np.asarray(exact.concentration_mg_l.index, dtype=float) - acquisition_seconds)))
    timestamp = float(exact.concentration_mg_l.index[index])
    truth = float(exact.concentration_mg_l.loc[:, node].iloc[index])
    noise = float(np.random.default_rng(seed).normal(0.0, NOISE_STD))
    observed = max(0.0, truth + noise)
    return (
        SensorSeries(
            node_id=node,
            timestamps_seconds=(timestamp,), concentration_mg_l=(observed,),
            pressure_m=(25.0,), health=(1.0,), missing=(False,), drift=(False,),
            delayed=(False,), frozen=(False,),
        ),
        {
            "measurement_timestamp_seconds": timestamp,
            "measurement_value_mg_l": observed,
            "measurement_noise_realization_mg_l": noise,
            "measurement_noise_free_value_mg_l": truth,
        },
    )


def _stop_reason(result: Any, evidence: list[SensorSeries], network: Any) -> str | None:
    if result.planning_allowed:
        return "planning_already_allowed"
    if not [node for node in network.junction_name_list if node not in {item.node_id for item in evidence}]:
        return "no_unobserved_candidate"
    if result.control_action.value != "REQUEST_SAMPLE":
        return f"control_action_{result.control_action.value}"
    if result.sample_result is None:
        return "missing_authoritative_sample_result"
    if result.sample_result.stop:
        return result.sample_result.stop_reason or "sampler_stopped"
    if result.sample_result.recommended_node is None:
        return "sampler_returned_no_node"
    return None


def _run_strategy(
    name: str,
    pipeline: Any,
    network: Any,
    scenario: Any,
    initial: list[SensorSeries],
    truth: str,
    seed: int,
) -> dict[str, Any]:
    evidence = list(initial)
    result = pipeline.analyze(uuid.uuid4(), network, evidence, sample_budget_remaining=MAX_SAMPLES)
    states = [_state(result, truth, sample_budget_remaining=MAX_SAMPLES, threshold=pipeline.evidence_threshold)]
    rounds: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for sample_index in range(MAX_SAMPLES):
        before = states[-1]
        stop_reason = _stop_reason(result, evidence, network)
        if stop_reason is not None:
            break
        choices = [node for node in network.junction_name_list if node not in {item.node_id for item in evidence}]
        if name == "EIG":
            candidate = next(
                item for item in result.sample_result.ranked
                if item.node_id == result.sample_result.recommended_node
            )
            node, expected, delay = candidate.node_id, candidate.expected_information_gain_bits, candidate.collection_time_minutes
        else:
            node = sorted(choices)[int(hashlib.sha256(f"{seed}:{sample_index}".encode()).hexdigest(), 16) % len(choices)]
            expected, delay = None, 30.0
        decision = max(timestamp for item in evidence for timestamp in item.timestamps_seconds)
        observation, measurement = _measurement(
            network, scenario, node, decision_seconds=decision, delay_minutes=delay, seed=seed + sample_index
        )
        evidence.append(observation)
        result = pipeline.reanalyze_after_sample(
            result, network, evidence, sample_budget_remaining=MAX_SAMPLES - sample_index - 1
        )
        after = _state(
            result, truth, sample_budget_remaining=MAX_SAMPLES - sample_index - 1,
            threshold=pipeline.evidence_threshold,
        )
        states.append(after)
        rounds.append({
            "strategy": name,
            "seed": seed,
            "true_source": truth,
            "round": sample_index + 1,
            "selected_node": node,
            "collection_delay_minutes": delay,
            "expected_information_gain": expected,
            **measurement,
            "posterior_entropy_before": before["entropy"],
            "posterior_entropy_after": after["entropy"],
            "realized_entropy_reduction": before["entropy"] - after["entropy"],
            "true_source_rank_before": before["true_source_rank"],
            "true_source_rank_after": after["true_source_rank"],
            "true_source_probability_before": before["true_source_probability"],
            "true_source_probability_after": after["true_source_probability"],
            "candidate_set_size_before": before["candidate_set_size"],
            "candidate_set_size_after": after["candidate_set_size"],
            "calibrated_before": before["calibrated"], "calibrated_after": after["calibrated"],
            "calibration_source_before": before["calibration_source"], "calibration_source_after": after["calibration_source"],
            "calibration_group_before": before["calibration_group_identifier"], "calibration_group_after": after["calibration_group_identifier"],
            "ood_before": before["ood_level"], "ood_after": after["ood_level"],
            "disagreement_before": before["neural_classical_disagreement"], "disagreement_after": after["neural_classical_disagreement"],
            "learned_evidence_before": before["learned_evidence_sufficiency"], "learned_evidence_after": after["learned_evidence_sufficiency"],
            "deterministic_evidence_before": before["deterministic_evidence_sufficient"], "deterministic_evidence_after": after["deterministic_evidence_sufficient"],
            "planning_allowed_before": before["planning_allowed"], "planning_allowed_after": after["planning_allowed"],
            "control_action_before": before["control_action"], "control_action_after": after["control_action"],
            "suppression_reasons_before": before["suppression_reasons"], "suppression_reasons_after": after["suppression_reasons"],
            "sample_budget_remaining": after["sample_budget_remaining"],
        })
    return {
        "strategy": name, "seed": seed, "truth": truth, "states": states, "rounds": rounds,
        "initial_actionable": states[0]["planning_allowed"],
        "final_top1": states[-1]["top1_correct"], "final_top3": states[-1]["true_source_rank"] <= 3,
        "final_rank": states[-1]["true_source_rank"], "final_entropy": states[-1]["entropy"],
        "actionable_within": 0 if states[0]["planning_allowed"] else next(
            (index for index, state in enumerate(states[1:], start=1) if state["planning_allowed"]), None
        ),
        "sampling_stop_reason": stop_reason,
    }


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    def ranks(values: list[float]) -> list[float]:
        order = sorted(enumerate(values), key=lambda item: item[1])
        out, at = [0.0] * len(values), 0
        while at < len(order):
            end = at
            while end + 1 < len(order) and order[end + 1][1] == order[at][1]:
                end += 1
            for index in range(at, end + 1):
                out[order[index][0]] = (at + end + 2) / 2
            at = end + 1
        return out
    x, y = ranks([item[0] for item in pairs]), ranks([item[1] for item in pairs])
    mx, my = fmean(x), fmean(y)
    denominator = math.sqrt(sum((value - mx) ** 2 for value in x) * sum((value - my) ** 2 for value in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / denominator if denominator else None


def _summary(records: list[dict[str, Any]], rounds: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [item["actionable_within"] for item in records if item["actionable_within"] is not None]
    return {
        "n": len(records),
        "actionable_initial": sum(item["initial_actionable"] for item in records) / len(records),
        **{f"actionable_within_{step}": sum(item["actionable_within"] is not None and item["actionable_within"] <= step for item in records) / len(records) for step in range(1, 4)},
        "median_samples_to_actionability": median(resolved) if resolved else None,
        "never_resolved": sum(item["actionable_within"] is None for item in records) / len(records),
        "final_top1": sum(item["final_top1"] for item in records) / len(records),
        "final_top3": sum(item["final_top3"] for item in records) / len(records),
        "median_rank_improvement": median([item["true_source_rank_before"] - item["true_source_rank_after"] for item in rounds]) if rounds else None,
        "median_candidate_contraction": median([item["candidate_set_size_before"] - item["candidate_set_size_after"] for item in rounds]) if rounds else None,
        "median_realized_entropy_reduction": median([item["realized_entropy_reduction"] for item in rounds]) if rounds else None,
        "sampling_stop_reasons": dict(Counter(item["sampling_stop_reason"] or "sample_budget_exhausted" for item in records)),
    }


def _blockers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_round: dict[str, Counter[str]] = defaultdict(Counter)
    intersections: Counter[str] = Counter()
    first, sole = Counter(), Counter()
    localization_correct_suppressed: Counter[str] = Counter()
    evidence: Counter[str] = Counter()
    for row in rows:
        for round_index, state in enumerate(row["states"]):
            reasons = tuple(state["suppression_reasons"])
            by_round[str(round_index)].update(reasons or ("NONE",))
            if reasons:
                intersections["|".join(sorted(reasons))] += 1
                if round_index == 0:
                    first[reasons[0]] += 1
                if len(reasons) == 1:
                    sole[reasons[0]] += 1
            if state["top1_correct"] and not state["planning_allowed"]:
                localization_correct_suppressed.update(reasons or ("OTHER",))
            learned = state["learned_evidence_sufficient"]
            if learned is not None:
                key = "correct" if state["top1_correct"] else "incorrect"
                evidence[f"{key}_{'sufficient' if learned else 'insufficient'}"] += 1
                if state["top1_correct"] and state["deterministic_evidence_sufficient"] and not learned:
                    evidence["false_insufficient"] += 1
                if not state["top1_correct"] and learned:
                    evidence["false_sufficient"] += 1
    relevant = evidence["correct_sufficient"] + evidence["false_insufficient"]
    sufficient = evidence["correct_sufficient"] + evidence["incorrect_sufficient"]
    return {
        "blocker_frequency_by_round": {key: dict(value) for key, value in sorted(by_round.items())},
        "blocker_intersections": dict(intersections), "first_remaining_blocker": dict(first),
        "sole_remaining_blocker": dict(sole),
        "localization_correct_but_suppressed": dict(localization_correct_suppressed),
        "evidence_sufficiency_confusion": dict(evidence),
        "evidence_sufficiency_precision_correct": evidence["correct_sufficient"] / sufficient if sufficient else None,
        "evidence_sufficiency_recall_correct": evidence["correct_sufficient"] / relevant if relevant else None,
        "evidence_sufficiency_false_insufficient_rate": evidence["false_insufficient"] / relevant if relevant else None,
        "evidence_sufficiency_false_sufficient_rate": evidence["false_sufficient"] / sufficient if sufficient else None,
    }


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"
    code_commit = _commit()
    path = ROOT / "data/frozen/golden_network.inp"
    base, generator, rows = wntr.network.WaterNetworkModel(str(path)), WNTRScenarioGenerator(), []
    factory = V4PipelineFactory(ROOT / "models/hydrocore-v4-release", project_root=ROOT)
    for seed in SEEDS:
        scenario, network = generator.generate_with_network(base, ScenarioGenerationConfig(seed=seed, network_id="golden-reference", network_family="golden-reference", split=DatasetSplit.DEVELOPMENT_HOLDOUT, stage=CurriculumStage.OPERATIONAL, sensor_count=4, sensor_noise_std=NOISE_STD))
        all_series = build_sensor_series(scenario, build_feature_context(network))
        initial = [SensorSeries(node_id=item.node_id, timestamps_seconds=item.timestamps_seconds[:INITIAL_STEPS], concentration_mg_l=item.concentration_mg_l[:INITIAL_STEPS], pressure_m=item.pressure_m[:INITIAL_STEPS], health=item.health[:INITIAL_STEPS], missing=item.missing[:INITIAL_STEPS], drift=item.drift[:INITIAL_STEPS], delayed=item.delayed[:INITIAL_STEPS], frozen=item.frozen[:INITIAL_STEPS]) for item in all_series[:2]]
        truth = scenario.manifest.incident.source_nodes[0]
        for strategy in ("EIG", "RANDOM"):
            rows.append(_run_strategy(strategy, factory(None, path), network, scenario, initial, truth, seed))
    by = {name: [row for row in rows if row["strategy"] == name] for name in ("EIG", "RANDOM")}
    eig_rounds = [item for row in by["EIG"] for item in row["rounds"]]
    random_rounds = [item for row in by["RANDOM"] for item in row["rounds"]]
    expected_realized = [(item["expected_information_gain"], item["realized_entropy_reduction"]) for item in eig_rounds if item["expected_information_gain"] is not None]
    provenance = {
        "code_under_test_commit": code_commit, "model_sha": factory.model_hash,
        "calibration_sha": json.loads((ROOT / "models/hydrocore-v4-release/calibration-status.json").read_text())["calibration_artifact_hash"],
        "feature_schema_sha": factory.identity.feature_schema_hash,
        "normalization_sha": factory.identity.normalization_hash,
        "signature_policy_hash": factory.identity.signature_policy_hash,
        "locked_test_opened": False,
    }
    payload = {
        "schema_version": 2, **provenance, "network": "golden-reference",
        "sensor_coverage": "50% (2 of 4 junctions; not the fully instrumented demo)",
        "initial_causal_steps": INITIAL_STEPS, "sample_budget": MAX_SAMPLES,
        "expected_collection_delay_minutes": 30.0, "measurement_noise_std_mg_l": NOISE_STD,
        "paired_design": "same incident, source, causal evidence, physics, acquisition delay, noise seed, and budget",
        "strategies": {name: _summary(by[name], eig_rounds if name == "EIG" else random_rounds) for name in by},
        "expected_vs_realized": {
            "spearman": _spearman(expected_realized),
            "median_realized_entropy_reduction": median([item["realized_entropy_reduction"] for item in eig_rounds]) if eig_rounds else None,
            "p25_realized_entropy_reduction": float(np.percentile([item["realized_entropy_reduction"] for item in eig_rounds], 25)) if eig_rounds else None,
            "p75_realized_entropy_reduction": float(np.percentile([item["realized_entropy_reduction"] for item in eig_rounds], 75)) if eig_rounds else None,
            "fraction_positive": sum(item["realized_entropy_reduction"] > 0 for item in eig_rounds) / len(eig_rounds) if eig_rounds else None,
            "fraction_rank_improving": sum(item["true_source_rank_after"] < item["true_source_rank_before"] for item in eig_rounds) / len(eig_rounds) if eig_rounds else None,
            "fraction_rank_worsening": sum(item["true_source_rank_after"] > item["true_source_rank_before"] for item in eig_rounds) / len(eig_rounds) if eig_rounds else None,
        },
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sampling.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "sampling-blockers.json").write_text(json.dumps({"schema_version": 1, **provenance, **_blockers(rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
