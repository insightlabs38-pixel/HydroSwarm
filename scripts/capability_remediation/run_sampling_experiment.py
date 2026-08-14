"""Matched causal EIG versus random valid-unsampled sampling experiment."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import uuid
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import wntr  # noqa: E402

from hydroswarm.data.scenarios import CurriculumStage, DatasetSplit, ScenarioGenerationConfig, WNTRScenarioGenerator  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.preprocessing.builder import SensorSeries  # noqa: E402
from hydroswarm.runtime import V4PipelineFactory  # noqa: E402
from hydroswarm.simulation import HydraulicSimulator  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402


SEEDS = tuple(range(2026081601, 2026081621))
NOISE_STD = 0.05
INITIAL_STEPS = 3
MAX_SAMPLES = 3
OUT = ROOT / "reports/evaluation/capability-remediation/sampling.json"


def _entropy(result: Any) -> float:
    return result.posterior_history[-1].entropy_bits


def _rank(result: Any, truth: str) -> int:
    nodes = sorted(result.fused_belief, key=lambda node: (-result.fused_belief[node], node))
    return nodes.index(truth) + 1


def _append(series: list[SensorSeries], observation: SensorSeries) -> list[SensorSeries]:
    return [*series, observation]


def _measurement(network: Any, scenario: Any, node: str, *, decision_seconds: float, delay_minutes: float, seed: int) -> SensorSeries:
    exact = HydraulicSimulator(network).simulate_incident(
        scenario.manifest.incident.source_nodes[0],
        strength_mg_min=10.0 * scenario.manifest.incident.relative_strength,
        start_minute=scenario.manifest.incident.start_minute,
        duration_minutes=scenario.manifest.incident.duration_minutes,
    )
    acquisition_seconds = decision_seconds + delay_minutes * 60.0
    index = int(np.argmin(np.abs(np.asarray(exact.concentration_mg_l.index, dtype=float) - acquisition_seconds)))
    timestamp = float(exact.concentration_mg_l.index[index])
    value = max(0.0, float(exact.concentration_mg_l.loc[:, node].iloc[index]) + np.random.default_rng(seed).normal(0.0, NOISE_STD))
    return SensorSeries(node_id=node, timestamps_seconds=(timestamp,), concentration_mg_l=(value,), pressure_m=(25.0,), health=(1.0,), missing=(False,), drift=(False,), delayed=(False,), frozen=(False,))


def _run_strategy(name: str, pipeline: Any, network: Any, scenario: Any, initial: list[SensorSeries], truth: str, seed: int) -> dict[str, Any]:
    evidence = list(initial)
    result = pipeline.analyze(uuid.uuid4(), network, evidence, sample_budget_remaining=MAX_SAMPLES)
    rounds: list[dict[str, Any]] = []
    initial_actionable = result.planning_allowed
    for sample_index in range(MAX_SAMPLES):
        if result.planning_allowed:
            break
        candidates = [node for node in network.junction_name_list if node not in {item.node_id for item in evidence}]
        if not candidates:
            break
        if name == "EIG":
            selected = result.sample_result.recommended_node if result.sample_result and not result.sample_result.stop else None
            candidate = next((item for item in (result.sample_result.ranked if result.sample_result else ()) if item.node_id == selected), None)
            if candidate is None:
                break
            node, expected, delay = candidate.node_id, candidate.expected_information_gain_bits, candidate.collection_time_minutes
        else:
            node = sorted(candidates)[int(hashlib.sha256(f"{seed}:{sample_index}".encode()).hexdigest(), 16) % len(candidates)]
            expected, delay = None, 30.0
        before_entropy, before_rank = _entropy(result), _rank(result, truth)
        decision = max(timestamp for item in evidence for timestamp in item.timestamps_seconds)
        evidence = _append(evidence, _measurement(network, scenario, node, decision_seconds=decision, delay_minutes=delay, seed=seed + sample_index))
        result = pipeline.reanalyze_after_sample(result, network, evidence, sample_budget_remaining=MAX_SAMPLES - sample_index - 1)
        rounds.append({
            "round": sample_index + 1, "node": node, "expected_information_gain": expected,
            "collection_delay_minutes": delay, "realized_entropy_change": _entropy(result) - before_entropy,
            "true_source_rank_before": before_rank, "true_source_rank_after": _rank(result, truth),
            "planning_allowed_after": result.planning_allowed,
        })
    return {
        "strategy": name, "initial_actionable": initial_actionable,
        "final_top1": _rank(result, truth) == 1, "final_top3": _rank(result, truth) <= 3,
        "final_rank": _rank(result, truth), "final_entropy": _entropy(result),
        "actionable_within": 0 if initial_actionable else next((item["round"] for item in rounds if item["planning_allowed_after"]), None),
        "rounds": rounds,
    }


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    def ranks(values: list[float]) -> list[float]:
        ordered = sorted(enumerate(values), key=lambda item: item[1])
        out = [0.0] * len(values)
        at = 0
        while at < len(ordered):
            end = at
            while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[at][1]:
                end += 1
            rank = (at + end + 2) / 2
            for index in range(at, end + 1):
                out[ordered[index][0]] = rank
            at = end + 1
        return out
    x, y = ranks([pair[0] for pair in pairs]), ranks([pair[1] for pair in pairs])
    mx, my = sum(x) / len(x), sum(y) / len(y)
    denominator = math.sqrt(sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / denominator if denominator else None


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"
    path = ROOT / "data/frozen/golden_network.inp"
    base = wntr.network.WaterNetworkModel(str(path))
    generator, rows = WNTRScenarioGenerator(), []
    for seed in SEEDS:
        scenario, network = generator.generate_with_network(base, ScenarioGenerationConfig(seed=seed, network_id="golden-reference", network_family="golden-reference", split=DatasetSplit.DEVELOPMENT_HOLDOUT, stage=CurriculumStage.OPERATIONAL, sensor_count=4, sensor_noise_std=NOISE_STD))
        all_series = build_sensor_series(scenario, build_feature_context(network))
        initial = [SensorSeries(node_id=item.node_id, timestamps_seconds=item.timestamps_seconds[:INITIAL_STEPS], concentration_mg_l=item.concentration_mg_l[:INITIAL_STEPS], pressure_m=item.pressure_m[:INITIAL_STEPS], health=item.health[:INITIAL_STEPS], missing=item.missing[:INITIAL_STEPS], drift=item.drift[:INITIAL_STEPS], delayed=item.delayed[:INITIAL_STEPS], frozen=item.frozen[:INITIAL_STEPS]) for item in all_series[:2]]
        truth = scenario.manifest.incident.source_nodes[0]
        for strategy in ("EIG", "RANDOM"):
            pipeline = V4PipelineFactory(ROOT / "models/hydrocore-v4-release", project_root=ROOT)(None, path)
            rows.append({"seed": seed, "truth": truth, **_run_strategy(strategy, pipeline, network, scenario, initial, truth, seed)})
    by = {name: [row for row in rows if row["strategy"] == name] for name in ("EIG", "RANDOM")}
    eig_rounds = [round_ for row in by["EIG"] for round_ in row["rounds"]]
    random_rounds = [round_ for row in by["RANDOM"] for round_ in row["rounds"]]
    def summary(records: list[dict[str, Any]], rounds: list[dict[str, Any]]) -> dict[str, Any]:
        changes = [-item["realized_entropy_change"] for item in rounds]
        return {
            "n": len(records), "top1_after_n": sum(row["final_top1"] for row in records) / len(records),
            "top3_after_n": sum(row["final_top3"] for row in records) / len(records),
            "actionable_within_3": sum(row["actionable_within"] is not None and row["actionable_within"] <= 3 for row in records) / len(records),
            "never_resolved": sum(row["actionable_within"] is None for row in records) / len(records),
            "median_realized_entropy_reduction": median(changes) if changes else None,
        }
    expected_realized = [(item["expected_information_gain"], -item["realized_entropy_change"]) for item in eig_rounds if item["expected_information_gain"] is not None]
    payload = {"schema_version": 1, "network": "golden-reference", "sensor_coverage": "50% (2 of 4 junctions; deliberately not the fully instrumented demo)", "initial_causal_steps": INITIAL_STEPS, "expected_collection_delay_minutes": 30.0, "measurement_noise_std_mg_l": NOISE_STD, "locked_test_opened_before": False, "locked_test_opened_after": locked_test_opened(ROOT), "paired_design": "same incident, source, initial evidence, physics, noise seed, and three-sample budget", "strategies": {name: summary(by[name], eig_rounds if name == "EIG" else random_rounds) for name in by}, "expected_vs_realized": {"spearman": _spearman(expected_realized), "median_realized_entropy_reduction": median([-item["realized_entropy_change"] for item in eig_rounds]) if eig_rounds else None, "fraction_positive": sum(item["realized_entropy_change"] < 0 for item in eig_rounds) / len(eig_rounds) if eig_rounds else None, "fraction_strongly_negative": sum(item["realized_entropy_change"] > 0.25 for item in eig_rounds) / len(eig_rounds) if eig_rounds else None}, "rows": rows}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
