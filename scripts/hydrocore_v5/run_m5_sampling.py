"""Milestone 5 (experiments.txt): decision-aware active sampling.

Core finding to address (M0 baseline / CAP-REM-02, reports/evaluation/
hydrocore-v5/m0-baseline.json's `current_sampling_baseline`): current EIG
reduces uncertainty substantially but did NOT beat random on
samples-to-actionability (actionable_within_three_samples: eig=0.375 vs
random_valid_unsampled=0.45). Entropy reduction is therefore not the
correct sole sampling objective.

5.1 Frozen predictor + calibration: the Milestone-1 winning predictor (arm
A) and the Milestone-3 frozen B_DEPTH_AWARE calibration (alpha=0.1),
refit identically to M3/M4 -- no predictor or calibration tuning happens
in this script.

Architecture note (why this script does NOT route sampling decisions
through `hydroswarm.inference.pipeline.HybridInferencePipeline`, unlike
`scripts/capability_remediation/run_sampling_experiment.py`'s v4
methodology): `HybridInferencePipeline.analyze()`'s own calibration gate
calls `calibrator.selection(network_id=governed_network_family(topology_hash),
condition=...)` -- `governed_network_family` returns a bare topology-family
string with no depth-bucket hook, so it cannot address M3's
B_DEPTH_AWARE Mondrian keys (`f"{network_id}:{depth_bucket}"`). Routing
through the full pipeline would therefore silently fall back to
global/condition-only calibration, abandoning M3's frozen policy. Instead
this script uses the same low-level primitives Milestones 1-4 already use
(direct `HydroCore` forward pass, `SplitConformalCalibrator.candidate_set`
called directly with the real depth-bucket key, `HydraulicFeatureBuilder`
called directly), extended to support genuinely NEW sensor nodes at
arbitrary acquisition times -- confirmed by inspection that `build()`
already unions per-series timestamps and tolerates per-node asynchronous
coverage (`hydroswarm/preprocessing/builder.py`'s `selected_times`/
`by_time` construction), so no new feature-alignment mechanism had to be
invented for active sampling.

`hydroswarm.training.corpus.model_input_classical_prior` (the documented
"live-serving equivalent" of the training-time classical prior) is reused
for classical_prior at every round, aligned onto the corpus's own 25-step
hourly report grid (`TARGET_TIMESTAMPS`, read from a real scenario, not
hardcoded).

5.2 Matched policies (all four use posterior/candidate-derived signals
only -- the true source is never used at decision time):
  RANDOM_VALID_UNSAMPLED  -- uniform over unsampled, currently-accessible
    junctions, seeded deterministically per (incident, round).
  CURRENT_EIG  -- `hydroswarm.sampling.active.rank_sample_locations`'s own
    default blended score (unchanged production code).
  CANDIDATE_CONTRACTION  -- re-ranks the SAME candidate list by
    `expected_candidate_reduction` alone (still an existing, already-
    computed `SampleCandidate` field; no new signature/physics code).
  DECISION_GAIN  -- re-ranks the same candidate list by a predeclared
    composite (5.3) that targets actionability instead of raw entropy:

      score = 0.25 * expected_information_gain_bits
            + 1.00 * expected_candidate_reduction
            + 0.75 * leading_hypothesis_separation
            + 0.50 * detection_probability
            - 0.25 * redundancy
            - 0.01 * collection_time_minutes
            - 0.05 * operational_cost

    (CURRENT_EIG's own blend weights expected_information_gain_bits at
    1.00 and collection_time_minutes at -0.002; DECISION_GAIN inverts the
    entropy/contraction priority and is ~5x more delay-averse, since
    actionability is reached by shrinking the CANDIDATE SET below the
    planning gate, not by lowering entropy per se -- directly targeting
    the M0/CAP-REM-02 finding.) Ties broken by node_id for determinism.

5.4 Experimental design: paired incidents share source, network, initial
causal evidence, sensor coverage, sample budget, and (per round) the exact
measurement-noise draw across all four arms, matching the golden-reference
network's own 4-junction scope (`_scenario_config`-style controlled
generation: all non-source/non-sensor-placement randomization disabled --
single start/duration/strength/demand-regime bin, zero sensor noise/fault/
outage/drift probabilities -- so `network_sha256` is provably identical to
the pristine base network for every generated incident; verified by
inspection this makes governed-family/topology identity irrelevant here
since this script never calls governed_network_family at all).

Stratified 3x3 (sensor coverage 25%/50%/75% = 1/2/3 of the network's 4
junctions initially sampled x initial causal evidence 2/3/6 reports,
matching Milestone-3's EARLY/EARLY/MID depth buckets) x N_PER_CELL=12
incidents/cell = 108 paired incidents (>= the 100-preferred target).
Sample budget = 3 per arm; a coverage stratum with few remaining
unsampled junctions (75% coverage leaves only 1) naturally caps how far
that stratum's curve can run -- reported honestly, not padded.

5.5 Primary metrics: actionable within <=1/2/3 samples, median samples to
actionability, never-actionable fraction. Secondary: final top-1/top-3,
candidate-set contraction, entropy reduction, source-rank improvement.
"Actionable" mirrors Milestone 4's own simplification (conformal
candidate-set size in [1, K=3], same K as
`hydroswarm.simulation.wrapper.MAXIMUM_EVALUATION_HYPOTHESES` /
production's `maximum_planning_candidates` default) rather than
reconstructing the full pipeline's OOD/disagreement/frozen-sensor gate
matrix, which this experiment does not vary.

Statistical comparison: paired bootstrap (5000 resamples, seeded) over
the 108 matched incidents for each non-random arm vs RANDOM_VALID_UNSAMPLED
on actionable_within_3 and (among incidents both arms resolved) samples-
to-resolution. Promotion requires the predeclared 5.5/experiments.txt bar:
>=10pp actionable<=3 improvement with a 95% CI excluding zero, or a
clearly (CI-excluding-zero) lower paired samples-to-resolution
distribution -- never a bare point estimate. If no arm clears this bar:
ACTIVE_SAMPLING_REMAINS_ADVISORY (matching the M0/CAP-REM-02 baseline
finding's own language) -- not overclaimed.

Uses development_holdout only (never locked_final_test /
locked_topology_test). No predictor, calibration, K, or verification
threshold is changed from Milestones 3/4.

Writes:
  reports/evaluation/hydrocore-v5/m5-sampling.json
  reports/evaluation/hydrocore-v5/m5-summary.md
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.calibration.conformal import (  # noqa: E402
    CalibrationExample,
    SplitConformalCalibrator,
    classify_runtime_condition,
)
from hydroswarm.classical.signatures import (  # noqa: E402
    SignatureBuilder,
    SignatureCache,
    SignatureCacheKey,
)
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.preprocessing import HydraulicFeatureBuilder, SensorSeries  # noqa: E402
from hydroswarm.sampling.active import SamplingConstraints, rank_sample_locations  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.corpus import build_feature_context, build_sensor_series, model_input_classical_prior  # noqa: E402
from run_m1_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m3_calibration import DEPTH_BUCKET_OF, _freeze_predictor  # noqa: E402

OUTPUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m5-sampling.json"
OUTPUT_SUMMARY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m5-summary.md"

ALPHA = 0.1
K_MAX_CANDIDATES = 3  # matches Milestone 4's K; never relaxed.
MAX_SAMPLES = 3
NOISE_STD_MG_L = 0.05
COVERAGE_STRATA: dict[str, int] = {"25%": 1, "50%": 2, "75%": 3}  # of 4 junctions
DEPTH_STRATA: tuple[int, ...] = (2, 3, 6)
N_PER_CELL = 12  # 3 coverage x 3 depth x 12 = 108 paired incidents (>=100 preferred)
ARMS: tuple[str, ...] = ("RANDOM_VALID_UNSAMPLED", "CURRENT_EIG", "CANDIDATE_CONTRACTION", "DECISION_GAIN")
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260814

#: Predeclared decision-gain weights (5.3); see module docstring for
#: rationale. Development-data-only signals; never includes the true
#: source at decision time.
DECISION_GAIN_WEIGHTS = {
    "expected_information_gain_bits": 0.25,
    "expected_candidate_reduction": 1.00,
    "leading_hypothesis_separation": 0.75,
    "detection_probability": 0.50,
    "redundancy": -0.25,
    "collection_time_minutes": -0.01,
    "operational_cost": -0.05,
}


def _decision_gain_score(candidate) -> float:
    return sum(getattr(candidate, field) * weight for field, weight in DECISION_GAIN_WEIGHTS.items())


def _seeded_rng(*parts: object) -> np.random.Generator:
    material = ":".join(str(part) for part in parts)
    seed_int = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    return np.random.default_rng(seed_int)


def main() -> int:  # noqa: C901
    assert not locked_test_opened(ROOT), "locked test must remain closed"

    export_path, use_adapters, predictor_description = _freeze_predictor()
    model = HydroCore.from_variant("small", use_adapters=use_adapters, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()

    calibration_records = build_scenario_pool("calibration", network_loader=build_wntr_network)
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)
    target_timestamps = build_sensor_series(train_records[0].scenario, train_records[0].feature_context)[0].timestamps_seconds

    def _collect(records, bucket: str, depth: int) -> list[dict[str, Any]]:
        collected = []
        with torch.no_grad():
            for record in records:
                scenario = record.scenario
                example = scenario_to_prefix_example(scenario, record.network, library, depth, feature_context=record.feature_context)
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
                truth = int(example.targets["source_node"].item())
                full_series = build_sensor_series(scenario, record.feature_context)
                truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
                condition = classify_runtime_condition(truncated_series)
                collected.append({
                    "probabilities": probs, "true_index": truth, "condition": condition,
                    "network_id": scenario.manifest.network_id, "depth": depth, "depth_bucket": bucket,
                })
        return collected

    full_calibration_examples: list[dict[str, Any]] = []
    with torch.no_grad():
        for depth in CAUSAL_PREFIX_DEPTHS:
            full_calibration_examples.extend(_collect(calibration_records, DEPTH_BUCKET_OF[depth], depth))
    calibrator = SplitConformalCalibrator.fit(
        [
            CalibrationExample(
                probabilities=tuple(item["probabilities"]), true_index=item["true_index"], condition=item["condition"],
                network_id=f"{item['network_id']}:{item['depth_bucket']}",
            )
            for item in full_calibration_examples
        ],
        alpha=ALPHA, model_hash=export_path, feature_schema_hash="n/a", dataset_manifest_hash="m5-sampling-pool",
    )

    base_network = build_wntr_network()
    junctions = tuple(sorted(base_network.junction_name_list))
    base_simulator = HydraulicSimulator(base_network)
    with tempfile.TemporaryDirectory() as cache_dir:
        signature_cache = SignatureCache(cache_dir)
        signature_key = SignatureCacheKey(
            network_hash=base_simulator.state_hash(), hydraulic_state_hash="m5-fixed-profile",
            simulator_version=base_simulator.simulator_version, configuration_hash="m5-source-location-only-v1",
            sensor_layout_hash="all-junctions",
        )
        signature_artifact = SignatureBuilder(base_simulator, signature_cache).build_or_load(
            key=signature_key, source_nodes=junctions, start_time_bins=(0,), duration_bins=(60,),
            strength_bins=(1.0,), demand_regimes=("baseline",), sensor_nodes=junctions,
            sample_times_seconds=list(target_timestamps),
        )

    generator = WNTRScenarioGenerator()
    incidents: list[dict[str, Any]] = []
    for coverage_label, sensor_count in COVERAGE_STRATA.items():
        for depth in DEPTH_STRATA:
            for index in range(N_PER_CELL):
                #: Python's built-in hash() is per-process randomized for str/tuple
                #: (PYTHONHASHSEED salting) unless explicitly disabled -- using it
                #: here would silently break run-to-run reproducibility of which
                #: scenarios get generated. hashlib is deterministic across
                #: processes/runs, matching every other seed derivation in this
                #: v5 script family (_seeded_rng below, run_m1_arm, etc.).
                seed_material = f"m5-sampling:{coverage_label}:{depth}:{index}"
                seed = 950_000_000 + int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "big") % 10_000_000
                network = build_wntr_network()
                config = ScenarioGenerationConfig(
                    seed=seed, network_id="golden-reference", network_family="golden-reference",
                    split=DatasetSplit.DEVELOPMENT_HOLDOUT, stage=CurriculumStage.OPERATIONAL,
                    event_type=EventType.CONTAMINATION, sensor_count=sensor_count,
                    start_time_bins_min=(0,), duration_bins_min=(60,), strength_bins=(1.0,),
                    demand_regimes=(1.0,), sensor_noise_std=0.0, missing_probability=0.0,
                    drift_per_hour=0.0, frozen_probability=0.0, communication_outage_probability=0.0,
                    unit_mismatch_probability=0.0, roughness_variation_fraction=0.0,
                    tank_level_variation_fraction=0.0, pipe_outage_probability=0.0,
                )
                scenario, randomized_network = generator.generate_with_network(network, config)
                feature_context = build_feature_context(randomized_network)
                full_series = build_sensor_series(scenario, feature_context)
                initial_series = [truncate_causal_prefix(item, depth) for item in full_series]
                incidents.append({
                    "coverage_label": coverage_label, "sensor_count": sensor_count, "depth": depth,
                    "seed": seed, "scenario": scenario, "network": randomized_network,
                    "feature_context": feature_context, "initial_series": initial_series,
                    "truth": scenario.manifest.incident.source_nodes[0],
                })

    node_ids_ref: list[str] = []

    def _posterior_and_candidates(network, feature_context, series, bucket: str) -> tuple[dict[str, float], tuple[str, ...], list[str]]:
        classical_prior = model_input_classical_prior(library, list(library.node_ids), series, target_timestamps)
        built = HydraulicFeatureBuilder().build(
            network, feature_context.graph, feature_context.state, series,
            classical_prior=classical_prior, window_steps=25,
        )
        with torch.no_grad():
            output = model(built.batch)
        probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
        node_ids = list(built.node_ids)
        if not node_ids_ref:
            node_ids_ref.extend(node_ids)
        condition = classify_runtime_condition(series)
        indices = calibrator.candidate_set(probs, condition=condition, network_id=f"golden-reference:{bucket}")
        candidate_nodes = [node_ids[index] for index in indices]
        fused_belief = dict(zip(node_ids, probs, strict=True))
        return fused_belief, tuple(candidate_nodes), node_ids

    def _entropy_bits(belief: dict[str, float]) -> float:
        values = np.asarray([max(v, 1e-12) for v in belief.values()], dtype=float)
        values = values / values.sum()
        return float(-(values * np.log2(values)).sum())

    def _rank(belief: dict[str, float], truth: str) -> int:
        ordered = sorted(belief, key=lambda node: (-belief[node], node))
        return ordered.index(truth) + 1 if truth in ordered else len(ordered)

    def _measurement(network, scenario, node: str, *, decision_seconds: float, delay_minutes: float, rng: np.random.Generator) -> tuple[SensorSeries, float]:
        incident = scenario.manifest.incident
        exact = HydraulicSimulator(network).simulate_incident(
            incident.source_nodes[0], strength_mg_min=10.0 * incident.relative_strength,
            start_minute=incident.start_minute, duration_minutes=incident.duration_minutes,
        )
        acquisition_seconds = decision_seconds + delay_minutes * 60.0
        index = int(np.argmin(np.abs(np.asarray(exact.concentration_mg_l.index, dtype=float) - acquisition_seconds)))
        timestamp = float(exact.concentration_mg_l.index[index])
        truth_value = float(exact.concentration_mg_l.loc[:, node].iloc[index])
        noise = float(rng.normal(0.0, NOISE_STD_MG_L))
        observed = max(0.0, truth_value + noise)
        return SensorSeries(
            node_id=node, timestamps_seconds=(timestamp,), concentration_mg_l=(observed,),
            pressure_m=(25.0,), health=(1.0,), missing=(False,), drift=(False,),
            delayed=(False,), frozen=(False,),
        ), timestamp

    def _pick_node(arm: str, ranked_candidates, unsampled: list[str], *, incident_key: tuple, round_index: int) -> str | None:
        accessible = [candidate for candidate in ranked_candidates if candidate.accessible]
        if arm == "RANDOM_VALID_UNSAMPLED":
            if not unsampled:
                return None
            rng = _seeded_rng("m5-random", *incident_key, round_index)
            return sorted(unsampled)[int(rng.integers(0, len(unsampled)))]
        if not accessible:
            return None
        if arm == "CURRENT_EIG":
            return max(accessible, key=lambda c: (c.score, c.node_id)).node_id
        if arm == "CANDIDATE_CONTRACTION":
            return max(accessible, key=lambda c: (c.expected_candidate_reduction, c.node_id)).node_id
        if arm == "DECISION_GAIN":
            return max(accessible, key=lambda c: (_decision_gain_score(c), c.node_id)).node_id
        raise ValueError(f"unknown arm: {arm}")

    rows: list[dict[str, Any]] = []
    for incident in incidents:
        bucket = DEPTH_BUCKET_OF[incident["depth"]]
        truth = incident["truth"]
        incident_key = (incident["coverage_label"], incident["depth"], incident["seed"])
        for arm in ARMS:
            series = list(incident["initial_series"])
            belief, candidates, node_ids = _posterior_and_candidates(
                incident["network"], incident["feature_context"], series, bucket
            )
            states = [{
                "candidate_set_size": len(candidates), "entropy_bits": _entropy_bits(belief),
                "true_source_rank": _rank(belief, truth), "top1_correct": _rank(belief, truth) == 1,
                "actionable": 1 <= len(candidates) <= K_MAX_CANDIDATES,
            }]
            rounds: list[dict[str, Any]] = []
            sampled_nodes = {item.node_id for item in series}
            for round_index in range(MAX_SAMPLES):
                if states[-1]["actionable"]:
                    break
                unsampled = [node for node in junctions if node not in sampled_nodes]
                if not unsampled:
                    break
                hypothesis_weights = {h.identifier: belief.get(h.source_node, 0.0) for h in signature_artifact.hypotheses}
                decision_seconds = max(timestamp for item in series for timestamp in item.timestamps_seconds)
                sampling_result = rank_sample_locations(
                    signature_artifact, hypothesis_weights,
                    constraints=SamplingConstraints(already_sampled=frozenset(sampled_nodes)),
                    noise_scale_mg_l=NOISE_STD_MG_L, target_sample_time_seconds=decision_seconds, top_k=20,
                )
                node = _pick_node(arm, sampling_result.ranked, unsampled, incident_key=incident_key, round_index=round_index)
                if node is None:
                    break
                rng = _seeded_rng("m5-noise", *incident_key, round_index)
                candidate_meta = next((c for c in sampling_result.ranked if c.node_id == node), None)
                delay = candidate_meta.collection_time_minutes if candidate_meta else 30.0
                observation, acquisition_ts = _measurement(
                    incident["network"], incident["scenario"], node,
                    decision_seconds=decision_seconds, delay_minutes=delay, rng=rng,
                )
                series.append(observation)
                sampled_nodes.add(node)
                before = states[-1]
                belief, candidates, node_ids = _posterior_and_candidates(
                    incident["network"], incident["feature_context"], series, bucket
                )
                after = {
                    "candidate_set_size": len(candidates), "entropy_bits": _entropy_bits(belief),
                    "true_source_rank": _rank(belief, truth), "top1_correct": _rank(belief, truth) == 1,
                    "actionable": 1 <= len(candidates) <= K_MAX_CANDIDATES,
                }
                states.append(after)
                rounds.append({
                    "round": round_index + 1, "selected_node": node,
                    "collection_delay_minutes": delay,
                    "candidate_set_size_before": before["candidate_set_size"], "candidate_set_size_after": after["candidate_set_size"],
                    "entropy_before": before["entropy_bits"], "entropy_after": after["entropy_bits"],
                    "true_source_rank_before": before["true_source_rank"], "true_source_rank_after": after["true_source_rank"],
                })
            actionable_within = 0 if states[0]["actionable"] else next(
                (index for index, state in enumerate(states[1:], start=1) if state["actionable"]), None
            )
            rows.append({
                "arm": arm, "coverage_label": incident["coverage_label"], "sensor_count": incident["sensor_count"],
                "depth": incident["depth"], "depth_bucket": bucket, "seed": incident["seed"], "truth": truth,
                "initial_actionable": states[0]["actionable"], "actionable_within": actionable_within,
                "unsampled_available": len(junctions) - incident["sensor_count"],
                "final_top1": states[-1]["top1_correct"], "final_top3": states[-1]["true_source_rank"] <= 3,
                "final_rank": states[-1]["true_source_rank"],
                "entropy_reduction": states[0]["entropy_bits"] - states[-1]["entropy_bits"],
                "candidate_contraction": states[0]["candidate_set_size"] - states[-1]["candidate_set_size"],
                "samples_used": len(rounds), "states": states, "rounds": rounds,
            })

    def _summary(arm_rows: list[dict[str, Any]]) -> dict[str, Any]:
        resolved = [row["actionable_within"] for row in arm_rows if row["actionable_within"] is not None]
        n = len(arm_rows)
        return {
            "n": n,
            "actionable_initial": sum(row["initial_actionable"] for row in arm_rows) / n,
            **{
                f"actionable_within_{step}": sum(
                    row["actionable_within"] is not None and row["actionable_within"] <= step for row in arm_rows
                ) / n
                for step in range(1, MAX_SAMPLES + 1)
            },
            "median_samples_to_actionability": statistics.median(resolved) if resolved else None,
            "never_actionable_fraction": sum(row["actionable_within"] is None for row in arm_rows) / n,
            "final_top1": sum(row["final_top1"] for row in arm_rows) / n,
            "final_top3": sum(row["final_top3"] for row in arm_rows) / n,
            "mean_candidate_contraction": statistics.fmean(row["candidate_contraction"] for row in arm_rows),
            "mean_entropy_reduction": statistics.fmean(row["entropy_reduction"] for row in arm_rows),
            "mean_samples_used": statistics.fmean(row["samples_used"] for row in arm_rows),
        }

    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    arm_summaries = {arm: _summary(arm_rows) for arm, arm_rows in by_arm.items()}
    by_arm_and_cell: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {} for arm in ARMS}
    for row in rows:
        cell = f"{row['coverage_label']}|depth{row['depth']}"
        by_arm_and_cell[row["arm"]].setdefault(cell, []).append(row)
    per_cell_summary = {
        arm: {cell: _summary(cell_rows) for cell, cell_rows in cells.items()}
        for arm, cells in by_arm_and_cell.items()
    }

    def _paired_bootstrap_actionable3(arm: str) -> dict[str, Any]:
        random_rows = {(row["coverage_label"], row["depth"], row["seed"]): row for row in by_arm["RANDOM_VALID_UNSAMPLED"]}
        arm_rows = {(row["coverage_label"], row["depth"], row["seed"]): row for row in by_arm[arm]}
        keys = sorted(set(random_rows) & set(arm_rows))
        arm_ind = np.asarray([
            1.0 if arm_rows[k]["actionable_within"] is not None and arm_rows[k]["actionable_within"] <= 3 else 0.0
            for k in keys
        ])
        random_ind = np.asarray([
            1.0 if random_rows[k]["actionable_within"] is not None and random_rows[k]["actionable_within"] <= 3 else 0.0
            for k in keys
        ])
        point_estimate = float(arm_ind.mean() - random_ind.mean())
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = len(keys)
        diffs = np.empty(BOOTSTRAP_RESAMPLES)
        for i in range(BOOTSTRAP_RESAMPLES):
            resample = rng.integers(0, n, size=n)
            diffs[i] = arm_ind[resample].mean() - random_ind[resample].mean()
        ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        return {
            "n_paired": n, "point_estimate_pp": point_estimate * 100,
            "ci95_low_pp": ci_low * 100, "ci95_high_pp": ci_high * 100,
            "ci_excludes_zero": ci_low > 0 or ci_high < 0,
            "meets_10pp_bar": point_estimate * 100 >= 10.0 and ci_low > 0,
        }

    def _paired_bootstrap_samples_to_resolution(arm: str) -> dict[str, Any]:
        random_rows = {(row["coverage_label"], row["depth"], row["seed"]): row for row in by_arm["RANDOM_VALID_UNSAMPLED"]}
        arm_rows = {(row["coverage_label"], row["depth"], row["seed"]): row for row in by_arm[arm]}
        keys = sorted(
            k for k in set(random_rows) & set(arm_rows)
            if random_rows[k]["actionable_within"] is not None and arm_rows[k]["actionable_within"] is not None
        )
        if not keys:
            return {"n_paired_resolved": 0, "point_estimate": None, "ci95_low": None, "ci95_high": None, "ci_excludes_zero": False}
        arm_vals = np.asarray([arm_rows[k]["actionable_within"] for k in keys], dtype=float)
        random_vals = np.asarray([random_rows[k]["actionable_within"] for k in keys], dtype=float)
        point_estimate = float(np.median(arm_vals) - np.median(random_vals))
        rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
        n = len(keys)
        diffs = np.empty(BOOTSTRAP_RESAMPLES)
        for i in range(BOOTSTRAP_RESAMPLES):
            resample = rng.integers(0, n, size=n)
            diffs[i] = np.median(arm_vals[resample]) - np.median(random_vals[resample])
        ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        return {
            "n_paired_resolved": n, "point_estimate": point_estimate,
            "ci95_low": ci_low, "ci95_high": ci_high, "ci_excludes_zero": ci_low > 0 or ci_high < 0,
            "clearly_lower": ci_high < 0,
        }

    comparisons = {
        arm: {
            "actionable_within_3_vs_random": _paired_bootstrap_actionable3(arm),
            "samples_to_resolution_vs_random": _paired_bootstrap_samples_to_resolution(arm),
        }
        for arm in ARMS if arm != "RANDOM_VALID_UNSAMPLED"
    }
    promoted_arms = [
        arm for arm, comparison in comparisons.items()
        if comparison["actionable_within_3_vs_random"]["meets_10pp_bar"]
        or comparison["samples_to_resolution_vs_random"]["clearly_lower"]
    ]
    exit_decision = f"PROMOTE_{promoted_arms[0]}" if promoted_arms else "ACTIVE_SAMPLING_REMAINS_ADVISORY"

    initial_actionable_rate = arm_summaries["RANDOM_VALID_UNSAMPLED"]["actionable_initial"]
    limitation_note = (
        f"Ceiling-effect caveat: {initial_actionable_rate:.1%} of incidents were ALREADY actionable "
        f"(candidate-set size in [1, {K_MAX_CANDIDATES}]) before any sample was taken, on the golden-reference "
        "network's 4-junction action space -- K=3 already covers most of that space, so there is limited room "
        "for any sampling policy (including CURRENT_EIG, the production baseline) to demonstrate a large effect "
        "on this specific network. The comparison above is real and not invalidated by this, but a null/near-null "
        "result here should not be read as ruling out a larger effect on a bigger network with more source "
        "candidates; it was not tested (out of scope for this milestone -- topology diversity is Milestone 7)."
    )

    report = {
        "schema_version": 1,
        "purpose": "Milestone 5 (experiments.txt): decision-aware active sampling.",
        "predictor": {"export_path": export_path, "use_adapters": use_adapters, "description": predictor_description},
        "calibration_scheme": "B_DEPTH_AWARE (frozen in Milestone 3, refit here identically over all CAUSAL_PREFIX_DEPTHS)",
        "alpha": ALPHA,
        "k_max_candidates": K_MAX_CANDIDATES,
        "max_samples": MAX_SAMPLES,
        "coverage_strata": COVERAGE_STRATA,
        "depth_strata": DEPTH_STRATA,
        "n_per_cell": N_PER_CELL,
        "n_paired_incidents": len(incidents),
        "arms": ARMS,
        "decision_gain_weights": DECISION_GAIN_WEIGHTS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "arm_summaries": arm_summaries,
        "per_cell_summary": per_cell_summary,
        "comparisons_vs_random": comparisons,
        "exit_decision": exit_decision,
        "limitation_note": limitation_note,
        "rows": rows,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Milestone 5 summary: decision-aware active sampling",
        "",
        f"Predictor: {predictor_description}",
        "Calibration: B_DEPTH_AWARE (Milestone 3 frozen scheme, refit identically here).",
        f"K = {K_MAX_CANDIDATES}, sample budget = {MAX_SAMPLES}, {len(incidents)} paired incidents "
        f"({', '.join(f'{k}={v}' for k, v in COVERAGE_STRATA.items())} coverage x depth {DEPTH_STRATA}, "
        f"{N_PER_CELL}/cell).",
        "",
        "## Arm summaries",
        "",
        "| arm | n | actionable<=1 | actionable<=2 | actionable<=3 | median samples | never actionable | final top-1 | final top-3 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        s = arm_summaries[arm]
        lines.append(
            f"| {arm} | {s['n']} | {s['actionable_within_1']:.3f} | {s['actionable_within_2']:.3f} | "
            f"{s['actionable_within_3']:.3f} | {s['median_samples_to_actionability']} | "
            f"{s['never_actionable_fraction']:.3f} | {s['final_top1']:.3f} | {s['final_top3']:.3f} |"
        )
    lines += [
        "",
        "## Paired bootstrap comparison vs RANDOM_VALID_UNSAMPLED (95% CI, "
        f"{BOOTSTRAP_RESAMPLES} resamples)",
        "",
        "| arm | actionable<=3 delta (pp) | 95% CI | meets >=10pp+CI bar | samples-to-resolution delta | 95% CI | clearly lower |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, comparison in comparisons.items():
        a3 = comparison["actionable_within_3_vs_random"]
        sr = comparison["samples_to_resolution_vs_random"]
        lines.append(
            f"| {arm} | {a3['point_estimate_pp']:.1f} | [{a3['ci95_low_pp']:.1f}, {a3['ci95_high_pp']:.1f}] | "
            f"{a3['meets_10pp_bar']} | {sr['point_estimate']} | "
            f"[{sr.get('ci95_low')}, {sr.get('ci95_high')}] | {sr.get('clearly_lower', False)} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        limitation_note,
        "",
        f"**Exit decision: {exit_decision}**",
        "",
        "Per the predeclared promotion rule (experiments.txt 5.5 / this script's module docstring): a policy is "
        "promoted only if it clears >=10pp actionable<=3 improvement with a 95% CI excluding zero, or a clearly "
        "(CI-excluding-zero) lower paired samples-to-resolution distribution -- never a bare point-estimate "
        "difference. If no arm clears this bar, active sampling remains advisory, matching the M0/CAP-REM-02 "
        "baseline finding this milestone was designed to test.",
    ]
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"exit_decision": exit_decision, "arm_summaries": arm_summaries, "comparisons": comparisons}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
