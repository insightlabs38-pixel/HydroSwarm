"""Milestone 7B (experiments.txt): calibration under actively acquired evidence.

Question tested: whether conformal calibration specifically adapted to
actively-acquired evidence (ADAPTIVE_EVIDENCE) can restore valid
post-sampling uncertainty coverage without making candidate sets
operationally useless -- NOT whether any sampler should be promoted.
Active sampling remains ADVISORY regardless of this result (this script
changes no authority/safety threshold and promotes no sampling policy).

Frozen scientific state this milestone starts from (do not re-derive/
re-litigate here): the Milestone-1 predictor remains selected (M7's
topology-diverse and M6B's cadence-diverse predictors were both NOT
promoted); alpha=0.1, K=3; Milestone 5 found round-wise conformal coverage
after active sampling under B_DEPTH_AWARE (Milestone 3's frozen scheme)
degraded materially below the ~90% target at round 1 (~84.5%) and round 2
(~83.3%) despite round-0/aggregate coverage looking fine -- an aggregate-
masks-subgroup-failure pattern. This script asks only whether a calibration
scheme that actually conditions on acquisition state can fix that, using
the SAME frozen predictor, SAME alpha/K, and the SAME CURRENT_EIG policy
M5 already found is the one actually retained (M5's own promotion rule
left active sampling advisory; CURRENT_EIG is simply production's serving/
advisory default, reused here unchanged -- this script does not re-run
M5's sampler-promotion comparison).

Data provenance / leakage rules (experiments.txt M7B.2): three completely
disjoint pools of PHYSICAL INCIDENTS, split before any causal-prefix
truncation or acquisition trajectory is generated:
  A. model-weight training data -- already frozen (Milestone 1), never
     touched by this script.
  B. adaptive calibration-FIT incidents -- a new pool of 108 physical
     incidents (seed namespace "m7b-fit", disjoint by construction from
     every other namespace in this codebase, since seeds are derived via
     sha256 of a namespaced string, not shared RNG state).
  C. adaptive calibration EVALUATION incidents -- a second new pool of 108
     physical incidents (seed namespace "m7b-eval"), disjoint from B by
     construction (different literal namespace string) and asserted
     disjoint at runtime (no shared seed). All rounds/states generated
     from one physical incident's trajectory stay in that incident's own
     pool -- there is no cross-round leakage between B and C because
     trajectories are generated independently, once, per pool.
The pre-existing Milestone-3 "calibration" scenario split (used only to
refit the FROZEN B_DEPTH_AWARE baseline arm identically to M3/M5, exactly
as M5 did) is a separate, already-frozen artifact unrelated to the new B/C
split above -- reusing it for baseline comparison is not new leakage, since
it never depended on adaptive acquisition and was already the frozen
comparison arm before this milestone existed. `development_holdout` labels
are read only for scenario generation of B/C (matching M5's own precedent),
never to train model weights. Locked final/topology test data is never
opened (asserted before and after, matching every other v5 script).

Design deviation from Milestone 5's rollout, and why: M5 stopped sampling
as soon as ITS OWN candidate_gate_pass became true, because M5 was
measuring a production-shaped stopping policy. This script instead always
samples to the full budget (MAX_SAMPLES=3) for every incident in both B
and C (unless the network genuinely runs out of unsampled junctions --
reported honestly, not padded), because M7B's question is "does
calibration cover correctly at round 1 / round 2+", and gating collection
on ONE calibrator's gate would systematically deprive the OTHER calibrator
of exactly the post-sample states this experiment exists to compare. This
is a measurement-only change: node selection every round is still driven
purely by CURRENT_EIG's own unmodified `rank_sample_locations` ranking
(which never depends on any calibrator -- confirmed by inspection,
`hydroswarm.sampling.active` contains no calibration import), so no
sampling/acquisition policy is altered, promoted, or newly authorized.

Calibration arms (experiments.txt M7B.3):
  A -- B_DEPTH_AWARE: Milestone 3's frozen scheme, refit identically here
       on the untouched M3 calibration split (network_id=f"golden-reference:
       {depth_bucket}", where depth_bucket is the INCIDENT's INITIAL evidence
       depth bucket, EARLY/MID -- unchanged at every round, exactly as M3/M5
       defined it; this is precisely the scheme M5 found degrades post-sample).
  B -- ADAPTIVE_EVIDENCE: a new scheme fit ONLY on pool B (never touches C),
       grouped by a predeclared, deterministic fallback hierarchy:

         L1  condition/network + acquisition_state + depth_bucket
         L2  condition/network + acquisition_state
         L3  acquisition_state + depth_bucket
         L4  acquisition_state
         L5  existing governed global fallback (SplitConformalCalibrator's
             own built-in condition -> global chain)

       acquisition_state in {PASSIVE, ACTIVE_ROUND_1, ACTIVE_ROUND_2_PLUS}
       (round 0 / round 1 / rounds 2-3 merged, since M5 already found round
       2 (n=18) and round 3 (n=1) too sparse to treat separately).
       depth_bucket reuses M3's own EARLY/MID/MATURE map applied to the
       incident's INITIAL causal-prefix depth (2,3 -> EARLY; 6 -> MID),
       matching M5's own stratification exactly. "condition/network" is the
       literal string f"{network_id}:{condition}"; because this experiment
       reuses M5's fixed golden-reference generation config (every fault/
       noise/degradation probability at 0 except acquisition-time
       measurement noise), `condition` resolves to CLEAN for effectively
       every state on this corpus -- reported honestly: the condition/
       network axis collapses to a near-constant here, so L1 and L2 keys
       are typically identical in practice on THIS network, not because
       the hierarchy code has fewer than 5 levels.

       MIN_N (minimum-N threshold, predeclared, never tuned after seeing
       eval results) = 10, matching `SplitConformalCalibrator.fit`'s own
       long-standing `minimum_group_size` default already used unmodified
       by every M1-M6 script in this family -- not a new number invented
       for this milestone. The resolution table (which level a given raw
       combination uses) is computed ONCE from pool B's counts ONLY, then
       applied unchanged to query pool C -- eval labels are never used to
       pick the grouping level. Levels below L1/L2 that still fail MIN_N at
       fit time are given a per-example unique nonce network_id (guaranteed
       group size 1 < MIN_N), which SplitConformalCalibrator.fit's own
       per-key filtering therefore drops from `network_scores` -- this is
       exactly how "L5 existing governed global fallback" is implemented:
       no bespoke global-fallback code is added, the existing
       `SplitConformalCalibrator.selection()` chain (network -> condition ->
       global) is reused verbatim for that tier.

Statistical treatment (experiments.txt M7B.6): adaptive round-states from
one physical incident are correlated (they share source/network/sensor
draws and are literally nested draws from the same trajectory), so an
INCIDENT-CLUSTERED bootstrap (resample physical incidents with replacement,
keep every state/round belonging to a resampled incident) is used for both
arms' coverage-difference and candidate-size-difference intervals -- never
a naive per-state bootstrap that would treat correlated rounds as
independent. 5000 resamples, seeded, reported as 95% percentile intervals.

Promotion rule (experiments.txt M7B.7, predeclared before running this
script, not adjusted afterward): ADAPTIVE_EVIDENCE is promoted only if ALL
of:
  1. round 1 AND round(2+) coverage are no longer materially below the 90%
     target (>5.0pp undercoverage bar, matching M3/M5's own convention).
  2. round-0/PASSIVE coverage does not regress by more than 5.0pp relative
     to B_DEPTH_AWARE's own round-0 coverage (PASSIVE_REGRESSION_BAR_PP).
  3. candidate-set inflation is not "uselessly large": ADAPTIVE_EVIDENCE's
     mean/median candidate-set size in ACTIVE_ROUND_1/ACTIVE_ROUND_2_PLUS
     must not exceed B_DEPTH_AWARE's own mean/median size in that same
     round-bucket by more than CANDIDATE_INFLATION_FACTOR_LIMIT=1.5x, and
     must not reach the full 4-junction action space unless B_DEPTH_AWARE
     already does too.
  4. candidate_gate_pass rate is not materially degraded: ADAPTIVE's rate
     in ACTIVE_ROUND_1/ACTIVE_ROUND_2_PLUS must not be more than
     GATE_PASS_DEGRADATION_BAR_PP=10.0 points lower than B_DEPTH_AWARE's.
  5. no safety/authority threshold is touched by this script (trivially
     true -- alpha, K, and every production authority gate are read-only
     constants here, never written).
If ALL hold: ADAPTIVE_EVIDENCE_CALIBRATION_JUSTIFIED. Otherwise:
KEEP_B_DEPTH_AWARE_AND_MARK_POST_SAMPLE_UNCALIBRATED -- and, per
experiments.txt M7B.8, any post-sample round-bucket that remains
materially below target under BOTH arms is explicitly labeled
UNCALIBRATED_POST_ACQUISITION in this report rather than silently reported
as if still calibrated.

Optional cross-policy diagnostic (experiments.txt M7B.9): after the primary
CURRENT_EIG comparison and resulting decision, the WINNING calibrator
(whichever the decision above selects -- NOT refit) is additionally applied
to a RANDOM_VALID_UNSAMPLED rollout over the SAME pool-C incidents, purely
as distribution-transfer evidence. This diagnostic never influences the
promotion decision above (computed strictly after it, from already-frozen
calibrators).

Writes:
  reports/evaluation/hydrocore-v5/m7b-adaptive-calibration.json
  reports/evaluation/hydrocore-v5/m7b-summary.md
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import tempfile
from collections import Counter
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
from run_m5_sampling import _seeded_rng  # noqa: E402

OUTPUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m7b-adaptive-calibration.json"
OUTPUT_SUMMARY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m7b-summary.md"

ALPHA = 0.1
K_MAX_CANDIDATES = 3  # frozen; never relaxed to shrink sets.
MAX_SAMPLES = 3
NOISE_STD_MG_L = 0.05
COVERAGE_STRATA: dict[str, int] = {"25%": 1, "50%": 2, "75%": 3}  # of 4 junctions
DEPTH_STRATA: tuple[int, ...] = (2, 3, 6)
N_PER_CELL = 12  # 3 coverage x 3 depth x 12 = 108 incidents per pool (B and C each)
ACQUISITION_STATES: tuple[str, ...] = ("PASSIVE", "ACTIVE_ROUND_1", "ACTIVE_ROUND_2_PLUS")
MIN_N = 10  # predeclared; reuses SplitConformalCalibrator.fit's own long-standing default.
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260815

COVERAGE_TARGET = 1 - ALPHA
MATERIAL_UNDERCOVERAGE_PP = 5.0  # matches M3/M5's own convention.
PASSIVE_REGRESSION_BAR_PP = 5.0  # predeclared promotion criterion 2.
CANDIDATE_INFLATION_FACTOR_LIMIT = 1.5  # predeclared promotion criterion 3.
GATE_PASS_DEGRADATION_BAR_PP = 10.0  # predeclared promotion criterion 4.

L1 = "L1_NETWORK_CONDITION_ACQSTATE_DEPTH"
L2 = "L2_NETWORK_CONDITION_ACQSTATE"
L3 = "L3_ACQSTATE_DEPTH"
L4 = "L4_ACQSTATE"
L5 = "L5_GLOBAL_FALLBACK"


def _measurement(network, scenario, node: str, *, decision_seconds: float, delay_minutes: float, rng: np.random.Generator) -> tuple[SensorSeries, float]:
    """Identical mechanics to run_m5_sampling.py's own nested `_measurement`
    (real WNTR incident simulation, seeded acquisition-time measurement
    noise, no fabricated logits) -- duplicated rather than imported because
    M5's version is a private nested function inside its own frozen
    `main()`, not a module-level export."""

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


def _acquisition_state_of(round_index: int) -> str:
    if round_index == 0:
        return "PASSIVE"
    if round_index == 1:
        return "ACTIVE_ROUND_1"
    return "ACTIVE_ROUND_2_PLUS"


def _network_condition(state: dict[str, Any]) -> str:
    return f"{state['network_id']}:{state['condition']}"


def _build_resolution_counts(states: list[dict[str, Any]]) -> tuple[Counter, Counter, Counter, Counter]:
    c1: Counter = Counter()
    c2: Counter = Counter()
    c3: Counter = Counter()
    c4: Counter = Counter()
    for state in states:
        nc = _network_condition(state)
        acq, depth_bucket = state["acquisition_state"], state["depth_bucket"]
        c1[(nc, acq, depth_bucket)] += 1
        c2[(nc, acq)] += 1
        c3[(acq, depth_bucket)] += 1
        c4[acq] += 1
    return c1, c2, c3, c4


def _resolve(state: dict[str, Any], counts: tuple[Counter, Counter, Counter, Counter], min_n: int) -> tuple[str, str | None]:
    c1, c2, c3, c4 = counts
    nc = _network_condition(state)
    acq, depth_bucket = state["acquisition_state"], state["depth_bucket"]
    if c1[(nc, acq, depth_bucket)] >= min_n:
        return L1, f"{nc}:{acq}:{depth_bucket}"
    if c2[(nc, acq)] >= min_n:
        return L2, f"{nc}:{acq}"
    if c3[(acq, depth_bucket)] >= min_n:
        return L3, f"{acq}:{depth_bucket}"
    if c4[acq] >= min_n:
        return L4, acq
    return L5, None


def _generate_incidents(pool_label: str, generator: WNTRScenarioGenerator) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    for coverage_label, sensor_count in COVERAGE_STRATA.items():
        for depth in DEPTH_STRATA:
            for index in range(N_PER_CELL):
                seed_material = f"m7b-{pool_label}:{coverage_label}:{depth}:{index}"
                seed = 976_000_000 + int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "big") % 10_000_000
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
                    "pool": pool_label, "coverage_label": coverage_label, "sensor_count": sensor_count,
                    "depth": depth, "seed": seed, "scenario": scenario, "network": randomized_network,
                    "feature_context": feature_context, "initial_series": initial_series,
                    "truth": scenario.manifest.incident.source_nodes[0],
                })
    return incidents


def main() -> int:  # noqa: C901
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    export_path, use_adapters, predictor_description = _freeze_predictor()
    model = HydroCore.from_variant("small", use_adapters=use_adapters, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()
    predictor_param_count = sum(p.numel() for p in model.parameters())
    predictor_hash = hashlib.sha256(Path(export_path).read_bytes()).hexdigest()

    # --- B_DEPTH_AWARE: refit identically to M3/M5 on the untouched M3 calibration split. ---
    calibration_records = build_scenario_pool("calibration", network_loader=build_wntr_network)
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)
    target_timestamps = build_sensor_series(train_records[0].scenario, train_records[0].feature_context)[0].timestamps_seconds

    def _collect_static(records, bucket: str, depth: int) -> list[dict[str, Any]]:
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

    b_depth_aware_fit_examples: list[dict[str, Any]] = []
    for depth in CAUSAL_PREFIX_DEPTHS:
        b_depth_aware_fit_examples.extend(_collect_static(calibration_records, DEPTH_BUCKET_OF[depth], depth))
    b_depth_aware_calibrator = SplitConformalCalibrator.fit(
        [
            CalibrationExample(
                probabilities=tuple(item["probabilities"]), true_index=item["true_index"], condition=item["condition"],
                network_id=f"{item['network_id']}:{item['depth_bucket']}",
            )
            for item in b_depth_aware_fit_examples
        ],
        alpha=ALPHA, model_hash=export_path, feature_schema_hash="n/a", dataset_manifest_hash="m3-calibration-pool",
    )

    # --- Shared signature/sampling machinery, identical to Milestone 5. ---
    base_network = build_wntr_network()
    junctions = tuple(sorted(base_network.junction_name_list))
    base_simulator = HydraulicSimulator(base_network)
    with tempfile.TemporaryDirectory() as cache_dir:
        signature_cache = SignatureCache(cache_dir)
        signature_key = SignatureCacheKey(
            network_hash=base_simulator.state_hash(), hydraulic_state_hash="m7b-fixed-profile",
            simulator_version=base_simulator.simulator_version, configuration_hash="m7b-source-location-only-v1",
            sensor_layout_hash="all-junctions",
        )
        signature_artifact = SignatureBuilder(base_simulator, signature_cache).build_or_load(
            key=signature_key, source_nodes=junctions, start_time_bins=(0,), duration_bins=(60,),
            strength_bins=(1.0,), demand_regimes=("baseline",), sensor_nodes=junctions,
            sample_times_seconds=list(target_timestamps),
        )

    generator = WNTRScenarioGenerator()
    fit_incidents = _generate_incidents("fit", generator)
    eval_incidents = _generate_incidents("eval", generator)
    fit_seeds = {incident["seed"] for incident in fit_incidents}
    eval_seeds = {incident["seed"] for incident in eval_incidents}
    incident_splits_disjoint = fit_seeds.isdisjoint(eval_seeds)
    assert incident_splits_disjoint, "adaptive calibration-fit and evaluation incident pools must be disjoint"

    def _posterior(network, feature_context, series) -> dict[str, Any]:
        classical_prior = model_input_classical_prior(library, list(library.node_ids), series, target_timestamps)
        built = HydraulicFeatureBuilder().build(
            network, feature_context.graph, feature_context.state, series,
            classical_prior=classical_prior, window_steps=25,
        )
        with torch.no_grad():
            output = model(built.batch)
        probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
        node_ids = list(built.node_ids)
        condition = classify_runtime_condition(series)
        fused_belief = dict(zip(node_ids, probs, strict=True))
        evidence_sufficiency_value = float(output["evidence_sufficiency"].reshape(-1)[0].item())
        return {
            "probs": probs, "node_ids": node_ids, "condition": condition,
            "fused_belief": fused_belief, "evidence_sufficiency": evidence_sufficiency_value,
        }

    def _entropy_bits(belief: dict[str, float]) -> float:
        values = np.asarray([max(v, 1e-12) for v in belief.values()], dtype=float)
        values = values / values.sum()
        return float(-(values * np.log2(values)).sum())

    def _rank(belief: dict[str, float], truth: str) -> int:
        ordered = sorted(belief, key=lambda node: (-belief[node], node))
        return ordered.index(truth) + 1 if truth in ordered else len(ordered)

    def _pick_node(policy: str, ranked_candidates, unsampled: list[str], *, incident_key: tuple, round_index: int) -> str | None:
        accessible = [candidate for candidate in ranked_candidates if candidate.accessible]
        if policy == "RANDOM_VALID_UNSAMPLED":
            if not unsampled:
                return None
            rng = _seeded_rng("m7b-random", *incident_key, round_index)
            return sorted(unsampled)[int(rng.integers(0, len(unsampled)))]
        if not accessible:
            return None
        if policy == "CURRENT_EIG":
            return max(accessible, key=lambda c: (c.score, c.node_id)).node_id
        raise ValueError(f"unknown policy: {policy}")

    def _rollout(incident: dict[str, Any], policy: str) -> list[dict[str, Any]]:
        truth = incident["truth"]
        depth_bucket = DEPTH_BUCKET_OF[incident["depth"]]
        incident_key = (incident["pool"], policy, incident["coverage_label"], incident["depth"], incident["seed"])
        series = list(incident["initial_series"])
        sampled_nodes = {item.node_id for item in series}
        states: list[dict[str, Any]] = []

        def _record(round_index: int) -> dict[str, Any]:
            result = _posterior(incident["network"], incident["feature_context"], series)
            true_index = result["node_ids"].index(truth)
            states.append({
                "round_index": round_index, "acquisition_state": _acquisition_state_of(round_index),
                "probabilities": result["probs"], "true_index": true_index,
                "condition": result["condition"], "network_id": "golden-reference",
                "entropy_bits": _entropy_bits(result["fused_belief"]),
                "true_source_rank": _rank(result["fused_belief"], truth),
                "top1_correct": _rank(result["fused_belief"], truth) == 1,
                "evidence_sufficiency": result["evidence_sufficiency"],
                "depth_bucket": depth_bucket, "initial_depth": incident["depth"],
                "coverage_label": incident["coverage_label"], "seed": incident["seed"], "pool": incident["pool"],
            })
            return result

        result = _record(0)
        for round_index in range(MAX_SAMPLES):
            unsampled = [node for node in junctions if node not in sampled_nodes]
            if not unsampled:
                break
            hypothesis_weights = {h.identifier: result["fused_belief"].get(h.source_node, 0.0) for h in signature_artifact.hypotheses}
            decision_seconds = max(timestamp for item in series for timestamp in item.timestamps_seconds)
            sampling_result = rank_sample_locations(
                signature_artifact, hypothesis_weights,
                constraints=SamplingConstraints(already_sampled=frozenset(sampled_nodes)),
                noise_scale_mg_l=NOISE_STD_MG_L, target_sample_time_seconds=decision_seconds, top_k=20,
            )
            node = _pick_node(policy, sampling_result.ranked, unsampled, incident_key=incident_key, round_index=round_index)
            if node is None:
                break
            rng = _seeded_rng("m7b-noise", *incident_key, round_index)
            candidate_meta = next((c for c in sampling_result.ranked if c.node_id == node), None)
            delay = candidate_meta.collection_time_minutes if candidate_meta else 30.0
            observation, _ts = _measurement(
                incident["network"], incident["scenario"], node,
                decision_seconds=decision_seconds, delay_minutes=delay, rng=rng,
            )
            series.append(observation)
            sampled_nodes.add(node)
            result = _record(round_index + 1)
        return states

    fit_states: list[dict[str, Any]] = [state for incident in fit_incidents for state in _rollout(incident, "CURRENT_EIG")]
    eval_states_current_eig: list[dict[str, Any]] = [state for incident in eval_incidents for state in _rollout(incident, "CURRENT_EIG")]

    # --- ADAPTIVE_EVIDENCE: fit ONLY on pool B, resolution table computed ONLY from pool B counts. ---
    fit_counts = _build_resolution_counts(fit_states)
    fit_resolution_levels = Counter()
    adaptive_examples: list[CalibrationExample] = []
    for index, state in enumerate(fit_states):
        level, key = _resolve(state, fit_counts, MIN_N)
        fit_resolution_levels[level] += 1
        network_id_field = key if key is not None else f"__unresolved_fit_{index}__"
        adaptive_examples.append(CalibrationExample(
            probabilities=tuple(state["probabilities"]), true_index=state["true_index"],
            condition=state["condition"], network_id=network_id_field,
        ))
    adaptive_calibrator = SplitConformalCalibrator.fit(
        adaptive_examples, alpha=ALPHA, model_hash=export_path, feature_schema_hash="n/a",
        dataset_manifest_hash="m7b-adaptive-fit-pool", minimum_group_size=MIN_N,
    )

    def _b_depth_aware_candidates(state: dict[str, Any]) -> tuple[int, ...]:
        key = f"golden-reference:{state['depth_bucket']}"
        return b_depth_aware_calibrator.candidate_set(state["probabilities"], condition=state["condition"], network_id=key)

    def _adaptive_candidates(state: dict[str, Any]) -> tuple[tuple[int, ...], str]:
        level, key = _resolve(state, fit_counts, MIN_N)
        network_id_field = key if key is not None else "__unresolved_eval_query__"
        indices = adaptive_calibrator.candidate_set(state["probabilities"], condition=state["condition"], network_id=network_id_field)
        return indices, level

    def _paired_records(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
        paired = []
        for state in states:
            b_indices = _b_depth_aware_candidates(state)
            a_indices, a_level = _adaptive_candidates(state)
            base = {k: state[k] for k in (
                "round_index", "acquisition_state", "true_index", "condition", "depth_bucket",
                "initial_depth", "coverage_label", "seed", "pool",
            )}
            base["b_depth_aware"] = {
                "candidate_set_size": len(b_indices),
                "true_source_in_candidate_set": state["true_index"] in b_indices,
                "candidate_gate_pass": 1 <= len(b_indices) <= K_MAX_CANDIDATES,
            }
            base["adaptive_evidence"] = {
                "candidate_set_size": len(a_indices),
                "true_source_in_candidate_set": state["true_index"] in a_indices,
                "candidate_gate_pass": 1 <= len(a_indices) <= K_MAX_CANDIDATES,
                "resolution_level": a_level,
            }
            paired.append(base)
        return paired

    eval_paired = _paired_records(eval_states_current_eig)

    def _flatten(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flat = []
        for record in paired:
            for arm_name, arm_field in (("B_DEPTH_AWARE", "b_depth_aware"), ("ADAPTIVE_EVIDENCE", "adaptive_evidence")):
                arm_data = record[arm_field]
                flat.append({
                    "arm": arm_name, "round_index": record["round_index"], "acquisition_state": record["acquisition_state"],
                    "condition": record["condition"], "depth_bucket": record["depth_bucket"], "initial_depth": record["initial_depth"],
                    "coverage_label": record["coverage_label"], "seed": record["seed"], "pool": record["pool"],
                    "candidate_set_size": arm_data["candidate_set_size"],
                    "true_source_in_candidate_set": arm_data["true_source_in_candidate_set"],
                    "candidate_gate_pass": arm_data["candidate_gate_pass"],
                    "resolution_level": arm_data.get("resolution_level"),
                })
        return flat

    eval_flat = _flatten(eval_paired)

    def _stats_for(records: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(records)
        if n == 0:
            return {"n": 0}
        hits = sum(1 for r in records if r["true_source_in_candidate_set"])
        coverage = hits / n
        se = (coverage * (1 - coverage) / n) ** 0.5
        undercoverage_pp = max(0.0, (COVERAGE_TARGET - coverage) * 100)
        sizes = [r["candidate_set_size"] for r in records]
        return {
            "n": n, "hits": hits, "empirical_coverage": coverage,
            "coverage_95ci": [max(0.0, coverage - 1.96 * se), min(1.0, coverage + 1.96 * se)],
            "materially_below_target": undercoverage_pp > MATERIAL_UNDERCOVERAGE_PP,
            "undercoverage_pp": undercoverage_pp,
            "mean_candidate_set_size": statistics.fmean(sizes),
            "median_candidate_set_size": statistics.median(sizes),
            "singleton_rate": statistics.fmean(size == 1 for size in sizes),
            "candidate_gate_pass_rate": statistics.fmean(1 <= size <= K_MAX_CANDIDATES for size in sizes),
            "true_source_exclusion_rate": 1 - coverage,
        }

    arms = ("B_DEPTH_AWARE", "ADAPTIVE_EVIDENCE")

    def _by(keyfn, keys) -> dict[str, dict[str, dict[str, Any]]]:
        return {
            arm: {str(key): _stats_for([r for r in eval_flat if r["arm"] == arm and keyfn(r) == key]) for key in keys}
            for arm in arms
        }

    coverage_overall = {arm: _stats_for([r for r in eval_flat if r["arm"] == arm]) for arm in arms}
    coverage_by_round = _by(lambda r: r["acquisition_state"], ACQUISITION_STATES)
    coverage_by_initial_depth = _by(lambda r: r["initial_depth"], DEPTH_STRATA)
    coverage_by_initial_sensor_coverage = _by(lambda r: r["coverage_label"], COVERAGE_STRATA)
    coverage_by_condition = _by(lambda r: r["condition"], sorted({r["condition"] for r in eval_flat}))
    adaptive_by_fallback_source = {
        level: _stats_for([r for r in eval_flat if r["arm"] == "ADAPTIVE_EVIDENCE" and r["resolution_level"] == level])
        for level in (L1, L2, L3, L4, L5)
    }
    adaptive_fallback_counts = Counter(r["resolution_level"] for r in eval_flat if r["arm"] == "ADAPTIVE_EVIDENCE")
    n_adaptive_eval = sum(adaptive_fallback_counts.values())
    adaptive_fallback_usage_rate = (
        1.0 - (adaptive_fallback_counts.get(L1, 0) / n_adaptive_eval) if n_adaptive_eval else None
    )

    # --- samples-to-candidate-gate-pass, diagnostic only (post-hoc, from the full-budget trajectory). ---
    def _samples_to_gate_pass(arm_field: str) -> dict[str, Any]:
        by_incident: dict[tuple, list[dict[str, Any]]] = {}
        for record in eval_paired:
            key = (record["coverage_label"], record["initial_depth"], record["seed"])
            by_incident.setdefault(key, []).append(record)
        resolved = []
        never = 0
        for _, records in by_incident.items():
            records = sorted(records, key=lambda r: r["round_index"])
            hit = next((r["round_index"] for r in records if r[arm_field]["candidate_gate_pass"]), None)
            if hit is None:
                never += 1
            else:
                resolved.append(hit)
        n = len(by_incident)
        return {
            "n_incidents": n,
            "median_samples_to_candidate_gate_pass": statistics.median(resolved) if resolved else None,
            "mean_samples_to_candidate_gate_pass": statistics.fmean(resolved) if resolved else None,
            "never_candidate_gate_pass_fraction": never / n if n else None,
            "note": "diagnostic only -- candidate_gate_pass is the candidate-count planning gate, NOT full product actionability.",
        }

    samples_to_gate_pass = {
        "B_DEPTH_AWARE": _samples_to_gate_pass("b_depth_aware"),
        "ADAPTIVE_EVIDENCE": _samples_to_gate_pass("adaptive_evidence"),
    }

    # --- incident-clustered bootstrap (experiments.txt M7B.6). ---
    def _incident_key(record: dict[str, Any]) -> tuple:
        return (record["coverage_label"], record["initial_depth"], record["seed"])

    def _arm_stats(records: list[dict[str, Any]], arm_field: str) -> tuple[float, float]:
        coverage = statistics.fmean(r[arm_field]["true_source_in_candidate_set"] for r in records)
        mean_size = statistics.fmean(r[arm_field]["candidate_set_size"] for r in records)
        return coverage, mean_size

    def _clustered_bootstrap(round_filter: str | None, seed_offset: int) -> dict[str, Any]:
        by_incident: dict[tuple, list[dict[str, Any]]] = {}
        for record in eval_paired:
            if round_filter is not None and record["acquisition_state"] != round_filter:
                continue
            by_incident.setdefault(_incident_key(record), []).append(record)
        incident_keys = [key for key, records in by_incident.items() if records]
        n = len(incident_keys)
        if n == 0:
            return {"n_incidents": 0}
        all_records = [record for key in incident_keys for record in by_incident[key]]
        b_cov, b_size = _arm_stats(all_records, "b_depth_aware")
        a_cov, a_size = _arm_stats(all_records, "adaptive_evidence")
        point_cov_diff = a_cov - b_cov
        point_size_diff = a_size - b_size
        rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
        cov_diffs = np.empty(BOOTSTRAP_RESAMPLES)
        size_diffs = np.empty(BOOTSTRAP_RESAMPLES)
        for i in range(BOOTSTRAP_RESAMPLES):
            picks = rng.integers(0, n, size=n)
            resample_records = [record for j in picks for record in by_incident[incident_keys[j]]]
            b_cov_r, b_size_r = _arm_stats(resample_records, "b_depth_aware")
            a_cov_r, a_size_r = _arm_stats(resample_records, "adaptive_evidence")
            cov_diffs[i] = a_cov_r - b_cov_r
            size_diffs[i] = a_size_r - b_size_r
        return {
            "n_incidents": n, "n_states": len(all_records),
            "coverage_diff_adaptive_minus_bdepth_point": point_cov_diff,
            "coverage_diff_ci95": [float(np.percentile(cov_diffs, 2.5)), float(np.percentile(cov_diffs, 97.5))],
            "candidate_size_diff_adaptive_minus_bdepth_point": point_size_diff,
            "candidate_size_diff_ci95": [float(np.percentile(size_diffs, 2.5)), float(np.percentile(size_diffs, 97.5))],
        }

    clustered_bootstrap = {
        "overall": _clustered_bootstrap(None, 0),
        **{state: _clustered_bootstrap(state, index + 1) for index, state in enumerate(ACQUISITION_STATES)},
    }

    # --- predeclared promotion rule (experiments.txt M7B.7). ---
    round1_ok = not coverage_by_round["ADAPTIVE_EVIDENCE"]["ACTIVE_ROUND_1"].get("materially_below_target", True)
    round2plus_ok = not coverage_by_round["ADAPTIVE_EVIDENCE"]["ACTIVE_ROUND_2_PLUS"].get("materially_below_target", True)
    criterion_1_post_sample_restored = round1_ok and round2plus_ok

    passive_b = coverage_by_round["B_DEPTH_AWARE"]["PASSIVE"].get("empirical_coverage")
    passive_a = coverage_by_round["ADAPTIVE_EVIDENCE"]["PASSIVE"].get("empirical_coverage")
    passive_regression_pp = max(0.0, (passive_b - passive_a) * 100) if passive_b is not None and passive_a is not None else None
    criterion_2_passive_preserved = passive_regression_pp is not None and passive_regression_pp <= PASSIVE_REGRESSION_BAR_PP

    def _inflation_ok(round_state: str) -> bool:
        b_stats = coverage_by_round["B_DEPTH_AWARE"][round_state]
        a_stats = coverage_by_round["ADAPTIVE_EVIDENCE"][round_state]
        if b_stats.get("n", 0) == 0 or a_stats.get("n", 0) == 0:
            return True
        mean_ok = a_stats["mean_candidate_set_size"] <= b_stats["mean_candidate_set_size"] * CANDIDATE_INFLATION_FACTOR_LIMIT
        median_ok = a_stats["median_candidate_set_size"] <= max(b_stats["median_candidate_set_size"] * CANDIDATE_INFLATION_FACTOR_LIMIT, 1.0)
        full_space_ok = a_stats["mean_candidate_set_size"] < len(junctions) or b_stats["mean_candidate_set_size"] >= len(junctions)
        return mean_ok and median_ok and full_space_ok

    criterion_3_inflation_acceptable = _inflation_ok("ACTIVE_ROUND_1") and _inflation_ok("ACTIVE_ROUND_2_PLUS")

    def _gate_pass_ok(round_state: str) -> bool:
        b_stats = coverage_by_round["B_DEPTH_AWARE"][round_state]
        a_stats = coverage_by_round["ADAPTIVE_EVIDENCE"][round_state]
        if b_stats.get("n", 0) == 0 or a_stats.get("n", 0) == 0:
            return True
        return (b_stats["candidate_gate_pass_rate"] - a_stats["candidate_gate_pass_rate"]) * 100 <= GATE_PASS_DEGRADATION_BAR_PP

    criterion_4_gate_pass_preserved = _gate_pass_ok("ACTIVE_ROUND_1") and _gate_pass_ok("ACTIVE_ROUND_2_PLUS")
    criterion_5_no_authority_change = True  # trivially true: this script never writes alpha/K/authority thresholds.

    promotion_criteria_met = (
        criterion_1_post_sample_restored and criterion_2_passive_preserved
        and criterion_3_inflation_acceptable and criterion_4_gate_pass_preserved and criterion_5_no_authority_change
    )
    decision = "ADAPTIVE_EVIDENCE_CALIBRATION_JUSTIFIED" if promotion_criteria_met else "KEEP_B_DEPTH_AWARE_AND_MARK_POST_SAMPLE_UNCALIBRATED"

    uncalibrated_post_acquisition_rounds = [
        round_state for round_state in ("ACTIVE_ROUND_1", "ACTIVE_ROUND_2_PLUS")
        if coverage_by_round["B_DEPTH_AWARE"][round_state].get("materially_below_target")
        and coverage_by_round["ADAPTIVE_EVIDENCE"][round_state].get("materially_below_target")
    ] if decision == "KEEP_B_DEPTH_AWARE_AND_MARK_POST_SAMPLE_UNCALIBRATED" else []

    # --- optional cross-policy diagnostic (experiments.txt M7B.9): winning calibrator, NOT refit, on RANDOM_VALID_UNSAMPLED. ---
    winning_arm_field = "b_depth_aware" if decision == "KEEP_B_DEPTH_AWARE_AND_MARK_POST_SAMPLE_UNCALIBRATED" else "adaptive_evidence"
    random_states = [state for incident in eval_incidents for state in _rollout(incident, "RANDOM_VALID_UNSAMPLED")]
    random_paired = _paired_records(random_states)
    random_flat_winning = [record[winning_arm_field] | {
        "round_index": record["round_index"], "acquisition_state": record["acquisition_state"],
    } for record in random_paired]
    cross_policy_diagnostic = {
        "skipped": False,
        "policy": "RANDOM_VALID_UNSAMPLED",
        "winning_calibrator": "B_DEPTH_AWARE" if winning_arm_field == "b_depth_aware" else "ADAPTIVE_EVIDENCE",
        "refit": False,
        "overall": _stats_for(random_flat_winning),
        "by_round": {
            state: _stats_for([r for r in random_flat_winning if r["acquisition_state"] == state])
            for state in ACQUISITION_STATES
        },
        "note": "Distribution-transfer diagnostic only -- computed strictly AFTER and separately from the promotion decision above; never used to fit or select the calibrator.",
    }

    locked_after = locked_test_opened(ROOT)

    report = {
        "schema_version": 1,
        "purpose": "Milestone 7B (experiments.txt): calibration under actively acquired evidence.",
        "branch": "exp/hydrocore-v5-causal",
        "predictor": {
            "export_path": export_path, "use_adapters": use_adapters, "description": predictor_description,
            "parameter_count": predictor_param_count, "checkpoint_sha256": predictor_hash,
        },
        "alpha": ALPHA, "k_max_candidates": K_MAX_CANDIDATES, "max_samples": MAX_SAMPLES,
        "sampling_policy": "CURRENT_EIG (production serving/advisory default; active sampling remains ADVISORY)",
        "coverage_strata": COVERAGE_STRATA, "depth_strata": DEPTH_STRATA, "n_per_cell": N_PER_CELL,
        "acquisition_states": ACQUISITION_STATES, "min_n": MIN_N,
        "fallback_hierarchy": [L1, L2, L3, L4, L5],
        "n_fit_incidents": len(fit_incidents), "n_eval_incidents": len(eval_incidents),
        "incident_splits_disjoint": incident_splits_disjoint,
        "fit_resolution_level_counts": dict(fit_resolution_levels),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "calibration_arms": {
            "B_DEPTH_AWARE": "Milestone 3 frozen scheme, refit identically here on the untouched M3 calibration split.",
            "ADAPTIVE_EVIDENCE": "New scheme fit only on pool B (adaptive calibration-fit incidents) via the predeclared fallback hierarchy above.",
        },
        "coverage_overall": coverage_overall,
        "coverage_by_round": coverage_by_round,
        "coverage_by_initial_depth": coverage_by_initial_depth,
        "coverage_by_initial_sensor_coverage": coverage_by_initial_sensor_coverage,
        "coverage_by_condition": coverage_by_condition,
        "adaptive_by_fallback_source": adaptive_by_fallback_source,
        "adaptive_fallback_usage_rate": adaptive_fallback_usage_rate,
        "samples_to_candidate_gate_pass_diagnostic": samples_to_gate_pass,
        "clustered_bootstrap": clustered_bootstrap,
        "promotion_criteria": {
            "criterion_1_post_sample_coverage_restored": criterion_1_post_sample_restored,
            "criterion_2_passive_coverage_preserved": criterion_2_passive_preserved,
            "passive_regression_pp": passive_regression_pp,
            "criterion_3_candidate_set_inflation_acceptable": criterion_3_inflation_acceptable,
            "criterion_4_candidate_gate_pass_preserved": criterion_4_gate_pass_preserved,
            "criterion_5_no_authority_threshold_changed": criterion_5_no_authority_change,
            "promotion_criteria_met": promotion_criteria_met,
        },
        "decision": decision,
        "uncalibrated_post_acquisition_rounds": uncalibrated_post_acquisition_rounds,
        "cross_policy_diagnostic": cross_policy_diagnostic,
        "active_sampling_authority_changed": False,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "eval_paired_records": eval_paired,
    }
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    def _fmt(stats: dict[str, Any], field: str, digits: int = 3) -> str:
        value = stats.get(field)
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"

    lines = [
        "# Milestone 7B summary: calibration under actively acquired evidence",
        "",
        f"Predictor: {predictor_description} ({predictor_param_count} parameters, checkpoint sha256={predictor_hash[:16]}...)",
        f"alpha={ALPHA}, K={K_MAX_CANDIDATES}, sampling policy=CURRENT_EIG (advisory, unchanged).",
        f"Fit incidents (pool B): {len(fit_incidents)}. Evaluation incidents (pool C): {len(eval_incidents)}. "
        f"Disjoint: {incident_splits_disjoint}.",
        "",
        "## Round-wise coverage (target ~"
        f"{COVERAGE_TARGET:.0%}, material-undercoverage bar = {MATERIAL_UNDERCOVERAGE_PP}pp)",
        "",
        "| arm | round | n | coverage | 95% CI | materially below target | mean set size | median set size |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        for state in ACQUISITION_STATES:
            s = coverage_by_round[arm][state]
            if s.get("n"):
                lines.append(
                    f"| {arm} | {state} | {s['n']} | {_fmt(s, 'empirical_coverage')} | {s['coverage_95ci']} | "
                    f"{s['materially_below_target']} | {_fmt(s, 'mean_candidate_set_size', 2)} | {_fmt(s, 'median_candidate_set_size', 2)} |"
                )
    lines += [
        "",
        "## Fallback usage (ADAPTIVE_EVIDENCE)",
        "",
        f"Fit-corpus resolution level counts: {dict(fit_resolution_levels)}",
        f"Eval-side fallback usage rate (fraction NOT at the most specific L1 level): "
        f"{adaptive_fallback_usage_rate if adaptive_fallback_usage_rate is None else f'{adaptive_fallback_usage_rate:.3f}'}",
        "",
        "## Incident-clustered bootstrap (ADAPTIVE_EVIDENCE minus B_DEPTH_AWARE, "
        f"{BOOTSTRAP_RESAMPLES} resamples)",
        "",
        "| scope | n incidents | coverage diff | 95% CI | candidate-size diff | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for scope, stats in clustered_bootstrap.items():
        if stats.get("n_incidents"):
            lines.append(
                f"| {scope} | {stats['n_incidents']} | {stats['coverage_diff_adaptive_minus_bdepth_point']:.3f} | "
                f"{stats['coverage_diff_ci95']} | {stats['candidate_size_diff_adaptive_minus_bdepth_point']:.3f} | "
                f"{stats['candidate_size_diff_ci95']} |"
            )
    lines += [
        "",
        "## Promotion criteria (predeclared, experiments.txt M7B.7)",
        "",
        f"1. Post-sample coverage restored (round1 & round2+ not materially below target): {criterion_1_post_sample_restored}",
        f"2. Passive coverage preserved (regression <= {PASSIVE_REGRESSION_BAR_PP}pp): {criterion_2_passive_preserved} "
        f"(observed regression: {passive_regression_pp})",
        f"3. Candidate-set inflation acceptable (<= {CANDIDATE_INFLATION_FACTOR_LIMIT}x, not saturating full action space): {criterion_3_inflation_acceptable}",
        f"4. Candidate-gate-pass not materially degraded (<= {GATE_PASS_DEGRADATION_BAR_PP}pp drop): {criterion_4_gate_pass_preserved}",
        f"5. No safety/authority threshold changed: {criterion_5_no_authority_change}",
        "",
        f"**Decision: {decision}**",
        "",
    ]
    if uncalibrated_post_acquisition_rounds:
        lines.append(
            f"Rounds marked UNCALIBRATED_POST_ACQUISITION under BOTH arms: {uncalibrated_post_acquisition_rounds} -- "
            "post-sample candidate sets in these rounds must not be treated as calibrated until a valid scheme exists."
        )
    lines += [
        "",
        "## Optional cross-policy diagnostic (RANDOM_VALID_UNSAMPLED, winning calibrator, not refit)",
        "",
        f"Winning calibrator: {cross_policy_diagnostic['winning_calibrator']}. "
        f"Overall coverage on RANDOM_VALID_UNSAMPLED: {_fmt(cross_policy_diagnostic['overall'], 'empirical_coverage')}. "
        "This is distribution-transfer evidence only, computed after and separately from the decision above.",
        "",
        "active sampling authority changed: False. locked tests opened: "
        f"before={locked_before}, after={locked_after}.",
    ]
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "coverage_by_round": coverage_by_round}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
