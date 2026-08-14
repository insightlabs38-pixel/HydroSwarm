"""Milestone 6 (experiments.txt): temporal realism -- cadence, detection
delay, and sensor observability.

Not in the original v4 design strongly enough; folded into v5. Uses the
frozen Milestone-1 winning predictor (arm A) and the frozen Milestone-3
B_DEPTH_AWARE calibration scheme, refit identically to M3/M4/M5 -- no
predictor or calibration tuning happens in this script (matches every
other M1-M5 script's "frozen predictor" discipline).

6.1 Telemetry cadence. Physically identical incidents sampled at
different report cadences, evaluated both by NUMBER OF REPORTS and by
ELAPSED PHYSICAL TIME.

Practicality note (empirically determined this session, not assumed):
the milestone's suggested cadence list is "5 15 30 60 min ... where
practical". Overriding `model.options.time.report_timestep` to 300s (5
min, matching the network's already-fine `quality_timestep`) was tried
first and made a single incident exceed a 180s simulation timeout
(confirmed via `HydraulicSimulator(..., timeout_seconds=180.0)`
directly) -- computationally impractical for a multi-incident corpus.
900s (15 min) was then tried and completed in 0.04s/incident (a >4000x
difference for a 3x finer-than-hourly-vs-5x request, indicating 300s
hits a genuine WNTR/EPANET pathological path on this build/platform, not
a smooth cost curve) -- so this experiment uses CADENCES_MINUTES = (15,
30, 60), dropping only the 5-minute point, with this timing evidence
recorded in the output JSON rather than silently substituting a
different resolution.

Every cadence in the retained set is generated from ONE shared 15-minute-
resolution `GeneratedScenario` per incident (same seed, same noise
draws): 30/60-minute cadences are produced by *subsampling* (stride 2/4)
that single high-resolution trajectory, never by re-simulating -- so
"physically identical incidents sampled at different cadences" is exact
by construction (same underlying truth AND same sensor noise draws at
every shared timestamp), not merely same-seed-different-run. (WNTR
itself forces `hydraulic_timestep` down to match `report_timestep` -- see
`build_high_resolution_network`'s docstring -- so this one shared
trajectory is solved at 15-minute hydraulic+quality resolution
throughout, applied identically across every incident.)

6.2 Detection delay. Reuses the SAME incident pool (their onset times
already vary via `start_time_bins_min`): the model only ever sees reports
with timestamp <= (true onset + delay), for delay in {0 ("near onset"),
60, 180, 360} minutes -- never reports from before that cutoff are
withheld and never a future report is leaked. The true onset itself is
never fed to the model as an input feature (only ever appears as the
`start_time` supervised TARGET elsewhere in this codebase, e.g.
`hydroswarm.training.causal_prefix.scenario_to_prefix_example`) -- this
script's own feature construction never reads `manifest.incident.start_minute`
into the model's input batch, confirmed by inspection below.

6.3 Irregular/asynchronous telemetry. Development-only stress cases
(never used for any promotion decision): timestamp jitter, unequal
per-sensor reporting intervals, delayed reports, gaps (forced missing),
and partial historical sensor availability. Compared against the same
incidents' matched 30-minute clean baseline from 6.1.

6.4 Sensor coverage and placement. RANDOM vs DEGREE_CENTRALITY (existing
static `demand_centrality` graph feature,
`hydroswarm.preprocessing.builder`) vs HYDRAULIC_OBSERVABILITY (physics-
based: the sensor subset minimizing mean pairwise source-signature
correlation, i.e. maximizing classical separability -- same diagnostic
family as Milestone 1.2's `build_m1_corpus.py::_identifiability_diagnostic`,
here conditioned on WHICH sensor columns are observed rather than on
evidence depth) placement policies, at fixed sensor budgets (1/2/3 of the
golden-reference network's 4 junctions), on matched incidents (same
source/timing draws across all three policies for a given incident
index; only which nodes are instrumented differs). Reports initial
top-1/top-3, candidate-set size (frozen B_DEPTH_AWARE calibration,
EARLY bucket, depth=3), a `candidate_gate_pass`-style actionability flag
(K=3, matching `hydroswarm.simulation.wrapper.MAXIMUM_EVALUATION_HYPOTHESES`
-- the same K every M4/M5 script uses, never relaxed), samples required
(greedy addition of the next-best-by-that-SAME-policy's-own-ranking
unplaced junction until the gate passes, re-evaluating the real model
each round -- no new active-sampling machinery invented), and physics
identifiability restricted to the placed sensor set.

6.5 Conditional architecture change. `hydroswarm.model.encoders.TemporalEncoder`
already docstrings itself as encoding "masked histories using elapsed
timestamps rather than array position", and
`hydroswarm.preprocessing.builder.HydraulicFeatureBuilder.build` already
feeds real per-timestep `age` (elapsed seconds since the latest
observation) into both `temporal_features` and `quality_features`, plus
real absolute `timestamps` into the batch -- confirmed by inspection, not
assumed. An explicit elapsed-time representation therefore already
exists in HydroCore's architecture; this milestone's own text only
requires testing a NEW explicit temporal representation "if the model is
strongly cadence-sensitive after causal-prefix training" (6.5). This
script's own 6.1 result is the actual go/no-go trigger, decided and
recorded in the output/summary -- not assumed in either direction ahead
of the data, and no new training run is launched unless 6.1 demonstrates
the need (Milestone 9's "do not add continuous-time complexity unless the
cadence study demonstrates the need" restated at 6.5).

No predictor, calibration, K, or verification threshold is changed from
Milestones 1/3/4/5. Uses development_holdout only (never train/
validation/calibration/locked).

Writes:
  reports/evaluation/hydrocore-v5/m6-cadence.json
  reports/evaluation/hydrocore-v5/m6-detection-delay.json
  reports/evaluation/hydrocore-v5/m6-irregular-telemetry.json
  reports/evaluation/hydrocore-v5/m6-sensor-placement.json
  reports/evaluation/hydrocore-v5/m6-summary.md
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

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
from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
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
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.simulation.wrapper import MAXIMUM_EVALUATION_HYPOTHESES, HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    NETWORK_FAMILY,
    build_scenario_pool,
    fit_pool_signature_library,
)
from hydroswarm.training.corpus import (  # noqa: E402
    build_feature_context,
    build_sensor_series,
    model_input_classical_prior,
)
from run_m1_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m3_calibration import DEPTH_BUCKET_OF, _freeze_predictor  # noqa: E402

OUT_CADENCE = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m6-cadence.json"
OUT_DELAY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m6-detection-delay.json"
OUT_IRREGULAR = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m6-irregular-telemetry.json"
OUT_PLACEMENT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m6-sensor-placement.json"
OUT_SUMMARY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m6-summary.md"

ALPHA = 0.1
K_MAX_CANDIDATES = MAXIMUM_EVALUATION_HYPOTHESES
assert K_MAX_CANDIDATES == 3
EPS = 1e-9

#: 6.1 practicality finding -- see module docstring. Recorded as data, not
#: just prose, so a reader of the JSON artifact sees the same evidence.
CADENCE_PRACTICALITY_NOTE = {
    "requested_by_milestone_text": [5, 15, 30, 60],
    "used": [15, 30, 60],
    "reason": (
        "report_timestep=300s (5 min) made a single HydraulicSimulator.simulate_incident call exceed "
        "a 180-second timeout (confirmed directly, not inferred); report_timestep=900s (15 min) completed "
        "the same call in 0.04s. A >4000x gap between a 3x and a 5x finer-than-hourly request indicates a "
        "pathological EPANET/WNTR code path at exactly 300s on this build/platform, not a smooth cost curve; "
        "generating a multi-incident corpus at 300s is computationally impractical here."
    ),
    "measured_300s_timeout_seconds": 180.0,
    "measured_900s_elapsed_seconds": 0.038,
}

REPORT_TIMESTEP_SECONDS = 900  # 15 minutes -- the finest practical cadence (see above).
CADENCES_MINUTES: tuple[int, ...] = (15, 30, 60)
STRIDE_OF: dict[int, int] = {cadence: cadence // 15 for cadence in CADENCES_MINUTES}
REPORT_COUNTS: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
ELAPSED_CHECKPOINTS_MIN: tuple[int, ...] = (60, 120, 180, 240, 360)
DETECTION_DELAYS_MIN: tuple[int, ...] = (0, 60, 180, 360)
REFERENCE_CADENCE_FOR_DELAY_AND_STRESS = 30

N_SOURCES_ONSETS_SEEDS = (4, 4, 3)  # 4 junctions x 4 onset bins x 3 seed repeats = 48 incidents.


def build_high_resolution_network():
    """Golden-reference network with `report_timestep` overridden to 15
    minutes (see CADENCE_PRACTICALITY_NOTE for why 15 rather than 5
    minutes). WNTR itself then forces `hydraulic_timestep` down to match
    (see the in-line comment below) -- both hydraulics and quality/
    reporting resolve at 15-minute granularity in this network, applied
    identically to every incident this experiment generates, so cross-
    cadence comparisons stay fair even though the "hydraulics stays
    hourly" framing originally assumed does not hold in practice."""

    network = build_wntr_network()
    network.options.time.report_timestep = REPORT_TIMESTEP_SECONDS
    #: WNTR requires report_timestep to be an integer multiple of
    #: hydraulic_timestep; since 900 < the default 3600, WNTR itself
    #: silently reduces hydraulic_timestep to 900 to satisfy that
    #: constraint (confirmed via the UserWarning it raises: "The report
    #: timestep must be an integer multiple of the hydraulic timestep.
    #: Reducing the hydraulic timestep..."). So this network solves BOTH
    #: hydraulics and reports every 15 minutes, not "hydraulics hourly,
    #: quality reported finer" as originally assumed -- still a valid,
    #: physically consistent finer-resolution simulation, applied
    #: identically to every incident in this experiment, just documented
    #: accurately here rather than left as the earlier (wrong) assumption.
    network.options.time.hydraulic_timestep = REPORT_TIMESTEP_SECONDS
    return network


def _load_frozen_predictor() -> tuple[HydroCore, str, str]:
    export_path, use_adapters, description = _freeze_predictor()
    model = HydroCore.from_variant("small", use_adapters=use_adapters, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()
    return model, export_path, description


def _seeded_rng(*parts: object) -> np.random.Generator:
    material = ":".join(str(part) for part in parts)
    seed_int = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    return np.random.default_rng(seed_int)


def _slice_series(series: SensorSeries, indices: Sequence[int]) -> SensorSeries:
    """`series` restricted to `indices` (must be non-empty, ascending --
    SensorSeries's own __post_init__ enforces non-decreasing timestamps).
    Factored out of `_resample_series`/`_truncate_series_by_time`/
    `_cadence_evidence_up_to_checkpoint` so all three share one
    field-projection implementation rather than three copies that could
    silently drift apart."""

    return SensorSeries(
        node_id=series.node_id,
        timestamps_seconds=tuple(series.timestamps_seconds[i] for i in indices),
        concentration_mg_l=tuple(series.concentration_mg_l[i] for i in indices),
        pressure_m=tuple(series.pressure_m[i] for i in indices),
        health=tuple(series.health[i] for i in indices),
        missing=tuple(series.missing[i] for i in indices),
        drift=tuple(series.drift[i] for i in indices),
        delayed=tuple(series.delayed[i] for i in indices),
        frozen=tuple(series.frozen[i] for i in indices),
    )


def _resample_series(series: SensorSeries, *, stride: int, depth: int) -> SensorSeries:
    """First `depth` points of `series` at spacing `stride` (indices
    0, stride, 2*stride, ..., (depth-1)*stride) -- growing evidence from
    incident onset, same "prefix" semantics as `hydroswarm.training.
    causal_prefix.truncate_causal_prefix`, generalized with a stride so it
    can express a coarser report cadence than the underlying array's own
    resolution.

    Milestone-6 correction note: `depth` points spaced `stride` apart span
    `(depth - 1) * stride` steps, i.e. `(depth - 1) * cadence_minutes` of
    ELAPSED TIME (first index 0, last index `(depth-1)*stride`), NOT
    `depth * cadence_minutes` -- a depth=1 series is a single instant and
    spans zero elapsed time regardless of cadence. This function itself
    was never wrong (it just returns "the first `depth` strided points");
    the bug fixed this milestone was every CALLER that inferred elapsed
    time from `depth * cadence` instead of reading the actual timestamps
    the resulting series carries -- see `run_cadence_experiment`."""

    n = len(series.timestamps_seconds)
    indices = [i for i in range(0, n, stride)][: max(depth, 1)]
    if not indices:
        indices = [0]
    return _slice_series(series, indices)


def _truncate_series_by_time(series: SensorSeries, *, cutoff_seconds: float) -> SensorSeries | None:
    """All points of `series` with timestamp <= cutoff_seconds. Returns
    None if the sensor has no report at or before the cutoff (a genuinely
    not-yet-reporting sensor, never synthesized)."""

    indices = [i for i, t in enumerate(series.timestamps_seconds) if t <= cutoff_seconds]
    if not indices:
        return None
    return _slice_series(series, indices)


def _cadence_evidence_up_to_checkpoint(
    full_series: Sequence[SensorSeries], *, stride: int, checkpoint_minutes: int,
) -> list[SensorSeries]:
    """Milestone-6 correction (item 1): the ROBUST construction the
    corrected matched-elapsed-time analysis uses instead of the retired
    `depth = checkpoint // cadence` formula. For each sensor, first takes
    the cadence's own strided report positions (0, stride, 2*stride, ...
    -- the reports that sensor ACTUALLY produces at this cadence), then
    keeps only those at or before `first_timestamp + checkpoint_minutes`
    -- i.e. "every report this cadence would really have delivered by this
    physical-time boundary", read from real timestamps rather than
    inferred from depth/cadence arithmetic. Always keeps at least the
    first (t=0) report, matching every other truncation helper in this
    file's "depth >= 1" convention. Robust to `checkpoint_minutes` not
    being an exact multiple of `stride` (falls back to whatever the
    cadence actually delivered by then, never rounds up past the
    boundary)."""

    if not full_series:
        raise ValueError("full_series must not be empty")
    first_ts = full_series[0].timestamps_seconds[0]
    cutoff = first_ts + checkpoint_minutes * 60.0
    out = []
    for item in full_series:
        n = len(item.timestamps_seconds)
        strided = list(range(0, n, stride))
        selected = [i for i in strided if item.timestamps_seconds[i] <= cutoff]
        if not selected:
            selected = [strided[0]] if strided else [0]
        out.append(_slice_series(item, selected))
    return out


def _cadence_evidence_since_onset(
    full_series: Sequence[SensorSeries], *, stride: int, onset_seconds: float, n_reports: int,
) -> list[SensorSeries] | None:
    """Milestone-6 final validity check (item 2): the post-onset-anchored
    construction. `onset_seconds` is used ONLY to select which of the
    cadence's own strided report positions to include -- it is never
    written into any SensorSeries field or fed to the model (the model
    only ever sees the resulting timestamps/concentrations, exactly like
    every other evidence-construction helper in this file).

    For each sensor, takes the cadence's own strided report positions (0,
    stride, 2*stride, ...) that fall AT OR AFTER `onset_seconds`, then
    keeps the first `n_reports` of those -- i.e. "the first N reports this
    cadence would actually deliver once contamination has truly begun",
    as opposed to `_resample_series`'s first N reports from t=0 regardless
    of onset (which `_cadence_evidence_up_to_checkpoint`'s sibling,
    `_resample_series`-based `depth_rows`, uses, and which this check
    exists to stress-test for a pre-onset confound).

    Returns None if fewer than `n_reports` post-onset positions exist in
    the underlying series for this cadence (never pads/fabricates a
    missing future report)."""

    if not full_series:
        raise ValueError("full_series must not be empty")
    out = []
    for item in full_series:
        n = len(item.timestamps_seconds)
        strided = list(range(0, n, stride))
        post_onset = [i for i in strided if item.timestamps_seconds[i] >= onset_seconds]
        if len(post_onset) < n_reports:
            return None
        out.append(_slice_series(item, post_onset[:n_reports]))
    return out


def _prediction_fields(probs: list[float], node_ids: list[str]) -> dict[str, Any]:
    """Milestone-6 correction (item 3): the paired-identity fields every
    cadence-comparison row now carries, so "identical prediction" claims
    can be verified per-incident instead of inferred from equal aggregate
    accuracy (two cadences can share top-1=0.5 while disagreeing on WHICH
    half of the incidents they get right)."""

    predicted_index = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "predicted_index": predicted_index,
        "predicted_node": node_ids[predicted_index],
        "probs": list(probs),
    }


def _l1_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """L1 (sum of absolute differences) distance between two probability
    vectors over the same node ordering -- chosen (over e.g.
    Jensen-Shannon) for being simple, bounded ([0, 2] for two probability
    vectors), and directly interpretable as "total probability mass moved
    between predicted sources"; used consistently for every probability-
    vector-distance figure this correction reports."""

    return float(sum(abs(x - y) for x, y in zip(a, b, strict=True)))


def _paired_prediction_analysis(rows_by_cadence: dict[int, dict[tuple, dict[str, Any]]]) -> dict[str, Any]:
    """Milestone-6 correction (item 3). `rows_by_cadence[cadence][incident_key]`
    is one inference row (must carry predicted_node/probs from
    `_prediction_fields`). Compares the SAME incidents (by `incident_key`)
    across every cadence that has data for them -- never aggregate-only.
    Distance metric: L1 (`_l1_distance`), see that function's docstring."""

    cadences = sorted(rows_by_cadence)
    common_keys = None
    for cadence in cadences:
        keys = set(rows_by_cadence[cadence])
        common_keys = keys if common_keys is None else (common_keys & keys)
    common_keys = common_keys or set()
    n = len(common_keys)

    identical_all = 0
    l1_distances: list[float] = []
    for key in common_keys:
        predicted_nodes = [rows_by_cadence[c][key]["predicted_node"] for c in cadences]
        if len(set(predicted_nodes)) == 1:
            identical_all += 1
        for c1, c2 in itertools.combinations(cadences, 2):
            l1_distances.append(_l1_distance(rows_by_cadence[c1][key]["probs"], rows_by_cadence[c2][key]["probs"]))

    pairwise_top1_agreement: dict[str, Any] = {}
    for c1, c2 in itertools.combinations(cadences, 2):
        pair_keys = set(rows_by_cadence[c1]) & set(rows_by_cadence[c2])
        if not pair_keys:
            pairwise_top1_agreement[f"{c1}v{c2}"] = None
            continue
        agree = sum(
            1 for key in pair_keys
            if rows_by_cadence[c1][key]["predicted_node"] == rows_by_cadence[c2][key]["predicted_node"]
        )
        pairwise_top1_agreement[f"{c1}v{c2}"] = {"n": len(pair_keys), "agreement_fraction": agree / len(pair_keys)}

    top1_by_cadence = {
        str(cadence): statistics.fmean(rows_by_cadence[cadence][key]["top1"] for key in common_keys)
        for cadence in cadences
    } if common_keys else {}

    return {
        "distance_metric": "L1 (sum of absolute differences between the two probability vectors over the same node ordering)",
        "n_incidents_matched_across_all_cadences": n,
        "fraction_identical_predicted_node_all_cadences": (identical_all / n) if n else None,
        "fraction_prediction_changes_across_cadences": (1.0 - identical_all / n) if n else None,
        "pairwise_top1_agreement": pairwise_top1_agreement,
        "mean_l1_probability_distance": statistics.fmean(l1_distances) if l1_distances else None,
        "max_l1_probability_distance": max(l1_distances) if l1_distances else None,
        "top1_by_cadence": top1_by_cadence,
    }


def _grouped_paired_analysis(
    rows: Sequence[dict[str, Any]], *, cadences: Sequence[int], cadence_field: str, group_field: str,
) -> dict[str, Any]:
    """Milestone-6 final validity check (item 1): groups `rows` by
    `group_field` (e.g. `onset_minutes`) and runs `_paired_prediction_analysis`
    independently within each group -- so a confound where one group (e.g.
    pre-onset incidents) dominates the pooled statistic cannot masquerade
    as a genuine cross-cadence finding. `rows` must carry `incident_key`,
    `predicted_node`, `probs`, `top1` (from `_prediction_fields`/
    `_row_metrics`)."""

    groups = sorted({row[group_field] for row in rows})
    out: dict[str, Any] = {}
    for group_value in groups:
        rows_by_cadence: dict[int, dict[tuple, dict[str, Any]]] = {}
        for cadence in cadences:
            subset = [row for row in rows if row[cadence_field] == cadence and row[group_field] == group_value]
            if subset:
                rows_by_cadence[cadence] = {row["incident_key"]: row for row in subset}
        if len(rows_by_cadence) >= 2:
            out[str(group_value)] = _paired_prediction_analysis(rows_by_cadence)
    return out


def _infer(
    model: HydroCore, network, feature_context, series: Sequence[SensorSeries], library, target_timestamps,
) -> dict[str, Any]:
    """One forward pass: real live-serving classical_prior (nearest-time
    alignment onto the TRAIN library's own hourly grid -- correct for
    irregular/non-hourly evidence, see `model_input_classical_prior`'s own
    docstring and `run_m5_sampling.py`'s architecture note for why this is
    used instead of `_prefix_classical_prior`'s positional-grid assumption),
    real per-example window_steps (never padded to a fixed 25 regardless of
    how much evidence actually exists)."""

    classical_prior = model_input_classical_prior(library, list(library.node_ids), series, target_timestamps)
    window_steps = max(len(item.timestamps_seconds) for item in series)
    built = HydraulicFeatureBuilder().build(
        network, feature_context.graph, feature_context.state, series,
        classical_prior=classical_prior, window_steps=window_steps,
    )
    started = time.perf_counter()
    with torch.no_grad():
        output = model(built.batch)
    latency = time.perf_counter() - started
    probs_tensor = torch.softmax(output["source_node_logits"][0], dim=-1)
    node_ids = list(built.node_ids)
    return {
        "probs": probs_tensor.tolist(),
        "probs_tensor": probs_tensor,
        "node_ids": node_ids,
        "evidence_sufficiency": float(output["evidence_sufficiency"].reshape(-1)[0].item()),
        "latency_seconds": latency,
        "condition": classify_runtime_condition(series),
    }


def _row_metrics(probs: list[float], node_ids: list[str], truth_node: str) -> dict[str, Any]:
    truth_index = node_ids.index(truth_node)
    row_probs = {index: value for index, value in enumerate(probs)}
    p_truth = probs[truth_index]
    onehot = [0.0] * len(probs)
    onehot[truth_index] = 1.0
    sorted_probs = sorted(probs, reverse=True)
    rank = sorted_probs.index(p_truth) + 1
    return {
        "top1": localization_top_k(row_probs, truth_index, k=1),
        "top3": localization_top_k(row_probs, truth_index, k=3),
        "mrr": mean_reciprocal_rank([row_probs], [truth_index]),
        "nll": -math.log(p_truth + EPS),
        "brier": sum((p - o) ** 2 for p, o in zip(probs, onehot, strict=True)),
        "posterior_entropy": -sum(p * math.log(p + EPS) for p in probs),
        "true_source_rank": rank,
    }


def _candidate_set(calibrator: SplitConformalCalibrator, probs: list[float], *, condition: str, bucket: str) -> tuple[int, ...]:
    return calibrator.candidate_set(probs, condition=condition, network_id=f"{NETWORK_FAMILY}:{bucket}")


def _fit_frozen_calibrator(model: HydroCore, library, calibration_records) -> SplitConformalCalibrator:
    """Refits the frozen B_DEPTH_AWARE scheme (reports/evaluation/hydrocore-v5/
    m3-summary.md's decision) identically to run_m3_calibration.py/
    run_m5_sampling.py: same alpha, same network_id encoding
    (f"{network_id}:{depth_bucket}"), same calibration split, no tuning."""

    from hydroswarm.training.causal_prefix import scenario_to_prefix_example, truncate_causal_prefix

    examples = []
    with torch.no_grad():
        for depth in CAUSAL_PREFIX_DEPTHS:
            for record in calibration_records:
                scenario = record.scenario
                example = scenario_to_prefix_example(scenario, record.network, library, depth, feature_context=record.feature_context)
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
                truth = int(example.targets["source_node"].item())
                full_series = build_sensor_series(scenario, record.feature_context)
                truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
                condition = classify_runtime_condition(truncated_series)
                bucket = DEPTH_BUCKET_OF[depth]
                examples.append(CalibrationExample(
                    probabilities=tuple(probs), true_index=truth, condition=condition,
                    network_id=f"{scenario.manifest.network_id}:{bucket}",
                ))
    return SplitConformalCalibrator.fit(
        examples, alpha=ALPHA, model_hash="m6-temporal", feature_schema_hash="n/a",
        dataset_manifest_hash="m6-calibration-pool",
    )


# ---------------------------------------------------------------------------
# Shared matched-incident pool (6.1 / 6.2 / 6.3)
# ---------------------------------------------------------------------------


def _generate_cadence_incident_pool(junctions: tuple[str, ...]) -> list[dict[str, Any]]:
    """One 15-minute-resolution GeneratedScenario per (source, onset, seed
    repeat) -- shared verbatim by 6.1 (cadence), 6.2 (detection delay), and
    6.3 (irregular telemetry stress cases), so all three subsections read
    "physically identical incidents" in the strict sense: literally the
    same underlying trajectory and noise draws, sliced differently.

    Full sensor coverage (sensor_count=len(junctions)) here -- placement is
    Milestone 6.4's own independent question; leaving it out here isolates
    the cadence/delay/irregularity variables cleanly."""

    onset_bins = (0, 60, 120, 240)
    n_sources, n_onsets, n_seeds = N_SOURCES_ONSETS_SEEDS
    incidents = []
    generator = WNTRScenarioGenerator()
    for source in junctions[:n_sources]:
        for onset in onset_bins[:n_onsets]:
            for repeat in range(n_seeds):
                seed_material = f"m6-cadence-pool:{source}:{onset}:{repeat}"
                seed = 960_000_000 + int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "big") % 10_000_000
                network = build_high_resolution_network()
                config = ScenarioGenerationConfig(
                    seed=seed, network_id=NETWORK_FAMILY, network_family=NETWORK_FAMILY,
                    split=DatasetSplit.DEVELOPMENT_HOLDOUT, stage=CurriculumStage.OPERATIONAL,
                    event_type=EventType.CONTAMINATION, source_node=source,
                    start_time_bins_min=(onset,), duration_bins_min=(60,), strength_bins=(1.0,),
                    demand_regimes=(1.0,), sensor_count=len(junctions), sensor_noise_std=0.01,
                    missing_probability=0.0, drift_per_hour=0.0, frozen_probability=0.0,
                    communication_outage_probability=0.0, unit_mismatch_probability=0.0,
                    roughness_variation_fraction=0.0, tank_level_variation_fraction=0.0,
                    pipe_outage_probability=0.0,
                )
                scenario, randomized_network = generator.generate_with_network(network, config)
                feature_context = build_feature_context(randomized_network)
                full_series = build_sensor_series(scenario, feature_context)
                incidents.append({
                    "seed": seed, "source": source, "onset_minutes": onset,
                    "scenario": scenario, "network": randomized_network,
                    "feature_context": feature_context, "full_series": full_series,
                })
    return incidents


# ---------------------------------------------------------------------------
# 6.1 Telemetry cadence
# ---------------------------------------------------------------------------


def run_cadence_experiment(model, library, target_timestamps, incidents) -> dict[str, Any]:
    """Milestone-6 correction: see this file's module-level "Milestone 6
    correction" note and `_cadence_evidence_up_to_checkpoint`/
    `_paired_prediction_analysis`'s docstrings for the specific bugs fixed
    (matched-elapsed-time off-by-one; fixed-report-count elapsed labels;
    paired prediction-identity analysis added)."""

    REPORTING_RESOLUTION_MINUTES = REPORT_TIMESTEP_SECONDS / 60.0  # 15.0 -- the finest cadence's own resolution.

    # ---- Fixed report-count rows (depth axis). Construction itself
    # (_resample_series) was never wrong; every row now additionally
    # carries the REAL first/last timestamp and predicted_node/probs so
    # elapsed time and prediction identity are read from data, not formula.
    depth_rows: list[dict[str, Any]] = []
    for incident in incidents:
        truth = incident["source"]
        incident_key = (incident["seed"], incident["source"], incident["onset_minutes"])
        for cadence in CADENCES_MINUTES:
            stride = STRIDE_OF[cadence]
            available = len(range(0, len(incident["full_series"][0].timestamps_seconds), stride))
            for depth in REPORT_COUNTS:
                if depth > available:
                    continue
                series = [_resample_series(item, stride=stride, depth=depth) for item in incident["full_series"]]
                first_timestamp = series[0].timestamps_seconds[0]
                last_timestamp = series[0].timestamps_seconds[-1]
                actual_elapsed_minutes = (last_timestamp - first_timestamp) / 60.0
                result = _infer(model, incident["network"], incident["feature_context"], series, library, target_timestamps)
                metrics = _row_metrics(result["probs"], result["node_ids"], truth)
                prediction = _prediction_fields(result["probs"], result["node_ids"])
                depth_rows.append({
                    "incident_key": incident_key, "seed": incident["seed"], "source": truth,
                    "onset_minutes": incident["onset_minutes"], "cadence_minutes": cadence, "report_count": depth,
                    "first_timestamp": first_timestamp, "last_timestamp": last_timestamp,
                    "actual_elapsed_minutes": actual_elapsed_minutes,
                    "latency_seconds": result["latency_seconds"], **metrics, **prediction,
                })

    def _aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        return {
            "n": len(group),
            **{
                metric: statistics.fmean(row[metric] for row in group)
                for metric in (
                    "top1", "top3", "mrr", "nll", "brier", "posterior_entropy", "true_source_rank",
                    "latency_seconds", "actual_elapsed_minutes",
                )
            },
        }

    by_cadence_depth: dict[str, dict[str, Any]] = {}
    for cadence in CADENCES_MINUTES:
        by_cadence_depth[str(cadence)] = {}
        for depth in REPORT_COUNTS:
            group = [r for r in depth_rows if r["cadence_minutes"] == cadence and r["report_count"] == depth]
            by_cadence_depth[str(cadence)][str(depth)] = _aggregate(group)

    #: Milestone-6 correction (item 2): elapsed time here is now the
    #: ACTUAL mean `actual_elapsed_minutes` already aggregated above
    #: (identical across incidents for this clean, shared-grid pool --
    #: verified, not assumed, via the per-row values feeding the mean),
    #: never `depth * cadence`.
    report_count_invariance: dict[str, Any] = {}
    for depth in REPORT_COUNTS:
        top1s, elapsed_by_cadence = {}, {}
        for cadence in CADENCES_MINUTES:
            entry = by_cadence_depth[str(cadence)].get(str(depth))
            if entry and entry.get("n", 0) > 0:
                top1s[str(cadence)] = entry["top1"]
                elapsed_by_cadence[str(cadence)] = entry["actual_elapsed_minutes"]
        if len(top1s) >= 2:
            report_count_invariance[str(depth)] = {
                "top1_by_cadence": top1s,
                "actual_elapsed_minutes_by_cadence": elapsed_by_cadence,
                "spread_pp": (max(top1s.values()) - min(top1s.values())) * 100,
            }

    #: Milestone-6 correction ("important interpretation"): depth=1 is a
    #: single observation and spans ZERO elapsed time at every cadence
    #: (first_timestamp == last_timestamp) -- identical behavior there is
    #: EXPECTED and is NEVER used as evidence of report-count/elapsed-time
    #: conflation. The earliest MEANINGFUL fixed-report-count comparison
    #: is depth >= 2 (and the go/no-go trigger below uses depth=2
    #: specifically, the earliest one).
    DEPTH1_ELAPSED_TIME_CAVEAT = (
        "depth=1 is a single observation and therefore spans 0 minutes of elapsed time regardless of cadence "
        "(first_timestamp == last_timestamp for every cadence). Identical behavior at depth=1 is EXPECTED and is "
        "NEVER used as evidence of report-count/elapsed-time conflation; the earliest meaningful fixed-report-"
        "count comparison in this analysis is depth >= 2."
    )

    #: Milestone-6 correction (item 3): paired per-incident predicted-node
    #: identity, fixed report count, depth >= 2 only (depth=1 excluded per
    #: DEPTH1_ELAPSED_TIME_CAVEAT). Distinguishes "aggregate top-1 accuracy
    #: is equal" from "predictions are identical" -- see
    #: _paired_prediction_analysis's docstring.
    paired_by_depth: dict[str, Any] = {}
    for depth in REPORT_COUNTS:
        if depth < 2:
            continue
        rows_by_cadence: dict[int, dict[tuple, dict[str, Any]]] = {}
        for cadence in CADENCES_MINUTES:
            group = [r for r in depth_rows if r["cadence_minutes"] == cadence and r["report_count"] == depth]
            if group:
                rows_by_cadence[cadence] = {r["incident_key"]: r for r in group}
        if len(rows_by_cadence) >= 2:
            paired_by_depth[str(depth)] = _paired_prediction_analysis(rows_by_cadence)

    #: Predeclared bar (item 4B): paired predictions are "substantially
    #: invariant" if >=90% of matched incidents predict the exact same
    #: node across every available cadence. Evaluated ONLY at depth=2 --
    #: the earliest meaningful fixed-report-count comparison.
    PAIRED_INVARIANCE_FRACTION_BAR = 0.9
    depth2_paired = paired_by_depth.get("2")
    report_count_conflation_at_depth2 = bool(
        depth2_paired
        and depth2_paired.get("fraction_identical_predicted_node_all_cadences") is not None
        and depth2_paired["fraction_identical_predicted_node_all_cadences"] >= PAIRED_INVARIANCE_FRACTION_BAR
    )

    # ---- Final validity check (item 1): the t=0-anchored depth_rows above
    # can confound "report-count/time conflation" with "no contamination
    # signal was in the window yet" -- the incident pool's onset bins are
    # 0/60/120/240 minutes, so at depth=2 (max elapsed span 60 minutes
    # across all three cadences) only the onset=0 incidents are guaranteed
    # to contain any post-onset evidence at every cadence; onset=60/120/240
    # incidents' depth=2 windows are wholly or mostly PRE-onset. Stratifying
    # paired_by_depth by true onset (never fed to the model -- see
    # depth_rows' own construction, which never reads onset into any
    # SensorSeries field) checks whether the pooled invariance above
    # actually holds when contamination is observable, or is dominated by
    # trivially-invariant pre-onset cases.
    onset_stratified_paired_by_depth: dict[str, Any] = {}
    for depth in REPORT_COUNTS:
        if depth < 2:
            continue
        depth_subset = [r for r in depth_rows if r["report_count"] == depth]
        stratified = _grouped_paired_analysis(
            depth_subset, cadences=CADENCES_MINUTES, cadence_field="cadence_minutes", group_field="onset_minutes",
        )
        if stratified:
            onset_stratified_paired_by_depth[str(depth)] = stratified

    onset0_depth2 = onset_stratified_paired_by_depth.get("2", {}).get("0")
    onset0_depth2_identical_fraction = (
        onset0_depth2.get("fraction_identical_predicted_node_all_cadences") if onset0_depth2 else None
    )
    #: The pooled depth=2 claim is confounded if it met the invariance bar
    #: while the ONLY onset stratum guaranteed real post-onset signal at
    #: every cadence (onset=0) does NOT meet that same bar -- i.e. the
    #: pooled number was materially propped up by pre-onset (no-signal)
    #: incidents rather than reflecting genuine cadence-invariant behavior
    #: on real contamination evidence.
    depth2_confound_suspected = bool(
        report_count_conflation_at_depth2
        and onset0_depth2_identical_fraction is not None
        and onset0_depth2_identical_fraction < PAIRED_INVARIANCE_FRACTION_BAR
    )

    # ---- Final validity check (item 2): post-onset-anchored diagnostic.
    # `onset_seconds` is used ONLY to select which real reports to include
    # (see _cadence_evidence_since_onset's docstring) -- never fed to the
    # model. Tests whether prediction invariance persists when the
    # evidence actually contains contamination signal from the start.
    POST_ONSET_REPORT_COUNTS: tuple[int, ...] = (2, 3)
    post_onset_rows: list[dict[str, Any]] = []
    for incident in incidents:
        truth = incident["source"]
        incident_key = (incident["seed"], incident["source"], incident["onset_minutes"])
        onset_seconds = incident["onset_minutes"] * 60.0
        for cadence in CADENCES_MINUTES:
            stride = STRIDE_OF[cadence]
            for n_reports in POST_ONSET_REPORT_COUNTS:
                series = _cadence_evidence_since_onset(
                    incident["full_series"], stride=stride, onset_seconds=onset_seconds, n_reports=n_reports,
                )
                if series is None:
                    continue  # fewer than n_reports post-onset positions available -- never fabricated.
                first_timestamp = series[0].timestamps_seconds[0]
                last_timestamp = series[0].timestamps_seconds[-1]
                result = _infer(model, incident["network"], incident["feature_context"], series, library, target_timestamps)
                metrics = _row_metrics(result["probs"], result["node_ids"], truth)
                prediction = _prediction_fields(result["probs"], result["node_ids"])
                #: `actual_elapsed_minutes` is an alias for
                #: `elapsed_since_first_post_onset_report_minutes` (same
                #: value): the shared `_aggregate` closure expects this
                #: exact key name across depth_rows/checkpoint_rows/
                #: post_onset_rows so one aggregation implementation works
                #: for all three row families.
                elapsed_within_window_minutes = (last_timestamp - first_timestamp) / 60.0
                post_onset_rows.append({
                    "incident_key": incident_key, "seed": incident["seed"], "source": truth,
                    "onset_minutes": incident["onset_minutes"], "cadence_minutes": cadence,
                    "n_reports": n_reports, "first_evidence_timestamp": first_timestamp,
                    "last_evidence_timestamp": last_timestamp,
                    "actual_elapsed_minutes": elapsed_within_window_minutes,
                    "elapsed_since_first_post_onset_report_minutes": elapsed_within_window_minutes,
                    "elapsed_since_true_onset_to_first_report_minutes": (first_timestamp - onset_seconds) / 60.0,
                    "elapsed_since_true_onset_to_last_report_minutes": (last_timestamp - onset_seconds) / 60.0,
                    "latency_seconds": result["latency_seconds"], **metrics, **prediction,
                })

    post_onset_paired_by_n: dict[str, Any] = {}
    post_onset_by_cadence_and_n: dict[str, dict[str, Any]] = {}
    for n_reports in POST_ONSET_REPORT_COUNTS:
        rows_by_cadence: dict[int, dict[tuple, dict[str, Any]]] = {}
        for cadence in CADENCES_MINUTES:
            subset = [r for r in post_onset_rows if r["cadence_minutes"] == cadence and r["n_reports"] == n_reports]
            if subset:
                rows_by_cadence[cadence] = {r["incident_key"]: r for r in subset}
                post_onset_by_cadence_and_n.setdefault(str(cadence), {})[str(n_reports)] = _aggregate(subset)
        if len(rows_by_cadence) >= 2:
            post_onset_paired_by_n[str(n_reports)] = _paired_prediction_analysis(rows_by_cadence)

    post_onset_n2 = post_onset_paired_by_n.get("2")
    post_onset_n2_identical_fraction = (
        post_onset_n2.get("fraction_identical_predicted_node_all_cadences") if post_onset_n2 else None
    )
    post_onset_n3 = post_onset_paired_by_n.get("3")
    post_onset_n3_identical_fraction = (
        post_onset_n3.get("fraction_identical_predicted_node_all_cadences") if post_onset_n3 else None
    )
    report_count_conflation_post_onset_n2 = bool(
        post_onset_n2_identical_fraction is not None and post_onset_n2_identical_fraction >= PAIRED_INVARIANCE_FRACTION_BAR
    )
    report_count_conflation_post_onset_n3 = bool(
        post_onset_n3_identical_fraction is not None and post_onset_n3_identical_fraction >= PAIRED_INVARIANCE_FRACTION_BAR
    )
    #: item 3's authoritative decision signal: genuine report-count/time
    #: conflation on REAL post-onset evidence, not the t=0-anchored
    #: (possibly pre-onset-confounded) depth=2 signal.
    temporal_confound_confirmed = depth2_confound_suspected
    post_onset_trigger = report_count_conflation_post_onset_n2 or report_count_conflation_post_onset_n3

    # ---- Matched-elapsed-time rows (item 1: the off-by-one fix). Built
    # directly from real timestamps via _cadence_evidence_up_to_checkpoint
    # -- never from a depth = checkpoint // cadence formula.
    checkpoint_rows: list[dict[str, Any]] = []
    for incident in incidents:
        truth = incident["source"]
        incident_key = (incident["seed"], incident["source"], incident["onset_minutes"])
        for cadence in CADENCES_MINUTES:
            stride = STRIDE_OF[cadence]
            for checkpoint in ELAPSED_CHECKPOINTS_MIN:
                series = _cadence_evidence_up_to_checkpoint(incident["full_series"], stride=stride, checkpoint_minutes=checkpoint)
                first_timestamp = series[0].timestamps_seconds[0]
                last_timestamp = series[0].timestamps_seconds[-1]
                actual_elapsed_minutes = (last_timestamp - first_timestamp) / 60.0
                report_count = max(len(item.timestamps_seconds) for item in series)
                #: A cadence that has not yet delivered any report beyond
                #: t=0 by this checkpoint has NOT actually reached the
                #: physical boundary being tested -- excluded from the
                #: matched comparison (still recorded, flagged) rather
                #: than silently compared against cadences that did reach it.
                matched_checkpoint = report_count > 1
                assert not matched_checkpoint or abs(actual_elapsed_minutes - checkpoint) <= REPORTING_RESOLUTION_MINUTES, (
                    f"matched-elapsed-time construction misaligned: cadence={cadence}min checkpoint={checkpoint}min "
                    f"produced actual_elapsed_minutes={actual_elapsed_minutes} "
                    f"(tolerance={REPORTING_RESOLUTION_MINUTES}min)"
                )
                result = _infer(model, incident["network"], incident["feature_context"], series, library, target_timestamps)
                metrics = _row_metrics(result["probs"], result["node_ids"], truth)
                prediction = _prediction_fields(result["probs"], result["node_ids"])
                checkpoint_rows.append({
                    "incident_key": incident_key, "seed": incident["seed"], "source": truth,
                    "onset_minutes": incident["onset_minutes"], "cadence_minutes": cadence,
                    "checkpoint_minutes": checkpoint, "report_count": report_count,
                    "first_timestamp": first_timestamp, "last_timestamp": last_timestamp,
                    "actual_elapsed_minutes": actual_elapsed_minutes, "matched_checkpoint": matched_checkpoint,
                    "latency_seconds": result["latency_seconds"], **metrics, **prediction,
                })

    by_cadence_elapsed: dict[str, dict[str, Any]] = {}
    for cadence in CADENCES_MINUTES:
        by_cadence_elapsed[str(cadence)] = {}
        for checkpoint in ELAPSED_CHECKPOINTS_MIN:
            group = [
                r for r in checkpoint_rows
                if r["cadence_minutes"] == cadence and r["checkpoint_minutes"] == checkpoint and r["matched_checkpoint"]
            ]
            if not group:
                by_cadence_elapsed[str(cadence)][str(checkpoint)] = {"n": 0}
                continue
            by_cadence_elapsed[str(cadence)][str(checkpoint)] = {
                "report_count_used": statistics.fmean(r["report_count"] for r in group),
                **_aggregate(group),
            }

    # Core scientific comparison: at a FIXED, VERIFIED-ALIGNED elapsed
    # time, does performance differ materially by cadence? Cadence
    # sensitivity = max spread across cadences of top1 at the same
    # elapsed-time checkpoint. Cross-cadence alignment is asserted before
    # any two cadences are compared at a checkpoint (item 1's explicit
    # "add assertions" requirement).
    cadence_sensitivity: dict[str, Any] = {}
    for checkpoint in ELAPSED_CHECKPOINTS_MIN:
        top1s: dict[str, float] = {}
        actual_elapsed_by_cadence: dict[str, float] = {}
        for cadence in CADENCES_MINUTES:
            entry = by_cadence_elapsed[str(cadence)].get(str(checkpoint))
            if entry and entry.get("n", 0) > 0:
                top1s[str(cadence)] = entry["top1"]
                actual_elapsed_by_cadence[str(cadence)] = entry["actual_elapsed_minutes"]
        if len(actual_elapsed_by_cadence) >= 2:
            spread_minutes = max(actual_elapsed_by_cadence.values()) - min(actual_elapsed_by_cadence.values())
            assert spread_minutes <= REPORTING_RESOLUTION_MINUTES, (
                f"matched-elapsed-time groups at checkpoint={checkpoint}min are NOT genuinely aligned: "
                f"actual elapsed minutes by cadence = {actual_elapsed_by_cadence} "
                f"(spread={spread_minutes}min, tolerance={REPORTING_RESOLUTION_MINUTES}min)"
            )
        if len(top1s) >= 2:
            spread_pp = (max(top1s.values()) - min(top1s.values())) * 100
            cadence_sensitivity[str(checkpoint)] = {
                "top1_by_cadence": top1s, "actual_elapsed_minutes_by_cadence": actual_elapsed_by_cadence,
                "spread_pp": spread_pp,
            }

    max_spread_pp = max((entry["spread_pp"] for entry in cadence_sensitivity.values()), default=0.0)
    #: Predeclared bar (mirrors Milestone 1's own 10pp "meaningful gain" bar,
    #: applied here to "meaningful sensitivity" instead -- the same
    #: magnitude convention used throughout this v5 protocol for "a real
    #: effect, not noise").
    STRONG_CADENCE_SENSITIVITY_BAR_PP = 10.0
    strongly_cadence_sensitive = max_spread_pp >= STRONG_CADENCE_SENSITIVITY_BAR_PP

    return {
        "schema_version": 2,
        "correction_note": (
            "Milestone-6 correction (this revision): fixed an off-by-one in the matched-elapsed-time analysis -- "
            "N reports spaced `cadence` apart span (N-1)*cadence minutes of elapsed time, not N*cadence (a "
            "SensorSeries's own first index is t=0). Matched-elapsed-time evidence is now constructed directly "
            "from real timestamps via _cadence_evidence_up_to_checkpoint (timestamp <= first_timestamp + "
            "checkpoint) rather than a depth=checkpoint//cadence formula, with explicit per-row and cross-cadence "
            "alignment assertions. Fixed-report-count elapsed-time labels now report actual_elapsed_minutes read "
            "from the resulting series rather than depth*cadence. depth=1 (zero elapsed span at every cadence) is "
            "no longer used as evidence of report-count/time conflation -- see depth1_elapsed_time_caveat. Added "
            "paired per-incident predicted-node identity analysis (paired_prediction_analysis_by_depth) to "
            "distinguish 'aggregate top-1 accuracy is equal' from 'predictions are identical'. Predictor, "
            "calibration, alpha, K, incident pool, topology, and seeds are all UNCHANGED from the prior revision."
        ),
        "final_validity_check_note": (
            "Milestone-6 final validity check (this revision): the incident pool's onset bins (0/60/120/240 "
            "minutes) mean the t=0-anchored depth=2 evidence window (max elapsed span 60 minutes across all "
            "cadences) is wholly or mostly PRE-onset for 3 of 4 onset strata -- so the pooled "
            "report_count_conflation_at_depth2 finding could reflect trivial invariance on no-signal evidence "
            "rather than genuine cadence-invariant behavior on real contamination evidence. Added (1) "
            "onset_stratified_paired_by_depth (the same paired analysis, split by true onset -- never fed to "
            "the model, used only to select which real reports to evaluate) and (2) a post-onset-anchored "
            "diagnostic (post_onset_rows/post_onset_paired_by_n: the first N=2/3 reports at or after TRUE onset "
            "per cadence, onset again used only for evidence selection, never as a model input). The FINAL 6.5 "
            "decision is now driven by the post-onset evidence (report_count_conflation_post_onset_n2/n3), not "
            "the potentially-confounded pooled depth=2 signal -- see temporal_confound_confirmed."
        ),
        "purpose": "Milestone 6.1: telemetry cadence -- report-count vs elapsed-physical-time curves.",
        "practicality_note": CADENCE_PRACTICALITY_NOTE,
        "cadences_minutes": list(CADENCES_MINUTES),
        "report_counts": list(REPORT_COUNTS),
        "elapsed_checkpoints_minutes": list(ELAPSED_CHECKPOINTS_MIN),
        "reporting_resolution_minutes": REPORTING_RESOLUTION_MINUTES,
        "n_incidents": len(incidents),
        "by_cadence_and_report_count": by_cadence_depth,
        "by_cadence_and_elapsed_time": by_cadence_elapsed,
        "cadence_sensitivity_at_matched_elapsed_time": cadence_sensitivity,
        "max_cadence_sensitivity_spread_pp": max_spread_pp,
        "strong_cadence_sensitivity_bar_pp": STRONG_CADENCE_SENSITIVITY_BAR_PP,
        "strongly_cadence_sensitive": strongly_cadence_sensitive,
        "report_count_invariance_at_fixed_depth": report_count_invariance,
        "depth1_elapsed_time_caveat": DEPTH1_ELAPSED_TIME_CAVEAT,
        "paired_prediction_analysis_by_depth": paired_by_depth,
        "paired_invariance_fraction_bar": PAIRED_INVARIANCE_FRACTION_BAR,
        "report_count_conflation_at_depth2": report_count_conflation_at_depth2,
        "onset_stratified_paired_by_depth": onset_stratified_paired_by_depth,
        "onset0_depth2_identical_fraction": onset0_depth2_identical_fraction,
        "depth2_confound_suspected": depth2_confound_suspected,
        "post_onset_report_counts": list(POST_ONSET_REPORT_COUNTS),
        "post_onset_by_cadence_and_n": post_onset_by_cadence_and_n,
        "post_onset_paired_by_n": post_onset_paired_by_n,
        "post_onset_n2_identical_fraction": post_onset_n2_identical_fraction,
        "post_onset_n3_identical_fraction": post_onset_n3_identical_fraction,
        "report_count_conflation_post_onset_n2": report_count_conflation_post_onset_n2,
        "report_count_conflation_post_onset_n3": report_count_conflation_post_onset_n3,
        "temporal_confound_confirmed": temporal_confound_confirmed,
        "post_onset_trigger": post_onset_trigger,
        "depth_rows": depth_rows,
        "checkpoint_rows": checkpoint_rows,
        "post_onset_rows": post_onset_rows,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }


# ---------------------------------------------------------------------------
# 6.2 Detection delay
# ---------------------------------------------------------------------------


def run_detection_delay_experiment(model, library, target_timestamps, incidents) -> dict[str, Any]:
    rows = []
    for incident in incidents:
        truth = incident["source"]
        onset_seconds = incident["onset_minutes"] * 60.0
        for delay in DETECTION_DELAYS_MIN:
            cutoff = onset_seconds + delay * 60.0
            series = [_truncate_series_by_time(item, cutoff_seconds=cutoff) for item in incident["full_series"]]
            series = [item for item in series if item is not None]
            if not series:
                continue
            n_reports = max(len(item.timestamps_seconds) for item in series)
            result = _infer(model, incident["network"], incident["feature_context"], series, library, target_timestamps)
            metrics = _row_metrics(result["probs"], result["node_ids"], truth)
            rows.append({
                "seed": incident["seed"], "source": truth, "onset_minutes": incident["onset_minutes"],
                "delay_minutes": delay, "n_reports_available": n_reports, **metrics,
            })

    by_delay: dict[str, Any] = {}
    for delay in DETECTION_DELAYS_MIN:
        group = [r for r in rows if r["delay_minutes"] == delay]
        by_delay[str(delay)] = {
            "n": len(group),
            **({
                metric: statistics.fmean(row[metric] for row in group)
                for metric in ("top1", "top3", "mrr", "nll", "true_source_rank")
            } if group else {}),
            "mean_reports_available": statistics.fmean(row["n_reports_available"] for row in group) if group else None,
        }

    #: Structural confirmation (not a statistic over rows): this script's
    #: own model input never reads manifest.incident.start_minute -- the
    #: only per-example inputs are `series` (real observed
    #: timestamps/concentrations) and the static network/graph. Verified
    #: by inspecting `_infer`'s call graph in this file: HydraulicFeatureBuilder.build
    #: never receives `incident` or `onset` as an argument.
    onset_not_fed_as_input = True

    return {
        "schema_version": 1,
        "purpose": "Milestone 6.2: detection delay -- analysis start offset from true contamination onset.",
        "delays_minutes": list(DETECTION_DELAYS_MIN),
        "delay_semantics": "near onset=0, then +1h/+3h/+6h; model sees exactly the reports with timestamp <= onset+delay",
        "n_incidents": len(incidents),
        "onset_never_fed_as_model_input": onset_not_fed_as_input,
        "by_delay": by_delay,
        "rows": rows,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }


# ---------------------------------------------------------------------------
# 6.3 Irregular / asynchronous telemetry (development-only stress cases)
# ---------------------------------------------------------------------------


def _clean_reference_series(incident, *, depth: int) -> list[SensorSeries]:
    stride = STRIDE_OF[REFERENCE_CADENCE_FOR_DELAY_AND_STRESS]
    return [_resample_series(item, stride=stride, depth=depth) for item in incident["full_series"]]


def _stress_timestamp_jitter(series: list[SensorSeries], *, rng: np.random.Generator, jitter_seconds: float = 300.0) -> list[SensorSeries]:
    """Perturbs each report's own timestamp (never reorders -- clamps to
    stay monotonic, matching SensorSeries's own ordering invariant).
    Production's `GeneratedScenario.timestamp_jitter_seconds` field is
    recorded by the generator but never actually applied to reported
    timestamps anywhere in the codebase (confirmed by inspection: no
    consumer adds it to `timestamps_seconds`) -- this is therefore this
    experiment's own explicit jitter injection, not a reuse of an existing
    (currently dead) mechanism."""

    out = []
    for item in series:
        jittered = list(item.timestamps_seconds)
        for i in range(len(jittered)):
            delta = float(rng.normal(0.0, jitter_seconds))
            lower = jittered[i - 1] if i > 0 else 0.0
            upper = jittered[i + 1] if i + 1 < len(jittered) else jittered[i] + jitter_seconds * 10
            jittered[i] = min(max(jittered[i] + delta, lower), upper)
        out.append(SensorSeries(
            node_id=item.node_id, timestamps_seconds=tuple(jittered), concentration_mg_l=item.concentration_mg_l,
            pressure_m=item.pressure_m, health=item.health, missing=item.missing, drift=item.drift,
            delayed=item.delayed, frozen=item.frozen,
        ))
    return out


def _stress_unequal_intervals(series: list[SensorSeries], *, rng: np.random.Generator) -> list[SensorSeries]:
    """Each sensor independently gets its own stride (1, 2, or 3x the
    reference cadence) -- real deployments rarely have every sensor
    report on the exact same schedule."""

    out = []
    for item in series:
        stride = int(rng.integers(1, 4))
        indices = list(range(0, len(item.timestamps_seconds), stride)) or [0]
        out.append(SensorSeries(
            node_id=item.node_id,
            timestamps_seconds=tuple(item.timestamps_seconds[i] for i in indices),
            concentration_mg_l=tuple(item.concentration_mg_l[i] for i in indices),
            pressure_m=tuple(item.pressure_m[i] for i in indices),
            health=tuple(item.health[i] for i in indices),
            missing=tuple(item.missing[i] for i in indices),
            drift=tuple(item.drift[i] for i in indices),
            delayed=tuple(item.delayed[i] for i in indices),
            frozen=tuple(item.frozen[i] for i in indices),
        ))
    return out


def _stress_delayed_reports(series: list[SensorSeries], *, rng: np.random.Generator, delay_seconds: float = 1800.0) -> list[SensorSeries]:
    """One randomly chosen sensor's most recent report is marked
    `delayed=True` and its timestamp pushed back -- a real communication-
    delay case (existing `delayed` schema field, already consumed by
    `HydraulicFeatureBuilder.build`'s `temporal_features` column 5)."""

    if not series:
        return series
    index = int(rng.integers(0, len(series)))
    out = list(series)
    item = out[index]
    timestamps = list(item.timestamps_seconds)
    timestamps[-1] = max(timestamps[-1] - delay_seconds, timestamps[-2] if len(timestamps) > 1 else 0.0)
    delayed = list(item.delayed)
    delayed[-1] = True
    out[index] = SensorSeries(
        node_id=item.node_id, timestamps_seconds=tuple(timestamps), concentration_mg_l=item.concentration_mg_l,
        pressure_m=item.pressure_m, health=item.health, missing=item.missing, drift=item.drift,
        delayed=tuple(delayed), frozen=item.frozen,
    )
    return out


def _stress_gaps(series: list[SensorSeries], *, rng: np.random.Generator, drop_probability: float = 0.3) -> list[SensorSeries]:
    """Forces extra reports missing (existing `missing` schema field) --
    never invents a future report, only hides an already-generated one."""

    out = []
    for item in series:
        missing = list(item.missing)
        for i in range(len(missing)):
            if rng.random() < drop_probability:
                missing[i] = True
        out.append(SensorSeries(
            node_id=item.node_id, timestamps_seconds=item.timestamps_seconds, concentration_mg_l=item.concentration_mg_l,
            pressure_m=item.pressure_m, health=item.health, missing=tuple(missing), drift=item.drift,
            delayed=item.delayed, frozen=item.frozen,
        ))
    return out


def _stress_partial_availability(series: list[SensorSeries], *, rng: np.random.Generator) -> list[SensorSeries]:
    """One randomly chosen sensor only starts reporting partway through
    this evidence window (e.g. a sensor commissioned mid-incident) -- its
    EARLY reports are dropped entirely (shorter history than its peers),
    not marked missing. HydraulicFeatureBuilder.build already supports
    per-sensor asynchronous time coverage (confirmed: `selected_times`
    unions per-series timestamps and independently looks up each
    (time, node) pair), so no new feature-alignment code is needed."""

    if len(series) < 2:
        return series
    index = int(rng.integers(0, len(series)))
    out = list(series)
    item = out[index]
    n = len(item.timestamps_seconds)
    if n < 2:
        return series
    start = int(rng.integers(1, n))
    sl = slice(start, n)
    out[index] = SensorSeries(
        node_id=item.node_id, timestamps_seconds=item.timestamps_seconds[sl], concentration_mg_l=item.concentration_mg_l[sl],
        pressure_m=item.pressure_m[sl], health=item.health[sl], missing=item.missing[sl], drift=item.drift[sl],
        delayed=item.delayed[sl], frozen=item.frozen[sl] if item.frozen else (),
    )
    return out


STRESS_CASES = {
    "TIMESTAMP_JITTER": _stress_timestamp_jitter,
    "UNEQUAL_SENSOR_INTERVALS": _stress_unequal_intervals,
    "DELAYED_REPORTS": _stress_delayed_reports,
    "GAPS": _stress_gaps,
    "PARTIAL_HISTORICAL_AVAILABILITY": _stress_partial_availability,
}


def run_irregular_telemetry_experiment(model, library, target_timestamps, incidents, *, n_incidents: int = 24, depth: int = 6) -> dict[str, Any]:
    subset = incidents[:n_incidents]
    rows = []
    for incident in subset:
        truth = incident["source"]
        clean_series = _clean_reference_series(incident, depth=depth)
        clean_result = _infer(model, incident["network"], incident["feature_context"], clean_series, library, target_timestamps)
        clean_metrics = _row_metrics(clean_result["probs"], clean_result["node_ids"], truth)
        rows.append({"stress": "CLEAN_BASELINE", "seed": incident["seed"], "source": truth, **clean_metrics})
        for stress_name, stress_fn in STRESS_CASES.items():
            rng = _seeded_rng("m6-stress", stress_name, incident["seed"])
            stressed_series = stress_fn(clean_series, rng=rng)
            result = _infer(model, incident["network"], incident["feature_context"], stressed_series, library, target_timestamps)
            metrics = _row_metrics(result["probs"], result["node_ids"], truth)
            rows.append({"stress": stress_name, "seed": incident["seed"], "source": truth, **metrics})

    by_stress: dict[str, Any] = {}
    for stress_name in ["CLEAN_BASELINE", *STRESS_CASES]:
        group = [r for r in rows if r["stress"] == stress_name]
        by_stress[stress_name] = {
            "n": len(group),
            **{metric: statistics.fmean(row[metric] for row in group) for metric in ("top1", "top3", "mrr", "nll", "true_source_rank")},
        }
    baseline_top1 = by_stress["CLEAN_BASELINE"]["top1"]
    degradation_pp = {
        stress_name: (baseline_top1 - by_stress[stress_name]["top1"]) * 100
        for stress_name in STRESS_CASES
    }

    return {
        "schema_version": 1,
        "purpose": (
            "Milestone 6.3: development-only irregular/asynchronous telemetry stress cases. Diagnostic only "
            "-- never used for any promotion decision (matches experiments.txt 6.3's own 'development-only' framing)."
        ),
        "reference_cadence_minutes": REFERENCE_CADENCE_FOR_DELAY_AND_STRESS,
        "reference_depth_reports": depth,
        "n_incidents": len(subset),
        "by_stress_case": by_stress,
        "top1_degradation_pp_vs_clean": degradation_pp,
        "rows": rows,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }


# ---------------------------------------------------------------------------
# 6.4 Sensor coverage and placement
# ---------------------------------------------------------------------------


def _degree_centrality_placement(feature_context, junctions: tuple[str, ...], k: int) -> tuple[str, ...]:
    scored = sorted(junctions, key=lambda node: (-feature_context.graph.nodes[node].get("demand_centrality", 0.0), node))
    return tuple(sorted(scored[:k]))


def _hydraulic_observability_placement(library, junctions: tuple[str, ...], k: int, *, depth_rows: int) -> tuple[str, ...]:
    """The sensor subset of size k minimizing mean pairwise source-
    signature correlation (restricted to those columns and the first
    `depth_rows` template rows), i.e. maximizing classical physics-based
    separability among source hypotheses -- purely a function of the
    static network/signature library, never any specific incident's
    realized noise. Golden-reference network has only 4 junctions, so an
    exact brute-force search over C(4,k) subsets is used (no heuristic
    needed)."""

    positions = {node: index for index, node in enumerate(library.node_ids)}
    best_subset, best_score = None, None
    for subset in itertools.combinations(junctions, k):
        columns = [positions[node] for node in subset]
        pairwise = []
        for a, b in itertools.combinations(junctions, 2):
            sig_a = np.asarray(library.signatures[a][:depth_rows, columns], dtype=np.float64).ravel()
            sig_b = np.asarray(library.signatures[b][:depth_rows, columns], dtype=np.float64).ravel()
            valid = np.isfinite(sig_a) & np.isfinite(sig_b)
            if valid.sum() < 2 or np.std(sig_a[valid]) < 1e-9 or np.std(sig_b[valid]) < 1e-9:
                continue
            correlation = float(np.corrcoef(sig_a[valid], sig_b[valid])[0, 1])
            if np.isfinite(correlation):
                pairwise.append(correlation)
        score = statistics.fmean(pairwise) if pairwise else 1.0  # worst case (fully correlated / uninformative)
        if best_score is None or score < best_score:
            best_score, best_subset = score, subset
    return tuple(sorted(best_subset))


def _random_placement(rng: np.random.Generator, junctions: tuple[str, ...], k: int) -> tuple[str, ...]:
    return tuple(sorted(map(str, rng.choice(junctions, size=k, replace=False))))


PLACEMENT_POLICIES = ("RANDOM", "DEGREE_CENTRALITY", "HYDRAULIC_OBSERVABILITY")
SENSOR_BUDGETS = (1, 2, 3)  # of 4 junctions -> 25% / 50% / 75% coverage.
PLACEMENT_DEPTH = 3  # EARLY bucket, matching DEPTH_BUCKET_OF; "initial" evaluation point.
N_PLACEMENT_INCIDENTS = 12


def _placement_measurement(network, source: str, node: str, *, decision_seconds: float, rng: np.random.Generator, noise_std: float = 0.01) -> SensorSeries:
    exact = HydraulicSimulator(network).simulate_incident(source, strength_mg_min=10.0, start_minute=0, duration_minutes=60)
    timestamps = np.asarray(exact.concentration_mg_l.index, dtype=float)
    index = int(np.argmin(np.abs(timestamps - decision_seconds)))
    values = exact.concentration_mg_l.loc[:, node].to_numpy(dtype=float)[: index + 1]
    noisy = np.maximum(0.0, values + rng.normal(0.0, noise_std, size=values.shape))
    return SensorSeries(
        node_id=node, timestamps_seconds=tuple(timestamps[: index + 1].tolist()),
        concentration_mg_l=tuple(map(float, noisy)), pressure_m=tuple(25.0 for _ in noisy),
        health=tuple(1.0 for _ in noisy), missing=tuple(False for _ in noisy),
        drift=tuple(False for _ in noisy), delayed=tuple(False for _ in noisy), frozen=tuple(False for _ in noisy),
    )


def run_sensor_placement_experiment(model, library, target_timestamps, network, feature_context, junctions: tuple[str, ...], calibrator) -> dict[str, Any]:
    onset_bins = (0, 60, 120, 240)
    incident_specs = []
    for index in range(N_PLACEMENT_INCIDENTS):
        source = junctions[index % len(junctions)]
        onset = onset_bins[index % len(onset_bins)]
        seed_material = f"m6-placement:{index}"
        seed = 970_000_000 + int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "big") % 10_000_000
        incident_specs.append({"index": index, "source": source, "onset_minutes": onset, "seed": seed})

    decision_seconds = PLACEMENT_DEPTH * 3600.0  # 3 hourly reports.
    rows = []
    for budget in SENSOR_BUDGETS:
        placements = {
            "DEGREE_CENTRALITY": _degree_centrality_placement(feature_context, junctions, budget),
            "HYDRAULIC_OBSERVABILITY": _hydraulic_observability_placement(library, junctions, budget, depth_rows=PLACEMENT_DEPTH),
        }
        for spec in incident_specs:
            rng_random_placement = _seeded_rng("m6-placement-random", spec["seed"])
            placements["RANDOM"] = _random_placement(rng_random_placement, junctions, budget)
            for policy in PLACEMENT_POLICIES:
                sensor_nodes = placements[policy]
                rng_noise = _seeded_rng("m6-placement-noise", policy, spec["seed"])
                series = [
                    _placement_measurement(network, spec["source"], node, decision_seconds=decision_seconds, rng=rng_noise)
                    for node in sensor_nodes
                ]
                result = _infer(model, network, feature_context, series, library, target_timestamps)
                metrics = _row_metrics(result["probs"], result["node_ids"], spec["source"])
                candidates = _candidate_set(calibrator, result["probs"], condition=result["condition"], bucket=DEPTH_BUCKET_OF[PLACEMENT_DEPTH])
                candidate_gate_pass = 1 <= len(candidates) <= K_MAX_CANDIDATES

                # Greedy "samples required": add the next unplaced junction
                # per this SAME policy's own ranking (random order for
                # RANDOM), re-evaluating the real model each round, until
                # the candidate gate passes or every junction is placed.
                remaining = [node for node in junctions if node not in sensor_nodes]
                if policy == "RANDOM":
                    rng_order = _seeded_rng("m6-placement-random-order", spec["seed"])
                    order = list(map(str, rng_order.permutation(remaining))) if remaining else []
                elif policy == "DEGREE_CENTRALITY":
                    order = sorted(remaining, key=lambda node: (-feature_context.graph.nodes[node].get("demand_centrality", 0.0), node))
                else:
                    order = []
                    pool = list(remaining)
                    current = list(sensor_nodes)
                    while pool:
                        best_node, best_score = None, None
                        for candidate_node in pool:
                            trial = tuple(sorted((*current, candidate_node)))
                            # Reuse the same objective inline (avoids recomputing full combinatorics for one fixed extra node).
                            positions = {node: idx for idx, node in enumerate(library.node_ids)}
                            columns = [positions[node] for node in trial]
                            pairwise = []
                            for a, b in itertools.combinations(junctions, 2):
                                sig_a = np.asarray(library.signatures[a][:PLACEMENT_DEPTH, columns], dtype=np.float64).ravel()
                                sig_b = np.asarray(library.signatures[b][:PLACEMENT_DEPTH, columns], dtype=np.float64).ravel()
                                valid = np.isfinite(sig_a) & np.isfinite(sig_b)
                                if valid.sum() < 2 or np.std(sig_a[valid]) < 1e-9 or np.std(sig_b[valid]) < 1e-9:
                                    continue
                                correlation = float(np.corrcoef(sig_a[valid], sig_b[valid])[0, 1])
                                if np.isfinite(correlation):
                                    pairwise.append(correlation)
                            trial_score = statistics.fmean(pairwise) if pairwise else 1.0
                            if best_score is None or trial_score < best_score:
                                best_score, best_node = trial_score, candidate_node
                        order.append(best_node)
                        current.append(best_node)
                        pool.remove(best_node)

                samples_used = None
                current_nodes = list(sensor_nodes)
                gate_pass_now = candidate_gate_pass
                for round_index, next_node in enumerate(order, start=1):
                    if gate_pass_now:
                        samples_used = round_index - 1
                        break
                    current_nodes.append(next_node)
                    rng_round_noise = _seeded_rng("m6-placement-noise", policy, spec["seed"], round_index)
                    round_series = [
                        _placement_measurement(network, spec["source"], node, decision_seconds=decision_seconds, rng=rng_round_noise)
                        for node in current_nodes
                    ]
                    round_result = _infer(model, network, feature_context, round_series, library, target_timestamps)
                    round_candidates = _candidate_set(calibrator, round_result["probs"], condition=round_result["condition"], bucket=DEPTH_BUCKET_OF[PLACEMENT_DEPTH])
                    gate_pass_now = 1 <= len(round_candidates) <= K_MAX_CANDIDATES
                if samples_used is None:
                    samples_used = len(order) if gate_pass_now else None

                # Identifiability restricted to this placement's sensor set (Milestone 1.2-style diagnostic).
                positions = {node: idx for idx, node in enumerate(library.node_ids)}
                columns = [positions[node] for node in sensor_nodes]
                pairwise = []
                for a, b in itertools.combinations(junctions, 2):
                    sig_a = np.asarray(library.signatures[a][:PLACEMENT_DEPTH, columns], dtype=np.float64).ravel()
                    sig_b = np.asarray(library.signatures[b][:PLACEMENT_DEPTH, columns], dtype=np.float64).ravel()
                    valid = np.isfinite(sig_a) & np.isfinite(sig_b)
                    if valid.sum() < 2 or np.std(sig_a[valid]) < 1e-9 or np.std(sig_b[valid]) < 1e-9:
                        continue
                    correlation = float(np.corrcoef(sig_a[valid], sig_b[valid])[0, 1])
                    if np.isfinite(correlation):
                        pairwise.append(correlation)
                mean_pairwise_correlation = statistics.fmean(pairwise) if pairwise else None

                rows.append({
                    "budget": budget, "policy": policy, "sensor_nodes": list(sensor_nodes),
                    "incident_index": spec["index"], "source": spec["source"], "onset_minutes": spec["onset_minutes"],
                    "seed": spec["seed"], "candidate_set_size": len(candidates), "candidate_gate_pass": candidate_gate_pass,
                    "samples_required": samples_used, "mean_pairwise_signature_correlation": mean_pairwise_correlation,
                    **metrics,
                })

    def _aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        resolved = [row["samples_required"] for row in group if row["samples_required"] is not None]
        return {
            "n": len(group),
            "top1": statistics.fmean(row["top1"] for row in group),
            "top3": statistics.fmean(row["top3"] for row in group),
            "mrr": statistics.fmean(row["mrr"] for row in group),
            "mean_candidate_set_size": statistics.fmean(row["candidate_set_size"] for row in group),
            "candidate_gate_pass_rate": statistics.fmean(row["candidate_gate_pass"] for row in group),
            "median_samples_required": statistics.median(resolved) if resolved else None,
            "never_resolved_fraction": statistics.fmean(row["samples_required"] is None for row in group),
            "mean_pairwise_signature_correlation": statistics.fmean(
                row["mean_pairwise_signature_correlation"] for row in group if row["mean_pairwise_signature_correlation"] is not None
            ) if any(row["mean_pairwise_signature_correlation"] is not None for row in group) else None,
        }

    by_budget_policy: dict[str, dict[str, Any]] = {}
    for budget in SENSOR_BUDGETS:
        by_budget_policy[str(budget)] = {
            policy: _aggregate([r for r in rows if r["budget"] == budget and r["policy"] == policy])
            for policy in PLACEMENT_POLICIES
        }

    return {
        "schema_version": 1,
        "purpose": "Milestone 6.4: sensor coverage and placement -- RANDOM vs DEGREE_CENTRALITY vs HYDRAULIC_OBSERVABILITY.",
        "sensor_budgets": list(SENSOR_BUDGETS),
        "n_junctions": len(junctions),
        "placement_evidence_depth_reports": PLACEMENT_DEPTH,
        "placement_evidence_bucket": DEPTH_BUCKET_OF[PLACEMENT_DEPTH],
        "k_max_candidates": K_MAX_CANDIDATES,
        "n_incidents_per_budget": N_PLACEMENT_INCIDENTS,
        "by_budget_and_policy": by_budget_policy,
        "rows": rows,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(*, rerun_all: bool = False) -> int:
    """Milestone-6 correction (item 5/6): by default (`rerun_all=False`)
    this is a NARROW cadence-analysis correction -- only 6.1 is
    recomputed; 6.2 (detection delay), 6.3 (irregular telemetry), and 6.4
    (sensor placement) are byte-for-byte scientifically unaffected by the
    matched-elapsed-time/report-count bug fixed here (verified by
    inspection: `run_detection_delay_experiment` and
    `run_sensor_placement_experiment` already derive their own timing from
    real timestamps -- `_truncate_series_by_time`'s `cutoff` and
    `_placement_measurement`'s `decision_seconds` -- never from a
    depth*cadence-style formula; `run_irregular_telemetry_experiment` uses
    `_resample_series` at one FIXED depth and never labels an elapsed-time
    value at all), so their existing committed JSON artifacts are loaded
    from disk unchanged rather than rerun. Pass `rerun_all=True` to force
    a full rerun of 6.2-6.4 as well (e.g. to double check preservation is
    still valid after an unrelated change) -- not used by default."""

    assert not locked_test_opened(ROOT), "locked test must remain closed"

    model, export_path, predictor_description = _load_frozen_predictor()
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)
    target_timestamps = build_sensor_series(train_records[0].scenario, train_records[0].feature_context)[0].timestamps_seconds

    junctions = tuple(sorted(train_records[0].network.junction_name_list))

    print(f"predictor: {predictor_description} ({export_path})")
    print("generating shared cadence/delay/stress incident pool...")
    incidents = _generate_cadence_incident_pool(junctions)
    print(f"  {len(incidents)} incidents generated")

    print("running 6.1 cadence experiment (CORRECTED this revision)...")
    cadence_report = run_cadence_experiment(model, library, target_timestamps, incidents)
    OUT_CADENCE.parent.mkdir(parents=True, exist_ok=True)
    OUT_CADENCE.write_text(json.dumps(cadence_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"  max cadence-sensitivity spread: {cadence_report['max_cadence_sensitivity_spread_pp']:.2f}pp; "
          f"strongly_cadence_sensitive={cadence_report['strongly_cadence_sensitive']}")

    if rerun_all:
        print("running 6.2 detection-delay experiment (--rerun-all)...")
        delay_report = run_detection_delay_experiment(model, library, target_timestamps, incidents)
        OUT_DELAY.write_text(json.dumps(delay_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        print("running 6.3 irregular-telemetry stress experiment (--rerun-all)...")
        irregular_report = run_irregular_telemetry_experiment(model, library, target_timestamps, incidents)
        OUT_IRREGULAR.write_text(json.dumps(irregular_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        print("running 6.4 sensor-placement experiment (--rerun-all)...")
        calibration_records = build_scenario_pool("calibration", network_loader=build_wntr_network)
        calibrator = _fit_frozen_calibrator(model, library, calibration_records)
        reference_network = train_records[0].network
        reference_context = train_records[0].feature_context
        placement_report = run_sensor_placement_experiment(
            model, library, target_timestamps, reference_network, reference_context, junctions, calibrator,
        )
        OUT_PLACEMENT.write_text(json.dumps(placement_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    else:
        print("6.2/6.3/6.4 PRESERVED (loaded from existing committed artifacts, not rerun -- see main()'s docstring).")
        delay_report = json.loads(OUT_DELAY.read_text())
        irregular_report = json.loads(OUT_IRREGULAR.read_text())
        placement_report = json.loads(OUT_PLACEMENT.read_text())

    # 6.5 conditional architecture decision, driven by 6.1's own CORRECTED
    # measured results (item 4: do not assume the prior decision remains
    # correct -- recompute from scratch). Milestone-6 final validity check:
    # the pooled t=0-anchored depth=2 signal (B_pooled below) can be
    # confounded by pre-onset (no-signal) incidents -- see
    # cadence_report["final_validity_check_note"] -- so the AUTHORITATIVE
    # decision signal is now the post-onset-anchored diagnostic (C), not
    # B_pooled. Three signals reported:
    #   A) corrected matched-physical-time results (verified-aligned).
    #   B_pooled) t=0-anchored fixed-report-count paired invariance,
    #      reported for continuity but explicitly NOT authoritative if
    #      temporal_confound_confirmed.
    #   C) post-onset-anchored paired invariance (N=2/3 real reports after
    #      TRUE onset, onset used only to select evidence, never fed to
    #      the model) -- this drives the final decision.
    strongly_sensitive = cadence_report["strongly_cadence_sensitive"]
    pooled_count_conflation = cadence_report["report_count_conflation_at_depth2"]
    confound_confirmed = cadence_report["temporal_confound_confirmed"]
    post_onset_trigger = cadence_report["post_onset_trigger"]
    trigger = strongly_sensitive or post_onset_trigger
    depth2_paired = cadence_report["paired_prediction_analysis_by_depth"].get("2")
    depth2_identical_fraction = (
        depth2_paired.get("fraction_identical_predicted_node_all_cadences") if depth2_paired else None
    )
    onset0_fraction = cadence_report["onset0_depth2_identical_fraction"]
    n2_fraction = cadence_report["post_onset_n2_identical_fraction"]
    n3_fraction = cadence_report["post_onset_n3_identical_fraction"]

    finding_a = (
        f"(A) Corrected matched-physical-time result: max cadence-sensitivity spread at a VERIFIED-ALIGNED "
        f"elapsed-time boundary is {cadence_report['max_cadence_sensitivity_spread_pp']:.2f}pp "
        f"(bar: {cadence_report['strong_cadence_sensitivity_bar_pp']:.0f}pp) -- "
        + ("material cadence-dependent behavior remains a measured problem." if strongly_sensitive
           else "below the predeclared bar; no material cadence-dependent behavior at matched physical time.")
    )
    finding_b_pooled = (
        f"(B_pooled, NOT authoritative -- see C) t=0-anchored depth=2 paired identity: "
        f"{'n/a' if depth2_identical_fraction is None else f'{depth2_identical_fraction:.3f}'} pooled across all "
        f"onset bins (bar {cadence_report['paired_invariance_fraction_bar']:.2f}), vs "
        f"{'n/a' if onset0_fraction is None else f'{onset0_fraction:.3f}'} for the onset=0 stratum ALONE (the "
        "only stratum guaranteed real post-onset signal at every cadence at depth=2) -- "
        + (
            "the pooled figure met the bar while the onset=0-only figure did NOT, confirming the pooled number "
            "is inflated by pre-onset (no-signal) incidents rather than reflecting genuine time-invariant "
            "behavior on real evidence; this pooled claim is RETRACTED as evidence of report-count/time "
            "conflation." if confound_confirmed else
            "onset stratification does not overturn the pooled figure; no confound detected in this signal."
        )
    )
    finding_c = (
        f"(C, AUTHORITATIVE) Post-onset-anchored paired identity: at N=2 real post-onset reports, "
        f"{'n/a' if n2_fraction is None else f'{n2_fraction:.3f}'} of matched incidents predict the identical "
        f"node across all cadences (bar {cadence_report['paired_invariance_fraction_bar']:.2f}); at N=3, "
        f"{'n/a' if n3_fraction is None else f'{n3_fraction:.3f}'}. "
        + ("Substantial invariance PERSISTS even when contamination evidence is genuinely present, confirming "
           "real report-count/time conflation -- not a pre-onset artifact." if post_onset_trigger else
           "Invariance does NOT persist once evidence is anchored to true onset -- the original report-count/"
           "time conflation claim does not hold up on real contamination evidence.")
    )

    architecture_note = {
        "existing_temporal_representation": (
            "hydroswarm.model.encoders.TemporalEncoder already encodes masked histories using elapsed "
            "timestamps rather than array position (its own docstring); HydraulicFeatureBuilder.build already "
            "feeds real per-timestep age (elapsed seconds since the latest observation) into temporal_features "
            "and quality_features, plus real absolute timestamps into the batch. An explicit elapsed-time "
            "representation already exists in HydroCore's architecture -- confirmed by inspection, not assumed. "
            "This remains true regardless of the corrected cadence finding below: HydroCore already has "
            "elapsed-time-aware temporal features, so even if the problem remains, the recommended next "
            "experiment continues to be cadence-diversified causal-prefix training and/or a controlled "
            "timestamp-conditioning ablation at matched model size -- never a new temporal architecture."
        ),
        "cadence_sensitivity_measured_pp": cadence_report["max_cadence_sensitivity_spread_pp"],
        "bar_pp": cadence_report["strong_cadence_sensitivity_bar_pp"],
        "strongly_cadence_sensitive": strongly_sensitive,
        "report_count_conflation_at_depth2_pooled": pooled_count_conflation,
        "temporal_confound_confirmed": confound_confirmed,
        "report_count_conflation_post_onset_n2": cadence_report["report_count_conflation_post_onset_n2"],
        "report_count_conflation_post_onset_n3": cadence_report["report_count_conflation_post_onset_n3"],
        "paired_invariance_fraction_bar": cadence_report["paired_invariance_fraction_bar"],
        "depth1_elapsed_time_caveat": cadence_report["depth1_elapsed_time_caveat"],
        "decision": (
            "NEW_TEMPORAL_REPRESENTATION_EXPERIMENT_WARRANTED" if trigger else
            "NEW_TEMPORAL_REPRESENTATION_EXPERIMENT_NOT_TRIGGERED"
        ),
        "rationale": (
            f"{finding_a} {finding_b_pooled} {finding_c} "
            + (
                (
                    "Since HydroCore's TemporalEncoder already encodes elapsed timestamps (not array position) "
                    "yet the post-onset-anchored evidence (C) still shows genuine report-count/time conflation, "
                    "the existing representation's GENERALIZATION -- not its presence -- is implicated: "
                    "train_records/calibration_records/M1's causal-prefix corpus all use a fixed hourly "
                    "reporting cadence, so the model has never actually been trained on report sequences where "
                    "consecutive reports span 15 or 30 minutes rather than 60. The correctly targeted follow-up "
                    "is therefore a training-distribution diversification (cadence-varied causal-prefix corpus, "
                    "i.e. re-running Milestone 1's arm construction with randomized inter-report spacing) and/or "
                    "a controlled ablation of the TemporalEncoder's timestamp-conditioning at matched model size "
                    "-- NOT a new architecture built from scratch. Not executed in this correction (a full "
                    "retrain is explicitly out of scope); recorded here as the concrete next milestone "
                    "recommendation, per experiments.txt's own decision tree ('M6: is cadence/delay robustness "
                    "poor? YES -> test explicit time encoding')."
                ) if trigger else
                (
                    "The post-onset-anchored diagnostic (C) -- the authoritative signal for this decision -- "
                    "did not show substantial paired-prediction invariance despite materially different elapsed "
                    "durations, and matched-physical-time sensitivity (A) is below its bar. Per experiments.txt "
                    "Milestone 6.5 ('do not add continuous-time complexity unless the cadence study demonstrates "
                    "the need') and Milestone 9's identical restatement, no new architecture experiment is "
                    "warranted. The prior revision's WARRANTED conclusion (driven by the pooled, pre-onset-"
                    "confounded depth=2 signal) is RETRACTED, not preserved: once anchored to true onset, the "
                    "apparent report-count/time conflation does not hold up on real contamination evidence."
                )
            )
        ),
    }

    summary_lines = _render_summary(
        predictor_description, export_path, cadence_report, delay_report, irregular_report, placement_report, architecture_note,
    )
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))

    assert not locked_test_opened(ROOT), "locked test must remain closed after M6"
    return 0


def _render_summary(predictor_description, export_path, cadence_report, delay_report, irregular_report, placement_report, architecture_note) -> list[str]:
    lines = [
        "# Milestone 6 summary: temporal realism -- cadence, detection delay, sensor observability",
        "",
        f"Predictor: {predictor_description} ({export_path})",
        "Calibration: B_DEPTH_AWARE (Milestone 3 frozen scheme, refit identically here).",
        "",
        "## 6.1 Telemetry cadence",
        "",
        "**Milestone-6 correction applied this revision** (see `correction_note` in m6-cadence.json for the "
        "full text): the matched-elapsed-time analysis had an off-by-one -- N reports spaced `cadence` apart "
        "span (N-1)*cadence minutes of elapsed time, not N*cadence. Matched-elapsed-time evidence is now built "
        "directly from real timestamps (`timestamp <= first_timestamp + checkpoint`) with explicit per-row and "
        "cross-cadence alignment assertions, never from a `depth = checkpoint // cadence` formula. "
        "Fixed-report-count elapsed-time labels now report the ACTUAL elapsed minutes read from the resulting "
        "series, never `depth * cadence`. Paired per-incident predicted-node identity analysis was added to "
        "distinguish 'aggregate top-1 accuracy is equal' from 'predictions are identical'. Predictor, "
        "calibration, alpha, K, incident pool, topology, and seeds are unchanged from the prior revision.",
        "",
        f"Practicality: used cadences {cadence_report['cadences_minutes']} min (5 min dropped -- see "
        "`practicality_note` in m6-cadence.json for the measured 180s-timeout-vs-0.04s evidence, unchanged "
        "this revision).",
        "",
        "### Fixed PHYSICAL elapsed-time comparison (corrected)",
        "",
        f"N incidents: {cadence_report['n_incidents']}. Reporting resolution (alignment tolerance): "
        f"{cadence_report['reporting_resolution_minutes']:.0f} min. Every row below passed a per-row "
        "actual-elapsed-vs-checkpoint assertion AND a cross-cadence alignment assertion before being compared "
        "(script raises, does not silently compare, on misalignment). Max cadence-sensitivity spread at a "
        f"VERIFIED-ALIGNED elapsed-time boundary: **{cadence_report['max_cadence_sensitivity_spread_pp']:.2f}pp** "
        f"(bar: {cadence_report['strong_cadence_sensitivity_bar_pp']:.0f}pp). "
        f"Strongly cadence-sensitive: **{cadence_report['strongly_cadence_sensitive']}**.",
        "",
        "| checkpoint (min) | actual elapsed @15min | actual elapsed @30min | actual elapsed @60min | top1 @15min | top1 @30min | top1 @60min | spread (pp) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for checkpoint, entry in sorted(cadence_report["cadence_sensitivity_at_matched_elapsed_time"].items(), key=lambda kv: int(kv[0])):
        t = entry["top1_by_cadence"]
        e = entry["actual_elapsed_minutes_by_cadence"]

        def _fmt(d, key):
            v = d.get(key)
            return "-" if v is None else f"{v:.1f}"

        lines.append(
            f"| {checkpoint} | {_fmt(e, '15')} | {_fmt(e, '30')} | {_fmt(e, '60')} | "
            f"{t.get('15', float('nan')):.3f} | {t.get('30', float('nan')):.3f} | {t.get('60', float('nan')):.3f} | "
            f"{entry['spread_pp']:.2f} |"
        )
    lines += [
        "",
        "### Fixed REPORT-COUNT comparison (corrected elapsed labels)",
        "",
        cadence_report["depth1_elapsed_time_caveat"],
        "",
        "| depth (reports) | actual elapsed @15min | actual elapsed @30min | actual elapsed @60min | top1 @15min | top1 @30min | top1 @60min | spread (pp) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for depth, entry in sorted(cadence_report["report_count_invariance_at_fixed_depth"].items(), key=lambda kv: int(kv[0])):
        t = entry["top1_by_cadence"]
        e = entry["actual_elapsed_minutes_by_cadence"]

        def _fmt2(d, key):
            v = d.get(key)
            return "-" if v is None else f"{v:.1f}"

        note = "  (depth=1: NOT evidence -- see caveat above)" if depth == "1" else ""
        lines.append(
            f"| {depth}{note} | {_fmt2(e, '15')} | {_fmt2(e, '30')} | {_fmt2(e, '60')} | "
            f"{t.get('15', float('nan')):.3f} | {t.get('30', float('nan')):.3f} | {t.get('60', float('nan')):.3f} | "
            f"{entry['spread_pp']:.2f} |"
        )
    lines += [
        "",
        "### Paired prediction-identity analysis (fixed report count, depth >= 2)",
        "",
        "Aggregate top-1 accuracy equal does NOT by itself imply 'predictions are identical' -- this table "
        "verifies per-incident predicted-node identity across cadences directly. L1 = sum of absolute "
        "differences between the two probability vectors over the same node ordering.",
        "",
        f"**report_count_conflation_at_depth2 (paired, bar={cadence_report['paired_invariance_fraction_bar']:.2f}): "
        f"{cadence_report['report_count_conflation_at_depth2']}**",
        "",
        "| depth | n matched | fraction identical predicted_node (all cadences) | 15v30 agreement | 15v60 agreement | 30v60 agreement | mean L1 dist | max L1 dist |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for depth, entry in sorted(cadence_report["paired_prediction_analysis_by_depth"].items(), key=lambda kv: int(kv[0])):
        pw = entry["pairwise_top1_agreement"]

        def _agree(key):
            v = pw.get(key)
            return "-" if not v else f"{v['agreement_fraction']:.3f} (n={v['n']})"

        frac = entry["fraction_identical_predicted_node_all_cadences"]
        mean_l1 = entry["mean_l1_probability_distance"]
        max_l1 = entry["max_l1_probability_distance"]
        lines.append(
            f"| {depth} | {entry['n_incidents_matched_across_all_cadences']} | "
            f"{'-' if frac is None else f'{frac:.3f}'} | {_agree('15v30')} | {_agree('15v60')} | {_agree('30v60')} | "
            f"{'-' if mean_l1 is None else f'{mean_l1:.4f}'} | {'-' if max_l1 is None else f'{max_l1:.4f}'} |"
        )
    pooled_depth2_fraction = cadence_report["paired_prediction_analysis_by_depth"].get("2", {}).get(
        "fraction_identical_predicted_node_all_cadences"
    )
    onset0_depth2_fraction = cadence_report["onset0_depth2_identical_fraction"]
    lines += [
        "",
        "### Milestone-6 final validity check: onset-stratified paired analysis (potential pre-onset confound)",
        "",
        "The incident pool's onset bins are 0/60/120/240 minutes; at depth=2 the max elapsed span (60 min) means "
        "only onset=0 incidents are guaranteed real post-onset evidence at EVERY cadence. Splitting the pooled "
        "paired analysis by true onset (onset used only to select which real incidents fall in which row here -- "
        "never fed to the model) checks whether pooled invariance survives when contamination is actually "
        "observable.",
        "",
        f"**depth2_confound_suspected: {cadence_report['depth2_confound_suspected']}** "
        f"(pooled depth=2 fraction identical: {'n/a' if pooled_depth2_fraction is None else f'{pooled_depth2_fraction:.3f}'}; "
        f"onset=0-only depth=2 fraction identical: {'n/a' if onset0_depth2_fraction is None else f'{onset0_depth2_fraction:.3f}'})",
        "",
        "| depth | onset (min) | n matched | fraction identical predicted_node | mean L1 dist | top1 @15min | top1 @30min | top1 @60min |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for depth, by_onset in sorted(cadence_report["onset_stratified_paired_by_depth"].items(), key=lambda kv: int(kv[0])):
        for onset, entry in sorted(by_onset.items(), key=lambda kv: int(kv[0])):
            frac = entry["fraction_identical_predicted_node_all_cadences"]
            mean_l1 = entry["mean_l1_probability_distance"]
            t1 = entry.get("top1_by_cadence", {})
            lines.append(
                f"| {depth} | {onset} | {entry['n_incidents_matched_across_all_cadences']} | "
                f"{'-' if frac is None else f'{frac:.3f}'} | {'-' if mean_l1 is None else f'{mean_l1:.4f}'} | "
                f"{t1.get('15', float('nan')):.3f} | {t1.get('30', float('nan')):.3f} | {t1.get('60', float('nan')):.3f} |"
            )
    lines += [
        "",
        "### Milestone-6 final validity check: post-onset-anchored diagnostic (AUTHORITATIVE for the 6.5 decision)",
        "",
        "Evidence = the first N real reports AT OR AFTER true contamination onset, per cadence (onset used only "
        "to select which reports to include; never written into any model input -- see "
        "`_cadence_evidence_since_onset`'s docstring). Compares the SAME incidents across cadences.",
        "",
        f"**report_count_conflation_post_onset_n2: {cadence_report['report_count_conflation_post_onset_n2']}** "
        f"(bar={cadence_report['paired_invariance_fraction_bar']:.2f}); "
        f"**report_count_conflation_post_onset_n3: {cadence_report['report_count_conflation_post_onset_n3']}**",
        "",
        "| N post-onset reports | n matched | fraction identical predicted_node (all cadences) | 15v30 agreement | 15v60 agreement | 30v60 agreement | mean L1 dist | max L1 dist | top1 @15min | top1 @30min | top1 @60min |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n_reports, entry in sorted(cadence_report["post_onset_paired_by_n"].items(), key=lambda kv: int(kv[0])):
        pw = entry["pairwise_top1_agreement"]

        def _agree2(key):
            v = pw.get(key)
            return "-" if not v else f"{v['agreement_fraction']:.3f} (n={v['n']})"

        frac = entry["fraction_identical_predicted_node_all_cadences"]
        mean_l1 = entry["mean_l1_probability_distance"]
        max_l1 = entry["max_l1_probability_distance"]
        t1 = entry.get("top1_by_cadence", {})
        lines.append(
            f"| {n_reports} | {entry['n_incidents_matched_across_all_cadences']} | "
            f"{'-' if frac is None else f'{frac:.3f}'} | {_agree2('15v30')} | {_agree2('15v60')} | {_agree2('30v60')} | "
            f"{'-' if mean_l1 is None else f'{mean_l1:.4f}'} | {'-' if max_l1 is None else f'{max_l1:.4f}'} | "
            f"{t1.get('15', float('nan')):.3f} | {t1.get('30', float('nan')):.3f} | {t1.get('60', float('nan')):.3f} |"
        )
    lines += [
        "",
        "## 6.2 Detection delay",
        "",
        f"N incidents: {delay_report['n_incidents']}. Onset never fed as model input: "
        f"**{delay_report['onset_never_fed_as_model_input']}**.",
        "",
        "| delay (min) | n | top1 | top3 | mrr | mean reports available |",
        "|---|---|---|---|---|---|",
    ]
    for delay in sorted(delay_report["by_delay"], key=int):
        entry = delay_report["by_delay"][delay]
        if entry["n"] == 0:
            continue
        lines.append(
            f"| {delay} | {entry['n']} | {entry['top1']:.3f} | {entry['top3']:.3f} | {entry['mrr']:.3f} | "
            f"{entry['mean_reports_available']:.2f} |"
        )
    lines += [
        "",
        "## 6.3 Irregular/asynchronous telemetry (development-only, diagnostic)",
        "",
        f"N incidents: {irregular_report['n_incidents']} at {irregular_report['reference_cadence_minutes']}min "
        f"cadence, depth={irregular_report['reference_depth_reports']} reports.",
        "",
        "| stress case | n | top1 | top1 degradation vs clean (pp) |",
        "|---|---|---|---|",
    ]
    for stress_name, entry in irregular_report["by_stress_case"].items():
        degradation = irregular_report["top1_degradation_pp_vs_clean"].get(stress_name)
        lines.append(f"| {stress_name} | {entry['n']} | {entry['top1']:.3f} | "
                      f"{'-' if degradation is None else f'{degradation:.2f}'} |")
    lines += [
        "",
        "## 6.4 Sensor coverage and placement",
        "",
        f"N junctions: {placement_report['n_junctions']}. Evidence depth: {placement_report['placement_evidence_depth_reports']} "
        f"reports ({placement_report['placement_evidence_bucket']} bucket). K={placement_report['k_max_candidates']}.",
        "",
        "| budget | policy | n | top1 | top3 | mean candidate size | gate-pass rate | median samples required |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for budget in sorted(placement_report["by_budget_and_policy"], key=int):
        for policy, entry in placement_report["by_budget_and_policy"][budget].items():
            if entry["n"] == 0:
                continue
            lines.append(
                f"| {budget} | {policy} | {entry['n']} | {entry['top1']:.3f} | {entry['top3']:.3f} | "
                f"{entry['mean_candidate_set_size']:.2f} | {entry['candidate_gate_pass_rate']:.3f} | "
                f"{'-' if entry['median_samples_required'] is None else entry['median_samples_required']} |"
            )
    lines += [
        "",
        "## 6.5 Conditional architecture change",
        "",
        f"**Decision: {architecture_note['decision']}**",
        "",
        architecture_note["existing_temporal_representation"],
        "",
        architecture_note["rationale"],
        "",
        "## Limitations",
        "",
        "6.1/6.2/6.3 use a single canonical (golden-reference, 4-junction) topology with full sensor coverage "
        "(placement isolated to 6.4); cadence/delay behavior on larger or differently-shaped networks was not "
        "tested here (topology diversity is Milestone 7's scope). 6.4's HYDRAULIC_OBSERVABILITY placement and "
        "identifiability diagnostic are network-structural (no incident-specific information), matching "
        "Milestone 1.2's identifiability baseline's own scope limits. 6.3's stress cases are development-only "
        "diagnostics per the milestone text and are not used to select or tune anything.",
        "",
        "## Milestone-6 correction scope",
        "",
        "This revision recomputes ONLY 6.1 (telemetry cadence): a matched-elapsed-time off-by-one and a "
        "fixed-report-count elapsed-time mislabeling, both described above. 6.2 (detection delay), 6.3 "
        "(irregular telemetry), and 6.4 (sensor placement) are PRESERVED byte-for-byte from the prior revision "
        "-- their own timing/elapsed-time constructions were verified by inspection to already derive from real "
        "timestamps (never a depth*cadence-style formula), so no direct implementation inconsistency from the "
        "6.1 fix carries over to them; they were not rerun. The frozen Milestone-1 predictor, Milestone-3 "
        "B_DEPTH_AWARE calibrator, alpha=0.1, K=3, topology, incident pool construction, and random seeds are "
        "all unchanged from the prior revision throughout this correction.",
    ]
    return lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun-all", action="store_true",
        help="Also rerun 6.2/6.3/6.4 instead of loading their preserved artifacts (see main()'s docstring).",
    )
    args = parser.parse_args()
    raise SystemExit(main(rerun_all=args.rerun_all))
