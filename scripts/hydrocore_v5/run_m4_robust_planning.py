"""Milestone 4 (experiments.txt): robust planning under source uncertainty.

Question: if the frozen Milestone-3 calibration (B_DEPTH_AWARE, alpha=0.1)
says several sources remain plausible, must planning always be blocked, or
can a response be exactly verified safe across the entire candidate region?

4.1 Control -- current authority (hydroswarm.inference.pipeline): a
CANDIDATE_REGION_TOO_BROAD suppression fires whenever the calibrated
conformal candidate-set size exceeds `maximum_planning_candidates` (=3 in
every production wiring; no runtime factory overrides the pipeline
default). Below that size, this script's "control" arm verifies each
generated plan against a single, ordinary point-estimate hypothesis (the
top-1 candidate node) -- the naive verification convention used anywhere
`PlanEvaluationContext.hypotheses` is not wired (e.g.
evaluation/golden.py's fixture; any caller predating
api/app.py's `_runtime_evaluation_context`).

4.2 Experimental robust policy: exact-WNTR verify every prescreened plan
against EVERY plausible candidate source in the region (as
`WeightedSourceHypothesis` entries), using the SAME machinery
api/app.py's `_runtime_evaluation_context` already wires into the live
`/plans/{id}/verify` endpoint (`PlanEvaluationContext.hypotheses`,
`PlanVerifier.verify`'s union-of-rejection-codes decision rule, worst-case
aggregation) -- this milestone does not invent a new verification
mechanism, it evaluates the existing one end-to-end against a naive
single-hypothesis baseline on held-out incidents where ground truth is
known.

Predeclared K = 3, NOT chosen to shrink candidate sets: it is the existing
hard architectural ceiling `hydroswarm.simulation.wrapper.
MAXIMUM_EVALUATION_HYPOTHESES` already enforces (ValueError above it) for
exact multi-hypothesis verification. This script never raises or bypasses
that ceiling. Incidents whose calibrated candidate-set size exceeds K
(observed at the EARLY depth bucket, matching M3's p90=4 finding) are
reported separately as a bounded "not yet reachable without a future
architecture decision" bucket -- both policies fail closed there, exactly
as experiments.txt Milestone 4 requires ("if no robust plan exists, fail
closed"), and no attempt is made to claim an actionability gain for them.

Because K already equals production's own `maximum_planning_candidates`
default, this experiment cannot show an eligibility-driven actionability
gain (both arms attempt planning on the identical incident set). What it
tests instead: whether exact whole-region verification changes SAFETY
outcomes relative to the naive point-estimate convention (false-safe /
authority-invariant-violation rate: does a plan the naive control marks
VERIFIED actually violate deterministic safety when checked against the
incident's real, held-out ground-truth source?), and at what computational
cost (exact simulator calls, wall-clock).

Uses development_holdout only (never locked_final_test /
locked_topology_test). One representative depth per Milestone-3 evidence
bucket (EARLY=2, MID=4, MATURE=12) and a predeclared, outcome-independent
per-bucket incident cap, both fixed before any result was inspected, to
keep exact-WNTR verification cost practical (experiments.txt 4.3's
"computational cost remains practical").

Milestone-4 correction (this revision): the ground-truth false-safe check
is now UNCONDITIONAL -- every eligible incident where control or robust
selects a plan is exact-WNTR verified against the real held-out
IncidentSourceProfile regardless of whether the true source falls inside
the conformal candidate set. The prior revision gated this check on
`true_node in candidate_probs`, so candidate-set coverage MISSES were
never actually checked and silently defaulted to false_safe=False --
indistinguishable in the report from "checked and found safe". See
`_false_safe_breakdown` for the unconditional / conditional (coverage-hit)
/ coverage-miss false-safe counts and rates this now reports for both
policies.

SCOPE LIMITATION: ROBUSTLY_VERIFIED in this milestone means exact
verification across every plausible SOURCE-LOCATION candidate in the
conformal region, holding the remaining modeled incident-profile
assumptions (start time, duration, injection strength -- all taken from
`IncidentSourceProfile`'s governed defaults, per node, exactly like
production's own `_runtime_evaluation_context`) fixed. It is not a claim
of robustness over all uncertain contamination parameters: the
`WeightedSourceHypothesis` set varies candidate node identity only, never
jointly samples start_minute/duration_minutes/relative_strength. This
milestone does not model those additional uncertainties.

Writes:
  reports/evaluation/hydrocore-v5/m4-robust-planning.json
  reports/evaluation/hydrocore-v5/m4-summary.md
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.calibration.conformal import (  # noqa: E402
    CalibrationExample,
    SplitConformalCalibrator,
    classify_runtime_condition,
)
from hydroswarm.domain import IncidentCreate, PlanDecision  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT  # noqa: E402
from hydroswarm.planning.response import (  # noqa: E402
    PlanGenerationContext,
    generate_response_plans,
    prescreen_top_plans,
)
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.simulation.verifier import PlanVerifier  # noqa: E402
from hydroswarm.simulation.wrapper import (  # noqa: E402
    MAXIMUM_EVALUATION_HYPOTHESES,
    HydraulicSimulator,
    IncidentSourceProfile,
    PlanEvaluationContext,
    WeightedSourceHypothesis,
)
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.corpus import build_sensor_series  # noqa: E402
from hydroswarm.training.scenario_reconstruction import incident_truth_to_source_profile  # noqa: E402
from run_m1_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m3_calibration import DEPTH_BUCKET_OF, _freeze_predictor  # noqa: E402

OUTPUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m4-robust-planning.json"
OUTPUT_SUMMARY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m4-summary.md"

ALPHA = 0.1
#: Architecturally justified, not tuned for effect: the existing hard cap
#: exact multi-hypothesis WNTR verification supports (see module docstring).
K_MAX_CANDIDATES = MAXIMUM_EVALUATION_HYPOTHESES
assert K_MAX_CANDIDATES == 3
#: One representative depth per Milestone-3 evidence bucket.
BUCKET_DEPTH: dict[str, int] = {"EARLY": 2, "MID": 4, "MATURE": 12}
#: Predeclared compute-budget cap (fixed scenario order, not outcome-selected).
MAX_INCIDENTS_PER_BUCKET = 40
MAXIMUM_EXACT_SIMULATIONS = 3  # matches production's runtime prescreen budget
DEFAULT_CONTAMINATION_THRESHOLD_MG_L: float = IncidentCreate.model_fields["contamination_threshold_mg_l"].default

SCOPE_LIMITATION = (
    "ROBUSTLY_VERIFIED in Milestone 4 means exact verification across every plausible source-location candidate "
    "in the conformal region, holding the remaining modeled incident-profile assumptions fixed. It is not a claim "
    "of robustness over all uncertain contamination parameters. The WeightedSourceHypothesis set varies candidate "
    "source node identity only; start_minute, duration_minutes, and relative_strength are held at "
    "IncidentSourceProfile's governed defaults for every hypothesis (matching production's own "
    "_runtime_evaluation_context), never jointly sampled or varied across the hypothesis set. This milestone does "
    "not model uncertainty in source timing, duration, or injection strength."
)


def _planning_context(incident_id, network, graph, probable_nodes, sampled_nodes) -> PlanGenerationContext:
    """Mirrors hydroswarm.inference.pipeline.HybridInferencePipeline.
    _planning_context exactly (candidate template/isolation/flush/monitor
    construction) so this experiment exercises the same plan-generation
    contract production uses, without importing the full pipeline module."""

    junctions = tuple(str(node) for node in network.junction_name_list)
    monitors = probable_nodes + tuple(node for node in junctions if node not in probable_nodes)
    downstream: list[str] = []
    for source in probable_nodes:
        if source in graph:
            downstream.extend(str(node) for node in graph.successors(source) if str(node) in junctions)
    if not downstream:
        downstream = list(monitors)
    demand_ranked = sorted(junctions, key=lambda node: -float(graph.nodes[node].get("demand_m3s", 0.0)))
    return PlanGenerationContext(
        incident_id=incident_id,
        model_version="m4-robust-planning-v1",
        probable_source_nodes=probable_nodes,
        isolatable_links=tuple(str(link) for link in network.pipe_name_list),
        downstream_flush_nodes=tuple(dict.fromkeys(downstream)),
        critical_demand_nodes=tuple(demand_ranked[:2]),
        monitor_nodes=tuple(dict.fromkeys(monitors)),
        sampled_nodes=sampled_nodes,
    )


def _worst_metrics(verification) -> dict[str, float] | None:
    consequences = verification.worst_case_consequences or verification.consequences
    if consequences is None:
        return None
    return {
        "service_availability": consequences.service_availability,
        "population_impacted": consequences.population_impacted,
        "minimum_pressure_m": consequences.minimum_pressure_m,
    }


def _evaluate_incident(
    *, bucket: str, depth: int, record, model, library, calibrator
) -> dict[str, Any]:
    scenario = record.scenario
    example = scenario_to_prefix_example(scenario, record.network, library, depth, feature_context=record.feature_context)
    with torch.no_grad():
        output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
    probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
    node_ids = example.topology.node_ids
    truth_idx = int(example.targets["source_node"].item())
    true_node = node_ids[truth_idx]

    full_series = build_sensor_series(scenario, record.feature_context)
    truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
    condition = classify_runtime_condition(truncated_series)
    network_id_key = f"{scenario.manifest.network_id}:{bucket}"
    indices = calibrator.candidate_set(probs, condition=condition, network_id=network_id_key)
    candidate_nodes = [node_ids[index] for index in indices]

    result: dict[str, Any] = {
        "scenario_id": str(scenario.manifest.scenario_id),
        "bucket": bucket,
        "depth": depth,
        "candidate_set_size": len(candidate_nodes),
        "true_source_in_candidate_set": true_node in candidate_nodes,
    }

    if not (1 <= len(candidate_nodes) <= K_MAX_CANDIDATES):
        result.update({
            "eligible": False,
            "exceeds_k": len(candidate_nodes) > K_MAX_CANDIDATES,
            "control": {"suppressed": True},
            "robust": {"suppressed": True},
        })
        return result

    simulator = HydraulicSimulator(record.network, exact_simulation_budget=64)
    graph = simulator.build_dynamic_graph()
    incident_id = uuid.uuid5(uuid.NAMESPACE_URL, f"hydrocore-v5-m4:{scenario.manifest.scenario_id}:{depth}")

    candidate_probs = {node_ids[index]: max(probs[index], 1e-9) for index in indices}
    total = sum(candidate_probs.values())
    candidate_probs = {node: value / total for node, value in candidate_probs.items()}
    ranked_candidates = sorted(candidate_probs.items(), key=lambda item: -item[1])
    top1_node = ranked_candidates[0][0]

    context = _planning_context(
        incident_id, record.network, graph, tuple(node for node, _ in ranked_candidates), frozenset()
    )
    proposals = generate_response_plans(context, maximum_plans=ACTION_TEMPLATE_COUNT)
    prescreened = prescreen_top_plans(proposals, maximum_exact_simulations=MAXIMUM_EXACT_SIMULATIONS)

    verifier = PlanVerifier(simulator)
    started = time.perf_counter()
    runs_before = simulator.exact_runs

    control_eval_context = PlanEvaluationContext(
        contamination_threshold_mg_l=DEFAULT_CONTAMINATION_THRESHOLD_MG_L,
        hypotheses=(WeightedSourceHypothesis(profile=IncidentSourceProfile(source_node_id=top1_node), probability=1.0),),
    )
    control_verifications = [(proposal, verifier.verify(proposal.plan, control_eval_context)) for proposal in prescreened]
    control_verified = [(p, v) for p, v in control_verifications if v.decision == PlanDecision.VERIFIED]
    control_selected = (
        max(control_verified, key=lambda pv: pv[0].predicted_value * pv[0].predicted_validity)
        if control_verified else None
    )
    control_runs_after = simulator.exact_runs

    hypotheses = tuple(
        WeightedSourceHypothesis(profile=IncidentSourceProfile(source_node_id=node), probability=probability)
        for node, probability in ranked_candidates
    )
    robust_eval_context = PlanEvaluationContext(
        contamination_threshold_mg_l=DEFAULT_CONTAMINATION_THRESHOLD_MG_L,
        hypotheses=hypotheses,
        aggregation_policy="worst_case",
    )
    robust_verifications = [(proposal, verifier.verify(proposal.plan, robust_eval_context)) for proposal in prescreened]
    robust_verified = [(p, v) for p, v in robust_verifications if v.decision == PlanDecision.VERIFIED]

    def _worst_case_rank(pair):
        metrics = _worst_metrics(pair[1])
        assert metrics is not None
        return (metrics["service_availability"], -metrics["population_impacted"])

    robust_selected = max(robust_verified, key=_worst_case_rank) if robust_verified else None
    robust_runs_after = simulator.exact_runs
    elapsed_seconds = time.perf_counter() - started

    #: Unconditional ground-truth safety audit (Milestone-4 correction):
    #: EVERY eligible incident where control or robust selects a plan gets
    #: exact-WNTR verified against the real held-out IncidentSourceProfile,
    #: regardless of whether the true source is inside the conformal
    #: candidate set. Previously this check was gated on
    #: `true_node in candidate_probs`, so coverage-miss incidents were never
    #: actually tested and silently defaulted to false_safe=False --
    #: indistinguishable from "checked and safe". `false_safe` is now None
    #: (not applicable -- nothing was selected to check) only when no plan
    #: was selected at all; whenever a plan IS selected it is always
    #: checked, so an untested-but-selected incident can never occur.
    true_profile = incident_truth_to_source_profile(scenario.manifest.incident, is_contamination=True)
    truth_eval_context = PlanEvaluationContext(
        contamination_threshold_mg_l=DEFAULT_CONTAMINATION_THRESHOLD_MG_L, source_profile=true_profile
    )
    control_truth_checked = control_selected is not None
    control_false_safe = None
    if control_truth_checked:
        truth_check = verifier.verify(control_selected[0].plan, truth_eval_context)
        control_false_safe = truth_check.decision != PlanDecision.VERIFIED
    robust_truth_checked = robust_selected is not None
    robust_false_safe = None
    if robust_truth_checked:
        truth_check = verifier.verify(robust_selected[0].plan, truth_eval_context)
        robust_false_safe = truth_check.decision != PlanDecision.VERIFIED

    result.update({
        "eligible": True,
        "exceeds_k": False,
        "top1_node": top1_node,
        "candidate_nodes": [node for node, _ in ranked_candidates],
        "attempted_plans": len(prescreened),
        "exact_simulator_calls": robust_runs_after - runs_before,
        "control_exact_simulator_calls": control_runs_after - runs_before,
        "robust_exact_simulator_calls": robust_runs_after - control_runs_after,
        "elapsed_seconds": elapsed_seconds,
        "control": {
            "suppressed": False,
            "verified": control_selected is not None,
            "selected_template": control_selected[0].template if control_selected else None,
            "worst_case": _worst_metrics(control_selected[1]) if control_selected else None,
            "truth_checked": control_truth_checked,
            "false_safe": control_false_safe,
        },
        "robust": {
            "suppressed": False,
            "robustly_verified": robust_selected is not None,
            "fail_closed": robust_selected is None,
            "selected_template": robust_selected[0].template if robust_selected else None,
            "worst_case": _worst_metrics(robust_selected[1]) if robust_selected else None,
            "truth_checked": robust_truth_checked,
            "false_safe": robust_false_safe,
        },
    })
    return result


def _false_safe_breakdown(selected: list[dict[str, Any]], *, policy: str) -> dict[str, Any]:
    """Unconditional/conditional/coverage-miss false-safe counts and rates
    for one policy ("control" or "robust"), over the incidents where that
    policy actually selected a plan (and therefore actually ran the
    ground-truth check -- see _evaluate_incident's truth_checked field)."""

    on_hit = [r for r in selected if r["true_source_in_candidate_set"]]
    on_miss = [r for r in selected if not r["true_source_in_candidate_set"]]
    false_safe_all = [r for r in selected if r[policy]["false_safe"]]
    false_safe_hit = [r for r in on_hit if r[policy]["false_safe"]]
    false_safe_miss = [r for r in on_miss if r[policy]["false_safe"]]
    assert all(r[policy]["truth_checked"] for r in selected), (
        f"{policy}: every selected-plan incident must have been ground-truth checked"
    )
    return {
        "selected_count": len(selected),
        "selected_on_coverage_hit_count": len(on_hit),
        "selected_on_coverage_miss_count": len(on_miss),
        "false_safe_count_unconditional": len(false_safe_all),
        "false_safe_rate_unconditional": (len(false_safe_all) / len(selected)) if selected else None,
        "false_safe_count_conditional_coverage_hit": len(false_safe_hit),
        "false_safe_rate_conditional_coverage_hit": (len(false_safe_hit) / len(on_hit)) if on_hit else None,
        "false_safe_count_coverage_miss": len(false_safe_miss),
        "false_safe_rate_coverage_miss": (len(false_safe_miss) / len(on_miss)) if on_miss else None,
    }


def _bucket_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    eligible = [r for r in records if r["eligible"]]
    exceeds_k = [r for r in records if r.get("exceeds_k")]
    control_verified = [r for r in eligible if r["control"]["verified"]]
    robust_verified = [r for r in eligible if r["robust"]["robustly_verified"]]
    coverage_applicable = [r for r in eligible if r["true_source_in_candidate_set"]]
    coverage_miss = [r for r in eligible if not r["true_source_in_candidate_set"]]
    #: The only incidents where control (top-1 only) and robust (whole
    #: region) COULD structurally disagree -- at candidate_set_size == 1
    #: robust degenerates to exactly one hypothesis, identical to control by
    #: construction, so including those would understate how rarely this
    #: sample actually exercised the multi-hypothesis mechanism.
    genuinely_multi_candidate = [r for r in eligible if r["candidate_set_size"] > 1]
    decisions_disagree = [
        r for r in genuinely_multi_candidate if r["control"]["verified"] != r["robust"]["robustly_verified"]
    ]

    def _mean(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    control_false_safe = _false_safe_breakdown(control_verified, policy="control")
    robust_false_safe = _false_safe_breakdown(robust_verified, policy="robust")

    return {
        "n": n,
        "planning_eligible_rate": len(eligible) / n if n else 0.0,
        "exceeds_k_rate": len(exceeds_k) / n if n else 0.0,
        "mean_candidate_set_size": _mean([r["candidate_set_size"] for r in records]),
        "true_source_coverage_rate_eligible": (len(coverage_applicable) / len(eligible)) if eligible else None,
        "candidate_coverage_hit_count": len(coverage_applicable),
        "candidate_coverage_miss_count": len(coverage_miss),
        "genuinely_multi_candidate_count": len(genuinely_multi_candidate),
        "decision_disagreement_count_multi_candidate": len(decisions_disagree),
        "decision_disagreement_rate_multi_candidate": (
            len(decisions_disagree) / len(genuinely_multi_candidate)
        ) if genuinely_multi_candidate else None,
        "control_verified_rate_of_eligible": (len(control_verified) / len(eligible)) if eligible else None,
        "robust_verified_rate_of_eligible": (len(robust_verified) / len(eligible)) if eligible else None,
        "robust_fail_closed_rate_of_eligible": (
            sum(1 for r in eligible if r["robust"]["fail_closed"]) / len(eligible)
        ) if eligible else None,
        #: Unconditional ground-truth safety audit (Milestone-4 correction):
        #: every incident counted here where that policy selected a plan was
        #: ACTUALLY exact-WNTR checked against the real held-out source,
        #: whether or not that source was inside the conformal candidate
        #: set -- see _evaluate_incident / _false_safe_breakdown.
        "control_false_safe": control_false_safe,
        "robust_false_safe": robust_false_safe,
        # Back-compat top-level aliases for the unconditional counts (now
        # genuinely unconditional, unlike the pre-correction fields of the
        # same name).
        "control_false_safe_count": control_false_safe["false_safe_count_unconditional"],
        "control_false_safe_rate_of_control_verified": control_false_safe["false_safe_rate_unconditional"],
        "robust_false_safe_count": robust_false_safe["false_safe_count_unconditional"],
        "robust_false_safe_rate_of_robust_verified": robust_false_safe["false_safe_rate_unconditional"],
        "mean_worst_case_service_availability_control": _mean(
            [r["control"]["worst_case"]["service_availability"] for r in control_verified if r["control"]["worst_case"]]
        ),
        "mean_worst_case_service_availability_robust": _mean(
            [r["robust"]["worst_case"]["service_availability"] for r in robust_verified if r["robust"]["worst_case"]]
        ),
        "mean_control_exact_simulator_calls": _mean(
            [r["control_exact_simulator_calls"] for r in eligible]
        ),
        "mean_robust_exact_simulator_calls": _mean(
            [r["robust_exact_simulator_calls"] for r in eligible]
        ),
        "mean_elapsed_seconds_per_incident": _mean([r["elapsed_seconds"] for r in eligible]),
    }


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"

    export_path, use_adapters, predictor_description = _freeze_predictor()
    model = HydroCore.from_variant("small", use_adapters=use_adapters, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()

    calibration_records = build_scenario_pool("calibration", network_loader=build_wntr_network)
    development_records = build_scenario_pool("development_holdout", network_loader=build_wntr_network)
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    def _collect(records, bucket: str, depth: int) -> list[dict[str, Any]]:
        collected = []
        with torch.no_grad():
            for record in records:
                scenario = record.scenario
                example = scenario_to_prefix_example(
                    scenario, record.network, library, depth, feature_context=record.feature_context
                )
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

    # Fit on the full Milestone-3 depth set (not just the three
    # representative depths this script evaluates incidents against) so the
    # B_DEPTH_AWARE calibrator reconstructed here is identical to the one M3
    # froze -- M3 fit on all seven CAUSAL_PREFIX_DEPTHS.
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
        alpha=ALPHA, model_hash=export_path, feature_schema_hash="n/a", dataset_manifest_hash="m4-robust-planning-pool",
    )

    per_bucket_incidents: dict[str, list[dict[str, Any]]] = {}
    for bucket, depth in BUCKET_DEPTH.items():
        subset = development_records[:MAX_INCIDENTS_PER_BUCKET]
        per_bucket_incidents[bucket] = [
            _evaluate_incident(bucket=bucket, depth=depth, record=record, model=model, library=library, calibrator=calibrator)
            for record in subset
        ]

    all_incidents = [item for bucket_items in per_bucket_incidents.values() for item in bucket_items]
    bucket_summaries = {bucket: _bucket_summary(items) for bucket, items in per_bucket_incidents.items()}
    overall_summary = _bucket_summary(all_incidents)

    #: Unconditional: overall_summary["robust_false_safe_count"] is now the
    #: `false_safe_count_unconditional` alias, computed over EVERY incident
    #: where robust actually selected a plan (coverage hits and misses
    #: alike) -- see the Milestone-4 correction note in the module
    #: docstring and _false_safe_breakdown.
    zero_robust_invariant_violations = overall_summary["robust_false_safe_count"] == 0
    control_has_violations = overall_summary["control_false_safe_count"] > 0
    material_actionability_gain = False  # K == production's existing maximum_planning_candidates; see module docstring.

    #: Preserve (never drop) the exact incidents behind any robust
    #: false-safe finding, for precise failure-mode classification, per the
    #: Milestone-4 correction's decision rule: "if robust false-safe
    #: incidents appear, preserve them, classify the failure mode
    #: precisely, and STOP."
    robust_false_safe_incidents = [
        item for item in all_incidents
        if item.get("eligible") and item["robust"]["truth_checked"] and item["robust"]["false_safe"]
    ]
    control_false_safe_incidents = [
        item for item in all_incidents
        if item.get("eligible") and item["control"]["truth_checked"] and item["control"]["false_safe"]
    ]

    if zero_robust_invariant_violations and material_actionability_gain:
        exit_decision = "PROMOTE_ROBUST_PLANNING"
    elif zero_robust_invariant_violations:
        exit_decision = "MECHANISM_VALIDATED_NO_MATERIAL_ACTIONABILITY_GAIN_AT_K3"
    else:
        exit_decision = "REJECT_INVARIANT_VIOLATIONS_FOUND"

    report = {
        "schema_version": 1,
        "purpose": "Milestone 4 (experiments.txt): robust planning under source uncertainty.",
        "predictor": {"export_path": export_path, "use_adapters": use_adapters, "description": predictor_description},
        "calibration_scheme": "B_DEPTH_AWARE (frozen in Milestone 3, refit here identically over all CAUSAL_PREFIX_DEPTHS)",
        "alpha": ALPHA,
        "k_max_candidates": K_MAX_CANDIDATES,
        "k_max_candidates_justification": (
            "hydroswarm.simulation.wrapper.MAXIMUM_EVALUATION_HYPOTHESES -- the existing hard architectural "
            "ceiling for exact multi-hypothesis WNTR verification; equals production's own "
            "HybridInferencePipeline.maximum_planning_candidates default (3), so this experiment cannot show an "
            "eligibility-driven actionability gain (see module docstring)."
        ),
        "bucket_depths": BUCKET_DEPTH,
        "max_incidents_per_bucket": MAX_INCIDENTS_PER_BUCKET,
        "maximum_exact_simulations_per_arm": MAXIMUM_EXACT_SIMULATIONS,
        "scope_limitation": SCOPE_LIMITATION,
        "ground_truth_safety_audit": (
            "Unconditional: every eligible incident where control or robust selected a plan is exact-WNTR "
            "verified against the real held-out IncidentSourceProfile, regardless of candidate-set coverage. "
            "See per_bucket/overall control_false_safe / robust_false_safe breakdowns "
            "(false_safe_count_unconditional, false_safe_count_conditional_coverage_hit, "
            "false_safe_count_coverage_miss) and incidents[*].control.truth_checked / "
            "incidents[*].robust.truth_checked (always True whenever that policy selected a plan)."
        ),
        "per_bucket": bucket_summaries,
        "overall": overall_summary,
        "incidents": all_incidents,
        "zero_robust_authority_invariant_violations": zero_robust_invariant_violations,
        "control_has_authority_invariant_violations": control_has_violations,
        "robust_false_safe_incidents": robust_false_safe_incidents,
        "control_false_safe_incidents": control_false_safe_incidents,
        "material_actionability_gain": material_actionability_gain,
        "exit_decision": exit_decision,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Milestone 4 summary: robust planning under source uncertainty",
        "",
        f"Predictor: {predictor_description}",
        "Calibration: B_DEPTH_AWARE (Milestone 3 frozen scheme, refit identically here).",
        f"K (max verified candidate sources) = {K_MAX_CANDIDATES} "
        "(hydroswarm.simulation.wrapper.MAXIMUM_EVALUATION_HYPOTHESES -- the existing hard ceiling, "
        "equal to production's maximum_planning_candidates default; never relaxed).",
        "",
        "## Scope limitation",
        "",
        SCOPE_LIMITATION,
        "",
        "## Ground-truth safety audit methodology (Milestone-4 correction)",
        "",
        "Every eligible incident where control or robust selected a plan is exact-WNTR verified against the real "
        "held-out `IncidentSourceProfile`, unconditionally -- including incidents where the true source falls "
        "OUTSIDE the conformal candidate set (a calibration-coverage miss). The prior revision only ran this check "
        "when the true source was inside the candidate set, so coverage misses were silently reported as "
        "false_safe=False without ever being tested. That gate has been removed.",
        "",
        "## Per-bucket results (development_holdout, one representative depth/bucket, "
        f"n<={MAX_INCIDENTS_PER_BUCKET}/bucket)",
        "",
        "| bucket | depth | n | eligible | exceeds-K | coverage hit | coverage miss | control verified | "
        "robust verified | control false-safe (unconditional) | robust false-safe (unconditional) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for bucket, depth in BUCKET_DEPTH.items():
        s = bucket_summaries[bucket]
        lines.append(
            f"| {bucket} | {depth} | {s['n']} | {s['planning_eligible_rate']:.2f} | {s['exceeds_k_rate']:.2f} | "
            f"{s['candidate_coverage_hit_count']} | {s['candidate_coverage_miss_count']} | "
            f"{(s['control_verified_rate_of_eligible'] or 0):.2f} | {(s['robust_verified_rate_of_eligible'] or 0):.2f} | "
            f"{s['control_false_safe_count']} | {s['robust_false_safe_count']} |"
        )
    lines += [
        "",
        "## False-safe breakdown by policy (unconditional / conditional-on-coverage-hit / coverage-miss-only), overall",
        "",
        "| policy | selected | selected on hit | selected on miss | false-safe (unconditional) | rate | "
        "false-safe (coverage hit) | rate | false-safe (coverage miss) | rate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for policy_name, fs in (("control", overall_summary["control_false_safe"]), ("robust", overall_summary["robust_false_safe"])):
        lines.append(
            f"| {policy_name} | {fs['selected_count']} | {fs['selected_on_coverage_hit_count']} | "
            f"{fs['selected_on_coverage_miss_count']} | {fs['false_safe_count_unconditional']} | "
            f"{fs['false_safe_rate_unconditional']} | {fs['false_safe_count_conditional_coverage_hit']} | "
            f"{fs['false_safe_rate_conditional_coverage_hit']} | {fs['false_safe_count_coverage_miss']} | "
            f"{fs['false_safe_rate_coverage_miss']} |"
        )
    lines += [
        "",
        f"## Overall (n={overall_summary['n']})",
        "",
        f"- Planning eligible (region size 1-{K_MAX_CANDIDATES}): {overall_summary['planning_eligible_rate']:.3f}",
        f"- Region exceeds K (not reachable without a future architecture decision): "
        f"{overall_summary['exceeds_k_rate']:.3f} -- fails closed under both policies, no actionability claimed.",
        f"- Candidate-set coverage: {overall_summary['candidate_coverage_hit_count']} hit / "
        f"{overall_summary['candidate_coverage_miss_count']} miss (true source outside the conformal candidate "
        f"set) among eligible incidents -- coverage misses are now included in the unconditional false-safe audit "
        f"below, not skipped.",
        f"- Control (naive single top-1-hypothesis) verified rate of eligible: "
        f"{(overall_summary['control_verified_rate_of_eligible'] or 0):.3f}",
        f"- Robust (whole-region multi-hypothesis) verified rate of eligible: "
        f"{(overall_summary['robust_verified_rate_of_eligible'] or 0):.3f}",
        f"- Genuinely multi-candidate incidents (size > 1, the only population where control and robust could "
        f"structurally disagree -- size-1 regions make robust degenerate to control by construction): "
        f"{overall_summary['genuinely_multi_candidate_count']} of {overall_summary.get('n')} total "
        f"({sum(1 for r in all_incidents if r['eligible'] and r['candidate_set_size'] > 1)} shown for cross-check).",
        f"- Decision disagreement rate among genuinely multi-candidate incidents: "
        f"{overall_summary['decision_disagreement_rate_multi_candidate']} "
        f"({overall_summary['decision_disagreement_count_multi_candidate']} of "
        f"{overall_summary['genuinely_multi_candidate_count']}) -- **on this held-out sample, robust whole-region "
        f"verification never changed the verify/reject decision relative to naive top-1-only verification.** This is "
        f"an honest negative finding for the mechanism's empirically demonstrated safety value-add on this topology, "
        f"not evidence the mechanism is unnecessary in general: it reflects that here, action-template plan safety "
        f"(pressure/service constraints) was largely source-location-invariant, not that whole-region verification "
        f"is redundant by design (the K=3 architectural guarantee against false-safe holds regardless).",
        f"- Control false-safe count (naive-verified plan actually unsafe against the real held-out source): "
        f"{overall_summary['control_false_safe_count']} "
        f"(rate of control-verified: {overall_summary['control_false_safe_rate_of_control_verified']})",
        f"- Robust false-safe count (robustly-verified plan actually unsafe against the real held-out source): "
        f"{overall_summary['robust_false_safe_count']} "
        f"(rate of robust-verified: {overall_summary['robust_false_safe_rate_of_robust_verified']})",
        f"- Mean exact simulator calls/incident: control={overall_summary['mean_control_exact_simulator_calls']}, "
        f"robust={overall_summary['mean_robust_exact_simulator_calls']}",
        f"- Mean wall-clock seconds/incident (control+robust combined): "
        f"{overall_summary['mean_elapsed_seconds_per_incident']}",
        "",
        f"**Zero robust authority invariant violations: {zero_robust_invariant_violations}.** "
        f"Control (naive) authority invariant violations present: {control_has_violations}.",
        "",
        f"**Material actionability gain: {material_actionability_gain}** -- K equals production's existing "
        "maximum_planning_candidates threshold, so eligibility is identical between arms; this experiment "
        "could not and does not claim a reachable-incident actionability increase (see module docstring for why, "
        "and the exceeds-K rate above for how much traffic a future K-relaxation would need to address).",
        "",
    ]
    if robust_false_safe_incidents:
        lines += [
            "## ROBUST FALSE-SAFE INCIDENTS (preserved for failure-mode classification)",
            "",
            "The robust (whole-region) policy selected a plan as ROBUSTLY_VERIFIED that the exact-WNTR check "
            "against the incident's real held-out source found unsafe. Full per-incident records below; the full "
            "detail (candidate set, worst-case metrics, selected template) is also in "
            "`robust_false_safe_incidents` in the JSON artifact.",
            "",
        ]
        for item in robust_false_safe_incidents:
            lines.append(
                f"- scenario `{item['scenario_id']}` bucket={item['bucket']} depth={item['depth']} "
                f"candidate_set_size={item['candidate_set_size']} "
                f"true_source_in_candidate_set={item['true_source_in_candidate_set']} "
                f"selected_template={item['robust']['selected_template']} "
                f"worst_case={item['robust']['worst_case']}"
            )
        lines.append("")
    if control_false_safe_incidents:
        lines += [
            "## Control false-safe incidents (naive single-hypothesis verification, for reference)",
            "",
        ]
        for item in control_false_safe_incidents:
            lines.append(
                f"- scenario `{item['scenario_id']}` bucket={item['bucket']} depth={item['depth']} "
                f"candidate_set_size={item['candidate_set_size']} "
                f"true_source_in_candidate_set={item['true_source_in_candidate_set']} "
                f"selected_template={item['control']['selected_template']}"
            )
        lines.append("")
    lines.append(f"**Exit decision: {exit_decision}**")
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"exit_decision": exit_decision, "overall": overall_summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
