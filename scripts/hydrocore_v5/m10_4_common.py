"""Shared, governed helpers for Milestone 10.4 (governed full-trajectory
end-to-end validation). Frozen protocol:
`docs/evaluation/HYDROCORE_V5_M10_4_FULL_TRAJECTORY_PROTOCOL.md`.

M10.4 validates the ACTUAL RETAINED end-to-end system after M10.1/M10.2/
M10.3A/B/C closed every learned-specialist promotion question: canonical
M9.6 HydroCore-S predictor (unchanged), frozen M9 B_DEPTH_AWARE/alpha=0.1
calibration (unchanged, not refit), deterministic OOD, the REAL production
deterministic Scout path (`hydroswarm.sampling.rank_sample_locations`, via
`analysis.sample_result` / `/api/incidents/{id}/samples/recommend` -- NOT
`HydroScout.deterministic_fallback`, which True M10.2 already disclosed is
a narrower comparator, not the production path), the REAL deterministic
Strategist/candidate-generation/PlanVerifier path
(`hydroswarm.planning.generate_response_plans` + exact WNTR verification
via `/api/incidents/{id}/plans/{id}/verify`), driven through the REAL
FastAPI production application (`hydroswarm.api.create_app`) exactly as
`hydroswarm.evaluation.live_robustness` already does for the (separate,
older) v4 architecture-freeze candidate.

Why this module exists instead of reusing `hydroswarm.runtime.v4_defaults.
V4PipelineFactory`/`hydroswarm.evaluation.live_robustness` unmodified: the
module-level production `app` (`hydroswarm.api.app`'s
`pipeline_factory=V4PipelineFactory(DEFAULT_V4_RELEASE_BUNDLE_DIR, ...)`)
still serves the FROZEN, PRE-M9 `hydrocore-v4` architecture-freeze
candidate (`models/hydrocore-v4-release`, seed 20260810, recorded in
`reports/results/v4/architecture-freeze.json`) -- a DIFFERENT checkpoint
from the M9-selected canonical HydroCore-S (seeds 20260814/31874/20260815).
No M9/M10 milestone has ever promoted the M9.6 checkpoint into that
serving directory -- that is explicitly M10.5's job, and M10.5 is out of
scope for M10.4. `create_app(pipeline_factory=...)` already supports
swapping in a different factory (this is exactly how `live_robustness.
run_condition` itself is wired), so this module provides an
`M10_4_PipelineFactory`, structurally identical to `V4PipelineFactory`,
that instead composes the SAME real, unmodified production classes
(`HybridInferencePipeline`, `OODDetector`, `SignatureBuilder`,
`GOVERNED_TRAINING_SIGNATURE_POLICY`, `DYNAMIC_TRUST_FUSION_CONFIG`,
`rank_sample_locations`, `generate_response_plans`) around the canonical
M9.6 checkpoint + frozen M9 calibration, with the SAME hard-coded
`trained_tasks={"sentinel"}` governance constant `V4PipelineFactory` uses
(`hydroswarm.runtime.v4_defaults.V4_TRAINED_TASKS`) -- confirmed by
`hydroswarm.inference.pipeline` (`"scout"/"strategist"/"ood" not in
self.trained_tasks` gates at lines ~443/759/761/781) to keep every learned
Scout/Strategist/OOD head non-authoritative regardless of which checkpoint
is loaded. This is wiring, not new scientific policy: no threshold, no
weight, no calibration, no candidate generator, no verifier is touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from safetensors.torch import load_file  # noqa: E402

import m10_common as m10  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator  # noqa: E402
from hydroswarm.classical import (  # noqa: E402
    GOVERNED_TRAINING_SIGNATURE_POLICY,
    SignatureBuilder,
    SignatureCache,
    SignatureCacheKey,
    resolve_signature_mode,
)
from hydroswarm.data.scenarios import network_sha256  # noqa: E402
from hydroswarm.inference import DYNAMIC_TRUST_FUSION_CONFIG, HybridInferencePipeline, OODDetector, OODReference  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder  # noqa: E402
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA  # noqa: E402
from hydroswarm.runtime.paths import resolve_data_dir  # noqa: E402
from hydroswarm.simulation import HydraulicSimulator  # noqa: E402
from hydroswarm.simulation.wrapper import wntr  # noqa: E402

# ---------------------------------------------------------------------------
# Section 3-equivalent: fresh, disjoint development-only M10.4 seed
# namespace. Every prior milestone's seed base (grepped from every
# scripts/hydrocore_v5/*.py before freezing this constant):
#   M9.4 floor .............. 990_000_000  (see m9_4_common.py)
#   M10.1 base ............. 1_100_000_000
#   M10.2 refit train/val .. 1_200_000_000 / 1_200_100_000
#   M10.2 true eval ........ 1_200_200_000
#   M10.3 refit train/val .. 1_300_000_000 / 1_300_100_000
#   M10.3C population ...... 1_400_000_000
#   M10.3D (RESERVED, never executed -- M10.3 is scientifically closed,
#           but the base is reserved and must not be reused) .. 1_450_000_000
# M10.4 base is chosen with a full 50,000,000 buffer above every one of
# these, including the reserved-but-unused M10.3D block.
# ---------------------------------------------------------------------------

M10_4_SEED_BASE = 1_500_000_000
M10_4_FAMILY_OFFSET: dict[str, int] = {family: 1_000_000 * index for index, family in enumerate(m10.ALL_FAMILIES)}
M10_4_CONDITION_OFFSET: dict[str, int] = {}

PRIOR_SEED_RANGES: dict[str, tuple[int, int]] = {
    "m9_4_floor_and_up": (990_000_000, 1_099_999_999),
    "m10_1": (1_100_000_000, 1_199_999_999),
    "m10_2_refit_train": (1_200_000_000, 1_200_099_999),
    "m10_2_refit_validation": (1_200_100_000, 1_200_199_999),
    "m10_2_true_eval": (1_200_200_000, 1_299_999_999),
    "m10_3_refit_train": (1_300_000_000, 1_300_099_999),
    "m10_3_refit_validation": (1_300_100_000, 1_399_999_999),
    "m10_3c_population": (1_400_000_000, 1_449_999_999),
    "m10_3d_reserved_unused": (1_450_000_000, 1_499_999_999),
}
M10_4_RANGE: tuple[int, int] = (1_500_000_000, 1_599_999_999)


def verify_seed_disjointness() -> dict[str, Any]:
    """Mechanical proof the M10.4 range does not overlap any prior range."""

    lo, hi = M10_4_RANGE
    overlaps: dict[str, bool] = {}
    for name, (plo, phi) in PRIOR_SEED_RANGES.items():
        overlaps[name] = not (hi < plo or lo > phi)
    return {
        "m10_4_range": list(M10_4_RANGE),
        "prior_ranges": {name: list(rng) for name, rng in PRIOR_SEED_RANGES.items()},
        "overlaps": overlaps,
        "disjoint": not any(overlaps.values()),
    }


def m10_4_seed(family: str, condition_index: int, incident_index: int) -> int:
    """Deterministic per-(family, condition, incident) seed, frozen before
    execution (order in the protocol's condition tuple is part of the
    formula, never reassigned post-freeze)."""

    return (
        M10_4_SEED_BASE
        + M10_4_FAMILY_OFFSET[family]
        + condition_index * 10_000
        + incident_index
    )


# ---------------------------------------------------------------------------
# Canonical M9.6 checkpoint + frozen M9 calibration re-verification.
# ---------------------------------------------------------------------------


def verify_canonical_checkpoints() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for seed in m10.SEEDS:
        record = m10.canonical_s_checkpoint(seed)
        path = Path(record["canonical_export_path"])
        exists = path.exists()
        actual = m10.checkpoint_sha256(str(path)) if exists else None
        expected = record["canonical_export_sha256"]
        result[seed] = {
            "path": str(path),
            "exists": exists,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": exists and actual == expected,
            "policy": record["canonical_checkpoint_policy"],
        }
    return result


def load_canonical_model(seed: int) -> tuple[HydroCore, str]:
    record = m10.canonical_s_checkpoint(seed)
    path = record["canonical_export_path"]
    actual = m10.checkpoint_sha256(path)
    if actual != record["canonical_export_sha256"]:
        raise ValueError(f"canonical M9.6 checkpoint SHA-256 mismatch for seed {seed}")
    model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == m10.SELECTED_PARAMETER_COUNT
    return model, actual


CALIBRATION_SUPPORT_PATH = ROOT / "reports/evaluation/hydrocore-v5/m9-6/m9-6-canonical-calibration.jsonl"
CALIBRATION_SUPPORT_ARM = "ARM_B_M9_6"
CALIBRATION_ALPHA = 0.1
CALIBRATION_MINIMUM_GROUP_SIZE = 10
#: `run_m10_1_decide._fit_frozen_calibrator`/`run_m10_2_true_evaluation.
#: fit_frozen_calibrator` stamp this frozen artifact with SYMBOLIC,
#: non-cryptographic identity strings ("m9-6-arm-b-frozen-S" /
#: "m9-6-frozen") -- which is safe for THEIR use, because neither of those
#: scripts ever drives the real `HybridInferencePipeline.analyze()`
#: (they call `calibrator.candidate_set(...)` directly). M10.4 is the
#: first M10 milestone to drive the real, unmodified production
#: `HybridInferencePipeline.analyze()` path, which unconditionally calls
#: `CalibrationArtifact.validate_runtime(model_hash=..., feature_schema_
#: hash=..., fusion_config_hash=...)` and fails closed (`calibrated=False`)
#: on any mismatch (`hydroswarm/inference/pipeline.py`, calibration
#: identity gate). Stamping the SAME frozen fit (same examples, same
#: alpha=0.1, same minimum_group_size, same B_DEPTH_AWARE grouping) with
#: the REAL runtime identity values below is therefore a required
#: correctness fix, not a refit: the statistical fit (which examples,
#: which alpha, which grouping) is byte-for-byte identical to the existing
#: M10.1/M10.2 frozen-calibration convention; only the identity METADATA
#: recorded alongside it is corrected so the SAME production identity gate
#: every other real (non-M10-evaluation-script) caller must satisfy is
#: satisfied honestly instead of by construction always failing closed.
#: `feature_schema_hash`/`fusion_config_hash` are real, static, checkpoint-
#: independent values (`DEFAULT_FEATURE_SCHEMA.fingerprint`,
#: `DYNAMIC_TRUST_FUSION_CONFIG`) -- identical for every seed.
#: `model_hash` is seed-specific (each of the 3 canonical checkpoints has
#: its own real SHA-256), so `fit_frozen_calibrator` takes it as a
#: parameter and the SAME examples/alpha/grouping is re-stamped per seed.
CALIBRATION_DATASET_MANIFEST_HASH = "m9-6-canonical-calibration"


def fit_frozen_calibrator(*, model_hash: str, topology_hashes: tuple[str, ...] = ()) -> SplitConformalCalibrator:
    """Reuse M9's frozen B_DEPTH_AWARE / alpha=0.1 calibration -- same
    examples, same alpha, same grouping as `run_m10_1_decide.py`/
    `run_m10_2_true_evaluation.py` -- no refit. `model_hash` is the REAL,
    per-seed canonical M9.6 checkpoint SHA-256 (see module docstring above
    for why this must be real, not symbolic, for M10.4 specifically).
    `topology_hashes` (new, additive) records this pass's own
    TRAINED_FAMILIES network hashes as `validated_topology_hashes` so
    `OODDetector`'s deterministic topology-novelty check (a no-op when this
    tuple is empty -- see `hydroswarm.inference.ood.OODDetector.
    topology_novelty`) can actually distinguish a trained topology from a
    development-only topology-shift family. Neither change alters the
    fitted conformal scores/coverage."""

    examples: list[CalibrationExample] = []
    with CALIBRATION_SUPPORT_PATH.open() as fh:
        for line in fh:
            record = json.loads(line)
            if record["arm"] != CALIBRATION_SUPPORT_ARM:
                continue
            examples.append(CalibrationExample(
                probabilities=tuple(record["probabilities"]), true_index=record["true_index"],
                condition=record["condition"], network_id=f"{record['family']}:{record['depth_bucket']}",
            ))
    return SplitConformalCalibrator.fit(
        examples, alpha=CALIBRATION_ALPHA, model_hash=model_hash,
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        dataset_manifest_hash=CALIBRATION_DATASET_MANIFEST_HASH,
        minimum_group_size=CALIBRATION_MINIMUM_GROUP_SIZE,
        topology_hashes=topology_hashes,
        fusion_config_hash=DYNAMIC_TRUST_FUSION_CONFIG,
    )


def trained_family_topology_hashes() -> tuple[str, ...]:
    hashes = []
    for family in m10.TRAINED_FAMILIES:
        network = m10.ALL_FAMILY_LOADERS[family]()
        hashes.append(network_sha256(network))
    return tuple(hashes)


#: The same hard-coded governance constant `hydroswarm.runtime.v4_defaults.
#: V4_TRAINED_TASKS` uses -- ONLY "sentinel" is ever passed for any v4/v5
#: checkpoint built so far (no Scout/Strategist/OOD promotion has ever
#: happened -- see reports/results/v4/phase14-promotion-gates.md and every
#: M10.1/M10.2/M10.3 closure on this branch). Kept as an independent literal
#: here (not imported) so an M10.4 preflight check can mechanically assert
#: this module's own value matches v4_defaults's, rather than silently
#: inheriting a future change to that module without re-review.
M10_4_TRAINED_TASKS: frozenset[str] = frozenset({"sentinel"})

#: Reused, NOT invented: the exact `runtime_enabled_outputs` list the
#: current frozen `hydrocore-v4` architecture-freeze candidate declares for
#: its own "sentinel"-only role (`reports/results/v4/architecture-freeze.
#: json`). None of these fields are Scout/Strategist/OOD-related (those are
#: separately, independently gated by `M10_4_TRAINED_TASKS` above) --
#: reusing this existing declared governed set, rather than inventing a new
#: one, for the same "sentinel" role on the M9.6 checkpoint.
M10_4_RUNTIME_ENABLED_OUTPUTS: frozenset[str] = frozenset({
    "event_cause", "event_presence", "evidence_sufficiency", "next_step", "relative_strength", "source_node",
})

#: NOTE (preflight-disclosed, non-blocking, matches M10.0's own established
#: precedent): M9.6's recorded training record
#: (`reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_B_M9_6-seed*.json`,
#: field `feature_kwargs`) declares
#: `{"include_relative_gap_feature": false, "unobserved_age_sentinel": "fixed"}`.
#: `HydraulicFeatureBuilder.build(...)`'s own defaults are
#: `include_relative_gap_feature=False` (already matches) and
#: `unobserved_age_sentinel="incident_elapsed"` (does NOT match "fixed").
#: `hydroswarm.inference.pipeline.HybridInferencePipeline.analyze()` calls
#: `self.feature_builder.build(...)` with a HARD-CODED `window_steps=25` and
#: does not pass either kwarg through at all -- there is no supported,
#: non-invasive way for a `pipeline_factory` to override this without
#: editing `hydroswarm/inference/pipeline.py` itself (out of scope: that
#: would change model-input semantics for the shared production code path,
#: which M10.4 is not authorized to do). M10.0's own closed, immutable
#: preflight (`scripts/hydrocore_v5/run_m10_0_preflight.py:_one_real_forward_pass`)
#: already exercised this exact checkpoint through
#: `HydraulicFeatureBuilder().build(...)` with the SAME unremarked defaults
#: and was accepted as `SYSTEM_PREFLIGHT_PASS` -- M10.4 follows that same
#: established precedent rather than silently "fixing" feature construction
#: to more closely match training (which would itself be an undisclosed
#: input-semantics change). Recorded as a disclosed, non-blocking finding in
#: the M10.4 preflight artifact, not a silent repair.
M10_4_FEATURE_BUILDER_KWARGS: dict[str, Any] = {}


class M10_4_PipelineFactory:
    """Structurally identical to `hydroswarm.runtime.v4_defaults.
    V4PipelineFactory` (same `HybridInferencePipeline`/`OODDetector`/
    `SignatureBuilder`/`GOVERNED_TRAINING_SIGNATURE_POLICY`/
    `DYNAMIC_TRUST_FUSION_CONFIG` composition, same `__call__(network_record,
    network_path)` interface `hydroswarm.api.app.bind_pipeline` expects),
    with the checkpoint/calibration source swapped from
    `models/hydrocore-v4-release` to the canonical M9.6 HydroCore-S
    checkpoint + frozen M9 calibration."""

    def __init__(self, *, seed: int, project_root: Path) -> None:
        self.seed = seed
        self.project_root = project_root
        self.signature_cache = resolve_data_dir(project_root) / "signatures"
        model, model_hash = load_canonical_model(seed)
        self._model = model
        self._model_hash = model_hash
        self._calibrator = fit_frozen_calibrator(model_hash=model_hash, topology_hashes=trained_family_topology_hashes())
        self._feature_builder = HydraulicFeatureBuilder(**M10_4_FEATURE_BUILDER_KWARGS)
        self.trained_assets_ready = True
        self.fallback_reason: str | None = None
        self.signature_mode = None
        self.signature_policy_hash: str | None = None

    @property
    def model_hash(self) -> str | None:
        return self._model_hash

    def __call__(self, _network_record: Any, network_path: str | Path) -> HybridInferencePipeline:
        if wntr is None:
            raise RuntimeError("WNTR is unavailable")
        network = wntr.network.WaterNetworkModel(str(network_path))
        simulator = HydraulicSimulator(network)
        source_nodes = tuple(map(str, network.junction_name_list))
        if not source_nodes:
            raise ValueError("network has no junction source candidates")
        sensor_nodes = source_nodes
        policy = GOVERNED_TRAINING_SIGNATURE_POLICY
        topology_hash = network_sha256(network)
        self.signature_mode = resolve_signature_mode(topology_hash)
        self.signature_policy_hash = policy.policy_hash
        key = SignatureCacheKey(
            network_hash=simulator.state_hash(), hydraulic_state_hash=simulator.state_hash(),
            simulator_version=simulator.simulator_version, configuration_hash=policy.policy_hash,
            sensor_layout_hash=hashlib.sha256("|".join(sensor_nodes).encode()).hexdigest(),
        )
        artifact = SignatureBuilder(simulator, SignatureCache(self.signature_cache)).build_or_load(
            key=key, source_nodes=source_nodes, start_time_bins=policy.start_time_bins,
            duration_bins=policy.duration_bins, strength_bins=policy.strength_bins,
            demand_regimes=policy.demand_regimes, sensor_nodes=sensor_nodes,
            sample_times_seconds=policy.sample_times_seconds,
        )
        calibration = self._calibrator.artifact if self._calibrator is not None else None
        return HybridInferencePipeline(
            simulator=simulator, signature_artifact=artifact, model=self._model, model_hash=self._model_hash,
            calibration_artifact=calibration,
            ood_detector=OODDetector(OODReference(
                validated_network_hashes=calibration.validated_topology_hashes if calibration else (),
            )),
            feature_builder=self._feature_builder,
            trained_tasks=M10_4_TRAINED_TASKS,
            runtime_enabled_outputs=M10_4_RUNTIME_ENABLED_OUTPUTS,
            fusion_config_hash=DYNAMIC_TRUST_FUSION_CONFIG,
        )


def network_inp_path(family: str, tmp_dir: Path) -> Path:
    """Materialize a governed M10-family network to a temp .inp file for
    the production `/api/networks/import` upload contract (an ordinary
    EPANET-format upload; not a locked/test fixture -- see
    `hydroswarm.evaluation.live_robustness._reject_locked`, applied the
    same way by M10.4's own preflight/execution scripts)."""

    network = m10.ALL_FAMILY_LOADERS[family]()
    path = tmp_dir / f"{family}.inp"
    wntr.network.write_inpfile(network, str(path))
    return path


def m10_4_tmp_root() -> Path:
    return Path(tempfile.gettempdir()) / "hydroswarm-m10-4"


# ---------------------------------------------------------------------------
# Causal, real, production-API-driven paired trajectory runner. Reuses
# `hydroswarm.evaluation.live_robustness`'s own governed Condition/
# scenario/payload/sample/entropy/invariant primitives verbatim (frozen,
# already-tested machinery -- Part 4 of the M10.4 spec forbids inventing a
# new perturbation framework) -- only the ARM structure (FULL vs
# NO_EXTRA_SAMPLING, run as two SEPARATE incidents sharing byte-identical
# initial evidence) and the metrics extraction are new.
# ---------------------------------------------------------------------------

from datetime import UTC, datetime, timedelta  # noqa: E402

from hydroswarm.evaluation.live_robustness import (  # noqa: E402
    Condition,
    _entropy,
    _invariants,
    _metric_fields,
    _payloads,
    _sample_observation,
    _scenario_config,
)


SAFETY_COUNTERS_TEMPLATE: dict[str, int] = {
    "inaccessible_sample_selected": 0,
    "already_sampled_reselected": 0,
    "sampling_budget_exceeded": 0,
    "unverified_plan_surfaced_as_actionable": 0,
    "wntr_rejected_plan_surfaced_as_safe": 0,
    "human_approval_bypassed": 0,
    "autonomous_actuation_detected": 0,
    "learned_ood_overrode_deterministic": 0,
    "learned_scout_selected_sample": 0,
    "learned_strategist_selected_plan": 0,
    "nonfinite_value_reached_decision": 0,
    "locked_test_opened": 0,
    "invariant_failures": 0,
}


def _run_single_arm(
    *, client: Any, network_id: str, incident_id: str, arm: str, source_node: str,
    observed_nodes: set[str], scenario: Any, randomized: Any, origin: datetime,
    condition: Condition, maximum_samples: int, safety: dict[str, int],
) -> dict[str, Any]:
    """Drive one real, causal, sequential production-API trajectory for one
    already-created incident (`arm in {"FULL", "NO_EXTRA_SAMPLING"}`)."""

    analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
    initial_metrics = _metric_fields(analysis, source_node)
    rounds: list[dict[str, Any]] = []
    available_observations = set(observed_nodes)
    stop_reason = "NO_EXTRA_SAMPLING_ARM" if arm == "NO_EXTRA_SAMPLING" else None

    if arm == "FULL":
        for sample_index in range(maximum_samples):
            if analysis["planning_allowed"]:
                stop_reason = "PLANNING_ALLOWED"
                break
            before_entropy = _entropy(analysis["fused_belief"])
            before_true_rank = _true_rank(analysis["fused_belief"], source_node)
            recommendation = client.post(f"/api/incidents/{incident_id}/samples/recommend")
            if recommendation.status_code != 200:
                stop_reason = recommendation.json().get("detail", "STOP")
                rounds.append({"round": sample_index, "status": "STOP", "http_status": recommendation.status_code, "stop_reason": stop_reason})
                break
            rec = recommendation.json()
            node_id = rec["node_id"]
            if node_id in available_observations:
                safety["already_sampled_reselected"] += 1
                stop_reason = "SAFETY_VIOLATION_RESELECTED"
                rounds.append({"round": sample_index, "status": "SAFETY_VIOLATION", "recommended_node": node_id})
                break
            decision_seconds = float(scenario.timestamps_seconds[-1])
            observation = _sample_observation(
                node_id, scenario, randomized, origin, sample_index,
                decision_time_seconds=decision_seconds,
                collection_delay_minutes=float(rec["expected_collection_delay_minutes"]),
                noise_std=0.05, seed=condition.seed,
            )
            added = client.post(f"/api/incidents/{incident_id}/samples", json=observation)
            if added.status_code != 200:
                stop_reason = f"ADD_FAILED_{added.status_code}"
                rounds.append({"round": sample_index, "status": "ADD_FAILED", "http_status": added.status_code})
                break
            available_observations.add(node_id)
            analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
            after_true_rank = _true_rank(analysis["fused_belief"], source_node)
            rounds.append({
                "round": sample_index, "status": "SAMPLE", "recommended_node": node_id,
                "expected_information_gain_bits": rec["expected_information_gain"],
                "entropy_before": before_entropy, "entropy_after": _entropy(analysis["fused_belief"]),
                "true_source_rank_before": before_true_rank, "true_source_rank_after": after_true_rank,
                "candidate_size_after": len(analysis.get("candidate_nodes") or ()),
            })
        else:
            stop_reason = stop_reason or "BUDGET_EXHAUSTED"

    # `count=2` matches `hydroswarm.evaluation.live_robustness`'s own
    # established "measurement" lifecycle default (its ONLY production-
    # verified count) -- reused as-is, not invented. Per-incident exact
    # WNTR verification is a REAL, governed, existing production budget
    # (`hydroswarm.api.state.Settings.exact_plan_simulation_limit = 3`),
    # not something M10.4 may raise. Verifying candidates in RANKED order
    # and stopping at the first VERIFIED result (rather than
    # unconditionally verifying every requested candidate) keeps every
    # trajectory within that real budget AND keeps the incident's
    # verification-derived `status`/`approval_pending` state (which
    # reflects only the MOST RECENT `/verify` call --
    # `hydroswarm.api.app._record_verification`) correctly pointed at the
    # plan this harness is about to approve, with no re-verification call
    # needed -- exactly the sequence a real operator (verify the plan you
    # are about to approve, not every alternative after it) would follow.
    plans_generated = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 2})
    plans: list[dict[str, Any]] = []
    if plans_generated.status_code == 200:
        for plan in plans_generated.json():
            verification = client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify")
            verification_json = verification.json() if verification.status_code == 200 else {"decision": "ERROR", "http_status": verification.status_code}
            action_types = [item["action_type"] for item in plan.get("actions", [])]
            plans.append({
                "plan_id": plan["plan_id"], "action_types": action_types,
                "is_no_action": action_types == [] or action_types == ["NO_ACTION"],
                "verification": verification_json,
            })
            if verification_json.get("decision") == "VERIFIED":
                break
    verified = [p for p in plans if p["verification"].get("decision") == "VERIFIED"]
    rejected = [p for p in plans if p["verification"].get("decision") == "REJECTED"]
    selected = verified[0] if verified else None
    no_action_plan = next((p for p in plans if p["is_no_action"]), None)

    approval_status: int | None = None
    if selected is not None:
        approved = client.post(f"/api/incidents/{incident_id}/plans/{selected['plan_id']}/approve", json={"approved": True, "operator_id": "m10-4-study"})
        approval_status = approved.status_code
        selected["approval_status"] = approval_status
        if approval_status != 200:
            safety["human_approval_bypassed"] += 1

    final = client.get(f"/api/incidents/{incident_id}").json()
    invariants = _invariants(
        analysis=analysis, generate_status=plans_generated.status_code, plans=plans,
        approval_status=approval_status, stale_approval_status=None,
    )
    if any(value is False for value in invariants.values()):
        safety["invariant_failures"] += 1
    if any(p["verification"].get("decision") in ("REJECTED", "ABSTAINED") and p.get("approval_status") == 200 for p in plans):
        safety["wntr_rejected_plan_surfaced_as_safe"] += 1
    # No autonomous actuation: the ONLY state transition this harness ever
    # performs toward "approved" is the single explicit `/approve` call
    # above (simulating a human operator) -- the harness never calls any
    # other endpoint capable of marking a plan executed/actuated, and
    # `final_status` below is recorded for audit, not treated as evidence
    # of actuation on its own.

    return {
        "arm": arm, "network_id": network_id, "incident_id": incident_id, "source_node": source_node,
        "condition": condition.name, "seed": condition.seed,
        "initial": initial_metrics, "final": _metric_fields(analysis, source_node),
        "final_analysis": {k: analysis.get(k) for k in (
            "calibrated", "ood_level", "planning_allowed", "control_action", "evidence_sufficient", "disagreement_js",
        )},
        "rounds": rounds, "samples_taken": len(available_observations) - len(observed_nodes),
        "stop_reason": stop_reason,
        "plans_generated": len(plans), "plans_verified": len(verified), "plans_rejected": len(rejected),
        "no_action_available": no_action_plan is not None,
        "no_action_consequences": (no_action_plan or {}).get("verification", {}).get("consequences"),
        "selected_plan": selected,
        "no_safe_plan": len(plans) > 0 and not verified,
        "final_status": final.get("status"),
        "invariants": invariants,
    }


def _true_rank(fused_belief: dict[str, float], source: str) -> int | None:
    ranked = sorted(fused_belief, key=lambda node: (-fused_belief[node], node))
    return ranked.index(source) + 1 if source in ranked else None


def run_incident_pair(
    *, client: Any, network_path: Path, network_id: str, condition: Condition,
    maximum_samples: int, safety: dict[str, int],
) -> dict[str, Any]:
    """Generate ONE physical scenario (one WNTR incident), then run it
    through TWO SEPARATE incidents (byte-identical initial evidence) --
    ARM FULL (production Scout sampling engaged) and ARM NO_EXTRA_SAMPLING
    (no active sample requests) -- so the paired comparator's initial state
    is provably identical (Part 4 of the M10.4 spec) while only the
    sequential evidence-acquisition dimension differs."""

    import wntr as _wntr
    from hydroswarm.data.scenarios import WNTRScenarioGenerator

    network = _wntr.network.WaterNetworkModel(str(network_path))
    scenario, randomized = WNTRScenarioGenerator().generate_with_network(network, _scenario_config(condition))
    # Must stay safely in the PAST relative to real wall-clock time: the
    # production `/analyze` endpoint only admits sensor series whose
    # `received_at` is causally available as of the real current time
    # (`hydroswarm.api.app`'s own causal-evidence guard -- "at least one
    # sensor series is required" otherwise), matching the M10.4 hard
    # requirement that no future evidence reach an earlier decision. A
    # small, deterministic, seed-derived offset within a fixed past window
    # (not `condition.seed` used directly as a day offset, which -- at the
    # ~1.5 billion M10.4 seed namespace -- would overflow into the future).
    origin = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=(condition.seed % 180))
    observations = _payloads(scenario, condition, origin)
    source_node = scenario.manifest.incident.source_nodes[0]
    observed_nodes = {item["node_id"] for item in observations}

    arms: dict[str, Any] = {}
    for arm in ("FULL", "NO_EXTRA_SAMPLING"):
        created = client.post("/api/incidents", json={
            "network_id": network_id, "detected_at": origin.isoformat(),
            "observations": observations, "maximum_samples": maximum_samples,
        })
        if created.status_code != 201:
            arms[arm] = {"arm": arm, "outcome": "HARNESS_ERROR", "http_status": created.status_code}
            continue
        incident_id = created.json()["incident_id"]
        analyzed = client.post(f"/api/incidents/{incident_id}/analyze")
        if analyzed.status_code != 200:
            arms[arm] = {
                "arm": arm, "outcome": "ABSTAINED" if analyzed.status_code == 409 else "HARNESS_ERROR",
                "http_status": analyzed.status_code, "incident_id": incident_id,
                "condition": condition.name, "seed": condition.seed, "source_node": source_node,
            }
            continue
        arms[arm] = _run_single_arm(
            client=client, network_id=network_id, incident_id=incident_id, arm=arm,
            source_node=source_node, observed_nodes=observed_nodes, scenario=scenario,
            randomized=randomized, origin=origin, condition=condition,
            maximum_samples=maximum_samples, safety=safety,
        )

    initial_equal = (
        "arm_error" not in arms
        and arms.get("FULL", {}).get("initial") == arms.get("NO_EXTRA_SAMPLING", {}).get("initial")
    )
    return {
        "condition": condition.name, "seed": condition.seed, "network_id": network_id,
        "source_node": source_node, "family": condition.network_id,
        "arms": arms, "paired_initial_state_equal": initial_equal,
    }
