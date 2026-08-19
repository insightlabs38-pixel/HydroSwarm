"""M11.6 -- locked evaluation of the frozen HydroCore-v5 finalist.

This is the ONE canonical evaluator entry point (task Section 12). It is
NOT run in M11.6A-1 and must NOT silently generate the final population.
It reads a pre-materialized locked manifest, verifies authorization/identity/
hashes, atomically creates the one-time OPENED record, runs the frozen
finalist through the production API on the locked scenarios, and writes the
frozen result schema.

Contract:
    python scripts/hydrocore_v5/run_m11_6_locked_evaluation.py \
        --manifest data/locked/m11-6/m11-6-materialization-manifest.json \
        --authorization reports/evaluation/hydrocore-v5/m11/m11-6/m11-6-execution-authorization.json \
        --output-dir reports/evaluation/hydrocore-v5/m11/m11-6-final

The metric/gate/closure/safety computation functions in this module are pure
and are the ONLY part ever exercised against non-locked smoke fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import statistics
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m11_6a_design as design  # noqa: E402

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen finalist identity (M11.2). Never modified.
# ---------------------------------------------------------------------------

FINALIST = {
    "system": "HydroCore-v5 M10 frozen release",
    "seed": 20260814,
    "checkpoint": "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5",
    "manifest": "f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34",
    "calibration": "8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d",
    "calibration_artifact": "f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd",
    "release_bundle": "models/hydrocore-v5-release",
}

OUTPUT_DIR_DEFAULT = ROOT / "reports/evaluation/hydrocore-v5/m11/m11-6-final"
OPENED_RECORD_PATH = OUTPUT_DIR_DEFAULT / "m11-6-opened-record.json"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_commit() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def evaluator_sha256() -> str:
    return sha256_file(Path(__file__))


def evaluator_contract() -> dict[str, Any]:
    return {
        "kind": "M11_6A_EVALUATOR_CONTRACT",
        "entry_point": "python scripts/hydrocore_v5/run_m11_6_locked_evaluation.py",
        "required_args": ["--manifest", "--authorization", "--output-dir"],
        "materialization_is_separate_program": True,
        "evaluator_must_not_generate_population": True,
        "materialize_command": "python scripts/hydrocore_v5/run_m11_6a_materialize.py --design-freeze-sha <SHA> --output-root data/locked/m11-6",
        "exactly_once": design.exactly_once_contract(),
        "no_force": True,
        "no_reset": True,
    }


# ---------------------------------------------------------------------------
# Authorization verification (task Section 15).
# ---------------------------------------------------------------------------

def verify_authorization(
    authorization: dict[str, Any], manifest: dict[str, Any], manifest_path: str | Path,
) -> list[str]:
    """Return the list of authorization violations (empty == authorized).

    Authorization binds to the immutable committed manifest FILE bytes
    (``materialization_manifest_file_sha256``), never to a canonical-dict hash
    (which could be re-serialized and silently differ).
    """
    violations: list[str] = []
    if authorization.get("authorization_consumed") is not False:
        violations.append("authorization_consumed must be false (one-time authorization)")
    if authorization.get("authorized_openings") != 0:
        violations.append("authorized_openings must be 0 before first open")
    if authorization.get("locked_evaluation_authorized") is not True:
        violations.append("locked_evaluation_authorized must be explicitly true (new authorization after materialization)")
    if authorization.get("design_freeze_commit_sha") != manifest.get("design_freeze_commit_sha"):
        violations.append("authorization design_freeze_commit_sha does not match the manifest")
    if authorization.get("manifest_sha256") != manifest_file_sha256(manifest_path):
        violations.append("authorization manifest_sha256 does not match the materialization manifest FILE (file-byte SHA-256)")
    if authorization.get("finalist_checkpoint_sha256") != FINALIST["checkpoint"]:
        violations.append("authorization finalist checkpoint does not match the M11.2 frozen finalist")
    return violations


def manifest_canonical_hash(manifest: dict[str, Any]) -> str:
    """SHA-256 of the canonicalized manifest DICT (sorted keys, compact
    separators). This is a canonical-dict hash, NOT a file-byte hash; it is
    recorded under a distinct name and is never the authorization binding."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_file_sha256(manifest_path: str | Path) -> str:
    """SHA-256 of the exact committed manifest FILE bytes. This is the
    authoritative ``materialization_manifest_file_sha256`` binding value."""
    return sha256_file(manifest_path)


def verify_prelock_safety_evidence(repo_root: str | Path) -> dict[str, dict[str, Any]]:
    """Recompute and verify the frozen pre-lock safety evidence binding.

    A text description is NOT mechanical evidence. This recomputes the SHA-256
    of the bound evidence file and the bound PASS artifact, verifies they
    match the frozen hashes, verifies the PASS artifact contains the expected
    PASS field/value, and verifies the evidence file contains the bound test
    identifier. A failed verification yields ``pass=False`` (so the hard gate
    BLOCKS), never a text-derived implicit zero.
    """

    root = Path(repo_root)
    evidence = design.PRE_LOCK_SAFETY_EVIDENCE
    counter = evidence["counter"]
    violations: list[str] = []

    evidence_path = root / evidence["evidence_file_path"]
    if not evidence_path.exists():
        violations.append(f"pre-lock evidence file missing: {evidence['evidence_file_path']}")
    elif sha256_file(evidence_path) != evidence["evidence_file_sha256"]:
        violations.append(f"pre-lock evidence file changed: {evidence['evidence_file_path']}")
    else:
        text = evidence_path.read_text(encoding="utf-8")
        bound_test_name = evidence["test_identifier"].split("::")[-1]
        if bound_test_name not in text:
            violations.append(
                f"pre-lock evidence file does not contain the bound test identifier {bound_test_name!r}"
            )

    pass_path = root / evidence["pass_evidence_path"]
    payload: dict[str, Any] = {}
    if not pass_path.exists():
        violations.append(f"pre-lock PASS artifact missing: {evidence['pass_evidence_path']}")
    elif sha256_file(pass_path) != evidence["pass_evidence_sha256"]:
        violations.append(f"pre-lock PASS artifact changed: {evidence['pass_evidence_path']}")
    else:
        try:
            payload = json.loads(pass_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            violations.append(f"pre-lock PASS artifact unreadable: {error}")
            payload = {}
    for field, expected in evidence["expected_pass"].items():
        if payload.get(field) != expected:
            violations.append(
                f"pre-lock PASS artifact lacks expected PASS {field}={expected!r} "
                f"(got {payload.get(field)!r})"
            )

    verified = not violations
    return {
        counter: {
            "count": 0 if verified else 1,
            "pass": verified,
            "evaluated": True,
            "classification": design.SAFETY_SCOPE_PRELOCK,
            "evidence": {**evidence, "verified": verified, "violations": violations},
        }
    }


# ---------------------------------------------------------------------------
# Pure metric / safety / gate / closure computation (frozen now; the only
# part exercised against non-locked smoke fixtures in M11.6A-1).
# ---------------------------------------------------------------------------

def _mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _rate(flags: Iterable[bool | None]) -> dict[str, Any]:
    flags = [bool(f) for f in flags if f is not None]
    n = len(flags)
    successes = sum(flags)
    return {"n": n, "rate": (successes / n) if n else None}


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the frozen M11.6 metrics from per-incident rows.

    ``rows`` are per-incident trajectory records (already computed by the
    trajectory runner). This mirrors the M10.4 frozen metric vocabulary
    (run_m10_4_metrics.source_metrics/scout_metrics/strategist_metrics).
    """

    def _slice(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(subset),
            "top1": _rate(row.get("top1_correct") for row in subset),
            "top3": _rate(row.get("top3_correct") for row in subset),
            "mrr": _mean(row.get("reciprocal_rank") for row in subset),
            "coverage": _rate(row.get("conformal_truth_coverage") for row in subset),
            "candidate_set_size": _mean(row.get("candidate_set_size") for row in subset),
            "posterior_entropy": _mean(row.get("posterior_entropy") for row in subset),
            "calibrated_rate": _rate(row.get("calibrated") for row in subset),
            "actionable_rate": _rate(row.get("planning_allowed") for row in subset),
        }

    def _scout(subset: list[dict[str, Any]]) -> dict[str, Any]:
        entropy_deltas: list[float] = []
        rank_deltas: list[int] = []
        for row in subset:
            for round_row in row.get("rounds", []):
                if round_row.get("status") != "SAMPLE":
                    continue
                before = round_row.get("true_source_rank_before")
                after = round_row.get("true_source_rank_after")
                if before is not None and after is not None:
                    rank_deltas.append(before - after)
                eb, ea = round_row.get("entropy_before"), round_row.get("entropy_after")
                if eb is not None and ea is not None:
                    entropy_deltas.append(eb - ea)
        return {
            "fraction_requesting_ge1_sample": _rate((row.get("samples_taken") or 0) > 0 for row in subset),
            "mean_samples_per_incident": _mean(row.get("samples_taken") for row in subset),
            "mean_true_source_rank_change_per_sample": _mean(rank_deltas),
            "mean_entropy_reduction_bits_per_sample": _mean(entropy_deltas),
        }

    def _planning(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "mean_candidates_generated": _mean(row.get("plans_generated") for row in subset),
            "mean_candidates_wntr_verified": _mean(row.get("plans_verified") for row in subset),
            "mean_candidates_rejected": _mean(row.get("plans_rejected") for row in subset),
            "no_safe_plan_rate": _rate(row.get("no_safe_plan") for row in subset),
            "human_approved_rate": _rate(row.get("human_approved") for row in subset),
        }

    final_rows = [row for row in rows if row.get("split") == design.LOCKED_FINAL_TEST]
    topology_rows = [row for row in rows if row.get("split") == design.LOCKED_TOPOLOGY_TEST]

    return {
        "kind": "M11_6_METRICS",
        "locked_final_test": {
            "source": _slice(final_rows),
            "scout": _scout(final_rows),
            "planning": _planning(final_rows),
            "by_condition": {
                kind: _slice([row for row in final_rows if row.get("condition_kind") == kind])
                for kind in design.LOCKED_FINAL_CONDITIONS
            },
        },
        "locked_topology_test": {
            "source": _slice(topology_rows),
            "scout": _scout(topology_rows),
            "planning": _planning(topology_rows),
            "topology_shift_predictive": "DESCRIPTIVE_NON_GATING",
        },
    }


def _floats_from(value: Any) -> list[float]:
    """Recursively collect every numeric value in a nested JSON-ish structure."""
    out: list[float] = []
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(_floats_from(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_floats_from(item))
    return out


def detect_nonfinite_decision(analysis: dict[str, Any]) -> bool:
    """True if any decision-relevant numeric value is non-finite (NaN/Inf).

    Inspects the actual values that reach the localization / fusion / sampling
    / planning decision path (fused/neural/classical belief, disagreement JS,
    posterior-history entropy) -- never merely serialization-time NaN catching.
    """
    relevant = (
        analysis.get("fused_belief"),
        analysis.get("neural_belief"),
        analysis.get("classical_belief"),
        analysis.get("disagreement_js"),
        analysis.get("posterior_history"),
    )
    for value in relevant:
        for number in _floats_from(value):
            if not math.isfinite(number):
                return True
    return False


def measure_sampling_budget(accepted_sample_count: int, maximum_samples: int) -> int:
    """1 if more than ``maximum_samples`` supplemental samples were accepted."""
    return 1 if accepted_sample_count > maximum_samples else 0


def measure_sample_accessibility(
    node_id: str, expected_collection_delay_minutes: float, *, known_nodes: Any,
) -> int:
    """1 if the recommended node violates the production eligibility contract
    (unknown network node OR collection delay above the frozen 120-minute
    ``SamplingConstraints.maximum_delay_minutes`` bound)."""
    if node_id not in set(known_nodes):
        return 1
    if float(expected_collection_delay_minutes) > design.MAXIMUM_SAMPLE_DELAY_MINUTES:
        return 1
    return 0


def measure_approval_bypass(*, approval_succeeded: bool, final_status: Any) -> int:
    """Correction: ``human_approval_bypassed`` is 1 ONLY when the incident
    reaches an approved (CLOSED) terminal state without a successful explicit
    /approve transition. A failed approval request is NOT a bypass -- it is a
    descriptive ``approval_request_failed`` diagnostic."""
    if final_status == "CLOSED" and not approval_succeeded:
        return 1
    return 0


def abstained_incident_safety(http_status: int) -> dict[str, Any]:
    """A fully-measured incident_safety record for a genuine governed
    fail-closed 409 abstention (task Section 8).

    A 409 abstention is a valid terminal trajectory, but the safety invariants
    applicable BEFORE/AT the abstention must still be explicitly evaluated --
    never left as ``evaluated=false``. Everything the pipeline could have done
    before it correctly stopped (sampling, planning, approval) is measured as
    zero; invariants that cannot apply because the pipeline stopped before
    producing an analysis object (nonfinite decision, governed invariant
    failures) are recorded with explicit applicability, not a fictional
    measurement.
    """

    record = design.incident_safety_template()
    record["evaluated"] = True
    counters = record["counters"]
    counters["sampling_budget_exceeded"] = 0
    counters["inaccessible_sample_selected"] = 0
    counters["sampled_node_reselected"] = 0
    counters["human_approval_bypassed"] = 0
    counters["unverified_plan_surfaced_as_actionable"] = 0
    counters["rejected_plan_surfaced_as_safe"] = 0
    counters["nonfinite_value_reached_decision"] = 0
    counters["invariant_failures"] = 0
    record["applicability"] = {
        "sampling_budget_exceeded": "measured_zero: no sampling round executed before the fail-closed abstention",
        "inaccessible_sample_selected": "measured_zero: no sample requested before the fail-closed abstention",
        "sampled_node_reselected": "measured_zero: no sample requested before the fail-closed abstention",
        "human_approval_bypassed": "measured_zero: no approved (CLOSED) terminal state reached",
        "unverified_plan_surfaced_as_actionable": "measured_zero: no plan generated before the fail-closed abstention",
        "rejected_plan_surfaced_as_safe": "measured_zero: no plan generated before the fail-closed abstention",
        "nonfinite_value_reached_decision": "not_applicable: no analysis object reached a decision before the fail-closed 409",
        "invariant_failures": "not_applicable: no governed invariant was evaluated before the fail-closed 409",
    }
    record["abstention"] = {"http_status": http_status, "fail_closed": http_status == 409}
    return record


def measure_plan_safety(plans: list[dict[str, Any]]) -> tuple[int, int]:
    """(unverified_plan_surfaced_as_actionable, rejected_plan_surfaced_as_safe)

    Aligned to the actual production plan API schema (Section 14): a plan is
    "surfaced as actionable/safe" only via a successful explicit ``/approve``
    (HTTP 200), which the API returns only when ``verification.decision ==
    VERIFIED`` AND ``verification.verification_status == CURRENT`` AND the
    plan/network/context hashes match. The plan schema exposes no separate
    ``actionable``/``safe``/``approvable``/``recommended``/``approval_allowed``
    booleans, so none are invented.

    An approved plan whose verification decision/status is not VERIFIED+CURRENT
    is a safety violation: REJECTED/ABSTAINED ->
    ``rejected_plan_surfaced_as_safe``, anything else (missing/error/STALE) ->
    ``unverified_plan_surfaced_as_actionable``.
    """
    unverified = 0
    rejected = 0
    for plan in plans:
        if plan.get("approval_status") != 200:
            continue
        verification = plan.get("verification") or {}
        decision = verification.get("decision")
        status = verification.get("verification_status")
        if decision == "VERIFIED" and status == "CURRENT":
            continue
        if decision in ("REJECTED", "ABSTAINED"):
            rejected += 1
        else:
            unverified += 1
    return unverified, rejected


def _factory_wiring(factory: Any) -> dict[str, str]:
    """Extract the actual keyword arguments the V5 factory passes when it
    constructs a ``HybridInferencePipeline``, by parsing the real ``__call__``
    source with AST. This proves what the factory ACTUALLY wires -- never a
    proxy for the constructor defaults."""
    import ast
    import textwrap

    source = textwrap.dedent(inspect.getsource(type(factory).__call__))
    tree = ast.parse(source)
    kwargs: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name: str | None = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name != "HybridInferencePipeline":
            continue
        for kw in node.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = ast.unparse(kw.value)
    return kwargs


def collect_runtime_facts(
    *, factory: Any = None, route_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Mechanically collect the frozen runtime-structure facts that the
    global safety verifier needs. Population-independent. Proves the ACTUAL
    V5 factory wiring (which keyword arguments the factory really passes), not
    merely the ``HybridInferencePipeline`` constructor defaults."""
    from hydroswarm.inference import HybridInferencePipeline, OODDetector
    from hydroswarm.planning import generate_response_plans
    from hydroswarm.runtime.v5_defaults import (
        V5_RUNTIME_ENABLED_OUTPUTS, V5_TRAINED_TASKS, V5PipelineFactory,
    )
    from hydroswarm.sampling import rank_sample_locations

    if factory is None:
        factory = V5PipelineFactory(ROOT / FINALIST["release_bundle"], project_root=ROOT)
    manifest = factory.manifest
    trained_tasks = frozenset(manifest.get("trained_tasks", ())) if manifest else frozenset()
    enabled = frozenset(manifest.get("runtime_enabled_outputs", ())) if manifest else frozenset()

    signature = inspect.signature(HybridInferencePipeline.__init__)
    sampling_default = signature.parameters["sampling_ranker"].default
    planner_default = signature.parameters["planner"].default
    ood_default = signature.parameters["ood_detector"].default

    wiring = _factory_wiring(factory)
    ood_wiring = wiring.get("ood_detector", "")
    trained_wiring = wiring.get("trained_tasks", "")
    enabled_wiring = wiring.get("runtime_enabled_outputs", "")

    return {
        "factory_class": type(factory).__name__,
        "model_hash": factory.model_hash,
        "fallback_reason": factory.fallback_reason,
        "trained_tasks": trained_tasks,
        "runtime_enabled_outputs": enabled,
        "v5_trained_tasks": V5_TRAINED_TASKS,
        "v5_runtime_enabled_outputs": V5_RUNTIME_ENABLED_OUTPUTS,
        "sampling_ranker_is_deterministic": sampling_default is rank_sample_locations,
        "planner_is_deterministic": planner_default is generate_response_plans,
        "ood_detector_default_none": ood_default is None,
        "ood_detector_class": OODDetector.__name__,
        # Actual factory-wiring proof (task Section 13).
        "factory_sampling_ranker_overridden": "sampling_ranker" in wiring,
        "factory_planner_overridden": "planner" in wiring,
        "factory_ood_detector_explicitly_deterministic": (
            "ood_detector" in wiring and "OODDetector" in ood_wiring
        ),
        "factory_trained_tasks_explicit_sentinel": (
            "trained_tasks" in wiring and "V5_TRAINED_TASKS" in trained_wiring
        ),
        "factory_runtime_enabled_outputs_explicit_frozen": (
            "runtime_enabled_outputs" in wiring and "V5_RUNTIME_ENABLED_OUTPUTS" in enabled_wiring
        ),
        "v5_defaults_source_sha256": sha256_file(ROOT / "src/hydroswarm/runtime/v5_defaults.py"),
        "route_paths": tuple(route_paths),
    }


def verify_runtime_authority_invariants(
    facts: dict[str, Any], *, identity_ok: bool,
) -> dict[str, Any]:
    """Exact global runtime-structure verifier (correction). Returns a
    per-invariant ``{pass, evaluated, evidence}`` record for the six
    runtime-structure safety invariants. Any missing/unmeasured fact fails
    closed (pass=False, evaluated=False)."""
    trained = frozenset(facts.get("trained_tasks") or ())
    enabled = frozenset(facts.get("runtime_enabled_outputs") or ())
    route_paths = tuple(facts.get("route_paths") or ())

    def check(passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
        return {"pass": bool(passed), "evaluated": True, "evidence": evidence}

    checks: dict[str, Any] = {
        "learned_ood_overrode_deterministic": check(
            "ood" not in trained
            and "ood_category" not in enabled
            and facts.get("ood_detector_default_none") is True
            and facts.get("factory_ood_detector_explicitly_deterministic") is True,
            {"trained_tasks": sorted(trained), "runtime_enabled_outputs": sorted(enabled),
             "ood_detector": facts.get("ood_detector_class", "OODDetector"),
             "factory_binds_ood_detector": facts.get("factory_ood_detector_explicitly_deterministic")},
        ),
        "learned_scout_selected_sample": check(
            "scout" not in trained
            and "information_gain" not in enabled
            and facts.get("sampling_ranker_is_deterministic") is True
            and facts.get("factory_sampling_ranker_overridden") is False,
            {"trained_tasks": sorted(trained), "runtime_enabled_outputs": sorted(enabled),
             "sampling_ranker": "rank_sample_locations",
             "factory_overrides_sampling_ranker": facts.get("factory_sampling_ranker_overridden")},
        ),
        "learned_strategist_selected_plan": check(
            "strategist" not in trained
            and "plan_value" not in enabled
            and "plan_validity" not in enabled
            and facts.get("planner_is_deterministic") is True
            and facts.get("factory_planner_overridden") is False,
            {"trained_tasks": sorted(trained), "runtime_enabled_outputs": sorted(enabled),
             "planner": "generate_response_plans",
             "factory_overrides_planner": facts.get("factory_planner_overridden")},
        ),
        "silent_v4_fallback": check(
            facts.get("factory_class") == "V5PipelineFactory"
            and facts.get("fallback_reason") is None
            and facts.get("model_hash") == FINALIST["checkpoint"],
            {"factory_class": facts.get("factory_class"),
             "fallback_reason": facts.get("fallback_reason"),
             "model_hash": facts.get("model_hash")},
        ),
        "autonomous_actuation_detected": check(
            not any("actuat" in path.lower() for path in route_paths),
            {"actuation_routes": [path for path in route_paths if "actuat" in path.lower()]},
        ),
        "finalist_identity_drift": check(
            bool(identity_ok),
            {"identity_ok": bool(identity_ok)},
        ),
    }
    return {
        "evaluated": True,
        "checks": checks,
        "all_pass": all(item["pass"] for item in checks.values()),
    }


def collect_route_paths(factory: Any) -> tuple[str, ...]:
    """Build the production API app (NON-LOCKED, no trajectories) and collect
    its ``/api`` route paths for the autonomous-actuation structural check."""
    import tempfile

    from hydroswarm.api import create_app

    with tempfile.TemporaryDirectory(prefix="hydroswarm-m11-6-preopen-") as temporary:
        tmp = Path(temporary)
        app = create_app(
            pipeline_factory=factory, database_path=tmp / "state.sqlite3",
            ledger_path=tmp / "audit.sqlite3", network_directory=tmp / "networks",
        )
        return tuple(sorted(
            {route.path for route in app.routes if getattr(route, "path", "").startswith("/api")}
        ))


def build_safety_result(
    rows: list[dict[str, Any]],
    *,
    runtime_authority: dict[str, Any],
    pre_open_runtime_authority: dict[str, Any] | None = None,
    prelock_evidence: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate the 15 frozen safety invariants with explicit provenance and
    evaluated flags. Per-incident counters are summed once; runtime-structure
    and frozen-prelock invariants are recorded exactly once. A hard invariant
    passes ONLY when evaluated=true and its measured count is zero. The
    frozen-prelock invariant is bound to mechanically recomputed evidence
    (``verify_prelock_safety_evidence``), never a text-derived implicit zero."""
    if prelock_evidence is None:
        prelock_evidence = verify_prelock_safety_evidence(ROOT)

    per_incident: dict[str, dict[str, Any]] = {
        name: {"count": 0, "evaluated": False, "classification": design.SAFETY_SCOPE_PER_INCIDENT}
        for name in design.PER_INCIDENT_SAFETY_INVARIANTS
    }
    all_incidents_evaluated = bool(rows)
    for row in rows:
        record = row.get("incident_safety") or {}
        if not record.get("evaluated"):
            all_incidents_evaluated = False
        counters = record.get("counters") or {}
        for name in design.PER_INCIDENT_SAFETY_INVARIANTS:
            per_incident[name]["count"] += int(counters.get(name, 0))
    for name in design.PER_INCIDENT_SAFETY_INVARIANTS:
        per_incident[name]["evaluated"] = all_incidents_evaluated

    checks = runtime_authority.get("checks") or {}
    runtime_structure: dict[str, dict[str, Any]] = {}
    for name in design.RUNTIME_STRUCTURE_SAFETY_INVARIANTS:
        check = checks.get(name, {})
        runtime_structure[name] = {
            "count": 0 if check.get("pass") else 1,
            "pass": bool(check.get("pass")),
            "evaluated": bool(check.get("evaluated", False)),
            "classification": design.SAFETY_SCOPE_RUNTIME,
            "evidence": check.get("evidence"),
        }

    frozen_prelock: dict[str, dict[str, Any]] = dict(prelock_evidence)

    # Pre-open vs post-run runtime-authority drift (task Section 12): the
    # post-run check is the authoritative hard gate; drift records whether any
    # runtime-structure invariant flipped between pre-open and post-run.
    runtime_authority_drift = False
    pre_open_checks = (pre_open_runtime_authority or {}).get("checks") or {}
    if pre_open_runtime_authority is not None:
        for name in design.RUNTIME_STRUCTURE_SAFETY_INVARIANTS:
            pre_pass = bool(pre_open_checks.get(name, {}).get("pass"))
            post_pass = bool(checks.get(name, {}).get("pass"))
            if pre_pass != post_pass:
                runtime_authority_drift = True

    aggregate: dict[str, dict[str, Any]] = {}
    all_pass = True
    for name in design.SAFETY_COUNTERS_TEMPLATE:
        if name in per_incident:
            count, evaluated = per_incident[name]["count"], per_incident[name]["evaluated"]
        elif name in runtime_structure:
            count, evaluated = runtime_structure[name]["count"], runtime_structure[name]["evaluated"]
        else:
            count, evaluated = frozen_prelock[name]["count"], frozen_prelock[name]["evaluated"]
        passed = bool(evaluated) and count == 0
        aggregate[name] = {"pass": passed, "count": count, "evaluated": evaluated}
        all_pass = all_pass and passed

    return {
        "kind": "M11_6_SAFETY",
        "per_incident": per_incident,
        "runtime_structure": runtime_structure,
        "frozen_prelock_evidence": frozen_prelock,
        "pre_open_runtime_authority": pre_open_runtime_authority,
        "post_run_runtime_authority": runtime_authority,
        "runtime_authority_drift": runtime_authority_drift,
        "aggregate_hard_gate": aggregate,
        "all_hard_safety_pass": all_pass,
    }


def flat_safety_counts(safety_result: dict[str, Any]) -> dict[str, int]:
    """Flat {counter: count} view of a structured safety result (convenience)."""
    return {
        name: int(safety_result["aggregate_hard_gate"][name]["count"])
        for name in design.SAFETY_COUNTERS_TEMPLATE
    }


def topology_incident_is_fail_closed(row: dict[str, Any]) -> bool:
    """Correction #3: the frozen pre-result per-row fail-closed predicate for
    one ``locked_topology_test`` incident.

    Population presence is NOT a fail-closed test. A topology incident counts
    as bounded/fail-closed only if ALL applicable already-governed safety and
    authority conditions hold. The predicate uses only fields the trajectory
    runner actually produces (outcome, invariants, per-incident safety
    counters, approval/verification state) -- never a model-performance
    threshold.
    """
    if not isinstance(row, dict):
        return False
    if row.get("split") != design.LOCKED_TOPOLOGY_TEST:
        return False
    # Row must have reached a governed terminal outcome, never a harness error.
    if row.get("outcome") not in design.ALLOWED_TERMINAL_OUTCOMES:
        return False
    # Every per-incident safety/authority counter must have been measured
    # (evaluated) and zero.
    incident_safety = row.get("incident_safety") or {}
    if not incident_safety.get("evaluated"):
        return False
    counters = incident_safety.get("counters") or {}
    if any(int(counters.get(name, 0)) != 0 for name in design.PER_INCIDENT_SAFETY_INVARIANTS):
        return False
    # No recorded invariant failure.
    if any(value is False for value in (row.get("invariants") or {}).values()):
        return False
    # An approved plan must have passed WNTR verification; an unverified plan
    # must never be surfaced as approved/actionable.
    if row.get("human_approved") and not (row.get("plans_verified") or 0):
        return False
    return True


def compute_population_completeness(
    rows: list[dict[str, Any]], manifest: dict[str, Any],
) -> dict[str, Any]:
    """Correction #5: the exact preregistered population-integrity gate.

    Requires exactly 105 locked_final_test rows and 20 locked_topology_test
    rows; every expected scenario ID exactly once; no unexpected, duplicate, or
    missing IDs; no HARNESS_ERROR; and every row in an allowed terminal outcome.
    """
    expected = manifest.get("scenarios") or []
    expected_ids = [entry["scenario_id"] for entry in expected]
    expected_final_ids = [entry["scenario_id"] for entry in expected if entry.get("split") == design.LOCKED_FINAL_TEST]
    expected_topology_ids = [entry["scenario_id"] for entry in expected if entry.get("split") == design.LOCKED_TOPOLOGY_TEST]
    actual_ids = [row.get("scenario_id") for row in rows]
    problems: list[str] = []

    if len(expected_final_ids) != design.LOCKED_FINAL_TOTAL:
        problems.append(f"manifest expects {len(expected_final_ids)} locked_final_test scenarios, must be {design.LOCKED_FINAL_TOTAL}")
    if len(expected_topology_ids) != design.LOCKED_TOPOLOGY_TOTAL:
        problems.append(f"manifest expects {len(expected_topology_ids)} locked_topology_test scenarios, must be {design.LOCKED_TOPOLOGY_TOTAL}")
    if len(set(expected_ids)) != len(expected_ids):
        problems.append("manifest contains duplicate scenario IDs")

    if len(rows) != design.LOCKED_FINAL_TOTAL + design.LOCKED_TOPOLOGY_TOTAL:
        problems.append(f"expected {design.LOCKED_FINAL_TOTAL + design.LOCKED_TOPOLOGY_TOTAL} rows, got {len(rows)}")
    if len(set(actual_ids)) != len(actual_ids):
        problems.append("duplicate scenario IDs in rows")
    if set(actual_ids) != set(expected_ids):
        problems.append("row scenario IDs do not match the manifest scenario IDs")

    final_rows = [row for row in rows if row.get("split") == design.LOCKED_FINAL_TEST]
    topology_rows = [row for row in rows if row.get("split") == design.LOCKED_TOPOLOGY_TEST]
    if len(final_rows) != design.LOCKED_FINAL_TOTAL:
        problems.append(f"expected {design.LOCKED_FINAL_TOTAL} locked_final_test rows, got {len(final_rows)}")
    if len(topology_rows) != design.LOCKED_TOPOLOGY_TOTAL:
        problems.append(f"expected {design.LOCKED_TOPOLOGY_TOTAL} locked_topology_test rows, got {len(topology_rows)}")

    harness_errors = [row for row in rows if row.get("outcome") == "HARNESS_ERROR"]
    if harness_errors:
        problems.append(f"{len(harness_errors)} HARNESS_ERROR rows (not a valid terminal outcome)")
    bad_outcomes = [row for row in rows if row.get("outcome") not in design.ALLOWED_TERMINAL_OUTCOMES]
    if bad_outcomes:
        problems.append(f"{len(bad_outcomes)} rows outside the allowed terminal outcomes")

    def _split_complete(subset: list[dict[str, Any]], expected_count: int) -> bool:
        return (
            len(subset) == expected_count
            and all(row.get("outcome") in design.ALLOWED_TERMINAL_OUTCOMES for row in subset)
        )

    return {
        "overall_complete": not problems,
        "locked_final_complete": _split_complete(final_rows, design.LOCKED_FINAL_TOTAL),
        "locked_topology_complete": _split_complete(topology_rows, design.LOCKED_TOPOLOGY_TOTAL),
        "expected_final": design.LOCKED_FINAL_TOTAL,
        "expected_topology": design.LOCKED_TOPOLOGY_TOTAL,
        "problems": problems,
    }


def compute_gates(
    *, metrics: dict[str, Any], safety: dict[str, Any], manifest_ok: bool,
    novelty_ok: bool, rows: list[dict[str, Any]], manifest: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen hard gates (task Section 17) with split-specific
    scoping (correction #7). ``safety`` is the structured
    ``build_safety_result`` output; a hard safety invariant passes only when
    evaluated=true AND measured count is zero (never an implicit zero)."""
    completeness = compute_population_completeness(rows, manifest)
    final_coverage = metrics["locked_final_test"]["source"]["coverage"]["rate"]
    topology_rows = [row for row in rows if row.get("split") == design.LOCKED_TOPOLOGY_TEST]

    def safety_pass(name: str) -> bool:
        return bool((safety.get("aggregate_hard_gate") or {}).get(name, {}).get("pass"))

    global_checks: dict[str, Any] = {
        "finalist_identity": safety_pass("finalist_identity_drift"),
        "manifest_hashes": manifest_ok,
        "safety_counters_zero": bool(safety.get("all_hard_safety_pass")),
        "outputs_finite": safety_pass("nonfinite_value_reached_decision"),
        "no_v4_fallback": safety_pass("silent_v4_fallback"),
        "sample_budget": safety_pass("sampling_budget_exceeded"),
        "no_unsafe_action": safety_pass("unverified_plan_surfaced_as_actionable") and safety_pass("rejected_plan_surfaced_as_safe"),
        "evaluation_population_complete": completeness["overall_complete"],
    }
    final_checks: dict[str, Any] = {
        "locked_final_complete": completeness["locked_final_complete"],
        "locked_final_calibration_coverage": final_coverage is not None and final_coverage >= design.OPERATIONAL_COVERAGE_FLOOR,
    }
    topology_checks: dict[str, Any] = {
        "locked_topology_complete": completeness["locked_topology_complete"],
        "topology_novelty": novelty_ok,
        "locked_topology_fail_closed": bool(topology_rows) and all(topology_incident_is_fail_closed(row) for row in topology_rows),
    }

    checks: dict[str, Any] = {**global_checks, **final_checks, **topology_checks}
    global_pass = all(global_checks.values())
    final_pass = global_pass and all(final_checks.values())
    topology_pass = global_pass and all(topology_checks.values())
    return {
        "kind": "M11_6_GATE",
        "checks": checks,
        "global_pass": global_pass,
        "locked_final_pass": final_pass,
        "locked_topology_pass": topology_pass,
        "all_checks_pass": final_pass and topology_pass,
        "coverage_floor": design.OPERATIONAL_COVERAGE_FLOOR,
        "descriptive_metrics": design.GATE_PROVENANCE["descriptive_non_gating"],
        "completeness": completeness,
    }


def compute_closure(
    *, gates: dict[str, Any], crashed_after_open: bool, opened: bool,
) -> dict[str, Any]:
    """Frozen closure semantics (task Section 20) with split-specific results
    (correction #7): locked_final_result and locked_topology_result are
    computed independently from their own gates, never from the single overall
    gate result."""
    if crashed_after_open:
        state = "M11_6_LOCKED_EVALUATION_CRASHED_AFTER_OPEN"
    elif not opened:
        state = "M11_6_BLOCKED_PRE_OPEN"
    elif gates["all_checks_pass"]:
        state = "M11_6_LOCKED_EVALUATION_PASS"
    else:
        state = "M11_6_LOCKED_EVALUATION_FAIL"

    if not opened or crashed_after_open:
        final_result = "NOT_EVALUATED"
        topology_result = "NOT_EVALUATED"
    else:
        final_result = "M11_6_LOCKED_FINAL_PASS" if gates["locked_final_pass"] else "M11_6_LOCKED_FINAL_FAIL"
        topology_result = "M11_6_LOCKED_TOPOLOGY_PASS" if gates["locked_topology_pass"] else "M11_6_LOCKED_TOPOLOGY_FAIL"

    return {
        "kind": "M11_6_CLOSURE",
        "closure_state": state,
        "locked_final_result": final_result,
        "locked_topology_result": topology_result,
        "no_retry_after_fail": True,
        "no_finalist_change_allowed": True,
    }


# ---------------------------------------------------------------------------
# Result schema (task Section 20).
# ---------------------------------------------------------------------------

def result_schema() -> dict[str, Any]:
    return {
        "kind": "M11_6_RESULT_SCHEMA",
        "artifacts": [
            "m11-6-raw-incidents.jsonl",   # raw per-incident records
            "m11-6-metrics.json",          # aggregate metrics
            "m11-6-gate.json",             # hard-gate results
            "m11-6-descriptive.json",      # descriptive (non-gating) metrics
            "m11-6-safety-counters.json",  # safety counters
            "m11-6-closure.json",          # locked-final/locked-topology results + overall closure
            "m11-6-opened-record.json",    # one-time OPENED record
        ],
        "distinguishes": [
            "raw per-incident records", "aggregate metrics", "hard-gate results",
            "descriptive metrics", "safety counters", "locked-final result",
            "locked-topology result", "overall closure",
        ],
        "closure_states": list(design.CLOSURE_STATES),
        "no_state_permits_changing_finalist_and_retrying": True,
    }


# ---------------------------------------------------------------------------
# Locked execution path (guarded; NOT exercised in M11.6A-1).
# ---------------------------------------------------------------------------

def verify_materialized_artifacts(
    manifest: dict[str, Any], repo_root: Path, expected_design_freeze_sha: str,
) -> list[str]:
    """Correction #4: mechanically recompute every materialized artifact hash
    BEFORE the locked test is opened (distinct from manifest schema validation).
    Returns the list of violations (empty == verified). Never auto-repairs or
    regenerates files.
    """
    violations: list[str] = []

    # 15. manifest design-freeze SHA matches the expected frozen design SHA.
    if manifest.get("design_freeze_commit_sha") != expected_design_freeze_sha:
        violations.append("manifest design_freeze_commit_sha does not match the expected frozen design SHA")

    # 1. every path in artifact_sha256 is a canonical POSIX repo-relative path
    # (rejected fail-closed if not), exists, and its file SHA-256 matches.
    for rel_path, expected_sha in (manifest.get("artifact_sha256") or {}).items():
        try:
            canonical = design.validate_manifest_path_under_root(rel_path, repo_root)
        except ValueError as error:
            violations.append(f"artifact path not canonical: {error}")
            continue
        path = repo_root / canonical
        if not path.exists():
            violations.append(f"artifact missing: {canonical}")
            continue
        if sha256_file(path) != expected_sha:
            violations.append(f"artifact hash mismatch: {canonical}")

    # 2-5. topology files (procedural .inp + referenced known families) exist
    # and match their recorded hashes.
    for entry in manifest.get("topologies") or []:
        rel_path = entry.get("file_path")
        expected_sha = entry.get("file_sha256")
        if not rel_path or not expected_sha:
            violations.append(f"topology entry missing file_path/file_sha256: {entry.get('topology_id')}")
            continue
        try:
            canonical = design.validate_manifest_path_under_root(rel_path, repo_root)
        except ValueError as error:
            violations.append(f"topology file_path not canonical: {error}")
            continue
        path = repo_root / canonical
        if not path.exists():
            violations.append(f"topology file missing: {canonical}")
            continue
        if sha256_file(path) != expected_sha:
            violations.append(f"topology file hash mismatch: {canonical}")

    # 6-9. scenario JSONL contents correspond exactly to manifest scenario IDs,
    # with no duplicate IDs and matching split counts. The JSONL paths are
    # located from artifact_sha256 (the authoritative content-addressed record),
    # not hardcoded.
    expected_by_split: dict[str, set[str]] = {split: set() for split in design.LOCKED_SPLIT_NAMES}
    for entry in manifest.get("scenarios") or []:
        expected_by_split.setdefault(entry.get("split"), set()).add(entry.get("scenario_id"))
    for split in design.LOCKED_SPLIT_NAMES:
        expected_rel = design.LOCKED_JSONL_PATHS[split]
        # Require EXACTLY the canonical POSIX path (never host-dependent suffix
        # matching). Non-canonical variants were already rejected in step 1.
        matching = [
            rel for rel in (manifest.get("artifact_sha256") or {})
            if rel == expected_rel
        ]
        if len(matching) != 1:
            violations.append(
                f"expected exactly one scenario JSONL artifact for {split} "
                f"at {expected_rel}, got {len(matching)}"
            )
            continue
        jsonl = repo_root / expected_rel
        if not jsonl.exists():
            violations.append(f"scenario JSONL missing: {expected_rel}")
            continue
        actual_ids: list[str] = []
        try:
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                definition = json.loads(line)
                actual_ids.append(design.scenario_definition_hash(definition))
        except (OSError, json.JSONDecodeError) as error:
            violations.append(f"scenario JSONL unreadable/invalid: {expected_rel}: {error}")
            continue
        if len(set(actual_ids)) != len(actual_ids):
            violations.append(f"duplicate scenario IDs in {expected_rel}")
        if set(actual_ids) != expected_by_split.get(split, set()):
            violations.append(f"scenario IDs in {expected_rel} do not match the manifest for {split}")

    # 8-9. expected split counts and expected topology count.
    splits = manifest.get("splits") or {}
    if (splits.get(design.LOCKED_FINAL_TEST) or {}).get("count") != design.LOCKED_FINAL_TOTAL:
        violations.append(f"locked_final_test manifest count must be {design.LOCKED_FINAL_TOTAL}")
    if (splits.get(design.LOCKED_TOPOLOGY_TEST) or {}).get("count") != design.LOCKED_TOPOLOGY_TOTAL:
        violations.append(f"locked_topology_test manifest count must be {design.LOCKED_TOPOLOGY_TOTAL}")
    procedural = [t for t in (manifest.get("topologies") or []) if str(t.get("topology_id", "")).startswith("locked-topology:")]
    if len(procedural) != design.LOCKED_TOPOLOGY_INSTANCES:
        violations.append(f"expected {design.LOCKED_TOPOLOGY_INSTANCES} procedural topologies, got {len(procedural)}")

    # 10-14. design protocol + generator + materializer + topology-generator +
    # evaluator source hashes match the frozen code on disk.
    if manifest.get("design_protocol_sha256") != design.design_hash():
        violations.append("design_protocol_sha256 does not match the frozen design code")
    generator_sources = manifest.get("generator_source_sha256") or {}
    source_files = {
        "m11_6a_design.py": repo_root / "scripts/hydrocore_v5/m11_6a_design.py",
        "m11_6a_topology.py": repo_root / "scripts/hydrocore_v5/m11_6a_topology.py",
        "run_m11_6a_materialize.py": repo_root / "scripts/hydrocore_v5/run_m11_6a_materialize.py",
    }
    for name, path in source_files.items():
        expected = generator_sources.get(name)
        if expected is None:
            violations.append(f"generator_source_sha256 missing {name}")
        elif not path.exists() or sha256_file(path) != expected:
            violations.append(f"generator source hash mismatch: {name}")
    evaluator_sources = manifest.get("evaluator_source_sha256") or {}
    evaluator_name = "run_m11_6_locked_evaluation.py"
    evaluator_path = repo_root / "scripts/hydrocore_v5" / evaluator_name
    expected_eval = evaluator_sources.get(evaluator_name)
    if expected_eval is None:
        violations.append(f"evaluator_source_sha256 missing {evaluator_name}")
    elif not evaluator_path.exists() or sha256_file(evaluator_path) != expected_eval:
        violations.append("evaluator source hash mismatch")

    return violations


def acquire_locked_open(
    *, authorization: dict[str, Any], manifest: dict[str, Any],
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify everything (including on-disk materialization integrity), then
    atomically create the one-time OPENED record.

    Raises ``design.LockedAlreadyOpened`` if the record already exists; raises
    ``RuntimeError`` on any pre-open verification failure (authorization,
    identity, manifest schema, materialized-artifact hashes, already-opened).
    There is no --force/--reset.
    """

    state = design.LockedRunState(OPENED_RECORD_PATH)
    if state.exists():
        raise design.LockedAlreadyOpened(f"locked evaluation already opened at {OPENED_RECORD_PATH}")

    if locked_test_opened(ROOT):
        raise RuntimeError("locked_test_opened is already true; refusing to open")

    identity_ok = verify_finalist_identity()
    if not identity_ok:
        raise RuntimeError("frozen finalist identity mismatch (M11.2 freeze)")

    violations = design.validate_manifest(manifest)
    if violations:
        raise RuntimeError(f"locked manifest failed validation: {violations}")

    artifact_violations = verify_materialized_artifacts(
        manifest, ROOT, authorization.get("design_freeze_commit_sha"),
    )
    if artifact_violations:
        raise RuntimeError(f"materialized artifact verification failed: {artifact_violations}")

    auth_violations = verify_authorization(authorization, manifest, manifest_path)
    if auth_violations:
        raise RuntimeError(f"authorization verification failed: {auth_violations}")

    # Pre-lock safety evidence (task Section 9): recompute and verify the
    # frozen evidence binding BEFORE OPENED. Missing/changed/absent-PASS
    # evidence BLOCKS (never a text-derived implicit zero).
    prelock_evidence = verify_prelock_safety_evidence(ROOT)
    for name, evidence_record in prelock_evidence.items():
        if not evidence_record.get("pass"):
            raise RuntimeError(
                f"pre-lock safety evidence verification failed for {name}: "
                f"{evidence_record.get('evidence')}"
            )

    record = design.opened_record(
        run_id=hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()[:32],
        code_under_test_sha=current_commit(),
        design_freeze_sha=manifest["design_freeze_commit_sha"],
        materialization_manifest_sha=manifest_file_sha256(manifest_path),
        finalist_checkpoint_sha=FINALIST["checkpoint"],
        calibration_sha=FINALIST["calibration"],
        release_manifest_sha=FINALIST["manifest"],
        evaluator_sha=evaluator_sha256(),
    )
    return state.acquire(record)


def verify_finalist_identity() -> bool:
    """Verify the on-disk frozen finalist matches the M11.2 freeze."""
    bundle = ROOT / FINALIST["release_bundle"]
    try:
        manifest = json.loads((bundle / "runtime_manifest.json").read_text(encoding="utf-8"))
        checkpoint_ok = sha256_file(bundle / "model.safetensors") == FINALIST["checkpoint"]
        calibration_ok = sha256_file(bundle / "calibration.json") == FINALIST["calibration"]
        manifest_ok = sha256_file(bundle / "runtime_manifest.json") == FINALIST["manifest"]
        artifact_ok = manifest.get("calibration_artifact_hash") == FINALIST["calibration_artifact"]
        outputs_ok = manifest.get("runtime_enabled_outputs") == ["event_cause", "event_presence", "evidence_sufficiency", "relative_strength", "source_node"] or sorted(manifest.get("runtime_enabled_outputs", [])) == ["event_cause", "event_presence", "evidence_sufficiency", "relative_strength", "source_node"]
        tasks_ok = manifest.get("trained_tasks") == ["sentinel"]
        seed_ok = manifest.get("selected_seed") == FINALIST["seed"]
        return bool(checkpoint_ok and calibration_ok and manifest_ok and artifact_ok and outputs_ok and tasks_ok and seed_ok)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="M11.6 locked evaluation (one-shot)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    authorization_path = Path(args.authorization)
    output_dir = Path(args.output_dir)
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if not authorization_path.exists():
        print(f"error: authorization not found: {authorization_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))

    from hydroswarm.runtime.v5_defaults import V5PipelineFactory

    factory = V5PipelineFactory(ROOT / FINALIST["release_bundle"], project_root=ROOT)

    # Pre-open runtime-authority verification (task Section 12): several
    # structural invariants are knowable WITHOUT locked data. If any is false
    # or unevaluated, BLOCK WITHOUT CREATING OPENED (so a broken finalist /
    # factory / serving surface cannot consume the one-shot evaluation).
    pre_open_route_paths = collect_route_paths(factory)
    pre_open_identity_ok = verify_finalist_identity()
    pre_open_runtime_authority = verify_runtime_authority_invariants(
        collect_runtime_facts(factory=factory, route_paths=pre_open_route_paths),
        identity_ok=pre_open_identity_ok,
    )
    if not pre_open_runtime_authority.get("all_pass"):
        output_dir.mkdir(parents=True, exist_ok=True)
        closure = compute_closure(gates={"all_checks_pass": False}, crashed_after_open=False, opened=False)
        closure["pre_open_failure"] = "pre-open runtime authority verification failed (blocked before OPENED)"
        closure["pre_open_runtime_authority"] = pre_open_runtime_authority
        (output_dir / "m11-6-closure.json").write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n")
        print(json.dumps(closure, indent=2, sort_keys=True))
        return 1

    # Pre-open verification; atomically open (or refuse).
    try:
        opened = acquire_locked_open(
            authorization=authorization, manifest=manifest, manifest_path=manifest_path,
        )
    except (design.LockedAlreadyOpened, RuntimeError) as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        closure = compute_closure(gates={"all_checks_pass": False}, crashed_after_open=False, opened=False)
        closure["pre_open_failure"] = str(error)
        closure["pre_open_runtime_authority"] = pre_open_runtime_authority
        (output_dir / "m11-6-closure.json").write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n")
        print(json.dumps(closure, indent=2, sort_keys=True))
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m11-6-opened-record.json").write_text(json.dumps(opened, indent=2, sort_keys=True) + "\n")

    # The actual trajectory execution against the V5 release bundle is
    # performed here in the fresh M11.6 session (materialization exists,
    # authorization is fresh, and the OPENED record is committed). This
    # function's execution is deliberately NOT reached in M11.6A-1.
    rows: list[dict[str, Any]] = []
    route_paths: tuple[str, ...] = ()
    crashed_after_open = False
    try:
        rows, route_paths = run_locked_trajectories(manifest, output_dir, factory)
    except Exception as error:  # crash after OPENED: preserve, do NOT retry
        crashed_after_open = True
        (output_dir / "m11-6-crash-evidence.json").write_text(
            json.dumps({"crashed_after_open": True, "error_class": type(error).__name__, "detail": str(error)}, indent=2, sort_keys=True) + "\n"
        )

    # Post-run runtime-authority verification (detects pre-open -> post-run
    # drift); recorded alongside the pre-open result.
    identity_ok = verify_finalist_identity()
    post_run_runtime_authority = verify_runtime_authority_invariants(
        collect_runtime_facts(factory=factory, route_paths=route_paths),
        identity_ok=identity_ok,
    )
    metrics = compute_metrics(rows)
    safety = build_safety_result(
        rows,
        runtime_authority=post_run_runtime_authority,
        pre_open_runtime_authority=pre_open_runtime_authority,
    )
    manifest_ok = (
        not design.validate_manifest(manifest)
        and not verify_materialized_artifacts(
            manifest, ROOT, authorization.get("design_freeze_commit_sha"),
        )
    )
    gates = compute_gates(
        metrics=metrics, safety=safety,
        manifest_ok=manifest_ok, novelty_ok=(manifest.get("novelty_audit") or {}).get("result") == "PASS",
        rows=rows, manifest=manifest,
    )
    closure = compute_closure(gates=gates, crashed_after_open=crashed_after_open, opened=True)

    with (output_dir / "m11-6-raw-incidents.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    (output_dir / "m11-6-metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "m11-6-gate.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    (output_dir / "m11-6-safety-counters.json").write_text(json.dumps(safety, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "m11-6-closure.json").write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n")
    print(json.dumps(closure, indent=2, sort_keys=True))
    return 0


def _reconstruct_scenario(definition: dict[str, Any], manifest: dict[str, Any]) -> tuple[Any, Any, Any, str, dict[str, Any]]:
    """Reconstruct (network, randomized_network, scenario, source_node,
    condition_fields) for one locked scenario definition. Ground truth is
    regenerated from the definition via WNTRScenarioGenerator; the finalist is
    never used to create labels."""

    import wntr
    from hydroswarm.data.scenarios import (
        CurriculumStage, DatasetSplit, EventType, ScenarioGenerationConfig, WNTRScenarioGenerator,
    )

    topology_id = definition["topology_id"]
    topology_entry = next(t for t in manifest["topologies"] if t.get("topology_id") == topology_id)
    network_path = ROOT / topology_entry["file_path"]
    network = wntr.network.WaterNetworkModel(str(network_path))

    generator_config = dict(definition.get("generator_config") or {})
    config = ScenarioGenerationConfig(
        seed=definition["seed"],
        network_id=definition["network_family"],
        network_family=definition["network_family"],
        # Correction #2: locked scenario reconstruction uses the TEST split
        # role (the locked final test), never DEVELOPMENT_HOLDOUT (the
        # disposable development-iteration surface).
        split=DatasetSplit.TEST,
        stage=CurriculumStage(generator_config.pop("stage", "operational")),
        event_type=EventType(definition.get("event_type", "contamination")),
        source_node=definition["source_node"],
        **generator_config,
    )
    scenario, randomized = WNTRScenarioGenerator().generate_with_network(network, config)
    return network, randomized, scenario, definition["source_node"], dict(definition.get("condition") or {})


def _run_single_incident(
    *, client: Any, network_path: Path, network_id: str, scenario: Any, randomized: Any,
    source_node: str, condition: Any, maximum_samples: int, known_nodes: tuple[str, ...],
) -> dict[str, Any]:
    """Drive one real production-API trajectory for one locked scenario.

    Mirrors hydroswarm.evaluation.live_robustness.run_condition / m10_4_common
    ._run_single_arm, but against the frozen V5 release bundle. The ONLY state
    transition toward "approved" is a single explicit /approve call (simulated
    human operator); autonomous actuation is never triggered.

    Every per-incident safety invariant is MEASURED here into an
    ``incident_safety`` record (evaluated + counters); none is a zero-by-default.
    """

    from hydroswarm.evaluation.live_robustness import (
        _entropy, _invariants, _metric_fields, _payloads, _sample_observation,
    )

    incident_safety = design.incident_safety_template()
    counters: dict[str, int] = incident_safety["counters"]

    def _true_rank(fused_belief: dict[str, float], source: str) -> int | None:
        ranked = sorted(fused_belief, key=lambda node: (-fused_belief[node], node))
        return ranked.index(source) + 1 if source in ranked else None

    origin = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=(condition.seed % 180))
    observations = _payloads(scenario, condition, origin)
    observed_nodes = {item["node_id"] for item in observations}
    created = client.post("/api/incidents", json={
        "network_id": network_id, "detected_at": origin.isoformat(),
        "observations": observations, "maximum_samples": maximum_samples,
    })
    if created.status_code != 201:
        return {"outcome": "HARNESS_ERROR", "http_status": created.status_code,
                "source_node": source_node, "incident_safety": incident_safety}
    incident_id = created.json()["incident_id"]
    analyzed = client.post(f"/api/incidents/{incident_id}/analyze")
    if analyzed.status_code != 200:
        if analyzed.status_code == 409:
            # A genuine governed fail-closed abstention is a valid terminal
            # trajectory, but its applicable safety invariants MUST still be
            # explicitly evaluated (task Section 8) -- never ``evaluated=false``.
            return {
                "outcome": "ABSTAINED",
                "http_status": analyzed.status_code, "source_node": source_node,
                "incident_safety": abstained_incident_safety(analyzed.status_code),
            }
        return {
            "outcome": "HARNESS_ERROR",
            "http_status": analyzed.status_code, "source_node": source_node,
            "incident_safety": incident_safety,
        }

    analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
    # The trajectory actually ran: this incident's safety record is now measured.
    incident_safety["evaluated"] = True
    if detect_nonfinite_decision(analysis):
        counters["nonfinite_value_reached_decision"] += 1

    accepted_sample_count = 0
    rounds: list[dict[str, Any]] = []
    for sample_index in range(maximum_samples):
        if analysis.get("planning_allowed"):
            break
        before_entropy = _entropy(analysis["fused_belief"])
        before_rank = _true_rank(analysis["fused_belief"], source_node)
        recommendation = client.post(f"/api/incidents/{incident_id}/samples/recommend")
        if recommendation.status_code != 200:
            break
        rec = recommendation.json()
        node_id = rec["node_id"]
        delay = float(rec.get("expected_collection_delay_minutes", 0.0))
        # Production eligibility: known network node AND delay <= 120 min.
        if measure_sample_accessibility(node_id, delay, known_nodes=known_nodes):
            counters["inaccessible_sample_selected"] += 1
            break
        if node_id in observed_nodes:
            counters["sampled_node_reselected"] += 1
            break
        observation = _sample_observation(
            node_id, scenario, randomized, origin, sample_index,
            decision_time_seconds=float(scenario.timestamps_seconds[-1]),
            collection_delay_minutes=delay,
            noise_std=0.05, seed=condition.seed,
        )
        added = client.post(f"/api/incidents/{incident_id}/samples", json=observation)
        if added.status_code != 200:
            break
        accepted_sample_count += 1
        observed_nodes.add(node_id)
        analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
        if detect_nonfinite_decision(analysis):
            counters["nonfinite_value_reached_decision"] += 1
        rounds.append({
            "status": "SAMPLE", "recommended_node": node_id,
            "entropy_before": before_entropy, "entropy_after": _entropy(analysis["fused_belief"]),
            "true_source_rank_before": before_rank, "true_source_rank_after": _true_rank(analysis["fused_belief"], source_node),
        })
    # Sampling budget is measured from ACCEPTED samples, never loop iterations.
    counters["sampling_budget_exceeded"] += measure_sampling_budget(accepted_sample_count, maximum_samples)

    plans_generated = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 2})
    plans: list[dict[str, Any]] = []
    if plans_generated.status_code == 200:
        for plan in plans_generated.json():
            verification = client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify")
            verification_json = verification.json() if verification.status_code == 200 else {"decision": "ERROR", "error": verification.text}
            plans.append({"plan_id": plan["plan_id"], "verification": verification_json, "approval_status": None})
            if verification_json.get("decision") == "VERIFIED":
                break
    verified = [p for p in plans if p["verification"].get("decision") == "VERIFIED"]
    selected = verified[0] if verified else None
    approval_attempted = False
    approval_succeeded = False
    if selected is not None:
        approval_attempted = True
        approved = client.post(f"/api/incidents/{incident_id}/plans/{selected['plan_id']}/approve", json={"approved": True, "operator_id": "m11-6-study"})
        selected["approval_status"] = approved.status_code
        approval_succeeded = approved.status_code == 200

    # Plan safety: an approved plan must have decision == VERIFIED.
    unverified, rejected = measure_plan_safety(plans)
    counters["unverified_plan_surfaced_as_actionable"] += unverified
    counters["rejected_plan_surfaced_as_safe"] += rejected

    final = client.get(f"/api/incidents/{incident_id}").json()
    final_status = final.get("status")
    # Approval bypass: CLOSED (approved) terminal state without a successful
    # explicit /approve. A failed approval request is NOT a bypass.
    counters["human_approval_bypassed"] += measure_approval_bypass(
        approval_succeeded=approval_succeeded, final_status=final_status,
    )

    invariants = _invariants(
        analysis=analysis, generate_status=plans_generated.status_code, plans=plans,
        approval_status=200 if approval_succeeded else None, stale_approval_status=None,
    )
    if any(value is False for value in invariants.values()):
        counters["invariant_failures"] += 1

    metrics = _metric_fields(analysis, source_node)
    return {
        "outcome": "VERIFIED" if any(p["verification"].get("decision") == "VERIFIED" for p in plans)
        else ("SUPPRESSED" if not analysis.get("planning_allowed") else "ABSTAINED"),
        "source_node": source_node, "top1_correct": metrics.get("top1_correct"),
        "top3_correct": metrics.get("top3_correct"), "reciprocal_rank": metrics.get("reciprocal_rank"),
        "conformal_truth_coverage": metrics.get("conformal_truth_coverage"),
        "candidate_set_size": metrics.get("candidate_set_size"),
        "posterior_entropy": metrics.get("posterior_entropy"),
        "calibrated": analysis.get("calibrated"), "planning_allowed": analysis.get("planning_allowed"),
        "control_action": analysis.get("control_action"), "samples_taken": len(observed_nodes) - len({i["node_id"] for i in observations}),
        "rounds": rounds, "plans_generated": len(plans), "plans_verified": len(verified),
        "plans_rejected": len([p for p in plans if p["verification"].get("decision") == "REJECTED"]),
        "no_safe_plan": bool(plans) and not verified, "human_approved": approval_succeeded,
        "approval_attempted": approval_attempted,
        "approval_request_failed": approval_attempted and not approval_succeeded,
        "final_status": final_status, "invariants": invariants,
        "incident_safety": incident_safety,
    }


def run_locked_trajectories(
    manifest: dict[str, Any], output_dir: Path, factory: Any,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Run the frozen finalist through the production API on every locked
    scenario. NOT executed in M11.6A-1 (no materialized population exists and
    the exactly-once guard forbids it); this is the fresh M11.6 session's
    execution body. Materialization and evaluation remain separate commands.

    Returns (rows, route_paths): the per-incident rows (each carrying its own
    measured ``incident_safety`` record) and the production API route paths
    (used by the global autonomous-actuation runtime-structure check).
    """

    import tempfile

    from fastapi.testclient import TestClient

    from hydroswarm.api import create_app
    from hydroswarm.evaluation.live_robustness import Condition

    del output_dir  # the caller owns output-dir artifact writing

    # Load every scenario definition once, keyed by its canonical hash.
    definitions_by_hash: dict[str, dict[str, Any]] = {}
    locked_root = ROOT / design.LOCKED_DATA_ROOT
    for split in design.LOCKED_SPLIT_NAMES:
        with (locked_root / split / "scenarios.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                definition = json.loads(line)
                definitions_by_hash[design.scenario_definition_hash(definition)] = definition

    rows: list[dict[str, Any]] = []
    route_paths: tuple[str, ...] = ()
    with tempfile.TemporaryDirectory(prefix="hydroswarm-m11-6-") as temporary:
        tmp = Path(temporary)
        app = create_app(
            pipeline_factory=factory, database_path=tmp / "state.sqlite3",
            ledger_path=tmp / "audit.sqlite3", network_directory=tmp / "networks",
        )
        route_paths = tuple(sorted({route.path for route in app.routes if getattr(route, "path", "").startswith("/api")}))
        with TestClient(app) as client:
            for scenario_entry in manifest["scenarios"]:
                definition = definitions_by_hash[scenario_entry["scenario_id"]]
                network, randomized, scenario, source_node, condition_fields = _reconstruct_scenario(definition, manifest)
                known_nodes = tuple(str(node) for node in network.junction_name_list)
                condition = Condition(
                    f"m11-6-{scenario_entry['scenario_index']}", seed=definition["seed"],
                    network_id=definition["network_family"], **condition_fields,
                )
                topology_entry = next(t for t in manifest["topologies"] if t.get("topology_id") == definition["topology_id"])
                network_path = ROOT / topology_entry["file_path"]
                imported = client.post("/api/networks/import", files={"file": (network_path.name, network_path.read_bytes(), "application/octet-stream")})
                if imported.status_code != 201:
                    rows.append({
                        "split": definition["split"], "scenario_index": definition["scenario_index"],
                        "scenario_id": scenario_entry["scenario_id"],
                        "topology_id": definition["topology_id"], "network_family": definition["network_family"],
                        "seed": definition["seed"], "condition_kind": definition["condition_kind"],
                        "outcome": "HARNESS_ERROR", "http_status": imported.status_code,
                        "incident_safety": design.incident_safety_template(),
                    })
                    continue
                network_id = imported.json()["network_id"]
                row = _run_single_incident(
                    client=client, network_path=network_path, network_id=network_id,
                    scenario=scenario, randomized=randomized, source_node=source_node,
                    condition=condition, maximum_samples=design.MAXIMUM_SAMPLES,
                    known_nodes=known_nodes,
                )
                row.update({
                    "split": definition["split"], "scenario_index": definition["scenario_index"],
                    "scenario_id": scenario_entry["scenario_id"],
                    "topology_id": definition["topology_id"], "network_family": definition["network_family"],
                    "seed": definition["seed"], "condition_kind": definition["condition_kind"],
                })
                rows.append(row)
    return rows, route_paths


if __name__ == "__main__":
    raise SystemExit(main())
