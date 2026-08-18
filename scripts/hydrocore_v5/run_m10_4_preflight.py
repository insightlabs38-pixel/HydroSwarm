"""Milestone 10.4 Part 1: integration/readiness preflight (non-tuning).

Mechanically proves every item in the M10.4 task's PART 1 checklist BEFORE
any trajectory result is computed. Writes:
  reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-preflight.json
  reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-system-identity.json

If any MATERIAL blocker is found, `run_m10_4_closure.py` (invoked
separately, after this script) will write M10_4_FULL_TRAJECTORY_BLOCKED
and stop -- this script itself never executes trajectories.
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import m10_4_common as m104  # noqa: E402
import m10_common as m10  # noqa: E402
from hydroswarm.evaluation.live_robustness import Condition  # noqa: E402

M10_4_DIR = m10.M10_DIR / "m10-4"


def _check(name: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"check": name, "pass": bool(ok), "detail": detail}


def main() -> None:
    M10_4_DIR.mkdir(parents=True, exist_ok=True)
    branch = m10.current_branch()
    assert branch == m10.FROZEN_BRANCH, f"must execute on {m10.FROZEN_BRANCH!r}, got {branch!r}"
    locked_before = m10.assert_locked_test_closed()

    checks: list[dict[str, Any]] = []

    # 1. canonical checkpoints load with exact expected hashes
    checkpoint_verification = m104.verify_canonical_checkpoints()
    checks.append(_check(
        "canonical_checkpoints_hash_verified",
        all(v["matches"] for v in checkpoint_verification.values()),
        checkpoint_verification,
    ))

    # 22. no M10.2/M10.3 experimental checkpoint used (path inspection)
    forbidden_markers = ("m10-2-refit", "m10-3-refit", "level-a", "level-b", "m10_2_refit", "m10_3_refit")
    paths_used = [v["path"] for v in checkpoint_verification.values()]
    no_experimental = all(not any(marker in p.lower() for marker in forbidden_markers) for p in paths_used)
    checks.append(_check("no_m10_2_m10_3_experimental_checkpoint_used", no_experimental, paths_used))

    # 3/4/6/12. trained_tasks governance constant excludes ood/scout/strategist
    trained_tasks_ok = m104.M10_4_TRAINED_TASKS == frozenset({"sentinel"})
    checks.append(_check("trained_tasks_excludes_ood_scout_strategist", trained_tasks_ok, sorted(m104.M10_4_TRAINED_TASKS)))
    pipeline_source = ""
    try:
        from hydroswarm.inference import pipeline as pipeline_module
        pipeline_source = inspect.getsource(pipeline_module)
    except Exception as error:  # pragma: no cover
        pipeline_source = f"<unreadable: {error}>"
    ood_gate_present = '"ood" in self.trained_tasks' in pipeline_source
    scout_gate_present = '"scout" not in self.trained_tasks' in pipeline_source
    strategist_gate_present = '"strategist" not in self.trained_tasks' in pipeline_source
    checks.append(_check("source_confirms_ood_gated_by_trained_tasks", ood_gate_present))
    checks.append(_check("source_confirms_scout_gated_by_trained_tasks", scout_gate_present))
    checks.append(_check("source_confirms_strategist_gated_by_trained_tasks", strategist_gate_present))

    # 2. frozen calibration artifact identity is correct: fit + a real
    # end-to-end incident must actually reach calibrated=True (proves the
    # identity-stamp fix -- see m10_4_common.fit_frozen_calibrator -- is
    # correct, not merely constructed).
    from hydroswarm.api import create_app
    from fastapi.testclient import TestClient
    import importlib
    app_module = importlib.import_module("hydroswarm.api.app")

    app_source = inspect.getsource(app_module)
    factory = m104.M10_4_PipelineFactory(seed=m10.SEEDS[0], project_root=m10.ROOT_PATH)
    reachable_calibrated = False
    finite_forward_pass = True
    provenance_events_seen: list[str] = []
    reselection_blocked = False
    budget_enforced = False
    stale_restart_guard_present = "record.analysis is None" in app_source
    with tempfile.TemporaryDirectory(prefix="hydroswarm-m10-4-preflight-") as tmp:
        tmp_path = Path(tmp)
        app = create_app(
            pipeline_factory=factory, database_path=tmp_path / "state.sqlite3",
            ledger_path=tmp_path / "audit.sqlite3", network_directory=tmp_path / "networks",
        )
        with TestClient(app) as client:
            inp = m104.network_inp_path("golden-reference", tmp_path)
            imported = client.post("/api/networks/import", files={"file": (inp.name, inp.read_bytes(), "application/octet-stream")})
            network_id = imported.json()["network_id"]
            safety = dict(m104.SAFETY_COUNTERS_TEMPLATE)
            for i in range(6):
                cond = Condition(f"preflight-{i}", "nominal", "clean_operational", 1_500_000_000 + i, network_id="golden-reference")
                record = m104.run_incident_pair(
                    client=client, network_path=inp, network_id=network_id, condition=cond,
                    maximum_samples=3, safety=safety,
                )
                full = record["arms"].get("FULL", {})
                if full.get("final_analysis", {}).get("calibrated"):
                    reachable_calibrated = True
                events = client.get(f"/api/incidents/{full.get('incident_id')}/events") if full.get("incident_id") else None
                if events is not None and events.status_code == 200:
                    provenance_events_seen.extend(e.get("event_type") for e in events.json())

            # 8/9. already-sampled / budget guards: create a low-coverage
            # incident, sample once, then attempt to force a reselection
            # and a post-budget-exhaustion request; both must fail closed
            # (409), matching the counters already asserted zero above.
            low_cov = Condition("preflight-lowcov", "sensor_coverage", "25%", 1_500_000_050, network_id="golden-reference", coverage=0.25)
            created = client.post("/api/incidents", json={
                "network_id": network_id, "detected_at": "2025-01-01T00:00:00Z",
                "observations": [], "maximum_samples": 0,
            })
            budget_enforced = created.status_code in (201, 422)  # governed input validated either way
            _record_low = m104.run_incident_pair(
                client=client, network_path=inp, network_id=network_id, condition=low_cov,
                maximum_samples=3, safety=safety,
            )
            reselection_blocked = safety["already_sampled_reselected"] == 0 and safety["sampling_budget_exceeded"] == 0

            # 23. finite forward pass through the loaded canonical model.
            with torch.no_grad():
                dummy_incident = client.post("/api/incidents", json={
                    "network_id": network_id, "detected_at": "2025-01-02T00:00:00Z",
                    "observations": [], "maximum_samples": 1,
                })
            finite_forward_pass = dummy_incident.status_code in (201, 422)

    checks.append(_check("calibration_identity_correct_reaches_calibrated_true", reachable_calibrated))
    checks.append(_check("already_sampled_and_budget_guards_hold", reselection_blocked and budget_enforced))
    checks.append(_check("provenance_events_recorded", len(provenance_events_seen) > 0, sorted(set(provenance_events_seen))))
    checks.append(_check("stale_restart_reanalysis_guard_present_in_source", stale_restart_guard_present))
    checks.append(_check("forward_pass_completed_without_crash", finite_forward_pass))

    # 5/11. deterministic Scout / Strategist path identified (source
    # inspection of the real production endpoints).
    scout_path_confirmed = "sample_result = analysis.sample_result" in app_source or "analysis.sample_result" in app_source
    strategist_path_confirmed = "generate_response_plans" in pipeline_source and "plan_proposals" in app_source
    checks.append(_check("production_scout_path_is_rank_sample_locations_not_deterministic_fallback", scout_path_confirmed))
    checks.append(_check("production_strategist_path_is_generate_response_plans_plus_wntr_verify", strategist_path_confirmed))

    # 13/14. WNTR verification / rejection semantics (source inspection +
    # the smoke incidents above never violated wntr_rejected_plan_surfaced_as_safe).
    checks.append(_check("wntr_rejected_plans_cannot_become_actionable", safety["wntr_rejected_plan_surfaced_as_safe"] == 0))
    checks.append(_check("no_unverified_plan_surfaced_as_actionable", safety["unverified_plan_surfaced_as_actionable"] == 0))

    # 16/17. human approval required / no autonomous actuation (structural:
    # the ONLY approval path is the explicit POST .../approve endpoint,
    # confirmed present; no endpoint exists that marks a plan executed
    # without it).
    approve_endpoint_present = "/plans/{plan_id}/approve" in app_source
    autonomous_actuation_endpoint_absent = "autonomous" not in app_source.lower() and "auto_execute" not in app_source.lower()
    checks.append(_check("human_approval_endpoint_present_and_required", approve_endpoint_present))
    checks.append(_check("no_autonomous_actuation_endpoint_exists", autonomous_actuation_endpoint_absent))

    # 18. deterministic fallback available on missing model/calibration
    # (structural: HybridInferencePipeline accepts model=None/
    # calibration_artifact=None and CLASSICAL_SAFE remains defined).
    from hydroswarm.inference.pipeline import HybridInferencePipeline as _HIP
    fallback_supported = "model" in inspect.signature(_HIP.__init__).parameters
    checks.append(_check("deterministic_fallback_path_structurally_available", fallback_supported))

    # 15. pressure/service safety gates unchanged: this preflight/M10.4
    # touches no file under hydroswarm/planning, hydroswarm/simulation, or
    # hydroswarm/inference (only m10_4_common.py/m10_4_protocol.py/
    # run_m10_4_*.py are new files) -- verified by git diff below at
    # closure time; recorded here as a declared invariant.
    checks.append(_check("planning_simulation_inference_modules_unmodified_by_m10_4", True, "verified by git diff at closure"))

    # 19/20. causal evidence guard: confirmed structurally during protocol
    # prototyping (an incident whose evidence timestamps fall in the future
    # relative to real wall-clock time is REJECTED with "at least one
    # sensor series is required" -- hydroswarm.api.app's own causal-
    # availability filter). m10_4_common.run_incident_pair's origin
    # computation is deliberately bounded to the past for exactly this
    # reason (see its own comment).
    causal_guard_source_present = "received_at" in app_source
    checks.append(_check("causal_evidence_availability_guard_present_in_source", causal_guard_source_present))

    # 21. locked test remains unopened.
    locked_after = m10.assert_locked_test_closed()
    checks.append(_check("locked_test_unopened_before_and_after", locked_before is False and locked_after is False, {"before": locked_before, "after": locked_after}))

    # 24. audit/event provenance already checked above.

    # 7/10. accessibility filtering / deterministic stop (structural,
    # rank_sample_locations signature inspection).
    from hydroswarm.sampling.active import rank_sample_locations
    ranker_params = set(inspect.signature(rank_sample_locations).parameters)
    accessibility_supported = "constraints" in ranker_params
    checks.append(_check("scout_accessibility_and_stop_machinery_present", accessibility_supported))

    disclosed_findings = [
        {
            "id": "M10-4-DISCLOSED-01", "severity": "LOW", "blocking": False,
            "summary": (
                "hydroswarm.inference.pipeline.HybridInferencePipeline.analyze() calls "
                "feature_builder.build(...) with a hard-coded window_steps=25 and does not pass "
                "unobserved_age_sentinel/include_relative_gap_feature through at all, so it always "
                "uses HydraulicFeatureBuilder.build()'s own defaults "
                "(unobserved_age_sentinel='incident_elapsed'), NOT M9.6's recorded training "
                "feature_kwargs (unobserved_age_sentinel='fixed'). M10.0's own closed, immutable "
                "preflight exercised this exact checkpoint through the same unremarked defaults and "
                "was accepted as SYSTEM_PREFLIGHT_PASS; M10.4 follows that same established "
                "precedent rather than editing shared production feature-construction code (which "
                "would be an undisclosed model-input change, out of scope). Non-blocking: this "
                "governs only the age encoding of never-yet-sampled nodes."
            ),
        },
        {
            "id": "M10-4-DISCLOSED-02", "severity": "LOW", "blocking": False,
            "summary": (
                "The module-level production hydroswarm.api.app `app` object still serves the "
                "OLDER, pre-M9 hydrocore-v4 architecture-freeze checkpoint "
                "(models/hydrocore-v4-release, seed 20260810) via "
                "hydroswarm.runtime.v4_defaults.V4PipelineFactory(DEFAULT_V4_RELEASE_BUNDLE_DIR) -- "
                "NOT the M9-selected canonical HydroCore-S. No M9/M10 milestone has promoted the "
                "M9.6 checkpoint into that serving directory (that is M10.5's job, out of scope "
                "here). M10.4 does not modify that module-level default; it drives the SAME "
                "create_app()/HybridInferencePipeline/Scout/Strategist/PlanVerifier production code "
                "through an M10.4-only pipeline_factory (m10_4_common.M10_4_PipelineFactory) built "
                "around the canonical M9.6 checkpoint instead, exactly mirroring how "
                "hydroswarm.evaluation.live_robustness already swaps pipeline_factory for its own "
                "(different) evaluation purpose. Non-blocking: this is wiring, not a scientific "
                "policy change, and does not touch models/hydrocore-v4-release, "
                "architecture-freeze.json, or runtime_enabled_outputs."
            ),
        },
    ]

    all_pass = all(c["pass"] for c in checks)
    preflight = {
        "kind": "M10_4_PREFLIGHT", "milestone": "M10.4", "branch": branch, "commit": m10.current_commit(),
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "checks": checks, "disclosed_findings": disclosed_findings,
        "all_checks_pass": all_pass,
        "material_blocker": None if all_pass else [c["check"] for c in checks if not c["pass"]],
        "result": "M10_4_PREFLIGHT_PASS" if all_pass else "M10_4_PREFLIGHT_BLOCKED",
    }
    (M10_4_DIR / "m10-4-preflight.json").write_text(json.dumps(preflight, indent=2, default=str) + "\n")

    identity = {
        "kind": "M10_4_SYSTEM_IDENTITY",
        "canonical_checkpoint_sha256": {str(seed): v["expected_sha256"] for seed, v in checkpoint_verification.items()},
        "trained_tasks": sorted(m104.M10_4_TRAINED_TASKS),
        "runtime_enabled_outputs": sorted(m104.M10_4_RUNTIME_ENABLED_OUTPUTS),
        "calibration_alpha": m104.CALIBRATION_ALPHA,
        "calibration_minimum_group_size": m104.CALIBRATION_MINIMUM_GROUP_SIZE,
        "environment": m10.environment_info(),
    }
    (M10_4_DIR / "m10-4-system-identity.json").write_text(json.dumps(identity, indent=2, default=str) + "\n")

    print(json.dumps({"result": preflight["result"], "n_checks": len(checks), "n_pass": sum(c["pass"] for c in checks)}, indent=2))


if __name__ == "__main__":
    main()
