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
import json
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


def compute_safety_counters(
    rows: list[dict[str, Any]], *, identity_ok: bool,
    global_counters: dict[str, int] | None = None,
) -> dict[str, int]:
    """Aggregate safety counters exactly once (correction #6).

    Per-incident counters live on each row and are summed once; truly global
    counters (finalist identity drift, silent v4 fallback, etc.) are merged
    once via ``global_counters``. The trajectory runner must NOT attach a
    single shared aggregate counter object to every row (that would multiply a
    real violation by the population size). ``invariant_failures`` is read only
    from each row's per-incident safety counters (the trajectory runner already
    records it there); the ``invariants`` diagnostic dict is not a second,
    independent counting source.
    """
    counters = design.zero_safety_counters()
    for row in rows:
        per_row = row.get("safety_counters") or {}
        for name in design.SAFETY_COUNTERS_TEMPLATE:
            counters[name] += int(per_row.get(name, 0))
    for name, value in (global_counters or {}).items():
        if name in counters:
            counters[name] += int(value)
    # finalist_identity_drift is global and authoritative from the identity check.
    counters["finalist_identity_drift"] = 0 if identity_ok else 1
    return counters


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
    # Every per-incident safety/authority counter must be zero.
    safety = row.get("safety_counters") or {}
    if any(int(safety.get(name, 0)) != 0 for name in design.SAFETY_COUNTERS_TEMPLATE):
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
    *, metrics: dict[str, Any], safety: dict[str, int], identity_ok: bool,
    manifest_ok: bool, novelty_ok: bool, rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen hard gates (task Section 17) with split-specific
    scoping (correction #7). No post-hoc thresholds."""
    completeness = compute_population_completeness(rows, manifest)
    final_coverage = metrics["locked_final_test"]["source"]["coverage"]["rate"]
    topology_rows = [row for row in rows if row.get("split") == design.LOCKED_TOPOLOGY_TEST]

    global_checks: dict[str, Any] = {
        "finalist_identity": identity_ok and safety["finalist_identity_drift"] == 0,
        "manifest_hashes": manifest_ok,
        "safety_counters_zero": all(safety[name] == 0 for name in design.SAFETY_COUNTERS_TEMPLATE),
        "outputs_finite": safety["nonfinite_value_reached_decision"] == 0,
        "no_v4_fallback": safety["silent_v4_fallback"] == 0,
        "sample_budget": safety["sampling_budget_exceeded"] == 0,
        "no_unsafe_action": safety["unverified_plan_surfaced_as_actionable"] == 0 and safety["rejected_plan_surfaced_as_safe"] == 0,
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

    # 1. every path in artifact_sha256 exists and its file SHA-256 matches.
    for rel_path, expected_sha in (manifest.get("artifact_sha256") or {}).items():
        path = repo_root / rel_path
        if not path.exists():
            violations.append(f"artifact missing: {rel_path}")
            continue
        if sha256_file(path) != expected_sha:
            violations.append(f"artifact hash mismatch: {rel_path}")

    # 2-5. topology files (procedural .inp + referenced known families) exist
    # and match their recorded hashes.
    for entry in manifest.get("topologies") or []:
        rel_path = entry.get("file_path")
        expected_sha = entry.get("file_sha256")
        if not rel_path or not expected_sha:
            violations.append(f"topology entry missing file_path/file_sha256: {entry.get('topology_id')}")
            continue
        path = repo_root / rel_path
        if not path.exists():
            violations.append(f"topology file missing: {rel_path}")
            continue
        if sha256_file(path) != expected_sha:
            violations.append(f"topology file hash mismatch: {rel_path}")

    # 6-9. scenario JSONL contents correspond exactly to manifest scenario IDs,
    # with no duplicate IDs and matching split counts. The JSONL paths are
    # located from artifact_sha256 (the authoritative content-addressed record),
    # not hardcoded.
    expected_by_split: dict[str, set[str]] = {split: set() for split in design.LOCKED_SPLIT_NAMES}
    for entry in manifest.get("scenarios") or []:
        expected_by_split.setdefault(entry.get("split"), set()).add(entry.get("scenario_id"))
    for split in design.LOCKED_SPLIT_NAMES:
        matching = [
            rel for rel in (manifest.get("artifact_sha256") or {})
            if rel.endswith(f"/{split}/scenarios.jsonl")
        ]
        if len(matching) != 1:
            violations.append(f"expected exactly one scenario JSONL artifact for {split}, got {len(matching)}")
            continue
        jsonl = repo_root / matching[0]
        if not jsonl.exists():
            violations.append(f"scenario JSONL missing: {matching[0]}")
            continue
        actual_ids: list[str] = []
        try:
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                definition = json.loads(line)
                actual_ids.append(design.scenario_definition_hash(definition))
        except (OSError, json.JSONDecodeError) as error:
            violations.append(f"scenario JSONL unreadable/invalid: {matching[0]}: {error}")
            continue
        if len(set(actual_ids)) != len(actual_ids):
            violations.append(f"duplicate scenario IDs in {matching[0]}")
        if set(actual_ids) != expected_by_split.get(split, set()):
            violations.append(f"scenario IDs in {matching[0]} do not match the manifest for {split}")

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

    # Pre-open verification; atomically open (or refuse).
    try:
        opened = acquire_locked_open(
            authorization=authorization, manifest=manifest, manifest_path=manifest_path,
        )
    except (design.LockedAlreadyOpened, RuntimeError) as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        closure = compute_closure(gates={"all_checks_pass": False}, crashed_after_open=False, opened=False)
        closure["pre_open_failure"] = str(error)
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
    crashed_after_open = False
    try:
        rows = run_locked_trajectories(manifest, output_dir)
    except Exception as error:  # crash after OPENED: preserve, do NOT retry
        crashed_after_open = True
        (output_dir / "m11-6-crash-evidence.json").write_text(
            json.dumps({"crashed_after_open": True, "error_class": type(error).__name__, "detail": str(error)}, indent=2, sort_keys=True) + "\n"
        )

    identity_ok = verify_finalist_identity()
    metrics = compute_metrics(rows)
    safety = compute_safety_counters(rows, identity_ok=identity_ok)
    manifest_ok = (
        not design.validate_manifest(manifest)
        and not verify_materialized_artifacts(
            manifest, ROOT, authorization.get("design_freeze_commit_sha"),
        )
    )
    gates = compute_gates(
        metrics=metrics, safety=safety, identity_ok=identity_ok,
        manifest_ok=manifest_ok, novelty_ok=(manifest.get("novelty_audit") or {}).get("result") == "PASS",
        rows=rows, manifest=manifest,
    )
    closure = compute_closure(gates=gates, crashed_after_open=crashed_after_open, opened=True)

    with (output_dir / "m11-6-raw-incidents.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    (output_dir / "m11-6-metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "m11-6-gate.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    (output_dir / "m11-6-safety-counters.json").write_text(json.dumps(safety, indent=2, sort_keys=True) + "\n")
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
    source_node: str, condition: Any, safety: dict[str, int], maximum_samples: int,
) -> dict[str, Any]:
    """Drive one real production-API trajectory for one locked scenario.

    Mirrors hydroswarm.evaluation.live_robustness.run_condition / m10_4_common
    ._run_single_arm, but against the frozen V5 release bundle. The ONLY state
    transition toward "approved" is a single explicit /approve call (simulated
    human operator); autonomous actuation is never triggered.
    """

    from hydroswarm.evaluation.live_robustness import (
        _entropy, _invariants, _metric_fields, _payloads, _sample_observation,
    )

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
        return {"outcome": "HARNESS_ERROR", "http_status": created.status_code, "source_node": source_node}
    incident_id = created.json()["incident_id"]
    analyzed = client.post(f"/api/incidents/{incident_id}/analyze")
    if analyzed.status_code != 200:
        return {
            "outcome": "ABSTAINED" if analyzed.status_code == 409 else "HARNESS_ERROR",
            "http_status": analyzed.status_code, "source_node": source_node,
        }

    analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
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
        if node_id in observed_nodes:
            safety["sampled_node_reselected"] += 1
            break
        observation = _sample_observation(
            node_id, scenario, randomized, origin, sample_index,
            decision_time_seconds=float(scenario.timestamps_seconds[-1]),
            collection_delay_minutes=float(rec["expected_collection_delay_minutes"]),
            noise_std=0.05, seed=condition.seed,
        )
        added = client.post(f"/api/incidents/{incident_id}/samples", json=observation)
        if added.status_code != 200:
            break
        observed_nodes.add(node_id)
        analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
        rounds.append({
            "status": "SAMPLE", "recommended_node": node_id,
            "entropy_before": before_entropy, "entropy_after": _entropy(analysis["fused_belief"]),
            "true_source_rank_before": before_rank, "true_source_rank_after": _true_rank(analysis["fused_belief"], source_node),
        })

    plans_generated = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 2})
    plans: list[dict[str, Any]] = []
    if plans_generated.status_code == 200:
        for plan in plans_generated.json():
            verification = client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify")
            verification_json = verification.json() if verification.status_code == 200 else {"decision": "ERROR"}
            plans.append({"plan_id": plan["plan_id"], "verification": verification_json})
            if verification_json.get("decision") == "VERIFIED":
                break
    verified = [p for p in plans if p["verification"].get("decision") == "VERIFIED"]
    selected = verified[0] if verified else None
    human_approved = False
    if selected is not None:
        approved = client.post(f"/api/incidents/{incident_id}/plans/{selected['plan_id']}/approve", json={"approved": True, "operator_id": "m11-6-study"})
        human_approved = approved.status_code == 200
        if approved.status_code != 200:
            safety["human_approval_bypassed"] += 1

    final = client.get(f"/api/incidents/{incident_id}").json()
    invariants = _invariants(
        analysis=analysis, generate_status=plans_generated.status_code, plans=plans,
        approval_status=200 if human_approved else None, stale_approval_status=None,
    )
    if any(value is False for value in invariants.values()):
        safety["invariant_failures"] += 1
    if any(p["verification"].get("decision") in ("REJECTED", "ABSTAINED") and p.get("approval_status") == 200 for p in plans):
        safety["rejected_plan_surfaced_as_safe"] += 1

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
        "no_safe_plan": bool(plans) and not verified, "human_approved": human_approved,
        "final_status": final.get("status"), "invariants": invariants,
    }


def run_locked_trajectories(manifest: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    """Run the frozen finalist through the production API on every locked
    scenario. NOT executed in M11.6A-1 (no materialized population exists and
    the exactly-once guard forbids it); this is the fresh M11.6 session's
    execution body. Materialization and evaluation remain separate commands.
    """

    import tempfile

    from fastapi.testclient import TestClient

    from hydroswarm.api import create_app
    from hydroswarm.evaluation.live_robustness import Condition
    from hydroswarm.runtime.v5_defaults import V5PipelineFactory

    del output_dir  # the caller owns output-dir artifact writing

    # Load every scenario definition once, keyed by its canonical hash.
    definitions_by_hash: dict[str, dict[str, Any]] = {}
    locked_root = ROOT / "data/locked/m11-6"
    for split in design.LOCKED_SPLIT_NAMES:
        with (locked_root / split / "scenarios.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                definition = json.loads(line)
                definitions_by_hash[design.scenario_definition_hash(definition)] = definition

    factory = V5PipelineFactory(ROOT / FINALIST["release_bundle"], project_root=ROOT)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hydroswarm-m11-6-") as temporary:
        tmp = Path(temporary)
        app = create_app(
            pipeline_factory=factory, database_path=tmp / "state.sqlite3",
            ledger_path=tmp / "audit.sqlite3", network_directory=tmp / "networks",
        )
        with TestClient(app) as client:
            for scenario_entry in manifest["scenarios"]:
                definition = definitions_by_hash[scenario_entry["scenario_id"]]
                _network, randomized, scenario, source_node, condition_fields = _reconstruct_scenario(definition, manifest)
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
                        "safety_counters": design.zero_safety_counters(),
                    })
                    continue
                network_id = imported.json()["network_id"]
                # Correction #6: a FRESH per-incident safety counter dict per
                # row -- never one shared global aggregate object attached to
                # every row (which would multiply a violation by population size).
                row_safety = design.zero_safety_counters()
                row = _run_single_incident(
                    client=client, network_path=network_path, network_id=network_id,
                    scenario=scenario, randomized=randomized, source_node=source_node,
                    condition=condition, safety=row_safety, maximum_samples=design.MAXIMUM_SAMPLES,
                )
                row.update({
                    "split": definition["split"], "scenario_index": definition["scenario_index"],
                    "scenario_id": scenario_entry["scenario_id"],
                    "topology_id": definition["topology_id"], "network_family": definition["network_family"],
                    "seed": definition["seed"], "condition_kind": definition["condition_kind"],
                    "safety_counters": row_safety,
                })
                rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
