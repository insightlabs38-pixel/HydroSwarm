"""Frozen M10.4 scientific protocol (governed full-trajectory end-to-end
validation). Written and hashed BEFORE any trajectory is executed or any
performance result is inspected -- Part 2 of the M10.4 task specification.

Companion document: `docs/evaluation/HYDROCORE_V5_M10_4_FULL_TRAJECTORY_PROTOCOL.md`.
Preflight (must PASS before this protocol is frozen for real execution):
`reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-preflight.json`.

This module contains ONLY frozen, pre-registered constants -- population
design, condition matrix, seed formula, sample budget, comparator design,
and the utility/quality gate thresholds. No result-driven modification is
permitted after `run_m10_4_execute.py` starts writing trajectory rows.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m10_4_common as m104  # noqa: E402
import m10_common as m10  # noqa: E402

PROTOCOL_VERSION = "M10.4-v1"

# ---------------------------------------------------------------------------
# System-under-test identity (Section: "M10.4 SYSTEM UNDER TEST").
# ---------------------------------------------------------------------------

SYSTEM_UNDER_TEST = {
    "predictor": "canonical M9.6 HydroCore-S (CLASSICAL_HYDROCORE_S / AGE_FIX_ONLY / "
                 "EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING), unmodified",
    "calibration": "frozen M9 B_DEPTH_AWARE, alpha=0.1, source-representative support 20/source, "
                   "re-stamped (not refit) with the real per-seed checkpoint/feature/fusion identity "
                   "-- see m10_4_common.fit_frozen_calibrator docstring",
    "ood": "hydroswarm.inference.ood.OODDetector (deterministic multi-signal); learned ood_category "
           "head excluded from trained_tasks -- advisory-only, non-authoritative",
    "fusion": "hydroswarm.inference.fusion.fuse_source_probabilities, DYNAMIC_TRUST_FUSION_CONFIG "
              "('fuse_source_probabilities-v1'), unmodified",
    "scout": "hydroswarm.sampling.rank_sample_locations, invoked via analysis.sample_result / "
             "POST /api/incidents/{id}/samples/recommend -- the REAL production endpoint. "
             "NOT HydroScout.deterministic_fallback (True M10.2's narrower comparator).",
    "strategist": "hydroswarm.planning.generate_response_plans (deterministic candidate generation) "
                  "+ exact WNTR/EPANET PlanVerifier via POST /api/incidents/{id}/plans/{id}/verify. "
                  "Learned Strategist heads are schema-unbuilt for M9.6 (M10.0 preflight: "
                  "strategist_named_proxy_heads_present=False) and separately excluded from "
                  "trained_tasks -- doubly non-authoritative.",
    "trained_tasks": sorted(m104.M10_4_TRAINED_TASKS),
    "runtime_enabled_outputs": sorted(m104.M10_4_RUNTIME_ENABLED_OUTPUTS),
    "driven_through": "hydroswarm.api.create_app (real FastAPI production application) via "
                       "starlette TestClient, with pipeline_factory=M10_4_PipelineFactory "
                       "(structurally identical to hydroswarm.runtime.v4_defaults.V4PipelineFactory, "
                       "checkpoint/calibration source swapped to canonical M9.6 + frozen M9 "
                       "calibration -- see m10_4_common.py module docstring for the full rationale).",
}

CANONICAL_CHECKPOINT_SHA256: dict[int, str] = {
    20260814: "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5",
    31874: "527e707b988870dff323b6a3e0f1bde36a152b7bbe7e4c7f4bf0e59cdaf19332",
    20260815: "b45f5afeabba820270130595dffb44152ec12fad6fa383cce67013f14524113c",
}

# ---------------------------------------------------------------------------
# Population design (Part 3 / TRAJECTORY CONDITIONS). Reuses
# hydroswarm.evaluation.live_robustness's own governed Condition
# constructors verbatim -- no new perturbation framework.
# ---------------------------------------------------------------------------

#: Frozen order -- index is baked into the seed formula, never reassigned.
CONDITION_KINDS: tuple[str, ...] = (
    "NOMINAL",
    "LOW_COVERAGE_ACTIVE_SAMPLING",
    "SENSOR_DROPOUT",
    "SENSOR_HEALTH_DEGRADED",
    "MEASUREMENT_NOISE",
    "SEVERITY_SHIFT",
    "AMBIGUITY_DISAGREEMENT",
)

#: kwargs for `hydroswarm.evaluation.live_robustness.Condition`, keyed by
#: condition kind (network_id filled in per-family at generation time).
CONDITION_KWARGS: dict[str, dict[str, Any]] = {
    "NOMINAL": dict(perturbation_type="nominal", perturbation_level="clean_operational"),
    "LOW_COVERAGE_ACTIVE_SAMPLING": dict(perturbation_type="sensor_coverage", perturbation_level="25%", coverage=0.25),
    "SENSOR_DROPOUT": dict(perturbation_type="missingness", perturbation_level="30%", missing_rate=0.3),
    "SENSOR_HEALTH_DEGRADED": dict(perturbation_type="sensor_health", perturbation_level="frozen:50%", health_mode="frozen", health_fraction=0.5),
    "MEASUREMENT_NOISE": dict(perturbation_type="measurement_noise", perturbation_level="moderate", noise_std=0.05),
    "SEVERITY_SHIFT": dict(perturbation_type="hydraulic_mismatch", perturbation_level="source_strength", hydraulic="source_strength"),
    "AMBIGUITY_DISAGREEMENT": dict(perturbation_type="ambiguity", perturbation_level="disagreement", ambiguity="disagreement"),
}

#: TRAINED_FAMILIES get the full condition matrix; UNSEEN (development-only
#: topology-shift) families get only NOMINAL -- the topology shift IS their
#: perturbation (Part 3: "exercise both TRAINED topology families ... and
#: development-only topology-shift families ... under already-governed
#: protocol semantics").
TRAINED_FAMILY_CONDITIONS: tuple[str, ...] = CONDITION_KINDS
UNSEEN_FAMILY_CONDITIONS: tuple[str, ...] = ("NOMINAL",)

INCIDENTS_PER_CELL = 5
MAXIMUM_SAMPLES = 3

MODEL_SEEDS: tuple[int, ...] = m10.SEEDS


def population_cells() -> tuple[tuple[str, str], ...]:
    """Every (family, condition_kind) cell, in frozen iteration order."""

    cells: list[tuple[str, str]] = []
    for family in m10.TRAINED_FAMILIES:
        for kind in TRAINED_FAMILY_CONDITIONS:
            cells.append((family, kind))
    for family in m10.UNSEEN_FAMILIES:
        for kind in UNSEEN_FAMILY_CONDITIONS:
            cells.append((family, kind))
    return tuple(cells)


def condition_index(kind: str) -> int:
    return CONDITION_KINDS.index(kind)


def incident_seed(model_seed: int, family: str, kind: str, incident_index: int) -> int:
    """Deterministic, frozen seed for one physical incident. `model_seed`
    only selects which of the 3 canonical checkpoints observes this
    incident's causal evidence -- the underlying WNTR scenario/seed
    namespace is shared across model seeds by design (each canonical
    checkpoint is evaluated against the SAME physical incident population,
    exactly like every prior M9/M10 multi-seed comparison)."""

    return m104.m10_4_seed(family, condition_index(kind), incident_index)


# ---------------------------------------------------------------------------
# Comparator design (Part 4).
# ---------------------------------------------------------------------------

COMPARATOR = {
    "arm_full": "ARM_FULL: retained end-to-end system, production deterministic Scout sampling "
                "(POST /samples/recommend) engaged up to MAXIMUM_SAMPLES before deterministic planning.",
    "arm_no_extra_sampling": "ARM_NO_EXTRA_SAMPLING: identical canonical checkpoint/calibration/OOD/"
                              "fusion/candidate-generation/Strategist/WNTR verification, identical "
                              "initial evidence -- but no active Scout sample request is ever issued; "
                              "plans are generated directly from the initial analysis.",
    "pairing": "Each physical incident is realized as TWO separate incidents (one per arm) built from "
               "byte-identical initial observations/network/calibration/model -- "
               "`m10_4_common.run_incident_pair` asserts `paired_initial_state_equal` for every pair.",
    "excluded_by_closed_decisions": [
        "learned Scout vs deterministic Scout (True M10.2, closed: M10_2_LEARNED_SCOUT_NOT_PROMOTED_DETERMINISTIC_RETAINED)",
        "learned Strategist vs deterministic Strategist (M10.3C, closed: M10_3C_LEARNED_STRATEGIST_NOT_JUSTIFIED)",
        "learned OOD vs deterministic OOD (M10.1, closed: M10_1_LEARNED_OOD_NOT_PROMOTED_DETERMINISTIC_RETAINED)",
    ],
}

# ---------------------------------------------------------------------------
# Fail-closed / failure-injection case list (Part 9). Small, targeted,
# deterministic -- not part of the statistical population.
# ---------------------------------------------------------------------------

FAIL_CLOSED_CASES: tuple[str, ...] = (
    "MODEL_UNAVAILABLE",
    "CALIBRATION_UNAVAILABLE",
    "SENSOR_STATE_INSUFFICIENT",
    "SAMPLING_BUDGET_PREEXHAUSTED",
    "NO_ACCESSIBLE_UNSAMPLED_CANDIDATE",
)
FAIL_CLOSED_SEED_BASE = m104.M10_4_SEED_BASE + 90_000_000

# ---------------------------------------------------------------------------
# Utility / quality gate (Part 10). Frozen BEFORE any trajectory result is
# inspected. Thresholds reuse existing project conventions (Wilson/paired
# bootstrap over incident-level pairs, per m10_common.wilson_interval_90)
# rather than inventing new statistical machinery.
# ---------------------------------------------------------------------------

UTILITY_GATE = {
    "A_all_hard_safety_gates_pass": "every counter in m10-4-safety-counters.json equals 0",
    "B_no_material_top1_regression": "ARM_FULL final top-1 accuracy >= ARM_NO_EXTRA_SAMPLING final "
                                      "top-1 accuracy - 0.05 (absolute), pooled across the population "
                                      "(non-inferiority margin, not a requirement to improve)",
    "C_active_sampling_non_harmful": "among paired incidents where ARM_FULL actually samples "
                                      "(samples_taken > 0), mean true-source-rank change "
                                      "(before -> after sampling) is <= 0 (rank number does not "
                                      "systematically worsen) OR the 90% Wilson/bootstrap CI on the "
                                      "fraction that improve is not systematically below the fraction "
                                      "that worsen by more than 0.10",
    "D_final_plans_wntr_valid": "every plan ever marked approved/actionable has verification.decision "
                                 "== VERIFIED (m10-4-safety-counters.json "
                                 "wntr_rejected_plan_surfaced_as_safe == 0)",
    "E_no_material_exposure_harm": "ARM_FULL selected-plan exposure/service/pressure consequences are "
                                    "not systematically worse than the NO_ACTION reference by more than "
                                    "a 10% relative margin, pooled across incidents where both are "
                                    "available",
    "F_all_outputs_finite": "m10-4-safety-counters.json nonfinite_value_reached_decision == 0",
    "G_fail_closed_valid": "every FAIL_CLOSED_CASES scenario in m10-4-fail-closed.json resolves to a "
                            "bounded, deterministic, non-escalating outcome (no learned-authority "
                            "escalation, no crash, no autonomous actuation)",
}

CLOSURE_STATES: tuple[str, ...] = (
    "M10_4_FULL_TRAJECTORY_PASS",
    "M10_4_FULL_TRAJECTORY_UTILITY_NOT_ESTABLISHED",
    "M10_4_FULL_TRAJECTORY_BLOCKED",
)


def protocol_document() -> dict[str, Any]:
    return {
        "kind": "M10_4_PROTOCOL",
        "protocol_version": PROTOCOL_VERSION,
        "system_under_test": SYSTEM_UNDER_TEST,
        "canonical_checkpoint_sha256": CANONICAL_CHECKPOINT_SHA256,
        "model_seeds": list(MODEL_SEEDS),
        "condition_kinds": list(CONDITION_KINDS),
        "condition_kwargs": CONDITION_KWARGS,
        "trained_family_conditions": list(TRAINED_FAMILY_CONDITIONS),
        "unseen_family_conditions": list(UNSEEN_FAMILY_CONDITIONS),
        "trained_families": list(m10.TRAINED_FAMILIES),
        "unseen_families": list(m10.UNSEEN_FAMILIES),
        "incidents_per_cell": INCIDENTS_PER_CELL,
        "maximum_samples": MAXIMUM_SAMPLES,
        "population_cells": [list(c) for c in population_cells()],
        "seed_namespace": {
            "base": m104.M10_4_SEED_BASE, "range": list(m104.M10_4_RANGE),
            "family_offset": m104.M10_4_FAMILY_OFFSET,
        },
        "comparator": COMPARATOR,
        "fail_closed_cases": list(FAIL_CLOSED_CASES),
        "fail_closed_seed_base": FAIL_CLOSED_SEED_BASE,
        "utility_gate": UTILITY_GATE,
        "closure_states": list(CLOSURE_STATES),
        "locked_test_authorized": False,
        "development_only": True,
        "no_post_result_tuning": True,
    }


def protocol_hash() -> str:
    payload = json.dumps(protocol_document(), sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    print(protocol_hash())
