"""M10.3B Strategist failure-diagnosis amendment: focused mechanical tests.

These are NOT training tests -- M10.3B trains nothing and touches no
checkpoint. They exercise the exact ranking/sign/alignment logic
`run_m10_3b_diagnosis.py`/`run_m10_3_level_a_gate.py` rely on, with small
deterministic synthetic examples of known ground truth (Section 9's own
"candidate A exact value = clearly better... assert every stage preserves
A > B > C" instruction), plus the plan_value/plan_regret_proxy bijection
and the deterministic-oracle-computation requirement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_m10_3_level_a_gate as gate_module  # noqa: E402

from hydroswarm.domain import ConsequenceMetrics  # noqa: E402
from hydroswarm.planning.plan_value_policy import evaluate_plan_value  # noqa: E402


def _metrics(**overrides) -> ConsequenceMetrics:
    base = dict(
        contaminant_mass_consumed_mg=0.0, pressure_violation_minutes=0.0,
        service_availability=1.0, containment_time_minutes=0.0,
        minimum_pressure_m=20.0, operation_count=0,
    )
    base.update(overrides)
    return ConsequenceMetrics(**base)


# --------------------------------------------------------------------------
# Pairwise ranking direction (the gate's own comparator function, exercised
# directly -- not reimplemented).
# --------------------------------------------------------------------------


def test_pairwise_ranking_perfect_agreement_scores_100_percent() -> None:
    pred = np.array([0.9, 0.5, 0.1])
    target = np.array([0.9, 0.5, 0.1])
    mask = np.array([True, True, True])
    correct, total = gate_module._pairwise_ranking_accuracy_per_incident(pred, target, mask)
    assert (correct, total) == (3, 3)


def test_pairwise_ranking_inverted_prediction_scores_0_percent() -> None:
    pred = np.array([0.1, 0.5, 0.9])  # exactly inverted vs target
    target = np.array([0.9, 0.5, 0.1])
    mask = np.array([True, True, True])
    correct, total = gate_module._pairwise_ranking_accuracy_per_incident(pred, target, mask)
    assert (correct, total) == (0, 3)


def test_pairwise_ranking_ties_excluded_from_denominator() -> None:
    pred = np.array([0.9, 0.5, 0.1])
    target = np.array([0.5, 0.5, 0.1])  # first two candidates tied on target
    mask = np.array([True, True, True])
    correct, total = gate_module._pairwise_ranking_accuracy_per_incident(pred, target, mask)
    assert total == 2  # only (0,2) and (1,2) are real, non-tied pairs
    assert correct == 2


def test_pairwise_ranking_padded_slot_never_enters_the_count() -> None:
    pred = np.array([0.9, 0.5, 999.0])  # position 2 "padding" with a poisoned/adversarial value
    target = np.array([0.9, 0.5, -999.0])
    mask = np.array([True, True, False])
    correct, total = gate_module._pairwise_ranking_accuracy_per_incident(pred, target, mask)
    assert total == 1
    assert correct == 1


def test_pairwise_ranking_metric_is_candidate_order_permutation_invariant() -> None:
    rng = np.random.default_rng(20260817)
    pred = rng.normal(size=9)
    target = rng.normal(size=9)
    mask = np.ones(9, dtype=bool)
    base = gate_module._pairwise_ranking_accuracy_per_incident(pred, target, mask)
    perm = rng.permutation(9)
    permuted = gate_module._pairwise_ranking_accuracy_per_incident(pred[perm], target[perm], mask[perm])
    assert base == permuted


# --------------------------------------------------------------------------
# plan_value / plan_regret_proxy: exact bijection (redundancy finding),
# deterministic given a fixed candidate pool.
# --------------------------------------------------------------------------


def test_plan_value_is_exact_deterministic_function_of_regret() -> None:
    no_response = _metrics(contaminant_mass_consumed_mg=1000.0, containment_time_minutes=240.0)
    a = _metrics(contaminant_mass_consumed_mg=100.0, containment_time_minutes=30.0)
    b = _metrics(contaminant_mass_consumed_mg=500.0, service_availability=0.98, containment_time_minutes=90.0)
    pool = [no_response, a, b]
    for candidate in pool:
        result = evaluate_plan_value(candidate, no_response=no_response, valid_candidate_metrics=pool)
        assert result.plan_regret_proxy == result.regret
        assert result.plan_value == pytest.approx(1.0 / (1.0 + result.regret), abs=1e-12)


def test_evaluate_plan_value_is_deterministic_and_reproducible() -> None:
    """Same inputs -> byte-identical outputs across repeated calls (Section
    23's "reproducible target construction"/"deterministic oracle
    computation" requirements) -- no hidden randomness anywhere in the
    formula."""

    no_response = _metrics(contaminant_mass_consumed_mg=1000.0, containment_time_minutes=240.0)
    a = _metrics(contaminant_mass_consumed_mg=250.0, pressure_violation_minutes=0.0, containment_time_minutes=60.0)
    pool = [no_response, a]
    first = evaluate_plan_value(a, no_response=no_response, valid_candidate_metrics=pool)
    second = evaluate_plan_value(a, no_response=no_response, valid_candidate_metrics=pool)
    assert first == second


def test_regret_is_zero_exactly_at_the_pool_minimum_cost_candidate() -> None:
    no_response = _metrics(contaminant_mass_consumed_mg=1000.0, containment_time_minutes=240.0)
    best = _metrics(contaminant_mass_consumed_mg=10.0, containment_time_minutes=5.0)  # unambiguously cheapest
    worse = _metrics(contaminant_mass_consumed_mg=900.0, containment_time_minutes=200.0)
    pool = [no_response, best, worse]
    result_best = evaluate_plan_value(best, no_response=no_response, valid_candidate_metrics=pool)
    result_worse = evaluate_plan_value(worse, no_response=no_response, valid_candidate_metrics=pool)
    assert result_best.regret == 0.0
    assert result_best.plan_value == 1.0
    assert result_worse.regret > 0.0
    assert result_worse.plan_value < 1.0


# --------------------------------------------------------------------------
# Oracle-style "best candidate" selection is a deterministic pure function
# of the (already exactly-verified) candidate pool -- no stochasticity.
# --------------------------------------------------------------------------


def test_oracle_best_candidate_selection_is_deterministic() -> None:
    no_response = _metrics(contaminant_mass_consumed_mg=1000.0, containment_time_minutes=240.0)
    candidates = [
        _metrics(contaminant_mass_consumed_mg=800.0, containment_time_minutes=180.0),
        _metrics(contaminant_mass_consumed_mg=50.0, containment_time_minutes=20.0),
        _metrics(contaminant_mass_consumed_mg=400.0, containment_time_minutes=100.0),
    ]
    pool = [no_response, *candidates]
    for _ in range(5):
        values = [evaluate_plan_value(c, no_response=no_response, valid_candidate_metrics=pool).plan_value for c in candidates]
        best_index = int(np.argmax(values))
        assert best_index == 1  # the unambiguously cheapest candidate, every time


# --------------------------------------------------------------------------
# pressure_risk_proxy structural-degeneracy finding: a valid (non-rejected)
# plan mathematically cannot have nonzero pressure_violation_minutes, given
# the verifier's own rejection rule (PlanVerifier/HydraulicSimulator use the
# SAME minimum_pressure_m threshold for both the rejection decision and the
# pressure_violation_minutes computation). Exercised here directly against
# the governed cost formula, not merely asserted from reading the source.
# --------------------------------------------------------------------------


def test_pressure_risk_proxy_cost_component_present_when_nonzero() -> None:
    no_response = _metrics(contaminant_mass_consumed_mg=1000.0, containment_time_minutes=240.0)
    candidate = _metrics(contaminant_mass_consumed_mg=200.0, pressure_violation_minutes=30.0, containment_time_minutes=60.0)
    pool = [no_response, candidate]
    result = evaluate_plan_value(candidate, no_response=no_response, valid_candidate_metrics=pool)
    # If pressure_violation_minutes WERE allowed to be nonzero for a
    # candidate reaching this formula, the formula correctly incorporates
    # it (0.5 == 30/60) -- confirming the ZERO seen in the real M10.3A
    # population is a property of WHICH candidates reach this formula at
    # all (the verifier's own gate), not a bug in this formula silently
    # discarding the term.
    assert result.pressure_risk_proxy == pytest.approx(0.5)
