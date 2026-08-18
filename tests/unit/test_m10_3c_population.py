"""M10.3C Strategist expanded-population identifiability/oracle-gate
amendment: focused mechanical tests.

These are NOT training tests -- M10.3C trains nothing and touches no
checkpoint. They exercise seed-namespace disjointness, family/depth
coverage, balanced/frozen cell counts, candidate-generation determinism/
depth-independence, the candidate-verification/rejection-code diagnostic,
and the closure-gate decision-tree logic, using small synthetic inputs of
known ground truth (mirroring `tests/unit/test_m10_3b_diagnosis.py`'s own
style).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

import m10_3c_population_protocol as proto  # noqa: E402
import run_m10_3b_diagnosis as m10_3b  # noqa: E402
import run_m10_3c_population as mod  # noqa: E402
from run_m7_topology import TRAINED_FAMILIES  # noqa: E402

from hydroswarm.planning.action_templates import ACTION_TEMPLATES, LINK_TARGET_TEMPLATES  # noqa: E402
from hydroswarm.training.causal_prefix import CAUSAL_PREFIX_DEPTHS  # noqa: E402
from hydroswarm.training.strategist_candidate_corpus import build_strategist_candidate_example  # noqa: E402
from hydroswarm.training.strategist_trajectory import build_strategist_trajectory  # noqa: E402


# ---------------------------------------------------------------------------
# Population definition: families / depths / seed namespace.
# ---------------------------------------------------------------------------


def test_families_match_governed_trained_families() -> None:
    assert proto.FAMILIES == tuple(name for name, _loader in TRAINED_FAMILIES)


def test_depth_buckets_are_governed_subset_of_causal_prefix_depths() -> None:
    assert set(proto.DEPTH_BUCKETS) <= set(CAUSAL_PREFIX_DEPTHS)
    assert proto.DEPTH_BUCKETS == tuple(sorted(set(proto.DEPTH_BUCKETS)))
    # exactly the task-specified {1,2,3,4,6,25} -- omits only 12, never invents a new depth.
    assert set(proto.DEPTH_BUCKETS) == {1, 2, 3, 4, 6, 25}


def test_population_is_balanced_across_family_and_depth_cells() -> None:
    assert proto.PER_FAMILY_COUNT % len(proto.DEPTH_BUCKETS) == 0
    assert proto.PER_CELL_COUNT == proto.PER_FAMILY_COUNT // len(proto.DEPTH_BUCKETS)
    assert proto.TOTAL_SCENARIO_COUNT == proto.PER_FAMILY_COUNT * len(proto.FAMILIES)
    # No family/cell is over-weighted relative to any other -- every family gets the SAME count.
    assert len({proto.PER_FAMILY_COUNT for _ in proto.FAMILIES}) == 1


def test_depth_label_round_robin_assignment_is_deterministic_and_balanced() -> None:
    counts = {depth: 0 for depth in proto.DEPTH_BUCKETS}
    assigned_twice = {depth: 0 for depth in proto.DEPTH_BUCKETS}
    for index in range(proto.PER_FAMILY_COUNT):
        depth_label = proto.DEPTH_BUCKETS[index % len(proto.DEPTH_BUCKETS)]
        counts[depth_label] += 1
    for index in range(proto.PER_FAMILY_COUNT):
        depth_label = proto.DEPTH_BUCKETS[index % len(proto.DEPTH_BUCKETS)]
        assigned_twice[depth_label] += 1
    assert counts == assigned_twice  # deterministic given the same index
    assert all(count == proto.PER_CELL_COUNT for count in counts.values())


def test_seed_namespace_family_ranges_are_pairwise_disjoint() -> None:
    ranges = {
        family: (proto.FAMILY_SEED_BASE[family], proto.FAMILY_SEED_BASE[family] + proto.PER_FAMILY_COUNT * 100)
        for family in proto.FAMILIES
    }
    names = list(ranges)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (a_lo, a_hi), (b_lo, b_hi) = ranges[names[i]], ranges[names[j]]
            assert not (a_lo <= b_hi and b_lo <= a_hi), f"{names[i]} overlaps {names[j]}"


def test_seed_namespace_disjoint_from_historical_milestone_ranges() -> None:
    historical = {
        "M10.1": (1_100_000_000, 1_199_999_999),
        "M10.2": (1_200_000_000, 1_299_999_999),
        "M10.3A/M10.3B": (1_300_000_000, 1_399_999_999),
    }
    m10_3c_lo = min(proto.FAMILY_SEED_BASE.values())
    m10_3c_hi = max(proto.FAMILY_SEED_BASE[f] + proto.PER_FAMILY_COUNT * 100 for f in proto.FAMILIES)
    for name, (lo, hi) in historical.items():
        assert not (m10_3c_lo <= hi and lo <= m10_3c_hi), f"M10.3C overlaps {name}"


def test_reserved_future_m10_3d_seed_base_disjoint_from_diagnostic_ranges() -> None:
    reserved_lo = proto.RESERVED_FUTURE_M10_3D_SEED_BASE
    for family in proto.FAMILIES:
        hi = proto.FAMILY_SEED_BASE[family] + proto.PER_FAMILY_COUNT * 100
        assert reserved_lo > hi, f"reserved M10.3D range collides with {family}'s diagnostic range"


def test_seed_bases_never_appear_elsewhere_in_the_repository_before_this_protocol() -> None:
    # Static-grep re-confirmation matching the protocol module's own frozen
    # claim, scoped to the EXACT seed-base constants (not a generic 10-digit
    # numeric range, which would false-positive on unrelated large integers
    # like unix timestamps or unrelated IDs elsewhere in the repo).
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    exact_bases = ["1_400_000_000", "1400000000", "1_410_000_000", "1410000000",
                   "1_420_000_000", "1420000000", "1_450_000_000", "1450000000"]
    pattern = "|".join(exact_bases)
    search_dirs = ["src", "scripts", "docs", "reports", "tests", "configs"]
    result = subprocess.run(
        ["grep", "-rEl", pattern, *search_dirs, "--include=*.py", "--include=*.json", "--include=*.md"],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    excluded_substrings = (
        "m10_3c_population_protocol.py", "run_m10_3c_population.py", "test_m10_3c_population.py",
        "/m10-3c-population/", "HYDROCORE_V5_M10_3C_STRATEGIST_POPULATION_AMENDMENT.md",
        # M10.4 (a later, separately-frozen milestone) legitimately and
        # necessarily cites M10.3C's own seed range while MECHANICALLY
        # PROVING its own fresh seed namespace is disjoint from every prior
        # milestone's, including M10.3C's -- a real, disclosed, intended
        # downstream reference, not an accidental collision. Does not
        # touch, weaken, or re-open M10.3C's own finding/closure/protocol
        # in any way; only widens this regression guard's own allowlist.
        "scripts/hydrocore_v5/m10_4_common.py",
        "docs/evaluation/HYDROCORE_V5_M10_4_FULL_TRAJECTORY_PROTOCOL.md",
        "reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-seed-disjointness.json",
    )
    hits = [line for line in result.stdout.splitlines() if not any(s in line for s in excluded_substrings)]
    assert hits == [], f"unexpected pre-existing hits for the M10.3C seed bases: {hits}"


# ---------------------------------------------------------------------------
# Depth-independence audit finding: candidate/target generation never
# receives a depth argument anywhere in the call chain (regression guard
# for the M10.3C module docstring's own load-bearing claim).
# ---------------------------------------------------------------------------


def test_build_strategist_candidate_example_signature_has_no_depth_argument() -> None:
    sig = inspect.signature(build_strategist_candidate_example)
    assert "depth" not in sig.parameters


def test_build_strategist_trajectory_signature_has_no_depth_argument() -> None:
    sig = inspect.signature(build_strategist_trajectory)
    assert "depth" not in sig.parameters


def test_strategist_trajectory_source_uses_full_sensor_series_not_truncated() -> None:
    source = inspect.getsource(build_strategist_trajectory)
    assert "build_sensor_series(scenario, feature_context)" in source
    assert "truncate_causal_prefix" not in source


# ---------------------------------------------------------------------------
# Candidate-verification / rejection-code diagnostic (Section 14).
# ---------------------------------------------------------------------------


class _FakeLabel:
    def __init__(self, action_template: str, plan_validity: bool, rejection_codes: tuple[str, ...] = ()) -> None:
        self.action_template = action_template
        self.plan_validity = plan_validity
        self.rejection_codes = rejection_codes


def test_candidate_verification_counts_and_rejection_codes() -> None:
    label_lists = [
        (
            _FakeLabel("NO_ACTION", True),
            _FakeLabel("ISOLATE_SOURCE", False, ("PRESSURE_BELOW_MINIMUM",)),
        ),
        (
            _FakeLabel("NO_ACTION", True),
            _FakeLabel("ISOLATE_SOURCE", False, ("PRESSURE_BELOW_MINIMUM",)),
        ),
        (
            _FakeLabel("NO_ACTION", True),
            _FakeLabel("ISOLATE_SOURCE", True),
        ),
    ]
    result = mod._candidate_verification(label_lists, "test-cell")
    assert result["per_template"]["NO_ACTION"]["n_proposed"] == 3
    assert result["per_template"]["NO_ACTION"]["n_verified"] == 3
    assert result["per_template"]["NO_ACTION"]["verification_rate"] == pytest.approx(1.0)
    assert result["per_template"]["ISOLATE_SOURCE"]["n_proposed"] == 3
    assert result["per_template"]["ISOLATE_SOURCE"]["n_verified"] == 1
    assert result["per_template"]["ISOLATE_SOURCE"]["verification_rate"] == pytest.approx(1 / 3)
    assert result["per_template"]["ISOLATE_SOURCE"]["rejection_code_frequency"] == {"PRESSURE_BELOW_MINIMUM": 2}
    assert result["isolation_template_summary"]["ISOLATE_SOURCE"]["ever_verified_on_this_cell"] is True
    assert result["isolation_template_summary"]["ALTERNATE_VALVE_CUT"]["n_proposed"] == 0
    assert result["isolation_template_summary"]["ALTERNATE_VALVE_CUT"]["ever_verified_on_this_cell"] is False


def test_isolation_templates_match_governed_link_target_templates() -> None:
    assert set(mod.ISOLATION_TEMPLATES) == LINK_TARGET_TEMPLATES
    assert set(mod.ISOLATION_TEMPLATES) <= set(ACTION_TEMPLATES)


# ---------------------------------------------------------------------------
# Reuse (not reimplementation) of M10.3B's own methodology / tolerances.
# ---------------------------------------------------------------------------


def test_near_tie_tolerance_is_the_same_object_reused_from_m10_3b() -> None:
    assert mod.NEAR_TIE_TOLERANCE is m10_3b.NEAR_TIE_TOLERANCE


def test_within_incident_and_oracle_functions_are_reused_from_m10_3b() -> None:
    assert mod._within_incident_variance is m10_3b._within_incident_variance
    assert mod._oracle_utility is m10_3b._oracle_utility
    assert mod._candidate_diversity is m10_3b._candidate_diversity
    assert mod._target_identifiability is m10_3b._target_identifiability


# ---------------------------------------------------------------------------
# Gate / closure-decision-tree logic (Section 18/20), synthetic inputs.
# ---------------------------------------------------------------------------


def _variance_doc(frac_2plus: float, frac_3plus: float, support: int = 100) -> dict:
    return {"per_target": {"plan_value": {
        "fraction_incidents_with_2plus_meaningfully_distinguishable": frac_2plus,
        "fraction_incidents_with_3plus_meaningfully_distinguishable_clusters": frac_3plus,
        "n_incidents_with_2plus_valid_candidates": support,
    }}}


def _oracle_doc(frac_meaningful: float, mean_gain: float, no_action_near_optimal: float) -> dict:
    return {
        "best_vs_no_action_plan_value_gain": {"fraction_meaningfully_positive": frac_meaningful, "mean": mean_gain},
        "fraction_incidents_where_no_action_is_already_near_optimal": no_action_near_optimal,
    }


def test_diversity_pass_requires_all_three_criteria() -> None:
    ok, _ = mod._diversity_pass(_variance_doc(0.5, 0.2), contributing_cells=5)
    assert ok is True
    ok, _ = mod._diversity_pass(_variance_doc(0.5, 0.2), contributing_cells=1)  # too few contributing cells
    assert ok is False
    ok, _ = mod._diversity_pass(_variance_doc(0.10, 0.2), contributing_cells=5)  # below 2plus threshold
    assert ok is False


def test_oracle_pass_requires_all_four_criteria() -> None:
    ok, _ = mod._oracle_pass(_oracle_doc(0.30, 0.08, 0.5), contributing_cells=5)
    assert ok is True
    ok, _ = mod._oracle_pass(_oracle_doc(0.30, 0.08, 0.95), contributing_cells=5)  # NO_ACTION too dominant
    assert ok is False
    ok, _ = mod._oracle_pass(_oracle_doc(0.05, 0.08, 0.5), contributing_cells=5)  # below gain-fraction threshold
    assert ok is False


def test_m10_3b_like_baseline_fails_the_gate() -> None:
    """Feeding literally the M10.3B negative baseline numbers into the
    SAME gate functions must fail -- the gate is not trivially satisfied."""

    diversity_ok, _ = mod._diversity_pass(_variance_doc(0.175, 0.0), contributing_cells=1)
    oracle_ok, _ = mod._oracle_pass(_oracle_doc(0.096, 0.022, 0.904), contributing_cells=1)
    assert diversity_ok is False
    assert oracle_ok is False


def test_decide_closure_pass_when_global_pass() -> None:
    decision, passing, clear_fail = mod._decide_closure(True, {"golden-reference": {"family_pass": True, "family_clear_fail": False}})
    assert decision == "M10_3C_POPULATION_IDENTIFIABILITY_PASS"


def test_decide_closure_conditional_when_one_family_passes_and_another_clearly_fails() -> None:
    family_gate = {
        "branched-loop": {"family_pass": True, "family_clear_fail": False},
        "golden-reference": {"family_pass": False, "family_clear_fail": True},
        "loop-grid": {"family_pass": False, "family_clear_fail": False},
    }
    decision, passing, clear_fail = mod._decide_closure(False, family_gate)
    assert decision == "M10_3C_POPULATION_IDENTIFIABILITY_CONDITIONAL"
    assert passing == ["branched-loop"]
    assert clear_fail == ["golden-reference"]


def test_decide_closure_not_justified_when_no_family_passes() -> None:
    family_gate = {family: {"family_pass": False, "family_clear_fail": False} for family in proto.FAMILIES}
    decision, _, _ = mod._decide_closure(False, family_gate)
    assert decision == "M10_3C_LEARNED_STRATEGIST_NOT_JUSTIFIED"


def test_decide_closure_not_justified_when_families_pass_but_none_clearly_fail() -> None:
    """No cherry-picking: a favorable-looking family alone, without a real
    separating regime (no clear-fail family), must NOT produce CONDITIONAL."""

    family_gate = {
        "golden-reference": {"family_pass": True, "family_clear_fail": False},
        "branched-loop": {"family_pass": False, "family_clear_fail": False},
        "loop-grid": {"family_pass": False, "family_clear_fail": False},
    }
    decision, _, _ = mod._decide_closure(False, family_gate)
    assert decision == "M10_3C_LEARNED_STRATEGIST_NOT_JUSTIFIED"


def test_decide_closure_not_justified_when_all_families_clear_fail() -> None:
    family_gate = {family: {"family_pass": False, "family_clear_fail": True} for family in proto.FAMILIES}
    decision, _, _ = mod._decide_closure(False, family_gate)
    assert decision == "M10_3C_LEARNED_STRATEGIST_NOT_JUSTIFIED"


# ---------------------------------------------------------------------------
# No locked-population usage (structural check).
# ---------------------------------------------------------------------------


def test_main_asserts_locked_test_closed_before_and_after() -> None:
    source = inspect.getsource(mod.main)
    assert source.count("assert_locked_test_closed") >= 2


def test_no_locked_split_token_used_for_candidate_diversity() -> None:
    # The module MAY discuss locked_final_test/locked_topology_test in
    # documentation/notes (it does, to explain why they're irrelevant to
    # seed-range disjointness) -- what matters is that the actual split
    # label passed into scenario generation is never one of them, and is
    # never the locked "test" split.
    assert proto.SPLIT_LABEL == "development_holdout"
    assert proto.SPLIT_LABEL not in ("test", "locked_final_test", "locked_topology_test")
    source = inspect.getsource(mod._build_family_population)
    assert "locked_final_test" not in source
    assert "locked_topology_test" not in source
    assert "proto.SPLIT_LABEL" in source
