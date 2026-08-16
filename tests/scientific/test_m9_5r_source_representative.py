"""Contract tests for Milestone 9.5R (`scripts/hydrocore_v5/m9_5r_common.py`,
`write_m9_5r_protocol.py`, `run_m9_5r_source_representative.py`,
`run_m9_5r_decide.py`) -- independent, one-shot confirmation of HydroCore-S
calibration at the already-predeclared adequate support level (20
independent physical calibration incidents/source) on a fresh,
source-representative population.

M9.5R does NOT reinterpret or overwrite M9.5 (formally closed,
M9_5_DECISION=E). These tests cover: exactly-one-support-level governance
(no 4/8/12 sweep), full source enumeration for trained families, exact
20/source calibration and development counts, seed-range disjointness from
M9.4/M9.5/M7/locked data, checkpoint identity/no-mutation, alpha/grouping/
fallback frozen, the corrected sanity gate's independence from any
historical-reproduction requirement, decision-D's restriction to actual
representativeness/implementation failures, and historical-artifact
immutability (M9.0a/M9.3/M9.4/M9.5).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT / "src"))

import m9_4_common as m4  # noqa: E402
import m9_5_common as m5  # noqa: E402
import m9_5r_common as m5r  # noqa: E402
import run_m7_topology as m7  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifacts_present() -> bool:
    return m5r.M9_5R_CANONICAL_CALIBRATION_PATH.exists() and m5r.M9_5R_CLOSURE_PATH.exists()


def _protocol_present() -> bool:
    return m5r.M9_5R_PROTOCOL_PATH.exists()


needs_artifacts = pytest.mark.skipif(not _artifacts_present(), reason="M9.5R pipeline has not been run yet in this environment")
needs_protocol = pytest.mark.skipif(not _protocol_present(), reason="M9.5R protocol has not been frozen yet in this environment")


# ---------------------------------------------------------------------------
# Governance constants.
# ---------------------------------------------------------------------------


def test_alpha_frozen_at_0_1():
    assert m5r.ALPHA == 0.1
    assert m5r.MINIMUM_GROUP_SIZE == 10
    assert m5r.OPERATIONAL_COVERAGE_FLOOR == 0.85
    assert m5r.NOMINAL_COVERAGE_TARGET == pytest.approx(0.90)


def test_exactly_one_primary_support_condition_of_20():
    assert m5r.PRIMARY_SUPPORT == 20
    assert m5r.CALIBRATION_REPEATS_PER_SOURCE == 20
    assert m5r.DEVELOPMENT_REPEATS_PER_SOURCE == 20


def test_no_support_sweep_exists():
    """M9.5R must NOT define any support-level tuple/sweep (Section 7/24:
    'no 4/8/12 support sweep exists')."""

    assert not hasattr(m5r, "SUPPORT_LEVELS")
    source_rep = (SCRIPTS_DIR / "run_m9_5r_source_representative.py").read_text()
    decide = (SCRIPTS_DIR / "run_m9_5r_decide.py").read_text()
    for forbidden in ("SUPPORT_LEVELS", "support_curve", "nested_support"):
        assert forbidden not in source_rep, f"source-representative module references {forbidden!r}"
        assert forbidden not in decide, f"decide module references {forbidden!r}"


def test_no_locked_test_access():
    assert m5r.assert_locked_test_closed() is False


def test_calibration_validity_scoped_to_trained_families_only():
    assert set(m5r.TRAINED_FAMILIES) == {"golden-reference", "branched-loop", "loop-grid"}
    assert set(m5r.TRAINED_FAMILIES) == set(m4.TRAINED_FAMILIES)


def test_decision_codes_frozen():
    assert set(m5r.DECISION_NAMES.keys()) == {"A", "B", "C", "D", "E", "F"}
    assert m5r.DECISION_NAMES["A"] == "INDEPENDENT_CALIBRATION_CONFIRMATION_PASS"
    assert m5r.DECISION_NAMES["D"] == "REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER"


# ---------------------------------------------------------------------------
# Section 8/9: complete source enumeration (trained families only), exactly
# 20 calibration + 20 development incidents per source, no truncation.
# ---------------------------------------------------------------------------


@pytest.mark.real_simulation
def test_full_junction_list_untruncated_for_trained_families():
    expected_counts = {"golden-reference": 4, "branched-loop": 7, "loop-grid": 8}
    for family, expected in expected_counts.items():
        junctions = m5r.full_junction_list(family, m5r.ALL_FAMILY_LOADERS[family])
        assert len(junctions) == expected
        assert junctions == tuple(sorted(junctions))


@pytest.mark.real_simulation
def test_generated_scenarios_are_exactly_20_per_source_calibration_and_development():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5r_source_representative as m95r  # noqa: E402

    family = "branched-loop"  # 7 sources -- would be truncated to 4 under the legacy EVAL_MAX_SOURCES policy.
    loader = m5r.ALL_FAMILY_LOADERS[family]
    junctions = set(m5r.full_junction_list(family, loader))
    for role, repeats in (("calibration_m9_5r", m5r.CALIBRATION_REPEATS_PER_SOURCE), ("development_m9_5r", m5r.DEVELOPMENT_REPEATS_PER_SOURCE)):
        pool = m95r._generate_m9_5r_scenarios(family, loader, role, repeats)
        covs = [cov for _r, cov in pool]
        sources = {cov["source_node"] for cov in covs}
        assert sources == junctions
        for j in junctions:
            assert sum(1 for cov in covs if cov["source_node"] == j) == 20


# ---------------------------------------------------------------------------
# Section 8: seed-range disjointness (M9.5R calibration vs development, and
# vs M9.4's, M9.5's, and M7's ranges).
# ---------------------------------------------------------------------------


def test_m9_5r_seed_bases_disjoint_from_m7_m9_4_and_m9_5_ranges():
    assert min(m7.SEED_BASES.values()) >= 940_000_000
    m9_4_ceiling = max(m4.M9_4_SEED_BASES.values()) + m4.M9_4_SEED_BASE_STEP
    assert m9_4_ceiling <= m5.M9_5_SEED_BASE_FLOOR
    m9_5_ceiling = max(m5.M9_5_SEED_BASES.values()) + m5.M9_5_SEED_BASE_STEP
    assert m9_5_ceiling <= m5r.M9_5R_SEED_BASE_FLOOR
    assert min(m5r.M9_5R_SEED_BASES.values()) >= m5r.M9_5R_SEED_BASE_FLOOR


def test_m9_5r_calibration_and_development_seed_ranges_disjoint_per_family():
    for family in m5r.TRAINED_FAMILIES:
        cal_base = m5r.m9_5r_seed_base(family, "calibration_m9_5r")
        dev_base = m5r.m9_5r_seed_base(family, "development_m9_5r")
        junctions = m5r.full_junction_list(family, m5r.ALL_FAMILY_LOADERS[family])
        cal_max = cal_base + (len(junctions) - 1) * m5r.M9_5R_SOURCE_STRIDE + (m5r.CALIBRATION_REPEATS_PER_SOURCE - 1)
        assert cal_max < dev_base, f"{family}: M9.5R calibration seed range overlaps development seed range"
        dev_max = dev_base + (len(junctions) - 1) * m5r.M9_5R_SOURCE_STRIDE + (m5r.DEVELOPMENT_REPEATS_PER_SOURCE - 1)
        next_family_index = m5r.TRAINED_FAMILIES.index(family) + 1
        if next_family_index < len(m5r.TRAINED_FAMILIES):
            next_cal_base = m5r.m9_5r_seed_base(m5r.TRAINED_FAMILIES[next_family_index], "calibration_m9_5r")
            assert dev_max < next_cal_base


def test_m9_5r_seed_base_table_covers_every_trained_family_and_role_exactly_once():
    expected_keys = {(family, role) for family in m5r.TRAINED_FAMILIES for role in m5r.M9_5R_ROLES}
    assert set(m5r.M9_5R_SEED_BASES.keys()) == expected_keys
    assert len(set(m5r.M9_5R_SEED_BASES.values())) == len(m5r.M9_5R_SEED_BASES)


@pytest.mark.real_simulation
def test_no_m9_4_or_m9_5_incident_reuse():
    """M9.5R's seed ranges must never overlap M9.4's or M9.5's seed ranges
    for the same family."""

    for family in m5r.TRAINED_FAMILIES:
        junctions = m5r.full_junction_list(family, m5r.ALL_FAMILY_LOADERS[family])
        n = len(junctions)

        m9_4_cal_base = m4.m9_4_seed_base(family, "calibration_m9_4")
        m9_4_dev_base = m4.m9_4_seed_base(family, "development_m9_4")
        m9_4_cal_max = m9_4_cal_base + (n - 1) * 1_000 + (m4.REPEATS_PER_SOURCE - 1)
        m9_4_dev_max = m9_4_dev_base + (n - 1) * 1_000 + (m4.REPEATS_PER_SOURCE - 1)

        m9_5_cal_base = m5.m9_5_seed_base(family, "calibration_m9_5")
        m9_5_dev_base = m5.m9_5_seed_base(family, "development_m9_5")
        m9_5_cal_max = m9_5_cal_base + (n - 1) * m5.M9_5_SOURCE_STRIDE + (m5.PRIMARY_SUPPORT - 1)
        m9_5_dev_max = m9_5_dev_base + (n - 1) * m5.M9_5_SOURCE_STRIDE + (m5.DEVELOPMENT_REPEATS_PER_SOURCE - 1)

        m9_5r_cal_base = m5r.m9_5r_seed_base(family, "calibration_m9_5r")
        m9_5r_dev_base = m5r.m9_5r_seed_base(family, "development_m9_5r")
        m9_5r_cal_max = m9_5r_cal_base + (n - 1) * m5r.M9_5R_SOURCE_STRIDE + (m5r.CALIBRATION_REPEATS_PER_SOURCE - 1)
        m9_5r_dev_max = m9_5r_dev_base + (n - 1) * m5r.M9_5R_SOURCE_STRIDE + (m5r.DEVELOPMENT_REPEATS_PER_SOURCE - 1)

        for other_base, other_max in (
            (m9_4_cal_base, m9_4_cal_max), (m9_4_dev_base, m9_4_dev_max),
            (m9_5_cal_base, m9_5_cal_max), (m9_5_dev_base, m9_5_dev_max),
        ):
            assert other_max < m9_5r_cal_base or m9_5r_cal_max < other_base
            assert other_max < m9_5r_dev_base or m9_5r_dev_max < other_base


# ---------------------------------------------------------------------------
# Checkpoint identity / no-mutation / no training.
# ---------------------------------------------------------------------------


def test_frozen_checkpoints_match_recorded_provenance_hashes():
    for seed in m5r.SEEDS:
        record = json.loads((m5r.RUNS_M8_7 / f"AGE_FIX_ONLY-seed{seed}.json").read_text())
        path = Path(record["export_path"])
        assert path.exists()
        assert _sha256_file(path) == record["checkpoint_sha256"]
    for seed in m5r.SEEDS:
        record = json.loads((m5r.RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
        ts = record["training_summary"]
        path = Path(ts["export_path"])
        assert path.exists()
        assert _sha256_file(path) == ts["export_sha256"]


@needs_artifacts
def test_manifest_records_checkpoint_hashes_unchanged_before_and_after_inference():
    manifest = json.loads(m5r.M9_5R_MANIFEST_PATH.read_text())
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m5r.SEEDS:
            entry = manifest["checkpoint_identities"][arm][str(seed)]
            assert entry["sha256_before"] == entry["sha256_after"], f"{arm} seed{seed} checkpoint mutated during M9.5R"


def test_source_representative_module_never_trains_or_backprops():
    source = (SCRIPTS_DIR / "run_m9_5r_source_representative.py").read_text()
    for forbidden in (".backward(", "optimizer.step(", "Optimizer(", "Trainer(", "torch.optim.", ".fit(resume_from"):
        assert forbidden not in source, f"forbidden training/backward invocation found: {forbidden!r}"
    assert "model.eval()" in source
    assert "torch.no_grad()" in source


def test_decide_module_never_trains_or_backprops():
    source = (SCRIPTS_DIR / "run_m9_5r_decide.py").read_text()
    for forbidden in (".backward(", "optimizer.step(", "Optimizer(", "Trainer("):
        assert forbidden not in source


def test_source_representative_module_never_opens_locked_test():
    source = (SCRIPTS_DIR / "run_m9_5r_source_representative.py").read_text()
    assert "locked_final_test" not in source
    assert "locked_topology_test" not in source
    assert "locked_test_opened" in source or "assert_locked_test_closed" in source


def test_decide_module_never_opens_locked_test():
    source = (SCRIPTS_DIR / "run_m9_5r_decide.py").read_text()
    assert "locked_final_test" not in source
    assert "locked_topology_test" not in source


# ---------------------------------------------------------------------------
# Calibration method identity: reused directly from hydroswarm.calibration.conformal.
# ---------------------------------------------------------------------------


def test_decide_reuses_unmodified_split_conformal_calibrator():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5r_decide as decide  # noqa: E402
    from hydroswarm.calibration.conformal import SplitConformalCalibrator

    assert decide.SplitConformalCalibrator is SplitConformalCalibrator


def test_no_alternative_calibration_method_introduced():
    for path in ("run_m9_5r_source_representative.py", "run_m9_5r_decide.py"):
        source = (SCRIPTS_DIR / path).read_text()
        for forbidden in ("APS", "RAPS", "TemperatureScal", "IsotonicRegression", "PlattScal", "alpha_sweep", "alpha=0.05", "alpha=0.15"):
            assert forbidden not in source, f"{path}: forbidden calibration-method deviation found: {forbidden!r}"


def test_candidate_set_inclusion_and_full_set_rate():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5r_decide as decide  # noqa: E402

    rows = [
        {"candidate_set_size": 1}, {"candidate_set_size": 4}, {"candidate_set_size": 4}, {"candidate_set_size": 2},
    ]
    for r in rows:
        r["candidate_covered"] = True
        r["depth_bucket"] = "MATURE"
    summary = decide._cov_summary(rows, n_nodes=4)
    assert summary["singleton_rate"] == pytest.approx(0.25)
    assert summary["full_set_rate"] == pytest.approx(0.5)
    assert summary["marginal_coverage"] == pytest.approx(1.0)


def test_candidate_set_guard_threshold_matches_m9_5_rule():
    # Same rule M9.5 used: pathological iff full_set_rate > 0.8 at the primary support level.
    assert m5r.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD == 0.8


# ---------------------------------------------------------------------------
# Section 11: corrected sanity gate must NOT depend on reproducing any
# historical coverage value.
# ---------------------------------------------------------------------------


def test_sanity_gate_never_references_historical_coverage_reproduction():
    source = (SCRIPTS_DIR / "run_m9_5r_decide.py").read_text()
    for forbidden in ("qualitatively_consistent_with_m9_4", "m9_4_like_bad_coverage", "reproduce_m9_4", "reproduce_m9_5"):
        assert forbidden not in source


def test_sanity_gate_checks_are_the_predeclared_twelve():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import inspect

    import run_m9_5r_decide as decide  # noqa: E402

    src = inspect.getsource(decide._sanity_gate)
    expected = [
        "A_checkpoint_sha_identity", "B_calibrator_class_identity_matches_m9_5", "C_alpha_equals_0_1",
        "D_nonconformity_score_implementation_identity_matches", "E_grouping_construction_identity_matches",
        "F_maturity_depth_mapping_identity_matches", "G_candidate_set_inclusion_rule_matches",
        "H_source_node_ordering_correct", "I_calibration_development_split_disjointness_passes",
        "J_all_source_nodes_represented", "K_all_outputs_finite", "L_resubstitution_diagnostic_numerically_plausible",
    ]
    for key in expected:
        assert key in src


@needs_artifacts
def test_sanity_gate_artifact_does_not_encode_any_specific_numerical_target():
    gate = json.loads(m5r.M9_5R_SANITY_GATE_PATH.read_text())
    assert set(gate["checks"].keys()) == {
        "A_checkpoint_sha_identity", "B_calibrator_class_identity_matches_m9_5", "C_alpha_equals_0_1",
        "D_nonconformity_score_implementation_identity_matches", "E_grouping_construction_identity_matches",
        "F_maturity_depth_mapping_identity_matches", "G_candidate_set_inclusion_rule_matches",
        "H_source_node_ordering_correct", "I_calibration_development_split_disjointness_passes",
        "J_all_source_nodes_represented", "K_all_outputs_finite", "L_resubstitution_diagnostic_numerically_plausible",
    }
    assert gate["M9_5R_SANITY_GATE"] in ("PASS", "FAIL")


# ---------------------------------------------------------------------------
# Decision D can only occur from an actual representativeness or
# implementation invariant failure -- never merely from differing coverage.
# ---------------------------------------------------------------------------


def test_decision_d_requires_representativeness_or_sanity_failure():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5r_decide as decide  # noqa: E402

    # representativeness fails, sanity passes -> D
    code, name, _ = decide._decide(
        False, {"M9_5R_SANITY_GATE": "PASS"}, {"all_3_pass": True}, {"all_9_cells_pass": True},
        {"pathological_full_set_behavior_detected": False},
    )
    assert code == "D"
    assert name == "REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER"

    # representativeness passes, sanity fails -> D
    code, name, _ = decide._decide(
        True, {"M9_5R_SANITY_GATE": "FAIL"}, {"all_3_pass": True}, {"all_9_cells_pass": True},
        {"pathological_full_set_behavior_detected": False},
    )
    assert code == "D"

    # both pass, but CURRENT control fails -> C, NOT D
    code, name, _ = decide._decide(
        True, {"M9_5R_SANITY_GATE": "PASS"}, {"all_3_pass": False}, {"all_9_cells_pass": True},
        {"pathological_full_set_behavior_detected": False},
    )
    assert code == "C"

    # both pass, control passes, interleaved fails -> B, NOT D
    code, name, _ = decide._decide(
        True, {"M9_5R_SANITY_GATE": "PASS"}, {"all_3_pass": True}, {"all_9_cells_pass": False},
        {"pathological_full_set_behavior_detected": False},
    )
    assert code == "B"

    # everything passes except candidate-set guard -> E, NOT D
    code, name, _ = decide._decide(
        True, {"M9_5R_SANITY_GATE": "PASS"}, {"all_3_pass": True}, {"all_9_cells_pass": True},
        {"pathological_full_set_behavior_detected": True},
    )
    assert code == "E"

    # everything passes -> A
    code, name, _ = decide._decide(
        True, {"M9_5R_SANITY_GATE": "PASS"}, {"all_3_pass": True}, {"all_9_cells_pass": True},
        {"pathological_full_set_behavior_detected": False},
    )
    assert code == "A"


# ---------------------------------------------------------------------------
# All 9 ARM_B2 cells / all 3 ARM_A cells required -- no averaging away.
# ---------------------------------------------------------------------------


def test_interleaved_gate_requires_all_9_cells_no_averaging():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5r_decide as decide  # noqa: E402

    families = list(m5r.TRAINED_FAMILIES)
    seeds = list(m5r.SEEDS)
    calibration_evaluation = {"ARM_B2": {str(s): {f: {"marginal_coverage": 0.90} for f in families} for s in seeds}}
    # degrade exactly one cell below the floor
    calibration_evaluation["ARM_B2"][str(seeds[0])][families[0]]["marginal_coverage"] = 0.80
    gate = decide._interleaved_confirmation_gate(calibration_evaluation)
    assert gate["all_9_cells_pass"] is False
    assert len(gate["per_family_seed_coverage"]) == 9


def test_current_control_gate_requires_all_3_seeds():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5r_decide as decide  # noqa: E402

    seeds = list(m5r.SEEDS)
    calibration_evaluation = {"ARM_A": {str(s): {"golden-reference": {"marginal_coverage": 0.90}} for s in seeds}}
    calibration_evaluation["ARM_A"][str(seeds[1])]["golden-reference"]["marginal_coverage"] = 0.5
    gate = decide._current_control_gate(calibration_evaluation)
    assert gate["all_3_pass"] is False
    assert len(gate["per_seed_coverage"]) == 3


# ---------------------------------------------------------------------------
# Historical artifact immutability (M9.0a/M9.3/M9.4 AND M9.5).
# ---------------------------------------------------------------------------


def test_historical_generators_m9_4_and_m9_5_files_untouched_in_place():
    import subprocess

    for path in (
        "scripts/hydrocore_v5/run_m7_topology.py",
        "scripts/hydrocore_v5/run_m9_0a_evaluate.py",
        "scripts/hydrocore_v5/run_m9_0a_decide.py",
        "scripts/hydrocore_v5/m9_3_common.py",
        "scripts/hydrocore_v5/m9_4_common.py",
        "scripts/hydrocore_v5/run_m9_4_source_representative.py",
        "scripts/hydrocore_v5/run_m9_4_decide.py",
        "scripts/hydrocore_v5/m9_5_common.py",
        "scripts/hydrocore_v5/run_m9_5_source_representative.py",
        "scripts/hydrocore_v5/run_m9_5_decide.py",
    ):
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", path], cwd=ROOT, capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "", f"{path} was modified relative to HEAD: {result.stdout}"


def test_m9_5_closure_still_reads_decision_e_unaltered():
    """M9.5R must never reinterpret or overwrite M9.5's formal closure."""

    closure = json.loads(m5.M9_5_CLOSURE_PATH.read_text())
    assert closure["M9_5_DECISION"] == "E"
    assert closure["M9_5_DECISION_NAME"] == "REPRESENTATIVENESS_OR_IMPLEMENTATION_BLOCKER"


# ---------------------------------------------------------------------------
# Protocol-freeze artifact (Section 20/23): must exist and be frozen BEFORE
# any inference-dependent artifact.
# ---------------------------------------------------------------------------


@needs_protocol
def test_protocol_freeze_predeclares_support_seed_gates_and_decision_rules():
    protocol = json.loads(m5r.M9_5R_PROTOCOL_PATH.read_text())
    assert protocol["primary_support_condition"]["calibration_repeats_per_source"] == 20
    assert protocol["primary_support_condition"]["development_repeats_per_source"] == 20
    assert protocol["primary_support_condition"]["no_support_sweep"] is True
    assert protocol["calibration_method_frozen"]["alpha"] == 0.1
    assert "seed_namespace" in protocol
    assert "corrected_sanity_gate" in protocol
    assert "decision_logic" in protocol


@needs_artifacts
def test_manifest_protocol_freeze_commit_precedes_or_equals_execution_commit():
    """The protocol must have been frozen at or before the commit the
    source-representative pipeline actually ran at (never generated after
    seeing results)."""

    manifest = json.loads(m5r.M9_5R_MANIFEST_PATH.read_text())
    assert manifest["protocol_frozen_at_commit"] == manifest["start_commit"]


# ---------------------------------------------------------------------------
# Artifact-dependent contract tests (require the M9.5R pipeline to have run).
# ---------------------------------------------------------------------------


@needs_artifacts
def test_representativeness_audit_covers_every_trained_family_source():
    audit = json.loads(m5r.M9_5R_REPRESENTATIVENESS_AUDIT_PATH.read_text())
    for family in m5r.TRAINED_FAMILIES:
        checks = audit["families"][family]["checks"]
        assert checks["all_sources_in_calibration"]
        assert checks["all_sources_in_development"]
        assert checks["exactly_20_calibration_incidents_per_source"]
        assert checks["exactly_20_development_incidents_per_source"]
        assert checks["no_zero_support_source"]
        assert checks["seed_disjoint_calibration_vs_development"]
        assert checks["no_scenario_id_overlap"]


@needs_artifacts
def test_source_policy_reports_exact_20_per_source_counts():
    policy = json.loads(m5r.M9_5R_SOURCE_POLICY_PATH.read_text())
    for family in m5r.TRAINED_FAMILIES:
        entry = policy["families"][family]
        n_sources = entry["n_sources"]
        assert entry["calibration"]["n_incidents"] == n_sources * 20
        assert entry["development"]["n_incidents"] == n_sources * 20
        for count in entry["calibration"]["n_incidents_per_source"].values():
            assert count == 20
        for count in entry["development"]["n_incidents_per_source"].values():
            assert count == 20


@needs_artifacts
def test_canonical_calibration_rows_have_no_cross_family_source_leakage():
    seen_sources_by_family: dict[str, set[str]] = {}
    with m5r.M9_5R_CANONICAL_CALIBRATION_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            seen_sources_by_family.setdefault(row["family"], set()).add(row["source_node"])
    for family, sources in seen_sources_by_family.items():
        expected = set(m5r.full_junction_list(family, m5r.ALL_FAMILY_LOADERS[family]))
        assert sources <= expected, f"{family}: unexpected source(s) {sources - expected}"


@needs_artifacts
def test_closure_alpha_and_coverage_floor_never_weakened():
    closure = json.loads(m5r.M9_5R_CLOSURE_PATH.read_text())
    assert closure["alpha"] == 0.1
    assert closure["coverage_floor"] == 0.85
    assert closure["locked_test_opened_before"] is False
    assert closure["locked_test_opened_after"] is False
    assert closure["no_training_performed"] is True
    assert closure["no_predictor_modified"] is True
    assert closure["calibration_repeats_per_source"] == 20
    assert closure["development_repeats_per_source"] == 20
    assert closure["M9_5R_DECISION"] in ("A", "B", "C", "D", "E", "F")


@needs_artifacts
def test_closure_decision_d_iff_representativeness_or_sanity_failed():
    closure = json.loads(m5r.M9_5R_CLOSURE_PATH.read_text())
    if closure["M9_5R_DECISION"] == "D":
        assert (not closure["representativeness_audit_passed"]) or (not closure["sanity_gate_passed"])
    else:
        assert closure["representativeness_audit_passed"] and closure["sanity_gate_passed"]


@needs_artifacts
def test_closure_requires_all_9_and_all_3_cells_present():
    closure = json.loads(m5r.M9_5R_CLOSURE_PATH.read_text())
    assert len(closure["interleaved"]["per_family_seed_coverage"]) == 9
    assert len(closure["current_control"]["per_seed_coverage"]) == 3
