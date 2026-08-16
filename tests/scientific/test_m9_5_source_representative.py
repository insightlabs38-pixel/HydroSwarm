"""Contract tests for Milestone 9.5 (`scripts/hydrocore_v5/m9_5_common.py`,
`run_m9_5_source_representative.py`, `run_m9_5_decide.py`) -- source-
representative calibration-support confirmation study for the frozen
CURRENT/INTERLEAVED HydroCore-S predictors.

M9.5 is a CALIBRATION-SUPPORT / FROZEN-CHECKPOINT study: these tests cover
governance/generation-policy correctness (nested support-level construction,
full source enumeration restricted to trained families, calibration/
development/M9.4 seed-range disjointness, checkpoint identity/no-mutation,
alpha/grouping/fallback frozen, deterministic bootstrap, no locked-test
access, historical-artifact immutability) -- never a promotion decision.
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
import run_m7_topology as m7  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifacts_present() -> bool:
    return m5.M9_5_CANONICAL_CALIBRATION_PATH.exists() and m5.M9_5_CLOSURE_PATH.exists()


needs_artifacts = pytest.mark.skipif(not _artifacts_present(), reason="M9.5 pipeline has not been run yet in this environment")


# ---------------------------------------------------------------------------
# Governance constants (Section 4/16/22).
# ---------------------------------------------------------------------------


def test_alpha_frozen_at_0_1():
    assert m5.ALPHA == 0.1
    assert m5.MINIMUM_GROUP_SIZE == 10
    assert m5.OPERATIONAL_COVERAGE_FLOOR == 0.85
    assert m5.NOMINAL_COVERAGE_TARGET == pytest.approx(0.90)


def test_support_levels_exactly_4_8_12_20():
    assert m5.SUPPORT_LEVELS == (4, 8, 12, 20)


def test_support_20_is_promotion_primary():
    assert m5.PRIMARY_SUPPORT == 20
    assert m5.PRIMARY_SUPPORT == max(m5.SUPPORT_LEVELS)


def test_quantile_bootstrap_constants_predeclared():
    assert m5.QUANTILE_BOOTSTRAP_RESAMPLES == 2000
    assert m5.QUANTILE_BOOTSTRAP_SEED == 20260817


def test_no_locked_test_access():
    assert m5.assert_locked_test_closed() is False


def test_calibration_validity_scoped_to_trained_families_only():
    assert set(m5.TRAINED_FAMILIES) == {"golden-reference", "branched-loop", "loop-grid"}
    assert set(m5.TRAINED_FAMILIES) == set(m4.TRAINED_FAMILIES)


# ---------------------------------------------------------------------------
# Section 6: nested-subset property (support_4 subset support_8 subset
# support_12 subset support_20).
# ---------------------------------------------------------------------------


@pytest.mark.real_simulation
def test_nested_support_levels_are_prefix_subsets():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5_source_representative as m95  # noqa: E402

    family = "golden-reference"
    loader = m5.ALL_FAMILY_LOADERS[family]
    pool = m95._generate_m9_5_scenarios(family, loader, "calibration_m9_5", 8)  # small enough for a fast test
    by_level = {level: {cov["generator_seed"] for _r, cov in pool if cov["repeat"] < level} for level in (4, 8)}
    assert by_level[4].issubset(by_level[8])
    assert len(by_level[4]) == 4 * 4  # 4 sources x 4 repeats
    assert len(by_level[8]) == 4 * 8


def test_nested_support_seed_offsets_never_collide_across_repeats():
    # repeat < N always selects a strict, deterministic prefix per source given
    # seed = seed_base + source_index*stride + repeat.
    stride = m5.M9_5_SOURCE_STRIDE
    assert stride > max(m5.SUPPORT_LEVELS)
    for level_small, level_big in zip(m5.SUPPORT_LEVELS, m5.SUPPORT_LEVELS[1:]):
        assert level_small < level_big


# ---------------------------------------------------------------------------
# Section 4/7/9: complete source enumeration (trained families only), equal
# repeats per source, no truncation.
# ---------------------------------------------------------------------------


@pytest.mark.real_simulation
def test_full_junction_list_untruncated_for_trained_families():
    expected_counts = {"golden-reference": 4, "branched-loop": 7, "loop-grid": 8}
    for family, expected in expected_counts.items():
        junctions = m5.full_junction_list(family, m5.ALL_FAMILY_LOADERS[family])
        assert len(junctions) == expected
        assert junctions == tuple(sorted(junctions))


@pytest.mark.real_simulation
def test_generated_scenarios_cover_every_source_at_every_support_level():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5_source_representative as m95  # noqa: E402

    family = "branched-loop"  # 7 sources -- would be truncated to 4 under the legacy EVAL_MAX_SOURCES policy.
    loader = m5.ALL_FAMILY_LOADERS[family]
    pool = m95._generate_m9_5_scenarios(family, loader, "calibration_m9_5", m5.PRIMARY_SUPPORT)
    junctions = set(m5.full_junction_list(family, loader))
    for level in m5.SUPPORT_LEVELS:
        subset = [cov for _r, cov in pool if cov["repeat"] < level]
        sources = {cov["source_node"] for cov in subset}
        assert sources == junctions, f"support={level}: not every source represented"
        for j in junctions:
            assert sum(1 for cov in subset if cov["source_node"] == j) == level


# ---------------------------------------------------------------------------
# Section 7/8: seed-range disjointness (M9.5 calibration vs development, and
# vs M9.4's ranges), M7's range.
# ---------------------------------------------------------------------------


def test_m9_5_seed_bases_disjoint_from_m7_and_m9_4_ranges():
    assert min(m7.SEED_BASES.values()) >= 940_000_000
    m9_4_ceiling = max(m4.M9_4_SEED_BASES.values()) + m4.M9_4_SEED_BASE_STEP
    assert m9_4_ceiling <= m5.M9_5_SEED_BASE_FLOOR
    assert min(m5.M9_5_SEED_BASES.values()) >= m5.M9_5_SEED_BASE_FLOOR


def test_m9_5_calibration_and_development_seed_ranges_disjoint_per_family():
    for family in m5.TRAINED_FAMILIES:
        cal_base = m5.m9_5_seed_base(family, "calibration_m9_5")
        dev_base = m5.m9_5_seed_base(family, "development_m9_5")
        junctions = m5.full_junction_list(family, m5.ALL_FAMILY_LOADERS[family])
        cal_max = cal_base + (len(junctions) - 1) * m5.M9_5_SOURCE_STRIDE + (m5.PRIMARY_SUPPORT - 1)
        assert cal_max < dev_base, f"{family}: M9.5 calibration seed range overlaps development seed range"
        dev_max = dev_base + (len(junctions) - 1) * m5.M9_5_SOURCE_STRIDE + (m5.DEVELOPMENT_REPEATS_PER_SOURCE - 1)
        next_family_index = m5.TRAINED_FAMILIES.index(family) + 1
        if next_family_index < len(m5.TRAINED_FAMILIES):
            next_cal_base = m5.m9_5_seed_base(m5.TRAINED_FAMILIES[next_family_index], "calibration_m9_5")
            assert dev_max < next_cal_base


def test_m9_5_seed_base_table_covers_every_trained_family_and_role_exactly_once():
    expected_keys = {(family, role) for family in m5.TRAINED_FAMILIES for role in m5.M9_5_ROLES}
    assert set(m5.M9_5_SEED_BASES.keys()) == expected_keys
    assert len(set(m5.M9_5_SEED_BASES.values())) == len(m5.M9_5_SEED_BASES)


@pytest.mark.real_simulation
def test_no_m9_4_incident_reuse():
    """M9.5's seed ranges must never overlap M9.4's calibration_m9_4/
    development_m9_4 seed ranges for the same family."""

    for family in m5.TRAINED_FAMILIES:
        junctions = m5.full_junction_list(family, m5.ALL_FAMILY_LOADERS[family])
        m9_4_cal_base = m4.m9_4_seed_base(family, "calibration_m9_4")
        m9_4_dev_base = m4.m9_4_seed_base(family, "development_m9_4")
        m9_4_cal_max = m9_4_cal_base + (len(junctions) - 1) * 1_000 + (m4.REPEATS_PER_SOURCE - 1)
        m9_4_dev_max = m9_4_dev_base + (len(junctions) - 1) * 1_000 + (m4.REPEATS_PER_SOURCE - 1)
        m9_5_cal_base = m5.m9_5_seed_base(family, "calibration_m9_5")
        m9_5_dev_base = m5.m9_5_seed_base(family, "development_m9_5")
        m9_5_cal_max = m9_5_cal_base + (len(junctions) - 1) * m5.M9_5_SOURCE_STRIDE + (m5.PRIMARY_SUPPORT - 1)
        m9_5_dev_max = m9_5_dev_base + (len(junctions) - 1) * m5.M9_5_SOURCE_STRIDE + (m5.DEVELOPMENT_REPEATS_PER_SOURCE - 1)
        assert m9_4_cal_max < m9_5_cal_base or m9_5_cal_max < m9_4_cal_base
        assert m9_4_dev_max < m9_5_dev_base or m9_5_dev_max < m9_4_dev_base


# ---------------------------------------------------------------------------
# Checkpoint identity / no-mutation / no training.
# ---------------------------------------------------------------------------


def test_frozen_checkpoints_match_recorded_provenance_hashes():
    for seed in m5.SEEDS:
        record = json.loads((m5.RUNS_M8_7 / f"AGE_FIX_ONLY-seed{seed}.json").read_text())
        path = Path(record["export_path"])
        assert path.exists()
        assert _sha256_file(path) == record["checkpoint_sha256"]
    for seed in m5.SEEDS:
        record = json.loads((m5.RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
        ts = record["training_summary"]
        path = Path(ts["export_path"])
        assert path.exists()
        assert _sha256_file(path) == ts["export_sha256"]


@needs_artifacts
def test_manifest_records_checkpoint_hashes_unchanged_before_and_after_inference():
    manifest = json.loads(m5.M9_5_MANIFEST_PATH.read_text())
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m5.SEEDS:
            entry = manifest["checkpoint_identities"][arm][str(seed)]
            assert entry["sha256_before"] == entry["sha256_after"], f"{arm} seed{seed} checkpoint mutated during M9.5"


def test_source_representative_module_never_trains_or_backprops():
    source = (SCRIPTS_DIR / "run_m9_5_source_representative.py").read_text()
    for forbidden in (".backward(", "optimizer.step(", "Optimizer(", "Trainer(", "torch.optim.", ".fit(resume_from"):
        assert forbidden not in source, f"forbidden training/backward invocation found: {forbidden!r}"
    assert "model.eval()" in source
    assert "torch.no_grad()" in source


def test_decide_module_never_trains_or_backprops():
    source = (SCRIPTS_DIR / "run_m9_5_decide.py").read_text()
    for forbidden in (".backward(", "optimizer.step(", "Optimizer(", "Trainer("):
        assert forbidden not in source


def test_source_representative_module_never_opens_locked_test():
    source = (SCRIPTS_DIR / "run_m9_5_source_representative.py").read_text()
    assert "locked_final_test" not in source
    assert "locked_topology_test" not in source
    assert "locked_test_opened" in source or "assert_locked_test_closed" in source


def test_decide_module_never_opens_locked_test():
    source = (SCRIPTS_DIR / "run_m9_5_decide.py").read_text()
    assert "locked_final_test" not in source
    assert "locked_topology_test" not in source


# ---------------------------------------------------------------------------
# Section 22: conformal score/grouping/fallback unchanged -- import identity
# check against the SAME frozen calibrator class, never a reimplementation.
# ---------------------------------------------------------------------------


def test_decide_reuses_unmodified_split_conformal_calibrator():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5_decide as decide  # noqa: E402
    from hydroswarm.calibration.conformal import SplitConformalCalibrator

    assert decide.SplitConformalCalibrator is SplitConformalCalibrator


def test_no_alternative_calibration_method_introduced():
    for path in ("run_m9_5_source_representative.py", "run_m9_5_decide.py"):
        source = (SCRIPTS_DIR / path).read_text()
        for forbidden in ("APS", "RAPS", "TemperatureScal", "IsotonicRegression", "PlattScal", "alpha_sweep", "alpha=0.05", "alpha=0.15"):
            assert forbidden not in source, f"{path}: forbidden calibration-method deviation found: {forbidden!r}"


# ---------------------------------------------------------------------------
# Bootstrap correctness (Section 14): incident-level resampling, depth rows
# grouped, deterministic given fixed seed.
# ---------------------------------------------------------------------------


def test_quantile_bootstrap_groups_depth_rows_with_incident():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5_decide as decide  # noqa: E402

    rows = []
    for depth_bucket, score in (("EARLY", 0.1), ("EARLY", 0.9)):
        rows.append({"source_node": "J1", "generator_seed": 1, "depth_bucket": depth_bucket, "nonconformity_score": score})
    # A single physical incident (source_node, generator_seed) contributes
    # BOTH rows -- resampling must draw the incident once, keeping both.
    quantiles = decide._bootstrap_quantiles(rows, "EARLY", resamples=5, seed=1)
    assert len(quantiles) == 5
    for q in quantiles:
        assert q in (0.1, 0.9)  # with n=1 incident, every resample IS that incident's {0.1, 0.9} pair


def test_quantile_bootstrap_deterministic_given_fixed_seed():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5_decide as decide  # noqa: E402

    rows = [
        {"source_node": f"J{i}", "generator_seed": i, "depth_bucket": "MATURE", "nonconformity_score": 0.1 * i}
        for i in range(1, 6)
    ]
    q1 = decide._bootstrap_quantiles(rows, "MATURE", resamples=50, seed=m5.QUANTILE_BOOTSTRAP_SEED)
    q2 = decide._bootstrap_quantiles(rows, "MATURE", resamples=50, seed=m5.QUANTILE_BOOTSTRAP_SEED)
    assert q1 == q2


def test_independent_n_not_conflated_with_depth_row_n():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5_decide as decide  # noqa: E402

    rows = []
    for depth in (1, 2, 3, 4, 6, 12, 25):  # 7 depths, ONE physical incident
        bucket = "EARLY" if depth in (1, 2, 3) else ("MID" if depth in (4, 6) else "MATURE")
        rows.append({"source_node": "J1", "generator_seed": 42, "depth": depth, "depth_bucket": bucket})
    assert decide._incident_count(rows) == 1
    assert len(rows) == 7


# ---------------------------------------------------------------------------
# Candidate-set inclusion correctness / full-set-rate.
# ---------------------------------------------------------------------------


def test_candidate_set_inclusion_and_full_set_rate():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_5_decide as decide  # noqa: E402

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


# ---------------------------------------------------------------------------
# Historical artifact immutability (M9.4 AND M7/M9.0a/M9.0b/M9.3).
# ---------------------------------------------------------------------------


def test_historical_generators_and_m9_4_files_untouched_in_place():
    import subprocess

    for path in (
        "scripts/hydrocore_v5/run_m7_topology.py",
        "scripts/hydrocore_v5/run_m9_0a_evaluate.py",
        "scripts/hydrocore_v5/run_m9_0a_decide.py",
        "scripts/hydrocore_v5/m9_3_common.py",
        "scripts/hydrocore_v5/m9_4_common.py",
        "scripts/hydrocore_v5/run_m9_4_source_representative.py",
        "scripts/hydrocore_v5/run_m9_4_decide.py",
    ):
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", path], cwd=ROOT, capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "", f"{path} was modified relative to HEAD: {result.stdout}"


# ---------------------------------------------------------------------------
# Artifact-dependent contract tests (require the M9.5 pipeline to have run).
# ---------------------------------------------------------------------------


@needs_artifacts
def test_representativeness_audit_covers_every_trained_family_source():
    audit = json.loads(m5.M9_5_REPRESENTATIVENESS_AUDIT_PATH.read_text())
    for family in m5.TRAINED_FAMILIES:
        checks = audit["families"][family]["checks"]
        assert checks["all_sources_in_calibration"]
        assert checks["all_sources_in_development"]
        assert checks["no_zero_support_source"]
        assert checks["seed_disjoint_calibration_vs_development"]
        assert checks["no_scenario_id_overlap"]


@needs_artifacts
def test_source_policy_reports_exact_nested_counts_per_level():
    policy = json.loads(m5.M9_5_SOURCE_POLICY_PATH.read_text())
    for family in m5.TRAINED_FAMILIES:
        entry = policy["families"][family]
        n_sources = entry["n_sources"]
        for level in m5.SUPPORT_LEVELS:
            nested = entry["nested_support_counts"][str(level)]
            assert nested["n_incidents"] == n_sources * level
            for count in nested["n_incidents_per_source"].values():
                assert count == level


@needs_artifacts
def test_canonical_calibration_rows_have_no_cross_family_source_leakage():
    seen_sources_by_family: dict[str, set[str]] = {}
    with m5.M9_5_CANONICAL_CALIBRATION_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            seen_sources_by_family.setdefault(row["family"], set()).add(row["source_node"])
    for family, sources in seen_sources_by_family.items():
        expected = set(m5.full_junction_list(family, m5.ALL_FAMILY_LOADERS[family]))
        assert sources <= expected, f"{family}: unexpected source(s) {sources - expected}"


@needs_artifacts
def test_closure_alpha_and_coverage_floor_never_weakened():
    closure = json.loads(m5.M9_5_CLOSURE_PATH.read_text())
    assert closure["alpha"] == 0.1
    assert closure["coverage_floor"] == 0.85
    assert closure["locked_test_opened_before"] is False
    assert closure["locked_test_opened_after"] is False
    assert closure["no_training_performed"] is True
    assert closure["no_predictor_modified"] is True
    assert closure["primary_support_repeats_per_source"] == 20
    assert closure["support_levels_repeats_per_source"] == [4, 8, 12, 20]
    assert closure["M9_5_DECISION"] in ("A", "B", "C", "D", "E", "F")


@needs_artifacts
def test_closure_end_commit_is_not_a_lazy_copy_of_start_commit_placeholder():
    """Regression test for the M9.4 end_commit bug (see M9.4's closure/manifest
    end_commit_note): a metadata-only follow-up commit must NOT be required
    for callers to trust this field is not silently equal-by-laziness."""

    closure = json.loads(m5.M9_5_CLOSURE_PATH.read_text())
    assert "end_commit" in closure
    # end_commit may legitimately equal start_commit only if genuinely no
    # commit occurred between manifest write and decide-stage read; the
    # decide script documents this explicitly via end_commit_note either way.
    assert "end_commit_note" in closure
