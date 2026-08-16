"""Contract tests for Milestone 9.4 (`scripts/hydrocore_v5/m9_4_common.py`,
`run_m9_4_source_representative.py`, `run_m9_4_decide.py`) -- source-
representative, exchangeability-corrected re-evaluation of the frozen
CURRENT/INTERLEAVED HydroCore-S predictors.

M9.4 is a FROZEN-CHECKPOINT RE-EVALUATION milestone: these tests cover
governance/generation-policy correctness (full source enumeration, no
EVAL_MAX_SOURCES-style truncation, calibration/development exchangeability
by construction, checkpoint identity/no-mutation, deterministic bootstrap,
alpha frozen at 0.1, no locked-test access, historical-artifact
immutability) -- never a promotion decision.
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
import run_m7_topology as m7  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifacts_present() -> bool:
    return m4.M9_4_PREDICTIONS_PATH.exists() and m4.M9_4_CLOSURE_PATH.exists()


needs_artifacts = pytest.mark.skipif(not _artifacts_present(), reason="M9.4 pipeline has not been run yet in this environment")


# ---------------------------------------------------------------------------
# Governance constants.
# ---------------------------------------------------------------------------


def test_alpha_frozen_at_0_1():
    assert m4.ALPHA == 0.1
    assert m4.MINIMUM_GROUP_SIZE == 10
    assert m4.OPERATIONAL_COVERAGE_FLOOR == 0.85
    assert m4.NOMINAL_COVERAGE_TARGET == pytest.approx(0.90)


def test_bootstrap_constants_predeclared():
    assert m4.BOOTSTRAP_RESAMPLES == 2000
    assert m4.BOOTSTRAP_SEED == 20260816
    assert m4.BOOTSTRAP_INTERVAL == 0.90


def test_guardrail_thresholds_match_frozen_m9_0a_values():
    assert m4.GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP == 5.0
    assert m4.GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP == 3.0
    assert m4.GUARDRAIL_MAX_MRR_REGRESSION == 0.03


def test_no_locked_test_access():
    assert m4.assert_locked_test_closed() is False


# ---------------------------------------------------------------------------
# Section 4: complete source enumeration, no EVAL_MAX_SOURCES-style truncation.
# ---------------------------------------------------------------------------


@pytest.mark.real_simulation
def test_full_junction_list_untruncated_for_every_family():
    expected_counts = {
        "golden-reference": 4, "branched-loop": 7, "loop-grid": 8,
        "coastal-branch": 6, "tree-branch": 5, "dense-loop": 6,
    }
    for family, expected in expected_counts.items():
        junctions = m4.full_junction_list(family, m4.ALL_FAMILY_LOADERS[family])
        assert len(junctions) == expected, f"{family}: expected {expected} junctions, got {len(junctions)}"
        assert junctions == tuple(sorted(junctions))


def test_families_with_more_than_four_sources_exceed_legacy_eval_max_sources():
    assert m7.EVAL_MAX_SOURCES == 4
    multi_source = [f for f in m4.ALL_FAMILIES if len(m4.full_junction_list(f, m4.ALL_FAMILY_LOADERS[f])) > 4]
    assert set(multi_source) == {"branched-loop", "loop-grid", "coastal-branch", "tree-branch", "dense-loop"}


@pytest.mark.real_simulation
def test_generated_scenarios_never_truncate_below_full_source_count():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_4_source_representative as m94  # noqa: E402

    family = "branched-loop"  # 7 sources -- would be truncated to 4 under the legacy policy.
    loader = m4.ALL_FAMILY_LOADERS[family]
    scenarios = m94._generate_m9_4_scenarios(family, loader, "development_m9_4")
    sources = {cov["source_node"] for _record, cov in scenarios}
    assert sources == set(m4.full_junction_list(family, loader))
    assert len(scenarios) == 7 * m4.REPEATS_PER_SOURCE


# ---------------------------------------------------------------------------
# Section 4/5: predeclared equal repeats, seed disjointness, no scenario overlap.
# ---------------------------------------------------------------------------


def test_repeats_per_source_predeclared():
    assert m4.REPEATS_PER_SOURCE == 4


def test_m9_4_seed_bases_disjoint_from_m7_seed_bases():
    assert min(m7.SEED_BASES.values()) >= 940_000_000
    assert max(m7.SEED_BASES.values()) < m4.M9_4_SEED_BASE_FLOOR
    assert min(m4.M9_4_SEED_BASES.values()) >= m4.M9_4_SEED_BASE_FLOOR


def test_calibration_and_development_seed_ranges_disjoint_per_family():
    for family in m4.ALL_FAMILIES:
        cal_base = m4.m9_4_seed_base(family, "calibration_m9_4")
        dev_base = m4.m9_4_seed_base(family, "development_m9_4")
        junctions = m4.full_junction_list(family, m4.ALL_FAMILY_LOADERS[family])
        cal_max = cal_base + (len(junctions) - 1) * 1_000 + (m4.REPEATS_PER_SOURCE - 1)
        assert cal_max < dev_base, f"{family}: calibration seed range overlaps development seed range"


def test_m9_4_seed_base_table_covers_every_family_and_role_exactly_once():
    expected_keys = {(family, role) for family in m4.ALL_FAMILIES for role in m4.M9_4_ROLES}
    assert set(m4.M9_4_SEED_BASES.keys()) == expected_keys
    assert len(set(m4.M9_4_SEED_BASES.values())) == len(m4.M9_4_SEED_BASES), "seed bases must be pairwise distinct"


@pytest.mark.real_simulation
def test_no_physical_scenario_seed_overlap_between_roles():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_4_source_representative as m94  # noqa: E402

    family = "golden-reference"
    loader = m4.ALL_FAMILY_LOADERS[family]
    cal = m94._generate_m9_4_scenarios(family, loader, "calibration_m9_4")
    dev = m94._generate_m9_4_scenarios(family, loader, "development_m9_4")
    cal_seeds = {cov["generator_seed"] for _r, cov in cal}
    dev_seeds = {cov["generator_seed"] for _r, cov in dev}
    assert cal_seeds.isdisjoint(dev_seeds)
    cal_ids = {cov["scenario_id"] for _r, cov in cal}
    dev_ids = {cov["scenario_id"] for _r, cov in dev}
    assert cal_ids.isdisjoint(dev_ids)


# ---------------------------------------------------------------------------
# Deterministic, macro-family-weighted bootstrap arithmetic (Section 13).
# ---------------------------------------------------------------------------


def test_macro_family_bootstrap_is_deterministic_given_fixed_seed():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_4_decide as decide  # noqa: E402

    rows = []
    for family, base in (("coastal-branch", 0.5), ("tree-branch", 0.6), ("dense-loop", 0.4)):
        for predictor_seed in m4.SEEDS:
            for generator_seed in range(3):
                for arm, bump in (("ARM_A", 0.0), ("ARM_B2", 0.05)):
                    rows.append({
                        "arm": arm, "family": family, "predictor_seed": predictor_seed, "generator_seed": generator_seed,
                        "depth_bucket": "MATURE", "metrics_neural": {"top1": base + bump},
                    })
    result_1 = decide._macro_family_bootstrap(rows, ("coastal-branch", "tree-branch", "dense-loop"), "MATURE", decide._top1_fn)
    result_2 = decide._macro_family_bootstrap(rows, ("coastal-branch", "tree-branch", "dense-loop"), "MATURE", decide._top1_fn)
    assert result_1["ci_lower"] == result_2["ci_lower"]
    assert result_1["ci_upper"] == result_2["ci_upper"]
    assert result_1["observed_macro_delta"] == pytest.approx(0.05, abs=1e-9)


def test_macro_family_bootstrap_resamples_incidents_not_depth_rows():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_m9_4_decide as decide  # noqa: E402

    # Two depths for the same incident must collapse to ONE resampling unit
    # (mean across depths within the maturity bucket), not two independent rows.
    rows = []
    for depth in (12, 25):
        rows.append({
            "arm": "ARM_A", "family": "coastal-branch", "predictor_seed": 20260814, "generator_seed": 1,
            "depth_bucket": "MATURE", "depth": depth, "metrics_neural": {"top1": 0.5},
        })
        rows.append({
            "arm": "ARM_B2", "family": "coastal-branch", "predictor_seed": 20260814, "generator_seed": 1,
            "depth_bucket": "MATURE", "depth": depth, "metrics_neural": {"top1": 0.75},
        })
    values = decide._incident_values(rows, "ARM_A", "coastal-branch", "MATURE", decide._top1_fn)
    assert values == {(20260814, 1): 0.5}


# ---------------------------------------------------------------------------
# Checkpoint identity / no-mutation.
# ---------------------------------------------------------------------------


def test_frozen_checkpoints_match_recorded_provenance_hashes():
    for seed in m4.SEEDS:
        record = json.loads((m4.RUNS_M8_7 / f"AGE_FIX_ONLY-seed{seed}.json").read_text())
        path = Path(record["export_path"])
        assert path.exists(), f"ARM_A seed{seed} checkpoint missing on disk"
        assert _sha256_file(path) == record["checkpoint_sha256"]
    for seed in m4.SEEDS:
        record = json.loads((m4.RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
        ts = record["training_summary"]
        path = Path(ts["export_path"])
        assert path.exists(), f"ARM_B2 seed{seed} checkpoint missing on disk"
        assert _sha256_file(path) == ts["export_sha256"]


@needs_artifacts
def test_manifest_records_checkpoint_hashes_unchanged_before_and_after_inference():
    manifest = json.loads(m4.M9_4_MANIFEST_PATH.read_text())
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m4.SEEDS:
            entry = manifest["checkpoint_identities"][arm][str(seed)]
            assert entry["sha256_before"] == entry["sha256_after"], f"{arm} seed{seed} checkpoint mutated during M9.4"


def test_source_representative_module_never_calls_backward_or_optimizer():
    source = (SCRIPTS_DIR / "run_m9_4_source_representative.py").read_text()
    for forbidden in (".backward(", "optimizer.step(", "Trainer(", ".fit(resume_from"):
        assert forbidden not in source, f"forbidden training/backward invocation found: {forbidden!r}"
    assert "model.eval()" in Path(str(ROOT / "scripts" / "hydrocore_v5" / "run_m9_0a_evaluate.py")).read_text()


def test_source_representative_module_never_opens_locked_test():
    source = (SCRIPTS_DIR / "run_m9_4_source_representative.py").read_text()
    assert "locked_final_test" not in source
    assert "locked_topology_test" not in source
    assert "locked_test_opened" in source  # asserts closed, never opens it


# ---------------------------------------------------------------------------
# Historical artifact immutability.
# ---------------------------------------------------------------------------


def test_historical_m7_m9_generators_untouched_in_place():
    import subprocess

    for path in (
        "scripts/hydrocore_v5/run_m7_topology.py",
        "scripts/hydrocore_v5/run_m9_0a_evaluate.py",
        "scripts/hydrocore_v5/run_m9_0a_decide.py",
        "scripts/hydrocore_v5/m9_3_common.py",
    ):
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", path], cwd=ROOT, capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "", f"{path} was modified relative to HEAD: {result.stdout}"


# ---------------------------------------------------------------------------
# Artifact-dependent contract tests (require the M9.4 pipeline to have run).
# ---------------------------------------------------------------------------


@needs_artifacts
def test_representativeness_audit_covers_every_family_source():
    audit = json.loads(m4.M9_4_REPRESENTATIVENESS_AUDIT_PATH.read_text())
    for family in m4.ALL_FAMILIES:
        checks = audit["families"][family]["checks"]
        assert checks["all_sources_in_calibration"], f"{family}: not every source present in calibration_m9_4"
        assert checks["all_sources_in_development"], f"{family}: not every source present in development_m9_4"
        assert checks["no_zero_support_source"]
        assert checks["seed_disjoint_calibration_vs_development"]
        assert checks["no_scenario_id_overlap"]


@needs_artifacts
def test_source_policy_reports_newly_included_sources_for_multi_source_families():
    policy = json.loads(m4.M9_4_SOURCE_POLICY_PATH.read_text())
    for family in ("branched-loop", "loop-grid", "coastal-branch", "tree-branch", "dense-loop"):
        entry = policy["families"][family]
        assert len(entry["newly_included_source_set"]) > 0, f"{family}: expected newly-included sources beyond the legacy 4-source subset"
        assert entry["n_sources"] == len(entry["legacy_included_source_set"]) + len(entry["newly_included_source_set"])


@needs_artifacts
def test_predictions_pairing_key_consistent_across_arms():
    seen_incidents: dict[str, set[str]] = {}
    with m4.M9_4_PREDICTIONS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            key = f"{row['family']}|{row['source_node']}|{row['generator_seed']}|{row['depth']}"
            seen_incidents.setdefault(row["arm"], set()).add(key)
    # Every ARM_A-evaluated key must also appear for ARM_B2 (ARM_B2 evaluates a superset of families).
    assert seen_incidents["ARM_A"].issubset(seen_incidents["ARM_B2"])


@needs_artifacts
def test_predictions_no_cross_role_row_leakage():
    """Every predictions.jsonl row's generator_seed must fall in that
    family's development_m9_4 seed range, never calibration_m9_4's."""

    with m4.M9_4_PREDICTIONS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            family = row["family"]
            junctions = m4.full_junction_list(family, m4.ALL_FAMILY_LOADERS[family])
            dev_base = m4.m9_4_seed_base(family, "development_m9_4")
            dev_max = dev_base + (len(junctions) - 1) * 1_000 + (m4.REPEATS_PER_SOURCE - 1)
            cal_base = m4.m9_4_seed_base(family, "calibration_m9_4")
            cal_max = cal_base + (len(junctions) - 1) * 1_000 + (m4.REPEATS_PER_SOURCE - 1)
            seed = row["generator_seed"]
            assert dev_base <= seed <= dev_max, f"row seed {seed} outside development_m9_4 range for {family}"
            assert not (cal_base <= seed <= cal_max), f"row seed {seed} leaked from calibration_m9_4 range for {family}"


@needs_artifacts
def test_legacy_reproduction_artifact_has_required_fields():
    legacy = json.loads(m4.M9_4_LEGACY_REPRODUCTION_PATH.read_text())
    assert legacy["M9_4_LEGACY_REPRODUCTION"] in ("PASS", "FAIL")
    assert "checks" in legacy and "reproduced" in legacy and "legacy_recorded" in legacy


@needs_artifacts
def test_closure_alpha_and_coverage_floor_never_weakened():
    closure = json.loads(m4.M9_4_CLOSURE_PATH.read_text())
    assert closure["calibration_gate"]["alpha"] == 0.1
    assert closure["calibration_gate"]["coverage_floor"] == 0.85
    assert closure["locked_test_opened_before"] is False
    assert closure["locked_test_opened_after"] is False
    assert closure["no_training_performed"] is True
    assert closure["no_predictor_modified"] is True
    assert closure["M9_4_DECISION"] in ("A", "B", "C", "D", "E", "F")
