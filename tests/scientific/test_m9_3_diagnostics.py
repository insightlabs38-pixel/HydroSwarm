"""Contract tests for the Milestone 9.3 calibration root-cause diagnostic
study (`scripts/hydrocore_v5/m9_3_common.py`, `m9_3_analysis_lib.py`,
`run_m9_3_build_table.py`, `run_m9_3_analyze.py`).

M9.3 is DIAGNOSTIC / ANALYSIS-ONLY. These tests cover governance/statistical
machinery correctness (checkpoint identity, alpha frozen at 0.1, finite-
sample quantile formula, fallback hierarchy, maturity bucket mapping, no
locked-test access, canonical row pairing, deterministic bootstrap, raw-
artifact immutability) -- never a promotion decision.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from tests.historical_artifact_portability import require_historical_artifact

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT / "src"))

import m9_3_analysis_lib as lib  # noqa: E402
import m9_3_common as m93  # noqa: E402
from hydroswarm.calibration.conformal import SplitConformalCalibrator, CalibrationExample, _quantile  # noqa: E402

CANONICAL_PATH = m93.M9_3_CANONICAL_PATH
MANIFEST_PATH = m93.M9_3_MANIFEST_PATH
REPRODUCTION_PATH = m93.M9_3_REPRODUCTION_PATH


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Alpha frozen at 0.1 everywhere.
# ---------------------------------------------------------------------------


def test_alpha_frozen_at_0_1():
    assert m93.ALPHA == 0.1
    assert m93.OPERATIONAL_COVERAGE_FLOOR == 0.85
    assert m93.NOMINAL_COVERAGE_TARGET == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Finite-sample quantile calculation (Section 21 audit target).
# ---------------------------------------------------------------------------


def test_finite_sample_quantile_formula_matches_hand_computation():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # n=10
    # rank = ceil((10+1)*0.9) = ceil(9.9) = 10 -> the max value.
    assert _quantile(scores, 0.1) == 1.0


def test_finite_sample_quantile_rank_never_exceeds_n():
    scores = [0.5]
    assert _quantile(scores, 0.1) == 0.5


def test_finite_sample_resolution_formula():
    assert m93.finite_sample_resolution(24) == pytest.approx(1 / 25)
    assert m93.finite_sample_resolution(0) != m93.finite_sample_resolution(0)  # nan


# ---------------------------------------------------------------------------
# Fallback hierarchy: NETWORK_SPECIFIC -> CONDITION_SPECIFIC -> GLOBAL.
# ---------------------------------------------------------------------------


def test_selection_fallback_hierarchy():
    examples = [CalibrationExample(probabilities=(0.9, 0.1), true_index=0, condition="CLEAN", network_id="fam:EARLY") for _ in range(15)]
    examples += [CalibrationExample(probabilities=(0.8, 0.2), true_index=0, condition="CLEAN", network_id="other:EARLY") for _ in range(15)]
    calibrator = SplitConformalCalibrator.fit(examples, alpha=0.1, model_hash="test", feature_schema_hash="n/a", dataset_manifest_hash="test", minimum_group_size=10)
    source, group, _scores = calibrator.selection(condition="CLEAN", network_id="fam:EARLY")
    assert source == "NETWORK_SPECIFIC"
    assert group == "fam:EARLY"
    source2, group2, _ = calibrator.selection(condition="CLEAN", network_id="unknown:EARLY")
    assert source2 == "CONDITION_SPECIFIC"
    assert group2 == "CLEAN"
    source3, group3, _ = calibrator.selection(condition="UNKNOWN_CONDITION", network_id="unknown:EARLY")
    assert source3 == "GLOBAL"
    assert group3 == "global"


def test_minimum_group_size_excludes_small_groups():
    examples = [CalibrationExample(probabilities=(0.9, 0.1), true_index=0, condition="CLEAN", network_id="tiny:EARLY") for _ in range(5)]
    calibrator = SplitConformalCalibrator.fit(examples, alpha=0.1, model_hash="test", feature_schema_hash="n/a", dataset_manifest_hash="test", minimum_group_size=10)
    assert "tiny:EARLY" not in calibrator.artifact.network_scores
    source, _group, _scores = calibrator.selection(condition="CLEAN", network_id="tiny:EARLY")
    assert source != "NETWORK_SPECIFIC"


# ---------------------------------------------------------------------------
# Maturity bucket mapping consistency.
# ---------------------------------------------------------------------------


def test_maturity_buckets_match_frozen_depth_grid():
    assert m93.EARLY_DEPTHS == (1, 2, 3)
    assert m93.MID_DEPTHS == (4, 6)
    assert m93.MATURE_DEPTHS == (12, 25)
    assert m93.DEPTHS == (1, 2, 3, 4, 6, 12, 25)


# ---------------------------------------------------------------------------
# No locked-test access.
# ---------------------------------------------------------------------------


LOCKED_TOKENS = ("locked_final_test", "locked_topology_test")


@pytest.mark.parametrize("script", ["m9_3_common.py", "m9_3_analysis_lib.py", "run_m9_3_build_table.py", "run_m9_3_analyze.py"])
def test_no_locked_test_path_construction(script):
    source = (SCRIPTS_DIR / script).read_text()
    path_construction_tokens = ("Path(", "open(", "/ \"", "/ '", "read_text", "read_bytes")
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("#", '"""', "'''", "*")):
            continue
        for token in LOCKED_TOKENS:
            if token in line and any(pc in line for pc in path_construction_tokens):
                pytest.fail(f"{script}:{lineno} constructs a path to {token}: {line!r}")


def test_locked_test_still_closed_right_now():
    assert m93.assert_locked_test_closed() is False


# ---------------------------------------------------------------------------
# No training code invoked by M9.3 scripts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", ["run_m9_3_build_table.py", "run_m9_3_analyze.py"])
def test_no_training_call_in_m9_3_scripts(script):
    source = (SCRIPTS_DIR / script).read_text()
    forbidden = ("optimizer.step", ".backward(", "Trainer(", "loss.backward", "requires_grad_(True")
    for token in forbidden:
        assert token not in source, f"{script} must never invoke training machinery ({token!r} found)"


# ---------------------------------------------------------------------------
# Deterministic bootstrap / learning-curve seeds, predeclared.
# ---------------------------------------------------------------------------


def test_bootstrap_and_learning_curve_seeds_predeclared():
    assert m93.M9_3_BOOTSTRAP_SEED == 20260816
    assert m93.LEARNING_CURVE_SEED == 20260816
    assert m93.M9_3_BOOTSTRAP_RESAMPLES == 2000
    assert m93.LEARNING_CURVE_FRACTIONS == (0.25, 0.50, 0.75, 1.0)


def test_quantile_stability_groups_by_incident_not_row():
    rows = []
    for arm in ("ARM_B2",):
        for incident in (1, 2, 3):
            for depth in (1, 2):
                rows.append({
                    "predictor_arm": arm, "training_seed": 20260814, "topology_family": "golden-reference",
                    "depth_bucket": "EARLY", "split": "calibration", "incident_id": incident, "nonconformity_score": 0.1 * incident,
                })
        for depth in (1,):
            for incident in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
                rows.append({
                    "predictor_arm": arm, "training_seed": 20260814, "topology_family": "golden-reference",
                    "depth_bucket": "EARLY", "split": "development", "incident_id": incident + 100, "nonconformity_score": 0.05,
                })
    df = pd.DataFrame(rows)
    result = lib.quantile_stability(df)
    entry = result["ARM_B2"]["20260814"]["golden-reference"]["EARLY"]
    assert entry["n_incidents"] == 3  # 3 distinct incidents, not 6 rows
    assert entry["resamples"] == 2000
    assert entry["bootstrap_seed"] == 20260816


# ---------------------------------------------------------------------------
# Development-oracle result must never enter a "selected"/promoted output.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", ["m9_3_common.py", "m9_3_analysis_lib.py", "run_m9_3_build_table.py", "run_m9_3_analyze.py"])
def test_no_promotion_or_selection_logic_present(script):
    source = (SCRIPTS_DIR / script).read_text()
    forbidden = ("SELECTED_SCHEME =", "PROMOTED", "promote_interleaved", "M9_3_SELECTED_CALIBRATION")
    for token in forbidden:
        assert token not in source


def test_counterfactual_oracle_label_present_in_lib():
    source = (SCRIPTS_DIR / "m9_3_analysis_lib.py").read_text()
    assert "DEVELOPMENT_ORACLE_NOT_VALID_FOR_DEPLOYMENT" in source


# ---------------------------------------------------------------------------
# Canonical table integrity (requires the table to already have been built).
# ---------------------------------------------------------------------------


canonical_missing = not CANONICAL_PATH.exists()


@pytest.mark.skipif(canonical_missing, reason="canonical table not built yet")
def test_canonical_table_family_labels_are_known_set():
    df = pd.read_json(CANONICAL_PATH, lines=True)
    valid_families = set(m93.KNOWN_FAMILIES) | set(m93.UNSEEN_FAMILIES)
    assert set(df["topology_family"].unique().tolist()) <= valid_families


@pytest.mark.skipif(canonical_missing, reason="canonical table not built yet")
def test_canonical_table_arm_a_only_has_golden_reference_as_known():
    df = pd.read_json(CANONICAL_PATH, lines=True)
    arm_a_known = df[(df.predictor_arm == "ARM_A") & (df.known_family)]
    assert set(arm_a_known["topology_family"].unique().tolist()) == {"golden-reference"}


@pytest.mark.skipif(canonical_missing, reason="canonical table not built yet")
def test_canonical_table_depths_and_seeds_match_frozen_grid():
    df = pd.read_json(CANONICAL_PATH, lines=True)
    assert set(df["prefix_depth"].unique().tolist()) <= set(m93.DEPTHS)
    assert set(df["training_seed"].unique().tolist()) == set(m93.SEEDS)


manifest_missing = not MANIFEST_PATH.exists()


@pytest.mark.skipif(manifest_missing, reason="manifest not built yet")
def test_checkpoint_sha256_recorded_for_all_arm_seed_combinations():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m93.SEEDS:
            assert str(seed) in manifest["checkpoint_sha256"][arm]
            assert len(manifest["checkpoint_sha256"][arm][str(seed)]["sha256"]) == 64


@pytest.mark.skipif(manifest_missing, reason="manifest not built yet")
def test_predictor_checkpoints_unchanged_after_diagnostics():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for arm in ("ARM_A", "ARM_B2"):
        for seed in m93.SEEDS:
            entry = manifest["checkpoint_sha256"][arm][str(seed)]
            on_disk = _sha256_file(require_historical_artifact(entry["export_path"], entry["sha256"], repo_root=ROOT))
            assert on_disk == entry["sha256"], f"{arm} seed{seed} checkpoint changed during/after M9.3 diagnostics"


@pytest.mark.skipif(manifest_missing, reason="manifest not built yet")
def test_manifest_records_lock_state_and_no_modification_flags():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["locked_test_opened_before"] is False
    assert manifest["locked_test_opened_after"] is False
    assert manifest["no_training_performed"] is True
    assert manifest["no_predictor_modified"] is True
    assert manifest["alpha"] == 0.1


reproduction_missing = not REPRODUCTION_PATH.exists()


@pytest.mark.skipif(reproduction_missing, reason="reproduction gate not run yet")
def test_reproduction_gate_status_recorded():
    reproduction = json.loads(REPRODUCTION_PATH.read_text())
    assert reproduction["status"] in ("EXACT_OR_WITHIN_DECLARED_FLOAT_TOLERANCE", "REPRODUCTION_FAILED")
    assert "M9_0A_REPRODUCTION" in reproduction
    assert "M9_0B_REPRODUCTION" in reproduction


@pytest.mark.skipif(manifest_missing, reason="manifest not built yet")
def test_raw_m9_0a_m9_0b_artifacts_never_overwritten():
    for path in (m93.M9_0A_RESULTS_PATH, m93.M9_0A_CALIBRATION_PATH, m93.M9_0A_TOPOLOGY_PATH, m93.M9_0B_RESULTS_PATH, m93.M9_0B_CAL_BY_SEED_PATH):
        assert path.exists()
    # Immutability is enforced structurally: M9.3 scripts only ever `.read_text()`
    # these paths (see build script), never `.write_text()` -- verified by grep.
    build_source = (SCRIPTS_DIR / "run_m9_3_build_table.py").read_text()
    for path in (m93.M9_0A_RESULTS_PATH, m93.M9_0A_CALIBRATION_PATH, m93.M9_0A_TOPOLOGY_PATH):
        var_names = [name for name in dir(m93) if getattr(m93, name, None) == path]
        for var in var_names:
            assert f"{var}.write_text" not in build_source
