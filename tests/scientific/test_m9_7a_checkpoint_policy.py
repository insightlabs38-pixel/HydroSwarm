"""Milestone 9.7A: checkpoint-selection consistency correction.

Proves, BEFORE any M9.8 execution and without editing any M9.6/M9.7
historical artifact:

1. M9.6's own canonical checkpoint policy is FINAL_STEP_1350 (not
   best-validation), for every real M9.6 training-run record.
2. Best-validation and canonical checkpoints genuinely diverge for at
   least one real, already-trained seed -- this is not a hypothetical
   risk.
3. M9.6's own evaluation script (`run_m9_6_evaluate._canonical_model`)
   loads the canonical checkpoint, never best-validation, for the numbers
   its frozen M9_6_DECISION=A actually rests on.
4. The M9.7A amendment (`m9-7a-amendment.json`) freezes FINAL_STEP_1350
   for BOTH M9.8 arms and marks best-validation diagnostic-only.
5. Every other M9.8-frozen parameter (architecture, seeds, step budget,
   primary endpoint, bootstrap procedure, practical-effect threshold,
   calibration policy, guardrails, HydroCore-L authorization) is
   unchanged from the original M9.7 freeze.
6. Every M9.7 artifact is byte-identical to its M9.7-closure snapshot
   (immutability).
7. locked_final_test/locked_topology_test remain unopened.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from hydroswarm.evaluation.live_robustness import locked_test_opened

ROOT = Path(__file__).resolve().parents[2]
M9_6_TRAINING_RUNS_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-6" / "m9-6-training-runs"
M9_7_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-7"
M9_7A_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-7a"

M9_6_RECORD_FILES = sorted(M9_6_TRAINING_RUNS_DIR.glob("*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_locked_test_unopened_before_module() -> None:
    assert locked_test_opened(ROOT) is False


def test_m9_6_training_runs_exist() -> None:
    # M9.7A's core evidence depends on these already-committed M9.6
    # artifacts -- fail loudly (not silently skip) if the historical
    # record has somehow gone missing.
    assert len(M9_6_RECORD_FILES) == 6
    names = {p.name for p in M9_6_RECORD_FILES}
    for arm in ("ARM_A_M9_6", "ARM_B_M9_6"):
        for seed in (20260814, 31874, 20260815):
            assert f"{arm}-seed{seed}.json" in names


@pytest.mark.parametrize("record_path", M9_6_RECORD_FILES, ids=lambda p: p.name)
def test_m9_6_canonical_checkpoint_policy_is_final_step_1350(record_path: Path) -> None:
    record = _load(record_path)
    assert record["canonical_checkpoint_policy"] == "FINAL_STEP_1350"
    assert record["canonical_global_step"] == 1350
    assert record["canonical_epoch"] == 19  # epoch 20 (0-indexed 19), matching config.epochs - 1
    assert record["canonical_export_path"]
    assert record["canonical_export_sha256"]


def test_best_validation_checkpoint_can_diverge_from_canonical() -> None:
    """Not hypothetical: a real, already-committed M9.6 seed genuinely
    diverges between best-validation and canonical."""
    record = _load(M9_6_TRAINING_RUNS_DIR / "ARM_B_M9_6-seed20260814.json")
    assert record["best_epoch"] != record["canonical_epoch"]
    assert record["best_validation_export_path"] != record["canonical_export_path"]
    assert record["best_validation_export_sha256"] != record["canonical_export_sha256"]


def test_m9_6_evaluation_used_canonical_checkpoint_not_best_validation() -> None:
    """Structural proof, from source: run_m9_6_evaluate._canonical_model
    reads record["canonical_export_path"], never
    record["best_validation_export_path"]."""
    sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5"))
    import run_m9_6_evaluate as m9_6_evaluate  # noqa: PLC0415

    import inspect

    source = inspect.getsource(m9_6_evaluate._canonical_model)
    assert 'record["canonical_export_path"]' in source
    assert "best_validation_export_path" not in source

    # And empirically: the loaded model's checkpoint SHA matches the
    # canonical record, for the one seed where canonical != best-validation.
    _model, sha, record = m9_6_evaluate._canonical_model("ARM_B_M9_6", 20260814)
    assert sha == record["canonical_export_sha256"]
    assert sha != record["best_validation_export_sha256"]


def test_m9_7a_amendment_exists_and_is_well_formed() -> None:
    amendment = _load(M9_7A_DIR / "m9-7a-amendment.json")
    assert amendment["kind"] == "M9_7A_PROTOCOL_CONSISTENCY_AMENDMENT"
    assert amendment["milestone"] == "M9.7A"
    assert amendment["locked_test_opened_before"] is False
    assert amendment["locked_test_opened_after"] is False


def test_m9_7a_freezes_final_step_1350_for_both_arms() -> None:
    amendment = _load(M9_7A_DIR / "m9-7a-amendment.json")
    policy = amendment["corrected_m9_8_checkpoint_policy"]
    assert "FINAL_STEP_1350" in policy["ARM_S"] or "canonical" in policy["ARM_S"].lower()
    assert "1350" in policy["ARM_M"]
    assert "canonical" in policy["ARM_M"].lower()
    assert "MUST NOT" in policy["best_validation_checkpoints"]
    assert "diagnostic" in policy["best_validation_checkpoints"].lower()


def test_m9_7a_does_not_alter_hydrocore_m_architecture() -> None:
    amendment = _load(M9_7A_DIR / "m9-7a-amendment.json")
    selected = _load(M9_7_DIR / "m9-7-selected-m-architecture.json")
    assert selected["total_parameters"] == 13_919_572
    assert "13,919,572" in amendment["unchanged_and_reverified"]["hydrocore_m_architecture"]
    assert "d_model=352" in amendment["unchanged_and_reverified"]["hydrocore_m_architecture"]


def test_m9_7a_does_not_alter_seeds_step_budget_or_l_authorization() -> None:
    amendment = _load(M9_7A_DIR / "m9-7a-amendment.json")
    unchanged = amendment["unchanged_and_reverified"]
    assert unchanged["seeds"] == [20260814, 31874, 20260815]
    assert unchanged["optimizer_step_budget"] == {"S": 1350, "M": 1350}
    assert unchanged["hydrocore_l_authorization"] is False


def test_m9_7a_does_not_alter_primary_endpoint_bootstrap_or_threshold() -> None:
    amendment = _load(M9_7A_DIR / "m9-7a-amendment.json")
    unchanged = amendment["unchanged_and_reverified"]

    prereg = _load(M9_7_DIR / "m9-7-m9-8-preregistration.json")
    assert prereg["statistical_procedure"]["bootstrap_seed"] == 20260819
    assert prereg["statistical_procedure"]["resamples"] == 2000
    assert prereg["statistical_procedure"]["confidence_interval"] == 0.90
    assert prereg["practical_effect_threshold"]["rules"]["A_primary_effect"]["condition"].startswith(
        "M - S >= +0.02 absolute"
    )
    assert "20260819" in unchanged["bootstrap_procedure"]
    assert "+0.02" in unchanged["practical_effect_threshold"]
    assert "0.1" in unchanged["calibration_policy"]


def test_m9_7_historical_artifacts_are_byte_identical_to_snapshot() -> None:
    """The amendment must be additive-only: every M9.7 report artifact's
    SHA-256 must still match the snapshot m9-7a-amendment.json recorded at
    freeze time -- proving M9.7A never silently edited a closed milestone's
    evidence."""
    amendment = _load(M9_7A_DIR / "m9-7a-amendment.json")
    snapshot = amendment["m9_7_artifacts_hash_snapshot"]
    assert len(snapshot) >= 12
    for name, expected_sha in snapshot.items():
        path = M9_7_DIR / name
        assert path.exists(), f"{name} referenced in snapshot but missing on disk"
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"{name} was modified since M9.7 closure"


def test_m9_7a_closure_decision() -> None:
    closure = _load(M9_7A_DIR / "m9-7a-closure.json")
    assert closure["M9_7A_DECISION"] == "CHECKPOINT_POLICY_CONSISTENCY_FIXED"
    assert closure["M9_8_CAPACITY_EXPERIMENT_READY"] is True
    assert closure["locked_test_opened_before"] is False
    assert closure["locked_test_opened_after"] is False


def test_locked_test_unopened_after_module() -> None:
    assert locked_test_opened(ROOT) is False
