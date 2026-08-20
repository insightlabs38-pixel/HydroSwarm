from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hydrocore_v5"))

import run_m11_2_finalist_freeze as m11  # noqa: E402


def test_m11_2_parent_selection_is_exact_and_unfrozen() -> None:
    parent = m11.parent_selection_verification()
    assert parent["all_checks_pass"] is True
    assert all(parent["checks"].values())


def test_m11_2_finalist_identity_binds_release_and_authority() -> None:
    identity = m11.finalist_identity()
    assert identity["assets"]["checkpoint"]["sha256"] == m11.EXPECTED["checkpoint"]
    assert identity["assets"]["calibration"]["sha256"] == m11.EXPECTED["calibration"]
    assert identity["assets"]["release_manifest"]["sha256"] == m11.EXPECTED["manifest"]
    assert identity["runtime_enabled_outputs"] == m11.OUTPUTS
    assert identity["deterministic_authority"] == m11.AUTHORITY
    assert not m11.identity_violations(identity)


def test_m11_2_identity_drift_is_rejected_for_every_governed_mutation() -> None:
    negative = m11.negative_identity_tests(m11.finalist_identity())
    assert negative["all_mutations_rejected"] is True
    assert all(not result["accepted_as_same_finalist"] for result in negative["tests"].values())


def test_m11_2_manual_output_and_authority_mutations_are_not_accepted() -> None:
    identity = m11.finalist_identity()
    changed = copy.deepcopy(identity)
    changed["runtime_enabled_outputs"].append("next_step")
    changed["deterministic_authority"]["human_approval_required"] = False
    changed["deterministic_authority"]["autonomous_actuation"] = True
    violations = m11.identity_violations(changed)
    assert "runtime_enabled_outputs" in violations
    assert "deterministic_authority" in violations


def test_m11_2_clean_load_reproducibility() -> None:
    result = m11.clean_load_reproducibility()
    assert result["all_checks_pass"] is True
    assert all(result["checks"].values())


def test_m11_2_historical_immutability_allows_additive_m11_5_evidence() -> None:
    history = m11.historical_immutability([
        ("A", "docs/evaluation/HYDROCORE_V5_M11_5_FULL_VALIDATION_RESULTS.md"),
        ("A", "scripts/hydrocore_v5/run_m11_5_full_validation.py"),
        ("A", "tests/scientific/test_m11_5_full_validation.py"),
        ("A", "reports/evaluation/hydrocore-v5/m11/m11-5/m11-5-matrix.json"),
    ])
    assert history["historical_artifacts_unchanged"] is True
    assert history["no_system_tuning_or_runtime_change"] is True
    assert len(history["later_milestone_additions"]) == 4


def test_m11_2_historical_immutability_rejects_protected_finalist_changes() -> None:
    history = m11.historical_immutability([
        ("M", "models/hydrocore-v5-release/runtime_manifest.json"),
        ("M", "src/hydroswarm/runtime/v5_defaults.py"),
        ("M", "reports/evaluation/hydrocore-v5/m10/m10-5/m10-5-closure.json"),
        ("M", "reports/evaluation/hydrocore-v5/m11/m11-1/final-selection.json"),
    ])
    assert history["historical_artifacts_unchanged"] is False
    assert history["no_system_tuning_or_runtime_change"] is False
    assert history["protected_path_violations"] == [path for _, path in [
        ("M", "models/hydrocore-v5-release/runtime_manifest.json"),
        ("M", "src/hydroswarm/runtime/v5_defaults.py"),
        ("M", "reports/evaluation/hydrocore-v5/m10/m10-5/m10-5-closure.json"),
        ("M", "reports/evaluation/hydrocore-v5/m11/m11-1/final-selection.json"),
    ]]


def test_m11_2_artifacts_freeze_without_authorizing_locked_evaluation(tmp_path: Path) -> None:
    output_dir = tmp_path / "m11-2"
    records = m11.build_artifacts(output_dir)
    certificate = records["finalist-freeze.json"]
    assert certificate["finalist_selected"] is True
    assert certificate["finalist_frozen"] is True
    assert certificate["tuning_closed"] is True
    assert certificate["locked_evaluation_authorized"] is False
    assert certificate["finalist_identity_manifest_path"] == "m11-2-finalist-identity.json"
    assert records["m11-2-closure.json"]["closure_state"] == "M11_2_FINALIST_FROZEN"
    assert (tmp_path / "m11-current-status.json").is_file()
