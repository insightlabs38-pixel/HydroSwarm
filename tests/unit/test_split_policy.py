from __future__ import annotations

import pytest

from hydroswarm.training import SplitPolicyViolation, authorize_locked_final_test, load_policy
from hydroswarm.training.split_policy import DEFAULT_POLICY_PATH


def test_default_policy_file_loads_and_covers_every_split_role() -> None:
    policy = load_policy()
    assert policy["schema_version"] == "hydroswarm-split-policy-v3"
    expected_roles = {
        "train",
        "validation",
        "calibration",
        "development_holdout",
        "ood_development",
        "locked_final_test",
        "locked_topology_test",
    }
    assert expected_roles.issubset(policy["split_roles"])


def test_default_policy_path_exists_in_repo() -> None:
    # Sanity check the module constant actually points at the committed file
    # (relative to repo root, as scripts are invoked from there).
    assert DEFAULT_POLICY_PATH.name == "evaluation_policy_v3.json"


def test_authorize_rejects_when_final_selection_missing(tmp_path) -> None:
    with pytest.raises(SplitPolicyViolation, match="does not exist"):
        authorize_locked_final_test(
            final_selection_path=tmp_path / "final-selection.json",
            registry_has_selected_configuration=True,
            all_required_tests_pass=True,
            manifest_hashes_match=True,
            calibration_fit_without_test_data=True,
            further_tuning_planned=False,
        )


def test_authorize_rejects_when_further_tuning_planned(tmp_path) -> None:
    selection = tmp_path / "final-selection.json"
    selection.write_text("{}", encoding="utf-8")
    with pytest.raises(SplitPolicyViolation, match="further tuning"):
        authorize_locked_final_test(
            final_selection_path=selection,
            registry_has_selected_configuration=True,
            all_required_tests_pass=True,
            manifest_hashes_match=True,
            calibration_fit_without_test_data=True,
            further_tuning_planned=True,
        )


def test_authorize_rejects_when_any_precondition_fails(tmp_path) -> None:
    selection = tmp_path / "final-selection.json"
    selection.write_text("{}", encoding="utf-8")
    with pytest.raises(SplitPolicyViolation, match="manifest_hashes_match"):
        authorize_locked_final_test(
            final_selection_path=selection,
            registry_has_selected_configuration=True,
            all_required_tests_pass=True,
            manifest_hashes_match=False,
            calibration_fit_without_test_data=True,
            further_tuning_planned=False,
        )


def test_authorize_succeeds_when_every_precondition_holds(tmp_path) -> None:
    selection = tmp_path / "final-selection.json"
    selection.write_text("{}", encoding="utf-8")
    authorize_locked_final_test(
        final_selection_path=selection,
        registry_has_selected_configuration=True,
        all_required_tests_pass=True,
        manifest_hashes_match=True,
        calibration_fit_without_test_data=True,
        further_tuning_planned=False,
    )  # must not raise
