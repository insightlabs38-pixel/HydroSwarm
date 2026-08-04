"""Load and enforce the governed split/final-test policy (Task 0.8).

This is the machine-readable half of configs/evaluation_policy_v3.json: a
small guard callers use before opening the locked final test, rather than
relying on every script to reimplement the precondition list correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_POLICY_PATH = Path("configs/evaluation_policy_v3.json")


class SplitPolicyViolation(Exception):
    """Raised when an action would violate the governed split policy."""


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def authorize_locked_final_test(
    *,
    final_selection_path: str | Path,
    registry_has_selected_configuration: bool,
    all_required_tests_pass: bool,
    manifest_hashes_match: bool,
    calibration_fit_without_test_data: bool,
    further_tuning_planned: bool,
) -> None:
    """Raise SplitPolicyViolation unless every Stage 6 precondition holds.

    Deliberately takes the preconditions as explicit booleans rather than
    trying to compute them itself: what counts as "all required tests pass"
    or "manifest hashes match" is decided by the caller's own governed
    tooling (pytest/ruff/pyright results, ExperimentRegistry hashes), and
    this function's only job is to refuse to proceed unless every one of
    them was already verified true.
    """

    if not Path(final_selection_path).exists():
        raise SplitPolicyViolation(
            f"{final_selection_path} does not exist; the locked final test may not be opened "
            "before a final-selection record is written"
        )
    if further_tuning_planned:
        raise SplitPolicyViolation(
            "further tuning is planned; the locked final test may only be opened once no more "
            "architecture or hyperparameter tuning is planned"
        )
    checks = {
        "registry_has_selected_configuration": registry_has_selected_configuration,
        "all_required_tests_pass": all_required_tests_pass,
        "manifest_hashes_match": manifest_hashes_match,
        "calibration_fit_without_test_data": calibration_fit_without_test_data,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SplitPolicyViolation(
            f"locked final test is not authorized; failing preconditions: {failed}"
        )
