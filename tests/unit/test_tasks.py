from __future__ import annotations

import pytest

from hydroswarm.tasks import CORPUS_SUPERVISED_TASKS, RUNTIME_TASKS, validate_tasks


def test_corpus_supervised_tasks_is_a_subset_of_runtime_tasks() -> None:
    assert CORPUS_SUPERVISED_TASKS <= RUNTIME_TASKS


def test_validate_tasks_accepts_known_tasks() -> None:
    validate_tasks(frozenset({"sentinel", "scout"}), label="test")


def test_validate_tasks_rejects_unknown_tasks() -> None:
    with pytest.raises(ValueError, match="unknown runtime tasks"):
        validate_tasks(frozenset({"sentinel", "made-up-task"}), label="test")
