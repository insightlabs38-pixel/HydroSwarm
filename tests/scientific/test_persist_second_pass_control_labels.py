"""core-issues4.txt Section F: streaming per-row persistence of
SecondPassControlLabel rows."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import persist_second_pass_control_labels  # noqa: E402

from hydroswarm.training.second_pass_control_labels import (  # noqa: E402
    SecondPassControlLabel,
    second_pass_control_policy_hash,
)
from hydroswarm.training.targets_v2 import NextStep  # noqa: E402


def _label(**overrides) -> SecondPassControlLabel:
    base = dict(
        scenario_id="s1",
        calibrated_candidate_set_size=2,
        candidate_covered=True,
        posterior_entropy_bits=0.5,
        classical_neural_disagreement_js=0.1,
        calibration_valid=True,
        evidence_sufficiency=True,
        next_step=NextStep.GENERATE_PLANS,
        teacher_checkpoint_hash="abc123",
        network_id="golden-reference",
        topology_hash="topo-hash",
    )
    base.update(overrides)
    return SecondPassControlLabel(**base)


def test_policy_hash_is_deterministic_and_changes_with_thresholds() -> None:
    first = second_pass_control_policy_hash()
    second = second_pass_control_policy_hash()
    assert first == second
    changed = second_pass_control_policy_hash(disagreement_threshold=0.9)
    assert changed != first


def test_row_dict_serializes_enum_as_plain_string_and_adds_source_split() -> None:
    row = persist_second_pass_control_labels._row_dict(_label(), split="validation")
    assert row["next_step"] == "GENERATE_PLANS"
    assert isinstance(row["next_step"], str)
    assert row["source_split"] == "validation"
    assert row["scenario_id"] == "s1"
    assert row["network_id"] == "golden-reference"
    assert row["topology_hash"] == "topo-hash"


def test_row_dict_preserves_none_candidate_covered() -> None:
    row = persist_second_pass_control_labels._row_dict(_label(candidate_covered=None), split="train")
    assert row["candidate_covered"] is None


def test_row_dict_is_json_serializable() -> None:
    import json

    row = persist_second_pass_control_labels._row_dict(_label(), split="train")
    reloaded = json.loads(json.dumps(row, sort_keys=True))
    assert reloaded["scenario_id"] == "s1"
    assert reloaded["next_step"] == "GENERATE_PLANS"
