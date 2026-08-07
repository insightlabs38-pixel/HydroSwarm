"""core-issues3.txt Phase 10 item 5 / core-issues4.txt Section I: real
multi-topology gradient/training smoke tests for every retained v4 output.

Scout (sample_node, information_gain, candidate_reduction,
should_continue_sampling) and Strategist (CandidatePlanEncoder,
plan_value, plan_validity, the five consequence proxies) are already
covered end to end by tests/integration/test_scout_state_dataset.py and
tests/integration/test_strategist_candidate_dataset.py respectively (both
added alongside this file, Phase 10.2/10.3). This file covers everything
else retained in v4: the governed Sentinel heads, event/control heads,
validated auxiliary heads, and the advisory ood_category head -- combined
in ONE real multi-topology batch (event_control_heads + auxiliary_heads +
ood_category_head + scout_control_heads all enabled together, matching a
genuine joint-multitask configuration, not one flag at a time) plus a
second, OOD-extension-specific batch exercising ood_category_head against
genuinely diverse real non-NONE classes (cycle-b2's own data is almost
entirely NONE-class; a batch drawn only from it would not actually
exercise the head's non-NONE logits).

Real data throughout: cycle-b2-trajectories-v3's completed validation
split (Phase 10.1, corrected for Phases 1-7) merged onto cycle-b2's base
tensors via scripts/merge_trajectory_targets.py, and
cycle-b2-ood-extension's real, verified per-category scenarios
(Phase 10.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import merge_trajectory_targets  # noqa: E402

from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training import ShardedScenarioDataset, collate_variable_topology  # noqa: E402
from hydroswarm.training.losses import compute_multitask_loss  # noqa: E402

_CYCLE_B2_TENSORS = Path("data/learning-v2/cycle-b2/tensors-normalized/validation")
_TRAJECTORY_JSONL = Path("data/learning-v2/cycle-b2-trajectories-v3/validation.jsonl")
_OOD_EXTENSION_ROOT = Path("data/learning-v2/cycle-b2-ood-extension/tensors-normalized")

pytestmark = pytest.mark.skipif(
    not (_CYCLE_B2_TENSORS.exists() and _TRAJECTORY_JSONL.exists() and _OOD_EXTENSION_ROOT.exists()),
    reason="requires the real cycle-b2, cycle-b2-trajectories-v3 (validation), and cycle-b2-ood-extension corpora",
)


def test_every_retained_sentinel_control_auxiliary_and_ood_head_receives_a_real_gradient(tmp_path: Path) -> None:
    merged_dir = tmp_path / "merged"
    merge_trajectory_targets.merge(_CYCLE_B2_TENSORS, _TRAJECTORY_JSONL, merged_dir, expected_split="validation")

    dataset = ShardedScenarioDataset(merged_dir, expected_split="validation")
    families: dict[str, list] = {}
    for index in range(len(dataset)):
        example = dataset[index]
        families.setdefault(example.network_id, []).append(example)
        if len(families) >= 3 and all(len(group) >= 8 for group in families.values()):
            break
    examples = [example for group in families.values() for example in group[:8]]
    assert len(families) >= 3, f"expected all 3 real topologies, got {sorted(families)}"

    inputs, targets = collate_variable_topology(examples)
    model = HydroCore.from_variant(
        "small",
        prior_mode="feature_only",
        event_control_heads=True,
        auxiliary_heads=True,
        ood_category_head=True,
        scout_control_heads=True,
    )
    model.train()
    output = model(inputs)
    result = compute_multitask_loss(output, targets)
    assert torch.isfinite(result.total)
    result.total.backward()

    # scout_control_heads/scout is deliberately excluded here -- this
    # batch carries no Scout targets (see test_scout_state_dataset.py for
    # that coverage), so those heads correctly receive zero gradient from
    # THIS batch; asserting otherwise would be testing the wrong thing.
    heads_expected_nonzero = (
        "source_node_head", "source_region_head", "profile_heads", "sensor_fault_head", "evidence_head",
        "event_presence_head", "event_cause_head", "next_step_head", "ood_category_head",
        "sensor_reconstruction_head", "travel_time_head",
    )
    for head_name in heads_expected_nonzero:
        head = getattr(model, head_name)
        gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
        assert gradients, f"{head_name} received no gradient at all"
        assert any(float(gradient.abs().sum()) > 0 for gradient in gradients), f"{head_name} gradient is all-zero"


def test_ood_category_head_receives_gradient_from_genuinely_diverse_real_classes() -> None:
    # cycle-b2's own data is almost entirely ood_class=NONE -- a batch
    # drawn only from it would not meaningfully exercise the head's
    # non-NONE logits. cycle-b2-ood-extension supplies real, verified,
    # non-NONE examples for exactly this purpose.
    examples = []
    for category in ("EXTREME_DEMAND", "TANK_STATE_SHIFT", "ROUGHNESS_MISMATCH", "FROZEN_DRIFTING_SENSOR"):
        dataset = ShardedScenarioDataset(_OOD_EXTENSION_ROOT / f"ood-{category}", expected_split="development_holdout")
        dataset.verify_shard_checksums()
        examples.append(dataset[0])
        examples.append(dataset[200])

    classes = {int(example.targets["ood_class"]) for example in examples}
    assert len(classes) == 4, f"expected 4 genuinely distinct real OOD classes, got {classes}"

    inputs, targets = collate_variable_topology(examples)
    model = HydroCore.from_variant("small", prior_mode="feature_only", ood_category_head=True)
    model.train()
    output = model(inputs)
    result = compute_multitask_loss(output, targets)
    assert torch.isfinite(result.total)
    assert "ood_class" in result.tasks
    result.total.backward()
    gradients = [parameter.grad for parameter in model.ood_category_head.parameters() if parameter.grad is not None]
    assert gradients and any(float(gradient.abs().sum()) > 0 for gradient in gradients)
