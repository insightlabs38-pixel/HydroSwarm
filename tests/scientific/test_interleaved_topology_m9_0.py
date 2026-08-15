"""Milestone 9.0 Section 25: interleaved multi-family training regression
tests. Proves, before Arm B is trained for real, that:

1. the per-family curriculum-stage-filtered scenario counts are balanced
   (no family can dominate a training epoch);
2. one `interleaved_optimizer_step` call accumulates gradients from ALL
   configured families onto the SAME model parameters before any
   `optimizer.step()` -- i.e. it is genuinely equivalent to summing each
   family's own independently-computed gradient, not silently dropping or
   overwriting any family's contribution;
3. each family's `CausalPrefixDatasetView` uses that family's own
   `SignatureLibrary` and never another family's (with the existing
   `scenario_to_prefix_example` cross-family-library assertion as the
   structural backstop);
4. `unobserved_age_sentinel="fixed"` / `include_relative_gap_feature=False`
   (AGE_FIX_ONLY) survives `stages_through()` for every one of Arm B's
   three families, not just golden-reference (M8.7's own regression only
   covered the single-family case);
5. no future evidence enters a depth-truncated prefix, for a non-golden
   topology family (M8.7's own causal-truncation test only covered
   golden-reference);
6. the frozen per-family scenario-budget accounting
   (`TRAIN_PER_FAMILY * NUM_FAMILIES == 600`, matching Arm A's own
   600-scenario budget) holds.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CausalPrefixDatasetView,
    fit_pool_signature_library,
    full_history_policy,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.data import CurriculumStage  # noqa: E402
from hydroswarm.training.trainer import set_deterministic_seed  # noqa: E402

from run_m7_topology import SEED_BASES, TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m9_0_arm_b import (  # noqa: E402
    CALIBRATION_PER_FAMILY,
    FAMILY_NAMES,
    FEATURE_KWARGS,
    NUM_FAMILIES,
    TRAIN_PER_FAMILY,
    interleaved_optimizer_step,
)

pytestmark = pytest.mark.real_simulation

#: Covers all 5 curriculum stages exactly 3 times (15 % 5 == 0) AND, with
#: source_round_robin=True, every junction of the largest trained family
#: (branched-loop/loop-grid, 13 junctions each) at least once.
SMALL_POOL_COUNT = 15


def _small_pools() -> dict[str, list]:
    # source_round_robin=True (M7's own _classical_fit_library convention)
    # guarantees fit_pool_signature_library sees every junction as a
    # source at least once even at this small a count -- plain random
    # source assignment (build_scenario_pool's default) needs hundreds of
    # scenarios for that same guarantee to hold reliably.
    return {
        family: _family_scenario_pool(
            "train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")],
            count=SMALL_POOL_COUNT, source_round_robin=True,
        )
        for family, loader in TRAINED_FAMILIES
    }


def _small_config() -> TrainingConfig:
    return TrainingConfig(task_weights={"source_node": 1.0}, batch_size=1, gradient_clip_norm=1.0)


def _small_model() -> HydroCore:
    return HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)


def test_family_pools_have_balanced_curriculum_stage_counts() -> None:
    pools = _small_pools()
    libraries = {family: fit_pool_signature_library(records) for family, records in pools.items()}
    views = {
        family: CausalPrefixDatasetView(
            pools[family], expected_split="train", signature_library=libraries[family],
            depth_policy=full_history_policy, base_seed=20260814, batch_size=1, **FEATURE_KWARGS,
        )
        for family in FAMILY_NAMES
    }
    for stage in CurriculumStage:
        counts = {family: len(view.stages_through(stage)) for family, view in views.items()}
        assert len(set(counts.values())) == 1, f"family curriculum-filtered counts diverge at stage {stage.name}: {counts}"


def test_gradient_accumulates_across_families_before_step() -> None:
    pools = _small_pools()
    libraries = {family: fit_pool_signature_library(records) for family, records in pools.items()}
    config = _small_config()

    examples = {
        family: scenario_to_prefix_example(
            pools[family][0].scenario, pools[family][0].network, libraries[family], 25,
            feature_context=pools[family][0].feature_context, **FEATURE_KWARGS,
        )
        for family in FAMILY_NAMES
    }
    family_batches = {
        family: ({k: v.unsqueeze(0) for k, v in ex.inputs.items()}, {k: v.unsqueeze(0) for k, v in ex.targets.items()})
        for family, ex in examples.items()
    }

    # Matches production training's own determinism guarantee
    # (TrainingConfig.deterministic=True everywhere in this repo) -- without
    # it, CPU BLAS reduction order is non-deterministic between the
    # combined run's 3 sequential forward/backward calls and each solo
    # run's single call, and near-zero gradient components (like
    # global_latents', which sit at the tail of many nearly-cancelling
    # contributions) can show large RELATIVE divergence from that
    # nondeterminism alone even though the true accumulation logic is
    # correct.
    set_deterministic_seed(0, deterministic=True)
    base_model = _small_model()
    base_state = copy.deepcopy(base_model.state_dict())

    # .eval() disables dropout so this test isolates the pure accumulation
    # arithmetic: without it, the combined run's 3 sequential forward
    # passes and each solo run's single forward pass draw dropout masks
    # from different points in torch's global RNG stream, and gradients
    # would legitimately differ for that unrelated reason.
    combined_model = _small_model().eval()
    combined_model.load_state_dict(base_state)
    combined_optimizer = torch.optim.AdamW(combined_model.parameters(), lr=1e-3)
    interleaved_optimizer_step(combined_model, combined_optimizer, family_batches, config=config, step=False, clip=False)
    combined_grads = {name: p.grad.clone() for name, p in combined_model.named_parameters() if p.grad is not None}

    summed_grads: dict[str, torch.Tensor] = {}
    for family, (inputs, targets) in family_batches.items():
        solo_model = _small_model().eval()
        solo_model.load_state_dict(base_state)
        solo_optimizer = torch.optim.AdamW(solo_model.parameters(), lr=1e-3)
        interleaved_optimizer_step(solo_model, solo_optimizer, {family: (inputs, targets)}, config=config, step=False, clip=False)
        for name, p in solo_model.named_parameters():
            if p.grad is None:
                continue
            # Each solo call already divides by len(family_batches)=1, so
            # rescale to the SAME per-family normalizer the combined call
            # used (1/NUM_FAMILIES) before summing across families.
            contribution = p.grad.clone() / NUM_FAMILIES
            summed_grads[name] = summed_grads.get(name, torch.zeros_like(contribution)) + contribution

    assert combined_grads.keys() == summed_grads.keys()
    assert len(combined_grads) > 0, "no parameter received any gradient -- test setup is broken"
    for name in combined_grads:
        assert torch.allclose(combined_grads[name], summed_grads[name], atol=1e-5, rtol=1e-4), (
            f"parameter {name}: combined-step gradient does not equal the sum of independent per-family "
            "gradients -- some family's contribution was dropped, overwritten, or double-counted"
        )


def test_optimizer_state_shared_not_per_family_weights() -> None:
    pools = _small_pools()
    libraries = {family: fit_pool_signature_library(records) for family, records in pools.items()}
    config = _small_config()
    examples = {
        family: scenario_to_prefix_example(
            pools[family][0].scenario, pools[family][0].network, libraries[family], 25,
            feature_context=pools[family][0].feature_context, **FEATURE_KWARGS,
        )
        for family in FAMILY_NAMES
    }
    family_batches = {
        family: ({k: v.unsqueeze(0) for k, v in ex.inputs.items()}, {k: v.unsqueeze(0) for k, v in ex.targets.items()})
        for family, ex in examples.items()
    }
    model = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: p.clone() for name, p in model.named_parameters()}
    interleaved_optimizer_step(model, optimizer, family_batches, config=config, step=True)
    after = dict(model.named_parameters())
    changed = [name for name in before if not torch.equal(before[name], after[name].detach())]
    assert changed, "no parameter changed after an interleaved step with 3 non-empty families -- optimizer.step() had no effect"
    # There is exactly ONE model / ONE optimizer instance handling all
    # three families -- structurally guaranteed here (no per-family model
    # variable exists anywhere in interleaved_optimizer_step), and the
    # AdamW state dict now has entries seeded by all three families'
    # gradients combined, not three separate optimizers.
    assert len(optimizer.state) > 0


def test_each_family_uses_its_own_signature_library() -> None:
    pools = _small_pools()
    libraries = {family: fit_pool_signature_library(records) for family, records in pools.items()}
    node_id_sets = {family: tuple(library.node_ids) for family, library in libraries.items()}
    assert len(set(node_id_sets.values())) == NUM_FAMILIES, (
        f"expected every family to have a structurally distinct node_id set, got {node_id_sets}"
    )
    golden, branched = FAMILY_NAMES[0], FAMILY_NAMES[1]
    with pytest.raises(ValueError):
        scenario_to_prefix_example(
            pools[golden][0].scenario, pools[golden][0].network, libraries[branched], 25,
            feature_context=pools[golden][0].feature_context, **FEATURE_KWARGS,
        )


def test_age_fix_only_propagates_through_stages_through_for_every_family() -> None:
    pools = _small_pools()
    libraries = {family: fit_pool_signature_library(records) for family, records in pools.items()}
    for family in FAMILY_NAMES:
        view = CausalPrefixDatasetView(
            pools[family], expected_split="train", signature_library=libraries[family],
            depth_policy=full_history_policy, base_seed=20260814, batch_size=1, **FEATURE_KWARGS,
        )
        staged = view.stages_through(CurriculumStage.ADVERSARIAL)
        assert staged._unobserved_age_sentinel == "fixed", family
        assert staged._include_relative_gap_feature is False, family
        example = staged[0]
        assert example.inputs["temporal_features"].shape[-1] == 6, family  # AGE_FIX_ONLY keeps Arm-A's dims.


def test_no_future_evidence_in_truncated_prefix_for_non_golden_family() -> None:
    family, loader = TRAINED_FAMILIES[1]  # branched-loop -- not golden-reference, unlike M8.7's own test.
    pool = _family_scenario_pool("train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")], count=1)
    record = pool[0]
    from hydroswarm.training.corpus import build_sensor_series

    full_series = build_sensor_series(record.scenario, record.feature_context)
    depth = 3
    truncated = [truncate_causal_prefix(item, depth) for item in full_series]
    for original, cut in zip(full_series, truncated, strict=True):
        assert len(cut.timestamps_seconds) <= depth
        if cut.timestamps_seconds:
            assert cut.timestamps_seconds == original.timestamps_seconds[: len(cut.timestamps_seconds)]
            if len(original.timestamps_seconds) > depth:
                assert max(cut.timestamps_seconds) <= original.timestamps_seconds[depth - 1]


def test_scenario_budget_accounting_matches_arm_a() -> None:
    ARM_A_TOTAL_TRAIN_SCENARIOS = 600  # reports/evaluation/hydrocore-v5/m8-7-runs/AGE_FIX_ONLY-seed*.json.
    assert TRAIN_PER_FAMILY * NUM_FAMILIES == ARM_A_TOTAL_TRAIN_SCENARIOS
    ARM_A_TOTAL_CALIBRATION_SCENARIOS = 150
    assert CALIBRATION_PER_FAMILY * NUM_FAMILIES == ARM_A_TOTAL_CALIBRATION_SCENARIOS
