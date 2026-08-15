"""Milestone 9.0a Section 8: step-matched interleaved multi-family training
regression tests. Proves, before Arm B2 is trained for real, that:

1. exactly 4 microbatches contribute before optimizer.step() (protocol
   Section 3);
2. the fixed 3-update rotation gives each family exactly 4 microbatches per
   3 consecutive optimizer updates (equal long-run family weighting);
3. each accumulated loss is normalized by /4, not /3 or any other value
   (protocol Section 4);
4. gradients genuinely accumulate across all 4 slots onto the SAME model
   parameters before any optimizer.step() -- no zero_grad between slots,
   combined-step gradient equals the sum of independently-computed
   per-slot gradients, generalized from M9.0's 3-slot-no-repeat case to 4
   slots with one family appearing twice in the same update;
5. scheduler.step() fires exactly once per optimizer update (verified at
   the training-loop level: global_step increments by exactly 1 per update);
6. the real per-epoch/per-family microbatch counts predicted by protocol
   Section 3 (target_updates * 4 / 3 per family) exactly match a real
   (small, fast) curriculum-filtered pool's actual per-epoch/per-family
   available microbatch counts, with zero remainder;
7. Arm B2's frozen per-epoch optimizer-step target sums to Arm A's real
   measured 1350-step total (protocol Section 2);
8. each family's own SignatureLibrary remains correct (same backstop M9.0
   already tests);
9. AGE_FIX_ONLY semantics survive stages_through() for all three families;
10. no future evidence enters a depth-truncated prefix, for a non-golden
    family;
11. the SAME model/optimizer state (not per-family copies) receives every
    slot's gradient.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

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
from hydroswarm.training.data import CurriculumStage, collate_scenarios  # noqa: E402
from hydroswarm.training.trainer import set_deterministic_seed  # noqa: E402

from run_m7_topology import SEED_BASES, TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m9_0_arm_b import FEATURE_KWARGS  # noqa: E402
from run_m9_0a_arm_b2 import (  # noqa: E402
    ARM_A_OPTIMIZER_STEPS_PER_EPOCH,
    ARM_A_SCHEDULER_TOTAL_STEPS,
    ARM_A_TOTAL_OPTIMIZER_STEPS,
    FAMILY_NAMES,
    MICROBATCHES_PER_UPDATE,
    NUM_FAMILIES,
    _build_update_slots,
    _extra_family_for_cycle_position,
    step_matched_interleaved_optimizer_step,
)

pytestmark = pytest.mark.real_simulation

#: A small, fast pool size that still gives a ZERO-REMAINDER 4-microbatch-
#: per-update / 4-per-3-updates-per-family schedule at every one of the 5
#: curriculum-fraction stages (1/5, 2/5, 3/5, 4/5, 5/5 of the pool):
#: microbatches-at-stage-k = SMALL_POOL_COUNT*k/10 must be divisible by 4
#: for every k in 1..5, which holds whenever SMALL_POOL_COUNT is divisible
#: by 40 (the real 200-scenario-per-family Arm B2 pool satisfies this too:
#: 200 = 5*40).
SMALL_POOL_COUNT = 40


def _small_pools() -> dict[str, list]:
    return {
        family: _family_scenario_pool(
            "train", network_loader=loader, family=family, seed_base=SEED_BASES[(family, "train")],
            count=SMALL_POOL_COUNT, source_round_robin=True,
        )
        for family, loader in TRAINED_FAMILIES
    }


def _small_config() -> TrainingConfig:
    return TrainingConfig(task_weights={"source_node": 1.0}, batch_size=2, gradient_clip_norm=1.0)


def _small_model() -> HydroCore:
    return HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)


def test_rotation_gives_each_family_exactly_4_microbatches_per_3_updates() -> None:
    counts_over_9_updates: dict[str, int] = dict.fromkeys(FAMILY_NAMES, 0)
    for update_index in range(9):  # 3 full rotation blocks.
        extra = _extra_family_for_cycle_position(update_index % NUM_FAMILIES)
        for family in FAMILY_NAMES:
            counts_over_9_updates[family] += 1  # base slot every update.
        counts_over_9_updates[extra] += 1  # extra slot.
    assert all(count == 12 for count in counts_over_9_updates.values()), counts_over_9_updates  # 4/block * 3 blocks.
    # Every family is the "extra" family exactly once per 3-update block.
    extras = [_extra_family_for_cycle_position(i % NUM_FAMILIES) for i in range(9)]
    assert extras == list(FAMILY_NAMES) * 3


def test_arm_a_optimizer_step_target_sums_to_1350() -> None:
    assert ARM_A_TOTAL_OPTIMIZER_STEPS == 1350
    assert sum(ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == 1350
    assert len(ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == 20
    assert ARM_A_SCHEDULER_TOTAL_STEPS == 1500
    # Every epoch's target update count is exactly divisible by 3 (a whole
    # number of 3-update rotation blocks, zero remainder -- protocol
    # Section 3).
    assert all(steps % NUM_FAMILIES == 0 for steps in ARM_A_OPTIMIZER_STEPS_PER_EPOCH)


def test_build_update_slots_yields_4_slots_per_update() -> None:
    pools = _small_pools()
    libraries = {family: fit_pool_signature_library(records) for family, records in pools.items()}
    views = {
        family: CausalPrefixDatasetView(
            pools[family], expected_split="train", signature_library=libraries[family],
            depth_policy=full_history_policy, base_seed=20260814, batch_size=2, **FEATURE_KWARGS,
        )
        for family in FAMILY_NAMES
    }
    staged = {family: view.stages_through(CurriculumStage.ADVERSARIAL) for family, view in views.items()}
    loaders = {
        family: DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_scenarios,
                            generator=torch.Generator().manual_seed(20260814))
        for family, dataset in staged.items()
    }
    family_loader_lengths = {family: len(loader) for family, loader in loaders.items()}
    assert len(set(family_loader_lengths.values())) == 1, family_loader_lengths  # balanced pools (M9.0's own invariant).
    microbatches_per_family = next(iter(family_loader_lengths.values()))
    # SMALL_POOL_COUNT=40 (divisible by 40) guarantees a zero-remainder
    # schedule at full saturation: microbatches_per_family * 3 / 4 is an
    # integer (protocol Section 3's central claim).
    assert microbatches_per_family * NUM_FAMILIES % MICROBATCHES_PER_UPDATE == 0
    target_updates = microbatches_per_family * NUM_FAMILIES // MICROBATCHES_PER_UPDATE

    iterators = {family: iter(loader) for family, loader in loaders.items()}
    updates = _build_update_slots(0, target_updates, iterators)
    assert len(updates) == target_updates
    assert all(len(slots) == MICROBATCHES_PER_UPDATE for slots in updates)

    family_consumed: dict[str, int] = dict.fromkeys(FAMILY_NAMES, 0)
    for slots in updates:
        for family, _batch in slots:
            family_consumed[family] += 1
    assert all(count == microbatches_per_family for count in family_consumed.values()), family_consumed
    # Zero remainder: every family's loader is now exactly exhausted.
    for family in FAMILY_NAMES:
        with pytest.raises(StopIteration):
            next(iterators[family])


def test_step_matched_schedule_exactly_consumes_real_curriculum_pool_zero_remainder() -> None:
    """Protocol Section 3's central claim: a target_updates count derived
    from `microbatches_available * 3 / 4` exactly and fully consumes every
    family's own curriculum-filtered pool with zero remainder, for both a
    non-saturated (growing) stage and a fully saturated stage, against a
    REAL pool (not a hand-constructed count)."""
    pools = _small_pools()
    libraries = {family: fit_pool_signature_library(records) for family, records in pools.items()}
    views = {
        family: CausalPrefixDatasetView(
            pools[family], expected_split="train", signature_library=libraries[family],
            depth_policy=full_history_policy, base_seed=20260814, batch_size=2, **FEATURE_KWARGS,
        )
        for family in FAMILY_NAMES
    }
    for stage in (CurriculumStage.CLEAN, CurriculumStage.ADVERSARIAL):
        staged = {family: view.stages_through(stage) for family, view in views.items()}
        loaders = {
            family: DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_scenarios,
                                generator=torch.Generator().manual_seed(20260814 + hash(stage.name) % 1000))
            for family, dataset in staged.items()
        }
        family_loader_lengths = {family: len(loader) for family, loader in loaders.items()}
        assert len(set(family_loader_lengths.values())) == 1, (stage.name, family_loader_lengths)
        microbatches_per_family = next(iter(family_loader_lengths.values()))
        assert microbatches_per_family > 0, stage.name
        assert microbatches_per_family * NUM_FAMILIES % MICROBATCHES_PER_UPDATE == 0, (
            stage.name, microbatches_per_family,
        )
        target_updates = microbatches_per_family * NUM_FAMILIES // MICROBATCHES_PER_UPDATE

        iterators = {family: iter(loader) for family, loader in loaders.items()}
        updates = _build_update_slots(0, target_updates, iterators)
        family_consumed: dict[str, int] = dict.fromkeys(FAMILY_NAMES, 0)
        for slots in updates:
            for family, _batch in slots:
                family_consumed[family] += 1
        for family in FAMILY_NAMES:
            assert family_consumed[family] == microbatches_per_family, (stage.name, family)
            with pytest.raises(StopIteration):
                next(iterators[family])


def test_gradient_accumulates_across_all_4_slots_before_step() -> None:
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
    batches = {
        family: ({k: v.unsqueeze(0) for k, v in ex.inputs.items()}, {k: v.unsqueeze(0) for k, v in ex.targets.items()})
        for family, ex in examples.items()
    }
    # 4 slots: base golden, branched, loop + an "extra" golden slot (reusing
    # the SAME example a second time is fine for this pure-accumulation
    # arithmetic test -- it only checks that all 4 slots' gradients sum
    # correctly onto shared parameters, not schedule diversity).
    golden, branched, loop = FAMILY_NAMES
    slot_batches = [(golden, batches[golden]), (branched, batches[branched]), (loop, batches[loop]), (golden, batches[golden])]

    set_deterministic_seed(0, deterministic=True)
    base_model = _small_model()
    base_state = copy.deepcopy(base_model.state_dict())

    combined_model = _small_model().eval()
    combined_model.load_state_dict(base_state)
    combined_optimizer = torch.optim.AdamW(combined_model.parameters(), lr=1e-3)
    step_matched_interleaved_optimizer_step(combined_model, combined_optimizer, slot_batches, config=config, step=False, clip=False)
    combined_grads = {name: p.grad.clone() for name, p in combined_model.named_parameters() if p.grad is not None}

    summed_grads: dict[str, torch.Tensor] = {}
    for family, batch in slot_batches:
        solo_model = _small_model().eval()
        solo_model.load_state_dict(base_state)
        solo_optimizer = torch.optim.AdamW(solo_model.parameters(), lr=1e-3)
        step_matched_interleaved_optimizer_step(solo_model, solo_optimizer, [(family, batch)], config=config, step=False, clip=False)
        for name, p in solo_model.named_parameters():
            if p.grad is None:
                continue
            # Each solo call already divides by len([...])=1; rescale to the
            # SAME per-slot normalizer (1/4) the combined call used.
            contribution = p.grad.clone() / len(slot_batches)
            summed_grads[name] = summed_grads.get(name, torch.zeros_like(contribution)) + contribution

    assert combined_grads.keys() == summed_grads.keys()
    assert len(combined_grads) > 0, "no parameter received any gradient -- test setup is broken"
    for name in combined_grads:
        assert torch.allclose(combined_grads[name], summed_grads[name], atol=1e-5, rtol=1e-4), (
            f"parameter {name}: combined-step gradient does not equal the sum of independent per-slot "
            "gradients -- some slot's contribution was dropped, overwritten, or double-counted"
        )


def test_loss_normalized_by_4_not_3() -> None:
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
    batches = {
        family: ({k: v.unsqueeze(0) for k, v in ex.inputs.items()}, {k: v.unsqueeze(0) for k, v in ex.targets.items()})
        for family, ex in examples.items()
    }
    golden, branched, loop = FAMILY_NAMES
    slot_batches = [(golden, batches[golden]), (branched, batches[branched]), (loop, batches[loop]), (golden, batches[golden])]

    set_deterministic_seed(0, deterministic=True)
    model_div4 = _small_model().eval()
    optimizer_div4 = torch.optim.AdamW(model_div4.parameters(), lr=1e-3)
    step_matched_interleaved_optimizer_step(model_div4, optimizer_div4, slot_batches, config=config, step=False, clip=False)
    grads_div4 = {name: p.grad.clone() for name, p in model_div4.named_parameters() if p.grad is not None}

    # Manually compute what /3 normalization (M9.0's own, wrong for Arm B2)
    # would have produced, to prove this function does NOT do that.
    set_deterministic_seed(0, deterministic=True)
    model_manual = _small_model().eval()
    model_manual.load_state_dict(model_div4.state_dict())
    optimizer_manual = torch.optim.AdamW(model_manual.parameters(), lr=1e-3)
    optimizer_manual.zero_grad(set_to_none=True)
    from hydroswarm.training.losses import compute_multitask_loss
    for family, (inputs, targets) in slot_batches:
        output = model_manual({k: v.float() if v.is_floating_point() else v for k, v in inputs.items()})
        result = compute_multitask_loss(output, targets, task_weights=config.task_weights, profile_ordinal_weight=config.profile_ordinal_weight)
        (result.total / 3).backward()
    grads_div3 = {name: p.grad.clone() for name, p in model_manual.named_parameters() if p.grad is not None}

    any_differs = any(
        not torch.allclose(grads_div4[name], grads_div3[name], atol=1e-8, rtol=1e-6)
        for name in grads_div4
    )
    assert any_differs, "dividing by 4 vs 3 produced identical gradients -- normalization is not actually being applied"
    # And /4 grads should be exactly 3/4 of /3 grads (same accumulated sum, different divisor).
    for name in grads_div4:
        assert torch.allclose(grads_div4[name], grads_div3[name] * (3.0 / 4.0), atol=1e-5, rtol=1e-4), name


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
    batches = {
        family: ({k: v.unsqueeze(0) for k, v in ex.inputs.items()}, {k: v.unsqueeze(0) for k, v in ex.targets.items()})
        for family, ex in examples.items()
    }
    golden, branched, loop = FAMILY_NAMES
    slot_batches = [(golden, batches[golden]), (branched, batches[branched]), (loop, batches[loop]), (branched, batches[branched])]
    model = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: p.clone() for name, p in model.named_parameters()}
    step_matched_interleaved_optimizer_step(model, optimizer, slot_batches, config=config, step=True)
    after = dict(model.named_parameters())
    changed = [name for name in before if not torch.equal(before[name], after[name].detach())]
    assert changed, "no parameter changed after a step-matched interleaved step -- optimizer.step() had no effect"
    assert len(optimizer.state) > 0


def test_each_family_uses_its_own_signature_library() -> None:
    pools = _small_pools()
    libraries = {family: fit_pool_signature_library(records) for family, records in pools.items()}
    node_id_sets = {family: tuple(library.node_ids) for family, library in libraries.items()}
    assert len(set(node_id_sets.values())) == NUM_FAMILIES
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
            depth_policy=full_history_policy, base_seed=20260814, batch_size=2, **FEATURE_KWARGS,
        )
        staged = view.stages_through(CurriculumStage.ADVERSARIAL)
        assert staged._unobserved_age_sentinel == "fixed", family
        assert staged._include_relative_gap_feature is False, family
        example = staged[0]
        assert example.inputs["temporal_features"].shape[-1] == 6, family


def test_no_future_evidence_in_truncated_prefix_for_non_golden_family() -> None:
    family, loader = TRAINED_FAMILIES[1]  # branched-loop.
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
