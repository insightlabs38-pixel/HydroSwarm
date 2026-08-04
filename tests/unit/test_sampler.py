from __future__ import annotations

import pytest
import torch

from hydroswarm.training import (
    CurriculumStage,
    GroupBalancedSampler,
    ScenarioExample,
    by_curriculum_stage,
    by_network,
    by_source_node,
    composite_key,
)


def _example(scenario_id: str, *, network_id: str, source_node: int = 0, stage: CurriculumStage = CurriculumStage.CLEAN) -> ScenarioExample:
    return ScenarioExample(
        scenario_id=scenario_id,
        network_id=network_id,
        split="train",
        seed=1,
        seed_family=f"family-{scenario_id}",
        stage=stage,
        inputs={"node_features": torch.zeros(2, 3)},
        targets={"source_node": torch.tensor(source_node)},
    )


def _imbalanced_examples() -> list[ScenarioExample]:
    # 90 examples from a large topology, 10 from a small one.
    large = [_example(f"large-{i}", network_id="big-topology") for i in range(90)]
    small = [_example(f"small-{i}", network_id="tiny-topology") for i in range(10)]
    return large + small


def test_group_weight_mass_is_equal_across_groups_regardless_of_size() -> None:
    examples = _imbalanced_examples()
    sampler = GroupBalancedSampler(examples, group_key=by_network, seed=0)
    mass = sampler.weight_mass_by_group(by_network, examples)
    assert mass[("network", "big-topology")] == pytest.approx(0.5, abs=1e-9)
    assert mass[("network", "tiny-topology")] == pytest.approx(0.5, abs=1e-9)


def test_within_group_examples_share_weight_equally() -> None:
    examples = _imbalanced_examples()
    sampler = GroupBalancedSampler(examples, group_key=by_network, seed=0)
    # 90 examples share 0.5 total mass -> each gets 0.5/90; 10 examples share
    # the other 0.5 -> each gets 0.5/10, five times larger per example.
    large_weight = sampler._weights[0]
    small_weight = sampler._weights[95]
    assert small_weight == pytest.approx(large_weight * 9, rel=1e-6)


def test_sampling_empirically_favors_small_group_members_over_many_draws() -> None:
    examples = _imbalanced_examples()
    sampler = GroupBalancedSampler(examples, group_key=by_network, seed=42, num_samples=20_000)
    drawn = list(sampler)
    small_indices = set(range(90, 100))
    small_draws = sum(1 for index in drawn if index in small_indices)
    # Each of the 10 small-group examples should be drawn roughly as often in
    # total as all 90 large-group examples combined (both groups get ~50%
    # mass); allow generous statistical slack since this is seed-dependent.
    assert 0.4 < small_draws / len(drawn) < 0.6


def test_set_epoch_changes_draw_order_deterministically() -> None:
    examples = _imbalanced_examples()
    sampler = GroupBalancedSampler(examples, group_key=by_network, seed=7, num_samples=50)

    sampler.set_epoch(0)
    first_epoch_a = list(sampler)
    sampler.set_epoch(0)
    first_epoch_b = list(sampler)
    assert first_epoch_a == first_epoch_b  # same epoch -> reproducible

    sampler.set_epoch(1)
    second_epoch = list(sampler)
    assert second_epoch != first_epoch_a  # different epoch -> different draw


def test_len_matches_num_samples() -> None:
    examples = _imbalanced_examples()
    sampler = GroupBalancedSampler(examples, group_key=by_network, seed=0, num_samples=37)
    assert len(sampler) == 37
    assert len(list(sampler)) == 37


def test_composite_key_balances_jointly_across_two_dimensions() -> None:
    examples = [
        *[_example(f"a{i}", network_id="netA", stage=CurriculumStage.CLEAN) for i in range(20)],
        *[_example(f"b{i}", network_id="netA", stage=CurriculumStage.DEGRADED) for i in range(2)],
        *[_example(f"c{i}", network_id="netB", stage=CurriculumStage.CLEAN) for i in range(5)],
    ]
    key = composite_key(by_network, by_curriculum_stage)
    sampler = GroupBalancedSampler(examples, group_key=key, seed=0)
    mass = sampler.weight_mass_by_group(key, examples)
    assert len(mass) == 3
    for group_mass in mass.values():
        assert group_mass == pytest.approx(1 / 3, rel=1e-6)


def test_by_source_node_groups_by_target_class() -> None:
    examples = [_example("a", network_id="net1", source_node=0), _example("b", network_id="net1", source_node=1)]
    sampler = GroupBalancedSampler(examples, group_key=by_source_node, seed=0)
    assert sampler.group_sizes() == {("source_node", 0): 1, ("source_node", 1): 1}


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        GroupBalancedSampler([], group_key=by_network, seed=0)


def test_sampler_integrates_with_dataloader() -> None:
    from torch.utils.data import DataLoader

    from hydroswarm.training import GovernedScenarioDataset, collate_scenarios

    examples = _imbalanced_examples()
    dataset = GovernedScenarioDataset(examples, expected_split="train")
    sampler = GroupBalancedSampler(examples, group_key=by_network, seed=0, num_samples=16)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=collate_scenarios)
    batches = list(loader)
    assert len(batches) == 4
    assert all(inputs["node_features"].shape[0] == 4 for inputs, _ in batches)
