from __future__ import annotations

import pytest
import torch

from hydroswarm.model import HydroCore
from hydroswarm.training import CurriculumStage, ScenarioExample, collate_variable_topology


def _tiny_model() -> HydroCore:
    return HydroCore(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        edge_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        latent_tokens=64,
        adapter_dims=(32, 32, 32),
        dropout=0.0,
    )


def _example(
    scenario_id: str,
    *,
    nodes: int,
    edges: list[tuple[int, int]],
    source_local_index: int,
    seed: int,
    missing_at: tuple[int, int] | None = None,
    extra_targets: dict[str, torch.Tensor] | None = None,
) -> ScenarioExample:
    """Builds a ScenarioExample matching what corpus generation actually
    stores in a shard: temporal_features/quality_features already
    NaN-replaced-with-zero (via pad_graph_batch's own nan_to_num), plus the
    separately-computed sensor_mask/quality_mask/node_mask/edge_mask that
    were derived from the real (pre-nan_to_num) values at generation time.
    `missing_at=(step, node)` marks one (step, node) pair as genuinely
    unobserved -- zeroed in the feature tensors and False in both masks --
    exactly like a real missing sensor reading."""

    generator = torch.Generator().manual_seed(seed)
    steps = 2
    edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros(2, 0, dtype=torch.long)
    temporal = torch.randn(steps, nodes, 2, generator=generator)
    quality = torch.randn(steps, nodes, 2, generator=generator)
    sensor_mask = torch.ones(steps, nodes, dtype=torch.bool)
    quality_mask = torch.ones(steps, nodes, dtype=torch.bool)
    if missing_at is not None:
        step, node = missing_at
        temporal[step, node] = 0.0
        quality[step, node] = 0.0
        sensor_mask[step, node] = False
        quality_mask[step, node] = False
    return ScenarioExample(
        scenario_id=scenario_id,
        network_id="net",
        split="train",
        seed=seed,
        seed_family=f"family-{scenario_id}",
        stage=CurriculumStage.CLEAN,
        inputs={
            "node_features": torch.randn(nodes, 3, generator=generator),
            "temporal_features": temporal,
            "quality_features": quality,
            "edge_index": edge_index,
            "edge_features": torch.randn(len(edges), 2, generator=generator) if edges else torch.zeros(0, 2),
            "travel_time": torch.rand(nodes, generator=generator),
            "reservoir_reachability": torch.rand(nodes, generator=generator),
            "demand_centrality": torch.rand(nodes, generator=generator),
            "source_candidate_mask": torch.ones(nodes, dtype=torch.bool),
            "node_mask": torch.ones(nodes, dtype=torch.bool),
            "sensor_mask": sensor_mask,
            "quality_mask": quality_mask,
            "edge_mask": torch.ones(len(edges), dtype=torch.bool),
        },
        targets={
            "source_node": torch.tensor(source_local_index),
            "sensor_fault": torch.zeros(nodes),
            **(extra_targets or {}),
        },
    )


def _small_example(seed: int = 1) -> ScenarioExample:
    return _example("small", nodes=3, edges=[(0, 1), (1, 2)], source_local_index=1, seed=seed)


def _large_example(seed: int = 2) -> ScenarioExample:
    return _example("large", nodes=6, edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], source_local_index=4, seed=seed)


def test_batch_pads_the_phase_6_auxiliary_node_indexed_targets() -> None:
    # Regression: sensor_reconstruction/future_concentration/travel_time
    # (core-issues2.txt Phase 6) are [node_count]-shaped exactly like
    # sensor_fault, but were not originally registered in
    # NODE_INDEXED_TARGET_KEYS -- collating two real different-sized graphs
    # (the whole point of this module) raised "inconsistent shape across
    # the batch" the first time a real multi-topology batch carried them,
    # caught by scripts/run_event_control_smoke_screening.py's own dry run.
    def _aux(nodes: int, seed: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return {
            "sensor_reconstruction": torch.rand(nodes, generator=generator),
            "sensor_reconstruction_mask": torch.ones(nodes, dtype=torch.bool),
            "future_concentration": torch.rand(nodes, generator=generator),
            "future_concentration_mask": torch.ones(nodes, dtype=torch.bool),
            "travel_time": torch.rand(nodes, generator=generator),
            "travel_time_mask": torch.ones(nodes, dtype=torch.bool),
        }

    small = _example(
        "small", nodes=3, edges=[(0, 1), (1, 2)], source_local_index=1, seed=1,
        extra_targets=_aux(3, seed=10),
    )
    large = _example(
        "large", nodes=6, edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], source_local_index=4, seed=2,
        extra_targets=_aux(6, seed=20),
    )
    _, targets = collate_variable_topology([small, large])  # must not raise
    for key in ("sensor_reconstruction", "future_concentration", "travel_time"):
        assert targets[key].shape == (2, 6)
        assert targets[f"{key}_mask"].shape == (2, 6)
        # padded (beyond each example's own node count) positions are False
        assert targets[f"{key}_mask"][0, 3:].tolist() == [False, False, False]
        assert targets[f"{key}_mask"][1, :].tolist() == [True] * 6


def test_batch_pads_scout_per_node_targets() -> None:
    # Regression (core-issues3.txt Phase 10.2): information_gain/
    # candidate_reduction were converted from a scalar (the single selected
    # sample_node's value) to a full [node_count]-shaped array by Phase 7.2
    # -- targets_v2.NODE_ARRAY_TARGETS was updated at the time, but this
    # module's own separately hand-maintained NODE_INDEXED_TARGET_KEYS
    # tuple was not, so collating two real different-sized graphs (the
    # whole point of this module) raised "inconsistent shape across the
    # batch" the first time a real Phase-10.2 Scout dataset actually
    # exercised this path -- not caught by any prior test, since nothing
    # before Phase 10.2 merged real per-node Scout targets into a batch.
    # NODE_INDEXED_TARGET_KEYS is now derived from NODE_ARRAY_TARGETS
    # directly, closing this specific gap and every future one like it.
    def _scout(nodes: int, seed: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return {
            "information_gain": torch.rand(nodes, generator=generator),
            "information_gain_mask": torch.ones(nodes, dtype=torch.bool),
            "candidate_reduction": torch.rand(nodes, generator=generator),
            "candidate_reduction_mask": torch.ones(nodes, dtype=torch.bool),
        }

    small = _example(
        "small", nodes=3, edges=[(0, 1), (1, 2)], source_local_index=1, seed=1,
        extra_targets=_scout(3, seed=10),
    )
    large = _example(
        "large", nodes=6, edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], source_local_index=4, seed=2,
        extra_targets=_scout(6, seed=20),
    )
    _, targets = collate_variable_topology([small, large])  # must not raise
    for key in ("information_gain", "candidate_reduction"):
        assert targets[key].shape == (2, 6)
        assert targets[f"{key}_mask"].shape == (2, 6)
        assert targets[f"{key}_mask"][0, 3:].tolist() == [False, False, False]
        assert targets[f"{key}_mask"][1, :].tolist() == [True] * 6


def test_batch_two_graphs_of_different_sizes() -> None:
    inputs, targets = collate_variable_topology([_small_example(), _large_example()])
    assert inputs["node_features"].shape == (2, 6, 3)  # padded to max_nodes=6
    assert inputs["node_mask"].tolist() == [
        [True, True, True, False, False, False],
        [True, True, True, True, True, True],
    ]
    assert inputs["source_candidate_mask"].shape == (2, 6)
    assert inputs["source_candidate_mask"][0].tolist() == [True, True, True, False, False, False]
    assert targets["source_node"].tolist() == [1, 4]
    assert targets["sensor_fault"].shape == (2, 6)


def test_forward_pass_through_hydrocore_with_padded_batch() -> None:
    model = _tiny_model().eval()
    inputs, _ = collate_variable_topology([_small_example(), _large_example()])
    with torch.no_grad():
        output = model(inputs)
    assert output["source_node_logits"].shape == (2, 6)


def test_source_logits_are_masked_correctly_for_padded_nodes() -> None:
    model = _tiny_model().eval()
    inputs, _ = collate_variable_topology([_small_example(), _large_example()])
    with torch.no_grad():
        output = model(inputs)
    padded_logits = output["source_node_logits"][0, 3:]  # small example's padded positions
    assert torch.all(padded_logits <= torch.finfo(padded_logits.dtype).min / 2)


def test_padding_does_not_affect_valid_node_predictions() -> None:
    model = _tiny_model().eval()
    small = _small_example()

    alone_inputs, _ = collate_variable_topology([small])
    with torch.no_grad():
        alone_output = model(alone_inputs)

    padded_inputs, _ = collate_variable_topology([small, _large_example()])
    with torch.no_grad():
        padded_output = model(padded_inputs)

    torch.testing.assert_close(
        alone_output["source_node_logits"][0, :3],
        padded_output["source_node_logits"][0, :3],
        atol=1e-5,
        rtol=1e-4,
    )


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_variable_topology([])


def test_stored_sensor_and_quality_masks_survive_collation_not_rederived_as_all_true() -> None:
    """Regression test (core-issues.txt repair item 1): collate_variable_topology
    used to drop the shard's own sensor_mask/quality_mask/edge_mask entirely and
    let pad_graph_batch re-derive them via torch.isfinite() on tensors that were
    already NaN-replaced-with-zero at corpus-generation time -- isfinite(0.0) is
    True, so a genuinely missing (step, node) reading silently became "observed"
    the moment two examples were batched together."""

    small = _small_example()  # nodes=3, no missing entries
    large = _example(
        "large-missing", nodes=6, edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        source_local_index=4, seed=2, missing_at=(1, 2),  # step 1, node 2 unobserved
    )
    inputs, _ = collate_variable_topology([small, large])

    # The missing (step, node) pair must still read False after padding --
    # not silently flipped to True by re-deriving from zero-filled data.
    assert inputs["sensor_mask"][1, 1, 2].item() is False
    assert inputs["quality_mask"][1, 1, 2].item() is False
    # Every other real (non-padded, non-missing) position for this example
    # must remain True -- the fix must not blank out unrelated positions.
    assert inputs["sensor_mask"][1, 0, 2].item() is True
    assert inputs["sensor_mask"][1, 1, 0].item() is True
    # node_mask/edge_mask (shape-derived, not NaN-derived) still behave as
    # documented by the existing padding test.
    assert inputs["node_mask"][0].tolist() == [True, True, True, False, False, False]
    assert inputs["edge_mask"][0].tolist() == [True, True, False, False, False]
    # Padded rows beyond either example's real span stay False, as before.
    assert inputs["sensor_mask"][0, :, 3:].any().item() is False


def test_missing_sensor_mask_input_falls_back_to_isfinite_derivation() -> None:
    """When an example genuinely has no precomputed mask (e.g. a plain
    single-example construction, matching HydraulicFeatureBuilder.build's
    own pad_graph_batch([sample]) call before any mask exists yet),
    collation must still derive one from isfinite() rather than crash or
    silently mark everything unobserved."""

    example = _small_example()
    del example.inputs["sensor_mask"]
    del example.inputs["quality_mask"]
    del example.inputs["edge_mask"]
    inputs, _ = collate_variable_topology([example])
    assert inputs["sensor_mask"][0].all().item() is True
    assert inputs["quality_mask"][0].all().item() is True
    assert inputs["edge_mask"][0].all().item() is True


def test_collate_rejects_mismatched_target_keys() -> None:
    small = _small_example()
    other = ScenarioExample(
        scenario_id="other",
        network_id="net",
        split="train",
        seed=3,
        seed_family="family-other",
        stage=CurriculumStage.CLEAN,
        inputs=dict(small.inputs),
        targets={"source_node": torch.tensor(0)},  # missing sensor_fault
    )
    with pytest.raises(ValueError, match="identical target keys"):
        collate_variable_topology([small, other])
