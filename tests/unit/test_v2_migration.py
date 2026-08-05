"""core-issues.txt repair item 2 introduced a real architecture change
(source_region_head: 1 output -> SOURCE_REGION_COUNT=3, applied to
incident_context instead of per node), which changes that head's tensor
shape. A checkpoint saved under the old (v2) architecture -- including the
currently promoted models/hydrocore-s-learning-v1.safetensors -- would
otherwise fail to load at all under strict=True. These tests cover the
narrow migration path that tolerates exactly that one, understood shape
change and nothing else."""

from __future__ import annotations

import copy

import pytest
import torch

from hydroswarm.model import HydroCore, load_state_dict_with_v2_migration
from hydroswarm.model.core import V2_TO_V3_RESHAPED_PARAMETERS


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


def _v2_shaped_state_dict(model: HydroCore) -> dict[str, torch.Tensor]:
    """A real v3 model's state dict, with source_region_head reshaped down
    to what a v2 checkpoint actually had: 1 output instead of 3."""

    state_dict = copy.deepcopy(model.state_dict())
    d_model = state_dict["source_region_head.network.1.weight"].shape[1]
    state_dict["source_region_head.network.1.weight"] = torch.randn(1, d_model)
    state_dict["source_region_head.network.1.bias"] = torch.randn(1)
    return state_dict


def test_v2_shaped_checkpoint_migrates_and_reports_exactly_the_reshaped_keys() -> None:
    trained = _tiny_model()
    v2_state = _v2_shaped_state_dict(trained)
    fresh = _tiny_model()

    migrated, dropped = load_state_dict_with_v2_migration(fresh, v2_state)

    assert migrated is True
    assert set(dropped) == V2_TO_V3_RESHAPED_PARAMETERS


def test_v2_migration_preserves_every_other_parameter_exactly() -> None:
    trained = _tiny_model()
    v2_state = _v2_shaped_state_dict(trained)
    fresh = _tiny_model()

    load_state_dict_with_v2_migration(fresh, v2_state)

    for name, tensor in fresh.state_dict().items():
        if name in V2_TO_V3_RESHAPED_PARAMETERS:
            continue
        torch.testing.assert_close(tensor, trained.state_dict()[name])
    # The reshaped head itself must NOT have loaded the (shape-incompatible,
    # never-meaningfully-trained) v2 weights -- it keeps its fresh v3 init.
    assert fresh.state_dict()["source_region_head.network.1.weight"].shape == (3, 32)


def test_a_v3_checkpoint_loads_normally_without_reporting_a_migration() -> None:
    trained = _tiny_model()
    fresh = _tiny_model()

    migrated, dropped = load_state_dict_with_v2_migration(fresh, trained.state_dict())

    assert migrated is False
    assert dropped == ()
    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, trained.state_dict()[name])


def test_an_unrelated_shape_mismatch_still_fails_closed() -> None:
    """The migration path must not become a general strict=False escape
    hatch: any mismatch beyond the one known, understood v2 reshape must
    still raise, exactly as strict loading would."""

    trained = _tiny_model()
    state_dict = _v2_shaped_state_dict(trained)
    # Corrupt an unrelated parameter's shape too.
    state_dict["source_node_head.network.1.weight"] = torch.randn(2, 32)
    fresh = _tiny_model()

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_state_dict_with_v2_migration(fresh, state_dict)


def test_a_solely_unrelated_shape_mismatch_also_fails_closed() -> None:
    trained = _tiny_model()
    state_dict = copy.deepcopy(trained.state_dict())
    state_dict["source_node_head.network.1.weight"] = torch.randn(2, 32)
    fresh = _tiny_model()

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_state_dict_with_v2_migration(fresh, state_dict)
