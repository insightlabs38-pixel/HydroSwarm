from __future__ import annotations

import torch

from hydroswarm.model import HydroCore


def _tiny_model() -> HydroCore:
    return HydroCore(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        adapter_dims=(32, 32, 32),
    )


def test_forward_handles_missing_sensors_and_node_masks() -> None:
    model = _tiny_model().eval()
    temporal = torch.randn(2, 3, 4, 2)
    temporal[0, 1, 1] = torch.nan
    quality = torch.randn(2, 3, 4, 2)
    quality[1, :, 2] = torch.nan
    node_mask = torch.tensor([[True, True, False, True], [True, True, True, False]])
    sensor_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    sensor_mask[0, :, 3] = False

    with torch.no_grad():
        output = model(
            {
                "node_features": torch.randn(2, 4, 3),
                "temporal_features": temporal,
                "quality_features": quality,
                "travel_time": torch.tensor([[0.0, 5.0, 10.0, 20.0], [0.0, 1.0, 2.0, 3.0]]),
                "reservoir_reachability": torch.tensor([[1.0, 1.0, 0.8, 0.5], [1.0, 0.5, 0.0, 0.0]]),
                "demand_centrality": torch.rand(2, 4),
                "node_mask": node_mask,
                "sensor_mask": sensor_mask,
            }
        )

    assert output["hidden_state"].shape == (2, 4, 32)
    assert output["sentinel"].shape == (2, 4, 2)
    assert output["scout"].shape == (2, 4, 2)
    assert output["strategist"].shape == (2, 4, 3)
    assert torch.isfinite(output["hidden_state"]).all()
    assert torch.count_nonzero(output["hidden_state"][~node_mask]) == 0
    assert torch.count_nonzero(output["strategist"][~node_mask]) == 0


def test_all_missing_observations_are_safe() -> None:
    model = _tiny_model().eval()
    with torch.no_grad():
        output = model(
            {
                "node_features": torch.zeros(1, 2, 3),
                "temporal_features": torch.full((1, 2, 2, 2), torch.nan),
                "quality_features": torch.full((1, 2, 2, 2), torch.nan),
                "travel_time": torch.zeros(1, 2),
                "reservoir_reachability": torch.zeros(1, 2),
                "demand_centrality": torch.zeros(1, 2),
                "sensor_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
                "quality_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
            }
        )
    assert torch.isfinite(output["sentinel"]).all()


def test_default_model_is_approximately_24_5m_parameters() -> None:
    model = HydroCore()
    report = model.parameter_report()
    assert report.total == model.parameter_count()
    assert report.trainable == report.total
    assert abs(report.total - 24_500_000) / 24_500_000 < 0.05
