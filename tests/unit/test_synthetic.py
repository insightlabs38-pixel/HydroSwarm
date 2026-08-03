from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from hydroswarm.data.synthetic import SyntheticConfig, generate_synthetic_data
from hydroswarm.simulation import network as network_module
from hydroswarm.simulation.network import build_networkx_network, build_wntr_network


def test_networkx_network_has_required_hydraulic_elements() -> None:
    graph = build_networkx_network()

    assert isinstance(graph, nx.MultiDiGraph)
    assert nx.is_weakly_connected(graph)
    assert graph.nodes["R1"]["node_type"] == "reservoir"
    assert graph.nodes["T1"]["node_type"] == "tank"
    assert all("elevation_m" in graph.nodes[name] for name in ("J1", "J2", "J3", "J4", "T1"))
    assert len(graph.graph["demand_pattern"]) == 24
    assert all(graph.nodes[name]["base_demand_m3s"] > 0 for name in ("J1", "J2", "J3", "J4"))


def test_wntr_network_is_valid_and_matches_topology() -> None:
    pytest.importorskip("wntr")
    model = build_wntr_network()

    assert set(model.reservoir_name_list) == {"R1"}
    assert set(model.tank_name_list) == {"T1"}
    assert set(model.junction_name_list) == {"J1", "J2", "J3", "J4"}
    assert len(model.pipe_name_list) == 7
    assert list(model.get_pattern("diurnal").multipliers) == pytest.approx(model.get_pattern("diurnal").multipliers)
    assert model.get_node("J3").elevation == pytest.approx(107.0)


def test_wntr_absence_has_actionable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(network_module, "wntr", None)
    monkeypatch.setattr(network_module, "_WNTR_IMPORT_ERROR", ImportError("not installed"))

    with pytest.raises(ImportError, match="use build_networkx_network"):
        network_module.build_wntr_network()


def test_synthetic_data_is_deterministic_aligned_and_imperfect() -> None:
    graph = build_networkx_network()
    config = SyntheticConfig(seed=17, periods=72, missing_probability=0.12)

    first = generate_synthetic_data(graph, config)
    second = generate_synthetic_data(graph, config)

    pd.testing.assert_frame_equal(first.sensor_readings, second.sensor_readings)
    pd.testing.assert_frame_equal(first.contamination, second.contamination)
    assert first.contamination.shape == (72, 5)
    assert first.sensor_readings.index.equals(first.contamination.index)
    assert first.missing_mask.to_numpy().any()
    assert first.sensor_readings.isna().equals(first.missing_mask)
    assert first.flow_reversal_mask.to_numpy().any()
    assert first.flow_reversal_mask.dtypes.eq(bool).all()
    assert first.noise.abs().to_numpy().max() > 0
    assert first.drift.abs().iloc[-1].max() > first.drift.abs().iloc[1].max()
    assert first.transport_delay_steps["J1"] == 0
    assert first.transport_delay_steps["J3"] > first.transport_delay_steps["J2"]
    assert first.contamination["J1"].idxmax() < first.contamination["J3"].idxmax()
    assert first.demand["J2"].max() > first.demand["J2"].min()


def test_synthetic_data_accepts_wntr_and_exports_tidy_records() -> None:
    pytest.importorskip("wntr")
    dataset = generate_synthetic_data(
        build_wntr_network(),
        SyntheticConfig(seed=8, periods=12, missing_probability=0.0),
    )

    frame = dataset.to_long_frame()
    assert len(frame) == 12 * 5
    assert {
        "timestamp",
        "node_id",
        "true_concentration",
        "sensor_reading",
        "demand_m3s",
        "is_missing",
        "flow_reversal",
        "transport_delay_steps",
    }.issubset(frame.columns)
    assert not frame["is_missing"].any()

