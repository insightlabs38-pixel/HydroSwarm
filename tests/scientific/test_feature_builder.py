
from hydroswarm.classical.state_estimation import HydraulicStateEstimator, OperationalTelemetry
from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder, SensorSeries
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.simulation.wrapper import HydraulicSimulator


def test_feature_builder_aligns_complete_native_schema() -> None:
    network = build_wntr_network()
    simulator = HydraulicSimulator(network)
    raw = simulator.calculate_state(3600)
    estimated = HydraulicStateEstimator().estimate(raw, OperationalTelemetry())
    graph = simulator.build_dynamic_graph(estimated.as_hydraulic_state())
    series = [SensorSeries(
        node_id="J1", timestamps_seconds=(0.0, 3600.0), concentration_mg_l=(0.0, 0.2),
        pressure_m=(30.0, 29.0), health=(1.0, 0.9), missing=(False, False),
        drift=(False, True), delayed=(False, False),
    )]
    built = HydraulicFeatureBuilder().build(
        network, graph, estimated, series,
        classical_prior={name: 1 / len(network.node_name_list) for name in network.node_name_list},
    )
    assert built.batch["node_features"].shape == (1, 6, 19)
    assert built.batch["edge_features"].shape[-1] == 13
    assert built.batch["temporal_features"].shape[-1] == 6
    assert built.batch["quality_features"].shape[-1] == 4
    assert built.batch["classical_prior"].shape == (1, 6)
    assert built.feature_schema_hash
