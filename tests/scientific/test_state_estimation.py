from hydroswarm.classical.state_estimation import HydraulicStateEstimator, OperationalTelemetry
from hydroswarm.simulation.wrapper import HydraulicState


def simulated_state() -> HydraulicState:
    return HydraulicState(
        timestamp_seconds=3600,
        pressure_m={"J1": 30.0, "J2": 28.0},
        demand_m3s={"J1": 0.001, "J2": 0.002},
        flow_m3s={"P1": 0.003},
        velocity_mps={"P1": 0.8},
    )


def test_estimator_reconciles_telemetry_and_reports_uncertainty() -> None:
    estimated = HydraulicStateEstimator(node_to_zone={"J1": "north", "J2": "north"}).estimate(
        simulated_state(),
        OperationalTelemetry(
            pressure_m={"J1": 27.0, "J2": None}, demand_m3s={"J1": 0.0015, "J2": None},
            flow_m3s={"P1": 0.0025}, tank_level_m={"T1": 4.5},
            pump_open={"PU1": False}, valve_open={"V1": True},
        ),
        model_tank_levels_m={"T1": 3.0}, model_pump_open={"PU1": True}, model_valve_open={"V1": False},
    )
    assert estimated.pressure_m["J1"].estimate == 27.0
    assert estimated.demand_m3s["J2"].estimate == 0.003
    assert estimated.zone_demand_multipliers["north"] == 1.5
    assert estimated.residuals.missing_values_imputed == 2
    assert estimated.residuals.reconciled_pumps == ("PU1",)
    assert estimated.residuals.reconciled_valves == ("V1",)
    assert 0.0 < estimated.normalized_uncertainty <= 1.0


def test_missing_telemetry_falls_back_to_model_with_wider_interval() -> None:
    estimated = HydraulicStateEstimator().estimate(
        simulated_state(), OperationalTelemetry(pressure_m={"J1": None})
    )
    item = estimated.pressure_m["J1"]
    assert item.estimate == 30.0
    assert item.lower < item.estimate < item.upper
    assert estimated.residuals.missing_values_imputed == 1
