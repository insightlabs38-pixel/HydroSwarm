from hydroswarm.sensors.health import (
    SensorFault,
    SensorTelemetry,
    analyze_sensor_health,
    classify_sensor_faults,
)


def test_detects_frozen_drift_jitter_and_communication_faults() -> None:
    frozen = analyze_sensor_health(SensorTelemetry(
        sensor_id="S1", node_id="J1", timestamps_seconds=(0, 300, 1200, 1500),
        values=(5.0, 5.0, None, 5.0), expected_values=(1.0, 1.0, 1.0, 1.0),
        calibration_age_days=500, expected_range=(0.0, 10.0),
    ))
    assert SensorFault.FROZEN in frozen.faults
    assert SensorFault.TIMESTAMP_JITTER in frozen.faults
    assert SensorFault.COMMUNICATION_INTERRUPTION in frozen.faults
    assert SensorFault.BIAS in frozen.faults
    assert SensorFault.CALIBRATION_OVERDUE in frozen.faults
    assert frozen.confidence < 0.5
    assert classify_sensor_faults([frozen])["S1"] == "probable_sensor_fault"


def test_correlated_change_is_classified_as_network_event() -> None:
    report = analyze_sensor_health(SensorTelemetry(
        sensor_id="S2", node_id="J2", timestamps_seconds=(0, 300, 600),
        values=(1.0, 2.0, 3.0), expected_values=(1.0, 1.0, 1.0), expected_range=(0.0, 10.0),
    ))
    assert classify_sensor_faults([report], cross_sensor_event_correlation={"S2": 0.9})["S2"] == "probable_network_event"
