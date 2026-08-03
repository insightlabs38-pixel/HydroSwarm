"""Deterministic live fault features before learned sensor-fault correction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

import numpy as np


class SensorFault(StrEnum):
    FROZEN = "FROZEN"
    TIMESTAMP_JITTER = "TIMESTAMP_JITTER"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    COMMUNICATION_INTERRUPTION = "COMMUNICATION_INTERRUPTION"
    BIAS = "BIAS"
    DRIFT = "DRIFT"
    CALIBRATION_OVERDUE = "CALIBRATION_OVERDUE"


@dataclass(frozen=True, slots=True)
class SensorTelemetry:
    sensor_id: str
    node_id: str
    timestamps_seconds: tuple[float, ...]
    values: tuple[float | None, ...]
    expected_values: tuple[float | None, ...] = ()
    calibration_age_days: float = 0.0
    expected_interval_seconds: float = 300.0
    expected_range: tuple[float, float] = (0.0, 100.0)

    def __post_init__(self) -> None:
        if not self.sensor_id or len(self.timestamps_seconds) != len(self.values):
            raise ValueError("sensor telemetry must be named and time/value aligned")
        if self.expected_values and len(self.expected_values) != len(self.values):
            raise ValueError("expected values must align with observations")


@dataclass(frozen=True, slots=True)
class SensorHealthReport:
    sensor_id: str
    node_id: str
    confidence: float
    faults: tuple[SensorFault, ...]
    frozen_score: float
    jitter_score: float
    interruption_score: float
    bias_estimate: float
    drift_per_hour: float
    model_residual_rmse: float
    recommendation: str


def analyze_sensor_health(
    telemetry: SensorTelemetry,
    *,
    frozen_tolerance: float = 1e-9,
    jitter_fraction: float = 0.35,
    calibration_limit_days: float = 365.0,
) -> SensorHealthReport:
    timestamps = np.asarray(telemetry.timestamps_seconds, dtype=float)
    if timestamps.size == 0 or not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
        raise ValueError("sensor timestamps must be finite and ordered")
    valid = np.asarray([value is not None and math.isfinite(value) for value in telemetry.values])
    values = np.asarray([float(value) if ok else np.nan for value, ok in zip(telemetry.values, valid, strict=True)])
    finite_values = values[valid]
    frozen_score = 0.0
    if finite_values.size >= 3:
        frozen_score = float(np.mean(np.abs(np.diff(finite_values)) <= frozen_tolerance))
    intervals = np.diff(timestamps)
    jitter_score = float(np.mean(
        np.abs(intervals - telemetry.expected_interval_seconds)
        > telemetry.expected_interval_seconds * jitter_fraction
    )) if intervals.size else 0.0
    missing = 1.0 - float(valid.mean()) if valid.size else 1.0
    gap_score = float(np.mean(intervals > 2.5 * telemetry.expected_interval_seconds)) if intervals.size else 0.0
    interruption = max(missing, gap_score)
    lower, upper = telemetry.expected_range
    unit_mismatch = bool(finite_values.size and np.mean((finite_values < lower) | (finite_values > upper)) > 0.5)

    bias = 0.0
    drift_per_hour = 0.0
    rmse = 0.0
    if telemetry.expected_values:
        expected = np.asarray([
            float(value) if value is not None and math.isfinite(value) else np.nan
            for value in telemetry.expected_values
        ])
        comparable = valid & np.isfinite(expected)
        residual = values[comparable] - expected[comparable]
        if residual.size:
            bias = float(residual.mean())
            rmse = float(np.sqrt(np.mean(residual**2)))
            if residual.size >= 3:
                hours = (timestamps[comparable] - timestamps[comparable][0]) / 3600.0
                if np.ptp(hours) > 0:
                    drift_per_hour = float(np.polyfit(hours, residual, 1)[0])

    scale = max(upper - lower, 1e-6)
    faults: list[SensorFault] = []
    if frozen_score >= 0.8:
        faults.append(SensorFault.FROZEN)
    if jitter_score >= 0.25:
        faults.append(SensorFault.TIMESTAMP_JITTER)
    if unit_mismatch:
        faults.append(SensorFault.UNIT_MISMATCH)
    if interruption >= 0.25:
        faults.append(SensorFault.COMMUNICATION_INTERRUPTION)
    if abs(bias) > 0.1 * scale:
        faults.append(SensorFault.BIAS)
    if abs(drift_per_hour) > 0.01 * scale:
        faults.append(SensorFault.DRIFT)
    if telemetry.calibration_age_days > calibration_limit_days:
        faults.append(SensorFault.CALIBRATION_OVERDUE)
    penalty = (
        0.25 * frozen_score + 0.15 * jitter_score + 0.25 * interruption
        + 0.15 * min(1.0, abs(bias) / scale) + 0.15 * min(1.0, abs(drift_per_hour) / scale)
        + 0.25 * float(unit_mismatch) + 0.1 * float(SensorFault.CALIBRATION_OVERDUE in faults)
    )
    confidence = max(0.0, min(1.0, 1.0 - penalty))
    recommendation = (
        "quarantine_sensor_and_inspect_units" if unit_mismatch
        else "inspect_or_recalibrate_sensor" if faults
        else "continue_monitoring"
    )
    return SensorHealthReport(
        sensor_id=telemetry.sensor_id, node_id=telemetry.node_id, confidence=confidence,
        faults=tuple(faults), frozen_score=frozen_score, jitter_score=jitter_score,
        interruption_score=interruption, bias_estimate=bias, drift_per_hour=drift_per_hour,
        model_residual_rmse=rmse, recommendation=recommendation,
    )


def classify_sensor_faults(
    reports: Sequence[SensorHealthReport],
    *,
    cross_sensor_event_correlation: Mapping[str, float] | None = None,
) -> Mapping[str, str]:
    """Distinguish isolated sensor faults from coherent network events."""
    classifications: dict[str, str] = {}
    for report in reports:
        correlation = (cross_sensor_event_correlation or {}).get(report.sensor_id, 0.0)
        if correlation >= 0.7 and SensorFault.UNIT_MISMATCH not in report.faults:
            classifications[report.sensor_id] = "probable_network_event"
        elif report.faults:
            classifications[report.sensor_id] = "probable_sensor_fault"
        else:
            classifications[report.sensor_id] = "healthy_or_ambiguous"
    return classifications

