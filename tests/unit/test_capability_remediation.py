"""Regression coverage for the capability-remediation contracts."""

from __future__ import annotations

import pytest
import wntr

from hydroswarm.calibration import CalibrationExample, SplitConformalCalibrator
from hydroswarm.calibration.conformal import classify_runtime_condition
from hydroswarm.classical.signature_policy import governed_network_family
from hydroswarm.data.scenarios import network_sha256
from hydroswarm.preprocessing.builder import SensorSeries
from hydroswarm.simulation import build_wntr_network


def _series(
    *,
    missing: tuple[bool, ...] = (False,),
    frozen: tuple[bool, ...] = (False,),
    drift: tuple[bool, ...] = (False,),
    delayed: tuple[bool, ...] = (False,),
) -> SensorSeries:
    steps = len(missing)
    return SensorSeries(
        node_id="J1",
        timestamps_seconds=tuple(float(index * 3600) for index in range(steps)),
        concentration_mg_l=tuple(0.1 for _ in range(steps)),
        pressure_m=tuple(25.0 for _ in range(steps)),
        health=tuple(1.0 for _ in range(steps)),
        missing=missing,
        frozen=frozen,
        drift=drift,
        delayed=delayed,
    )


def test_structural_identity_ignores_runtime_demand_and_tank_state() -> None:
    network = build_wntr_network()
    baseline = network_sha256(network)
    network.get_node("J1").demand_timeseries_list[0].base_value *= 1.25
    network.get_node("T1").init_level += 1.0
    assert network_sha256(network) == baseline


def test_static_link_change_changes_canonical_identity() -> None:
    network = build_wntr_network()
    baseline = network_sha256(network)
    network.get_link("P_J1_J2").roughness += 0.01
    assert network_sha256(network) != baseline


def test_epanet_round_trip_preserves_governed_structural_identity(tmp_path) -> None:
    network = build_wntr_network()
    baseline = network_sha256(network)
    path = tmp_path / "canonical-golden.inp"
    wntr.network.write_inpfile(network, str(path))
    reparsed = wntr.network.WaterNetworkModel(str(path))
    assert network_sha256(reparsed) == baseline


def test_governed_family_is_hash_based_not_display_name() -> None:
    network = build_wntr_network()
    assert governed_network_family(network_sha256(network)) == "golden-reference"
    network.name = "golden-reference"
    network.get_link("P_J1_J2").roughness += 0.01
    assert governed_network_family(network_sha256(network)) is None


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (_series(), "CLEAN"),
        (_series(delayed=(True,)), "OPERATIONAL"),
        (_series(missing=(True,)), "DEGRADED"),
        (_series(frozen=(True,)), "DEGRADED"),
        (_series(drift=(True,)), "DEGRADED"),
    ],
)
def test_runtime_condition_uses_evidence_semantics(series: SensorSeries, expected: str) -> None:
    assert classify_runtime_condition((series,)) == expected


def test_calibration_selection_reports_network_condition_and_global_groups() -> None:
    examples = (
        CalibrationExample((0.9, 0.1), 0, "CLEAN", "golden-reference"),
        CalibrationExample((0.8, 0.2), 0, "CLEAN", "golden-reference"),
        CalibrationExample((0.6, 0.4), 0, "DEGRADED", "other"),
        CalibrationExample((0.7, 0.3), 0, "DEGRADED", "other"),
    )
    calibrator = SplitConformalCalibrator.fit(
        examples,
        alpha=0.1,
        model_hash="m",
        feature_schema_hash="f",
        dataset_manifest_hash="d",
        minimum_group_size=2,
    )
    assert calibrator.selection(network_id="golden-reference", condition="DEGRADED")[:2] == (
        "NETWORK_SPECIFIC",
        "golden-reference",
    )
    assert calibrator.selection(network_id="unknown", condition="DEGRADED")[:2] == (
        "CONDITION_SPECIFIC",
        "DEGRADED",
    )
    assert calibrator.selection(network_id="unknown", condition="unknown")[:2] == ("GLOBAL", "global")
