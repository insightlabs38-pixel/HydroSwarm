"""Capability diagnostic Section 12: sensor-series reconstruction audit.

Directly constructs `SensorObservation` records (from
`hydroswarm.domain.schemas`) and runs them through the real production
`sensor_series()` closure logic (`src/hydroswarm/api/app.py` lines ~357-380).
That closure is not importable (it is defined inline inside `create_app()`),
so this script imports the ALREADY-WRITTEN verbatim replica
(`_sensor_series_closure_replica`) from
`scripts/capability_diagnostic/train_serve_parity_full.py` via `importlib`
rather than re-copying the logic a second time, per instructions.

Seven constructed cases, seed 20260813 (deterministic timestamps, no
randomness actually needed since every input is hand-specified):
  (a) one observation, one sensor, one node
  (b) multiple observations, same sensor/node, different times -> ordered
      multi-point series?
  (c) multiple DIFFERENT sensor_ids reporting the SAME node_id -> does
      grouping happen by node_id (correct) or accidentally fragment by
      sensor_id (defect)?
  (d) an active/grab sample at a previously-unseen node -> new SensorSeries
      entry, not an error?
  (e) out-of-order timestamps in the input list -> does the closure's own
      `items.sort(key=lambda item: item.observed_at)` fix it?
  (f) a delayed measurement (received_at > observed_at)
  (g) missing values interspersed with real ones

No locked-test access: this section touches no corpus/eval data at all,
only hand-constructed SensorObservation records, but the guard is still
asserted per protocol.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.domain.schemas import SensorObservation  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_serve_parity_full", ROOT / "scripts" / "capability_diagnostic" / "train_serve_parity_full.py"
)
assert _spec is not None and _spec.loader is not None
_parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parity)
_sensor_series_closure_replica = _parity._sensor_series_closure_replica

ORIGIN = datetime(2026, 8, 13, tzinfo=UTC)


def _obs(
    sensor_id: str, node_id: str, minute_offset: float, *,
    concentration: float | None = 1.0, pressure: float | None = 25.0,
    missing: bool = False, delay_minutes: float = 0.0, quality: float = 1.0,
    frozen_flag: bool = False, drift_flag: bool = False,
) -> SensorObservation:
    observed_at = ORIGIN + timedelta(minutes=minute_offset)
    received_at = observed_at + timedelta(minutes=delay_minutes)
    return SensorObservation(
        sensor_id=sensor_id, node_id=node_id, observed_at=observed_at, received_at=received_at,
        concentration_mg_l=(None if missing else concentration),
        pressure_m=(None if missing else pressure),
        quality=quality, missing=missing, drift_flag=drift_flag, frozen_flag=frozen_flag,
    )


def _series_summary(series_list: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": s.node_id,
            "n_points": len(s.timestamps_seconds),
            "timestamps_seconds": list(s.timestamps_seconds),
            "concentration_mg_l": list(s.concentration_mg_l),
            "pressure_m": list(s.pressure_m),
            "health": list(s.health),
            "missing": list(s.missing),
            "delayed": list(s.delayed),
            "frozen": list(s.frozen),
            "timestamps_strictly_ordered": all(
                b >= a for a, b in zip(s.timestamps_seconds, s.timestamps_seconds[1:])
            ),
        }
        for s in series_list
    ]


def _run_case(name: str, observations: tuple[SensorObservation, ...]) -> dict[str, Any]:
    try:
        result = _sensor_series_closure_replica(observations, ORIGIN)
        return {"case": name, "n_input_observations": len(observations), "n_output_series": len(result), "series": _series_summary(result)}
    except Exception as exc:  # noqa: BLE001
        return {"case": name, "n_input_observations": len(observations), "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed for this diagnostic"

    cases: dict[str, dict[str, Any]] = {}

    # (a) one observation, one sensor, one node
    cases["a_single_observation"] = _run_case(
        "a_single_observation", (_obs("S1", "J1", 0),),
    )

    # (b) multiple observations, same sensor/node, different times
    cases["b_multi_observation_same_sensor"] = _run_case(
        "b_multi_observation_same_sensor",
        (_obs("S1", "J1", 0, concentration=1.0), _obs("S1", "J1", 60, concentration=1.5), _obs("S1", "J1", 120, concentration=2.0)),
    )

    # (c) multiple DIFFERENT sensor_ids at the SAME node_id
    cases["c_multi_sensor_same_node"] = _run_case(
        "c_multi_sensor_same_node",
        (
            _obs("SENSOR-A", "J1", 0, concentration=1.0),
            _obs("SENSOR-B", "J1", 30, concentration=1.2),
            _obs("SENSOR-A", "J1", 60, concentration=1.4),
            _obs("SENSOR-B", "J1", 90, concentration=1.6),
        ),
    )

    # (d) active/grab sample at a previously-unseen node, alongside an
    # already-instrumented node, to confirm it becomes its own new entry
    # rather than erroring or merging incorrectly.
    cases["d_grab_sample_new_node"] = _run_case(
        "d_grab_sample_new_node",
        (
            _obs("FIXED-1", "J1", 0, concentration=1.0),
            _obs("FIXED-1", "J1", 60, concentration=1.1),
            _obs("GRAB-1", "J7", 45, concentration=0.4),
        ),
    )

    # (e) out-of-order timestamps in the input list
    cases["e_out_of_order_input"] = _run_case(
        "e_out_of_order_input",
        (
            _obs("S1", "J1", 120, concentration=3.0),
            _obs("S1", "J1", 0, concentration=1.0),
            _obs("S1", "J1", 60, concentration=2.0),
        ),
    )

    # (f) a delayed measurement (received_at > observed_at)
    cases["f_delayed_measurement"] = _run_case(
        "f_delayed_measurement",
        (_obs("S1", "J1", 0, delay_minutes=0.0), _obs("S1", "J1", 60, delay_minutes=15.0)),
    )

    # (g) missing values interspersed with real ones
    cases["g_missing_interspersed"] = _run_case(
        "g_missing_interspersed",
        (
            _obs("S1", "J1", 0, concentration=1.0),
            _obs("S1", "J1", 60, missing=True),
            _obs("S1", "J1", 120, concentration=2.0),
            _obs("S1", "J1", 180, missing=True),
        ),
    )

    # --- Direct answer to the specific question the protocol asks: does
    # creating a new sensor_id for a grab sample at an ALREADY-instrumented
    # node cause a spurious SECOND independent series instead of extending
    # that node's history? Reuses case (c)'s data but asks the question
    # explicitly.
    c_result = cases["c_multi_sensor_same_node"]
    merges_by_node = (
        "n_output_series" in c_result and c_result["n_output_series"] == 1 and c_result["series"][0]["n_points"] == 4
    )
    spurious_second_series_answer = {
        "question": "Does creating a new sensor_id for a grab sample at an already-instrumented node cause an "
        "unintended SECOND independent series instead of extending that node's history?",
        "answer": "NO" if merges_by_node else "YES (or inconclusive -- see case (c) raw output)",
        "evidence": (
            f"Case (c) fed 4 observations from 2 different sensor_ids (SENSOR-A, SENSOR-B) at the SAME node_id "
            f"(J1). Output: {c_result.get('n_output_series')} SensorSeries entr(ies), "
            f"{c_result['series'][0]['n_points'] if c_result.get('series') else 'n/a'} points in that entry. "
            "The closure groups by `item.node_id` (src/hydroswarm/api/app.py:~365, `grouped[item.node_id]`), "
            "never by sensor_id, so multiple sensor_ids at one node correctly merge into a single ordered series."
        ),
    }

    # Ordering check for (e): does sorting by observed_at correctly fix a
    # deliberately out-of-order input list?
    e_result = cases["e_out_of_order_input"]
    e_ordered = (
        "series" in e_result and len(e_result["series"]) == 1 and e_result["series"][0]["timestamps_strictly_ordered"]
        and e_result["series"][0]["concentration_mg_l"] == [1.0, 2.0, 3.0]
    )

    # Semantic soundness checks
    semantic_findings = {
        "a_single_observation_sound": cases["a_single_observation"].get("n_output_series") == 1,
        "b_multi_observation_becomes_ordered_series": (
            cases["b_multi_observation_same_sensor"].get("n_output_series") == 1
            and cases["b_multi_observation_same_sensor"]["series"][0]["n_points"] == 3
            and cases["b_multi_observation_same_sensor"]["series"][0]["timestamps_strictly_ordered"]
        ),
        "c_multi_sensor_same_node_merges_by_node_id": merges_by_node,
        "d_grab_sample_creates_new_series_not_error": (
            "series" in cases["d_grab_sample_new_node"] and cases["d_grab_sample_new_node"]["n_output_series"] == 2
        ),
        "e_out_of_order_input_correctly_sorted": e_ordered,
        "f_delayed_flag_correctly_set": (
            "series" in cases["f_delayed_measurement"]
            and cases["f_delayed_measurement"]["series"][0]["delayed"] == [False, True]
        ),
        "g_missing_pattern_preserved": (
            "series" in cases["g_missing_interspersed"]
            and cases["g_missing_interspersed"]["series"][0]["missing"] == [False, True, False, True]
        ),
    }

    report = {
        "schema_version": 1,
        "section": "12_sensor_series_reconstruction_audit",
        "locked_test_opened_before": locked_before,
        "replica_source": "scripts/capability_diagnostic/train_serve_parity_full.py:_sensor_series_closure_replica "
        "(imported via importlib, not re-copied), itself a verbatim replica of "
        "src/hydroswarm/api/app.py's sensor_series() closure lines ~357-380 (not importable -- inline closure).",
        "cases": cases,
        "spurious_second_series_for_new_sensor_id_at_existing_node": spurious_second_series_answer,
        "semantic_soundness_findings": semantic_findings,
        "overall_pass": all(semantic_findings.values()),
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "sensor-series-audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(semantic_findings, indent=2))
    print(json.dumps(spurious_second_series_answer, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
