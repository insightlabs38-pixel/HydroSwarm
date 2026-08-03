"""Streamlit incident console for offline HydroSwarm demonstrations."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import streamlit as st


def demo_console_data() -> dict[str, Any]:
    """Return a deterministic, safety-oriented console snapshot."""

    return {
        "mode": "OFFLINE DEMO",
        "network_id": "hydroswarm-demo",
        "incident_id": "3e8991a4-5b39-4891-82dc-fd090c061f33",
        "status": "APPROVAL",
        "updated_at": "2026-08-03T06:15:00+00:00",
        "candidates": [
            {"node_id": "J1", "probability": 0.46},
            {"node_id": "J2", "probability": 0.29},
            {"node_id": "J4", "probability": 0.17},
            {"node_id": "J3", "probability": 0.08},
        ],
        "coverage": {"measured": 0.92, "target": 0.90, "calibrated": True},
        "sensors": [
            {"sensor_id": "S-J1", "node_id": "J1", "health": 0.98, "staleness_s": 18, "flags": []},
            {"sensor_id": "S-J2", "node_id": "J2", "health": 0.81, "staleness_s": 44, "flags": ["drift"]},
            {"sensor_id": "S-J3", "node_id": "J3", "health": 0.64, "staleness_s": 310, "flags": ["stale"]},
            {"sensor_id": "S-T1", "node_id": "T1", "health": 0.93, "staleness_s": 25, "flags": []},
        ],
        "disagreement_js": 0.12,
        "ood_level": "CAUTION",
        "plans": [
            {
                "name": "Isolate J1 and flush J2",
                "decision": "PENDING_APPROVAL",
                "approval_pending": True,
                "consequences": {
                    "population_impacted": 420,
                    "minimum_pressure_m": 23.7,
                    "service_availability": 0.94,
                    "containment_time_minutes": 38,
                    "operation_count": 3,
                },
                "rejection_codes": [],
            },
            {
                "name": "Close P_R1_J1 immediately",
                "decision": "REJECTED",
                "approval_pending": False,
                "consequences": {
                    "population_impacted": 1_860,
                    "minimum_pressure_m": 7.4,
                    "service_availability": 0.51,
                    "containment_time_minutes": 22,
                    "operation_count": 1,
                },
                "rejection_codes": ["MINIMUM_PRESSURE", "SERVICE_AVAILABILITY"],
            },
        ],
        "audit": [
            {"sequence": 1, "timestamp": "2026-08-03T06:00:00+00:00", "actor": "sentinel", "event_type": "INCIDENT_DETECTED"},
            {"sequence": 2, "timestamp": "2026-08-03T06:04:00+00:00", "actor": "scout", "event_type": "CANDIDATES_RANKED"},
            {"sequence": 3, "timestamp": "2026-08-03T06:11:00+00:00", "actor": "strategist", "event_type": "PLANS_PROPOSED"},
            {"sequence": 4, "timestamp": "2026-08-03T06:15:00+00:00", "actor": "verifier", "event_type": "APPROVAL_REQUESTED"},
        ],
    }


def _local_api_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def load_console_data(api_url: str | None = None, *, timeout_seconds: float = 1.5) -> tuple[dict[str, Any], str | None]:
    """Load a local API snapshot, falling back to deterministic demo data."""

    api_url = api_url or os.environ.get("HYDROSWARM_API_URL")
    if not api_url:
        return demo_console_data(), None
    if not _local_api_url(api_url):
        return demo_console_data(), "Only a localhost API is allowed in offline mode."

    endpoint = api_url.rstrip("/")
    if not endpoint.endswith("/api/v1/console"):
        endpoint += "/api/v1/console"
    try:
        request = Request(endpoint, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - localhost checked above
            payload = json.load(response)
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            payload = payload["data"]
        if not isinstance(payload, dict):
            raise ValueError("console payload must be an object")
        return payload, None
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        return demo_console_data(), f"Local API unavailable ({exc}); showing frozen demo data."


def _percent(value: Any) -> str:
    return f"{100 * float(value):.1f}%"


def render_console(data: dict[str, Any], notice: str | None = None) -> None:
    """Render a console snapshot without mutating incident state."""

    st.set_page_config(page_title="HydroSwarm Console", page_icon="💧", layout="wide")
    st.title("HydroSwarm incident console")
    st.caption("Physics-first, local decision support. No cloud connection or autonomous actuation.")
    if notice:
        st.warning(notice)

    top = st.columns(5)
    top[0].metric("Runtime", data.get("mode", "OFFLINE"))
    top[1].metric("Incident", str(data.get("status", "UNKNOWN")))
    top[2].metric("Network", str(data.get("network_id", "unknown")))
    top[3].metric("Disagreement (JS)", f"{float(data.get('disagreement_js', 0.0)):.3f}")
    top[4].metric("OOD", str(data.get("ood_level", "UNKNOWN")))

    candidates, health = st.columns(2)
    with candidates:
        st.subheader("Candidate source probabilities")
        candidate_rows = data.get("candidates", [])
        st.bar_chart(
            {row["node_id"]: float(row["probability"]) for row in candidate_rows},
            horizontal=True,
        )
        coverage = data.get("coverage", {})
        measured = float(coverage.get("measured", 0.0))
        target = float(coverage.get("target", 0.0))
        st.progress(min(1.0, measured), text=f"Coverage {_percent(measured)} / target {_percent(target)}")
        st.caption("Calibrated" if coverage.get("calibrated") else "Not calibrated")
    with health:
        st.subheader("Sensor health and staleness")
        sensor_rows = []
        for sensor in data.get("sensors", []):
            sensor_rows.append(
                {
                    "sensor": sensor.get("sensor_id"),
                    "node": sensor.get("node_id"),
                    "health": _percent(sensor.get("health", 0.0)),
                    "staleness_s": sensor.get("staleness_s", 0),
                    "flags": ", ".join(sensor.get("flags", [])) or "healthy",
                }
            )
        st.dataframe(sensor_rows, use_container_width=True, hide_index=True)

    st.subheader("Verified plan consequences")
    for plan in data.get("plans", []):
        decision = str(plan.get("decision", "UNKNOWN"))
        title = f"{plan.get('name', 'Unnamed plan')} — {decision}"
        with st.expander(title, expanded=bool(plan.get("approval_pending"))):
            if plan.get("approval_pending"):
                st.warning("Operator approval pending. HydroSwarm will not execute this plan.")
            if plan.get("rejection_codes"):
                st.error("Rejected: " + ", ".join(plan["rejection_codes"]))
            consequences = plan.get("consequences") or {}
            st.dataframe(
                [{"metric": key.replace("_", " "), "value": value} for key, value in consequences.items()],
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Tamper-evident audit timeline")
    for event in data.get("audit", []):
        st.markdown(
            f"**{event.get('sequence', '?')} · {event.get('event_type', 'EVENT')}**  "
            f"{event.get('timestamp', '')} · `{event.get('actor', 'unknown')}`"
        )
    st.caption(f"Snapshot updated {data.get('updated_at', datetime.now(UTC).isoformat())}")


def main() -> None:
    data, notice = load_console_data()
    render_console(data, notice)


if __name__ == "__main__":
    main()

