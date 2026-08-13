"""SUB-12.1 #23: real, black-box HTTP verification of a running HydroSwarm
Docker container -- the PR gate's actual test body. Talks only to the
container's published port (never `docker exec`, so this script is
identical whether it is driving a bare-metal `hydroswarm start` process
or a hardened container); the workflow YAML handles anything that
genuinely requires the Docker CLI (build, run, restart, `docker exec
... hydroswarm self-test --strict` for the exact frozen-hash check).

Subcommands:
  health              -- /api/health, /api/readiness, /api/reference-demo,
                          and the built frontend's index all respond for real.
  live-workflow        -- drives the full real LIVE example sequence (network
                          import, incident creation, real analysis, real
                          sampling recommendation, real WNTR verification,
                          real approval) through the real production API,
                          the same sequence validated in-process against
                          `TestClient` earlier in SUB-12.1 -- this is that
                          same sequence over real HTTP against a real
                          container instead. Writes the resulting incident
                          id to --state-file for the persistence check below.
  verify-persistence   -- after a `docker restart`, confirms the incident
                          written by live-workflow is still present with
                          the same real, previously-recorded status --
                          proof the SQLite volume mount actually persists,
                          not just that the container restarts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8765"
STARTUP_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 2.0


def _request(
    base_url: str,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    timeout: float = 30.0,
) -> tuple[int, Any]:
    url = f"{base_url}{path}"
    headers = {"Accept": "application/json"}
    data: bytes | None = None
    if files is not None:
        boundary = "----hydroswarm-docker-ci-boundary"
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        body = b""
        for field, (filename, content, content_type) in files.items():
            body += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
            body += content
            body += b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        data = body
    elif json_body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(json_body).encode()
    elif method == "POST":
        data = b""

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body_bytes = response.read()
            return response.status, (json.loads(body_bytes) if body_bytes else None)
    except urllib.error.HTTPError as error:
        body_bytes = error.read()
        try:
            return error.code, json.loads(body_bytes)
        except ValueError:
            return error.code, body_bytes.decode(errors="replace")


def _wait_for_health(base_url: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, _ = _request(base_url, "GET", "/api/health", timeout=5.0)
            if status == 200:
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            last_error = error
        time.sleep(POLL_INTERVAL_SECONDS)
    raise RuntimeError(f"container did not report healthy within {STARTUP_TIMEOUT_SECONDS}s: {last_error}")


def cmd_health(args: argparse.Namespace) -> int:
    base_url = args.base_url
    _wait_for_health(base_url)

    status, body = _request(base_url, "GET", "/api/health")
    assert status == 200, f"/api/health -> {status}: {body}"
    print(f"[docker-ci] GET /api/health -> 200: {body}")

    status, body = _request(base_url, "GET", "/api/readiness")
    assert status == 200, f"/api/readiness -> {status}: {body}"
    assert body.get("status") == "ready", f"/api/readiness reported status={body.get('status')!r}: {body}"
    print(f"[docker-ci] GET /api/readiness -> 200, status=ready: {body}")

    status, body = _request(base_url, "GET", "/api/reference-demo", timeout=30.0)
    assert status == 200, f"/api/reference-demo -> {status}"
    assert body is not None and len(body) > 0, "/api/reference-demo returned an empty body"
    print("[docker-ci] GET /api/reference-demo -> 200 (real artifact)")

    request = urllib.request.Request(f"{base_url}/", headers={"Accept": "text/html"})
    with urllib.request.urlopen(request, timeout=10.0) as response:
        assert response.status == 200, f"built frontend index -> {response.status}"
        html = response.read()
        assert b"<div id=\"root\">" in html or b"<div id='root'>" in html, (
            "served page does not look like the built frontend shell"
        )
    print("[docker-ci] GET / -> 200 (built frontend)")
    return 0


def cmd_live_workflow(args: argparse.Namespace) -> int:
    base_url = args.base_url
    _wait_for_health(base_url)

    status, inputs = _request(base_url, "GET", "/api/live-example-inputs", timeout=30.0)
    assert status == 200, f"/api/live-example-inputs -> {status}: {inputs}"
    print(f"[docker-ci] live-example-inputs: network={inputs['network_filename']}, "
          f"{len(inputs['candidate_signatures_mg_l'])} node signatures")

    status, network = _request(
        base_url,
        "POST",
        "/api/networks/import",
        files={
            "file": (
                inputs["network_filename"],
                inputs["network_inp_text"].encode(),
                "text/plain",
            )
        },
    )
    assert status == 201, f"import failed: {status}: {network}"
    network_id = network["network_id"]
    print(f"[docker-ci] imported real network: {network_id}")

    now = _utc_now()
    observation = inputs["initial_observation"]
    status, incident = _request(
        base_url,
        "POST",
        "/api/incidents",
        json_body={
            "network_id": network_id,
            "detected_at": now,
            "observations": [
                {
                    "sensor_id": observation["sensor_id"],
                    "node_id": observation["node_id"],
                    "observed_at": now,
                    "received_at": now,
                    "concentration_mg_l": observation["concentration_mg_l"],
                    "pressure_m": observation["pressure_m"],
                }
            ],
            "contamination_threshold_mg_l": inputs["contamination_threshold_mg_l"],
        },
    )
    assert status == 201, f"incident creation failed: {status}: {incident}"
    incident_id = incident["incident_id"]
    print(f"[docker-ci] created real incident: {incident_id}")

    status, _ = _request(base_url, "POST", f"/api/incidents/{incident_id}/analyze")
    assert status == 200, f"analyze failed: {status}"
    print("[docker-ci] real initial analysis complete")

    status, recommendation = _request(base_url, "POST", f"/api/incidents/{incident_id}/samples/recommend")
    assert status == 200, f"sample recommendation failed: {status}: {recommendation}"
    node_id = recommendation["node_id"]
    concentration = inputs["candidate_signatures_mg_l"].get(node_id)
    assert concentration is not None, f"no reference signature for recommended node {node_id!r}"
    print(f"[docker-ci] real sampling recommendation: {node_id}")

    now = _utc_now()
    status, _ = _request(
        base_url,
        "POST",
        f"/api/incidents/{incident_id}/samples",
        json_body={
            "sensor_id": f"S-{node_id}",
            "node_id": node_id,
            "observed_at": now,
            "received_at": now,
            "concentration_mg_l": concentration,
            "pressure_m": 25.0,
        },
    )
    assert status == 200, f"sample submission failed: {status}"

    status, _ = _request(base_url, "POST", f"/api/incidents/{incident_id}/analyze")
    assert status == 200, f"re-analysis failed: {status}"
    print("[docker-ci] real re-analysis complete")

    status, plans = _request(
        base_url, "POST", f"/api/incidents/{incident_id}/plans/generate", json_body={"count": 2}
    )
    assert status == 200, f"plan generation failed: {status}: {plans}"
    print(f"[docker-ci] {len(plans)} real bounded response plan(s) generated")

    verified_plan_id = None
    for plan in plans:
        status, verification = _request(
            base_url, "POST", f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify"
        )
        assert status == 200, f"plan verification failed: {status}: {verification}"
        print(f"[docker-ci] real WNTR/EPANET verification of {plan['plan_id']}: {verification['decision']}")
        if verification["decision"] == "VERIFIED" and verified_plan_id is None:
            verified_plan_id = plan["plan_id"]
            # A CURRENT VERIFIED plan moves the incident to APPROVAL.  Do
            # not verify a later candidate before this approval: a rejected
            # later verification would correctly move the incident back to
            # PLANNING and invalidate the approval boundary we are testing.
            # This gate needs one real verified plan, not a weaker
            # multi-verification lifecycle.
            break

    assert verified_plan_id is not None, "no plan was VERIFIED by real WNTR verification -- cannot reach approval"

    status, approval = _request(
        base_url,
        "POST",
        f"/api/incidents/{incident_id}/plans/{verified_plan_id}/approve",
        json_body={"approved": True, "operator_id": "docker-ci-gate"},
    )
    assert status == 200, f"approval failed: {status}: {approval}"
    print(f"[docker-ci] real human-approval recorded for {verified_plan_id}")

    status, view = _request(base_url, "GET", f"/api/incidents/{incident_id}/view")
    assert status == 200, f"final view failed: {status}: {view}"
    runtime_mode = view.get("runtime_mode") or view.get("runtimeMode")
    print(f"[docker-ci] incident view: runtime_mode={runtime_mode}")

    # /view is an operator projection and deliberately does not duplicate
    # IncidentState.status.  Read the authority-bearing lifecycle state
    # from its own endpoint before carrying it across the restart boundary.
    status, terminal_state = _request(base_url, "GET", f"/api/incidents/{incident_id}")
    assert status == 200, f"terminal incident state failed: {status}: {terminal_state}"
    terminal_status = terminal_state.get("status")
    assert terminal_status == "CLOSED", f"approved incident did not reach CLOSED: {terminal_status!r}"

    if args.state_file:
        Path(args.state_file).write_text(
            json.dumps({"incident_id": incident_id, "runtime_mode": runtime_mode, "status": terminal_status})
        )
        print(f"[docker-ci] wrote persistence-check state to {args.state_file}")

    print("[docker-ci] live-workflow PASSED")
    return 0


def cmd_verify_persistence(args: argparse.Namespace) -> int:
    base_url = args.base_url
    _wait_for_health(base_url)

    state = json.loads(Path(args.state_file).read_text())
    incident_id = state["incident_id"]

    # Incident evidence, plans, verifications, approvals, and audit events
    # are durable; hybrid analysis is deliberately in-memory.  The workflow
    # approves its selected plan before restart, making CLOSED terminal.
    # Re-analysis would therefore be an invalid lifecycle transition rather
    # than a persistence check.  Check the persisted terminal state and the
    # persisted audit chain directly instead.
    status, incident = _request(base_url, "GET", f"/api/incidents/{incident_id}")
    assert status == 200, (
        f"incident {incident_id} not found after restart (status {status}) -- "
        "the data volume did not actually persist"
    )
    assert incident.get("status") == state["status"], (
        f"incident status changed across restart: was {state['status']!r}, now {incident.get('status')!r}"
    )

    status, replay = _request(base_url, "POST", f"/api/incidents/{incident_id}/replay")
    assert status == 200, f"persisted incident replay failed after restart: {status}: {replay}"
    assert replay.get("chain_valid") is True, "persisted incident audit chain is invalid after restart"
    event_types = {event.get("event_type") for event in replay.get("events", [])}
    assert "PLAN_APPROVED" in event_types, "persisted audit chain lacks the approval event"
    print(f"[docker-ci] incident {incident_id} survived restart with status={incident['status']} and a valid audit chain")
    print("[docker-ci] verify-persistence PASSED")
    return 0


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health")

    live_workflow_parser = subparsers.add_parser("live-workflow")
    live_workflow_parser.add_argument("--state-file", default=None)

    persistence_parser = subparsers.add_parser("verify-persistence")
    persistence_parser.add_argument("--state-file", required=True)

    args = parser.parse_args(argv)
    handlers = {
        "health": cmd_health,
        "live-workflow": cmd_live_workflow,
        "verify-persistence": cmd_verify_persistence,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"[docker-ci] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"[docker-ci] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
