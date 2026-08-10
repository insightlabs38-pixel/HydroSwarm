"""SUB-11 (submission.txt task list): offline audit. HydroSwarm's README and
docs repeatedly claim "runtime operation makes no internet calls" -- this
test makes that claim mechanically checked rather than merely asserted in
prose. It blocks every outbound (non-loopback) TCP connection attempt at
the socket layer for the duration of a real self-test + a real FastAPI
request cycle, and fails loudly if anything tries to reach the network.

Loopback (127.0.0.1 / ::1, used by the app's own dev-server binding and by
WNTR/EPANET's own subprocess plumbing) remains allowed; only genuinely
external destinations are blocked.
"""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

from hydroswarm.api import create_app
from hydroswarm.cli import run_self_test

LOOPBACK_ADDRESSES = {"127.0.0.1", "::1", "localhost"}


class OutboundConnectionBlocked(AssertionError):
    pass


@pytest.fixture
def block_outbound_network(monkeypatch: pytest.MonkeyPatch):
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address, *args, **kwargs):  # noqa: ANN001
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOOPBACK_ADDRESSES:
            raise OutboundConnectionBlocked(
                f"runtime code attempted an outbound connection to {address!r} -- "
                "HydroSwarm's runtime must never make a network call outside loopback"
            )
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.mark.real_simulation
def test_self_test_makes_no_outbound_network_call(block_outbound_network) -> None:
    report = run_self_test()
    assert report["ok"] is True


@pytest.mark.real_simulation
def test_api_health_and_reference_demo_make_no_outbound_network_call(block_outbound_network) -> None:
    client = TestClient(create_app())
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/reference-demo").status_code == 200
