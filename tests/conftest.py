"""Root conftest: enforces that only tests marked ``real_simulation`` (see
pyproject.toml) ever make a real HydraulicSimulator/WNTR/EPANET call.

Windows CI's broad job (``pytest -m "not real_simulation"``) is only cheap
because that set is audited to contain ZERO real simulator calls -- not
merely believed to. This wraps ``HydraulicSimulator._run_with_timeout`` for
the whole session and, via a ``pytest_runtest_protocol`` hookwrapper (not a
regular fixture -- a function-scoped autouse fixture sets up AFTER any
module/session-scoped fixtures the test depends on, so it would miss real
calls that happen during THEIR setup, e.g. a shared corpus-building fixture
used by several tests), attributes any real call made during setup, call,
or teardown to whichever test item was running. A future test -- or a
fixture several tests share -- that starts exercising the real simulator
cannot silently slip into the "broad" set and reintroduce the pathological
Windows spawn-overhead regression this audit exists to prevent (every real
WNTR/EPANET call there pays a fresh interpreter startup, no fork() on
Windows).

Violations are collected across the whole session and reported together at
the end (rather than failing each test individually, which would race
against a hookwrapper timing that no longer has a clean way to inject a
per-test failure after its own report already went out) -- a single clear,
aggregated summary that fails the run.

The dedicated Windows real-simulator smoke file
(``test_windows_simulator_smoke.py``) is deliberately exempt: it stays
unmarked on purpose so it remains separately selectable by path (ignored in
the broad job, run as its own CI step), not by marker exclusion.
"""

from __future__ import annotations

import pytest

_SMOKE_FILE_BASENAME = "test_windows_simulator_smoke.py"
_state = {"called": False}
_violations: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    from hydroswarm.simulation import wrapper

    original = wrapper.HydraulicSimulator._run_with_timeout

    def _instrumented(self, operation, function, args=()):
        _state["called"] = True
        return original(self, operation, function, args)

    wrapper.HydraulicSimulator._run_with_timeout = _instrumented


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    # A true wrapper around this item's entire setup+call+teardown -- unlike
    # a function-scoped fixture, this resets/checks around ANY fixture this
    # item depends on, at whatever scope, since fixture setup for an item
    # happens inside the yielded default protocol below.
    _state["called"] = False
    yield
    if not _state["called"]:
        return
    if item.get_closest_marker("real_simulation") is not None:
        return
    if _SMOKE_FILE_BASENAME in str(item.fspath):
        return
    _violations.append(item.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _violations:
        return
    lines = [
        "",
        "=" * 78,
        "REAL-SIMULATION MARKER AUDIT FAILED",
        "=" * 78,
        f"{len(_violations)} test(s) made a real HydraulicSimulator call (via "
        "_run_with_timeout -- including possibly during a shared module/"
        "session-scoped fixture's setup) but are not marked "
        "@pytest.mark.real_simulation:",
    ]
    lines += [f"  - {nodeid}" for nodeid in _violations]
    lines += [
        "",
        'Windows CI\'s broad job assumes `-m "not real_simulation"` contains '
        "ZERO real simulator calls. Add the marker (and "
        "@pytest.mark.full_simulation too if this test makes >=10 real "
        "calls) to each test above -- or, if the real call happens in a "
        "fixture several tests share, to every test that depends on that "
        "fixture -- so it doesn't silently reintroduce the pathological "
        "spawn-overhead regression this audit exists to prevent.",
        "=" * 78,
    ]
    message = "\n".join(lines)
    terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_line(message, red=True, bold=True)
    else:
        print(message)
    session.exitstatus = 1
