from __future__ import annotations

import sys

import pytest

from hydroswarm.training import job_runner


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_launch_writes_running_status_and_log(tmp_path) -> None:
    run_dir = tmp_path / "job-a"
    handle = job_runner.launch(
        _python("import time; time.sleep(0.05); print('hello-from-job')"),
        run_dir=run_dir,
        workdir=tmp_path,
    )
    status = handle.status()
    assert status["state"] == "RUNNING"
    assert status["pid"] == handle.pid
    assert status["resume_command"] is None

    exit_code = _wait_for_exit(handle.pid)
    job_runner.mark_finished(run_dir, exit_code=exit_code)
    final = handle.status()
    assert final["state"] == "COMPLETED"
    assert final["exit_code"] == 0
    assert "hello-from-job" in handle.log_path.read_text(encoding="utf-8")


def _wait_for_exit(pid: int, timeout: float = 5.0) -> int:
    # os.waitpid's os.WNOHANG flag is POSIX-only (not defined on Windows),
    # so a manual non-blocking poll loop around it isn't cross-platform.
    # psutil.Process.wait() is: it uses the real OS reap primitive on POSIX
    # and WaitForSingleObject on Windows, and returns the exit code either
    # way.
    import psutil

    try:
        return psutil.Process(pid).wait(timeout=timeout)
    except psutil.TimeoutExpired:
        raise TimeoutError("job did not exit in time") from None


def test_overlapping_launch_in_same_run_dir_is_rejected(tmp_path) -> None:
    run_dir = tmp_path / "job-b"
    handle = job_runner.launch(
        _python("import time; time.sleep(2)"),
        run_dir=run_dir,
        workdir=tmp_path,
    )
    try:
        with pytest.raises(job_runner.JobRunnerError, match="already RUNNING"):
            job_runner.launch(
                _python("print('should not start')"),
                run_dir=run_dir,
                workdir=tmp_path,
            )
    finally:
        handle.terminate(grace_period_seconds=1.0)


def test_launch_refuses_to_start_below_disk_guard(tmp_path, monkeypatch) -> None:
    import shutil

    class _Usage:
        free = 1 * 1024**3  # 1 GiB

    monkeypatch.setattr(shutil, "disk_usage", lambda _path: _Usage())
    with pytest.raises(job_runner.JobRunnerError, match="free disk"):
        job_runner.launch(
            _python("print('unreachable')"),
            run_dir=tmp_path / "job-c",
            workdir=tmp_path,
            min_free_disk_gb=5.0,
        )


def test_terminate_sends_sigterm_then_escalates_and_marks_terminated(tmp_path) -> None:
    run_dir = tmp_path / "job-d"
    handle = job_runner.launch(
        _python(
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(30)\n"
        ),
        run_dir=run_dir,
        workdir=tmp_path,
    )
    state = handle.terminate(grace_period_seconds=0.5)
    assert state == "TERMINATED"
    assert handle.status()["state"] == "TERMINATED"


def test_attach_reconnects_to_an_existing_job(tmp_path) -> None:
    run_dir = tmp_path / "job-e"
    original = job_runner.launch(
        _python("import time; time.sleep(0.05)"),
        run_dir=run_dir,
        workdir=tmp_path,
    )
    reattached = job_runner.attach(run_dir)
    assert reattached.pid == original.pid
    _wait_for_exit(original.pid)
    job_runner.mark_finished(run_dir, exit_code=0)
    assert reattached.status()["state"] == "COMPLETED"


def test_poll_detects_a_crashed_job_whose_pid_died_without_terminal_status(tmp_path) -> None:
    run_dir = tmp_path / "job-f"
    handle = job_runner.launch(
        _python("import sys; sys.exit(1)"),
        run_dir=run_dir,
        workdir=tmp_path,
    )
    _wait_for_exit(handle.pid)
    # No mark_finished() call: simulate a supervisor that never reaped the
    # child, so status.json is stuck at RUNNING even though the pid is gone.
    assert handle.poll() == "CRASHED"
