"""``run_process_group``: subprocess.run whose timeout kills the whole worker tree, and is a bound."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from scan_analysis.services._subprocess import run_process_group

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="process groups are POSIX"
)


def _gone(pid: int, within: float = 5.0) -> bool:
    """Whether ``pid`` no longer exists (or is a zombie) within ``within`` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        if not state or state.startswith("Z"):
            return True
        time.sleep(0.05)
    return False


def test_it_returns_what_subprocess_run_returns():
    done = run_process_group(["sh", "-c", "echo out; echo err >&2; exit 2"], timeout=10)
    assert (done.returncode, done.stdout, done.stderr) == (2, "out\n", "err\n")


def test_a_timeout_kills_a_grandchild_in_the_group(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    with pytest.raises(subprocess.TimeoutExpired):
        run_process_group(
            ["sh", "-c", f'sleep 300 & echo $! > "{pidfile}"; wait'], timeout=1
        )
    assert _gone(int(pidfile.read_text()))


def test_the_timeout_stays_a_bound_when_an_escaped_process_holds_a_pipe():
    """A process outside the group (wineserver, daemonized) may keep stderr open.

    Draining the pipes after the kill would wait for it; the review of #1242
    reproduced a 1 s timeout returning after 6 s.
    """
    holder = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(6)'], "
        "start_new_session=True); time.sleep(300)"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_process_group([sys.executable, "-c", holder], timeout=1)
    assert time.monotonic() - started < 4
