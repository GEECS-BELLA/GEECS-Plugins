"""Pin ``GeecsBluesky/qserver/launch_re_manager.sh``'s exit-status mapping (#804).

bluesky-queueserver's SIGTERM handler exits 1 after a clean shutdown, so a
normal ``systemctl stop`` logged ``status=1/FAILURE``. The launcher maps
exactly that case — exit 1 after a SIGTERM — to 0, and passes every other
status through so ``Restart=on-failure`` still sees a real startup failure.

``start-re-manager`` is replaced by a stub on PATH. The stop is delivered the
way systemd's ``KillMode=control-group`` delivers it: SIGTERM to every
process at once (here, the launcher's process group).
"""

from __future__ import annotations

import errno
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the launcher is a bash script",
)

LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "GeecsBluesky"
    / "qserver"
    / "launch_re_manager.sh"
)

# The stub manager. STUB_MODE picks its behaviour:
#   term:  run until SIGTERM, then "clean up" and exit 1 (upstream's AtTerm)
#   term3: run until SIGTERM, then exit 3 (a stop that went wrong)
#   crash: exit 1 at once (a startup failure: start_manager returns 1)
STUB = """#!/usr/bin/env bash
case "$STUB_MODE" in
    term)  trap 'sleep 0.2; exit 1' TERM ;;
    term3) trap 'exit 3' TERM ;;
    crash) exit 1 ;;
esac
: > "$STUB_READY"
while :; do sleep 0.05; done
"""


@pytest.fixture
def redis_port():
    """Something answering on 127.0.0.1:6379, so the launcher skips Redis.

    A bare listening socket is enough: the launcher's probe only connects.
    If the port is taken, whatever holds it already answers.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 6379))
        sock.listen(8)
    except OSError as exc:
        sock.close()
        if exc.errno != errno.EADDRINUSE:
            pytest.skip(f"cannot provide 127.0.0.1:6379: {exc}")
        sock = None
    yield
    if sock is not None:
        sock.close()


def _launch(tmp_path: Path, mode: str) -> tuple[subprocess.Popen, Path]:
    stubbin = tmp_path / "stubbin"
    stubbin.mkdir()
    stub = stubbin / "start-re-manager"
    stub.write_text(STUB, encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    ready = tmp_path / "ready"
    env = {
        **os.environ,
        "PATH": f"{stubbin}{os.pathsep}{os.environ.get('PATH', '')}",
        "QS_DOC_PROXY": "OFF",
        "STUB_MODE": mode,
        "STUB_READY": str(ready),
    }
    proc = subprocess.Popen(
        ["bash", str(LAUNCHER)],
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc, ready


def _stop_like_systemd(proc: subprocess.Popen, ready: Path) -> int:
    deadline = time.monotonic() + 10
    while not ready.exists():
        if proc.poll() is not None or time.monotonic() > deadline:
            out, err = proc.communicate()
            pytest.fail(f"stub manager never started: {out}{err}")
        time.sleep(0.02)
    os.killpg(proc.pid, signal.SIGTERM)
    return proc.wait(timeout=10)


def test_clean_stop_exits_zero(tmp_path, redis_port):
    """The #804 case: SIGTERM → manager cleans up → exit 1 → launcher 0."""
    proc, ready = _launch(tmp_path, "term")
    assert _stop_like_systemd(proc, ready) == 0


def test_other_status_after_stop_passes_through(tmp_path, redis_port):
    """Only 1 is upstream's normal-stop status; anything else stays visible."""
    proc, ready = _launch(tmp_path, "term3")
    assert _stop_like_systemd(proc, ready) == 3


def test_startup_failure_still_fails(tmp_path, redis_port):
    """Exit 1 with no SIGTERM is a real failure — Restart=on-failure must see it."""
    proc, _ = _launch(tmp_path, "crash")
    assert proc.wait(timeout=10) == 1
