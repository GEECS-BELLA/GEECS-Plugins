"""Run a vendor worker under a launcher (Wine) so that a timeout reaches its whole tree.

The FROG and WaveKit launchers start a Windows interpreter behind a launcher
(``env WINEDEBUG=-all wine``). On Wine 9 the launcher (``start.exe /exec``)
starts that interpreter as a child of its own, so ``subprocess.run``'s
timeout, which kills only the direct child, left the interpreter running and
holding its memory, once per timed-out shot.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from typing import Optional, Sequence

__all__ = ["run_process_group"]


def run_process_group(
    cmd: Sequence[str],
    *,
    capture_output: bool = True,
    text: bool = True,
    timeout: Optional[float] = None,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """``subprocess.run`` whose timeout kills the worker's whole process group.

    The worker runs as the leader of its own session. On a timeout, or any
    other interruption, the group is killed and the leader reaped, and the
    exception propagates (``subprocess.TimeoutExpired`` for a timeout).

    The pipes are not drained after the kill. A process that left the group
    may still hold one: the Wine prefix's ``wineserver`` daemonizes into a
    session of its own when a ``wine`` call starts it. Reading to EOF would
    then wait for that process, and the timeout would no longer be a bound.
    """
    pipe = subprocess.PIPE if capture_output else None
    with subprocess.Popen(
        list(cmd), stdout=pipe, stderr=pipe, text=text, env=env, start_new_session=True
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except BaseException:
            _kill_group(proc)
            proc.wait()
            raise
    return subprocess.CompletedProcess(list(cmd), proc.returncode, stdout, stderr)


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill ``proc``'s process group (POSIX), else ``proc`` alone."""
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        return
    proc.kill()
