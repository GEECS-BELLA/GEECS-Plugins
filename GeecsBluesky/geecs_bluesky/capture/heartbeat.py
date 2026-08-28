"""The capture daemon's liveness heartbeat — the toggle-off safety signal.

The engine suppresses native image saving blind to whether the capture
daemon is actually running; this module closes that gap (the HARD Phase-6
precondition recorded in ``Planning/data_capture/01_central_pva_capture_scope.md``):

- the daemon calls :func:`write_heartbeat` every ``HEARTBEAT_PERIOD_S``
  from a side thread while its dispatcher runs;
- the engine calls :func:`heartbeat_age` in its pre-claim preflight and
  **refuses a toggle-off scan** when the heartbeat is missing or older
  than ``STALE_AFTER_S`` — fail-closed, because the alternative is a scan
  whose camera images exist nowhere.

The heartbeat is a small JSON file on the filesystem the daemon and the
worker already share (the daemon must see the worker's save paths — an
existing deployment constraint, so this adds no new one). Default location
under the user state dir; override via ``[capture] heartbeat_path`` in the
shared GEECS config.

What a fresh heartbeat proves — and doesn't: the daemon *process* is alive
and its heartbeat thread is running. It does not prove the 0MQ document
subscription is healthy (the dispatcher reconnects on its own, and between
scans no documents flow, so there is no cheap doc-side freshness signal)
nor that PVA delivery will succeed — those residual gaps stay covered by
the daemon's per-scan reconciliation counters.
"""

from __future__ import annotations

import configparser
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

HEARTBEAT_PERIOD_S = 10.0
STALE_AFTER_S = 30.0

_USER_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")


def heartbeat_path() -> Path:
    """The heartbeat file location: ``[capture] heartbeat_path`` else default."""
    cfg_path = _USER_CONFIG_PATH.expanduser()
    if cfg_path.exists():
        parser = configparser.ConfigParser()
        try:
            parser.read(cfg_path)
            override = parser.get("capture", "heartbeat_path", fallback="").strip()
            if override:
                return Path(override).expanduser()
        except configparser.Error:
            logger.warning("Could not parse %s for [capture]", cfg_path)
    return Path("~/.local/state/geecs-capture/heartbeat.json").expanduser()


def write_heartbeat(n_targets: int, *, path: Path | None = None) -> None:
    """Write/refresh the heartbeat (atomic replace; creates its own dir).

    The heartbeat's directory is daemon state, never scan data — creating
    it does not touch the scan-folder invariant.
    """
    path = path or heartbeat_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": time.time(), "pid": os.getpid(), "targets": int(n_targets)}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def heartbeat_age(*, path: Path | None = None) -> float | None:
    """Seconds since the last heartbeat, or ``None`` when absent/unreadable."""
    path = path or heartbeat_path()
    try:
        payload = json.loads(path.read_text())
        stamp = float(payload["time"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return max(0.0, time.time() - stamp)


def daemon_looks_alive(*, path: Path | None = None) -> bool:
    """The engine-side verdict: a heartbeat exists and is fresh."""
    age = heartbeat_age(path=path)
    return age is not None and age <= STALE_AFTER_S
