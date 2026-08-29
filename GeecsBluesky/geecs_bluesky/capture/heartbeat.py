"""The capture daemon's liveness heartbeat — the toggle-off safety signal.

The engine suppresses native image saving blind to whether the capture
daemon is actually running; this module closes that gap (the HARD Phase-6
precondition recorded in ``Planning/data_capture/01_central_pva_capture_scope.md``):

- the daemon calls :func:`write_heartbeat` every ``HEARTBEAT_PERIOD_S``
  from a side thread while its dispatcher runs, and removes the file on
  clean shutdown (:func:`clear_heartbeat`) so a ``systemctl stop`` refuses
  toggle-off scans *immediately* instead of after the stale window;
- the engine calls :func:`read_heartbeat` in its pre-claim preflight and
  **refuses a toggle-off scan** when the heartbeat is missing, older than
  ``STALE_AFTER_S``, or does not cover every requested capture device —
  fail-closed, because the alternative is a scan whose camera images
  exist nowhere.

The heartbeat is a small JSON file local to the daemon's host and service
user — the engine-side check assumes it runs on the SAME host as the
daemon under the same account (true for the systemd deployment, where the
worker and daemon share one service user; a headless ``run_scan_request``
on a different machine cannot see a healthy daemon's heartbeat and will
be refused). Default location under the user state dir; override via
``[capture] heartbeat_path`` in the shared GEECS config (must be an
absolute path — a relative one would resolve against each process's CWD
and split the writer from the reader, so it is ignored with a warning).

What a fresh heartbeat proves — and doesn't: the daemon *process* is alive
and its heartbeat thread is running, monitoring the ``targets`` named in
the payload. It does not prove the 0MQ document subscription is healthy
(the dispatcher reconnects on its own, and between scans no documents
flow, so there is no cheap doc-side freshness signal) nor that PVA
delivery will succeed — those residual gaps stay covered by the daemon's
per-scan reconciliation counters. A daemon that dies WITHOUT running its
cleanup (SIGKILL, power loss) still takes up to ``STALE_AFTER_S`` to read
as down.
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
                candidate = Path(override).expanduser()
                if candidate.is_absolute():
                    return candidate
                logger.warning(
                    "[capture] heartbeat_path %r is not absolute — a relative "
                    "path resolves against each process's CWD and splits the "
                    "daemon from the engine; using the default instead",
                    override,
                )
        except configparser.Error:
            logger.warning("Could not parse %s for [capture]", cfg_path)
    return Path("~/.local/state/geecs-capture/heartbeat.json").expanduser()


def write_heartbeat(targets: list[str], *, path: Path | None = None) -> None:
    """Write/refresh the heartbeat (atomic replace; creates its own dir).

    *targets* is the list of device names the daemon is monitoring — the
    engine's preflight checks coverage against it, so a daemon started
    before a camera joined the roster reads as not-covering that camera.
    The heartbeat's directory is daemon state, never scan data — creating
    it does not touch the scan-folder invariant.
    """
    path = path or heartbeat_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": time.time(),
        "pid": os.getpid(),
        "targets": sorted(str(t) for t in targets),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def clear_heartbeat(*, path: Path | None = None) -> None:
    """Remove the heartbeat on clean daemon shutdown (best-effort).

    Makes an intentional stop (``systemctl stop``, Ctrl-C) refuse
    toggle-off scans immediately instead of leaving a ``STALE_AFTER_S``
    window during which the preflight would still pass.
    """
    path = path or heartbeat_path()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not remove heartbeat %s", path, exc_info=True)


def read_heartbeat(*, path: Path | None = None) -> dict | None:
    """The heartbeat payload, or ``None`` when absent/unreadable/corrupt."""
    path = path or heartbeat_path()
    try:
        payload = json.loads(path.read_text())
        float(payload["time"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return payload


def heartbeat_age(*, path: Path | None = None) -> float | None:
    """Seconds since the last heartbeat, or ``None`` when absent/unreadable."""
    payload = read_heartbeat(path=path)
    if payload is None:
        return None
    return max(0.0, time.time() - float(payload["time"]))


def daemon_looks_alive(*, path: Path | None = None) -> bool:
    """The engine-side verdict: a heartbeat exists and is fresh."""
    age = heartbeat_age(path=path)
    return age is not None and age <= STALE_AFTER_S
