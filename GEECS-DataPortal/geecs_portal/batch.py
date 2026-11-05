"""The portal's link to the analysis worker (``scan-analysis-worker``, its own service).

The day page's "Analyze this day" panel asks the worker, through the
portal (``/api/batch/...``, :mod:`geecs_portal.routes.batch`), for the
analysis groups, a day's plan, and to run, follow and cancel jobs.  The
portal runs no batch itself and imports nothing of the worker: the edge is
HTTP, as with the logbook, so a batch outlives a portal restart and the
portal's memory never carries a day's analysis.

``--analysis-worker-url`` names the worker.  ``--analysis-worker-autostart``
(development) lets the portal start one on this host when the URL is a
loopback address and nothing answers: ``python -m scan_analysis.worker``
from this interpreter over the portal's ``--processing-configs`` tree, in
its own session so it keeps running when the portal stops.  A deployment
runs the worker as its own unit and never passes the flag.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

__all__ = ["AutoStart", "WorkerLink", "WorkerUnavailable"]

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
#: how long a started worker gets to answer /health
_START_SECONDS = 30.0
_TIMEOUT = httpx.Timeout(30.0, connect=2.0)


class WorkerUnavailable(RuntimeError):
    """The worker did not answer (and could not be started)."""


@dataclasses.dataclass(frozen=True)
class AutoStart:
    """How to start a worker on this host when none answers (development)."""

    config_dir: Path
    experiment: str


@dataclasses.dataclass
class WorkerLink:
    """The worker's address, and optionally how to start it."""

    base_url: str
    autostart: Optional[AutoStart] = None
    client: Optional[httpx.Client] = None
    _started: Optional[subprocess.Popen] = dataclasses.field(default=None, repr=False)
    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, repr=False
    )

    def __post_init__(self) -> None:
        """Normalize the base URL (no trailing slash)."""
        self.base_url = self.base_url.rstrip("/")

    @property
    def can_autostart(self) -> bool:
        """Autostart is on and the URL is this host's."""
        host = urlparse(self.base_url).hostname or ""
        return self.autostart is not None and host in _LOOPBACK

    def call(self, method: str, path: str, *, json: Any = None) -> tuple[int, Any]:
        """``(status, body)`` of one request; starts the worker first if allowed.

        Raises :class:`WorkerUnavailable` when nothing answers.
        """
        try:
            return self._send(method, path, json)
        except httpx.TransportError as exc:
            if not self.can_autostart:
                raise WorkerUnavailable(
                    f"the analysis worker at {self.base_url} does not answer ({exc})"
                ) from exc
        self._start()
        try:
            return self._send(method, path, json)
        except httpx.TransportError as exc:
            raise WorkerUnavailable(
                f"the analysis worker at {self.base_url} does not answer ({exc})"
            ) from exc

    def reachable(self) -> bool:
        """Whether the worker answers /health now (never starts it)."""
        try:
            return self._send("GET", "/health", None)[0] == 200
        except httpx.TransportError:
            return False

    # ------------------------------------------------------------ internals

    def _send(self, method: str, path: str, json: Any) -> tuple[int, Any]:
        url = self.base_url + path
        if self.client is not None:
            response = self.client.request(method, url, json=json)
        else:
            with httpx.Client(timeout=_TIMEOUT) as client:
                response = client.request(method, url, json=json)
        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text}
        return response.status_code, body

    def _start(self) -> None:
        """Start a worker on this host and wait for it to answer."""
        assert self.autostart is not None
        with self._lock:
            if self._started is None or self._started.poll() is not None:
                port = urlparse(self.base_url).port or 8600
                log = Path(tempfile.gettempdir()) / "scan-analysis-worker.log"
                command = [
                    sys.executable,
                    "-m",
                    "scan_analysis.worker",
                    "--configs",
                    str(self.autostart.config_dir),
                    "--experiment",
                    self.autostart.experiment,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ]
                logger.info("starting the analysis worker: %s (log %s)", command, log)
                with log.open("ab") as out:
                    # its own session: it keeps running when the portal stops
                    self._started = subprocess.Popen(
                        command,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
            end = time.monotonic() + _START_SECONDS
            while time.monotonic() < end:
                if self.reachable():
                    return
                if self._started.poll() is not None:
                    break
                time.sleep(0.25)
        raise WorkerUnavailable(
            "the analysis worker did not start (see "
            f"{Path(tempfile.gettempdir()) / 'scan-analysis-worker.log'})"
        )
