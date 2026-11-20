"""The portal's link to the analysis worker (``scan-analysis-worker``, its own service).

The day page's "Analyze this day" panel asks the worker, through the
portal (``/api/batch/...``, :mod:`geecs_portal.routes.batch`), for the
analysis groups, a day's plan, and to run, follow and cancel jobs.  The
portal runs no batch itself and imports nothing of the worker: the edge is
HTTP, as with the logbook, so a batch outlives a portal restart and the
portal's memory never carries a day's analysis.

``--analysis-worker-url`` names the worker, and the portal never starts
one: the worker is its own process (its own unit on a services host, a
second terminal on a laptop), so a worker that does not answer is a 503
here, not something to repair.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import httpx

__all__ = ["WorkerLink", "WorkerUnavailable"]

_TIMEOUT = httpx.Timeout(30.0, connect=2.0)


class WorkerUnavailable(RuntimeError):
    """The worker did not answer."""


@dataclasses.dataclass
class WorkerLink:
    """The worker's address."""

    base_url: str
    client: Optional[httpx.Client] = None

    def __post_init__(self) -> None:
        """Normalize the base URL (no trailing slash)."""
        self.base_url = self.base_url.rstrip("/")

    def call(self, method: str, path: str, *, json: Any = None) -> tuple[int, Any]:
        """``(status, body)`` of one request.

        Raises :class:`WorkerUnavailable` when nothing answers.
        """
        try:
            return self._send(method, path, json)
        except httpx.TransportError as exc:
            raise WorkerUnavailable(
                f"the analysis worker at {self.base_url} does not answer ({exc})"
            ) from exc

    def reachable(self) -> bool:
        """Whether the worker answers /health now."""
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
