"""The GEECS device module's routes on Canopy (``/geecs/v1``), over HTTP.

A thin client: the module's answers pass through as it gave them, its
``{"code", "detail"}`` envelope included, so the page shows the module's
own words. A put is sent once and never retried: a timeout is not a
refusal, and only the module's journal knows whether it went out.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

#: Long enough for a settling put (the module waits up to 30 s for a move's callback).
PUT_TIMEOUT_S, READ_TIMEOUT_S = 45.0, 5.0


class ModuleUnavailable(Exception):
    """The module could not be reached, or answered with something that is not JSON."""


class ModuleClient:
    """Reads and puts through ``/geecs/v1`` with the display's read key."""

    def __init__(
        self,
        canopy_url: str,
        read_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=canopy_url.rstrip("/") + "/geecs/v1",
            headers={"X-API-Key": read_key},
            timeout=READ_TIMEOUT_S,
            transport=transport,
        )

    async def _send(self, method: str, path: str, **kw: Any) -> tuple[int, Any]:
        try:
            r = await self._http.request(method, path, **kw)
            return r.status_code, r.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModuleUnavailable(f"{method} {path}: {exc!r}") from exc

    async def get(self, path: str, **params: Any) -> tuple[int, Any]:
        """``GET /geecs/v1<path>``: the module's status and body."""
        return await self._send("GET", path, params=params or None)

    async def devices(self) -> tuple[int, Any]:
        """The roster: ``{experiment, devices: [{name, family, axes, state}]}``."""
        return await self.get("/devices")

    async def device(self, name: str) -> tuple[int, Any]:
        """One device: its state, heartbeat and every role with its verdict."""
        return await self.get(f"/devices/{quote(name, safe='')}")

    async def journal(self, entries: int = 50) -> tuple[int, Any]:
        """The newest journal rows, oldest first."""
        return await self.get("/journal", entries=entries)

    async def put(self, name: str, body: dict[str, Any]) -> tuple[int, Any]:
        """One write, sent once: ``{role, axis?, value, operator, confirm?}``."""
        path = f"/devices/{quote(name, safe='')}/put"
        return await self._send("POST", path, json=body, timeout=PUT_TIMEOUT_S)

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._http.aclose()
