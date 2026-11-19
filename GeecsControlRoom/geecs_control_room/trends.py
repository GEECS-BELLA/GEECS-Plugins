"""A device's trend history, from Vole (Canopy's live buffer), for the settables page's trends.

``GET /api/devices/{name}/trends?seconds=<window>[&axis=<axis>]`` serves the
past of the plots a family's panel names (``Panel.trends``): for each trend
role the device serves, the PV the module's device route names (never
derived here, never taken from the request) is subscribed in Vole
(``POST /vole/v1/pvs/subscribe/{pv}``: Vole buffers it from then on) and its
samples over the window are read (``GET /vole/v1/pvs/{pv}?from=&to=&maxPoints=``),
with this process's read key. The page draws them and extends each line
from Hare's stream; nothing here judges, averages or fills a value.

- **200**: ``{"device", "from", "to", "roles": {<role>: {"pv", "samples",
  "why"}}}``. Times are milliseconds since the epoch (Vole's are integer
  nanoseconds, more than a browser's number holds exactly); ``to`` is this
  server's clock, which the page takes as "now" so a viewer's own clock
  never shifts the plot. ``samples`` is ``[[t, value], ...]``, oldest first,
  a value ``null`` where the record was undefined, INVALID or no finite number: a gap
  in the line, never a zero. ``why`` is ``null``, or the words for a role
  with no history (Vole silent or refusing, nothing buffered yet): the page
  then starts that line at page load and says so.
- **404**: the device's family names no trend this device serves. The
  module's own refusals and silence keep the device pages' statuses
  (``device_pages.unread``).

Vole silent is never an error of this route: the rest of the page does not
wait on it (``VOLE_TIMEOUT_S`` per call, every role concurrently).
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Optional
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from geecs_control_room.device_pages import (
    DEFAULT_WINDOW_S,
    WINDOWS_S,
    classify,
    unread,
)
from geecs_control_room.module_client import ModuleUnavailable

#: One call to Vole: a subscribe or a history read. The page's plots wait at most two of these.
VOLE_TIMEOUT_S = 3.0
#: The most samples asked of Vole per PV and window.
MAX_POINTS = 4000


class VoleUnavailable(Exception):
    """Vole could not be reached, refused, or answered with something that is no history."""


class VoleClient:
    """Vole's buffer routes on Canopy (``/vole/v1``) with the display's read key: subscribe and read, nothing else."""

    def __init__(
        self,
        canopy_url: str,
        read_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=canopy_url.rstrip("/") + "/vole/v1",
            headers={"X-API-Key": read_key},
            timeout=VOLE_TIMEOUT_S,
            transport=transport,
        )

    async def subscribe(self, pv: str) -> None:
        """Ask Vole to buffer ``pv`` (already buffered: the same answer)."""
        try:
            r = await self._http.post(f"/pvs/subscribe/{quote(pv, safe='')}")
        except httpx.HTTPError as exc:
            raise VoleUnavailable(f"Vole did not answer: {exc!r}") from exc
        if r.status_code != 200:
            raise VoleUnavailable(f"Vole answered {r.status_code} to the subscribe")

    async def history(self, pv: str, from_ns: int, to_ns: int, max_points: int = MAX_POINTS) -> list[dict[str, Any]]:  # fmt: skip
        """Vole's samples of ``pv`` between two times (integer nanoseconds since the epoch), as it gave them."""
        params = {"from": from_ns, "to": to_ns, "maxPoints": max_points}
        try:
            r = await self._http.get(f"/pvs/{quote(pv, safe='')}", params=params)
            body = r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError) as exc:
            raise VoleUnavailable(f"Vole did not answer: {exc!r}") from exc
        if r.status_code != 200:
            raise VoleUnavailable(f"Vole answered {r.status_code}")
        samples = body.get("samples") if isinstance(body, dict) else None
        if not isinstance(samples, list):
            raise VoleUnavailable("Vole's answer carried no samples")
        return samples

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._http.aclose()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def points(samples: list[Any]) -> list[list[Optional[float]]]:
    """Vole's samples as the page draws them: ``[ms, value]`` oldest first, the value ``None`` unless a finite number of a record that is not INVALID (the live line's rule); a sample without a time is dropped."""
    out: list[list[Optional[float]]] = []
    for s in samples:
        if not isinstance(s, dict) or not _number(s.get("tsNs")):
            continue
        v = s.get("value") if s.get("severity") != "INVALID" else None
        out.append([s["tsNs"] / 1e6, float(v) if _number(v) and math.isfinite(v) else None])  # fmt: skip
    return sorted(out, key=lambda p: p[0])


async def _role_history(vole: VoleClient, pv: str, from_ns: int, to_ns: int) -> dict[str, Any]:  # fmt: skip
    try:
        await vole.subscribe(pv)
        found = points(await vole.history(pv, from_ns, to_ns))
    except VoleUnavailable as exc:
        return {"pv": pv, "samples": [], "why": str(exc)}
    return {"pv": pv, "samples": found, "why": None if found else "nothing buffered yet"}  # fmt: skip


def trend_pvs(body: dict[str, Any], axis: Optional[str]) -> dict[str, str]:
    """The PV of each trend role the device serves on ``axis``, as the module's device route names it; empty for a family without trends."""
    family, _, _ = classify(body)
    if family is None or family.panel is None:
        return {}
    wanted = {name for trend in family.panel.trends for name in trend.roles}
    on = (axis or None) if family.per_axis else None
    return {
        e["role"]: e["pv"]
        for e in body.get("roles") or []
        if e.get("role") in wanted and e.get("axis") == on and e.get("pv")
    }


def router(vole: VoleClient) -> APIRouter:
    """The trends' history, read only."""
    r = APIRouter()

    @r.get("/api/devices/{name}/trends")
    async def trends(request: Request, name: str, seconds: int = DEFAULT_WINDOW_S, axis: Optional[str] = None) -> Any:  # fmt: skip
        try:
            status, body = await request.app.state.module.device(name)
        except ModuleUnavailable as exc:
            status, body = 0, {"detail": str(exc)}
        if why := unread(status, body, name):
            return JSONResponse({"code": "unavailable", "detail": f"{why.title}: {why.detail}"}, status_code=why.status)  # fmt: skip
        pvs = trend_pvs(body, axis)
        if not pvs:
            return JSONResponse({"code": "not_found", "detail": f"{name} serves no trend role"}, status_code=404)  # fmt: skip
        to_ns = time.time_ns()
        from_ns = to_ns - min(max(seconds, 1), max(WINDOWS_S)) * 1_000_000_000
        found = await asyncio.gather(*(_role_history(vole, pv, from_ns, to_ns) for pv in pvs.values()))  # fmt: skip
        return JSONResponse(
            {
                "device": name,
                "from": from_ns / 1e6,
                "to": to_ns / 1e6,
                "roles": dict(zip(pvs, found)),
            },  # fmt: skip
            headers={"Cache-Control": "no-store"},
        )

    return r
