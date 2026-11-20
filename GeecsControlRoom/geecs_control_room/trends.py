"""A device's trend history, from Vole (Canopy's live buffer), for the settables page's trends.

``GET /api/devices/{name}/trends?seconds=<window>[&axis=<axis>]`` serves the
past of the plots a family's panel names (``Panel.trends``): for each trend
role the device serves, the PV the module's device route names (never
derived here, never taken from the request) is subscribed in Vole
(``POST /vole/v1/pvs/subscribe/{pv}``: Vole buffers it from then on) and its
samples over the window are read (``GET /vole/v1/pvs/{pv}?from=&to=&maxPoints=``),
with this process's read key. The page draws them and extends each line
from Hare's stream; nothing here judges, averages or fills a value.

- **200**: ``{"device", "from", "to", "now", "roles": {<role>: {"pv",
  "samples", "state", "why"}}}``. Times are milliseconds since the epoch
  (Vole's are integer nanoseconds, more than a browser's number holds
  exactly). ``from`` and ``to`` are the window asked of Vole; ``now`` is
  this server's clock when the answer was built, after Vole's calls, which
  the page takes as "now" so neither a viewer's own clock nor a slow Vole
  shifts the plot. ``samples`` is ``[[t, value], ...]``, oldest first, a
  value ``null`` where the record was undefined, INVALID or no finite
  number: a gap in the line, never a zero. ``state`` is ``history`` (there
  are samples), ``starting`` (subscribed and read, none yet: Vole has only
  now begun to buffer the PV) or ``failed`` (Vole silent or refusing, with
  nothing to show); ``why`` is ``null`` or the words for the last two. The
  page starts a line without history at page load and says which it is.
- **404**: the device's family names no trend this device serves. The
  module's own refusals and silence keep the device pages' statuses
  (``device_pages.unread``).

Vole silent is never an error of this route, and the rest of the page does
not wait on it: a role's subscribe and its history read are each given
``VOLE_TIMEOUT_S`` (two phases, so twice that at most), every role
concurrently. The history is read whatever the subscribe said (a PV another
viewer subscribed has a past even if this subscribe failed); the
subscribe's failure is reported only when there is no history to show.

Vole as observed on 2026-11-18, against Canopy at the fleet's pin (the
route relies on each; none is a documented contract here):

- A repeated ``POST subscribe`` answers 200.
- The value a PV held on entering the window comes back as a first sample
  stamped exactly at ``from`` (a PV unchanged for minutes, read over a 20 s
  window: one sample, at ``from``). So a buffered PV never answers empty,
  and an empty answer means "not buffered until now".
- ``maxPoints=N`` thins: N real samples spread across the whole window
  (the first at ``from``, the last near ``to``), neither the oldest N nor
  averages. A busy PV's past is therefore a thinned one, of real values.
- ``severity`` is a word: ``NO_ALARM``, ``INVALID``.
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

#: One phase of a role's read: its subscribe, then its history read, each at most this long.
VOLE_TIMEOUT_S = 3.0
#: The most samples asked of Vole per PV and window (Vole thins a busier past to this many real samples).
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
        if not r.is_success:
            raise VoleUnavailable(f"Vole answered {r.status_code} to the subscribe")

    async def history(self, pv: str, from_ns: int, to_ns: int, max_points: int = MAX_POINTS) -> list[dict[str, Any]]:  # fmt: skip
        """Vole's samples of ``pv`` between two times (integer nanoseconds since the epoch), as it gave them."""
        params = {"from": from_ns, "to": to_ns, "maxPoints": max_points}
        try:
            r = await self._http.get(f"/pvs/{quote(pv, safe='')}", params=params)
            body = r.json() if r.is_success else None
        except (httpx.HTTPError, ValueError) as exc:
            raise VoleUnavailable(f"Vole did not answer: {exc!r}") from exc
        if not r.is_success:
            raise VoleUnavailable(f"Vole answered {r.status_code}")
        samples = body.get("samples") if isinstance(body, dict) else None
        if not isinstance(samples, list):
            raise VoleUnavailable("Vole's answer carried no samples")
        return samples

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._http.aclose()


def _finite(value: Any) -> Optional[float]:
    """``value`` as a finite float; ``None`` for anything else (a word, a boolean, NaN, a whole number too large for a float)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def points(samples: list[Any]) -> list[list[Optional[float]]]:
    """Vole's samples as the page draws them: ``[ms, value]`` oldest first, the value ``None`` unless a finite number of a record that is not INVALID (the live line's rule); a sample without a time is dropped."""
    out: list[list[Optional[float]]] = []
    for s in samples:
        t = _finite(s.get("tsNs")) if isinstance(s, dict) else None
        if t is None:
            continue
        out.append([t / 1e6, None if s.get("severity") == "INVALID" else _finite(s.get("value"))])  # fmt: skip
    return sorted(out, key=lambda p: p[0])


async def _role_history(vole: VoleClient, pv: str, from_ns: int, to_ns: int) -> dict[str, Any]:  # fmt: skip
    """One role's past: subscribed, then read whatever the subscribe said; a failure is words only when there is nothing to show."""
    refused: Optional[str] = None
    try:
        await vole.subscribe(pv)
    except VoleUnavailable as exc:
        refused = str(exc)
    try:
        found = points(await vole.history(pv, from_ns, to_ns))
    except VoleUnavailable as exc:
        return {"pv": pv, "samples": [], "state": "failed", "why": refused or str(exc)}  # fmt: skip
    if found:
        return {"pv": pv, "samples": found, "state": "history", "why": None}
    if refused:
        return {"pv": pv, "samples": [], "state": "failed", "why": refused}
    return {"pv": pv, "samples": [], "state": "starting", "why": "not buffered until now"}  # fmt: skip


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
                "now": time.time_ns() / 1e6,
                "roles": dict(zip(pvs, found)),
            },  # fmt: skip
            headers={"Cache-Control": "no-store"},
        )

    return r
