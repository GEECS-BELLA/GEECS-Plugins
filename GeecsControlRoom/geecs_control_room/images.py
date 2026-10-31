"""A camera's image and arrays, read over Channel Access by this server: read only, at the PVs the module names.

Hare carries scalars; a camera's frame and its plots' arrays do not travel
through it (``Planning/device_pages.md``, slice 1). Two routes serve them:

- ``GET /api/devices/{name}/frame.jpg``: the JPEG the IOC's MJPEG server
  holds. Its URL is the string at the PV the module's device route names
  for role ``image`` (``read: once``), read as a char string, its host
  replaced with the address CA reached the IOC at (lab DNS names the IOC
  host; a client off it cannot resolve that). One fetch in flight per
  device, a frame reused for everyone within ``FRAME_EVERY_S``; the
  ``X-Frame-Tag`` names the bytes, so a page never decodes the same frame
  twice (a frozen camera's frame is not "new").
- ``GET /api/devices/{name}/arrays``: every ``read: on_request`` role of the
  device (lineouts, histogram) by role name, read at most every
  ``ARRAYS_EVERY_S`` and shared by every viewer. The page polls it only
  while its plots are open.

PV names come from the module's device route only, never derived here; the
route's answer is reused for ``ROUTE_TTL_S``, one read in flight per device,
and the last good answer is served while the module is silent. CA addressing is the standard
``EPICS_CA_*`` environment (``site.env``'s keys, loaded by the unit). The
transports are injected (:class:`Ca`, ``fetch``) so the logic runs in tests
without a CA library or an IOC; :class:`AiocaCa` is the real one. No write
ever leaves this module: every write is the module's put route.
"""

from __future__ import annotations

import asyncio
import math
import time
import zlib
from collections.abc import Awaitable, Callable
from typing import Any, Generic, Optional, Protocol, TypeVar
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from geecs_control_room.module_client import ModuleClient, ModuleUnavailable

#: The module's device route is reused this long (the roster's own cadence).
ROUTE_TTL_S = 10.0
#: The floor between two fetches of one camera's frame, however many viewers poll (10 fps).
FRAME_EVERY_S = 0.1
#: The plots' arrays are read at most this often per device.
ARRAYS_EVERY_S = 0.5
#: One CA read or one frame fetch: the IOC answers in milliseconds.
CA_TIMEOUT_S = FETCH_TIMEOUT_S = 3.0

T = TypeVar("T")


class Unavailable(Exception):
    """The route cannot serve: ``status`` (404 the module names no such thing, 503 a source is silent) and why."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


class Ca(Protocol):
    """The three CA reads this module makes."""

    async def text(self, pv: str) -> str:
        """``pv`` read as a char string (a ``JPG_URL_RBV`` waveform)."""
        ...

    async def host(self, pv: str) -> str:
        """The ``host:port`` CA reached ``pv``'s server at."""
        ...

    async def arrays(self, pvs: list[str]) -> list[list[float]]:
        """Each of ``pvs`` read as numbers, in order."""
        ...


class AiocaCa:
    """:class:`Ca` over aioca, imported on first use (libca reads ``EPICS_CA_*`` then)."""

    async def text(self, pv: str) -> str:
        """``pv`` as a char string."""
        import aioca

        return str(await aioca.caget(pv, datatype=aioca.DBR_CHAR_STR, timeout=CA_TIMEOUT_S))  # fmt: skip

    async def host(self, pv: str) -> str:
        """The address CA reached ``pv`` at."""
        import aioca

        return str((await aioca.cainfo(pv, timeout=CA_TIMEOUT_S)).host)

    async def arrays(self, pvs: list[str]) -> list[list[float]]:
        """Each of ``pvs`` as numbers."""
        import aioca

        values = await aioca.caget(pvs, timeout=CA_TIMEOUT_S)
        return [[float(x) for x in v] for v in values]


def with_ca_host(url: str, ca_host: str) -> str:
    """Replace ``url``'s host with the IOC's CA address (``host:port``), keeping the HTTP port.

    >>> with_ca_host("http://camserv:8081/CAM1.MJPG.jpg", "10.0.0.5:5064")
    'http://10.0.0.5:8081/CAM1.MJPG.jpg'
    """
    parts = urlsplit(url)
    ip = ca_host.rsplit(":", 1)[0]
    return urlunsplit(parts._replace(netloc=f"{ip}:{parts.port}" if parts.port else ip))


class Shared(Generic[T]):
    """One value read for everyone: reused within ``every`` of the read's start, one read in flight.

    The read runs as a task of its own that every caller awaits through a
    shield, so a viewer going away never cancels the read the others wait
    for. A failure is not kept: the next caller reads again.
    """

    def __init__(self, read: Callable[[], Awaitable[T]], every: float, clock: Callable[[], float] = time.monotonic) -> None:  # fmt: skip
        self._read, self._every, self._clock = read, every, clock
        self._value: Optional[T] = None
        self._at = -math.inf
        self._task: Optional[asyncio.Task[T]] = None

    async def get(self) -> T:
        """The held value while young enough, else the one read in flight."""
        if self._value is not None and self._clock() - self._at < self._every:
            return self._value
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run(self._clock()))  # fmt: skip
            self._task.add_done_callback(lambda t: None if t.cancelled() else t.exception())  # fmt: skip
        return await asyncio.shield(self._task)

    async def _run(self, asked: float) -> T:
        try:
            value = await self._read()
            self._value, self._at = value, asked
            return value
        finally:
            self._task = None


def _finite(values: list[float]) -> list[Optional[float]]:
    """The values with NaN and ±inf as ``None``: the browser's ``JSON.parse`` rejects them."""
    return [v if math.isfinite(v) else None for v in values]


class Images:
    """Every camera's frame and arrays, each shared by its viewers; the module's PV names, cached."""

    def __init__(self, ca: Ca, fetch: Callable[[str], Awaitable[bytes]], clock: Callable[[], float] = time.monotonic) -> None:  # fmt: skip
        self._ca, self._fetch, self._clock = ca, fetch, clock
        self._routes: dict[str, Shared[dict[str, dict[str, Any]]]] = {}
        self._good: dict[str, dict[str, dict[str, Any]]] = {}
        self._frames: dict[tuple[str, str], Shared[tuple[bytes, str]]] = {}
        self._urls: dict[str, str] = {}
        self._arrays: dict[tuple[str, tuple[str, ...]], Shared[dict[str, Any]]] = {}

    async def roles(self, module: ModuleClient, name: str) -> dict[str, dict[str, Any]]:  # fmt: skip
        """The device's roles by name as the module's device route gave them (at most ``ROUTE_TTL_S`` old).

        One read in flight per device, shared by every waiter (a slow module
        holds up only its own device's first viewers, never another camera).
        While the module is silent or failing, the last good answer is served
        and held for another ``ROUTE_TTL_S``: PV names do not change between
        reads, and a frame from a healthy IOC need not wait on the module. Any
        other 4xx (off the roster, the key refused) forgets it.
        """
        source = self._routes.get(name)
        if source is None:
            source = self._routes[name] = Shared(lambda: self._read_roles(module, name), ROUTE_TTL_S, self._clock)  # fmt: skip
        return await source.get()

    async def _read_roles(self, module: ModuleClient, name: str) -> dict[str, dict[str, Any]]:  # fmt: skip
        good = self._good.get(name)
        try:
            status, body = await module.device(name)
        except ModuleUnavailable as exc:
            if good is not None:
                return good
            raise Unavailable(503, f"the module did not answer: {exc}") from exc
        if status == 404:
            self._good.pop(name, None)
            raise Unavailable(404, f"{name} is not on the module's roster")
        if status != 200 or not isinstance(body, dict):
            if status >= 500 and good is not None:  # the module failing: as silent
                return good
            if (
                status < 500
            ):  # a refusal (the key revoked): forgotten, as the page's denied pane
                self._good.pop(name, None)
            raise Unavailable(503, f"the module answered {status} for {name}")
        roles = {r["role"]: r for r in body.get("roles") or [] if r.get("pv") and not r.get("axis")}  # fmt: skip
        self._good[name] = roles
        return roles

    async def frame(self, module: ModuleClient, name: str) -> tuple[bytes, str]:
        """The latest JPEG and its tag (the bytes' checksum)."""
        role = (await self.roles(module, name)).get("image")
        if role is None:
            raise Unavailable(404, f"the module names no image role for {name}")
        pv = role["pv"]
        source = self._frames.get((name, pv))
        if source is None:
            source = self._frames[(name, pv)] = Shared(lambda: self._read_frame(pv), FRAME_EVERY_S, self._clock)  # fmt: skip
        return await source.get()

    async def _read_frame(self, pv: str) -> tuple[bytes, str]:
        url = self._urls.get(pv)
        if url is None:
            try:
                url = self._urls[pv] = with_ca_host(await self._ca.text(pv), await self._ca.host(pv))  # fmt: skip
            except Exception as exc:  # any CA failure: said, never a 500
                raise Unavailable(503, f"the image URL ({pv}) could not be read: {exc}") from exc  # fmt: skip
        try:
            jpeg = await self._fetch(url)
        except Exception as exc:
            self._urls.pop(pv, None)  # read the URL again: the IOC may have restarted
            raise Unavailable(503, f"no frame from the IOC: {exc}") from exc
        return jpeg, f"{zlib.crc32(jpeg):08x}"

    async def arrays(self, module: ModuleClient, name: str) -> dict[str, Any]:
        """Every ``read: on_request`` role's array by role name, and the PVs read."""
        pvs = {r: e["pv"] for r, e in (await self.roles(module, name)).items() if e.get("read") == "on_request"}  # fmt: skip
        if not pvs:
            raise Unavailable(404, f"the module names no on-request array for {name}")
        key = (name, tuple(sorted(pvs.items())))
        source = self._arrays.get(key)
        if source is None:
            source = self._arrays[key] = Shared(lambda: self._read_arrays(pvs), ARRAYS_EVERY_S, self._clock)  # fmt: skip
        return await source.get()

    async def _read_arrays(self, pvs: dict[str, str]) -> dict[str, Any]:
        try:
            values = await self._ca.arrays(list(pvs.values()))
        except Exception as exc:  # any CA failure: said, never a 500
            raise Unavailable(503, f"the arrays could not be read: {exc}") from exc
        return {"arrays": {r: _finite(v) for r, v in zip(pvs, values)}, "pvs": pvs}


class HttpFetch:
    """One shared HTTP client for every camera's frames."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=FETCH_TIMEOUT_S)

    async def __call__(self, url: str) -> bytes:
        """GET one JPEG."""
        r = await self._client.get(url)
        r.raise_for_status()
        return r.content

    async def aclose(self) -> None:
        """Close the client."""
        await self._client.aclose()


def router(images: Images) -> APIRouter:
    """The frame and the arrays, read only."""
    r = APIRouter()

    def refused(exc: Unavailable) -> JSONResponse:
        return JSONResponse({"code": "unavailable", "detail": exc.detail}, status_code=exc.status)  # fmt: skip

    @r.get("/api/devices/{name}/frame.jpg")
    async def frame(request: Request, name: str) -> Response:
        try:
            jpeg, tag = await images.frame(request.app.state.module, name)
        except Unavailable as exc:
            return refused(exc)
        headers = {"Cache-Control": "no-store", "X-Frame-Tag": tag}
        return Response(jpeg, media_type="image/jpeg", headers=headers)

    @r.get("/api/devices/{name}/arrays")
    async def arrays(request: Request, name: str) -> Any:
        try:
            return await images.arrays(request.app.state.module, name)
        except Unavailable as exc:
            return refused(exc)

    return r
