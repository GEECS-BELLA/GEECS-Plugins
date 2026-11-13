"""A camera's image and arrays, read over Channel Access by this server: read only, at the PVs the module names.

Hare carries scalars; a camera's frame and its plots' arrays do not travel
through it (``GeecsControlRoom/CLAUDE.md`` § "The device pages"). Two routes serve them:

- ``GET /api/devices/{name}/frame.jpg``: the JPEG the IOC's MJPEG server
  holds. Its URL is the string at the PV the module's device route names
  for role ``image`` (``read: once``), read as a char string, its host
  replaced with the address CA reached the IOC at (lab DNS names the IOC
  host; a client off it cannot resolve that). Held on role
  ``image_counter`` (MJPG1's ``ArrayCounter_RBV``) as the raw frame is on
  its counter: probed at most every ``FRAME_EVERY_S``, an unmoved counter
  serves the held JPEG without a fetch for up to ``JPEG_REUSE_S``; ``ETag``
  the counter and the bytes' checksum, ``304`` on a matching
  ``If-None-Match``; one fetch in flight per device, shared. An unreadable
  counter (or no such role) fetches as before, with no ``ETag``. The ``X-Frame-Tag`` names the bytes, so a page never decodes the
  same frame twice (a frozen camera's frame is not "new").
- ``GET /api/devices/{name}/arrays``: every ``read: on_request`` role of type
  ``array`` (lineouts, histogram) by role name, read at most every
  ``ARRAYS_EVERY_S`` and shared by every viewer. The page polls it only
  while its plots are open.
- ``GET /api/devices/{name}/frame.raw``: the driver's frame at role
  ``raw_frame`` (``ARR:ArrayData``), uint16 little-endian row-major, gzip
  framed, its geometry and ``raw_counter`` tag in headers, ``304`` on a
  matching ``If-None-Match`` (weak, a list or ``*`` match too); the
  ``ETag`` is the counter and, once read, the full scale. One
  read in flight per device, held ``RAW_HOLD_S``; the counter is probed first
  so an unmoved camera costs one scalar read, never an array read.

The CA transport (:class:`Ca`), :class:`Shared`, :class:`Unavailable`, the
module-route cache (:meth:`Images.roles`, :meth:`Images.family`), the gzip
framing and the ``If-None-Match`` match are shared with the trace route
(``traces.py``).

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
from dataclasses import dataclass, replace
from typing import Any, Generic, Optional, Protocol, TypeVar
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from geecs_control_room.module_client import GATEWAY, ModuleClient, ModuleUnavailable

#: The module's device route is reused this long (the roster's own cadence).
ROUTE_TTL_S = 10.0
#: The floor between two fetches of one camera's frame, however many viewers poll (10 fps).
FRAME_EVERY_S = 0.1
#: The longest a JPEG is reused under an unchanged counter: should a counter ever be posted before its JPEG,
#: the stale frame lasts this long, never until the next trigger.
JPEG_REUSE_S = 1.0
#: The plots' arrays are read at most this often per device.
ARRAYS_EVERY_S = 0.5
#: A raw frame is held and shared this long (at most 2 Hz per camera, however many viewers poll).
RAW_HOLD_S = 0.5
#: One CA read or one frame fetch: the IOC answers in milliseconds (a 2592×2048 frame in ~0.2 s).
CA_TIMEOUT_S = FETCH_TIMEOUT_S = 3.0

T = TypeVar("T")


class Unavailable(Exception):
    """The route cannot serve: ``status`` (404 the module names no such thing, 503 a source is silent) and why."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


class Ca(Protocol):
    """The CA reads this module makes."""

    async def text(self, pv: str) -> str:
        """``pv`` read as a char string (a ``JPG_URL_RBV`` waveform)."""
        ...

    async def host(self, pv: str) -> str:
        """The ``host:port`` CA reached ``pv``'s server at."""
        ...

    async def arrays(self, pvs: list[str]) -> list[list[float]]:
        """Each of ``pvs`` read as numbers, in order."""
        ...

    async def scalar(self, pv: str) -> float:
        """``pv`` read as one number (a counter, the full scale)."""
        ...

    async def floats(self, pvs: list[str], count: Optional[int] = None) -> list[np.ndarray]:  # fmt: skip
        """Each of ``pvs`` as float32 (``DBR_FLOAT``), in order (its first ``count`` values only, when given): a trace and its axis."""
        ...

    async def frame(self, frame: str, width: str, height: str) -> tuple[np.ndarray, int, int]:  # fmt: skip
        """The frame's values as 16-bit integers (int16 as CA delivers ``DBR_SHORT``) and its geometry, in one batch."""
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

    async def scalar(self, pv: str) -> float:
        """``pv`` as one number."""
        import aioca

        return float(await aioca.caget(pv, timeout=CA_TIMEOUT_S))

    async def floats(self, pvs: list[str], count: Optional[int] = None) -> list[np.ndarray]:  # fmt: skip
        """Each of ``pvs`` asked as ``DBR_FLOAT`` (count 0: the IOC's current length)."""
        import aioca

        values = await aioca.caget(pvs, datatype=np.float32, count=count or 0, timeout=CA_TIMEOUT_S)  # fmt: skip
        return [np.asarray(v, dtype=np.float32) for v in values]

    async def frame(self, frame: str, width: str, height: str) -> tuple[np.ndarray, int, int]:  # fmt: skip
        """The frame (count 0: the IOC's current length) as ``DBR_SHORT`` and its geometry, read together.

        A ``USHORT`` waveform's native CA type is ``DBR_LONG`` (int32, twice
        the bytes on the wire); asked as ``DBR_SHORT`` the IOC sends each value's
        16 bits unchanged, so a count above 32767 arrives negative and
        :func:`_cast` restores it.
        """
        import aioca

        data, (w, h) = await asyncio.gather(
            aioca.caget(frame, datatype=np.int16, timeout=CA_TIMEOUT_S),
            aioca.caget([width, height], timeout=CA_TIMEOUT_S),
        )
        return np.asarray(data), int(w), int(h)


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


@dataclass(frozen=True)
class Jpeg:
    """One JPEG as served: its bytes, their checksum (``X-Frame-Tag``) and the ``image_counter`` read before its fetch.

    ``counter`` is ``None`` when the counter could not be read (or the module
    names none): the JPEG is served without an ``ETag`` and never held.
    """

    body: bytes
    tag: str
    counter: Optional[str]


@dataclass(frozen=True)
class RawFrame:
    """One raw frame as served: the gzip body, its geometry, its tag (``raw_counter``) and the full scale."""

    body: bytes
    width: int
    height: int
    tag: str
    full_scale: Optional[float]


def _cast(data: np.ndarray) -> np.ndarray:
    """The ``USHORT`` values, read as int16, as uint16 little-endian: a ``view`` of the same 16 bits, exact, never a clamp.

    The earlier "``astype``, never ``view``" rule was for the int32 delivery
    (a view there doubles the element count); the read is 16-bit now, so the
    view replaces it. No copy on a little-endian host; a big-endian one swaps.
    """
    return np.asarray(data).view(np.uint16).astype(np.dtype("<u2"), copy=False)


def gzip_frame(frame: np.ndarray) -> bytes:
    """Gzip framing at level 1 (``wbits=31``): what a browser's ``Content-Encoding: gzip`` decodes."""
    z = zlib.compressobj(1, zlib.DEFLATED, 31)
    return z.compress(frame) + z.flush()


def _finite(values: list[float]) -> list[Optional[float]]:
    """The values with NaN and ±inf as ``None``: the browser's ``JSON.parse`` rejects them."""
    return [v if math.isfinite(v) else None for v in values]


class Images:
    """Every camera's frame and arrays, each shared by its viewers; the module's PV names, cached."""

    def __init__(self, ca: Ca, fetch: Callable[[str], Awaitable[bytes]], clock: Callable[[], float] = time.monotonic) -> None:  # fmt: skip
        self._ca, self._fetch, self._clock = ca, fetch, clock
        self._routes: dict[str, Shared[dict[str, dict[str, Any]]]] = {}
        self._good: dict[str, dict[str, dict[str, Any]]] = {}
        self._family: dict[str, Optional[str]] = {}
        self._frames: dict[tuple[str, str, Optional[str]], Shared[Jpeg]] = {}
        self._jpegs: dict[tuple[str, str, Optional[str]], tuple[Jpeg, float]] = {}
        self._counters_down: dict[str, float] = {}
        self._urls: dict[str, str] = {}
        self._arrays: dict[tuple[str, tuple[str, ...]], Shared[dict[str, Any]]] = {}
        self._raws: dict[tuple[str, tuple[str, ...]], Shared[RawFrame]] = {}
        self._held: dict[tuple[str, tuple[str, ...]], RawFrame] = {}
        self._scales: dict[tuple[str, str], tuple[Optional[float], float]] = {}
        self._scale_reads: dict[tuple[str, str], asyncio.Task[None]] = {}

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
        # the source decides first: a DB device's family is its DB device type, never looked up as a family file
        family = None if body.get("source") == GATEWAY else body.get("family")
        self._good[name], self._family[name] = roles, family
        return roles

    @property
    def ca(self) -> Ca:
        """The CA transport, shared with the trace route."""
        return self._ca

    def family(self, name: str) -> Optional[str]:
        """The device's family template as the module's device route last named it (after :meth:`roles`); ``None`` for a DB device."""
        return self._family.get(name)

    async def frame(self, module: ModuleClient, name: str) -> Jpeg:
        """The latest JPEG, held on ``image_counter``, shared by every viewer within ``FRAME_EVERY_S``."""
        roles = await self.roles(module, name)
        role = roles.get("image")
        if role is None:
            raise Unavailable(404, f"the module names no image role for {name}")
        pv, counter_pv = role["pv"], roles.get("image_counter", {}).get("pv")
        key = (name, pv, counter_pv)
        source = self._frames.get(key)
        if source is None:
            source = self._frames[key] = Shared(lambda: self._read_frame(key), FRAME_EVERY_S, self._clock)  # fmt: skip
        return await source.get()

    async def _image_counter(self, pv: Optional[str]) -> Optional[str]:
        """MJPG1's counter, or ``None`` (no role, any CA failure, a NaN): the JPEG is then fetched untagged.

        An unreadable counter is not asked again for ``ROUTE_TTL_S``: a silent
        one would otherwise hold every frame for the CA timeout.
        """
        if pv is None or self._clock() - self._counters_down.get(pv, -math.inf) < ROUTE_TTL_S:  # fmt: skip
            return None
        # any CA failure is an unread counter, never a failed image
        (read,) = await asyncio.gather(self._ca.scalar(pv), return_exceptions=True)
        if isinstance(read, (int, float)) and math.isfinite(read):
            self._counters_down.pop(pv, None)
            return str(int(read))
        self._counters_down[pv] = self._clock()
        return None

    async def _read_frame(self, key: tuple[str, str, Optional[str]]) -> Jpeg:
        asked = self._clock()
        counter = await self._image_counter(key[2])
        held = self._jpegs.get(key)
        if counter is not None and held is not None and held[0].counter == counter and asked - held[1] < JPEG_REUSE_S:  # fmt: skip
            return held[0]  # the plugin has not moved: one scalar read, no fetch
        # tagged with the counter read BEFORE the fetch: the JPEG is at least that new (the plugin
        # stores its JPEG, then posts the counter), so a later counter change always fetches again
        body = await self._fetch_jpeg(key[1])
        jpeg = Jpeg(body, f"{zlib.crc32(body):08x}", counter)
        if counter is None:
            self._jpegs.pop(key, None)
        else:
            self._jpegs[key] = (jpeg, asked)
        return jpeg

    async def _fetch_jpeg(self, pv: str) -> bytes:
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
        return jpeg

    async def arrays(self, module: ModuleClient, name: str) -> dict[str, Any]:
        """Every ``read: on_request`` role of type ``array`` by role name, and the PVs read.

        An on-request scalar (a raw frame's geometry and counter, read with the
        frame) is not an array of the page's; the raw frame itself is ``read:
        never`` and served by its own route, never as JSON.
        """
        pvs = {r: e["pv"] for r, e in (await self.roles(module, name)).items() if e.get("read") == "on_request" and e.get("type") == "array"}  # fmt: skip
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

    async def raw(self, module: ModuleClient, name: str) -> RawFrame:
        """The latest raw frame, shared by every viewer within ``RAW_HOLD_S``."""
        roles = await self.roles(module, name)
        if "raw_frame" not in roles:
            raise Unavailable(404, f"the module names no raw_frame role for {name}")
        missing = [r for r in ("raw_width", "raw_height", "raw_counter") if r not in roles]  # fmt: skip
        if missing:
            raise Unavailable(404, f"the module names no {', '.join(missing)} role{'s' * (len(missing) > 1)} for {name}")  # fmt: skip
        pvs = tuple(roles[r]["pv"] for r in ("raw_frame", "raw_width", "raw_height", "raw_counter"))  # fmt: skip
        scale = roles.get("full_scale", {}).get("pv")
        key = (name, pvs + ((scale,) if scale else ()))
        source = self._raws.get(key)
        if source is None:
            source = self._raws[key] = Shared(lambda: self._read_raw(key, pvs, scale), RAW_HOLD_S, self._clock)  # fmt: skip
        return await source.get()

    async def _counter(self, pv: str) -> str:
        try:
            return str(int(await self._ca.scalar(pv)))
        except Exception as exc:  # any CA failure (or a NaN): said, never a 500
            raise Unavailable(503, f"the frame counter ({pv}) could not be read") from exc  # fmt: skip

    async def _batch(self, frame: str, width: str, height: str) -> tuple[np.ndarray, int, int]:  # fmt: skip
        try:
            return await self._ca.frame(frame, width, height)
        except Exception as exc:  # any CA failure: said, never a 500
            raise Unavailable(503, f"the raw frame ({frame}) could not be read: {exc}") from exc  # fmt: skip

    async def _read_raw(self, key: tuple[str, tuple[str, ...]], pvs: tuple[str, ...], scale: Optional[str]) -> RawFrame:  # fmt: skip
        frame_pv, width_pv, height_pv, counter_pv = pvs
        name = key[0]
        tag = await self._counter(counter_pv)
        self._ask_full_scale(name, scale)  # beside the frame, never in its way
        held = self._held.get(key)
        if held is not None and held.tag == tag:
            # the camera has not moved: one scalar read, no array read
            full_scale = self._full_scale(name, scale)
            if full_scale != held.full_scale:
                held = self._held[key] = replace(held, full_scale=full_scale)
            return held
        data, width, height = await self._batch(frame_pv, width_pv, height_pv)
        after = await self._counter(counter_pv)
        if after != tag:
            # a frame landed mid-batch: read once more, tagged by the second counter
            tag = after
            del data
            data, width, height = await self._batch(frame_pv, width_pv, height_pv)
        if data.dtype.kind not in "iu" or data.dtype.itemsize != 2:
            raise Unavailable(503, f"the raw frame arrived as {data.dtype}, not 16-bit")  # fmt: skip
        n = int(np.size(data))
        if n == 0 or n != width * height:
            # mono only: a colour frame (ArraySize0 = 3) fails here too
            raise Unavailable(503, f"geometry disagrees: {n} values for {width}×{height}")  # fmt: skip
        frame = await asyncio.to_thread(_cast, data)
        del data  # the cast is a view of it: both go once compressed (10.6 MB for a 2592×2048 frame)
        body = await asyncio.to_thread(gzip_frame, frame)
        del frame
        raw = self._held[key] = RawFrame(body, width, height, tag, self._full_scale(name, scale))  # fmt: skip
        return raw

    def _full_scale(self, name: str, pv: Optional[str]) -> Optional[float]:
        """The device's full scale as last read; ``None`` (the header omitted) while unread or unreadable."""
        return self._scales.get((name, pv), (None, 0.0))[0] if pv else None

    def _ask_full_scale(self, name: str, pv: Optional[str]) -> None:
        """Read the full scale in the background unless known (read once) or asked within ``ROUTE_TTL_S``.

        Never awaited by a frame: an unreadable ``FullScale`` would otherwise
        hold every viewer's frame for the CA timeout. A healthy read (a
        scalar, milliseconds) lands while the frame is read and cast.
        """
        if pv is None or (name, pv) in self._scale_reads:
            return
        held = self._scales.get((name, pv))
        if held is not None and (held[0] is not None or self._clock() - held[1] < ROUTE_TTL_S):  # fmt: skip
            return
        task = asyncio.get_running_loop().create_task(self._read_full_scale(name, pv))
        self._scale_reads[(name, pv)] = task

    async def _read_full_scale(self, name: str, pv: str) -> None:
        try:
            # any CA failure is an unreadable full scale: the header is omitted and the page assumes 4095
            (read,) = await asyncio.gather(self._ca.scalar(pv), return_exceptions=True)
            value = read if isinstance(read, float) and math.isfinite(read) and read > 0 else None  # fmt: skip
            self._scales[(name, pv)] = (value, self._clock())
        finally:
            self._scale_reads.pop((name, pv), None)


def etag_matches(if_none_match: Optional[str], etag: str) -> bool:
    """Whether ``If-None-Match`` names ``etag`` (weak comparison, a list, or ``*``), RFC 9110 § 13.1.2."""
    if not if_none_match:
        return False
    tags = {t.strip().removeprefix("W/") for t in if_none_match.split(",")}
    return "*" in tags or etag in tags


def _etag(raw: RawFrame) -> str:
    """The representation's validator: the counter, and the full scale once known (``"7"``, then ``"7;4095"``).

    The full scale is part of what a 200 says, so a scale landing after the
    first frame changes the ``ETag`` and the next conditional ask is a 200
    carrying it (the held body, no second array read), never a 304 that hides it.
    """
    return (
        f'"{raw.tag}"'
        if raw.full_scale is None
        else f'"{raw.tag};{_number(raw.full_scale)}"'
    )


def _number(value: float) -> str:
    """A header's number: ``4095``, never ``4095.0``."""
    return str(int(value)) if float(value).is_integer() else repr(value)


def _jpeg_response(jpeg: Jpeg, if_none_match: Optional[str]) -> Response:
    """The JPEG, or ``304`` when ``If-None-Match`` names its ``ETag`` (none while the counter is unread).

    The ``ETag`` is the counter and the bytes' checksum (``"3;0a1b2c3d"``):
    a fetch at ``JPEG_REUSE_S`` under the same counter that brings other
    bytes is a 200, never a 304 that hides them.
    """
    headers = {"Cache-Control": "no-store", "X-Frame-Tag": jpeg.tag}
    if jpeg.counter is not None:
        headers["ETag"] = f'"{jpeg.counter};{jpeg.tag}"'
        if etag_matches(if_none_match, headers["ETag"]):
            return Response(status_code=304, headers=headers)
    return Response(jpeg.body, media_type="image/jpeg", headers=headers)


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
    """The frame, the raw frame and the arrays, read only."""
    r = APIRouter()

    def refused(exc: Unavailable) -> JSONResponse:
        return JSONResponse({"code": "unavailable", "detail": exc.detail}, status_code=exc.status)  # fmt: skip

    @r.get("/api/devices/{name}/frame.jpg")
    async def frame(request: Request, name: str) -> Response:
        try:
            jpeg = await images.frame(request.app.state.module, name)
        except Unavailable as exc:
            return refused(exc)
        return _jpeg_response(jpeg, request.headers.get("if-none-match"))

    @r.get("/api/devices/{name}/frame.raw")
    async def frame_raw(request: Request, name: str) -> Response:
        try:
            raw = await images.raw(request.app.state.module, name)
        except Unavailable as exc:
            return refused(exc)
        etag = _etag(raw)
        headers = {
            "Cache-Control": "no-store",
            "ETag": etag,
            "X-Frame-Tag": f'"{raw.tag}"',
        }
        if etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
        headers |= {"Content-Encoding": "gzip", "X-Frame-Width": str(raw.width), "X-Frame-Height": str(raw.height)}  # fmt: skip
        if raw.full_scale is not None:
            headers["X-Frame-Full-Scale"] = _number(raw.full_scale)
        return Response(raw.body, media_type="application/octet-stream", headers=headers)  # fmt: skip

    @r.get("/api/devices/{name}/arrays")
    async def arrays(request: Request, name: str) -> Any:
        try:
            return await images.arrays(request.app.state.module, name)
        except Unavailable as exc:
            return refused(exc)

    return r
