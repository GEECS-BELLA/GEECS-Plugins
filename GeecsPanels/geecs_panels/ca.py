"""The real transports: aioca monitors, the IOC's stream URL, httpx fetches.

**Import order matters.** libca reads ``EPICS_CA_ADDR_LIST`` when its
context is created, at ``import aioca``; the command line applies the
``config.ini`` addressing (:func:`geecs_core.epics_env.apply_epics_address_config`)
before importing this module.

**The stream URL names the IOC host by its hostname**
(``http://<host>:8081/CAM1.MJPG.jpg``), which a client without lab DNS
cannot resolve. CA already knows where the IOC is — the address its
channel connected to — so the URL's host is replaced with that address
(:func:`with_ca_host`), keeping the HTTP port.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import asyncio
import time

import aioca
import httpx

from geecs_panels.camera import (
    ARRAYS,
    IMAGE_PLUGIN,
    JPG_URL_SUFFIX,
    SETTLED_BY_READBACK,
    counter_suffix,
)
from geecs_panels.feed import (
    ArraysSource,
    CameraFeed,
    FrameSource,
    ResolveUrl,
    Subscription,
    WriteFailed,
)
from geecs_panels.motor import NO_WAIT as MOTOR_NO_WAIT
from geecs_panels.motor_feed import MotorFeed
from geecs_panels.registry import Device
from geecs_panels.scope import (
    ARRAY_DATA,
    ARRAY_FIELDS,
    CHAR_STRING_SUFFIXES,
    SETTLED_BY_READBACK as SCOPE_SETTLED_BY_READBACK,
    ENUM_SETTINGS,
    TIME_AXIS,
)
from geecs_panels.scope_feed import ScopeFeed, ScopeTraces

#: Upstream frame timeout: the IOC answers a snapshot in ~50 ms.
FETCH_TIMEOUT_S = 3.0
#: The plots' arrays are read at most this often (and only while asked for).
ARRAYS_EVERY_S = 0.5
#: A write waits for the IOC to process the record, at most this long.
PUT_TIMEOUT_S = 3.0
#: How often a readback-settled write re-reads its readback.
SETTLE_POLL_S = 0.1


def subscribe(
    pv: str, as_string: bool, callback: Callable[[Optional[object]], None]
) -> Subscription:
    """Monitor *pv*; *callback* gets each value, or ``None`` on a disconnect."""

    def on_update(value: object) -> None:
        if not getattr(value, "ok", True):
            callback(None)
            return
        # aioca's augmented values subclass the plain type; hand on plain ones.
        if as_string:
            callback(str(value))
        elif isinstance(value, float):
            callback(float(value))
        elif isinstance(value, int):
            callback(int(value))
        else:
            callback(str(value))

    return aioca.camonitor(
        pv,
        on_update,
        datatype=str if as_string else None,
        notify_disconnect=True,
    )


async def settle(pv: str, want: object) -> None:
    """Poll *pv* until it reads *want*, or raise ``WriteFailed`` after ``PUT_TIMEOUT_S``."""
    deadline = time.monotonic() + PUT_TIMEOUT_S
    got: object = None
    while True:
        try:
            got = await aioca.caget(pv, timeout=PUT_TIMEOUT_S)
        except Exception as exc:
            raise WriteFailed(f"{pv}: {exc}") from exc
        if got == want:
            return
        if time.monotonic() >= deadline:
            raise WriteFailed(f"{pv} stayed {got!r} (wanted {want!r})")
        await asyncio.sleep(SETTLE_POLL_S)


def with_ca_host(url: str, ca_host: str) -> str:
    """Replace *url*'s host with the IOC's CA address (``ip:port``), keeping the HTTP port.

    >>> with_ca_host("http://camserv:8081/CAM1.MJPG.jpg", "192.168.7.166:5064")
    'http://192.168.7.166:8081/CAM1.MJPG.jpg'
    """
    parts = urlsplit(url)
    ip = ca_host.rsplit(":", 1)[0]
    netloc = f"{ip}:{parts.port}" if parts.port else ip
    return urlunsplit(parts._replace(netloc=netloc))


def url_resolver(prefix: str) -> ResolveUrl:
    """The resolver for one camera: its ``JPG_URL_RBV``, rehosted to the CA address."""
    pv = prefix + JPG_URL_SUFFIX

    async def resolve() -> str:
        url = await aioca.caget(pv, datatype=aioca.DBR_CHAR_STR, timeout=3.0)
        info = await aioca.cainfo(pv, timeout=3.0)
        return with_ca_host(str(url), info.host)

    return resolve


class Fetcher:
    """One shared HTTP client for every camera's frames."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=FETCH_TIMEOUT_S)

    async def __call__(self, url: str) -> bytes:
        """GET one JPEG."""
        response = await self._client.get(url)
        response.raise_for_status()
        return response.content

    async def aclose(self) -> None:
        """Close the client."""
        await self._client.aclose()


class LiveCameras:
    """Each camera's feed and frame source, created on first use and shared by every viewer."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._fetch = Fetcher()
        self._feeds: dict[str, CameraFeed] = {}
        self._frames: dict[str, FrameSource] = {}
        self._arrays: dict[str, ArraysSource] = {}

    def feed(self, device: Device) -> CameraFeed:
        """The camera's feed, subscribed on first use."""
        feed = self._feeds.get(device.name)
        if feed is None:
            feed = self._feeds[device.name] = CameraFeed(device.prefix, subscribe)
            feed.start()
        return feed

    def frames(self, device: Device) -> FrameSource:
        """The camera's frame source."""
        source = self._frames.get(device.name)
        if source is None:
            feed = self.feed(device)
            counter = counter_suffix(IMAGE_PLUGIN)
            source = self._frames[device.name] = FrameSource(
                url_resolver(device.prefix),
                self._fetch,
                self._min_interval,
                counter=lambda: feed.values.get(counter),
            )
        return source

    def arrays(self, device: Device) -> ArraysSource:
        """The camera's STAT arrays: one read per ARRAYS_EVERY_S, shared by every viewer."""
        source = self._arrays.get(device.name)
        if source is None:
            pvs = {key: device.prefix + suffix for key, suffix in ARRAYS.items()}

            async def fetch() -> dict[str, list[float]]:
                values = await aioca.caget(list(pvs.values()), timeout=3.0)
                return {key: [float(x) for x in v] for key, v in zip(pvs, values)}

            source = self._arrays[device.name] = ArraysSource(fetch, ARRAYS_EVERY_S)
        return source

    async def put(self, device: Device, suffix: str, value: object) -> None:
        """Write one PV and wait for the IOC to process it (or, for some, for its readback)."""
        pv = device.prefix + suffix
        readback = SETTLED_BY_READBACK.get(suffix)
        try:
            # An access-security refusal raises here either way.
            await aioca.caput(pv, value, wait=readback is None, timeout=PUT_TIMEOUT_S)
        except Exception as exc:
            raise WriteFailed(f"{pv}: {exc}") from exc
        if readback is not None:
            await settle(device.prefix + readback, value)

    async def aclose(self) -> None:
        """Drop every subscription and the HTTP client."""
        for feed in self._feeds.values():
            feed.close()
        await self._fetch.aclose()


#: The traces are read at most this often, and only for a new capture while a
#: page asks: a running scope is drawn at 2 Hz (4 x 100 k samples is ~1.6 MB
#: a read over the VPN).
TRACES_EVERY_S = 0.5


def subscribe_scope(
    pv: str, as_string: bool, callback: Callable[[Optional[object]], None]
) -> Subscription:
    """:func:`subscribe`, except a char-array message (``LastError_RBV``) is read as its text."""
    if not pv.endswith(CHAR_STRING_SUFFIXES):
        return subscribe(pv, as_string, callback)

    def on_update(value: object) -> None:
        callback(str(value) if getattr(value, "ok", True) else None)

    return aioca.camonitor(
        pv, on_update, datatype=aioca.DBR_CHAR_STR, notify_disconnect=True
    )


async def read_enums(feed: ScopeFeed) -> None:
    """Fill *feed*'s ``enums`` with each enum setting's choices, from the PV itself."""
    pvs = [feed.prefix + suffix for suffix in ENUM_SETTINGS]
    values = await aioca.caget(pvs, format=aioca.FORMAT_CTRL, timeout=3.0, throw=False)
    for suffix, value in zip(ENUM_SETTINGS, values):
        names = getattr(value, "enums", None)
        if getattr(value, "ok", False) and names:
            feed.enums[suffix] = [str(n) for n in names]


class LiveScopes:
    """Each scope's feed and trace source, created on first use and shared by every viewer."""

    def __init__(self) -> None:
        self._feeds: dict[str, ScopeFeed] = {}
        self._traces: dict[str, ScopeTraces] = {}
        self._enum_reads: dict[str, asyncio.Task[None]] = {}

    def feed(self, device: Device) -> ScopeFeed:
        """The scope's feed, subscribed on first use; its choices read until they are known."""
        feed = self._feeds.get(device.name)
        if feed is None:
            feed = self._feeds[device.name] = ScopeFeed(device.prefix, subscribe_scope)
            feed.start()
        self.refresh(device)
        return feed

    def refresh(self, device: Device) -> None:
        """Start a read of the choices if they are incomplete and none is running."""
        feed = self._feeds.get(device.name)
        if feed is None or len(feed.enums) >= len(ENUM_SETTINGS):
            return
        task = self._enum_reads.get(device.name)
        if task is None or task.done():
            # The IOC may be down at first: retried until complete, one read at a time.
            self._enum_reads[device.name] = asyncio.get_running_loop().create_task(
                read_enums(feed)
            )

    def traces(self, device: Device) -> ScopeTraces:
        """The time axis and the whole capture (``ARR``), read together, once per capture."""
        source = self._traces.get(device.name)
        if source is None:

            async def fetch() -> dict[str, list[float]]:
                # One request: the samples with their identity and shape.
                keys = list(ARRAY_FIELDS)
                axis, data, *meta = await aioca.caget(
                    [device.prefix + TIME_AXIS, device.prefix + ARRAY_DATA]
                    + [device.prefix + ARRAY_FIELDS[k] for k in keys],
                    timeout=3.0,
                )
                return {
                    "t": list(map(float, axis)),
                    "data": list(map(float, data)),
                    **{k: [float(v)] for k, v in zip(keys, meta)},
                }

            feed = self.feed(device)
            source = self._traces[device.name] = ScopeTraces(
                fetch, feed.present_channels, TRACES_EVERY_S
            )
        return source

    async def put(self, device: Device, suffix: str, value: object) -> None:
        """Write one setting and wait for the IOC to process it (Acquire: for its readback)."""
        pv = device.prefix + suffix
        readback = SCOPE_SETTLED_BY_READBACK.get(suffix)
        counter_pv = device.prefix + "ArrayCounter_RBV"
        try:
            before = (
                await aioca.caget(counter_pv, timeout=PUT_TIMEOUT_S)
                if readback
                else None
            )
            await aioca.caput(pv, value, wait=readback is None, timeout=PUT_TIMEOUT_S)
        except Exception as exc:
            raise WriteFailed(f"{pv}: {exc}") from exc
        if readback is None:
            return
        try:
            await settle(device.prefix + readback, value)
        except WriteFailed:
            # A Single capture can start and finish between two polls: a new
            # capture is the start having happened.
            if (
                not value
                or await aioca.caget(counter_pv, timeout=PUT_TIMEOUT_S) == before
            ):
                raise

    def close(self) -> None:
        """Drop every subscription."""
        for feed in self._feeds.values():
            feed.close()


class LiveMotors:
    """Each positioner's feed, created on first use and shared by every viewer."""

    def __init__(self) -> None:
        self._feeds: dict[str, MotorFeed] = {}

    def feed(self, device: Device) -> MotorFeed:
        """The positioner's feed, subscribed on first use."""
        feed = self._feeds.get(device.name)
        if feed is None:
            feed = self._feeds[device.name] = MotorFeed(
                device.prefix, device.axes, subscribe
            )
            feed.start()
        return feed

    async def put(self, device: Device, axis: str, field: str, value: object) -> None:
        """Write one field of one axis.

        A move's put (``.VAL``, a tweak, a home) completes only when the
        motion ends — put-callback on a motor record — so it is dispatched
        without waiting; the stream shows the move and ``.DMOV`` its end.
        A setting (``.TWV``, ``.VELO``) and ``.STOP`` are awaited, so a
        refused one surfaces: the record completes those at once, mid-move
        too (simmotor1: 0.02 s, 0.04 s and 0.28 s during a 6 s move).
        """
        pv = device.prefix + axis + field
        try:
            await aioca.caput(
                pv, value, wait=field not in MOTOR_NO_WAIT, timeout=PUT_TIMEOUT_S
            )
        except Exception as exc:
            raise WriteFailed(f"{pv}: {exc}") from exc

    def close(self) -> None:
        """Drop every subscription."""
        for feed in self._feeds.values():
            feed.close()
