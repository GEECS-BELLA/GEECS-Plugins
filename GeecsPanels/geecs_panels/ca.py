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

import aioca
import httpx

from geecs_panels.camera import IMAGE_PLUGIN, JPG_URL_SUFFIX, counter_suffix
from geecs_panels.feed import (
    CameraFeed,
    FrameSource,
    ResolveUrl,
    Subscription,
    WriteFailed,
)
from geecs_panels.registry import Device

#: Upstream frame timeout: the IOC answers a snapshot in ~50 ms.
FETCH_TIMEOUT_S = 3.0
#: A write waits for the IOC to process the record, at most this long.
PUT_TIMEOUT_S = 3.0


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

    async def put(self, device: Device, suffix: str, value: object) -> None:
        """Write one PV and wait for the IOC to process it."""
        pv = device.prefix + suffix
        try:
            await aioca.caput(pv, value, wait=True, timeout=PUT_TIMEOUT_S)
        except Exception as exc:
            raise WriteFailed(f"{pv}: {exc}") from exc

    async def aclose(self) -> None:
        """Drop every subscription and the HTTP client."""
        for feed in self._feeds.values():
            feed.close()
        await self._fetch.aclose()
