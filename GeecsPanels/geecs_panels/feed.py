"""One camera's live state: its PVs, monitored, and its frames, shared.

A :class:`CameraFeed` subscribes to the camera's monitored PVs on first
use and keeps the latest value of each, the connection state and when each
plugin's frame counter last advanced; :meth:`CameraFeed.snapshot` turns that
into the JSON the page renders. Every viewer of the camera reads the same
feed, so N open panels cost the IOC one set of subscriptions.

A :class:`FrameSource` serves the latest JPEG with at most one upstream
request in flight, reusing a frame for everyone who asks within
``min_interval`` — N viewers cost the camera host one stream — and, when
it is given the camera's frame counter, never fetching again until the
counter moves: a 1 Hz camera polled at 10 Hz costs one fetch a second.

The transports are injected (``subscribe``, ``fetch``) so the logic runs in
tests without a CA library or an IOC; :mod:`geecs_panels.ca` supplies the
real ones.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Optional, Protocol

from geecs_panels.camera import (
    COUNTER_PLUGINS,
    IMAGE_PLUGIN,
    READINGS,
    counter_suffix,
    judge,
    monitored_suffixes,
    string_suffixes,
)

_COUNTER_SUFFIXES = frozenset(counter_suffix(p) for p in COUNTER_PLUGINS)

#: Seconds after subscribing before a never-connected PV counts as missing.
CONNECT_GRACE_S = 5.0


class Subscription(Protocol):
    """What a subscribe call returns: something that can be closed."""

    def close(self) -> None:
        """Stop the monitor."""


#: (pv, as_string, callback(value or None when disconnected)) -> Subscription
Subscribe = Callable[[str, bool, Callable[[Optional[object]], None]], Subscription]


@dataclass
class CameraFeed:
    """The monitored PVs of one camera, latest values and frame-counter ages."""

    prefix: str
    subscribe: Subscribe
    clock: Callable[[], float] = time.monotonic
    values: dict[str, object] = field(default_factory=dict)
    connected: dict[str, bool] = field(default_factory=dict)
    _ticks: dict[str, float] = field(default_factory=dict)
    _subs: list[Subscription] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)
    _started: Optional[float] = None

    def start(self) -> None:
        """Subscribe to every monitored PV (idempotent)."""
        if self._started is not None:
            return
        self._started = self.clock()
        strings = string_suffixes()
        for suffix in monitored_suffixes():
            self.connected[suffix] = False
            self._subs.append(
                self.subscribe(
                    self.prefix + suffix,
                    suffix in strings,
                    lambda value, s=suffix: self.update(s, value),
                )
            )

    def close(self) -> None:
        """Drop every subscription."""
        for sub in self._subs:
            sub.close()
        self._subs.clear()

    def update(self, suffix: str, value: Optional[object]) -> None:
        """Take one monitor event: a value, or ``None`` for a disconnect."""
        if value is None:
            self.connected[suffix] = False
            if suffix in _COUNTER_SUFFIXES:
                # The first value after reconnecting (an IOC restart, a VPN
                # blip) is the counter's position again, not a frame.
                self.values.pop(suffix, None)
            return
        previous = self.values.get(suffix)
        self._seen.add(suffix)
        self.connected[suffix] = True
        self.values[suffix] = value
        for plugin in COUNTER_PLUGINS:
            # The first value after (re)connecting is the counter's current
            # position, not an advance: only a change is a processed frame.
            if (
                suffix == counter_suffix(plugin)
                and previous is not None
                and value != previous
            ):
                self._ticks[plugin] = self.clock()

    def counter_ages(self) -> dict[str, Optional[float]]:
        """Plugin → seconds since its counter last advanced (``None``: not seen to)."""
        now = self.clock()
        return {
            plugin: (now - self._ticks[plugin]) if plugin in self._ticks else None
            for plugin in COUNTER_PLUGINS
        }

    def missing(self) -> list[str]:
        """Suffixes never connected once the grace period has passed."""
        if self._started is None or self.clock() - self._started < CONNECT_GRACE_S:
            return []
        return [s for s, ok in self.connected.items() if not ok and s not in self._seen]

    def snapshot(self, stale_after: float) -> dict[str, object]:
        """The page's view: every reading with its state, the image's age, what is missing."""
        ages = self.counter_ages()
        readings = []
        for reading in READINGS:
            state = judge(
                reading,
                connected=self.connected,
                values=self.values,
                counter_age=ages,
                stale_after=stale_after,
            )
            readings.append(
                {
                    "key": reading.key,
                    "value": self.values.get(reading.suffix),
                    "state": state,
                    "age": ages.get(reading.plugin) if reading.plugin else None,
                }
            )
        image_age = ages[IMAGE_PLUGIN]
        image_connected = self.connected.get(counter_suffix(IMAGE_PLUGIN), False)
        if not image_connected:
            image_state = "disconnected"
        elif image_age is None or image_age > stale_after:
            image_state = "stale"
        else:
            image_state = "fresh"
        return {
            "readings": readings,
            "image": {"state": image_state, "age": image_age},
            "missing": self.missing(),
        }


#: () -> the URL of one JPEG frame from the IOC.
ResolveUrl = Callable[[], Awaitable[str]]
#: url -> the JPEG bytes.
Fetch = Callable[[str], Awaitable[bytes]]


@dataclass(frozen=True)
class Frame:
    """One JPEG, the frame counter it was fetched at, and its upstream fetch number.

    ``tag`` names this exact fetch for the page (``"<counter>.<fetch>"``):
    the page redraws exactly when the tag changes, so a safety refetch under
    an unchanged counter is still drawn.
    """

    jpeg: bytes
    counter: Optional[object] = None
    fetch: int = 0

    @property
    def tag(self) -> str:
        """The page's identity for this frame."""
        return f"{'' if self.counter is None else self.counter}.{self.fetch}"


class FrameUnavailable(RuntimeError):
    """No frame could be fetched; the message says why."""


@dataclass
class FrameSource:
    """The latest JPEG of one camera, one upstream request at a time."""

    resolve_url: ResolveUrl
    fetch: Fetch
    min_interval: float
    #: The camera's current frame counter (MJPG1's), ``None`` when unknown.
    counter: Callable[[], Optional[object]] = lambda: None
    #: The longest a frame is reused under an unchanged counter.
    max_reuse: float = 1.0
    clock: Callable[[], float] = time.monotonic
    _fetches: int = 0
    _url: Optional[str] = None
    _frame: Optional[Frame] = None
    _at: float = float("-inf")
    _inflight: Optional[asyncio.Future[Frame]] = None

    async def latest(self) -> Frame:
        """The newest frame, shared with concurrent callers.

        Three rules, in order:

        - **A floor for everyone:** within ``min_interval`` of the last
          fetch, the cached frame is served — however many viewers poll and
          however fast the camera, the IOC sees at most one fetch per
          ``min_interval`` (N viewers cost one stream).
        - **The counter:** past the floor, a frame is reused while the
          camera's counter has not moved — a 1 Hz camera polled at 10 Hz is
          fetched once a second.
        - **A safety net:** a frame is never reused longer than
          ``max_reuse``, counter or not — should a counter ever arrive
          before its JPEG, the stale frame lasts this long, not a camera
          period. (ffmpegServer stores the JPEG, then posts its counter.)

        All ages are timed from the request, not the reply: timed from the
        reply, a poll at the cap's period landed inside the window behind a
        slow fetch and was handed the old frame again (a third of them).
        """
        counter = self.counter()
        if self._frame is not None:
            age = self.clock() - self._at
            if age < self.min_interval:
                return self._frame
            unchanged = counter is not None and counter == self._frame.counter
            if unchanged and age < self.max_reuse:
                return self._frame
        if self._inflight is not None:
            return await asyncio.shield(self._inflight)
        asked = self.clock()
        self._inflight = asyncio.get_running_loop().create_future()
        try:
            # Tagged with the counter read BEFORE the fetch: the JPEG is at
            # least that new, so a later counter change always refetches.
            self._fetches += 1
            frame = Frame(await self._fetch(), counter, self._fetches)
        except BaseException as exc:
            self._inflight.set_exception(exc)
            # Mark it retrieved: a request with no concurrent waiter must
            # not log "exception was never retrieved".
            self._inflight.exception()
            raise
        else:
            self._frame, self._at = frame, asked
            self._inflight.set_result(frame)
            return frame
        finally:
            self._inflight = None

    async def _fetch(self) -> bytes:
        if self._url is None:
            try:
                self._url = await self.resolve_url()
            except Exception as exc:
                raise FrameUnavailable(f"stream URL unavailable: {exc}") from exc
        try:
            return await self.fetch(self._url)
        except Exception as exc:
            # Re-resolve next time: the IOC may have moved or restarted.
            self._url = None
            raise FrameUnavailable(f"frame fetch failed: {exc}") from exc
