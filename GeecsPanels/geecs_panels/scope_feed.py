"""One oscilloscope's live state: its settings and status, monitored; its traces, read on request.

A :class:`ScopeFeed` subscribes to the scope's readbacks on first use and
keeps the latest of each; :meth:`ScopeFeed.snapshot` is the JSON the page
renders, the chain's per-channel statistics included. The choices of every
enum setting are the PVs' own (``enums``, filled once by the transport), so
nothing here knows a vendor. Every viewer shares one feed. The traces come
through the camera's :class:`~geecs_panels.feed.ArraysSource` (one
background read at a time, at most one per interval, only while a page
asks) and are split and decimated for the page by :func:`traces_payload`.

The transport (``subscribe``) is injected, as for the camera.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Optional

from geecs_panels.feed import ArraysSource, ArraysUnavailable, Subscribe, Subscription
from geecs_panels.scope import (
    CHANNEL_FIELDS,
    CHANNELS,
    SCOPE_FIELDS,
    STAT_FIELDS,
    TIMEBASE_KEYS,
    decimate,
    monitored,
    split_channels,
)

#: Shown instead of a capture taken at an older timebase than the axis.
WAITING = "the timebase changed: waiting for a capture at the new one"

#: The page draws a trace at most this many min/max columns wide.
TRACE_COLUMNS = 1500


@dataclass
class ScopeFeed:
    """The monitored readbacks of one scope and when it last captured."""

    prefix: str
    subscribe: Subscribe
    clock: Callable[[], float] = time.monotonic
    values: dict[str, object] = field(default_factory=dict)
    connected: dict[str, bool] = field(default_factory=dict)
    #: enum setting suffix → its PV's choices, as the transport read them
    enums: dict[str, list[str]] = field(default_factory=dict)
    #: The capture identity when the timebase last changed: a capture with
    #: this identity or older was taken at the old timebase (None: no change
    #: since the feed connected).
    timebase_uid: Optional[float] = None
    _subs: list[Subscription] = field(default_factory=list)
    _captured_at: Optional[float] = None
    _started: bool = False

    def start(self) -> None:
        """Subscribe to every readback (idempotent)."""
        if self._started:
            return
        self._started = True
        for suffix, (key, as_string) in monitored().items():
            self.connected[key] = False
            self._subs.append(
                self.subscribe(
                    self.prefix + suffix,
                    as_string,
                    lambda value, k=key: self.update(k, value),
                )
            )

    def close(self) -> None:
        """Drop every subscription."""
        for sub in self._subs:
            sub.close()
        self._subs.clear()

    def update(self, key: str, value: Optional[object]) -> None:
        """Take one monitor event: a value, or ``None`` for a disconnect."""
        if value is None:
            self.connected[key] = False
            if key == "arr_uid":
                # An IOC restart renumbers captures: the marker means nothing now.
                self.timebase_uid = None
            if key == "count":
                self.values.pop(key, None)
            return
        previous = self.values.get(key)
        self.connected[key] = True
        self.values[key] = value
        if key in TIMEBASE_KEYS and previous is not None and value != previous:
            uid = self.values.get("arr_uid")
            self.timebase_uid = float(uid) if isinstance(uid, (int, float)) else -1.0
        # The first value after (re)connecting is the count's position, not a capture.
        if key == "count" and previous is not None and value != previous:
            self._captured_at = self.clock()

    def present_channels(self) -> list[str]:
        """The channels the model has, in order: the rows of the capture array."""
        return [ch for ch in CHANNELS if self.values.get(f"{ch}.present") == "Present"]

    def enabled_channels(self) -> list[str]:
        """The channels present on the model and switched on."""
        return [
            ch
            for ch in CHANNELS
            if self.values.get(f"{ch}.present") == "Present"
            and self.values.get(f"{ch}.enable") == "On"
        ]

    def snapshot(self) -> dict[str, object]:
        """The scope's settings, status and channels for the page."""
        reachable = any(self.connected.values())
        scope = {key: self.values.get(key) for key in SCOPE_FIELDS}
        channels = [
            {
                "name": ch,
                **{key: self.values.get(f"{ch}.{key}") for key in CHANNEL_FIELDS},
                "stats": {
                    key: self.values.get(f"{ch}.stat.{key}") for key in STAT_FIELDS
                },
            }
            for ch in CHANNELS
        ]
        age = None if self._captured_at is None else self.clock() - self._captured_at
        absent = {ch for ch in CHANNELS if self.values.get(f"{ch}.present") == "Absent"}
        return {
            "reachable": reachable,
            # A channel the model lacks has no chain plugins: not missing.
            "missing": sorted(
                k
                for k, ok in self.connected.items()
                if not ok and not (".stat." in k and k[0] in absent)
            ),
            "enums": self.enums,
            "scope": scope,
            "channels": channels,
            "capture_age": age,
        }


def traces_payload(
    arrays: dict[str, list[float]],
    present: list[str],
    taken_after: Optional[float] = None,
    columns: int = TRACE_COLUMNS,
) -> dict[str, object]:
    """Each channel with data, against the time axis, decimated to *columns*.

    *arrays* holds ``t`` (the time axis), ``data`` (the capture,
    [channels, samples]) and ``uid``, ``channels``, ``samples`` (the
    capture's identity and shape, read with it); *present* names the
    capture's rows. A disabled channel's row is all NaN and is left out.
    A capture whose identity is not after *taken_after* (the feed's
    ``timebase_uid``) was taken at an older timebase than the axis
    describes: no traces until a new one arrives.
    """
    uid = (arrays.get("uid") or [None])[0]
    if taken_after is not None and not _after(uid, taken_after):
        return {"traces": {}, "samples": 0, "waiting": WAITING}
    times = arrays.get("t", [])
    channels = int((arrays.get("channels") or [0])[0])
    samples = int((arrays.get("samples") or [0])[0])
    rows = split_channels(arrays.get("data", []), channels, samples)
    out: dict[str, object] = {}
    for ch, row in zip(present, rows):
        if any(v == v for v in row):  # not all NaN
            xs, ys = decimate(times, row, columns)
            out[ch] = {"t": xs, "v": ys}
    return {"traces": out, "samples": samples}


@dataclass
class ScopeTraces:
    """The capture for the page: read once per new capture, built once, shared.

    :meth:`kick` starts a read only when the monitored capture identity
    differs from the one held (the :class:`ArraysSource` keeps one read in
    flight, at most one per ``min_interval``), so an idle scope costs no
    reads. Each read is split and decimated in a worker thread, off the
    event loop (a 1 M-sample capture is half a second of Python), once for
    every viewer; :meth:`payload` then only applies the timebase check.
    """

    fetch_raw: Callable[[], Awaitable[dict[str, list[float]]]]
    present: Callable[[], list[str]]
    min_interval: float
    source: ArraysSource = field(init=False)

    def __post_init__(self) -> None:
        """Wrap the raw read in the build, so the source holds built payloads."""

        async def fetch() -> dict[str, list[float]]:
            raw = await self.fetch_raw()
            built = await asyncio.to_thread(traces_payload, raw, self.present())
            built["uid"] = (raw.get("uid") or [None])[0]
            return built  # type: ignore[return-value]  # the source holds the built payload

        self.source = ArraysSource(fetch, self.min_interval)

    def held_uid(self) -> Optional[object]:
        """The identity of the capture held (None before the first read or after a failure)."""
        try:
            return self.source.current().get("uid")
        except ArraysUnavailable:  # the last read failed: the next kick retries
            return None

    def kick(self, uid: Optional[object]) -> None:
        """Read the capture if *uid* (the monitored identity) is not the one held."""
        if uid is None or uid != self.held_uid():
            self.source.kick()

    def payload(self, taken_after: Optional[float]) -> dict[str, object]:
        """The held capture's traces, or "waiting" if it predates the timebase; ``key`` names it."""
        built: dict[str, object] = dict(
            self.source.current()
        )  # ArraysUnavailable propagates
        uid = built.get("uid")
        if taken_after is not None and not _after(uid, taken_after):
            return {
                "traces": {},
                "samples": 0,
                "waiting": WAITING,
                "key": f"waiting|{taken_after}",
            }
        built["key"] = f"{uid}"
        return built


def _after(uid: object, taken_after: float) -> bool:
    """Whether capture *uid* was taken after the timebase change marked *taken_after*."""
    return isinstance(uid, (int, float)) and uid > taken_after
