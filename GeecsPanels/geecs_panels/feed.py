"""One camera's live state: its PVs, monitored, and its frames, shared.

A :class:`CameraFeed` subscribes to the camera's monitored PVs on first
use and keeps the latest value of each, the connection state and when each
plugin's frame counter last advanced; :meth:`CameraFeed.snapshot` turns that
into the JSON the page renders. Every viewer of the camera reads the same
feed, so N open panels cost the IOC one set of subscriptions.

A :class:`FrameSource` serves the latest JPEG with at most one upstream
request in flight, reusing a frame for everyone who asks within
``min_interval`` — N viewers cost the camera host one stream — and, when
it is given the camera's frame counter, not fetching again until the
counter moves (or ``max_reuse`` passes): a 1 Hz camera polled at 10 Hz
costs one fetch a second.

The transports are injected (``subscribe``, ``fetch``) so the logic runs in
tests without a CA library or an IOC; :mod:`geecs_panels.ca` supplies the
real ones.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Optional, Protocol

from geecs_panels.camera import (
    COUNTER_PLUGINS,
    CROSSHAIR_FIELDS,
    CROSSHAIRS,
    ACQUISITION_FIELDS,
    IMAGE_PLUGIN,
    IMAGE_SIZE_FIELDS,
    LOCK_FIELDS,
    PLOT_FIELDS,
    PROCESSING_FIELDS,
    READINGS,
    ROI_FIELDS,
    counter_suffix,
    crosshair_suffix,
    judge,
    lock_state,
    monitored_suffixes,
    saturation,
    string_suffixes,
)
from geecs_panels.coords import Axis, to_image

_COUNTER_SUFFIXES = frozenset(counter_suffix(p) for p in COUNTER_PLUGINS)

#: The statistics the service keeps a history of, in each point's order
#: after its timestamp (the history lives here: STAT's own time-series
#: buffers are IOC-global and idle).
HISTORY_KEYS = ("cx", "cy", "mean", "max")
HISTORY_SUFFIXES = (
    "STAT:CentroidX_RBV",
    "STAT:CentroidY_RBV",
    "STAT:MeanValue_RBV",
    "STAT:MaxValue_RBV",
)
#: The flag each history value needs on to be a measurement at all.
HISTORY_FLAGS = (
    "STAT:ComputeCentroid",
    "STAT:ComputeCentroid",
    "STAT:ComputeStatistics",
    "STAT:ComputeStatistics",
)
#: One history point per this many seconds at most, and this many kept.
HISTORY_EVERY_S = 0.2
HISTORY_POINTS = 3000  # ten minutes at the decimated rate

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
    #: The held background predates a transform change (see _track_background).
    background_stale: bool = False
    #: Wall-clock time, for history timestamps (``clock`` is monotonic).
    wall: Callable[[], float] = time.time
    _history: deque[tuple[float, ...]] = field(
        default_factory=lambda: deque(maxlen=HISTORY_POINTS)
    )
    _history_at: float = float("-inf")
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
        self._track_background(suffix, previous, value)
        for plugin in COUNTER_PLUGINS:
            # The first value after (re)connecting is the counter's current
            # position, not an advance: only a change is a processed frame.
            if (
                suffix == counter_suffix(plugin)
                and previous is not None
                and value != previous
            ):
                self._ticks[plugin] = self.clock()
                if plugin == "STAT":
                    self._sample_history()

    def _sample_history(self) -> None:
        """Record the statistics on a STAT frame, decimated to HISTORY_EVERY_S.

        Sampled on the counter's tick, not the values' own monitors: a
        steady value posts nothing but still makes evenly spaced points.
        The counter and the values are separate records posted together, so
        a sample can pair one frame's tick with the previous frame's values —
        harmless at this rate.
        """
        now = self.clock()
        if now - self._history_at < HISTORY_EVERY_S:
            return
        self._history_at = now
        point = [self.wall()]
        for suffix, flag in zip(HISTORY_SUFFIXES, HISTORY_FLAGS):
            # A value not computed (its flag off) or not connected is frozen,
            # not measured: a gap, never a repeat of its last value.
            v = self.values.get(suffix)
            live = self.connected.get(suffix) and self.values.get(flag)
            point.append(
                float(v) if live and isinstance(v, (int, float)) else float("nan")
            )
        self._history.append(tuple(point))

    def history(self, seconds: float) -> list[tuple[float, ...]]:
        """The recorded statistics of the last *seconds*, oldest first."""
        since = self.wall() - seconds
        return [p for p in self._history if p[0] >= since]

    def _track_background(
        self, suffix: str, previous: Optional[object], value: object
    ) -> None:
        """Mark the held background stale on a transform change; a grab clears it.

        The background is grabbed AFTER the transform and NDProcess checks
        only its size, so after a rotation it is subtracted in the wrong
        orientation. Seen on the IOC's own records — a change from any client
        — and kept here, for every viewer of this service; a grab completing
        (``SaveBackground_RBV`` back from 1 to 0) clears it.
        """
        transform = suffix == PROCESSING_FIELDS["transform"]
        if transform and previous is not None and value != previous:
            self.background_stale = True
        grabbed = suffix == PROCESSING_FIELDS["background_saving"]
        if grabbed and previous == 1 and value == 0:
            self.background_stale = False

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
            "image": {
                "state": image_state,
                "age": image_age,
                **{f: self.values.get(s) for f, s in IMAGE_SIZE_FIELDS.items()},
            },
            "missing": self.missing(),
            "crosshairs": self.crosshairs(),
            "roi": self.roi(),
            "processing": {
                **{f: self.values.get(s) for f, s in PROCESSING_FIELDS.items()},
                "background_stale": self.background_stale,
            },
            "lock": {
                "state": lock_state(self.values, self.connected),
                **{
                    f: self.values.get(s) for f, s in LOCK_FIELDS.items() if f != "lock"
                },
            },
            "acquisition": {
                f: self.values.get(s) for f, s in ACQUISITION_FIELDS.items()
            },
            "saturation": self._saturation(readings),
        }

    def _saturation(self, readings: list[dict[str, object]]) -> dict[str, object]:
        """The saturation check, on a FRESH raw max only (a stale one says unknown)."""
        raw = next(r for r in readings if r["key"] == "raw_max")
        fmt = next(r for r in readings if r["key"] == "pixel_format")
        if raw["state"] != "fresh":
            return saturation(None, None)
        return saturation(fmt["value"], raw["value"])

    def crosshairs(self) -> list[dict[str, object]]:
        """Each crosshair's stored state (``None`` fields: not read yet)."""
        return [
            {
                "n": n,
                **{
                    f: self.values.get(crosshair_suffix(n, f)) for f in CROSSHAIR_FIELDS
                },
            }
            for n in CROSSHAIRS
        ]

    def roi(self) -> dict[str, object]:
        """The analysis ROI's bounds, binning and direction."""
        return {field: self.values.get(suffix) for field, suffix in ROI_FIELDS.items()}

    def roi_axes(self) -> tuple[Axis, Axis]:
        """ROI1's two axes for :mod:`geecs_panels.coords`; raises if not read yet."""
        r = self.roi()
        if any(r[k] is None for k in ("minx", "miny", "sizex", "sizey")):
            raise LookupError("the analysis ROI has not been read yet")
        return (
            Axis(int(r["minx"]), int(r["sizex"]), int(r["binx"] or 1), bool(r["revx"])),
            Axis(int(r["miny"]), int(r["sizey"]), int(r["biny"] or 1), bool(r["revy"])),
        )


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


class WriteFailed(RuntimeError):
    """The IOC did not take a write; the message says why."""


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
    _inflight: Optional[asyncio.Task[Frame]] = None

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

        The fetch runs as a task of its own that every caller awaits through
        a shield: one caller going away (its request cancelled) never cancels
        the fetch the others are waiting for.
        """
        counter = self.counter()
        if self._frame is not None:
            age = self.clock() - self._at
            if age < self.min_interval:
                return self._frame
            unchanged = counter is not None and counter == self._frame.counter
            if unchanged and age < self.max_reuse:
                return self._frame
        if self._inflight is None:
            self._inflight = asyncio.get_running_loop().create_task(
                self._fetch_frame(counter, self.clock())
            )
            # Every caller may have gone before a failure: retrieve it here,
            # so asyncio does not log "Task exception was never retrieved".
            self._inflight.add_done_callback(
                lambda t: None if t.cancelled() else t.exception()
            )
        return await asyncio.shield(self._inflight)

    async def _fetch_frame(self, counter: Optional[object], asked: float) -> Frame:
        try:
            # Tagged with the counter read BEFORE the fetch: the JPEG is at
            # least that new, so a later counter change always refetches.
            self._fetches += 1
            frame = Frame(await self._fetch(), counter, self._fetches)
            self._frame, self._at = frame, asked
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


class ArraysUnavailable(RuntimeError):
    """STAT's arrays could not be read; the message says why."""


#: () -> {ARRAYS key: list of numbers}, read from the IOC.
FetchArrays = Callable[[], Awaitable[dict[str, list[float]]]]


@dataclass
class ArraysSource:
    """STAT's arrays, read in the background and shared: at most one read per ``min_interval``.

    The SSE stream never WAITS for a read: it asks for one (:meth:`kick`)
    and sends whatever is held (:meth:`current`). A slow or failing read
    (a PV not served, the VPN blinking: a 3 s CA timeout) would otherwise
    hold back every push of the readings and turn the page stale. A failure
    is held for ``min_interval`` like a result, so it is not retried on
    every push either. The read runs as a task of its own: a viewer
    closing its stream never cancels the read the others share.
    """

    fetch: FetchArrays
    min_interval: float
    clock: Callable[[], float] = time.monotonic
    _arrays: Optional[dict[str, list[float]]] = None
    _error: Optional[str] = None
    _at: float = float("-inf")
    _task: Optional[asyncio.Task[dict[str, list[float]]]] = None

    def kick(self) -> None:
        """Start a read if none is running and the last one is older than ``min_interval``."""
        if self._task is not None or self.clock() - self._at < self.min_interval:
            return
        asked = self.clock()
        self._task = asyncio.get_running_loop().create_task(self.fetch())
        self._task.add_done_callback(lambda task: self._done(task, asked))

    def _done(self, task: asyncio.Task[dict[str, list[float]]], asked: float) -> None:
        self._task, self._at = None, asked
        if task.cancelled():
            return
        exc = task.exception()  # retrieved here, so never "never retrieved"
        if exc is not None:
            self._error = f"plots unavailable: {exc}"
        else:
            self._arrays, self._error = task.result(), None

    def current(self) -> dict[str, list[float]]:
        """The arrays held now; :class:`ArraysUnavailable` if the last read failed."""
        if self._error is not None:
            raise ArraysUnavailable(self._error)
        return self._arrays or {}


def _axis_map(axis: Axis) -> dict[str, float]:
    """An ROI output axis as image pixels: the first sample's position and the step."""
    start = to_image(axis, 0)
    return {"start": start, "step": to_image(axis, 1) - start}


def _finite(values: list[float], n: Optional[object]) -> list[Optional[float]]:
    """The first *n* values (the waveform's valid length); NaN and ±inf as ``None``.

    Browsers' ``JSON.parse`` rejects ``NaN`` and ``Infinity``, which Python's
    encoder writes by default: one would freeze the page.
    """
    count = int(n) if isinstance(n, (int, float)) else len(values)
    return [round(v, 6) if math.isfinite(v) else None for v in values[:count]]


def plot_payload(
    feed: CameraFeed,
    arrays: dict[str, list[float]],
    history_s: float,
    stale_after: float = 3.0,
) -> dict[str, object]:
    """Everything the plots draw, in image pixels where it is a position.

    ``stat_fresh`` says whether STAT is processing frames: a read of its
    arrays always succeeds, so held arrays must not pass for live ones.
    """
    v = feed.values
    age = feed.counter_ages().get("STAT")
    out: dict[str, object] = {
        "stat_fresh": bool(
            feed.connected.get(counter_suffix("STAT"))
            and age is not None
            and age <= stale_after
        ),
        "profiles_on": bool(v.get(PLOT_FIELDS["profiles_on"])),
        "histogram_on": bool(v.get(PLOT_FIELDS["histogram_on"])),
        "histogram": _finite(arrays.get("histogram", []), None),
        "hist_min": v.get(PLOT_FIELDS["hist_min"]),
        "hist_max": v.get(PLOT_FIELDS["hist_max"]),
        "history_keys": list(HISTORY_KEYS),
        "history": [
            [p[0]] + [x if math.isfinite(x) else None for x in p[1:]]
            for p in feed.history(history_s)
        ],
    }
    nx, ny = v.get(PLOT_FIELDS["profile_size_x"]), v.get(PLOT_FIELDS["profile_size_y"])
    out["profile_x"] = _finite(arrays.get("profile_x", []), nx)
    out["profile_y"] = _finite(arrays.get("profile_y", []), ny)
    out["cursor_x"] = _finite(arrays.get("cursor_x", []), nx)
    out["cursor_y"] = _finite(arrays.get("cursor_y", []), ny)
    try:
        ax, ay = feed.roi_axes()
    except LookupError:
        out["x_axis"] = out["y_axis"] = out["cursor"] = None
        return out
    out["x_axis"], out["y_axis"] = _axis_map(ax), _axis_map(ay)
    cx, cy = v.get(PLOT_FIELDS["cursor_x"]), v.get(PLOT_FIELDS["cursor_y"])
    out["cursor"] = (
        [to_image(ax, float(cx)), to_image(ay, float(cy))]
        if isinstance(cx, (int, float)) and isinstance(cy, (int, float))
        else None
    )
    return out
