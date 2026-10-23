"""One oscilloscope's live state: its settings and status, monitored; its traces, read on request.

A :class:`ScopeFeed` subscribes to the scope's readbacks on first use and
keeps the latest of each; :meth:`ScopeFeed.snapshot` is the JSON the page
renders. Every viewer shares one feed. The traces come through the
camera's :class:`~geecs_panels.feed.ArraysSource` (one background read at a
time, at most one per interval, only while a page asks) and are decimated
for the page by :func:`traces_payload`.

The transport (``subscribe``) is injected, as for the camera.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from geecs_panels.feed import Subscribe, Subscription
from geecs_panels.scope import (
    CHANNEL_FIELDS,
    CHANNELS,
    SCOPE_FIELDS,
    decimate,
    monitored,
)

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
            if key == "count":
                self.values.pop(key, None)
            return
        previous = self.values.get(key)
        self.connected[key] = True
        self.values[key] = value
        # The first value after (re)connecting is the count's position, not a capture.
        if key == "count" and previous is not None and value != previous:
            self._captured_at = self.clock()

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
            }
            for ch in CHANNELS
        ]
        age = None if self._captured_at is None else self.clock() - self._captured_at
        return {
            "reachable": reachable,
            "missing": sorted(k for k, ok in self.connected.items() if not ok),
            "scope": scope,
            "channels": channels,
            "capture_age": age,
        }


def traces_payload(
    arrays: dict[str, list[float]], columns: int = TRACE_COLUMNS
) -> dict[str, object]:
    """Each read channel's trace against the time axis, decimated to *columns*."""
    times = arrays.get("t", [])
    out: dict[str, object] = {}
    for ch in CHANNELS:
        if ch in arrays:
            xs, ys = decimate(times, arrays[ch], columns)
            out[ch] = {"t": xs, "v": ys}
    return {"traces": out, "samples": len(times)}
