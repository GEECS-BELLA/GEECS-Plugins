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

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from geecs_panels.feed import Subscribe, Subscription
from geecs_panels.scope import (
    CHANNEL_FIELDS,
    CHANNELS,
    SCOPE_FIELDS,
    STAT_FIELDS,
    decimate,
    monitored,
    split_channels,
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
    #: enum setting suffix → its PV's choices, as the transport read them
    enums: dict[str, list[str]] = field(default_factory=dict)
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
    arrays: dict[str, list[float]], present: list[str], columns: int = TRACE_COLUMNS
) -> dict[str, object]:
    """Each channel with data, against the time axis, decimated to *columns*.

    *arrays* holds ``t`` (the time axis), ``data`` (the capture,
    [channels, samples]) and ``shape`` ([channels, samples]); *present* names
    the capture's rows. A disabled channel's row is all NaN and is left out.
    """
    times = arrays.get("t", [])
    shape = arrays.get("shape", [])
    if len(shape) != 2:
        return {"traces": {}, "samples": 0}
    rows = split_channels(arrays.get("data", []), int(shape[0]), int(shape[1]))
    out: dict[str, object] = {}
    for ch, row in zip(present, rows):
        if any(v == v for v in row):  # not all NaN
            xs, ys = decimate(times, row, columns)
            out[ch] = {"t": xs, "v": ys}
    return {"traces": out, "samples": int(shape[1])}
