"""Bounded PVA frame monitors joined to shot readings by the camera timestamp."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from threading import Condition
from time import monotonic
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from geecs_bluesky.exceptions import GeecsDeviceDownError


class FrameSource(Protocol):
    """The production/fake frame source boundary."""

    def open(self) -> None:
        """Open the source."""
        ...

    def close(self) -> None:
        """Release the source."""
        ...

    def wait_connected(self, timeout: float) -> None:
        """Wait for an image or raise a device-down error."""
        ...

    def await_frames(
        self, stamps: Sequence[float], timeout: float
    ) -> dict[float, NDArray]:
        """Return available timestamp-matched images within the timeout."""
        ...


class LiveFrameSource:
    """Hold one monitor; retain owned arrays until their acquisition bin is evaluated."""

    def __init__(self, pv: str, *, keep: int = 64) -> None:
        self.pv = pv
        self._frames: deque[tuple[float, NDArray]] = deque(maxlen=keep)
        self._condition = Condition()
        self._context = None
        self._monitor = None
        self._closed = False

    def open(self) -> None:
        """Open the gating image monitor lazily (p4p is a worker extra)."""
        from p4p.client.thread import Context

        self._context = Context("pva")
        self._monitor = self._context.monitor(self.pv, self._on_update)

    def _on_update(self, value: object) -> None:
        if isinstance(value, Exception):
            return
        with self._condition:
            self._frames.append((float(value.timestamp), np.array(value, copy=True)))
            self._condition.notify_all()

    def close(self) -> None:
        """Release the monitor and transport context, including partial opens."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._monitor is not None:
            self._monitor.close()
        if self._context is not None:
            self._context.close()

    def wait_connected(self, timeout: float) -> None:
        """Wait for the first cached image before the trigger box is armed."""
        with self._condition:
            if (
                not self._condition.wait_for(
                    lambda: bool(self._frames) or self._closed, timeout
                )
                or not self._frames
            ):
                raise GeecsDeviceDownError(self.pv)

    def frame_at(self, stamp: float, tolerance: float = 1e-3) -> NDArray | None:
        """Return a unique timestamp match; never substitute a nearby shot."""
        with self._condition:
            matches = {
                ts: frame for ts, frame in self._frames if abs(ts - stamp) <= tolerance
            }
            return next(iter(matches.values())) if len(matches) == 1 else None

    def await_frames(
        self, stamps: Sequence[float], timeout: float
    ) -> dict[float, NDArray]:
        """Wait a bounded time for the exact requested frames; omit missing shots."""
        deadline = monotonic() + timeout
        with self._condition:
            while True:
                found = {
                    stamp: frame
                    for stamp in stamps
                    if (frame := self.frame_at(stamp)) is not None
                }
                remaining = deadline - monotonic()
                if len(found) == len(set(stamps)) or remaining <= 0 or self._closed:
                    return found
                self._condition.wait(remaining)
