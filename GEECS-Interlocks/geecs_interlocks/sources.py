"""Where the readings come from: GEECS devices over geecs_core, or a stand-in.

A :class:`Source` keeps the latest :class:`DeviceReading` of every watched
device and answers :meth:`Source.reading` from any thread. The monitor never
touches a socket: it reads these snapshots on its own tick.

:class:`DeviceSource` is the real one: one ``geecs_core.GeecsDevice``
subscription per device. geecs_core keeps a subscription alive through
device restarts once it is up; the first connection is this module's job,
retried in the background until it succeeds, so a device that is off when
the server starts is reported disconnected — never skipped — and picked up
when it comes back.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Seconds between two attempts to make a device's first connection.
CONNECT_RETRY_S = 5.0
#: What a device watched only for staleness subscribes to: the stamp every
#: GEECS device publishes with each update.
STAMP_VARIABLE = "acq_timestamp"


@dataclass(frozen=True)
class DeviceReading:
    """One device's latest state, as the monitor reads it.

    Attributes
    ----------
    values :
        The latest value of each subscribed variable.
    connected :
        Whether the subscription is up now.
    updated :
        ``time.monotonic()`` of the last update; ``None`` before the first.
    error :
        Why the device is not connected, when known (shown to operators).
    """

    values: Mapping[str, Any] = field(default_factory=dict)
    connected: bool = False
    updated: float | None = None
    error: str = ""


#: The reading of a device nothing has been heard from.
NOT_CONNECTED = DeviceReading()


class Source(Protocol):
    """What the monitor reads: the latest reading per device, from any thread."""

    def start(self) -> None:
        """Begin collecting readings (returns at once)."""

    def stop(self) -> None:
        """Stop collecting and release every connection."""

    def reading(self, device: str) -> DeviceReading:
        """The latest reading of *device* (:data:`NOT_CONNECTED` if none yet)."""


class ReadingCache:
    """The thread-safe latest-reading store every source shares."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._readings: dict[str, DeviceReading] = {}

    def get(self, device: str) -> DeviceReading:
        """The latest reading of *device*."""
        with self._lock:
            return self._readings.get(device, NOT_CONNECTED)

    def update(self, device: str, frame: Mapping[str, Any]) -> None:
        """Merge *frame* into *device*'s values: connected, updated now."""
        with self._lock:
            old = self._readings.get(device, NOT_CONNECTED)
            values = {**old.values, **frame}
            self._readings[device] = DeviceReading(values, True, time.monotonic())

    def disconnected(self, device: str, error: str = "") -> None:
        """Mark *device* down, keeping its last values and update time."""
        with self._lock:
            old = self._readings.get(device, NOT_CONNECTED)
            self._readings[device] = DeviceReading(
                old.values, False, old.updated, error
            )


class DeviceSource:
    """GEECS devices over direct UDP/TCP, one ``GeecsDevice`` subscription each.

    Parameters
    ----------
    variables :
        Device name → the variables to subscribe on it. A device with none
        (watched only for staleness) subscribes :data:`STAMP_VARIABLE`.
    endpoints :
        Device name → ``(host, port)``, skipping the experiment DB lookup
        (the local fake devices; off-network tests).
    """

    def __init__(
        self,
        variables: Mapping[str, Iterable[str]],
        endpoints: Mapping[str, tuple[str, int]] | None = None,
    ) -> None:
        self._variables = {
            name: sorted(set(v)) or [STAMP_VARIABLE] for name, v in variables.items()
        }
        self._endpoints = dict(endpoints or {})
        self._cache = ReadingCache()
        self._devices: dict[
            str, Any
        ] = {}  # Any: geecs_core.GeecsDevice, imported lazily
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Connect every device in the background, retrying the ones that fail."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._connect_loop, name="interlock-connect", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop retrying and close every device."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(CONNECT_RETRY_S + 1)
        for name, device in list(self._devices.items()):
            try:
                device.close()
            except Exception:
                logger.exception("closing %s failed", name)
        self._devices.clear()

    def reading(self, device: str) -> DeviceReading:
        """The latest reading of *device*, its connection state from geecs_core."""
        reading = self._cache.get(device)
        live = self._devices.get(device)
        if live is not None and reading.connected and not live.state.get("connected"):
            self._cache.disconnected(device, "subscription dropped (reconnecting)")
            return self._cache.get(device)
        return reading

    def _connect_loop(self) -> None:
        pending = list(self._variables)
        while pending and not self._stop.is_set():
            pending = [name for name in pending if not self._connect(name)]
            if pending:
                self._stop.wait(CONNECT_RETRY_S)

    def _connect(self, name: str) -> bool:
        from geecs_core import GeecsDevice

        host_port = self._endpoints.get(name)
        try:
            if host_port is None:
                device = GeecsDevice(name)
            else:
                device = GeecsDevice(name, host=host_port[0], port=host_port[1])
        except Exception as exc:  # noqa: BLE001 — a lookup failure is a disconnect
            self._cache.disconnected(name, f"lookup failed: {exc}")
            logger.warning("%s: lookup failed (%s); retrying", name, exc)
            return False
        try:
            device.subscribe(
                self._variables[name],
                on_update=lambda frame, name=name: self._on_update(name, frame),
            )
        except Exception as exc:  # noqa: BLE001 — retried; never fatal
            device.close()
            self._cache.disconnected(name, f"cannot connect: {exc}")
            logger.warning("%s: cannot connect (%s); retrying", name, exc)
            return False
        self._devices[name] = device
        logger.info("%s: subscribed to %s", name, ", ".join(self._variables[name]))
        return True

    def _on_update(self, name: str, frame: Mapping[str, Any]) -> None:
        # Runs on geecs_core's loop thread: store and return.
        self._cache.update(
            name, {k: v for k, v in frame.items() if k not in ("shot number",)}
        )


__all__ = [
    "CONNECT_RETRY_S",
    "NOT_CONNECTED",
    "DeviceReading",
    "DeviceSource",
    "ReadingCache",
    "STAMP_VARIABLE",
    "Source",
]
