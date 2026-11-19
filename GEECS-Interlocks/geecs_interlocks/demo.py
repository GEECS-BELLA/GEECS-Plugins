"""Stand-ins for the lab: scripted readings, and fake GEECS devices on localhost.

Both replay the same :class:`DemoScript`, made from the interlocks file
itself, so every state shows up within a couple of minutes with no
hardware, no DB and no network:

- each threshold's value swings across its limit about a quarter of the time;
- each alignment's value drifts out of tolerance and back;
- the first threshold's value goes NaN for a few seconds every 90 s;
- each stale-watched device goes quiet (a ``max_age_s`` condition) or drops
  out (a disconnect-only one) for a few seconds now and then.

:class:`DemoSource` feeds the monitor directly (``serve --demo``).
:class:`FakeDevices` runs one ``geecs_core.testing.FakeGeecsServer`` per
device and drives their values, so ``serve --fake-devices`` exercises the
real geecs_core subscribe and reconnect path end to end.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
import time
from collections.abc import Coroutine, Iterable
from dataclasses import dataclass, field
from typing import Any

from geecs_schemas import (
    AlignmentCondition,
    InterlockSet,
    StaleCondition,
    ThresholdCondition,
)

from geecs_interlocks.sources import STAMP_VARIABLE, DeviceReading, ReadingCache

logger = logging.getLogger(__name__)

#: Updates per second from each demo device (a GEECS device's usual 5 Hz).
UPDATE_HZ = 5.0
#: The NaN window: every this many seconds…
NAN_EVERY_S = 90.0
#: …for this long.
NAN_FOR_S = 3.0


@dataclass(frozen=True)
class _Wave:
    """value(t) = center + amplitude * (bias - sin(2πt/period + phase))."""

    center: float
    amplitude: float
    bias: float
    period: float
    phase: float

    def at(self, t: float) -> float:
        return self.center + self.amplitude * (
            self.bias - math.sin(2 * math.pi * t / self.period + self.phase)
        )


@dataclass(frozen=True)
class _Outage:
    """The device is out for *length* seconds every *every* seconds; *quiet*: still connected."""

    every: float
    length: float
    offset: float
    quiet: bool

    def active(self, t: float) -> bool:
        return (t + self.offset) % self.every < self.length


@dataclass
class DemoScript:
    """What every watched device reads at time *t* (seconds since the start)."""

    waves: dict[tuple[str, str], _Wave] = field(default_factory=dict)
    outages: dict[str, _Outage] = field(default_factory=dict)
    nan_variable: tuple[str, str] | None = None
    devices: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_interlocks(cls, interlocks: InterlockSet) -> DemoScript:
        """Script a believable session for every condition in *interlocks*."""
        script = cls()
        for k, interlock in enumerate(interlocks.interlocks):
            condition = interlock.condition
            period, phase = 40.0 + 13.0 * k, float(k)
            if isinstance(condition, ThresholdCondition):
                key = (condition.device, condition.variable)
                amplitude = max(
                    abs(condition.limit) * 0.2, condition.hysteresis * 3, 1e-9
                )
                # The safe side is below a '>' limit and above a '<' one.
                safe = -1.0 if condition.operator in (">", ">=") else 1.0
                script.waves.setdefault(
                    key, _Wave(condition.limit, safe * amplitude, 0.7, period, phase)
                )
                script.nan_variable = script.nan_variable or key
                script._watch(*key)
            elif isinstance(condition, AlignmentCondition):
                target_device = condition.target_device or condition.device
                target = (target_device, condition.target_variable)
                script.waves.setdefault(target, _Wave(0.0, 0.0, 0.0, period, 0.0))
                script.waves.setdefault(
                    (condition.device, condition.variable),
                    _Wave(0.0, condition.tolerance * 1.3, 0.0, period, phase),
                )
                script._watch(condition.device, condition.variable)
                script._watch(*target)
            elif isinstance(condition, StaleCondition):
                quiet = condition.max_age_s is not None
                length = (condition.max_age_s or 3.0) * 2 + 1.0
                every = 50.0 + 7.0 * k
                # First outage 20 s in, so a demo shows one soon.
                script.outages.setdefault(
                    condition.device, _Outage(every, length, every - 20.0, quiet)
                )
                script._watch(condition.device, None)
        return script

    def _watch(self, device: str, variable: str | None) -> None:
        variables = self.devices.setdefault(device, set())
        if variable is not None:
            variables.add(variable)

    def values(self, device: str, t: float) -> dict[str, Any]:
        """Every variable of *device* at *t* (its stamp included)."""
        out: dict[str, Any] = {STAMP_VARIABLE: round(time.time(), 3)}
        for variable in self.devices.get(device, ()):
            wave = self.waves.get((device, variable))
            value = wave.at(t) if wave is not None else 0.0
            if (device, variable) == self.nan_variable and t % NAN_EVERY_S > (
                NAN_EVERY_S - NAN_FOR_S
            ):
                value = math.nan
            out[variable] = value
        return out

    def outage(self, device: str, t: float) -> _Outage | None:
        """The outage *device* is in at *t*, if any."""
        outage = self.outages.get(device)
        return outage if outage is not None and outage.active(t) else None


class DemoSource:
    """A :class:`~geecs_interlocks.sources.Source` replaying a :class:`DemoScript`."""

    def __init__(self, script: DemoScript) -> None:
        self._script = script
        self._cache = ReadingCache()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.monotonic()

    def start(self) -> None:
        """Begin replaying (a thread updating every device at :data:`UPDATE_HZ`)."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="interlock-demo", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop replaying."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)

    def reading(self, device: str) -> DeviceReading:
        """The latest scripted reading of *device*."""
        return self._cache.get(device)

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = time.monotonic() - self._t0
            for device in self._script.devices:
                outage = self._script.outage(device, t)
                if outage is None:
                    self._cache.update(device, self._script.values(device, t))
                elif not outage.quiet:
                    self._cache.disconnected(device, "demo outage")
                # a quiet outage: connected, no updates — the reading ages
            self._stop.wait(1.0 / UPDATE_HZ)


class FakeDevices:
    """One ``FakeGeecsServer`` per scripted device on localhost, on a loop thread of its own.

    :attr:`endpoints` (``device → (host, port)``) is filled by :meth:`start`
    and handed to ``DeviceSource(endpoints=...)``. An outage stops a
    device's server and starts it again on the same port, so the client
    really loses and regains the subscription.
    """

    def __init__(self, script: DemoScript) -> None:
        self._script = script
        self.endpoints: dict[str, tuple[str, int]] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="interlock-fake-devices", daemon=True
        )
        self._driver: asyncio.Future[None] | None = None

    def start(self) -> dict[str, tuple[str, int]]:
        """Start every device's server; return the endpoints."""
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._start(), self._loop)
        future.result(timeout=10)
        self._driver = asyncio.run_coroutine_threadsafe(self._drive(), self._loop)
        return dict(self.endpoints)

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float = 10.0) -> Any:
        """Run *coro* on the devices' loop and wait for it (tests stop and start servers)."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def stop(self) -> None:
        """Stop driving and close every server."""
        if self._driver is not None:
            self._driver.cancel()
        asyncio.run_coroutine_threadsafe(self._stop_all(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)

    async def _start(self) -> None:
        from geecs_core.testing.fake_device_server import (
            FakeGeecsDevice,
            FakeGeecsServer,
        )

        self._fakes: dict[str, FakeGeecsDevice] = {}
        self._servers: dict[str, FakeGeecsServer] = {}
        for device in self._script.devices:
            fake = FakeGeecsDevice(device, self._script.values(device, 0.0))
            server = FakeGeecsServer(fake)
            await server.start()
            self._fakes[device], self._servers[device] = fake, server
            self.endpoints[device] = (server.host, server.port)
            logger.info("fake device %s on %s:%d", device, server.host, server.port)

    async def _drive(self) -> None:
        t0 = time.monotonic()
        down: set[str] = set()
        while True:
            t = time.monotonic() - t0
            for device, fake in self._fakes.items():
                out = self._script.outage(device, t) is not None
                if out and device not in down:
                    down.add(device)
                    await self._servers[device].stop()
                elif not out and device in down:
                    with contextlib.suppress(OSError):
                        await self._servers[device].start()
                        down.discard(device)
                fake.variables.update(self._script.values(device, t))
            await asyncio.sleep(1.0 / UPDATE_HZ)

    async def _stop_all(self) -> None:
        for server in getattr(self, "_servers", {}).values():
            with contextlib.suppress(Exception):
                await server.stop()


def watched_variables(script: DemoScript) -> dict[str, Iterable[str]]:
    """``device → variables`` for a ``DeviceSource`` over the fake devices."""
    return {device: sorted(v) for device, v in script.devices.items()}


__all__ = [
    "DemoScript",
    "DemoSource",
    "FakeDevices",
    "NAN_EVERY_S",
    "UPDATE_HZ",
    "watched_variables",
]
