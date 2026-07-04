"""Helpers for bounded fake-server tests."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, AsyncExitStack
from types import TracebackType
from typing import Any

from bluesky import RunEngine
from ophyd_async.core import Device

from geecs_ca_gateway.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer

FireCallback = Callable[[Sequence[FakeGeecsDevice]], None]


class BackgroundFakeServers(AbstractContextManager["BackgroundFakeServers"]):
    """Run one or more fake servers in a stoppable background thread."""

    def __init__(
        self,
        devices: FakeGeecsDevice | Sequence[FakeGeecsDevice],
        *,
        fire: FireCallback | None = None,
        initial_delay: float = 0.0,
        interval: float = 0.1,
        startup_timeout: float = 5.0,
        shutdown_timeout: float = 5.0,
    ) -> None:
        self._devices = (
            [devices] if isinstance(devices, FakeGeecsDevice) else list(devices)
        )
        self._fire = fire
        self._initial_delay = initial_delay
        self._interval = interval
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._errors: queue.SimpleQueue[BaseException] = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run,
            name="fake-geecs-server",
            daemon=True,
        )
        self.endpoints: list[tuple[str, int]] = []

    @property
    def endpoint(self) -> tuple[str, int]:
        """Return the first server endpoint."""
        return self.endpoints[0]

    def __enter__(self) -> "BackgroundFakeServers":
        self._thread.start()
        if not self._ready.wait(timeout=self._startup_timeout):
            self._stop.set()
            self._thread.join(timeout=self._shutdown_timeout)
            self._raise_thread_error()
            raise TimeoutError("FakeGeecsServer failed to start")
        self._raise_thread_error()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        self._stop.set()
        self._thread.join(timeout=self._shutdown_timeout)
        if self._thread.is_alive() and exc_type is None:
            raise TimeoutError("FakeGeecsServer thread failed to stop")
        if exc_type is None:
            self._raise_thread_error()
        return None

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:
            self._errors.put(exc)
            self._ready.set()

    async def _main(self) -> None:
        async with AsyncExitStack() as stack:
            servers = [
                await stack.enter_async_context(FakeGeecsServer(device))
                for device in self._devices
            ]
            self.endpoints = [(server.host, server.port) for server in servers]
            self._ready.set()

            if self._initial_delay:
                await self._sleep_until_stopped(self._initial_delay)

            while not self._stop.is_set():
                if self._fire is not None:
                    self._fire(self._devices)
                await self._sleep_until_stopped(self._interval)

    async def _sleep_until_stopped(self, duration: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration
        while not self._stop.is_set():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.05))

    def _raise_thread_error(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("FakeGeecsServer thread failed") from error


def connect_devices(re: RunEngine, *devices: Device, timeout: float = 10.0) -> None:
    """Connect devices on the RunEngine event loop."""
    for device in devices:
        asyncio.run_coroutine_threadsafe(device.connect(), re._loop).result(
            timeout=timeout
        )


def disconnect_devices(re: RunEngine, *devices: Any, timeout: float = 10.0) -> None:
    """Best-effort disconnect for devices connected on the RunEngine loop."""
    for device in devices:
        disconnect = getattr(device, "disconnect", None)
        if disconnect is None:
            continue
        asyncio.run_coroutine_threadsafe(disconnect(), re._loop).result(timeout=timeout)
