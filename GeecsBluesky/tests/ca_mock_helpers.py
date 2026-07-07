"""Shared helpers for CA-mock plan tests (no gateway, no network).

The direct backend's ``FakeGeecsServer`` role is played by ophyd-async mock
backends: ``set_mock_value`` on ``acq_timestamp`` is a shot, a pacer coroutine
scheduled on the RunEngine's event loop is the free-running trigger, and a
setpoint→readback follower stands in for GEECS's native move convergence.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from typing import Any, Sequence

from bluesky import RunEngine
from ophyd_async.core import callback_on_mock_put, set_mock_value


def connect_mock(run_engine: RunEngine, *devices: Any) -> None:
    """Connect devices with mock backends in the RE's persistent event loop."""
    for device in devices:
        asyncio.run_coroutine_threadsafe(
            device.connect(mock=True), run_engine._loop
        ).result(timeout=10.0)


def follow_setpoint(motor: Any) -> None:
    """Make the mock readback track :SP puts (GEECS native convergence stand-in)."""
    callback_on_mock_put(
        motor._setpoint,
        lambda value, **kwargs: set_mock_value(motor.position, value),
    )


def start_pacer(
    run_engine: RunEngine,
    targets: Sequence[tuple[Any, float]],
    *,
    initial_delay: float = 1.0,
    interval: float = 0.3,
    period: float = 1.0,
) -> Future:
    """Advance each (device, t0) acq_timestamp on the RE loop — the fake trigger.

    Starts after ``initial_delay`` (so a t0-sync stage sees static caches),
    then advances every ``interval`` seconds by ``period`` (the simulated
    trigger period).  Runs inside the RE event loop, so monitor callbacks and
    queues stay single-threaded.  Cancel the returned future to stop.
    """

    async def pace() -> None:
        ticks = 0
        await asyncio.sleep(initial_delay)
        while True:
            ticks += 1
            for device, t0 in targets:
                set_mock_value(device.acq_timestamp, t0 + ticks * period)
            await asyncio.sleep(interval)

    return asyncio.run_coroutine_threadsafe(pace(), run_engine._loop)
