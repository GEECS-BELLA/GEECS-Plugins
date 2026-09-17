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


class DocCollector:
    """Collect RunEngine documents; pick the ``primary`` stream's events."""

    def __init__(self) -> None:
        from collections import defaultdict

        self.docs: dict[str, list[dict]] = defaultdict(list)
        self.ordered: list[tuple[str, dict]] = []  # every document, in order

    def __call__(self, name: str, doc: dict) -> None:
        self.docs[name].append(doc)
        self.ordered.append((name, doc))

    def primary_events(self) -> list[dict]:
        uids = {d["uid"] for d in self.docs["descriptor"] if d["name"] == "primary"}
        return [e for e in self.docs["event"] if e["descriptor"] in uids]


def read_scan_info(path: Any) -> dict[str, str]:
    """The ``[Scan Info]`` section of a ScanInfo ini, values unquoted."""
    from configparser import ConfigParser

    parser = ConfigParser()
    parser.optionxform = str  # type: ignore[assignment]
    parser.read(path)
    return {k: v.strip('"') for k, v in parser.items("Scan Info")}


def wait_for_native_files(directory: Any, expected: int, timeout: float = 15.0) -> list:
    """Every expected native file exists and has stopped growing (two agreeing stats).

    Any regular file counts — the device's own format, not a fixed suffix.
    """
    import time
    from pathlib import Path

    directory = Path(directory)
    deadline = time.monotonic() + timeout
    while True:
        files = (
            sorted(f for f in directory.iterdir() if f.is_file())
            if directory.is_dir()
            else []
        )
        sizes = [f.stat().st_size for f in files]
        if len(files) >= expected and all(sizes):
            time.sleep(0.5)
            if [f.stat().st_size for f in files] == sizes:
                return files
        if time.monotonic() > deadline:
            raise AssertionError(
                f"{directory}: {len(files)} files after {timeout:.0f} s, expected {expected}"
            )
        time.sleep(0.5)
