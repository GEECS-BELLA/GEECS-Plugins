"""Tests for the strict single-shot acquisition path.

No free-running firer here: the fake device fires **only** when the plan's
``fire_shot`` callable executes, proving the plan owns every shot.
"""

from __future__ import annotations

import asyncio
import threading

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine

from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.exceptions import GeecsTriggerTimeoutError
from geecs_bluesky.plans.step_scan import geecs_step_scan
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer


def _run_server(
    device: FakeGeecsDevice, ready: threading.Event, host_port: list
) -> None:
    """Thread target: serve the fake device; never fire on its own."""

    async def _main() -> None:
        async with FakeGeecsServer(device) as srv:
            host_port.extend([srv.host, srv.port])
            ready.set()
            await asyncio.sleep(3600)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_main())
    except Exception:
        pass
    finally:
        loop.close()


def _setup_scan() -> tuple[
    FakeGeecsDevice, GeecsMotor, GeecsGenericDetector, RunEngine, list
]:
    fake = FakeGeecsDevice(
        name="U_Combined",
        variables={"Position (mm)": 0.0, "Sig": 1.0, "acq_timestamp": 1000.0},
    )
    ready = threading.Event()
    host_port: list = []
    threading.Thread(
        target=_run_server, args=(fake, ready, host_port), daemon=True
    ).start()
    ready.wait(timeout=5.0)
    assert host_port, "Server failed to start"
    host, port = host_port

    motor = GeecsMotor("U_Combined", "Position (mm)", host, port, name="scan_motor")
    cam = GeecsGenericDetector("U_Combined", ["Sig"], host, port, name="cam")
    cam.configure_shot_id(rep_rate_hz=1.0)

    events: list[dict] = []
    RE = RunEngine()
    RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)
    for dev in (motor, cam):
        asyncio.run_coroutine_threadsafe(dev.connect(), RE._loop).result(timeout=10)
    return fake, motor, cam, RE, events


def test_strict_scan_fires_each_shot_itself() -> None:
    """One fire → one complete row; no free-running trigger required."""
    fake, motor, cam, RE, events = _setup_scan()
    fire_count = 0

    def fire():
        nonlocal fire_count
        fire_count += 1
        fake.fire_shot()
        yield from bps.null()

    RE(
        geecs_step_scan(
            motor=motor,
            positions=[0.0, 1.0],
            detectors=[cam],
            shots_per_step=2,
            fire_shot=fire,
        )
    )

    assert fire_count == 4
    assert len(events) == 4
    # Plan-owned shots are consecutive: the trigger only ticks when fired
    assert [ev["data"]["cam-shot_id"] for ev in events] == [1, 2, 3, 4]
    for ev in events:
        assert ev["data"]["cam-valid"] is True
        assert ev["data"]["cam-shot_offset"] == 0


def test_strict_scan_hard_fails_when_device_misses_the_shot() -> None:
    """A device not responding to the plan's own shot aborts the scan."""
    fake, motor, cam, RE, events = _setup_scan()
    cam._trigger_timeout = 0.5  # keep the failure fast

    def dead_fire():
        # Trigger source broken: nothing fires, the camera never advances
        yield from bps.null()

    with pytest.raises(Exception) as excinfo:
        RE(
            geecs_step_scan(
                motor=motor,
                positions=[0.0],
                detectors=[cam],
                shots_per_step=1,
                fire_shot=dead_fire,
            )
        )
    # The timeout names the offending device through the status failure
    assert "U_Combined" in str(excinfo.value) or isinstance(
        excinfo.value.__cause__, GeecsTriggerTimeoutError
    )
    assert events == []
