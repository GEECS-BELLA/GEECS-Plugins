"""Tests for GeecsMotor, GeecsTriggerable (via GeecsCameraBase), and geecs_step_scan.

All tests use FakeGeecsServer — no real hardware required.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from bluesky import RunEngine
from bluesky.protocols import Movable, Triggerable

from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.exceptions import GeecsTriggerTimeoutError
from geecs_bluesky.devices.camera import GeecsCameraBase
from geecs_bluesky.plans.step_scan import geecs_step_scan
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def motor_device() -> FakeGeecsDevice:
    return FakeGeecsDevice(
        name="U_TestMotor",
        variables={"Position (mm)": 0.0},
    )


@pytest.fixture
def camera_device() -> FakeGeecsDevice:
    return FakeGeecsDevice(
        name="U_TestCam",
        variables={
            "SavedFile": "/data/shot001.tif",
            "acq_timestamp": 1000.0,
        },
    )


@pytest.fixture
def combined_device() -> FakeGeecsDevice:
    """One fake device with motor + camera variables for integration tests."""
    return FakeGeecsDevice(
        name="U_Combined",
        variables={
            "Position (mm)": 0.0,
            "SavedFile": "/data/shot001.tif",
            "acq_timestamp": 1000.0,
        },
    )


# ---------------------------------------------------------------------------
# GeecsMotor tests
# ---------------------------------------------------------------------------


class TestGeecsMotor:
    async def test_set_moves_to_position(self, motor_device: FakeGeecsDevice) -> None:
        async with FakeGeecsServer(motor_device) as srv:
            motor = GeecsMotor(
                "U_TestMotor", "Position (mm)", srv.host, srv.port, name="test_motor"
            )
            await motor.connect()
            await motor.set(5.0)
            reading = await motor.read()
            assert reading["test_motor-position"]["value"] == pytest.approx(5.0)

    async def test_motor_implements_movable(
        self, motor_device: FakeGeecsDevice
    ) -> None:
        async with FakeGeecsServer(motor_device) as srv:
            motor = GeecsMotor(
                "U_TestMotor", "Position (mm)", srv.host, srv.port, name="test_motor"
            )
            await motor.connect()
            assert isinstance(motor, Movable)

    async def test_set_returns_async_status(
        self, motor_device: FakeGeecsDevice
    ) -> None:
        """set() must return something awaitable (AsyncStatus)."""
        from ophyd_async.core import AsyncStatus

        async with FakeGeecsServer(motor_device) as srv:
            motor = GeecsMotor(
                "U_TestMotor", "Position (mm)", srv.host, srv.port, name="test_motor"
            )
            await motor.connect()
            status = motor.set(3.0)
            assert isinstance(status, AsyncStatus)
            await status

    async def test_multiple_sequential_moves(
        self, motor_device: FakeGeecsDevice
    ) -> None:
        async with FakeGeecsServer(motor_device) as srv:
            motor = GeecsMotor(
                "U_TestMotor", "Position (mm)", srv.host, srv.port, name="test_motor"
            )
            await motor.connect()
            for pos in (1.0, 2.0, 3.0):
                await motor.set(pos)
            reading = await motor.read()
            assert reading["test_motor-position"]["value"] == pytest.approx(3.0)

    async def test_motor_position_in_describe(
        self, motor_device: FakeGeecsDevice
    ) -> None:
        async with FakeGeecsServer(motor_device) as srv:
            motor = GeecsMotor(
                "U_TestMotor",
                "Position (mm)",
                srv.host,
                srv.port,
                name="test_motor",
                units="mm",
            )
            await motor.connect()
            desc = await motor.describe()
            assert "test_motor-position" in desc
            assert desc["test_motor-position"]["dtype"] == "number"


# ---------------------------------------------------------------------------
# GeecsTriggerable tests (via GeecsCameraBase)
# ---------------------------------------------------------------------------


class TestGeecsTriggerable:
    async def test_trigger_completes_when_shot_fired(
        self, camera_device: FakeGeecsDevice
    ) -> None:
        """trigger() must resolve once acq_timestamp advances."""
        async with FakeGeecsServer(camera_device) as srv:
            cam = GeecsCameraBase("U_TestCam", srv.host, srv.port, name="test_cam")
            await cam.connect()

            async def fire_later() -> None:
                await asyncio.sleep(0.1)
                camera_device.fire_shot()

            asyncio.create_task(fire_later())
            status = cam.trigger()
            await asyncio.wait_for(status, timeout=2.0)
            assert status.done

    async def test_trigger_timeout_if_no_shot(
        self, camera_device: FakeGeecsDevice
    ) -> None:
        """trigger() must raise GeecsTriggerTimeoutError when no shot arrives."""
        async with FakeGeecsServer(camera_device) as srv:
            cam = GeecsCameraBase("U_TestCam", srv.host, srv.port, name="test_cam")
            cam._trigger_timeout = 0.3  # short timeout for fast test
            await cam.connect()

            status = cam.trigger()
            with pytest.raises(GeecsTriggerTimeoutError):
                await asyncio.wait_for(status, timeout=2.0)

    async def test_trigger_then_read_filepath(
        self, camera_device: FakeGeecsDevice
    ) -> None:
        """After trigger completes, read() returns the updated file path."""
        async with FakeGeecsServer(camera_device) as srv:
            cam = GeecsCameraBase("U_TestCam", srv.host, srv.port, name="test_cam")
            await cam.connect()

            async def fire_and_update() -> None:
                await asyncio.sleep(0.05)
                camera_device.variables["SavedFile"] = "/data/shot002.tif"
                camera_device.fire_shot()

            asyncio.create_task(fire_and_update())
            await asyncio.wait_for(cam.trigger(), timeout=2.0)
            reading = await cam.read()
            assert reading["test_cam-filepath"]["value"] == "/data/shot002.tif"

    async def test_multiple_triggers(self, camera_device: FakeGeecsDevice) -> None:
        """trigger() should succeed for each successive shot."""
        async with FakeGeecsServer(camera_device) as srv:
            cam = GeecsCameraBase("U_TestCam", srv.host, srv.port, name="test_cam")
            await cam.connect()

            for _i in range(3):

                async def fire() -> None:
                    await asyncio.sleep(0.05)
                    camera_device.fire_shot()

                asyncio.create_task(fire())
                await asyncio.wait_for(cam.trigger(), timeout=2.0)


# ---------------------------------------------------------------------------
# GeecsCameraBase protocol tests
# ---------------------------------------------------------------------------


class TestGeecsCameraBase:
    async def test_implements_triggerable(self, camera_device: FakeGeecsDevice) -> None:
        async with FakeGeecsServer(camera_device) as srv:
            cam = GeecsCameraBase("U_TestCam", srv.host, srv.port, name="test_cam")
            await cam.connect()
            assert isinstance(cam, Triggerable)

    async def test_describe_includes_filepath(
        self, camera_device: FakeGeecsDevice
    ) -> None:
        async with FakeGeecsServer(camera_device) as srv:
            cam = GeecsCameraBase("U_TestCam", srv.host, srv.port, name="test_cam")
            await cam.connect()
            desc = await cam.describe()
            assert "test_cam-filepath" in desc
            assert desc["test_cam-filepath"]["dtype"] == "string"

    async def test_from_db_raises_without_mysql(
        self, camera_device: FakeGeecsDevice
    ) -> None:
        """from_db() must raise ImportError (or RuntimeError) without a DB."""
        import sys
        import unittest.mock

        # Simulate missing mysql.connector
        with unittest.mock.patch.dict(
            sys.modules, {"mysql": None, "mysql.connector": None}
        ):
            with pytest.raises((ImportError, Exception)):
                GeecsCameraBase.from_db("U_NonExistent")


# ---------------------------------------------------------------------------
# geecs_step_scan integration test (synchronous RunEngine)
# ---------------------------------------------------------------------------


def _run_server_with_shot_firer(
    device: FakeGeecsDevice,
    ready: threading.Event,
    host_port: list,
    shot_interval: float = 0.15,
) -> None:
    """Background thread: run the fake server and fire shots periodically."""

    async def _main() -> None:
        async with FakeGeecsServer(device) as srv:
            host_port.extend([srv.host, srv.port])
            ready.set()
            try:
                while True:
                    await asyncio.sleep(shot_interval)
                    device.fire_shot()
            except asyncio.CancelledError:
                pass

    loop = asyncio.new_event_loop()
    task_holder: list = []

    async def _wrapper() -> None:
        task = loop.create_task(_main())
        task_holder.append(task)
        await task

    try:
        loop.run_until_complete(_wrapper())
    except Exception:
        pass
    finally:
        loop.close()


def test_geecs_step_scan_collects_events(combined_device: FakeGeecsDevice) -> None:
    """End-to-end: step scan over 2 positions × 2 shots = 4 event documents."""
    ready = threading.Event()
    host_port: list = []

    srv_thread = threading.Thread(
        target=_run_server_with_shot_firer,
        args=(combined_device, ready, host_port),
        daemon=True,
    )
    srv_thread.start()
    ready.wait(timeout=5.0)
    assert host_port, "Server failed to start"

    host, port = host_port[0], host_port[1]

    # The RunEngine owns a persistent asyncio loop running in a background thread.
    # Devices must be connected in *that* loop so their transports are registered
    # with the same selector the RE uses for all I/O.  Use run_coroutine_threadsafe
    # to schedule the connect coroutines into the RE's loop synchronously.
    motor = GeecsMotor("U_Combined", "Position (mm)", host, port, name="scan_motor")
    cam = GeecsCameraBase(
        "U_Combined",
        host,
        port,
        name="scan_cam",
        filepath_variable="SavedFile",
    )

    events: list[dict] = []
    RE = RunEngine()
    RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)

    # Connect in RE's loop (blocks until both connects complete)
    asyncio.run_coroutine_threadsafe(motor.connect(), RE._loop).result(timeout=10)
    asyncio.run_coroutine_threadsafe(cam.connect(), RE._loop).result(timeout=10)

    # Run the step scan
    RE(
        geecs_step_scan(
            motor=motor,
            positions=[0.0, 1.0],
            detectors=[cam],
            shots_per_step=2,
            md={"test": True},
        )
    )

    # 2 positions × 2 shots = 4 events
    assert len(events) == 4, f"Expected 4 events, got {len(events)}"
    for ev in events:
        assert "scan_cam-filepath" in ev["data"]
        assert "scan_motor-position" in ev["data"]
