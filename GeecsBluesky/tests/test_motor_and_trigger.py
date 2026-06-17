"""Tests for GeecsMotor, GeecsTriggerable, and geecs_step_scan.

All tests use FakeGeecsServer — no real hardware required.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from bluesky import RunEngine
from bluesky.protocols import Movable, Triggerable

from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.devices.snapshot import GeecsSnapshotReadable
from geecs_bluesky.exceptions import GeecsTriggerTimeoutError
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
def detector_device() -> FakeGeecsDevice:
    return FakeGeecsDevice(
        name="U_TestDet",
        variables={
            "Signal": 1.0,
            "acq_timestamp": 1000.0,
        },
    )


@pytest.fixture
def combined_device() -> FakeGeecsDevice:
    """One fake device with motor + detector variables for integration tests."""
    return FakeGeecsDevice(
        name="U_Combined",
        variables={
            "Position (mm)": 0.0,
            "Signal": 1.0,
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
# GeecsTriggerable tests
# ---------------------------------------------------------------------------


class TestGeecsTriggerable:
    async def test_trigger_completes_when_shot_fired(
        self, detector_device: FakeGeecsDevice
    ) -> None:
        """trigger() must resolve once acq_timestamp advances."""
        async with FakeGeecsServer(detector_device) as srv:
            det = GeecsGenericDetector(
                "U_TestDet", ["Signal"], srv.host, srv.port, name="test_det"
            )
            await det.connect()

            async def fire_later() -> None:
                await asyncio.sleep(0.1)
                detector_device.fire_shot()

            asyncio.create_task(fire_later())
            status = det.trigger()
            await asyncio.wait_for(status, timeout=2.0)
            assert status.done

    async def test_trigger_timeout_if_no_shot(
        self, detector_device: FakeGeecsDevice
    ) -> None:
        """trigger() must raise GeecsTriggerTimeoutError when no shot arrives."""
        async with FakeGeecsServer(detector_device) as srv:
            det = GeecsGenericDetector(
                "U_TestDet", ["Signal"], srv.host, srv.port, name="test_det"
            )
            det._trigger_timeout = 0.3  # short timeout for fast test
            await det.connect()

            status = det.trigger()
            with pytest.raises(GeecsTriggerTimeoutError):
                await asyncio.wait_for(status, timeout=2.0)

    async def test_trigger_then_read_signal(
        self, detector_device: FakeGeecsDevice
    ) -> None:
        """After trigger completes, read() returns the updated signal."""
        async with FakeGeecsServer(detector_device) as srv:
            det = GeecsGenericDetector(
                "U_TestDet", ["Signal"], srv.host, srv.port, name="test_det"
            )
            await det.connect()

            async def fire_and_update() -> None:
                await asyncio.sleep(0.05)
                detector_device.variables["Signal"] = 2.0
                detector_device.fire_shot()

            asyncio.create_task(fire_and_update())
            await asyncio.wait_for(det.trigger(), timeout=2.0)
            reading = await det.read()
            assert reading["test_det-signal"]["value"] == pytest.approx(2.0)

    async def test_multiple_triggers(self, detector_device: FakeGeecsDevice) -> None:
        """trigger() should succeed for each successive shot."""
        async with FakeGeecsServer(detector_device) as srv:
            det = GeecsGenericDetector(
                "U_TestDet", ["Signal"], srv.host, srv.port, name="test_det"
            )
            await det.connect()

            for _i in range(3):

                async def fire() -> None:
                    await asyncio.sleep(0.05)
                    detector_device.fire_shot()

                asyncio.create_task(fire())
                await asyncio.wait_for(det.trigger(), timeout=2.0)

    async def test_implements_triggerable(
        self, detector_device: FakeGeecsDevice
    ) -> None:
        async with FakeGeecsServer(detector_device) as srv:
            det = GeecsGenericDetector(
                "U_TestDet", ["Signal"], srv.host, srv.port, name="test_det"
            )
            await det.connect()
            assert isinstance(det, Triggerable)


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
    det = GeecsGenericDetector(
        "U_Combined",
        ["Signal"],
        host,
        port,
        name="scan_det",
    )
    async_snapshot = GeecsSnapshotReadable(
        "U_Combined",
        ["Position (mm)"],
        host,
        port,
        name="async_snapshot",
    )

    events: list[dict] = []
    RE = RunEngine()
    RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)

    # Connect in RE's loop (blocks until both connects complete)
    asyncio.run_coroutine_threadsafe(motor.connect(), RE._loop).result(timeout=10)
    asyncio.run_coroutine_threadsafe(det.connect(), RE._loop).result(timeout=10)
    asyncio.run_coroutine_threadsafe(async_snapshot.connect(), RE._loop).result(
        timeout=10
    )

    # Run the step scan
    RE(
        geecs_step_scan(
            motor=motor,
            positions=[0.0, 1.0],
            detectors=[det, async_snapshot],
            shots_per_step=2,
            md={"test": True},
        )
    )

    # 2 positions × 2 shots = 4 events
    assert len(events) == 4, f"Expected 4 events, got {len(events)}"
    for ev in events:
        assert "scan_det-signal" in ev["data"]
        assert "scan_motor-position" in ev["data"]
        assert "async_snapshot-position__mm" in ev["data"]
        assert "bin_number" in ev["data"]
        assert "shot_index_in_bin" in ev["data"]
        assert "scan_event_index" in ev["data"]

    assert [ev["data"]["bin_number"] for ev in events] == [1, 1, 2, 2]
    assert [ev["data"]["shot_index_in_bin"] for ev in events] == [1, 2, 1, 2]
    assert [ev["data"]["scan_event_index"] for ev in events] == [1, 2, 3, 4]


def test_geecs_step_scan_statistics_collection(
    combined_device: FakeGeecsDevice,
) -> None:
    """motor=None, positions=[None]: N shots at fixed settings (former NOSCAN).

    Routes statistics collection through the same plan as a motor scan; the
    only difference is no scan variable is moved and there is a single bin.
    """
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

    det = GeecsGenericDetector("U_Combined", ["Signal"], host, port, name="scan_det")

    events: list[dict] = []
    RE = RunEngine()
    RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)
    asyncio.run_coroutine_threadsafe(det.connect(), RE._loop).result(timeout=10)

    RE(
        geecs_step_scan(
            motor=None,
            positions=[None],
            detectors=[det],
            shots_per_step=3,
            md={"test": True},
        )
    )

    assert len(events) == 3, f"Expected 3 events, got {len(events)}"
    for ev in events:
        assert "scan_det-signal" in ev["data"]
        # No scan variable was moved — no motor column
        assert not any(k.endswith("-position") for k in ev["data"])
        assert ev["data"]["bin_number"] == 1
    assert [ev["data"]["shot_index_in_bin"] for ev in events] == [1, 2, 3]
    assert [ev["data"]["scan_event_index"] for ev in events] == [1, 2, 3]


async def test_nonscalar_generic_detector_logs_save_path_and_acq_timestamp(
    tmp_path, combined_device: FakeGeecsDevice
) -> None:
    """Non-scalar detectors emit scanner save path and device acq_timestamp."""
    combined_device.variables.update(
        {
            "Signal": 1.0,
            "localsavingpath": "",
            "save": "off",
        }
    )
    async with FakeGeecsServer(combined_device) as srv:
        save_path = tmp_path / "Scan047" / "U_Combined"
        det = GeecsGenericDetector(
            "U_Combined",
            ["Signal"],
            srv.host,
            srv.port,
            name="scan_cam",
            save_nonscalar_data=True,
        )
        det.configure_nonscalar_file_logging(save_path)
        await det.connect()

        desc = await det.describe()
        assert desc["scan_cam-acq_timestamp"]["dtype"] == "number"
        assert desc["scan_cam-nonscalar_save_path"]["dtype"] == "string"

        for _index in range(2):

            async def fire() -> None:
                await asyncio.sleep(0.05)
                combined_device.fire_shot()

            asyncio.create_task(fire())
            await asyncio.wait_for(det.trigger(), timeout=2.0)
            reading = await det.read()

            assert reading["scan_cam-nonscalar_save_path"]["value"] == str(save_path)
            assert "scan_cam-acq_timestamp" in reading
            assert reading["scan_cam-acq_timestamp"]["value"] == pytest.approx(
                combined_device.variables["acq_timestamp"]
            )


async def test_generic_detector_derives_shot_id_from_acq_timestamp(
    combined_device: FakeGeecsDevice,
) -> None:
    """Shot ID follows device acq_timestamp jumps, not event count."""
    combined_device.variables.update({"Signal": 1.0})
    async with FakeGeecsServer(combined_device) as srv:
        det = GeecsGenericDetector(
            "U_Combined",
            ["Signal"],
            srv.host,
            srv.port,
            name="scan_cam",
        )
        await det.connect()
        det.configure_shot_id(rep_rate_hz=1.0)

        desc = await det.describe()
        assert desc["scan_cam-acq_timestamp"]["dtype"] == "number"
        assert desc["scan_cam-t0_acq_timestamp"]["dtype"] == "number"
        assert desc["scan_cam-shot_id"]["dtype"] == "number"
        assert desc["scan_cam-shot_offset"]["dtype"] == "number"
        assert desc["scan_cam-valid"]["dtype"] == "boolean"

        async def fire_one_period() -> None:
            await asyncio.sleep(0.05)
            combined_device.fire_shot()

        asyncio.create_task(fire_one_period())
        await asyncio.wait_for(det.trigger(), timeout=2.0)
        first_reading = await det.read()

        first_acq_timestamp = combined_device.variables["acq_timestamp"]
        assert first_reading["scan_cam-t0_acq_timestamp"]["value"] == pytest.approx(
            first_acq_timestamp
        )
        assert first_reading["scan_cam-shot_id"]["value"] == 1
        assert first_reading["scan_cam-shot_offset"]["value"] == 0
        assert first_reading["scan_cam-valid"]["value"] is True

        async def fire_after_missed_period() -> None:
            await asyncio.sleep(0.05)
            combined_device.variables["acq_timestamp"] = (
                float(combined_device.variables["acq_timestamp"]) + 2.0
            )

        asyncio.create_task(fire_after_missed_period())
        await asyncio.wait_for(det.trigger(), timeout=2.0)
        second_reading = await det.read()

        assert second_reading["scan_cam-t0_acq_timestamp"]["value"] == pytest.approx(
            first_acq_timestamp
        )
        assert second_reading["scan_cam-shot_id"]["value"] == 3
        assert second_reading["scan_cam-valid"]["value"] is True
