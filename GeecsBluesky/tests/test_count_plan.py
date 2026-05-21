"""End-to-end tests: GEECS device → ophyd-async → Bluesky plan.

Two levels of testing:
- Direct device API (``read()``, ``describe()``, ``set()``) — no RunEngine needed
- Bluesky ``count`` plan via RunEngine — full integration smoke test
"""

from __future__ import annotations

import asyncio
import threading

import pytest
import bluesky.plans as bp
from bluesky import RunEngine

from geecs_bluesky.devices.geecs_device import GeecsDevice
from geecs_bluesky.signals import geecs_signal_r, geecs_signal_rw
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer
from geecs_bluesky.transport.udp_client import GeecsUdpClient

_DEV = "U_TestDevice"


# ---------------------------------------------------------------------------
# Concrete test device
# ---------------------------------------------------------------------------


class FakeMotor(GeecsDevice):
    """Minimal GEECS motor device for testing."""

    def __init__(self, host: str, port: int, name: str = "fake_motor") -> None:
        udp = GeecsUdpClient(host, port)
        with self.add_children_as_readables():
            self.position = geecs_signal_rw(
                float, _DEV, "Position (mm)", host, port, units="mm", shared_udp=udp
            )
            self.velocity = geecs_signal_r(
                float, _DEV, "Velocity (mm/s)", host, port, units="mm/s", shared_udp=udp
            )
        super().__init__(name=name, shared_udp=udp)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_device() -> FakeGeecsDevice:
    return FakeGeecsDevice(
        name="U_TestDevice",
        variables={
            "Position (mm)": 0.0,
            "Velocity (mm/s)": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# Direct device API tests (async)
# ---------------------------------------------------------------------------


class TestDeviceAPI:
    async def test_connect_and_describe(self, sim_device: FakeGeecsDevice) -> None:
        async with FakeGeecsServer(sim_device) as srv:
            motor = FakeMotor(srv.host, srv.port)
            await motor.connect()
            description = await motor.describe()
            assert "fake_motor-position" in description
            assert "fake_motor-velocity" in description
            assert description["fake_motor-position"]["dtype"] == "number"

    async def test_read_returns_values(self, sim_device: FakeGeecsDevice) -> None:
        async with FakeGeecsServer(sim_device) as srv:
            motor = FakeMotor(srv.host, srv.port)
            await motor.connect()
            reading = await motor.read()
            assert "fake_motor-position" in reading
            assert reading["fake_motor-position"]["value"] == pytest.approx(0.0)

    async def test_set_position(self, sim_device: FakeGeecsDevice) -> None:
        async with FakeGeecsServer(sim_device) as srv:
            motor = FakeMotor(srv.host, srv.port)
            await motor.connect()
            await motor.position.set(5.0)
            reading = await motor.read()
            assert reading["fake_motor-position"]["value"] == pytest.approx(5.0)

    async def test_read_multiple_times(self, sim_device: FakeGeecsDevice) -> None:
        async with FakeGeecsServer(sim_device) as srv:
            motor = FakeMotor(srv.host, srv.port)
            await motor.connect()
            readings = [await motor.read() for _ in range(5)]
            assert all("fake_motor-position" in r for r in readings)

    async def test_position_reflects_external_change(
        self, sim_device: FakeGeecsDevice
    ) -> None:
        """External state change is reflected after the next 5-Hz TCP push."""
        async with FakeGeecsServer(sim_device) as srv:
            motor = FakeMotor(srv.host, srv.port)
            await motor.connect()
            sim_device.variables["Position (mm)"] = 42.0
            # TCP push rate is 5 Hz (200 ms); wait for the next push to update cache
            await asyncio.sleep(0.3)
            reading = await motor.read()
            assert reading["fake_motor-position"]["value"] == pytest.approx(42.0)

    async def test_disconnect_closes_shared_udp(
        self, sim_device: FakeGeecsDevice
    ) -> None:
        """disconnect() must close the shared UDP client sockets."""
        async with FakeGeecsServer(sim_device) as srv:
            motor = FakeMotor(srv.host, srv.port)
            await motor.connect()
            assert motor._shared_udp is not None
            assert motor._shared_udp._cmd_transport is not None  # connected

            await motor.disconnect()

            assert motor._shared_udp._cmd_transport is None  # sockets released

    async def test_reconnect_after_disconnect(
        self, sim_device: FakeGeecsDevice
    ) -> None:
        """connect(force_reconnect=True) after disconnect() should restore operation."""
        async with FakeGeecsServer(sim_device) as srv:
            motor = FakeMotor(srv.host, srv.port)
            await motor.connect()
            await motor.disconnect()

            await motor.connect(force_reconnect=True)
            reading = await motor.read()
            assert "fake_motor-position" in reading


# ---------------------------------------------------------------------------
# RunEngine integration test (synchronous, server in background thread)
# ---------------------------------------------------------------------------


def _run_server(
    device: FakeGeecsDevice, ready: threading.Event, host_port: list
) -> None:
    """Target for the background thread: run the fake server until stop is requested."""

    async def _main() -> None:
        async with FakeGeecsServer(device) as srv:
            host_port.extend([srv.host, srv.port])
            ready.set()
            # Run until cancelled from outside
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

    loop = asyncio.new_event_loop()
    task_holder: list[asyncio.Task] = []

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


def test_bluesky_count_plan(sim_device: FakeGeecsDevice) -> None:
    """Smoke test: ``count([motor], num=3)`` collects 3 events."""
    ready = threading.Event()
    host_port: list = []

    srv_thread = threading.Thread(
        target=_run_server,
        args=(sim_device, ready, host_port),
        daemon=True,
    )
    srv_thread.start()
    ready.wait(timeout=5.0)
    assert host_port, "Server failed to start"

    host, port = host_port[0], host_port[1]

    # Connect the device in its own event loop (ophyd-async requires this)
    motor = FakeMotor(host, port)
    asyncio.run(motor.connect())

    # Collect data
    events: list[dict] = []
    RE = RunEngine()
    RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)

    RE(bp.count([motor], num=3))

    assert len(events) == 3, f"Expected 3 events, got {len(events)}"
    for ev in events:
        assert "fake_motor-position" in ev["data"]
        assert "fake_motor-velocity" in ev["data"]
