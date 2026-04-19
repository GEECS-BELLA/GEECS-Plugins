"""Unit tests for the asyncio UDP and TCP transport layers.

All tests run against ``FakeGeecsServer`` on localhost — no real hardware required.
"""

import asyncio
import pytest

from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer
from geecs_bluesky.transport.udp_client import GeecsUdpClient
from geecs_bluesky.transport.tcp_subscriber import GeecsTcpSubscriber


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_device() -> FakeGeecsDevice:
    return FakeGeecsDevice(
        name="U_TestDevice",
        variables={
            "Position (mm)": 5.0,
            "Velocity (mm/s)": 1.5,
            "Status": 0,
        },
    )


# ---------------------------------------------------------------------------
# UDP tests
# ---------------------------------------------------------------------------


class TestUdpClient:
    async def test_get_float(self, fake_device: FakeGeecsDevice) -> None:
        async with FakeGeecsServer(fake_device) as srv:
            async with GeecsUdpClient(srv.host, srv.port) as client:
                value = await client.get("Position (mm)")
                assert value == pytest.approx(5.0)

    async def test_set_and_get(self, fake_device: FakeGeecsDevice) -> None:
        async with FakeGeecsServer(fake_device) as srv:
            async with GeecsUdpClient(srv.host, srv.port) as client:
                await client.set("Position (mm)", 7.5)
                value = await client.get("Position (mm)")
                assert value == pytest.approx(7.5)

    async def test_set_integer(self, fake_device: FakeGeecsDevice) -> None:
        async with FakeGeecsServer(fake_device) as srv:
            async with GeecsUdpClient(srv.host, srv.port) as client:
                await client.set("Status", 1)
                value = await client.get("Status")
                assert value == 1

    async def test_get_unknown_variable_raises(
        self, fake_device: FakeGeecsDevice
    ) -> None:
        async with FakeGeecsServer(fake_device) as srv:
            async with GeecsUdpClient(srv.host, srv.port) as client:
                with pytest.raises(RuntimeError, match="failed"):
                    await client.get("NonExistent")

    async def test_multiple_sequential_commands(
        self, fake_device: FakeGeecsDevice
    ) -> None:
        async with FakeGeecsServer(fake_device) as srv:
            async with GeecsUdpClient(srv.host, srv.port) as client:
                for v in (1.0, 2.0, 3.0):
                    await client.set("Position (mm)", v)
                final = await client.get("Position (mm)")
                assert final == pytest.approx(3.0)

    async def test_concurrent_gets_serialized(
        self, fake_device: FakeGeecsDevice
    ) -> None:
        """asyncio.gather of multiple gets on a shared client must all succeed.

        Without the asyncio.Lock in _exchange, concurrent calls to arm() would
        overwrite each other's futures — only the last armed future would ever
        resolve, causing the others to time out.  This test catches that
        regression: if the lock is removed all three gathers would hang.
        """
        async with FakeGeecsServer(fake_device) as srv:
            async with GeecsUdpClient(srv.host, srv.port) as client:
                pos, vel, status = await asyncio.gather(
                    client.get("Position (mm)"),
                    client.get("Velocity (mm/s)"),
                    client.get("Status"),
                )
        assert pos == pytest.approx(5.0)
        assert vel == pytest.approx(1.5)
        assert status == 0


# ---------------------------------------------------------------------------
# TCP subscription tests
# ---------------------------------------------------------------------------


class TestTcpSubscriber:
    async def test_receives_updates(self, fake_device: FakeGeecsDevice) -> None:
        received: list[dict] = []

        def on_update(update: dict) -> None:
            received.append(update)

        async with FakeGeecsServer(fake_device) as srv:
            async with GeecsTcpSubscriber(srv.host, srv.port) as sub:
                await sub.subscribe(["Position (mm)"], on_update)
                # Wait for a few pushes (server runs at 5 Hz)
                await asyncio.sleep(0.5)

        assert len(received) >= 2, f"Expected >=2 updates, got {len(received)}"
        assert "Position (mm)" in received[0]
        assert received[0]["Position (mm)"] == pytest.approx(5.0)

    async def test_reflects_set_value(self, fake_device: FakeGeecsDevice) -> None:
        """TCP pushes should reflect a value changed via UDP."""
        received: list[dict] = []

        def on_update(update: dict) -> None:
            received.append(update)

        async with FakeGeecsServer(fake_device) as srv:
            # UDP set
            async with GeecsUdpClient(srv.host, srv.port) as udp:
                await udp.set("Position (mm)", 9.9)

            async with GeecsTcpSubscriber(srv.host, srv.port) as sub:
                await sub.subscribe(["Position (mm)"], on_update)
                await asyncio.sleep(0.5)

        values = [r["Position (mm)"] for r in received if "Position (mm)" in r]
        assert any(v == pytest.approx(9.9) for v in values)

    async def test_multi_variable_subscription(
        self, fake_device: FakeGeecsDevice
    ) -> None:
        received: list[dict] = []

        def on_update(update: dict) -> None:
            received.append(update)

        async with FakeGeecsServer(fake_device) as srv:
            async with GeecsTcpSubscriber(srv.host, srv.port) as sub:
                await sub.subscribe(["Position (mm)", "Velocity (mm/s)"], on_update)
                await asyncio.sleep(0.5)

        assert len(received) >= 1
        assert "Position (mm)" in received[0]
        assert "Velocity (mm/s)" in received[0]
