"""Unit tests for the asyncio UDP and TCP transport layers.

All tests run against ``FakeGeecsServer`` on localhost — no real hardware required.
"""

import asyncio
import logging

import pytest

from geecs_ca_gateway.exceptions import GeecsCommandFailedError
from geecs_ca_gateway.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer
from geecs_ca_gateway.transport.udp_client import GeecsUdpClient
from geecs_ca_gateway.transport.tcp_subscriber import (
    GeecsTcpSubscriber,
    _compile_frame_pattern,
    _parse_subscription,
)

pytestmark = pytest.mark.fake_server


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
                with pytest.raises(GeecsCommandFailedError):
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

    async def test_missing_variable_warns_and_continues(
        self, fake_device: FakeGeecsDevice, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing subscribed variables should warn without stopping updates."""
        received: list[dict] = []

        def on_update(update: dict) -> None:
            received.append(update)

        caplog.set_level(logging.WARNING, logger="geecs_ca_gateway.transport")
        async with FakeGeecsServer(fake_device) as srv:
            async with GeecsTcpSubscriber(srv.host, srv.port) as sub:
                await sub.subscribe(["Position (mm)", "Not In Frame"], on_update)
                await asyncio.sleep(0.5)

        assert len(received) >= 2
        assert all("Position (mm)" in update for update in received)
        assert all("Not In Frame" not in update for update in received)
        missing_warnings = [
            record
            for record in caplog.records
            if "missing variable(s) in push frame" in record.message
        ]
        assert len(missing_warnings) == 1
        assert "Not In Frame" in missing_warnings[0].message

    async def test_callback_keyerror_does_not_stop_listener(
        self, fake_device: FakeGeecsDevice, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A callback assuming a missing key should not kill the TCP listener."""
        calls = 0

        def on_update(update: dict) -> None:
            nonlocal calls
            calls += 1
            update["Not In Frame"]

        caplog.set_level(logging.WARNING, logger="geecs_ca_gateway.transport")
        async with FakeGeecsServer(fake_device) as srv:
            async with GeecsTcpSubscriber(srv.host, srv.port) as sub:
                await sub.subscribe(["Position (mm)", "Not In Frame"], on_update)
                await asyncio.sleep(0.5)

        assert calls >= 2
        assert any(
            "TCP subscription callback failed" in record.message
            for record in caplog.records
        )

    async def test_string_value_reaches_callback_verbatim(self) -> None:
        """A string-typed value like '007' must survive the wire untouched.

        Numeric coercion would round-trip it through float ('007' -> 7), which
        corrupts zero-padded IDs and version strings on the readback PV.
        """
        device = FakeGeecsDevice(
            name="U_TestDevice",
            variables={"SerialNum": "007", "Position (mm)": 5.0},
        )
        received: list[dict] = []

        def on_update(update: dict) -> None:
            received.append(update)

        async with FakeGeecsServer(device) as srv:
            async with GeecsTcpSubscriber(srv.host, srv.port) as sub:
                await sub.subscribe(
                    ["SerialNum", "Position (mm)"],
                    on_update,
                    text_variables={"SerialNum"},
                )
                await asyncio.sleep(0.5)

        assert len(received) >= 1
        assert received[0]["SerialNum"] == "007"
        # Numeric variables are still coerced as before.
        assert received[0]["Position (mm)"] == pytest.approx(5.0)
        assert isinstance(received[0]["Position (mm)"], float)


# ---------------------------------------------------------------------------
# Push-frame parser tests (no sockets — frames fed directly to the parser)
# ---------------------------------------------------------------------------


def parse(
    msg: str, variables: list[str], text_variables: set[str] | None = None
) -> dict:
    """Compile the frame pattern for ``variables`` and parse ``msg``."""
    pattern = _compile_frame_pattern(variables)
    return _parse_subscription(msg, pattern, frozenset(text_variables or ()))


class TestSubscriptionFrameParsing:
    """Pin the comma-in-value and text-preservation parsing behaviour."""

    def test_value_containing_comma_parses_all_variables(self) -> None:
        """A comma inside a value must not drop it or corrupt its neighbours."""
        variables = ["Position (mm)", "localsavingpath", "Status"]
        msg = (
            "U_TestDevice>>42>>"
            "Position (mm) nval,5.0 nvar,"
            "localsavingpath nval,Z:/data/run1,repeat nvar,"
            "Status nval,0 nvar"
        )
        parsed = parse(msg, variables, text_variables={"localsavingpath"})
        assert parsed == {
            "Position (mm)": 5.0,
            "localsavingpath": "Z:/data/run1,repeat",
            "Status": 0,
        }

    def test_comma_value_as_last_variable_in_frame(self) -> None:
        variables = ["Status", "localsavingpath"]
        msg = (
            "U_TestDevice>>1>>"
            "Status nval,1 nvar,"
            "localsavingpath nval,Z:/data/run1,repeat nvar"
        )
        parsed = parse(msg, variables, text_variables={"localsavingpath"})
        assert parsed == {"Status": 1, "localsavingpath": "Z:/data/run1,repeat"}

    def test_value_with_comma_name_lookalike_text(self) -> None:
        """',<name>' text inside a value is not mistaken for a pair boundary."""
        variables = ["localsavingpath", "Status"]
        msg = (
            "U_TestDevice>>1>>"
            "localsavingpath nval,Z:/a,Status,b nvar,"
            "Status nval,3 nvar"
        )
        parsed = parse(msg, variables, text_variables={"localsavingpath"})
        assert parsed == {"localsavingpath": "Z:/a,Status,b", "Status": 3}

    def test_crlf_separated_pairs(self) -> None:
        """Real LabVIEW frames may join pairs with ',\\r\\n' (legacy format)."""
        variables = ["Position (mm)", "Velocity (mm/s)"]
        msg = (
            "U_TestDevice>>7>>"
            "Position (mm) nval,5.0 nvar,\r\n"
            "Velocity (mm/s) nval,1.5 nvar"
        )
        parsed = parse(msg, variables)
        assert parsed == {"Position (mm)": 5.0, "Velocity (mm/s)": 1.5}

    def test_unsubscribed_variable_does_not_corrupt_neighbours(self) -> None:
        """An extra pair the client never asked for is skipped cleanly."""
        variables = ["Position (mm)", "Status"]
        msg = (
            "U_TestDevice>>2>>"
            "Position (mm) nval,5.0 nvar,"
            "Extra nval,junk nvar,"
            "Status nval,0 nvar"
        )
        parsed = parse(msg, variables)
        assert parsed == {"Position (mm)": 5.0, "Status": 0}

    def test_value_containing_angle_delimiters(self) -> None:
        """'>>' inside a value must not truncate the payload."""
        variables = ["Note"]
        msg = "U_TestDevice>>3>>Note nval,a>>b nvar"
        parsed = parse(msg, variables, text_variables={"Note"})
        assert parsed == {"Note": "a>>b"}

    def test_name_prefix_of_another_name(self) -> None:
        """A name that is a prefix of another must not shadow the longer one."""
        variables = ["Position", "Position (mm)"]
        msg = "U_TestDevice>>4>>Position (mm) nval,5.0 nvar,Position nval,9.0 nvar"
        parsed = parse(msg, variables)
        assert parsed == {"Position (mm)": 5.0, "Position": 9.0}

    def test_string_dtype_values_pass_through_verbatim(self) -> None:
        """'007', '1.10', '1e5' keep their exact text for text variables."""
        variables = ["SerialNum", "Version", "Tag"]
        msg = (
            "U_TestDevice>>5>>"
            "SerialNum nval,007 nvar,"
            "Version nval,1.10 nvar,"
            "Tag nval,1e5 nvar"
        )
        parsed = parse(msg, variables, text_variables=set(variables))
        assert parsed == {"SerialNum": "007", "Version": "1.10", "Tag": "1e5"}

    def test_numeric_variables_still_coerced(self) -> None:
        """Variables not marked as text keep the historical numeric coercion."""
        variables = ["Position (mm)", "Shots"]
        msg = "U_TestDevice>>6>>Position (mm) nval,5.5 nvar,Shots nval,12 nvar"
        parsed = parse(msg, variables)
        assert parsed["Position (mm)"] == pytest.approx(5.5)
        assert isinstance(parsed["Position (mm)"], float)
        assert parsed["Shots"] == 12
        assert isinstance(parsed["Shots"], int)

    def test_malformed_or_empty_frames(self) -> None:
        assert parse("garbage", ["A"]) == {}
        assert parse("Dev>>1>>", ["A"]) == {}
        assert _parse_subscription("Dev>>1>>A nval,1 nvar", None) == {}
