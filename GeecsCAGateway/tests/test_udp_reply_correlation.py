"""Pinning tests for UDP exe-reply correlation in ``GeecsUdpClient``.

Regression tests for the stale-reply desync bug: a device taking longer than
``exe_timeout`` on ``set('A', x)`` times out; the caller then issues
``set('B', y)``, arming fresh futures on the same exe socket.  Without
correlation, A's late exe reply resolved B's future, so ``set('B')``
"succeeded" with A's value and B's real reply desynced every later exchange.

These tests never open a socket — they install fake transports on the client
and inject datagrams directly via ``_Oneshot.datagram_received``.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from geecs_ca_gateway.exceptions import GeecsConnectionError
from geecs_ca_gateway.transport.udp_client import GeecsUdpClient, _Oneshot

_UDP_LOGGER = "geecs_ca_gateway.transport.udp_client"
_DEVICE_ADDR = ("127.0.0.1", 50000)


class _FakeTransport:
    """Records datagrams instead of sending them — no network involved."""

    def __init__(self) -> None:
        self.sent: list[tuple[bytes, object]] = []

    def sendto(self, data: bytes, addr: object = None) -> None:
        self.sent.append((data, addr))

    def close(self) -> None:
        pass


@pytest.fixture
def client() -> GeecsUdpClient:
    """A GeecsUdpClient wired to fake transports (never connect()ed)."""
    c = GeecsUdpClient("127.0.0.1", 50000, device_name="U_FakeDev")
    c._cmd_transport = _FakeTransport()  # type: ignore[assignment]
    c._exe_transport = _FakeTransport()  # type: ignore[assignment]
    c._cmd_proto = _Oneshot()
    c._exe_proto = _Oneshot()
    return c


async def _wait_for_send(client: GeecsUdpClient, count: int) -> None:
    """Spin the loop until the client has sent *count* commands."""
    transport = client._cmd_transport
    assert isinstance(transport, _FakeTransport)
    for _ in range(100):
        if len(transport.sent) >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"client never sent command #{count}")


class TestUdpReplyCorrelation:
    async def test_late_reply_for_other_variable_is_discarded(
        self, client: GeecsUdpClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The full desync scenario from the review finding.

        set('A') times out at the exe stage; set('B') is then issued, and A's
        late exe reply arrives mid-exchange.  The stale reply must be dropped
        (with a warning) and B's real reply must resolve B's future with B's
        value.
        """
        caplog.set_level(logging.WARNING, logger=_UDP_LOGGER)
        client.exe_timeout = 0.05  # make the 'slow device' timeout fast

        # --- Exchange 1: set('A') — ACK arrives, exe reply never does.
        task_a = asyncio.create_task(client.set("A", 1.0))
        await _wait_for_send(client, 1)
        client._cmd_proto.datagram_received(b"accepted", _DEVICE_ADDR)
        with pytest.raises(GeecsConnectionError, match="no exe response"):
            await task_a

        # --- Exchange 2: set('B') on the same sockets.
        client.exe_timeout = 5.0
        task_b = asyncio.create_task(client.set("B", 2.0))
        await _wait_for_send(client, 2)
        client._cmd_proto.datagram_received(b"accepted", _DEVICE_ADDR)
        await asyncio.sleep(0)

        # A's late exe reply lands while B's exchange is in flight.
        client._exe_proto.datagram_received(
            b"U_FakeDev>>A>>1.0>>no error,", _DEVICE_ADDR
        )
        await asyncio.sleep(0)
        assert not task_b.done(), "stale reply for A must not resolve B's future"

        # B's genuine reply resolves B's future with B's value.
        client._exe_proto.datagram_received(
            b"U_FakeDev>>B>>2.0>>no error,", _DEVICE_ADDR
        )
        assert await task_b == pytest.approx(2.0)

        discards = [
            r
            for r in caplog.records
            if "does not match in-flight exchange" in r.message
        ]
        assert len(discards) == 1
        assert "U_FakeDev/B" in discards[0].getMessage()

    async def test_matching_reply_resolves_normally(
        self, client: GeecsUdpClient
    ) -> None:
        """No regression: a reply naming the in-flight variable still resolves."""
        task = asyncio.create_task(client.get("Position (mm)"))
        await _wait_for_send(client, 1)
        client._cmd_proto.datagram_received(b"accepted", _DEVICE_ADDR)
        client._exe_proto.datagram_received(
            b"U_FakeDev>>Position (mm)>>5.0>>no error,", _DEVICE_ADDR
        )
        assert await task == pytest.approx(5.0)

    async def test_command_echo_form_also_matches(self, client: GeecsUdpClient) -> None:
        """Real hardware echoes the command ('getVar'/'setVar') in field 2.

        The legacy GEECS-PythonAPI parser (``geecs_device.handle_response``)
        shows the second field carries the op prefix on real devices; the
        matcher must accept that form too, not just the bare variable name.
        """
        task = asyncio.create_task(client.set("Position (mm)", 7.5))
        await _wait_for_send(client, 1)
        client._cmd_proto.datagram_received(b"accepted", _DEVICE_ADDR)
        client._exe_proto.datagram_received(
            b"U_FakeDev>>setPosition (mm)>>7.5>>no error,", _DEVICE_ADDR
        )
        assert await task == pytest.approx(7.5)

    async def test_reply_with_no_exchange_in_flight_is_dropped(
        self, client: GeecsUdpClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Datagrams arriving between exchanges are logged and discarded."""
        caplog.set_level(logging.WARNING, logger=_UDP_LOGGER)
        client._exe_proto.datagram_received(
            b"U_FakeDev>>A>>1.0>>no error,", _DEVICE_ADDR
        )
        client._cmd_proto.datagram_received(b"accepted", _DEVICE_ADDR)
        stale = [r for r in caplog.records if "no exchange in flight" in r.message]
        assert len(stale) == 2

    async def test_malformed_reply_is_discarded_not_misattributed(
        self, client: GeecsUdpClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A reply with no attributable variable field must not resolve the future."""
        caplog.set_level(logging.WARNING, logger=_UDP_LOGGER)
        task = asyncio.create_task(client.get("A"))
        await _wait_for_send(client, 1)
        client._cmd_proto.datagram_received(b"accepted", _DEVICE_ADDR)
        # Short error form (">>error,msg") carries no variable name.
        client._exe_proto.datagram_received(b">>error,something broke", _DEVICE_ADDR)
        await asyncio.sleep(0)
        assert not task.done()
        # The genuine reply still gets through afterwards.
        client._exe_proto.datagram_received(
            b"U_FakeDev>>A>>3.0>>no error,", _DEVICE_ADDR
        )
        assert await task == pytest.approx(3.0)
        assert any(
            "does not match in-flight exchange" in r.message for r in caplog.records
        )
