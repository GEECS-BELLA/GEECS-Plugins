"""A ``GeecsUdpClient`` whose device replies the test delivers by hand.

Test code only. No socket is opened: the client's two transports record
what it sends, and the test plays the device by feeding datagrams to
``client._cmd_proto`` (the ACK) and ``client._exe_proto`` (the exe reply)
through ``datagram_received``.
"""

from __future__ import annotations

import asyncio

from geecs_core.transport.udp_client import GeecsUdpClient, _Oneshot


class RecordingTransport:
    """Records ``sendto`` instead of sending: no network involved."""

    def __init__(self) -> None:
        self.sent: list[tuple[bytes, object]] = []

    def sendto(self, data: bytes, addr: object = None) -> None:
        """Record one datagram."""
        self.sent.append((data, addr))

    def close(self) -> None:
        """Nothing to close."""


def hand_fed_udp_client(
    device_name: str, *, host: str = "127.0.0.1", port: int = 50000
) -> GeecsUdpClient:
    """Return a ``GeecsUdpClient`` wired to recording transports, never connected."""
    client = GeecsUdpClient(host, port, device_name=device_name)
    client._cmd_transport = RecordingTransport()  # type: ignore[assignment]
    client._exe_transport = RecordingTransport()  # type: ignore[assignment]
    client._cmd_proto = _Oneshot()
    client._exe_proto = _Oneshot()
    return client


async def wait_for_command(client: GeecsUdpClient, count: int) -> None:
    """Spin the loop until the client has sent *count* commands."""
    transport = client._cmd_transport
    assert isinstance(transport, RecordingTransport)
    for _ in range(100):
        if len(transport.sent) >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"client never sent command #{count}")
