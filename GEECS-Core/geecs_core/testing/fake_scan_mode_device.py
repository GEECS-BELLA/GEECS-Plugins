"""Fake GEECS devices on the scan-mode side of the protocol, for offline tests.

A :class:`FakeScanModeGroup` is one listening UDP socket standing in for the
experiment's multicast group: every datagram it receives reaches every
:class:`FakeScanModeDevice` registered with it, as a multicast reaches every
device.  By default it listens on a unicast loopback address — multicast on
a loopback interface is not routable on every CI host — and joins a real
multicast group when one is given, for the one test that exercises
multicast sending.

Each fake device does what the real device framework does, as captured
(``geecs_core.transport.scan_mode`` has the protocol):

- on ``ScanStart`` it records the lock IP (the command's last argument),
  acknowledges by unicast to that IP at its own command-port number —
  twice, like the real device — and connects to ``{lock IP}:{tcp_port}``;
- on the path channel it answers every keepalive with six zero bytes and
  records each ``Path>>`` it receives, bumping its stamp on every **new**
  shot number (``checkIfToAcqu``: a repeated number is not re-acquired);
- on the ``ScanEnd`` **frame** on the path channel it acknowledges (by UDP,
  like ``ScanStart``), leaves scan mode and closes the connection, as the
  real device does; the multicast ``ScanEnd`` it ignores, as the real
  device's listener does (empty case, read 2026-10-13).

``acknowledge=False`` plays a device that missed the command (the live
scan-5 failure mode: silent, never in scan mode); ``ack_status`` plays an
error acknowledgement.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import socket
from collections.abc import Iterable

from ..transport._framing import read_frame
from ..transport.scan_mode import (
    ACK_OK_STATUS,
    _wait_event,
    KEEPALIVE_PAYLOAD,
    PATH_KEY,
    SCAN_END,
    SCAN_MODE_TCP_PORT,
    SCAN_START,
    SEPARATOR,
)

logger = logging.getLogger(__name__)

#: What the real device writes on connect and in answer to every keepalive.
ZERO_REPLY = b"\x00" * 6
_PATH_PREFIX = f"{PATH_KEY}{SEPARATOR}".encode("latin-1")
_SHOT_NUMBER = re.compile(r"_(\d+)\.fileextension$")


class FakeScanModeDevice:
    """One simulated device's scan-mode behaviour (see the module docstring).

    Parameters
    ----------
    name :
        The device name, the first field of its acknowledgements.
    command_port :
        The device's command port number — where its acknowledgements go on
        the client host.  Nothing is bound on it here.
    tcp_port :
        The path-channel port to connect to; tests set it to the port their
        :class:`~geecs_core.transport.scan_mode.ScanModeServer` chose.
    acknowledge :
        ``False`` plays a device that never saw the command.
    ack_status :
        The acknowledgement's status field.
    """

    def __init__(
        self,
        name: str,
        command_port: int,
        *,
        tcp_port: int = SCAN_MODE_TCP_PORT,
        acknowledge: bool = True,
        ack_status: str = ACK_OK_STATUS,
        ack_repeats: int = 2,
        ack_gap_s: float = 0.02,
        disconnect_delay_s: float = 0.04,
    ) -> None:
        self.name = name
        self.command_port = int(command_port)
        self.tcp_port = int(tcp_port)
        self.acknowledge = acknowledge
        self.ack_status = ack_status
        self._ack_repeats = ack_repeats
        self._ack_gap_s = ack_gap_s
        self._disconnect_delay_s = disconnect_delay_s

        self.lock_ip: str | None = None
        self.received: list[bytes] = []
        self.acks_sent: list[bytes] = []
        self.paths: list[str] = []
        self.scan_ends: list[str] = []
        self.shot_numbers: list[int] = []
        self.stamp = 0
        self.keepalives = 0

        self._last_shot: int | None = None
        self._udp: asyncio.DatagramTransport | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._path_event = asyncio.Event()
        self._connected = asyncio.Event()

    @property
    def in_scan_mode(self) -> bool:
        """Whether a ``ScanStart`` was processed and no ``ScanEnd`` since."""
        return self.lock_ip is not None

    @property
    def connected(self) -> bool:
        """Whether the path-channel connection is open."""
        return self._connected.is_set()

    async def start(self, bind_host: str) -> None:
        """Open the socket the acknowledgements are sent from."""
        loop = asyncio.get_running_loop()
        self._udp, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol, local_addr=(bind_host, 0), family=socket.AF_INET
        )

    async def stop(self) -> None:
        """Drop the path-channel connection and release the socket."""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._disconnect()
        if self._udp is not None:
            self._udp.close()
            self._udp = None

    def handle(self, data: bytes) -> None:
        """React to one group datagram (scheduled; the group calls this from its protocol)."""
        self.received.append(data)
        task = asyncio.create_task(
            self._react(data), name=f"fake-scan-mode[{self.name}]"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_for_paths(self, count: int, timeout: float) -> list[str]:
        """Wait until *count* paths have arrived (or *timeout*); returns the paths so far."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            self._path_event.clear()
            remaining = deadline - loop.time()
            if len(self.paths) >= count or remaining <= 0:
                return list(self.paths)
            await _wait_event(self._path_event, remaining)

    async def wait_connected(self, timeout: float) -> bool:
        """Wait until the path-channel connection is open; ``False`` on timeout."""
        await _wait_event(self._connected, timeout)
        return self.connected

    # ------------------------------------------------------------------

    async def _react(self, data: bytes) -> None:
        parts = data.decode("latin-1").split(SEPARATOR)
        if len(parts) < 4 or not self.acknowledge:
            return  # ``Empty`` (the real device clears its path), or a missed command
        if parts[0] == SCAN_START:
            self.lock_ip = parts[3]
            await self._ack(parts, self.lock_ip)
            await self._connect(self.lock_ip)
        # A multicast ``ScanEnd`` is a dead letter on the real device: ignored.

    async def _ack(self, parts: list[str], ip: str) -> None:
        assert self._udp is not None, "start() first"
        payload = SEPARATOR.join([self.name, *parts, self.ack_status]).encode("latin-1")
        for repeat in range(self._ack_repeats):
            if repeat:
                await asyncio.sleep(self._ack_gap_s)
            self._udp.sendto(payload, (ip, self.command_port))
            self.acks_sent.append(payload)

    async def _connect(self, ip: str) -> None:
        await self._disconnect()
        try:
            reader, writer = await asyncio.open_connection(ip, self.tcp_port)
        except OSError as exc:
            logger.warning(
                "%s: path channel %s:%s refused: %s", self.name, ip, self.tcp_port, exc
            )
            return
        self._writer = writer
        writer.write(ZERO_REPLY)  # the real device's first bytes on connect
        self._connected.set()
        self._reader_task = asyncio.create_task(
            self._read_frames(reader, writer), name=f"fake-scan-mode-tcp[{self.name}]"
        )

    async def _read_frames(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                payload = await read_frame(reader)
                if payload == KEEPALIVE_PAYLOAD:
                    self.keepalives += 1
                    writer.write(ZERO_REPLY)
                elif payload.startswith(_PATH_PREFIX):
                    self._on_path(payload[len(_PATH_PREFIX) :].decode("latin-1"))
                elif payload.startswith(SCAN_END.encode("latin-1")):
                    # The real device's TCP loop: scan end -> end commands, ack, close.
                    text = payload.decode("latin-1")
                    parts = text.split(SEPARATOR)
                    self.scan_ends.append(text)
                    if self.acknowledge and len(parts) >= 4:
                        await self._ack(parts, parts[3])
                    self.lock_ip = None
                    await asyncio.sleep(self._disconnect_delay_s)
                    break  # the loop ends; the finally closes the connection
                else:
                    logger.debug(
                        "%s: ignoring %r on the path channel", self.name, payload
                    )
        except (asyncio.IncompleteReadError, OSError):
            pass
        finally:
            self._connected.clear()
            if self._writer is writer:
                self._writer = None
                writer.close()

    def _on_path(self, path: str) -> None:
        self.paths.append(path)
        match = _SHOT_NUMBER.search(path)
        shot = int(match.group(1)) if match else None
        if shot is not None:
            self.shot_numbers.append(shot)
        if shot is None or shot != self._last_shot:
            self.stamp += 1  # a new shot number: the real device acquires
        self._last_shot = shot
        self._path_event.set()

    async def _disconnect(self) -> None:
        # Take the writer first: cancelling the reader runs its ``finally``,
        # which clears ``self._writer`` — taken afterwards it would be None and
        # the connection would only close when garbage-collected.
        writer, self._writer = self._writer, None
        task, self._reader_task = self._reader_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if writer is not None:
            writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
        self._connected.clear()


class _GroupProtocol(asyncio.DatagramProtocol):
    def __init__(self, group: FakeScanModeGroup) -> None:
        self._group = group

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._group._on_datagram(data, addr)


class FakeScanModeGroup:
    """One listening socket standing in for the experiment's multicast group.

    Parameters
    ----------
    devices :
        The fake devices every datagram is delivered to.
    host :
        The address to listen on (and the interface a multicast group is
        joined on); the devices send their acknowledgements from it too.
    port :
        ``0`` lets the OS choose; the bound port is then :attr:`port`.
    group :
        A multicast group to join instead of listening on *host* alone.
    """

    def __init__(
        self,
        devices: Iterable[FakeScanModeDevice],
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        group: str | None = None,
    ) -> None:
        self.devices = tuple(devices)
        self.host = host
        self.port = int(port)
        self.group = group
        self.received: list[bytes] = []
        self._transport: asyncio.DatagramTransport | None = None

    async def start(self) -> None:
        """Bind (joining the multicast group when one was given) and start the devices."""
        loop = asyncio.get_running_loop()
        if self.group is None:
            self._transport, _ = await loop.create_datagram_endpoint(
                lambda: _GroupProtocol(self),
                local_addr=(self.host, self.port),
                family=socket.AF_INET,
            )
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", self.port))
            membership = socket.inet_aton(self.group) + socket.inet_aton(self.host)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
            self._transport, _ = await loop.create_datagram_endpoint(
                lambda: _GroupProtocol(self), sock=sock
            )
        self.port = self._transport.get_extra_info("sockname")[1]
        for device in self.devices:
            await device.start(self.host)

    async def stop(self) -> None:
        """Stop every device and release the socket."""
        for device in self.devices:
            await device.stop()
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    async def __aenter__(self) -> FakeScanModeGroup:
        """Start and return ``self``."""
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Stop the group."""
        await self.stop()

    def _on_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        logger.debug("fake group: %r from %s", data, addr)
        self.received.append(data)
        for device in self.devices:
            device.handle(data)
