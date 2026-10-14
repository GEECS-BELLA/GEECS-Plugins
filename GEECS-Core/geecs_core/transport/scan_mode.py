r"""Master Control's scan-mode device protocol — the client side, as captured.

Under Master Control a GEECS device in **scan mode** acquires once per
announced shot and saves to the announced, shot-numbered path.  The Bluesky
worker reproduces Master Control's side of that protocol so devices behave
exactly as they do under it (issue #1084).  Everything here was captured
live from Master Control scans (Undulator, 2026-10-13); nothing is
inferred.

The three channels
------------------
**Multicast commands** (UDP, to the experiment's ``expt.MultiCastIP`` group
at ``MultiCastRemotePort``; Master Control sends them from its
``MultiCastLocalPort``, which no device depends on)::

    Empty
    Empty
    ScanStart>>{experiment}>>{scan parameter}>>{client address}
    ...
    ScanEnd>>{experiment}>>{argument}>>{client address}

Every enabled device of the experiment enters scan mode on ``ScanStart``
(applying its DB start values and storing the named address as its
"Master Control IP").  :class:`ScanModeMulticaster` sends these.  The
multicast ``ScanEnd`` is a **dead letter**: the device framework's listener
has an empty case for it.  The scan end the device acts on is the same
``ScanEnd>>…`` text pushed as a frame on its path-channel connection (below,
:meth:`ScanModeServer.push_scan_end`); Master Control sends both, the frame
40 ms after the multicast, and the device then runs its end commands,
acknowledges by UDP and closes the connection.  Verified live 2026-10-13.  The address is normally the
sender's IP; it may have to be the name the devices resolve that IP to
(``client_ip``), a facility fact the consumer supplies.

**Acknowledgements** (UDP unicast): a device answers both commands with
``{device}>>ScanStart>>{experiment}>>{scan parameter}>>{client IP}>>no error,``
sent to the client IP **at the device's own command-port number**, twice
(identical datagrams a few hundred milliseconds apart).
:class:`ScanModeAckListener` binds those port numbers and collects the
replies, deduplicated.  A device that never acknowledges never entered scan
mode — observed live: its start values stay unapplied and every fire fails
— so a caller waits for every device it needs before firing.

**The path channel** (TCP): after ``ScanStart`` every device connects to
the client IP on port :data:`SCAN_MODE_TCP_PORT` (a framework constant) and
keeps the connection for the scan.  The client sends a length-prefixed
``dummy`` keepalive about every 40 ms on every connection (the device
answers each with six zero bytes), per shot pushes the length-prefixed
``Path>>{device-side shot path}`` to every connection, and at the end pushes
the ``ScanEnd>>…`` frame, on which the device leaves scan mode and closes.
The path reads
``{root}\...\Scan006\xxx\Scan006_xxx_001.fileextension`` and the device
substitutes ``xxx`` and the extension (:func:`device_shot_path`).
:class:`ScanModeServer` is that server.  A connection carries no device
name — only the acknowledgement does — and a device may open two and let
one be reset, so the server accepts every connection and judges presence
by peer host.

Framing is the GEECS TCP framing throughout: a 4-byte big-endian length
then the payload (:func:`frame`).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import socket
from collections.abc import Collection, Iterable
from dataclasses import dataclass, replace

from geecs_core.exceptions import GeecsConnectionError

from ._framing import frame

logger = logging.getLogger(__name__)

#: The port every device connects to after ``ScanStart`` — a device-framework
#: constant, not a DB value.
SCAN_MODE_TCP_PORT = 6342
#: Master Control's keepalive period per connection (captured: ~40 ms).
KEEPALIVE_INTERVAL_S = 0.04
#: Master Control's multicast TTL (captured); the groups span routed subnets.
MULTICAST_TTL = 5
SEPARATOR = ">>"
SCAN_START = "ScanStart"
SCAN_END = "ScanEnd"
EMPTY_MESSAGE = b"Empty"
KEEPALIVE_PAYLOAD = b"dummy"
PATH_KEY = "Path"
#: The status field of a successful acknowledgement (trailing comma included).
ACK_OK_STATUS = "no error,"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


__all__ = [
    "ScanModeAck",
    "ScanModeAckListener",
    "ScanModeMulticaster",
    "ScanModeServer",
    "device_shot_path",
    "frame",
    "parse_acknowledgement",
    "path_message",
    "scan_end_message",
    "scan_start_message",
]


def scan_start_message(experiment: str, scan_parameter: str, client_ip: str) -> bytes:
    """``ScanStart>>{experiment}>>{scan parameter}>>{client IP}`` — the IP is the one devices lock to."""
    return SEPARATOR.join((SCAN_START, experiment, scan_parameter, client_ip)).encode(
        "latin-1"
    )


def scan_end_message(experiment: str, argument: str, client_ip: str) -> bytes:
    """``ScanEnd>>{experiment}>>{argument}>>{client IP}``.

    Master Control's third argument differed from the scan parameter in the
    capture (``U_ProbeCamStage Home``) and its meaning is still open; callers
    echo the scan parameter until it is known.
    """
    return SEPARATOR.join((SCAN_END, experiment, argument, client_ip)).encode("latin-1")


def path_message(path: str) -> bytes:
    """``Path>>{path}`` — the per-shot push on the path channel (unframed)."""
    return f"{PATH_KEY}{SEPARATOR}{path}".encode("latin-1")


def device_shot_path(scan_folder: str, scan_number: int, shot_number: int) -> str:
    r"""The shot-path template devices expect, from the device-side scan folder.

    ``{scan_folder}\xxx\Scan{NNN}_xxx_{SSS}.fileextension`` with Windows
    separators: the device replaces ``xxx`` with its own name and
    ``fileextension`` with its own, and creates the folder if missing.
    *scan_folder* is the device-side ``scans\ScanNNN`` folder, however the
    worker derives it (the DB's ``RootPath`` or its own device-side
    translation of the claimed folder — slice 2's call); nothing here
    resolves or checks it: the worker knows the folder, the device owns the
    drive letter.

    The reader side spells the same ``Scan{NNN}_{device}_{SSS}`` name in
    ``geecs_data_utils.scan_paths`` and ``native_files`` — a second home by
    necessity, since Core imports nothing from Data-Utils; keep them in step.
    """
    folder = scan_folder.rstrip("\\")
    return f"{folder}\\xxx\\Scan{scan_number:03d}_xxx_{shot_number:03d}.fileextension"


@dataclass(frozen=True)
class ScanModeAck:
    """One device's acknowledgement of ``ScanStart`` or ``ScanEnd``.

    Attributes
    ----------
    device :
        The device name, the datagram's first field.
    command :
        ``ScanStart`` or ``ScanEnd``.
    status :
        The trailing field: ``no error,`` on success, else the device's
        error text.  The reply carries the verdict — read it.
    raw :
        The datagram as text.
    received_at :
        The listener's ``loop.time()`` at receipt (``0.0`` from the parser).
        The device sends every acknowledgement twice, a few hundred
        milliseconds apart, so a run tells its own replies from the previous
        run's repeat by time — :meth:`ScanModeAckListener.wait_for`'s
        ``since``.
    """

    device: str
    command: str
    status: str
    raw: str
    received_at: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether the device reported success."""
        return self.status.startswith("no error")


def parse_acknowledgement(data: bytes) -> ScanModeAck | None:
    """Parse ``{device}>>{command}>>...>>{status}``; ``None`` for anything else."""
    text = data.decode("latin-1").strip()
    parts = text.split(SEPARATOR)
    if len(parts) < 3 or not parts[0] or parts[1] not in (SCAN_START, SCAN_END):
        return None
    return ScanModeAck(
        device=parts[0], command=parts[1], status=parts[-1].strip(), raw=text
    )


# ---------------------------------------------------------------------------
# Multicast sender
# ---------------------------------------------------------------------------


class _SendOnly(asyncio.DatagramProtocol):
    """The multicaster's protocol: nothing is expected back on its socket."""

    def datagram_received(self, data: bytes, addr: object) -> None:
        logger.debug(
            "scan mode: ignoring %r from %s on the multicast socket", data, addr
        )

    def error_received(self, exc: Exception) -> None:
        logger.warning("scan mode: multicast socket error: %s", exc)


class ScanModeMulticaster:
    """Sends ``Empty`` / ``ScanStart`` / ``ScanEnd`` on the experiment's group.

    Parameters
    ----------
    group, port :
        ``expt.MultiCastIP`` and ``expt.MultiCastRemotePort``.  A unicast
        address works too (tests send to loopback).
    local_ip :
        The address to send from — the gateway's address on the device
        network (``detect_local_ip``).  It is also the IP named in
        ``ScanStart``, the one every device locks its command port to and
        connects back to on the path channel, so it must be the host the
        :class:`ScanModeServer` listens on.
    local_port :
        ``0`` (the default) lets the OS choose.  Binding Master Control's
        ``MultiCastLocalPort`` is not advised: a device replies to the IP
        named in the command at its own port, never to the source port, and
        58431 sits in Linux's ephemeral range, where a gateway on the same
        host holds two ports per device — a collision refuses :meth:`open`.
    ttl :
        Multicast TTL; the default is Master Control's.
    """

    def __init__(
        self,
        group: str,
        port: int,
        *,
        local_ip: str,
        local_port: int = 0,
        ttl: int = MULTICAST_TTL,
        client_ip: str | None = None,
    ) -> None:
        self._group = group
        self._port = int(port)
        self._local_ip = local_ip
        self._client_ip = client_ip or local_ip
        self._local_port = int(local_port)
        self._ttl = int(ttl)
        self._transport: asyncio.DatagramTransport | None = None

    @property
    def client_ip(self) -> str:
        """The address named in ``ScanStart`` — the one devices lock to and connect back to.

        Defaults to *local_ip*.  *client_ip* overrides it with the string the
        devices will see our datagrams coming from: a device converts the
        source address to text before comparing it with the stored address,
        and a host that resolves to a name (mDNS) must announce that name or
        every later command is dropped as foreign (found live, 2026-10-13).
        """
        return self._client_ip

    async def open(self) -> None:
        """Bind the sending socket (multicast interface and TTL set for a group address)."""
        if self._transport is not None:
            raise RuntimeError("ScanModeMulticaster already open")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if ipaddress.ip_address(self._group).is_multicast:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self._ttl)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
                sock.setsockopt(
                    socket.IPPROTO_IP,
                    socket.IP_MULTICAST_IF,
                    socket.inet_aton(self._local_ip),
                )
            sock.bind((self._local_ip, self._local_port))
        except OSError as exc:
            sock.close()
            raise GeecsConnectionError(
                f"scan mode: cannot bind the multicast sender to "
                f"{self._local_ip}:{self._local_port} ({exc.strerror or exc})"
            ) from exc
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(_SendOnly, sock=sock)

    async def close(self) -> None:
        """Release the socket."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    async def __aenter__(self) -> ScanModeMulticaster:
        """Open and return ``self``."""
        await self.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the socket."""
        await self.close()

    def send(self, payload: bytes) -> None:
        """Send one datagram to the group."""
        if self._transport is None:
            raise RuntimeError("ScanModeMulticaster not open — call open() first")
        self._transport.sendto(payload, (self._group, self._port))
        logger.debug("scan mode: tx → %s:%s %r", self._group, self._port, payload)

    async def announce_scan_start(
        self, experiment: str, scan_parameter: str, *, gap_s: float = 0.02
    ) -> bytes:
        """``Empty``, ``Empty``, then ``ScanStart`` naming this client's IP — the captured sequence.

        Returns the ``ScanStart`` datagram sent.
        """
        self.send(EMPTY_MESSAGE)
        await asyncio.sleep(gap_s)
        self.send(EMPTY_MESSAGE)
        await asyncio.sleep(gap_s)
        message = scan_start_message(experiment, scan_parameter, self._client_ip)
        self.send(message)
        return message

    async def announce_scan_end(self, experiment: str, argument: str) -> bytes:
        """``ScanEnd`` on the group, naming the client address.  Returns the datagram sent.

        Sent for fidelity with Master Control: the device framework's
        multicast listener does nothing with it (empty case, read 2026-10-13).
        The scan end a device acts on is :meth:`ScanModeServer.push_scan_end`.
        """
        message = scan_end_message(experiment, argument, self._client_ip)
        self.send(message)
        return message


# ---------------------------------------------------------------------------
# Waiting without losing a cancellation
# ---------------------------------------------------------------------------


async def _wait_event(event: asyncio.Event, timeout: float) -> None:
    """Wait for *event* up to *timeout* seconds; a cancellation always propagates.

    ``asyncio.wait_for(event.wait(), timeout)`` on Python 3.10 and 3.11
    returns normally instead of raising when the event was set in the same
    loop tick as the caller's cancellation (its ``except CancelledError: if
    fut.done(): return fut.result()`` branch), so a loop built on it would
    shrug off an abort and run on to its own deadline.  ``asyncio.wait``
    has no such branch.
    """
    waiter = asyncio.ensure_future(event.wait())
    try:
        await asyncio.wait({waiter}, timeout=timeout)
    finally:
        if not waiter.done():
            waiter.cancel()


# ---------------------------------------------------------------------------
# Acknowledgement listener
# ---------------------------------------------------------------------------


class _AckProtocol(asyncio.DatagramProtocol):
    def __init__(self, listener: ScanModeAckListener) -> None:
        self._listener = listener

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._listener._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        logger.warning("scan mode: acknowledgement socket error: %s", exc)


class ScanModeAckListener:
    """Collects the ``ScanStart`` / ``ScanEnd`` acknowledgements of a set of devices.

    A device acknowledges to the client IP at **its own command-port
    number**, so the listener binds one UDP socket per distinct port in
    *ports* on *local_ip* for as long as it is open.  The TCP connections
    cannot stand in for this: they arrive from ephemeral ports and carry no
    name.  A port already held by another process raises
    :class:`GeecsConnectionError` at :meth:`open` naming the port.

    The device sends each acknowledgement **twice**, a few hundred
    milliseconds apart, so a listener reused across runs must not take an
    old reply for a new one: read ``loop.time()`` before announcing and pass
    it as ``since`` to :meth:`wait_for`.  (A repeat that lands after that
    instant still counts — runs start more than the repeat gap apart.)

    Parameters
    ----------
    local_ip :
        The address to bind: the multicaster's ``local_ip`` (devices reply
        to the address named in ``ScanStart``, which resolves to it).
    ports :
        The command ports of the devices whose acknowledgements matter —
        ``GeecsDb.get_experiment_devices`` gives them; duplicates collapse.
    """

    def __init__(self, local_ip: str, ports: Iterable[int]) -> None:
        self._local_ip = local_ip
        self._ports = tuple(sorted({int(port) for port in ports}))
        self._transports: list[asyncio.DatagramTransport] = []
        self._acks: dict[str, dict[str, ScanModeAck]] = {}
        self._changed: asyncio.Event | None = None

    @property
    def ports(self) -> tuple[int, ...]:
        """The distinct ports the listener binds."""
        return self._ports

    async def open(self) -> None:
        """Bind every port; on failure release what was bound and raise."""
        if self._changed is not None:
            raise RuntimeError("ScanModeAckListener already open")
        loop = asyncio.get_running_loop()
        self._changed = asyncio.Event()
        for port in self._ports:
            try:
                transport, _ = await loop.create_datagram_endpoint(
                    lambda: _AckProtocol(self),
                    local_addr=(self._local_ip, port),
                    family=socket.AF_INET,
                )
            except OSError as exc:
                await self.close()
                raise GeecsConnectionError(
                    f"scan mode: cannot listen for acknowledgements on "
                    f"{self._local_ip}:{port} ({exc.strerror or exc}) — "
                    "another process holds the port"
                ) from exc
            self._transports.append(transport)

    async def close(self) -> None:
        """Release every socket."""
        for transport in self._transports:
            transport.close()
        self._transports.clear()
        self._changed = None

    async def __aenter__(self) -> ScanModeAckListener:
        """Open and return ``self``."""
        await self.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Release the sockets."""
        await self.close()

    def _on_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        ack = parse_acknowledgement(data)
        if ack is None:
            logger.debug("scan mode: ignoring %r from %s on an ack port", data, addr)
            return
        ack = replace(ack, received_at=asyncio.get_running_loop().time())
        bucket = self._acks.setdefault(ack.command, {})
        if ack.device in bucket:
            logger.debug("scan mode: repeat %s ack from %s", ack.command, ack.device)
        else:
            logger.debug(
                "scan mode: %s ack from %s: %s", ack.command, ack.device, ack.status
            )
        bucket[ack.device] = ack
        if self._changed is not None:
            self._changed.set()

    def acknowledgements(
        self, command: str, *, since: float | None = None
    ) -> dict[str, ScanModeAck]:
        """Every acknowledgement of *command* received so far (at or after *since*), by device."""
        return {
            device: ack
            for device, ack in self._acks.get(command, {}).items()
            if since is None or ack.received_at >= since
        }

    def clear(self) -> None:
        """Forget every acknowledgement received so far."""
        self._acks.clear()

    async def wait_for(
        self,
        command: str,
        devices: Collection[str],
        timeout: float,
        *,
        since: float,
    ) -> dict[str, ScanModeAck]:
        """Wait until every device in *devices* has acknowledged *command*, or *timeout* elapses.

        *since* is the ``loop.time()`` read **before** the command was
        announced; nothing received earlier counts (the device's repeat of
        its previous reply would otherwise pass a silent device).  Returns
        the acknowledgements received for *devices* at or after it, by
        device.  A device absent from the result never answered; one present
        may still report an error — check its ``ok``.
        """
        if self._changed is None:
            raise RuntimeError("ScanModeAckListener not open — call open() first")
        wanted = set(devices)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            self._changed.clear()
            have = {
                device: ack
                for device, ack in self.acknowledgements(command, since=since).items()
                if device in wanted
            }
            remaining = deadline - loop.time()
            if wanted <= have.keys() or remaining <= 0:
                return have
            await _wait_event(self._changed, remaining)


# ---------------------------------------------------------------------------
# Path channel server
# ---------------------------------------------------------------------------


async def _bounded_drain(writer: asyncio.StreamWriter, timeout: float) -> bool:
    """``await writer.drain()`` with a deadline; ``False`` when the deadline passed.

    Not ``asyncio.wait_for``: on Python 3.10 and 3.11 that helper returns the
    inner result instead of raising when the inner awaitable completed in the
    same loop tick as the outer cancellation — and ``drain()`` on an empty
    buffer completes at once, so the keepalive task was seen to survive its
    own cancellation and :meth:`ScanModeServer.stop` waited on it forever
    (live, 2026-10-13).  ``asyncio.wait`` neither cancels nor swallows;
    :func:`_wait_event` is the same cure for the event waits.
    """
    task = asyncio.ensure_future(writer.drain())
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    finally:
        if not task.done():
            task.cancel()
    if task not in done:
        return False
    task.result()  # re-raises an OSError from the transport
    return True


class ScanModeServer:
    """The port-6342 server devices connect to after ``ScanStart``.

    Accepts every connection, keeps each alive with a framed ``dummy``
    every *keepalive_interval_s*, drains whatever the device sends back,
    and pushes a framed ``Path>>…`` to every connection on
    :meth:`push_path`.  A connection whose write fails (reset, closed) is
    dropped at once; one whose peer has stopped reading is only noticed
    when the socket's send buffer fills and ``drain`` stalls past
    *write_timeout_s* — minutes of keepalives — and the device's zero-byte
    replies are discarded, not read as a heartbeat.  So this server knows
    **presence, not liveness**: :meth:`push_path` counts connections
    written (a device may hold two), and :meth:`wait_for_hosts` judges by
    peer host, since a connection carries no device name.  Whether a stuck
    device should be dropped — and after how long — waits for live
    evidence; Master Control itself resets some connections for reasons the
    capture did not show.

    Parameters
    ----------
    host :
        The address to listen on — what the address named in ``ScanStart``
        resolves to (the multicaster's ``local_ip``).
    port :
        :data:`SCAN_MODE_TCP_PORT`; ``0`` lets the OS choose (tests), the
        bound port is then :attr:`port`.
    """

    def __init__(
        self,
        host: str,
        port: int = SCAN_MODE_TCP_PORT,
        *,
        keepalive_interval_s: float = KEEPALIVE_INTERVAL_S,
        write_timeout_s: float = 1.0,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._keepalive_interval_s = keepalive_interval_s
        self._write_timeout_s = write_timeout_s
        self._server: asyncio.Server | None = None
        self._writers: dict[asyncio.StreamWriter, tuple[str, int]] = {}
        self._inbound: dict[tuple[str, int], int] = {}
        self._reader_tasks: set[asyncio.Task[None]] = set()
        self._keepalive_task: asyncio.Task[None] | None = None
        self._changed: asyncio.Event | None = None
        self._stopping = False

    @property
    def host(self) -> str:
        """The listening address."""
        return self._host

    @property
    def port(self) -> int:
        """The listening port (the OS-chosen one after :meth:`start` when 0 was given)."""
        return self._port

    @property
    def connection_count(self) -> int:
        """How many device connections are open."""
        return len(self._writers)

    def inbound_bytes(self) -> dict[tuple[str, int], int]:
        """Bytes each **open** connection's peer has sent this run, by peer address.

        The devices' six-byte answers to the keepalives, on the builds that
        answer at all.  Reset at :meth:`start`; a dropped connection's entry
        goes with it.
        """
        return dict(self._inbound)

    @property
    def peer_hosts(self) -> frozenset[str]:
        """The IPs with at least one open connection."""
        return frozenset(peer[0] for peer in self._writers.values())

    async def start(self) -> None:
        """Listen and start the keepalive ticker."""
        if self._server is not None:
            raise RuntimeError("ScanModeServer already started")
        if self._keepalive_task is not None and not self._keepalive_task.done():
            raise RuntimeError(
                "ScanModeServer: the previous keepalive ticker is still stopping"
            )
        self._stopping = False
        self._inbound.clear()
        try:
            self._server = await asyncio.start_server(
                self._on_connection, host=self._host, port=self._port
            )
        except OSError as exc:
            raise GeecsConnectionError(
                f"scan mode: cannot listen on {self._host}:{self._port} "
                f"({exc.strerror or exc})"
            ) from exc
        self._changed = asyncio.Event()
        self._port = self._server.sockets[0].getsockname()[1]
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name="scan-mode-keepalive"
        )
        logger.info("scan mode: path server listening on %s:%s", self._host, self._port)

    async def stop(self) -> None:
        """Stop the ticker, close every connection, stop listening.

        Bounded: the ticker is asked to stop by flag and by cancellation and
        waited for briefly, never indefinitely (see :func:`_bounded_drain`).
        """
        await self.stop_keepalive()
        if self._server is not None:
            self._server.close()
        for writer in list(self._writers):
            self._drop(writer)
        tasks = [task for task in self._reader_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> ScanModeServer:
        """Start and return ``self``."""
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Stop the server."""
        await self.stop()

    async def stop_keepalive(self) -> None:
        """Stop the keepalive ticker; every connection stays open.

        Not part of the exit: the devices leave scan mode on the ``ScanEnd``
        frame (:meth:`push_scan_end`) with the ticker running, as verified
        live.  Offered because Master Control's keepalives are seen to stop
        mid-scan in the capture and a consumer may want to quiesce the
        channel before closing it; :meth:`stop` calls it first.
        """
        self._stopping = True
        task, self._keepalive_task = self._keepalive_task, None
        if task is None:
            return
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=2.0)
        if not done:
            # Keep the reference: start() refuses until the orphan has ended,
            # so it can never write into a new run's connections.
            logger.warning("scan mode: the keepalive ticker did not stop in time")
            self._keepalive_task = task

    async def push_path(self, path: str) -> int:
        """Push ``Path>>{path}`` to every connection; returns how many connections took it."""
        count = await self._write_all(frame(path_message(path)))
        logger.debug("scan mode: pushed %r to %d connection(s)", path, count)
        return count

    async def push_scan_end(
        self, experiment: str, argument: str, client_ip: str
    ) -> int:
        """Push the ``ScanEnd`` frame to every connection; returns how many connections took it.

        **This is the exit the devices act on.**  Master Control sends
        ``ScanEnd`` twice: on the multicast group, where the device
        framework's listener has an empty ``ScanEnd`` case (a dead letter),
        and 40 ms later as a frame on every path-channel connection, where
        the device's TCP loop takes it as the scan end — it runs the stored
        end commands, acknowledges by UDP to *client_ip* at its own port and
        closes the connection (captured 2026-10-13, 200 frames per scan at
        Undulator, and read in the device framework's diagrams).  Send both,
        this one after the multicast.
        """
        count = await self._write_all(
            frame(scan_end_message(experiment, argument, client_ip))
        )
        logger.debug("scan mode: pushed ScanEnd to %d connection(s)", count)
        return count

    async def wait_for_hosts(
        self, hosts: Collection[str], timeout: float
    ) -> frozenset[str]:
        """Wait until every IP in *hosts* has an open connection, or *timeout* elapses.

        Returns the hosts still without one; empty means every host connected.
        """
        if self._changed is None:
            raise RuntimeError("ScanModeServer not started — call start() first")
        wanted = frozenset(hosts)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            self._changed.clear()
            missing = wanted - self.peer_hosts
            remaining = deadline - loop.time()
            if not missing or remaining <= 0:
                return missing
            await _wait_event(self._changed, remaining)

    # ------------------------------------------------------------------

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername") or ("?", 0)
        self._writers[writer] = (str(peer[0]), int(peer[1]))
        task = asyncio.current_task()
        if task is not None:
            self._reader_tasks.add(task)
        if self._changed is not None:
            self._changed.set()
        logger.info(
            "scan mode: device connection from %s:%s (%d open)",
            peer[0],
            peer[1],
            len(self._writers),
        )
        try:
            # The device sends six zero bytes per keepalive and nothing else
            # of interest: drain so its send buffer never fills, and treat
            # end-of-stream as the device leaving scan mode.
            while data := await reader.read(4096):
                self._inbound[(str(peer[0]), int(peer[1]))] = self._inbound.get(
                    (str(peer[0]), int(peer[1])), 0
                ) + len(data)
        except OSError as exc:
            logger.info(
                "scan mode: connection from %s:%s lost: %s", peer[0], peer[1], exc
            )
        finally:
            self._drop(writer)
            if task is not None:
                self._reader_tasks.discard(task)

    def _drop(self, writer: asyncio.StreamWriter) -> None:
        peer = self._writers.pop(writer, None)
        if peer is None:
            return
        self._inbound.pop((str(peer[0]), int(peer[1])), None)
        with contextlib.suppress(Exception):
            writer.close()
        if self._changed is not None:
            self._changed.set()
        logger.info(
            "scan mode: device connection from %s:%s closed (%d open)",
            peer[0],
            peer[1],
            len(self._writers),
        )

    async def _keepalive_loop(self) -> None:
        payload = frame(KEEPALIVE_PAYLOAD)
        while not self._stopping:
            await asyncio.sleep(self._keepalive_interval_s)
            if not self._stopping:
                await self._write_all(payload)

    async def _write_all(self, data: bytes) -> int:
        """Write *data* to every connection at once; returns how many took it.

        Every connection is written and drained independently and the drains
        run concurrently, so a peer whose buffer has filled costs only itself
        the *write_timeout_s* stall — the others keep the 40 ms cadence.
        """
        results = await asyncio.gather(
            *(self._write_one(writer, data) for writer in list(self._writers))
        )
        return sum(1 for ok in results if ok)

    async def _write_one(self, writer: asyncio.StreamWriter, data: bytes) -> bool:
        """Write *data* to one connection; drop it on a failed or stalled write."""
        peer = self._writers.get(writer)
        if peer is None:
            return False  # dropped by its reader in the meantime
        try:
            writer.write(data)
            flushed = await _bounded_drain(writer, self._write_timeout_s)
        except OSError as exc:
            flushed = False
            reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = f"no progress in {self._write_timeout_s}s"
        if flushed:
            return True
        logger.warning(
            "scan mode: dropping the connection from %s:%s — write failed (%s)",
            peer[0],
            peer[1],
            reason,
        )
        self._drop(writer)
        with contextlib.suppress(Exception):
            writer.transport.abort()
        return False
