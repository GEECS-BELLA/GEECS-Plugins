"""Asyncio UDP client implementing the GEECS command/ACK/exe protocol.

Command flow
------------
1. Send ``"set{var}>>{value}"`` or ``"get{var}>>"`` to ``(device_ip, device_port)``.
2. Await ACK on the client's own cmd socket (last ``>>`` token == ``"accepted"`` or ``"ok"``).
3. Await exe response on the client's exe socket (cmd_port + 1):
   ``"DevName>>VarName>>value>>no error,"``  or  ``">>error,msg"``.

Both sockets are created by :meth:`connect` and released by :meth:`close`.

Local-IP detection
------------------
GEECS devices sit on a private network reached via a PPP/VPN link on the lab
machines.  Binding a UDP socket to ``""`` raises ``EADDRNOTAVAIL`` on macOS
when the only route to the destination goes through a PPP interface.
:func:`_detect_local_ip` probes which interface the OS would use and we bind
explicitly to that address.  This has no effect on ordinary Ethernet/Wi-Fi
connections (where the probed IP happens to be the default one anyway).
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Any

from geecs_bluesky.exceptions import (
    GeecsCommandFailedError,
    GeecsCommandRejectedError,
    GeecsConnectionError,
)

logger = logging.getLogger(__name__)

_ACK_TIMEOUT = 1.5  # seconds
_EXE_TIMEOUT = 10.0  # seconds


# ---------------------------------------------------------------------------
# Internal datagram protocol helpers
# ---------------------------------------------------------------------------


class _Oneshot(asyncio.DatagramProtocol):
    """DatagramProtocol that resolves a single Future on the first datagram received."""

    def __init__(self) -> None:
        self._future: asyncio.Future[bytes] | None = None

    def arm(self, loop: asyncio.AbstractEventLoop) -> asyncio.Future[bytes]:
        """Return a new Future that will be resolved by the next datagram."""
        self._future = loop.create_future()
        return self._future

    def datagram_received(self, data: bytes, addr: object) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(data)

    def error_received(self, exc: Exception) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_exception(exc)

    def connection_lost(self, exc: Exception | None) -> None:
        if exc and self._future is not None and not self._future.done():
            self._future.set_exception(exc)


# ---------------------------------------------------------------------------
# Local-IP auto-detection
# ---------------------------------------------------------------------------


def _detect_local_ip(remote_host: str, remote_port: int = 80) -> str:
    """Return the local IP the OS would use to reach *remote_host*.

    Uses a no-op UDP ``connect`` (no packets sent) so it works even when
    the remote host is unreachable.  Falls back to ``""`` on failure.
    """
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect((remote_host, remote_port))
        local_ip = probe.getsockname()[0]
        probe.close()
        return local_ip
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------


class GeecsUdpClient:
    """Asyncio UDP client for a single GEECS device.

    Parameters
    ----------
    host:
        Device IP address.
    port:
        Device UDP port.
    ack_timeout:
        Seconds to wait for command ACK.
    exe_timeout:
        Seconds to wait for execution response.
    local_host:
        Local IP address to bind sockets to.  Leave empty (default) to
        auto-detect the interface that routes to *host* at :meth:`connect`
        time — necessary on PPP/VPN links where binding to ``""`` raises
        ``EADDRNOTAVAIL``.

    Notes
    -----
    GEECS devices process one UDP command at a time and reject any command
    that arrives while they are still handling the previous one.  An
    :class:`asyncio.Lock` inside :meth:`_exchange` ensures that concurrent
    coroutine calls are serialised even when multiple signal backends share
    the same client instance.
    """

    def __init__(
        self,
        host: str,
        port: int,
        ack_timeout: float = _ACK_TIMEOUT,
        exe_timeout: float = _EXE_TIMEOUT,
        local_host: str = "",
        device_name: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._device_name = device_name
        self.ack_timeout = ack_timeout
        self.exe_timeout = exe_timeout
        # Stored as-supplied; resolved at connect() time if empty.
        self._local_host_override = local_host
        self._local_host: str = local_host  # updated in connect()

        self._cmd_transport: asyncio.DatagramTransport | None = None
        self._exe_transport: asyncio.DatagramTransport | None = None
        self._cmd_proto: _Oneshot | None = None
        self._exe_proto: _Oneshot | None = None
        self._cmd_port: int = -1
        self._exe_port: int = -1
        # Serialise concurrent get/set — see class docstring.
        self._lock: asyncio.Lock = asyncio.Lock()

    async def connect(self) -> None:
        """Bind cmd and exe sockets on OS-assigned ports.

        Auto-detects the local interface IP at call time (not at construction)
        so that any PPP/VPN link established after the object was created is
        correctly picked up.
        """
        # Resolve the local bind address fresh each time we connect.
        self._local_host = self._local_host_override or _detect_local_ip(self._host)

        loop = asyncio.get_running_loop()

        cmd_proto = _Oneshot()
        cmd_transport, _ = await loop.create_datagram_endpoint(
            lambda: cmd_proto,
            local_addr=(self._local_host, 0),
            family=socket.AF_INET,
        )
        self._cmd_transport = cmd_transport  # type: ignore[assignment]
        self._cmd_proto = cmd_proto
        self._cmd_port = cmd_transport.get_extra_info("sockname")[1]

        # exe socket binds to cmd_port + 1 — may fail if that port is taken.
        exe_proto = _Oneshot()
        exe_transport, _ = await loop.create_datagram_endpoint(
            lambda: exe_proto,
            local_addr=(self._local_host, self._cmd_port + 1),
            family=socket.AF_INET,
        )
        self._exe_transport = exe_transport  # type: ignore[assignment]
        self._exe_proto = exe_proto
        self._exe_port = self._cmd_port + 1

        logger.debug(
            "GeecsUdpClient bound: local=%s cmd=%s exe=%s → device %s:%s",
            self._local_host,
            self._cmd_port,
            self._exe_port,
            self._host,
            self._port,
        )

    async def close(self) -> None:
        """Release both sockets."""
        for t in (self._cmd_transport, self._exe_transport):
            if t is not None:
                t.close()
        self._cmd_transport = self._exe_transport = None
        self._cmd_proto = self._exe_proto = None

    async def __aenter__(self) -> "GeecsUdpClient":
        """Connect and return ``self``."""
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close both sockets."""
        await self.close()

    # ------------------------------------------------------------------

    async def get(self, variable: str, timeout: float | None = None) -> Any:
        """Send a get command and return the parsed value from the exe response."""
        cmd = f"get{variable}>>"
        return await self._exchange(variable, cmd, timeout or self.exe_timeout)

    async def set(self, variable: str, value: Any, timeout: float | None = None) -> Any:
        """Send a set command and return the confirmed value from the exe response."""
        if isinstance(value, bool):
            cmd = f"set{variable}>>{int(value)}"
        elif isinstance(value, float):
            cmd = f"set{variable}>>{value:.12f}"
        else:
            cmd = f"set{variable}>>{value}"
        return await self._exchange(variable, cmd, timeout or self.exe_timeout)

    # ------------------------------------------------------------------

    async def _exchange(self, variable: str, cmd: str, exe_timeout: float) -> Any:
        """Acquire the device lock then run the full cmd/ACK/exe round-trip."""
        async with self._lock:
            return await self._exchange_inner(variable, cmd, exe_timeout)

    async def _exchange_inner(self, variable: str, cmd: str, exe_timeout: float) -> Any:
        """Inner exchange — called while holding ``self._lock``.

        Checking connection state inside the lock (rather than before acquiring
        it) avoids a TOCTOU window where ``close()`` could run between the
        check and the first socket operation.
        """
        if self._cmd_transport is None or self._cmd_proto is None:
            raise RuntimeError(
                f"GeecsUdpClient not connected — call connect() first "
                f"({self._host}:{self._port})"
            )

        loop = asyncio.get_running_loop()

        # Arm futures *before* sending so we don't miss the reply.
        ack_future = self._cmd_proto.arm(loop)
        exe_future = self._exe_proto.arm(loop)  # type: ignore[union-attr]

        self._cmd_transport.sendto(cmd.encode("ascii"), (self._host, self._port))
        logger.debug("UDP tx → %s:%s  %r", self._host, self._port, cmd)

        # 1) Wait for ACK
        try:
            ack_data = await asyncio.wait_for(ack_future, timeout=self.ack_timeout)
        except asyncio.TimeoutError:
            raise GeecsConnectionError(
                f"{self._device_name or self._host}/{variable}: "
                f"no ACK within {self.ack_timeout}s"
            ) from None

        ack_str = ack_data.decode("ascii", errors="replace").split(">>")[-1]
        if ack_str not in ("accepted", "ok"):
            raise GeecsCommandRejectedError(
                self._device_name or self._host,
                variable,
                f"ACK was {ack_str!r}",
            )

        # 2) Wait for exe response
        try:
            exe_data = await asyncio.wait_for(exe_future, timeout=exe_timeout)
        except asyncio.TimeoutError:
            raise GeecsConnectionError(
                f"{self._device_name or self._host}/{variable}: "
                f"no exe response within {exe_timeout}s"
            ) from None

        return _parse_exe_response(
            exe_data.decode("ascii", errors="replace"),
            self._device_name or self._host,
            variable,
        )


def _parse_exe_response(msg: str, device_name: str = "", variable: str = "") -> Any:
    """Parse ``DevName>>VarName>>value>>no error,`` and return the value.

    The status field (``parts[3]``) is ``"no error,"`` on success and starts
    with ``"error"`` on failure — ``"ok,"`` was expected from the protocol
    docs but real hardware uses ``"no error,"``.  Checking only the prefix
    handles both.
    """
    parts = msg.split(">>")
    if len(parts) < 4:
        raise ValueError(f"Malformed exe response: {msg!r}")
    status_field = parts[3]
    if status_field.startswith("error"):
        detail = status_field.split(",", 1)[1].strip() if "," in status_field else ""
        raise GeecsCommandFailedError(device_name, variable, detail or status_field)
    raw_value = parts[2]
    # Try to coerce to numeric
    try:
        f = float(raw_value)
        return int(f) if f == int(f) and "." not in raw_value else f
    except ValueError:
        return raw_value
