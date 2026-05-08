"""Fake GEECS device server speaking the real UDP/TCP wire protocol.

.. note::

   :meth:`FakeGeecsDevice.fire_shot` advances the ``acq_timestamp`` variable
   to simulate a laser shot.  Tests that exercise
   :class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable` should include
   ``"acq_timestamp": 1000.0`` in the device variables and call
   ``device.fire_shot()`` (from a background task) to unblock ``trigger()``.

Runs entirely in-process on localhost, no real hardware required.
Use ``FakeGeecsServer`` as an async context manager in tests.

Wire protocol recap
-------------------
UDP (device_port):
  client → server: ``"set{var}>>{value}"``  or  ``"get{var}>>"``
  server → client (src_port):     ACK: ``"accepted"``
  server → client (src_port + 1): exe: ``"DevName>>VarName>>value>>ok,"``

TCP (same device_port):
  client → server (framed): ``"Wait>>var1,var2"``
  server → client (framed, 5 Hz): ``"DevName>>shot>>var1 nval,val1 nvar,var2 nval,val2 nvar"``

Framing: 4-byte big-endian signed int length prefix + ASCII payload.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_PUSH_HZ = 5.0
_PUSH_INTERVAL = 1.0 / _PUSH_HZ


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FakeGeecsDevice:
    """In-memory state for one simulated GEECS device.

    Parameters
    ----------
    name:
        Device name as it appears in protocol messages (e.g. ``"U_ESP_JetXYZ"``).
    variables:
        Initial state dict mapping variable name → value.
        Values may be float, int, bool, or str.
    """

    name: str
    variables: dict[str, Any] = field(default_factory=dict)

    def get(self, var: str) -> Any:
        """Return the current value of ``var``, or ``None`` if not found."""
        return self.variables.get(var)

    def set(self, var: str, value: Any) -> bool:
        """Set ``var`` to ``value``; return ``False`` if the variable is unknown."""
        if var not in self.variables:
            return False
        self.variables[var] = value
        return True

    def build_subscription_payload(self, var_names: list[str], shot: int) -> str:
        """Format the 5-Hz push message for the requested variables."""
        parts = []
        for v in var_names:
            val = self.variables.get(v, 0.0)
            parts.append(f"{v} nval,{val} nvar")
        return f"{self.name}>>{shot}>>" + ",".join(parts)

    def fire_shot(self) -> None:
        """Simulate a laser shot by advancing ``acq_timestamp``.

        Increments ``acq_timestamp`` by 1.0 (or sets it to
        ``time.time()`` if the variable is not already present).
        Call this from a background task in tests that exercise
        :class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable`.
        """
        import time

        if "acq_timestamp" in self.variables:
            self.variables["acq_timestamp"] = (
                float(self.variables["acq_timestamp"]) + 1.0
            )
        else:
            self.variables["acq_timestamp"] = float(time.time())

    def build_exe_response(self, var: str, error: bool = False) -> str:
        """Build a GEECS exe-response string for ``var``."""
        val = self.variables.get(var, 0.0)
        if error:
            return f"{self.name}>>{var}>>{val}>>error,unknown variable"
        return f"{self.name}>>{var}>>{val}>>ok,"


# ---------------------------------------------------------------------------
# UDP protocol handler
# ---------------------------------------------------------------------------


class _GeecsUdpProtocol(asyncio.DatagramProtocol):
    """Handles incoming UDP commands and sends ACK + exe responses."""

    def __init__(self, device: FakeGeecsDevice) -> None:
        self._device = device
        self._transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
        self._transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        client_ip, client_cmd_port = addr
        client_exe_port = client_cmd_port + 1

        try:
            msg = data.decode("ascii").strip()
        except UnicodeDecodeError:
            logger.warning("received non-ASCII UDP datagram from %s", addr)
            return

        logger.debug("UDP rx from %s:%s  →  %r", client_ip, client_cmd_port, msg)

        # Parse get/set
        if ">>" not in msg:
            logger.warning("malformed UDP command (no >>): %r", msg)
            return

        sep = msg.index(">>")
        op_and_var = msg[:sep]  # e.g. "setJet_X (mm)" or "getJet_X (mm)"
        value_str = msg[sep + 2 :]  # everything after >>

        if op_and_var.startswith("set"):
            var = op_and_var[3:]
            value = self._coerce(value_str)
            ok = self._device.set(var, value)
        elif op_and_var.startswith("get"):
            var = op_and_var[3:]
            ok = var in self._device.variables
        else:
            logger.warning("unknown UDP op: %r", op_and_var)
            return

        assert self._transport is not None

        # 1) ACK on the client's cmd port
        self._transport.sendto(b"accepted", (client_ip, client_cmd_port))

        # 2) Exe response on the client's exe port (cmd_port + 1)
        exe_msg = self._device.build_exe_response(var, error=not ok)
        self._transport.sendto(exe_msg.encode("ascii"), (client_ip, client_exe_port))
        logger.debug("UDP tx exe → %s:%s  %r", client_ip, client_exe_port, exe_msg)

    @staticmethod
    def _coerce(s: str) -> Any:
        """Try to parse as float, then int, then leave as str."""
        try:
            f = float(s)
            # Preserve int if it looks like one
            return int(f) if f == int(f) and "." not in s else f
        except ValueError:
            return s

    def error_received(self, exc: Exception) -> None:
        logger.error("UDP error: %s", exc)


# ---------------------------------------------------------------------------
# TCP subscription handler
# ---------------------------------------------------------------------------


class _TcpSubscriptionHandler:
    """Manages one TCP subscription connection."""

    def __init__(
        self,
        device: FakeGeecsDevice,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._device = device
        self._reader = reader
        self._writer = writer
        self._task: asyncio.Task | None = None

    async def run(self) -> None:
        """Read the subscription command then push updates until closed."""
        try:
            # Read 4-byte length prefix
            header = await self._reader.readexactly(4)
            msg_len = struct.unpack(">i", header)[0]
            payload = await self._reader.readexactly(msg_len)
            cmd = payload.decode("ascii")
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return

        # Expected format: "Wait>>var1,var2,..."
        if not cmd.startswith("Wait>>"):
            logger.warning("TCP: unexpected command %r", cmd)
            return

        var_names = cmd[len("Wait>>") :].split(",")
        logger.debug("TCP subscription for vars: %s", var_names)

        shot = 0
        try:
            while True:
                shot += 1
                msg = self._device.build_subscription_payload(var_names, shot)
                encoded = msg.encode("ascii")
                frame = struct.pack(">i", len(encoded)) + encoded
                try:
                    self._writer.write(frame)
                    await self._writer.drain()
                except (ConnectionResetError, BrokenPipeError):
                    break
                await asyncio.sleep(_PUSH_INTERVAL)
        finally:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Top-level server
# ---------------------------------------------------------------------------


class FakeGeecsServer:
    """Asyncio-based server that speaks the GEECS UDP/TCP wire protocol.

    Typical test usage::

        async with FakeGeecsServer(device) as srv:
            # srv.host, srv.port available
            client = GeecsUdpClient(srv.host, srv.port)
            ...

    The server binds to an OS-assigned port on localhost by default.
    ``port=0`` means "let the OS choose"; the chosen port is then exposed
    on the ``port`` attribute after ``__aenter__``.
    """

    def __init__(
        self,
        device: FakeGeecsDevice,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._device = device
        self.host = host
        self.port = port
        self._udp_transport: asyncio.DatagramTransport | None = None
        self._tcp_server: asyncio.Server | None = None

    async def start(self) -> None:
        """Start UDP and TCP listeners."""
        loop = asyncio.get_running_loop()

        # UDP
        udp_transport, _ = await loop.create_datagram_endpoint(
            lambda: _GeecsUdpProtocol(self._device),
            local_addr=(self.host, self.port),
        )
        self._udp_transport = udp_transport  # type: ignore[assignment]
        udp_port = udp_transport.get_extra_info("sockname")[1]

        # TCP on the same port number
        self._tcp_server = await asyncio.start_server(
            self._handle_tcp,
            host=self.host,
            port=udp_port,
        )

        self.port = udp_port
        logger.info(
            "FakeGeecsServer '%s' listening on %s:%s (UDP+TCP)",
            self._device.name,
            self.host,
            self.port,
        )

    async def stop(self) -> None:
        """Shut down both listeners."""
        if self._tcp_server is not None:
            self._tcp_server.close()
            await self._tcp_server.wait_closed()
        if self._udp_transport is not None:
            self._udp_transport.close()

    async def _handle_tcp(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        handler = _TcpSubscriptionHandler(self._device, reader, writer)
        await handler.run()

    async def __aenter__(self) -> "FakeGeecsServer":
        """Start the server and return ``self``."""
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Stop the server."""
        await self.stop()
