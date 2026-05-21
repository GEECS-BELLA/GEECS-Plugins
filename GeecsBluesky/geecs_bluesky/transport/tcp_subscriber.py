"""Asyncio TCP subscription client for GEECS devices.

Protocol
--------
1. Connect to ``(device_ip, device_port)`` over TCP.
2. Send framed subscription command: 4-byte big-endian length + ``"Wait>>var1,var2"``.
3. Server pushes framed updates at 5 Hz:
   ``"DevName>>shot>>var1 nval,val1 nvar,var2 nval,val2 nvar"``.

Framing: ``struct.pack(">i", len(payload)) + payload``.

Usage::

    async def my_callback(update: dict[str, Any]) -> None:
        print(update)  # {"Jet_X (mm)": 5.23, ...}

    async with GeecsTcpSubscriber("127.0.0.1", 9000) as sub:
        await sub.subscribe(["Jet_X (mm)", "Jet_Y (mm)"], my_callback)
        await asyncio.sleep(5)
"""

from __future__ import annotations

import asyncio
import logging
import re
import struct
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Matches "varname nval,value nvar" pairs inside the payload
_VAR_PATTERN = re.compile(r"[^,]+ nval,[^,]+ nvar")

Callback = Callable[[dict[str, Any]], Awaitable[None] | None]


class GeecsTcpSubscriber:
    """Asyncio TCP subscription client for one GEECS device.

    Parameters
    ----------
    host:
        Device IP address.
    port:
        Device TCP port (same as UDP port for GEECS devices).
    connect_timeout:
        Seconds allowed for the initial TCP connection.
    """

    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout: float = 5.0,
    ) -> None:
        self._host = host
        self._port = port
        self.connect_timeout = connect_timeout

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._listen_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Open the TCP connection."""
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port),
            timeout=self.connect_timeout,
        )
        logger.debug("TCP connected to %s:%s", self._host, self._port)

    async def close(self) -> None:
        """Cancel listener task and close the TCP connection."""
        if self._listen_task is not None and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
        self._reader = None

    async def __aenter__(self) -> "GeecsTcpSubscriber":
        """Connect and return ``self``."""
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the connection."""
        await self.close()

    # ------------------------------------------------------------------

    async def subscribe(
        self,
        variables: list[str],
        callback: Callback,
    ) -> None:
        """Send subscription command and start the background push listener.

        Parameters
        ----------
        variables:
            List of variable names to subscribe to.
        callback:
            Called with ``{var_name: value}`` on every push received.
            May be a plain function or a coroutine function.
        """
        if self._writer is None:
            raise RuntimeError(
                "GeecsTcpSubscriber not connected — call connect() first"
            )
        cmd = ("Wait>>" + ",".join(variables)).encode("ascii")
        self._writer.write(struct.pack(">i", len(cmd)) + cmd)
        await self._writer.drain()
        logger.debug("TCP subscribed: %s", variables)

        self._listen_task = asyncio.create_task(
            self._listen_loop(callback),
            name=f"tcp-sub[{self._host}:{self._port}]",
        )

    async def _listen_loop(self, callback: Callback) -> None:
        """Read framed messages in a loop and dispatch to callback."""
        assert self._reader is not None
        try:
            while True:
                # Read 4-byte header
                header = await self._reader.readexactly(4)
                msg_len = struct.unpack(">i", header)[0]
                if msg_len <= 0:
                    continue
                payload = await self._reader.readexactly(msg_len)
                msg = payload.decode("ascii", errors="replace")
                logger.debug("TCP rx: %r", msg)

                parsed = _parse_subscription(msg)
                if parsed:
                    result = callback(parsed)
                    if asyncio.iscoroutine(result):
                        await result

        except asyncio.IncompleteReadError:
            logger.debug("TCP connection closed by server")
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("unexpected error in TCP listener")


def _parse_subscription(msg: str) -> dict[str, Any]:
    """Parse a GEECS subscription push into ``{var_name: value}``."""
    # Format: "DevName>>shot>>var1 nval,val1 nvar,var2 nval,val2 nvar"
    blocks = msg.split(">>")
    if len(blocks) < 3:
        return {}
    payload = blocks[-1]
    result: dict[str, Any] = {}
    for match in _VAR_PATTERN.finditer(payload):
        s = match.group()
        # s = "varname nval,value nvar"
        left, right = s.split(",", 1)
        var = left[:-5].strip()  # strip " nval"
        raw_val = right[:-5].strip()  # strip " nvar"
        result[var] = _coerce(raw_val)
    return result


def _coerce(s: str) -> Any:
    try:
        f = float(s)
        return int(f) if f == int(f) and "." not in s else f
    except ValueError:
        return s
