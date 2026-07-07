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
from typing import Any, Callable, Awaitable, Collection, Sequence

logger = logging.getLogger(__name__)

Callback = Callable[[dict[str, Any]], Awaitable[None] | None]


def _compile_frame_pattern(variables: Sequence[str]) -> re.Pattern[str] | None:
    """Build the push-frame regex anchored on the subscribed variable names.

    The payload is ``"var1 nval,val1 nvar,var2 nval,val2 nvar"``.  Variable
    *names* come from the experiment DB and never contain commas, but *values*
    may (e.g. a ``localsavingpath`` like ``Z:/data/run1,repeat``), so the frame
    cannot be tokenised on commas.  Instead each pair is anchored on a known
    subscribed name followed by the literal `` nval,`` token, and the value runs
    (non-greedily, newlines included) to the `` nvar`` token that sits at a pair
    boundary — i.e. one followed by ``,<comma-free name> nval,`` (the next pair,
    subscribed or not) or by the end of the frame (optionally with a trailing
    comma/whitespace).  Longer names are tried first so a name that is a prefix
    of another cannot shadow it.

    Residual ambiguity, inherent to the wire format: a value that itself
    contains the full boundary text `` nvar,<something-comma-free> nval,`` is
    indistinguishable from a real pair boundary and will be truncated there.
    (The legacy GEECS-PythonAPI parser stops at the *first* `` nvar`` regardless
    of what follows, so this parser is strictly more tolerant.)

    Returns ``None`` for an empty variable list (nothing can match).
    """
    if not variables:
        return None
    names = "|".join(
        re.escape(name) for name in sorted(set(variables), key=len, reverse=True)
    )
    return re.compile(
        rf"(?:^|,)\s*(?P<name>{names}) nval,"
        rf"(?P<value>.*?) nvar"
        rf"(?=,\s*[^,]+ nval,|,?\s*$)",
        re.DOTALL,
    )


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
        self._warned_missing_variables: set[str] = set()

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
                await asyncio.wait_for(self._writer.wait_closed(), timeout=1.0)
            except Exception:
                transport = getattr(self._writer, "transport", None) or getattr(
                    self._writer, "_transport", None
                )
                if transport is not None:
                    transport.abort()
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
        text_variables: Collection[str] = (),
    ) -> None:
        """Send subscription command and start the background push listener.

        Parameters
        ----------
        variables:
            List of variable names to subscribe to.
        callback:
            Called with ``{var_name: value}`` on every push received.
            May be a plain function or a coroutine function.
        text_variables:
            Variable names whose values must be delivered as the exact raw text
            from the wire (string/path-typed channels).  All other variables get
            numeric coercion, which is lossy for text: ``'007'`` → ``7``,
            ``'1.10'`` → ``1.1``, ``'1e5'`` → ``100000.0``.
        """
        if self._writer is None:
            raise RuntimeError(
                "GeecsTcpSubscriber not connected — call connect() first"
            )
        cmd = ("Wait>>" + ",".join(variables)).encode("ascii")
        self._writer.write(struct.pack(">i", len(cmd)) + cmd)
        await self._writer.drain()
        logger.debug("TCP subscribed: %s", variables)
        self._warned_missing_variables.clear()

        self._listen_task = asyncio.create_task(
            self._listen_loop(callback, variables, frozenset(text_variables)),
            name=f"tcp-sub[{self._host}:{self._port}]",
        )

    async def _listen_loop(
        self,
        callback: Callback,
        variables: list[str],
        text_variables: frozenset[str],
    ) -> None:
        """Read framed messages in a loop and dispatch to callback."""
        assert self._reader is not None
        subscribed = tuple(variables)
        pattern = _compile_frame_pattern(subscribed)
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

                parsed = _parse_subscription(msg, pattern, text_variables)
                self._warn_missing_variables(subscribed, parsed)
                if parsed:
                    try:
                        result = callback(parsed)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.warning(
                            "TCP subscription callback failed for %s:%s; "
                            "continuing listener",
                            self._host,
                            self._port,
                            exc_info=True,
                        )

        except asyncio.IncompleteReadError:
            logger.debug("TCP connection closed by server")
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("unexpected error in TCP listener")

    def _warn_missing_variables(
        self, variables: tuple[str, ...], frame: dict[str, Any]
    ) -> None:
        """Warn once for subscribed variables absent from a TCP push frame."""
        missing = [
            var
            for var in variables
            if var not in frame and var not in self._warned_missing_variables
        ]
        if not missing:
            return
        self._warned_missing_variables.update(missing)
        logger.warning(
            "TCP subscription from %s:%s missing variable(s) in push frame: %s",
            self._host,
            self._port,
            ", ".join(missing),
        )


def _parse_subscription(
    msg: str,
    pattern: re.Pattern[str] | None,
    text_variables: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Parse a GEECS subscription push into ``{var_name: value}``.

    Format: ``"DevName>>shot>>var1 nval,val1 nvar,var2 nval,val2 nvar"``.
    The payload is everything after the *second* ``>>`` (values may themselves
    contain ``>>``, so splitting the whole message on it would truncate them)
    and is tokenised by *pattern* — see :func:`_compile_frame_pattern`.

    Values of variables in ``text_variables`` are returned as the exact raw
    text; all others are numerically coerced via :func:`_coerce`.
    """
    if pattern is None:
        return {}
    i1 = msg.find(">>")
    i2 = msg.find(">>", i1 + 2) if i1 >= 0 else -1
    if i2 < 0:
        return {}
    payload = msg[i2 + 2 :]
    result: dict[str, Any] = {}
    for match in pattern.finditer(payload):
        var = match.group("name")
        raw_val = match.group("value")
        result[var] = raw_val if var in text_variables else _coerce(raw_val.strip())
    return result


def _coerce(s: str) -> Any:
    """Best-effort numeric conversion; non-numeric text passes through as-is.

    Lossy for text that merely *looks* numeric (``'007'`` → ``7``) — string-typed
    variables must bypass this via the ``text_variables`` parse parameter.
    """
    try:
        f = float(s)
        return int(f) if f == int(f) and "." not in s else f
    except ValueError:
        return s
