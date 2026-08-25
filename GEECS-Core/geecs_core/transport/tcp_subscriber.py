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
import socket
import struct
from typing import Any, Callable, Awaitable, Collection, Sequence

from ._coerce import coerce_scalar

logger = logging.getLogger(__name__)

Callback = Callable[[dict[str, Any]], Awaitable[None] | None]


def _apply_keepalive(
    sock: socket.socket, idle_s: float, interval_s: float, count: int
) -> None:
    """Enable TCP keepalive on *sock*, tuned where the platform allows.

    Why: the subscription is read-only after the ``Wait>>`` command, so the
    OS never sends on it — an *ungracefully* dead peer (host crash, power
    cycle, a partition that eats the FIN/RST) leaves a half-open socket the
    listener waits on forever, with readbacks frozen at valid-stale values
    (live incident 2026-08-17, u_s1h; issue #611).  Keepalive probes are
    answered by the peer's TCP stack even when the application is silent,
    so a legitimately idle GEECS device is never disturbed — only a truly
    dead peer fails the probes, which surfaces as a connection reset ending
    the listener (the supervised-reconnect path).

    Tuning knobs are platform-guarded: ``TCP_KEEPIDLE`` (Linux, newer
    Windows) / ``TCP_KEEPALIVE`` (macOS) for the idle time, then
    ``TCP_KEEPINTVL`` / ``TCP_KEEPCNT`` where present.  Where only
    ``SO_KEEPALIVE`` exists the OS defaults apply (typically ~2 h — still
    strictly better than never).  A tuning failure is logged and ignored;
    it must never fail the connect.
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        logger.warning("could not enable TCP keepalive", exc_info=True)
        return
    idle = max(1, round(idle_s))
    interval = max(1, round(interval_s))
    probes = max(1, int(count))
    try:
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle)
        elif hasattr(socket, "TCP_KEEPALIVE"):  # macOS spelling of the idle knob
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, idle)
        if hasattr(socket, "TCP_KEEPINTVL"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval)
        if hasattr(socket, "TCP_KEEPCNT"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, probes)
    except OSError:
        logger.warning("could not tune TCP keepalive timing", exc_info=True)


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
    keepalive:
        Enable TCP keepalive on the connection (default True) — the OS
        probes an idle peer, so an *ungracefully* dead one (host crash,
        power cycle, FIN-eating partition) resets the connection instead
        of leaving it half-open with the listener waiting forever (issue
        #611).  Probes never disturb a live-but-quiet device, so the
        "silence is not a drop" doctrine is preserved.
    keepalive_idle_s, keepalive_interval_s, keepalive_count:
        Probe timing where the platform exposes it (see
        :func:`_apply_keepalive`): start probing after ``idle`` seconds of
        silence, every ``interval`` seconds, declaring the peer dead after
        ``count`` unanswered probes — defaults detect a dead peer in
        roughly a minute.
    """

    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout: float = 5.0,
        keepalive: bool = True,
        keepalive_idle_s: float = 30.0,
        keepalive_interval_s: float = 10.0,
        keepalive_count: int = 3,
    ) -> None:
        self._host = host
        self._port = port
        self.connect_timeout = connect_timeout
        self._keepalive = keepalive
        self._keepalive_idle_s = keepalive_idle_s
        self._keepalive_interval_s = keepalive_interval_s
        self._keepalive_count = keepalive_count

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._listen_task: asyncio.Task | None = None
        self._warned_missing_variables: set[str] = set()

    async def connect(self) -> None:
        """Open the TCP connection (keepalive-enabled unless disabled)."""
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port),
            timeout=self.connect_timeout,
        )
        if self._keepalive:
            sock = self._writer.get_extra_info("socket")
            if sock is not None:
                _apply_keepalive(
                    sock,
                    self._keepalive_idle_s,
                    self._keepalive_interval_s,
                    self._keepalive_count,
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

    async def wait_disconnected(self) -> None:
        """Return when the push listener exits (socket drop or ``close()``).

        Lets a caller supervise the subscription: await this after
        :meth:`subscribe`, and reconnect when it returns. Returns immediately
        if no listener is running.
        """
        if self._listen_task is not None:
            await asyncio.shield(self._listen_task)

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
        include_shot: bool = False,
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
        include_shot:
            When True, each callback dict additionally carries the frame's shot
            counter under the reserved key ``"shot number"`` (an ``int``).  The
            key is only attached to frames where at least one subscribed
            variable matched, so the default callback-gating behavior is
            unchanged.  Subscribing a GEECS variable literally named
            ``"shot number"`` together with this option raises ``ValueError``
            (the counter would silently overwrite the variable's value).
        """
        if include_shot and "shot number" in variables:
            raise ValueError(
                'include_shot=True reserves the key "shot number"; it cannot '
                "also be a subscribed variable name"
            )
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
            self._listen_loop(
                callback, variables, frozenset(text_variables), include_shot
            ),
            name=f"tcp-sub[{self._host}:{self._port}]",
        )

    async def _listen_loop(
        self,
        callback: Callback,
        variables: list[str],
        text_variables: frozenset[str],
        include_shot: bool = False,
    ) -> None:
        """Read framed messages in a loop and dispatch to callback."""
        assert self._reader is not None
        subscribed = tuple(variables)
        pattern = _compile_frame_pattern(subscribed)
        warned_bad_shot = False
        try:
            while True:
                # Read 4-byte header
                header = await self._reader.readexactly(4)
                msg_len = struct.unpack(">i", header)[0]
                if msg_len <= 0:
                    continue
                payload = await self._reader.readexactly(msg_len)
                # latin-1 is a lossless byte<->str map: binary payloads (image
                # frames) survive exactly, and ASCII scalar frames are unchanged.
                msg = payload.decode("latin-1")
                # Truncated repr: image frames are multi-MB, scalar frames tiny.
                logger.debug("TCP rx (%d bytes): %.200r", len(msg), msg)

                parsed = _parse_subscription(
                    msg, pattern, text_variables, include_shot=include_shot
                )
                if (
                    include_shot
                    and parsed
                    and "shot number" not in parsed
                    and not warned_bad_shot
                ):
                    # A malformed shot field is a device-configuration property
                    # and recurs on every ~5 Hz frame — warn once, like
                    # _warn_missing_variables.
                    warned_bad_shot = True
                    logger.warning(
                        "non-integer shot field in push frames from %s:%s; "
                        '"shot number" will be omitted (warning once)',
                        self._host,
                        self._port,
                    )
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
        except OSError as exc:
            # The ungraceful peer-death path — e.g. ConnectionResetError from
            # a keepalive-detected dead peer (#611) or an RST from a rebooted
            # host.  Ordinary connection loss, not "unexpected": the caller's
            # supervisor owns the once-per-episode down logging.
            logger.info("TCP connection to %s:%s lost: %s", self._host, self._port, exc)
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
    include_shot: bool = False,
) -> dict[str, Any]:
    """Parse a GEECS subscription push into ``{var_name: value}``.

    Format: ``"DevName>>shot>>var1 nval,val1 nvar,var2 nval,val2 nvar"``.
    The payload is everything after the *second* ``>>`` (values may themselves
    contain ``>>``, so splitting the whole message on it would truncate them)
    and is tokenised by *pattern* — see :func:`_compile_frame_pattern`.

    Values of variables in ``text_variables`` are returned as the exact raw
    text; all others are numerically coerced via :func:`coerce_scalar`.

    With ``include_shot=True``, the frame's shot counter (the field between the
    first and second ``>>``) is added under the reserved key ``"shot number"``
    — but only when at least one subscribed variable matched and the counter
    parses as an integer (otherwise the key is silently omitted; the listener
    loop warns once per subscription), so empty frames still return ``{}``.
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
        result[var] = (
            raw_val if var in text_variables else coerce_scalar(raw_val.strip())
        )
    if include_shot and result:
        try:
            result["shot number"] = int(msg[i1 + 2 : i2])
        except ValueError:
            pass
    return result
