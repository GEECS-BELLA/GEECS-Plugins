"""ShotGate — a facility's software interlock server, read before every fire.

The server broadcasts its interlocks' verdict one way over TCP, a frame
on connect and then every ~0.1 s (the heartbeat).  Under a trigger
profile naming it (``TriggerProfile.shot_gate``) the worker waits for
permission before every ``SINGLESHOT``; a permission to fire, not a
safety system.  When a shot may fire is the server's protocol (below);
how the worker waits for it is :mod:`geecs_bluesky.plans.shot_gate`.

- **The frame**: a 4-byte big-endian length, then UTF-8 JSON (``v`` 1:
  ``ok``, ``reason``, and per interlock ``unsafe`` and ``data_stamp``).
  Safe only when ``ok`` and the interlocks (at least one) all say safe;
  anything else — another version, a malformed frame — is not safe.
- **One fresh verdict per fire**: after :meth:`ShotGate.fired`, a safe
  frame permits the next fire only when every ``data_stamp`` it carries is
  later than the fire (+ ``holdoff_s``), so the verdict rests on data
  taken after the shot; a frame with no ``data_stamp`` must be *received*
  that late instead.  Before the first fire any safe frame permits.
- **Fail closed**: unreachable, no frame within :data:`HEARTBEAT_S`, a
  frame over :data:`MAX_FRAME` — not safe, drop and reconnect.
- **One reader per address** for the process (:func:`gate_for`), so the
  last-fire mark follows the physical server across runs and profiles;
  each profile's hold settings stay its own.

Standard library only: the web scanner runs a read-only reader for its chip.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import struct
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Seconds between reconnect attempts after the connection drops or is refused.
RECONNECT_S = 1.0
#: Seconds a run waits at its start for a connection and its first frame.
CONNECT_TIMEOUT_S = 5.0
#: Default seconds a slot holds for permission before it pauses the run.
DEFAULT_HOLD_PAUSE_S = 60.0
#: Seconds without a frame before the connection counts as lost (10× the
#: server's 0.1 s broadcast).
HEARTBEAT_S = 1.0
#: Largest frame body accepted; anything larger is a protocol error (reconnect).
MAX_FRAME = 64 * 1024
#: The protocol version this client speaks.
PROTOCOL_VERSION = 1

_LENGTH = struct.Struct(">I")


@dataclass(frozen=True)
class GateMessage:
    """One verdict: safe or not, why not, and the data stamps it rests on."""

    safe: bool
    reason: str = ""
    #: The interlocks' non-null ``data_stamp`` values (Unix seconds).
    data_stamps: tuple[float, ...] = ()


def _not_understood(why: str) -> GateMessage:
    return GateMessage(False, f"frame not understood: {why}")


def _read_interlocks(interlocks: dict) -> tuple[list[str], list[float]]:
    """The unsafe interlocks' diagnostics and every non-null data stamp.

    Raises :class:`ValueError` for an interlock without a boolean
    ``unsafe`` or with a ``data_stamp`` that is not a finite number.
    """
    unsafe: list[str] = []
    stamps: list[float] = []
    for name, state in interlocks.items():
        if not isinstance(state, dict) or not isinstance(state.get("unsafe"), bool):
            raise ValueError(f"interlock {name!r} has no unsafe flag")
        if state["unsafe"]:
            unsafe.append(f"{name}: {state.get('diagnostic') or 'unsafe'}")
        stamp = state.get("data_stamp")
        if stamp is None:
            continue
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
            raise ValueError(f"interlock {name!r} data_stamp {stamp!r}")
        try:
            value = float(stamp)
        except OverflowError:  # an integer too large for a float
            value = math.inf
        if not math.isfinite(value):
            raise ValueError(f"interlock {name!r} data_stamp is not finite")
        stamps.append(value)
    return unsafe, stamps


def parse_frame(body: bytes) -> GateMessage:
    """One frame's JSON body → a verdict; anything not understood is not safe."""
    try:
        frame = json.loads(body.decode("utf-8"))
    except (ValueError, RecursionError) as exc:  # bad UTF-8 or JSON; nesting
        return _not_understood(f"{type(exc).__name__}: {exc}"[:200])
    if not isinstance(frame, dict):
        return _not_understood("not a JSON object")
    if frame.get("v") != PROTOCOL_VERSION:
        return GateMessage(False, f"protocol version {frame.get('v')!r} not understood")
    interlocks = frame.get("interlocks")
    if not isinstance(interlocks, dict):
        return _not_understood("no interlocks")
    if not interlocks:
        return GateMessage(False, "no interlocks registered")
    try:
        unsafe, stamps = _read_interlocks(interlocks)
    except ValueError as exc:
        return _not_understood(str(exc))
    if unsafe or frame.get("ok") is not True:
        reason = str(frame.get("reason") or "") or "; ".join(unsafe) or "not ok"
        return GateMessage(False, reason, tuple(stamps))
    return GateMessage(True, "", tuple(stamps))


class _Link:
    """One address's reader and last-fire mark, shared by every :class:`ShotGate` on it."""

    def __init__(self, host: str, port: int, parse: Callable[[bytes], GateMessage]):
        self.host = host
        self.port = int(port)
        self.parse = parse
        self.latest = GateMessage(False, "not connected to the shot gate")
        self.received: float | None = None  # monotonic, of the latest frame
        self.fire_mono: float | None = None
        self.fire_wall: float | None = None  # Unix, for the data stamps
        self.connected = False  # a connection that has delivered a frame
        self.changed: asyncio.Event | None = None
        self.task: asyncio.Task | None = None

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def set(self, message: GateMessage, connected: bool) -> None:
        self.latest = message
        self.received = time.monotonic()
        self.connected = connected
        # Wake every waiter, then arm a fresh event for the next change.
        if self.changed is not None:
            self.changed.set()
            self.changed = asyncio.Event()

    def report(self, message: GateMessage, connected: bool) -> None:
        """:meth:`set`, from the current reader only: a superseded one says nothing."""
        if self.task is asyncio.current_task():
            self.set(message, connected)

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        task = self.task
        if task is not None and not task.done():
            if task.get_loop() is loop:
                return
            # A closed loop raises RuntimeError: its task died with it.
            with contextlib.suppress(RuntimeError):
                task.get_loop().call_soon_threadsafe(task.cancel)
        # Not safe until this loop's reader hears the server (no set(): the
        # old event belongs to the other loop).
        self.latest = GateMessage(False, "not connected to the shot gate")
        self.connected = False
        self.changed = asyncio.Event()
        self.task = loop.create_task(self._run(), name=f"shot-gate {self.address}")

    async def stop(self) -> None:
        task, self.task = self.task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.set(GateMessage(False, "not connected to the shot gate"), False)

    async def wait_change(self, timeout: float) -> None:
        if self.changed is None:  # never started: nothing will change
            await asyncio.sleep(timeout)
            return
        event = self.changed
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(timeout):
                await event.wait()

    async def _run(self) -> None:
        logged_down = False
        while True:
            try:
                async with asyncio.timeout(CONNECT_TIMEOUT_S):
                    reader, writer = await asyncio.open_connection(self.host, self.port)
            except (OSError, TimeoutError) as exc:
                if not logged_down:
                    logger.warning(
                        "shot gate %s unreachable (%s) — not safe; retrying",
                        self.address,
                        str(exc) or "connect timed out",
                    )
                    logged_down = True
                why = str(exc) or "connect timed out"
                self.report(GateMessage(False, f"shot gate unreachable: {why}"), False)
                await asyncio.sleep(RECONNECT_S)
                continue
            logger.info("shot gate %s connected", self.address)
            logged_down = False
            self.report(GateMessage(False, "connected, waiting for a frame"), False)
            reason = "the shot gate reader stopped"
            try:
                async for body in read_frames(reader):
                    message = self.parse(body)
                    if not message.safe and message.reason != self.latest.reason:
                        logger.info("shot gate: not safe — %s", message.reason)
                    self.report(message, True)
                reason = "the shot gate closed the connection"
            except TimeoutError:
                reason = f"no frame from the shot gate for {HEARTBEAT_S:g} s"
            except (OSError, ValueError, asyncio.IncompleteReadError) as exc:
                reason = f"shot gate connection lost: {exc}"
            finally:
                # Not safe before the teardown awaits: a fire must not pass
                # on a connection already known lost.
                self.report(GateMessage(False, reason), False)
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
            logger.warning("%s (%s) — not safe; reconnecting", reason, self.address)
            await asyncio.sleep(RECONNECT_S)


async def read_frames(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    """The framing: a 4-byte big-endian length, then the body.

    Ends at a clean close between frames; raises :class:`TimeoutError` when
    a frame does not arrive whole within :data:`HEARTBEAT_S`,
    :class:`ValueError` for a frame over :data:`MAX_FRAME`, and
    :class:`asyncio.IncompleteReadError` for a connection cut mid-frame.
    ``asyncio.timeout``, never ``wait_for``: on Python 3.11 ``wait_for`` can
    swallow a cancellation that lands as its read completes, and frames
    arrive every ~0.1 s, so a stopped reader would read on.
    """
    while True:
        async with asyncio.timeout(HEARTBEAT_S):
            try:
                header = await reader.readexactly(_LENGTH.size)
            except asyncio.IncompleteReadError as exc:
                if not exc.partial:
                    return
                raise
            (size,) = _LENGTH.unpack(header)
            if size > MAX_FRAME:
                raise ValueError(f"frame of {size} bytes (the limit is {MAX_FRAME})")
            body = await reader.readexactly(size)
        yield body


class ShotGate:
    """One profile's view of a shot gate: the shared reader, its own hold settings.

    Parameters
    ----------
    host, port :
        The server's address (the trigger profile's ``shot_gate``).
    hold_pause_s :
        Seconds a slot waits for permission before pausing the run (the
        operator's resume waits again).
    holdoff_s :
        Seconds added to the fire's time before a verdict counts as the
        next fire's permission (a device clock ahead of the worker's; one
        behind it cannot be covered, so the clocks must agree).
    parse :
        The frame's body → a verdict (:func:`parse_frame`).
    link :
        The reader to share (:func:`gate_for`); a new one by default.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        hold_pause_s: float = DEFAULT_HOLD_PAUSE_S,
        holdoff_s: float = 0.0,
        parse: Callable[[bytes], GateMessage] = parse_frame,
        link: _Link | None = None,
    ) -> None:
        if hold_pause_s <= 0:
            raise ValueError(f"hold_pause_s must be positive, got {hold_pause_s}")
        if holdoff_s < 0:
            raise ValueError(f"holdoff_s must not be negative, got {holdoff_s}")
        self.hold_pause_s = float(hold_pause_s)
        self.holdoff_s = float(holdoff_s)
        self._link = link if link is not None else _Link(host, port, parse)

    # ------------------------------------------------------------- state
    @property
    def address(self) -> str:
        """``host:port``, for messages."""
        return self._link.address

    @property
    def latest(self) -> GateMessage:
        """The server's latest verdict (a lost connection is a not-safe one)."""
        return self._link.latest

    @property
    def connected(self) -> bool:
        """Whether the reader holds a connection that has delivered a frame."""
        return self._link.connected

    @property
    def reason(self) -> str:
        """Why the gate does not permit a fire now (``""`` when it does)."""
        if self.permits():
            return ""
        latest = self._link.latest
        if not latest.safe:
            return latest.reason
        if latest.data_stamps:
            return "waiting for a verdict on data taken after the last shot"
        return "waiting for a safe frame after the last shot"

    def permits(self) -> bool:
        """Whether the latest verdict permits the next fire (one fresh verdict)."""
        link = self._link
        latest = link.latest
        if not (link.connected and latest.safe and link.received is not None):
            return False
        if link.fire_mono is None or link.fire_wall is None:
            return True
        if latest.data_stamps:
            return min(latest.data_stamps) > link.fire_wall + self.holdoff_s
        return link.received > link.fire_mono + self.holdoff_s

    def fired(self) -> None:
        """Record a fire: the next one needs a verdict fresher than now."""
        self._link.fire_mono = time.monotonic()
        self._link.fire_wall = time.time()

    def describe(self) -> dict[str, object]:
        """The start document's ``shot_gate`` entry."""
        return {
            "address": self.address,
            "hold_pause_s": self.hold_pause_s,
            "holdoff_s": self.holdoff_s,
        }

    # ------------------------------------------------------------- async
    async def start(self) -> None:
        """Start the reader on the calling loop (idempotent there).

        The reader outlives a RunEngine loop (one per address for the
        process, :func:`gate_for`), so one left on another loop — an
        environment reopened, a new RunEngine — is cancelled there and
        started again here.
        """
        await self._link.start()

    async def stop(self) -> None:
        """Stop the reader and drop the connection."""
        await self._link.stop()

    async def wait_connected(self, timeout: float) -> bool:
        """Wait up to *timeout* s for a connection and its first frame."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self._link.connected:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await self._link.wait_change(remaining)
        return True

    async def wait_safe(self, timeout: float) -> bool:
        """Wait up to *timeout* s until the gate permits a fire; whether it does."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self.permits():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await self._link.wait_change(remaining)
        return True


#: One reader per server address for the process: the connection and the
#: last-fire mark belong to the physical server, shared by every profile
#: that names it.
_LINKS: dict[tuple[str, int], _Link] = {}


def gate_for(spec: Any) -> ShotGate:
    """*spec*'s gate: the process's reader for its address, *spec*'s hold settings.

    *spec* is a ``ShotGateSpec``-shaped object (``key``, ``hold_pause_s``,
    ``holdoff_s``).  Each call is a new view, so one profile's settings
    never change another's.
    """
    host, port = spec.key
    link = _LINKS.get((host, port))
    if link is None:
        link = _LINKS[(host, port)] = _Link(host, port, parse_frame)
    return ShotGate(
        host,
        port,
        hold_pause_s=spec.hold_pause_s,
        holdoff_s=spec.holdoff_s,
        link=link,
    )
