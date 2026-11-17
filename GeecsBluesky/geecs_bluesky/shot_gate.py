"""ShotGate — a facility's permission-to-fire service, read before every fire.

A TCP service that checks hardware between shots and streams ``safe`` /
``not_safe <reason>`` one way.  Under a trigger profile naming one
(``TriggerProfile.shot_gate``) the worker waits for it before every
``SINGLESHOT``; a permission to fire, not a safety system.

- **One fresh ``safe`` per fire**: :meth:`ShotGate.permits` is true only
  for a ``safe`` received after :meth:`ShotGate.fired` (+ ``holdoff_s``) —
  the service may not yet have reported the last shot's cleanup.  Before
  the first fire the latest ``safe`` permits.  :func:`gate_for` keeps one
  client per address for the process, so the mark follows the physical
  service across runs and profiles.
- **Fail closed**: not connected, connection lost, a line not understood —
  not safe, the reason saying which; the reader reconnects by itself.
- **The wire format is** :func:`parse_message` (newline-framed lines), a
  placeholder until the facility's format is known: swap it (and
  :meth:`ShotGate._read_lines` if the framing differs), nothing else.

Standard library only: the plan stubs (:mod:`geecs_bluesky.plans.shot_gate`)
run the reader on the RunEngine's loop, and the web scanner runs a second,
read-only one for its chip.  The rules in full: ``GeecsBluesky/CLAUDE.md``
§ "The shot gate".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Seconds between reconnect attempts after the connection drops or is refused.
RECONNECT_S = 1.0
#: Seconds a run waits at its start for the first connection before refusing.
CONNECT_TIMEOUT_S = 5.0
#: Default seconds a slot holds for permission before it pauses the run.
DEFAULT_HOLD_PAUSE_S = 60.0
#: Longest line accepted; anything longer is a protocol error (reconnect).
MAX_LINE = 4096


@dataclass(frozen=True)
class GateMessage:
    """One verdict from the service: safe or not, and why not."""

    safe: bool
    reason: str = ""


def parse_message(line: str) -> GateMessage:
    """One line of the wire format → a verdict; anything not understood is not safe.

    The placeholder format: ``safe``, or ``not_safe`` with an optional
    reason after a space or a colon.  Case-insensitive on the keyword.
    """
    text = line.strip()
    keyword, _, rest = text.replace(":", " ", 1).partition(" ")
    keyword = keyword.lower()
    if keyword == "safe" and not rest.strip():
        return GateMessage(True)
    if keyword == "not_safe":
        return GateMessage(False, rest.strip() or "no reason given")
    return GateMessage(False, f"message not understood: {text!r}")


class ShotGate:
    """The client of one permission-to-fire service.

    Parameters
    ----------
    host, port :
        The service's address (the trigger profile's ``shot_gate``).
    hold_pause_s :
        Seconds a slot waits for permission before pausing the run (the
        operator's resume waits again).  Both hold settings are the
        profile's (:func:`gate_for` refreshes them per run).
    holdoff_s :
        Seconds after a fire before a ``safe`` counts as the next fire's
        permission (0: any ``safe`` received after the fire).
    parse :
        The wire format (:func:`parse_message`).
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        hold_pause_s: float = DEFAULT_HOLD_PAUSE_S,
        holdoff_s: float = 0.0,
        parse: Callable[[str], GateMessage] = parse_message,
    ) -> None:
        if hold_pause_s <= 0:
            raise ValueError(f"hold_pause_s must be positive, got {hold_pause_s}")
        if holdoff_s < 0:
            raise ValueError(f"holdoff_s must not be negative, got {holdoff_s}")
        self.host = host
        self.port = int(port)
        self.hold_pause_s = float(hold_pause_s)
        self.holdoff_s = float(holdoff_s)
        self._parse = parse
        self._latest = GateMessage(False, "not connected to the shot gate")
        self._received: float | None = None  # monotonic, of the latest message
        self._last_fire: float | None = None  # monotonic
        self._connected = False
        self._changed: asyncio.Event | None = None
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------- state
    @property
    def address(self) -> str:
        """``host:port``, for messages."""
        return f"{self.host}:{self.port}"

    @property
    def latest(self) -> GateMessage:
        """The service's latest verdict (a lost connection is a not-safe one)."""
        return self._latest

    @property
    def connected(self) -> bool:
        """Whether the reader holds a connection now."""
        return self._connected

    @property
    def reason(self) -> str:
        """Why the gate does not permit a fire now (``""`` when it does)."""
        if self.permits():
            return ""
        if self._latest.safe:
            return "waiting for a safe message after the last shot"
        return self._latest.reason

    def permits(self) -> bool:
        """Whether the latest message permits the next fire (one fresh ``safe``)."""
        if not (self._connected and self._latest.safe and self._received is not None):
            return False
        if self._last_fire is None:
            return True
        return self._received > self._last_fire + self.holdoff_s

    def fired(self) -> None:
        """Record a fire: the next one needs a ``safe`` received after now."""
        self._last_fire = time.monotonic()

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

        The client outlives a RunEngine loop (one per address for the
        process, :func:`gate_for`), so a reader left on another loop — an
        environment reopened, a new RunEngine — is cancelled there and
        started again here.
        """
        loop = asyncio.get_running_loop()
        task = self._task
        if task is not None and not task.done():
            if task.get_loop() is loop:
                return
            # A closed loop raises RuntimeError: its task died with it.
            with contextlib.suppress(RuntimeError):
                task.get_loop().call_soon_threadsafe(task.cancel)
        # Not safe until this loop's reader hears the service (no _set: the
        # old event belongs to the other loop).
        self._latest = GateMessage(False, "not connected to the shot gate")
        self._connected = False
        self._changed = asyncio.Event()
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name=f"shot-gate {self.address}"
        )

    async def stop(self) -> None:
        """Stop the reader and drop the connection."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._set(GateMessage(False, "not connected to the shot gate"), False)

    async def wait_connected(self, timeout: float) -> bool:
        """Wait up to *timeout* s for a connection; whether one is held."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self._connected:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await self._wait_change(remaining)
        return True

    async def wait_safe(self, timeout: float) -> bool:
        """Wait up to *timeout* s until the gate permits a fire; whether it does."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self.permits():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await self._wait_change(remaining)
        return True

    async def _wait_change(self, timeout: float) -> None:
        if self._changed is None:  # never started: nothing will change
            await asyncio.sleep(timeout)
            return
        event = self._changed
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(event.wait(), timeout)

    def _set(self, message: GateMessage, connected: bool) -> None:
        self._latest = message
        self._received = time.monotonic()
        self._connected = connected
        # Wake every waiter, then arm a fresh event for the next change.
        if self._changed is not None:
            self._changed.set()
            self._changed = asyncio.Event()

    async def _run(self) -> None:
        logged_down = False
        while True:
            try:
                reader, writer = await asyncio.open_connection(
                    self.host, self.port, limit=MAX_LINE
                )
            except OSError as exc:
                if not logged_down:
                    logger.warning(
                        "shot gate %s unreachable (%s) — not safe; retrying",
                        self.address,
                        exc,
                    )
                    logged_down = True
                self._set(GateMessage(False, f"shot gate unreachable: {exc}"), False)
                await asyncio.sleep(RECONNECT_S)
                continue
            logger.info("shot gate %s connected", self.address)
            logged_down = False
            self._set(GateMessage(False, "connected, no message yet"), True)
            try:
                async for line in self._read_lines(reader):
                    message = self._parse(line)
                    if not message.safe and message.reason != self._latest.reason:
                        logger.info("shot gate: not safe — %s", message.reason)
                    self._set(message, True)
                reason = "the shot gate closed the connection"
            except (OSError, ValueError, asyncio.LimitOverrunError) as exc:
                reason = f"shot gate connection lost: {exc}"
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
            logger.warning("%s (%s) — not safe; reconnecting", reason, self.address)
            self._set(GateMessage(False, reason), False)
            await asyncio.sleep(RECONNECT_S)

    @staticmethod
    async def _read_lines(reader: asyncio.StreamReader) -> AsyncIterator[str]:
        """The framing: newline-delimited UTF-8 lines, blank lines skipped."""
        while True:
            raw = await reader.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                yield line


#: One client per service address for the process: the reader and the
#: last-fire mark belong to the physical service, shared by every profile
#: that names it.
_GATES: dict[tuple[str, int], ShotGate] = {}


def gate_for(spec: Any) -> ShotGate:
    """The process's client for *spec*'s address, its hold settings refreshed.

    *spec* is a ``ShotGateSpec``-shaped object (``host``, ``port``,
    ``hold_pause_s``, ``holdoff_s``).  The hold settings are the profile's,
    taken at every call, so an edited profile applies at the next run.
    """
    key = (str(spec.host), int(spec.port))
    gate = _GATES.get(key)
    if gate is None:
        gate = _GATES[key] = ShotGate(key[0], key[1])
    gate.hold_pause_s = float(spec.hold_pause_s)
    gate.holdoff_s = float(spec.holdoff_s)
    return gate
