"""The Master Control status stream's wire format: framing, two messages, the parser.

A frame is a 4-byte big-endian length, then that many bytes of UTF-8 text,
every :data:`SEND_INTERVAL_S`. The text comes in one of two formats
(:data:`WireFormat`), chosen per server:

``summary`` (the default) — the overall verdict first, then one part per
interlock::

    SAFE | OK: <name> | OK: <name>
    WARNING! TRIPPED | TRIPPED: <name> - <reason> | OK: <name>
    WARNING! TRIPPED | OFF: <name>                      nothing armed
    WARNING! TRIPPED | NO INTERLOCKS                    no interlocks file

``legacy`` — the original server's: one part per armed interlock, joined
with ``" | "``::

    SAFE: <name> | SAFE: <name>                         every armed interlock OK
    WARNING! <name> - <reason> | SAFE: <name>           a trip
    WARNING! No monitors registered                     nothing armed, or no file

**Master Control reads a frame only when it holds a single keyword** — one
``SAFE`` or one ``WARNING``, in any case. ``summary`` always does (pinned by
the tests over every mix of states): its one keyword is the leading word,
``SAFE`` for an all-clear and ``WARNING! TRIPPED`` otherwise. ``legacy``
repeats a keyword per interlock, so Master Control can read it only while a
single interlock is armed; it stays for that case and for other readers.
In both formats a frame holds ``WARNING`` exactly when it is not an
all-clear. Inside a name or a
reason "safe" is sent as ``sa-fe``, "warning" as ``warn-ing``, and ``|`` as
``/``.

Waiting (nothing received yet) and settling (clear again, holding for
``settle_s``) are sent as trips, the reason saying which: on this wire,
anything not OK is a trip. ``legacy`` leaves ``off`` interlocks out, as the
original server never listed them.

Nothing here imports Qt or the web stack.
"""

from __future__ import annotations

import re
import socket
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from geecs_interlocks.monitor import InterlockStatus, Snapshot

#: Seconds between two frames to one client (the original server's cadence).
SEND_INTERVAL_S = 0.1
#: A frame longer than this is a broken stream, not a message.
MAX_FRAME_BYTES = 16 * 1024 * 1024

#: The two message formats; only ``summary`` keeps Master Control's one-keyword rule.
WireFormat = Literal["legacy", "summary"]
WIRE_FORMATS: tuple[WireFormat, ...] = get_args(WireFormat)
DEFAULT_WIRE_FORMAT: WireFormat = "summary"

#: ``summary``: the verdict that leads every frame.
ALL_SAFE = "SAFE"
TRIPPED = "WARNING! TRIPPED"
NO_INTERLOCKS = "NO INTERLOCKS"

#: One interlock's state, as the wire names it.
PART_OK = "OK"
PART_TRIPPED = "TRIPPED"
PART_OFF = "OFF"

#: ``legacy``: one part's prefixes, and the frame of nothing armed.
LEGACY_SAFE = "SAFE: "
LEGACY_WARNING = "WARNING! "
LEGACY_NOTHING = "WARNING! No monitors registered"

_LENGTH = struct.Struct(">I")
_KEYWORDS = re.compile("safe|warn", re.IGNORECASE)


def frame(message: str) -> bytes:
    """*message* with its 4-byte big-endian length prefix."""
    payload = message.encode("utf-8")
    return _LENGTH.pack(len(payload)) + payload


def _recv_exactly(sock: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = sock.recv(min(size, 65536))
        if not chunk:
            raise ConnectionError("the server closed the stream")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> str:
    """Block until one whole frame has arrived on *sock*; its message.

    Raises
    ------
    ConnectionError
        The server closed the stream, or sent a length past
        :data:`MAX_FRAME_BYTES`.
    TimeoutError
        *sock*'s timeout passed with the frame incomplete.
    """
    (size,) = _LENGTH.unpack(_recv_exactly(sock, _LENGTH.size))
    if size > MAX_FRAME_BYTES:
        raise ConnectionError(f"frame of {size} bytes: not an interlock stream")
    return _recv_exactly(sock, size).decode("utf-8", errors="replace")


def clean(text: str) -> str:
    """*text* fit for a part: no separator, no keyword ("Safety" → "Sa-fety", "warning" → "wa-rning")."""
    text = text.replace("|", "/")
    return _KEYWORDS.sub(lambda m: f"{m.group(0)[:2]}-{m.group(0)[2:]}", text)


def _part(status: InterlockStatus) -> str:
    name = clean(status.name)
    if not status.armed:
        return f"{PART_OFF}: {name}"
    if not status.unsafe:
        return f"{PART_OK}: {name}"
    reason = status.diagnostic or status.state
    return f"{PART_TRIPPED}: {name} - {clean(reason)}"


def render_summary(snapshot: Snapshot) -> str:
    """The ``summary`` message: the overall verdict, then one part per interlock."""
    if not snapshot.statuses:
        return f"{TRIPPED} | {NO_INTERLOCKS}"
    verdict = ALL_SAFE if snapshot.safe else TRIPPED
    return " | ".join([verdict, *(_part(s) for s in snapshot.statuses)])


def _legacy_part(status: InterlockStatus) -> str:
    name = clean(status.name)
    if not status.unsafe:
        return f"{LEGACY_SAFE}{name}"
    reason = status.diagnostic or status.state
    return f"{LEGACY_WARNING}{name} - {clean(reason)}"


def render_legacy(snapshot: Snapshot) -> str:
    """The ``legacy`` message: one part per armed interlock (the original server's)."""
    armed = [s for s in snapshot.statuses if s.armed]
    if not armed:
        return LEGACY_NOTHING
    return " | ".join(_legacy_part(s) for s in armed)


def render_status(
    snapshot: Snapshot, wire_format: WireFormat = DEFAULT_WIRE_FORMAT
) -> str:
    """The message one frame carries, in *wire_format*."""
    if wire_format == "summary":
        return render_summary(snapshot)
    if wire_format == "legacy":
        return render_legacy(snapshot)
    raise ValueError(f"unknown wire format {wire_format!r} (one of {WIRE_FORMATS})")


@dataclass(frozen=True)
class StatusPart:
    """One interlock's part of a frame, read back."""

    state: str
    name: str
    reason: str = ""


@dataclass(frozen=True)
class ParsedStatus:
    """A frame read back: the verdict and the parts after it."""

    verdict: str
    parts: tuple[StatusPart, ...]
    #: True when the frame says nothing is watched (no file, or nothing armed in ``legacy``).
    no_interlocks: bool = False
    #: The format the frame was read as.
    wire_format: WireFormat = "summary"

    @property
    def safe(self) -> bool:
        """Whether the frame is an all-clear."""
        return self.verdict == ALL_SAFE


_PREFIXES: Sequence[tuple[str, str]] = tuple(
    (f"{state}: ", state) for state in (PART_TRIPPED, PART_OFF, PART_OK)
)


def _parse_part(text: str) -> StatusPart:
    for prefix, state in _PREFIXES:
        if text.startswith(prefix):
            name, _, reason = text[len(prefix) :].partition(" - ")
            return StatusPart(state, name, reason)
    # A part this reader does not know: never read it as safe.
    return StatusPart(PART_TRIPPED, text, "unrecognised part")


def _parse_legacy(message: str) -> ParsedStatus:
    if message == LEGACY_NOTHING:
        return ParsedStatus(TRIPPED, (), no_interlocks=True, wire_format="legacy")
    parts = []
    for text in message.split(" | "):
        if text.startswith(LEGACY_SAFE):
            parts.append(StatusPart(PART_OK, text[len(LEGACY_SAFE) :]))
        elif text.startswith(LEGACY_WARNING):
            name, _, reason = text[len(LEGACY_WARNING) :].partition(" - ")
            parts.append(StatusPart(PART_TRIPPED, name, reason))
        else:
            parts.append(StatusPart(PART_TRIPPED, text, "unrecognised part"))
    safe = bool(parts) and all(p.state == PART_OK for p in parts)
    return ParsedStatus(
        ALL_SAFE if safe else TRIPPED, tuple(parts), wire_format="legacy"
    )


def parse_status(message: str) -> ParsedStatus:
    """Read a frame of either format back into its verdict and parts.

    A frame leading with exactly ``SAFE`` or ``WARNING! TRIPPED`` is
    ``summary``; anything else is read as ``legacy``. Anything that is not
    clearly an all-clear is read as tripped, as Master Control reads it. A
    part's reason is everything after its first ``" - "``, so a name holding
    ``" - "`` splits early (the rest lands in the reason).
    """
    head, *rest = message.split(" | ")
    if head not in (ALL_SAFE, TRIPPED):
        return _parse_legacy(message)
    verdict = ALL_SAFE if head == ALL_SAFE else TRIPPED
    if rest == [NO_INTERLOCKS]:
        return ParsedStatus(verdict, (), no_interlocks=True)
    return ParsedStatus(verdict, tuple(_parse_part(part) for part in rest))


__all__ = [
    "ALL_SAFE",
    "DEFAULT_WIRE_FORMAT",
    "LEGACY_NOTHING",
    "WIRE_FORMATS",
    "WireFormat",
    "MAX_FRAME_BYTES",
    "NO_INTERLOCKS",
    "PART_OFF",
    "PART_OK",
    "PART_TRIPPED",
    "SEND_INTERVAL_S",
    "TRIPPED",
    "ParsedStatus",
    "StatusPart",
    "clean",
    "frame",
    "parse_status",
    "read_frame",
    "render_legacy",
    "render_status",
    "render_summary",
]
