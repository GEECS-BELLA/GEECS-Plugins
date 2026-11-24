"""The Master Control status stream's wire format: framing, the message, the parser.

A frame is a 4-byte big-endian length, then that many bytes of UTF-8 text,
every :data:`SEND_INTERVAL_S`: the overall verdict first, then one part per
interlock, joined with ``" | "``::

    SAFE | OK: <name> | OK: <name>                      every armed interlock OK
    WARNING! TRIPPED | TRIPPED: <name> - <reason> | OK: <name>
    WARNING! TRIPPED | OFF: <name>                      nothing armed
    WARNING! TRIPPED | NO INTERLOCKS                    no interlocks file

**The keyword rule** (pinned by the tests over every mix of states): a frame
holds exactly **one** keyword, its leading word: ``SAFE`` for an all-clear,
``WARNING! TRIPPED`` otherwise. Master Control fires a shot on a new frame
holding ``SAFE``, so the word appears nowhere else: inside a name or a
reason "safe" is sent as ``sa-fe``, "warning" as ``wa-rning``, and ``|`` as
``/``. (The original server sent ``SAFE: <name>`` per interlock, so a frame
could carry ``SAFE`` while another interlock was tripped; that format is
gone.)

Waiting (nothing received yet) and settling (clear again, holding for
``settle_s``) are sent as trips, the reason saying which: on this wire,
anything not OK is a trip.

This is not the JSON frame GeecsBluesky's ``ShotGate`` reads: that client
cannot use this server as its gate (yet).

Nothing here imports the web stack.
"""

from __future__ import annotations

import re
import socket
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from geecs_interlocks.monitor import InterlockStatus, Snapshot

#: Seconds between two frames to one client (the original server's cadence).
SEND_INTERVAL_S = 0.1
#: A frame longer than this is a broken stream, not a message.
MAX_FRAME_BYTES = 16 * 1024 * 1024

#: The verdict that leads every frame.
ALL_SAFE = "SAFE"
TRIPPED = "WARNING! TRIPPED"
NO_INTERLOCKS = "NO INTERLOCKS"

#: One interlock's state, as the wire names it.
PART_OK = "OK"
PART_TRIPPED = "TRIPPED"
PART_OFF = "OFF"

#: The one part of a frame sent while the evaluation has stopped (no snapshot for a while).
STALLED_NAME = "interlock server"

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


def render_status(snapshot: Snapshot) -> str:
    """The message one frame carries: the overall verdict, then one part per interlock."""
    if not snapshot.statuses:
        return f"{TRIPPED} | {NO_INTERLOCKS}"
    verdict = ALL_SAFE if snapshot.safe else TRIPPED
    return " | ".join([verdict, *(_part(s) for s in snapshot.statuses)])


def render_stalled(age_s: float) -> str:
    """The frame sent while no snapshot has replaced the last for *age_s*: a trip, whatever it said."""
    return f"{TRIPPED} | {PART_TRIPPED}: {STALLED_NAME} - evaluation stopped {age_s:.0f} s ago"


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
    #: True when the frame says there is no interlocks file at all.
    no_interlocks: bool = False

    @property
    def safe(self) -> bool:
        """Whether the frame is an all-clear (the leading ``SAFE``)."""
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


def parse_status(message: str) -> ParsedStatus:
    """Read a frame back into its verdict and parts.

    Anything but a leading ``SAFE`` is read as tripped. A part's reason is
    everything after its first ``" - "``, so a name holding ``" - "`` splits
    early (the rest lands in the reason).
    """
    head, *rest = message.split(" | ")
    verdict = ALL_SAFE if head == ALL_SAFE else TRIPPED
    if rest == [NO_INTERLOCKS]:
        return ParsedStatus(verdict, (), no_interlocks=True)
    return ParsedStatus(verdict, tuple(_parse_part(part) for part in rest))


__all__ = [
    "ALL_SAFE",
    "MAX_FRAME_BYTES",
    "NO_INTERLOCKS",
    "PART_OFF",
    "PART_OK",
    "PART_TRIPPED",
    "SEND_INTERVAL_S",
    "STALLED_NAME",
    "TRIPPED",
    "ParsedStatus",
    "StatusPart",
    "clean",
    "frame",
    "parse_status",
    "read_frame",
    "render_stalled",
    "render_status",
]
