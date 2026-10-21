"""The interlock streams' wire format, import-light for the GUI clients.

Both streams the worker serves share one framing: a 4-byte big-endian
length, then that many bytes of UTF-8.

- **The status stream** (``[interlocks] status_port``, the one LabVIEW
  reads): a text message, the overall verdict first, then one part per
  interlock, joined with ``" | "`` (:mod:`.status_server` writes it;
  :func:`parse_status` reads it back).
- **The detail stream** (``[interlocks] detail_port``): one JSON object
  per frame, :data:`DETAIL_VERSION` in its ``"v"`` key (:mod:`.detail`
  writes it; the server window reads it).

Nothing here imports bluesky, ophyd-async or Qt.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

#: Seconds between two status frames to one client (the retired server's cadence).
SEND_INTERVAL_S = 0.1
#: Seconds between two detail frames to one client.
DETAIL_INTERVAL_S = 0.25
#: The detail document's ``"v"``.
DETAIL_VERSION = 1
#: A frame longer than this is a broken stream, not a message.
MAX_FRAME_BYTES = 16 * 1024 * 1024

#: The verdict that leads every status frame.
ALL_SAFE = "SAFE"
TRIPPED = "WARNING! TRIPPED"
NO_INTERLOCKS = "NO INTERLOCKS"

#: One interlock's state, as both streams and the windows name it.
STATE_OK = "OK"
STATE_TRIPPED = "TRIPPED"
STATE_NOT_ARMED = "NOT ARMED"
#: A client's own state when frames stop arriving (never sent by the worker).
STATE_NO_SIGNAL = "NO SIGNAL"

_LENGTH = struct.Struct(">I")


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


@dataclass(frozen=True)
class StatusPart:
    """One interlock's part of a status frame."""

    state: str
    name: str
    reason: str = ""


@dataclass(frozen=True)
class ParsedStatus:
    """A status frame read back: the verdict and the parts after it."""

    verdict: str
    parts: tuple[StatusPart, ...]
    #: True when the frame says no interlock is registered at all.
    no_interlocks: bool = False

    @property
    def safe(self) -> bool:
        """Whether the frame is an all-clear (the leading ``SAFE``)."""
        return self.verdict == ALL_SAFE


_PREFIXES = (
    ("NOT ARMED: ", STATE_NOT_ARMED),
    ("TRIPPED: ", STATE_TRIPPED),
    ("OK: ", STATE_OK),
)


def _parse_part(text: str) -> StatusPart:
    for prefix, state in _PREFIXES:
        if text.startswith(prefix):
            name, _, reason = text[len(prefix) :].partition(" - ")
            return StatusPart(state, name, reason)
    # A part this client does not know: never read it as safe.
    return StatusPart(STATE_TRIPPED, text, "unrecognised part")


def parse_status(message: str) -> ParsedStatus:
    """Read a status frame back into its verdict and parts.

    Anything but a leading ``SAFE`` is read as tripped, as the LabVIEW
    side reads it.  A part's reason is everything after its first
    ``" - "``, so a name holding ``" - "`` splits early (the name is
    still shown, the rest lands in the reason).
    """
    head, *rest = message.split(" | ")
    verdict = ALL_SAFE if head == ALL_SAFE else TRIPPED
    if rest == [NO_INTERLOCKS]:
        return ParsedStatus(verdict, (), no_interlocks=True)
    return ParsedStatus(verdict, tuple(_parse_part(part) for part in rest))


__all__ = [
    "ALL_SAFE",
    "DETAIL_INTERVAL_S",
    "DETAIL_VERSION",
    "MAX_FRAME_BYTES",
    "NO_INTERLOCKS",
    "SEND_INTERVAL_S",
    "STATE_NOT_ARMED",
    "STATE_NO_SIGNAL",
    "STATE_OK",
    "STATE_TRIPPED",
    "TRIPPED",
    "ParsedStatus",
    "StatusPart",
    "frame",
    "parse_status",
    "read_frame",
]
