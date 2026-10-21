"""The interlock streams' wire format, read back as the GUI clients read it."""

from __future__ import annotations

import socket
import struct

import pytest
from geecs_bluesky.interlocks.install import InterlockStatus
from geecs_bluesky.interlocks.status_server import render_status
from geecs_bluesky.interlocks.wire import (
    ALL_SAFE,
    MAX_FRAME_BYTES,
    STATE_NOT_ARMED,
    STATE_OK,
    STATE_TRIPPED,
    TRIPPED,
    StatusPart,
    frame,
    parse_status,
    read_frame,
)


def test_an_all_clear_frame_reads_back_safe() -> None:
    parsed = parse_status(
        render_status([InterlockStatus("cam", False), InterlockStatus("vac", False)])
    )
    assert parsed.safe and parsed.verdict == ALL_SAFE
    assert parsed.parts == (StatusPart(STATE_OK, "cam"), StatusPart(STATE_OK, "vac"))


def test_every_state_and_reason_reads_back() -> None:
    message = render_status(
        [
            InterlockStatus("vacuum", True, "Pressure=0.001 (limit: >1e-05)"),
            InterlockStatus("cam", False),
            InterlockStatus("gauge", True),
            InterlockStatus(
                "stamp", True, "not armed: not connected: U_Gauge", armed=False
            ),
        ]
    )
    parsed = parse_status(message)
    assert not parsed.safe and parsed.verdict == TRIPPED
    assert parsed.parts == (
        StatusPart(STATE_TRIPPED, "vacuum", "Pressure=0.001 (limit: >1e-05)"),
        StatusPart(STATE_OK, "cam"),
        StatusPart(STATE_TRIPPED, "gauge"),
        StatusPart(STATE_NOT_ARMED, "stamp", "not connected: U_Gauge"),
    )


def test_no_interlocks_is_not_safe() -> None:
    parsed = parse_status(render_status([]))
    assert not parsed.safe and parsed.no_interlocks and parsed.parts == ()


@pytest.mark.parametrize("message", ["", "safe | OK: a", "SAFE: a", "garbage"])
def test_anything_but_a_leading_SAFE_is_tripped(message: str) -> None:
    assert not parse_status(message).safe


def test_an_unknown_part_never_reads_as_ok() -> None:
    (part,) = parse_status("WARNING! TRIPPED | SOMETHING NEW: x").parts
    assert part.state == STATE_TRIPPED


def test_read_frame_reads_what_frame_writes() -> None:
    a, b = socket.socketpair()
    with a, b:
        a.sendall(frame("SAFE | OK: µ") + frame("second"))
        assert read_frame(b) == "SAFE | OK: µ"
        assert read_frame(b) == "second"


def test_read_frame_refuses_a_closed_or_absurd_stream() -> None:
    a, b = socket.socketpair()
    with b:
        a.sendall(struct.pack(">I", 10) + b"short")
        a.close()
        with pytest.raises(ConnectionError):
            read_frame(b)
    a, b = socket.socketpair()
    with a, b:
        a.sendall(struct.pack(">I", MAX_FRAME_BYTES + 1))
        with pytest.raises(ConnectionError):
            read_frame(b)
