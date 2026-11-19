"""The Master Control frame: the keyword rule, the parts, the parser, the framing."""

import itertools
import re
import socket

import pytest

from geecs_interlocks.monitor import STATES, InterlockStatus, Snapshot
from geecs_interlocks.wire import (
    ALL_SAFE,
    NO_INTERLOCKS,
    clean,
    frame,
    parse_status,
    read_frame,
    render_status,
)


def status(name, state, diagnostic=""):
    return InterlockStatus(name, "", "", state, diagnostic)


def snap(*statuses):
    return Snapshot(tuple(statuses))


class TestRender:
    def test_all_ok(self):
        assert render_status(snap(status("a", "ok"), status("b", "off"))) == (
            "SAFE | OK: a | OFF: b"
        )

    def test_a_trip(self):
        message = render_status(snap(status("a", "tripped", "P=5"), status("b", "ok")))
        assert message == "WARNING! TRIPPED | TRIPPED: a - P=5 | OK: b"

    @pytest.mark.parametrize("state", ["waiting", "settling"])
    def test_waiting_and_settling_are_trips(self, state):
        message = render_status(snap(status("a", state, "why")))
        assert message == "WARNING! TRIPPED | TRIPPED: a - why"

    def test_no_interlocks(self):
        assert render_status(snap()) == f"WARNING! TRIPPED | {NO_INTERLOCKS}"

    def test_nothing_armed(self):
        assert render_status(snap(status("a", "off"))).startswith("WARNING! TRIPPED")


class TestKeywordRule:
    """'safe' appears only as the leading SAFE of an all-clear, whatever the mix."""

    NAMES = ["Safety shutter", "unsafe | gauge", "plain"]

    @pytest.mark.parametrize("mix", list(itertools.product(STATES, repeat=2)))
    def test_every_mix(self, mix):
        statuses = [
            status(name, state, "SAFE? unsafe!") for name, state in zip(self.NAMES, mix)
        ]
        message = render_status(snap(*statuses))
        hits = [m.start() for m in re.finditer("safe", message, re.IGNORECASE)]
        if snap(*statuses).safe:
            assert hits == [0]
        else:
            assert hits == []
        assert message.count(" | ") == len(statuses)

    def test_clean(self):
        assert clean("Safety | unSAFE") == "Sa-fety / unSA-FE"


class TestParse:
    def test_round_trip(self):
        parsed = parse_status(
            render_status(
                snap(
                    status("a", "tripped", "P=5"), status("b", "ok"), status("c", "off")
                )
            )
        )
        assert not parsed.safe
        assert [(p.state, p.name, p.reason) for p in parsed.parts] == [
            ("TRIPPED", "a", "P=5"),
            ("OK", "b", ""),
            ("OFF", "c", ""),
        ]

    def test_unknown_parts_read_as_tripped(self):
        assert parse_status("SAFE | WHAT: x").parts[0].state == "TRIPPED"

    def test_no_interlocks(self):
        assert parse_status(f"WARNING! TRIPPED | {NO_INTERLOCKS}").no_interlocks

    def test_only_a_leading_safe_is_safe(self):
        assert parse_status(ALL_SAFE).safe
        assert not parse_status("SAFE-ISH | OK: a").safe


class TestFraming:
    def test_round_trip_over_a_socket(self):
        a, b = socket.socketpair()
        with a, b:
            a.sendall(frame("héllo") + frame("two"))
            assert read_frame(b) == "héllo"
            assert read_frame(b) == "two"

    def test_closed_stream(self):
        a, b = socket.socketpair()
        with b:
            a.close()
            with pytest.raises(ConnectionError):
                read_frame(b)

    def test_length_prefix(self):
        assert frame("ab")[:4] == b"\x00\x00\x00\x02"
