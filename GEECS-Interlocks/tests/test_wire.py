"""The Master Control frame: the one-keyword rule, the parts, the parser, the framing."""

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
    render_stalled,
    render_status,
)


def status(name, state, diagnostic=""):
    return InterlockStatus(name, "", "", state, diagnostic)


def snap(*statuses):
    return Snapshot(tuple(statuses))


def keywords(message):
    return re.findall("safe|warning", message, re.IGNORECASE)


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

    def test_a_trip_without_a_reason_names_its_state(self):
        assert render_status(snap(status("a", "tripped"))).endswith(
            "TRIPPED: a - tripped"
        )


NAMES = ["Safety shutter", "unsafe | warning gauge", "plain"]
MIXES = list(itertools.product(STATES, repeat=2))


def mixed(mix):
    return snap(
        *(
            status(name, state, "SAFE? unsafe! WARNING")
            for name, state in zip(NAMES, mix)
        )
    )


class TestKeywordRule:
    """Master Control fires on a new frame holding SAFE: exactly one keyword, the leading one."""

    @pytest.mark.parametrize("mix", MIXES)
    def test_exactly_one_keyword(self, mix):
        snapshot = mixed(mix)
        message = render_status(snapshot)
        assert keywords(message) == (["SAFE"] if snapshot.safe else ["WARNING"])

    @pytest.mark.parametrize("mix", MIXES)
    def test_safe_leads_only_an_all_clear(self, mix):
        snapshot = mixed(mix)
        message = render_status(snapshot)
        hits = [m.start() for m in re.finditer("safe", message, re.IGNORECASE)]
        assert hits == ([0] if snapshot.safe else [])
        assert message.count(" | ") == len(snapshot.statuses)

    @pytest.mark.parametrize("statuses", [(), (status("a", "off"),)])
    def test_one_keyword_when_nothing_is_watched(self, statuses):
        assert keywords(render_status(snap(*statuses))) == ["WARNING"]

    def test_a_stalled_frame_is_one_warning(self):
        assert keywords(render_stalled(12)) == ["WARNING"]

    def test_clean(self):
        assert clean("Safety | unSAFE warning") == "Sa-fety / unSA-FE wa-rning"

    def test_the_servers_own_reasons_need_no_cleaning(self):
        """Reasons the server writes itself never hold a keyword (only names and packets do)."""
        from geecs_schemas import ExternalCondition

        from geecs_interlocks.conditions import Inputs, evaluate
        from geecs_interlocks.external import SignalReading

        condition = ExternalCondition(kind="external", port=1, source="stage")
        signal = SignalReading(True, 0.0, "x")
        alarm = evaluate(condition, Inputs(signal=signal), tripped=False)
        for reason in (alarm.diagnostic, "clear again; OK in 2.0 s"):
            assert clean(reason) == reason


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
        assert not parse_status("SAFE: a | SAFE: b").safe  # the old per-interlock form
        assert not parse_status("").safe


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
