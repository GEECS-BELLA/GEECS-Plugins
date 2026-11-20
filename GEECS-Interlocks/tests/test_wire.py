"""The Master Control frame in both formats: the keyword rules, the parts, the parser, the framing."""

import itertools
import re
import socket

import pytest

from geecs_interlocks.monitor import STATES, InterlockStatus, Snapshot
from geecs_interlocks.wire import (
    ALL_SAFE,
    DEFAULT_WIRE_FORMAT,
    LEGACY_NOTHING,
    NO_INTERLOCKS,
    WIRE_FORMATS,
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


def summary(snapshot):
    return render_status(snapshot, "summary")


def legacy(snapshot):
    return render_status(snapshot, "legacy")


def test_summary_is_the_default():
    assert DEFAULT_WIRE_FORMAT == "summary"
    assert render_status(snap(status("a", "ok"))) == "SAFE | OK: a"


class TestSummary:
    def test_all_ok(self):
        assert summary(snap(status("a", "ok"), status("b", "off"))) == (
            "SAFE | OK: a | OFF: b"
        )

    def test_a_trip(self):
        message = summary(snap(status("a", "tripped", "P=5"), status("b", "ok")))
        assert message == "WARNING! TRIPPED | TRIPPED: a - P=5 | OK: b"

    @pytest.mark.parametrize("state", ["waiting", "settling"])
    def test_waiting_and_settling_are_trips(self, state):
        message = summary(snap(status("a", state, "why")))
        assert message == "WARNING! TRIPPED | TRIPPED: a - why"

    def test_no_interlocks(self):
        assert summary(snap()) == f"WARNING! TRIPPED | {NO_INTERLOCKS}"

    def test_nothing_armed(self):
        assert summary(snap(status("a", "off"))).startswith("WARNING! TRIPPED")


class TestLegacy:
    """The original server's format, which Master Control reads today."""

    def test_all_ok(self):
        assert legacy(snap(status("a", "ok"), status("b", "ok"))) == "SAFE: a | SAFE: b"

    def test_a_trip(self):
        message = legacy(snap(status("a", "tripped", "P=5"), status("b", "ok")))
        assert message == "WARNING! a - P=5 | SAFE: b"

    @pytest.mark.parametrize("state", ["waiting", "settling"])
    def test_waiting_and_settling_are_trips(self, state):
        assert legacy(snap(status("a", state, "why"))) == "WARNING! a - why"

    def test_a_trip_without_a_reason_names_its_state(self):
        assert legacy(snap(status("a", "tripped"))) == "WARNING! a - tripped"

    def test_off_is_left_out(self):
        assert legacy(snap(status("a", "ok"), status("b", "off"))) == "SAFE: a"

    @pytest.mark.parametrize("statuses", [(), (status("a", "off"),)])
    def test_nothing_armed(self, statuses):
        assert legacy(snap(*statuses)) == LEGACY_NOTHING


NAMES = ["Safety shutter", "unsafe | warning gauge", "plain"]
MIXES = list(itertools.product(STATES, repeat=2))


def mixed(mix):
    return snap(
        *(
            status(name, state, "SAFE? unsafe! WARNING")
            for name, state in zip(NAMES, mix)
        )
    )


class TestKeywordRules:
    """Whatever the mix of states, names and reasons, a keyword search reads the frame right."""

    @pytest.mark.parametrize("wire_format", WIRE_FORMATS)
    @pytest.mark.parametrize("mix", MIXES)
    def test_warning_appears_exactly_when_unsafe(self, wire_format, mix):
        snapshot = mixed(mix)
        message = render_status(snapshot, wire_format)
        assert ("WARNING" in message.upper()) is (not snapshot.safe)

    @pytest.mark.parametrize("mix", MIXES)
    def test_summary_holds_exactly_one_keyword(self, mix):
        """Master Control reads a frame only when it holds a single SAFE or WARNING."""
        snapshot = mixed(mix)
        message = summary(snapshot)
        keywords = re.findall("safe|warning", message, re.IGNORECASE)
        assert keywords == (["SAFE"] if snapshot.safe else ["WARNING"])

    @pytest.mark.parametrize("statuses", [(), (status("a", "off"),)])
    def test_summary_one_keyword_when_nothing_is_watched(self, statuses):
        assert re.findall("safe|warning", summary(snap(*statuses)), re.IGNORECASE) == [
            "WARNING"
        ]

    def test_legacy_repeats_the_keyword(self):
        """Why summary is the default: two armed interlocks, two keywords."""
        message = legacy(snap(status("a", "ok"), status("b", "ok")))
        assert len(re.findall("safe", message, re.IGNORECASE)) == 2

    @pytest.mark.parametrize("mix", MIXES)
    def test_summary_safe_only_leading_an_all_clear(self, mix):
        snapshot = mixed(mix)
        message = summary(snapshot)
        hits = [m.start() for m in re.finditer("safe", message, re.IGNORECASE)]
        assert hits == ([0] if snapshot.safe else [])
        assert message.count(" | ") == len(snapshot.statuses)

    @pytest.mark.parametrize("mix", MIXES)
    def test_legacy_safe_only_as_a_part_prefix(self, mix):
        message = legacy(mixed(mix))
        hits = {m.start() for m in re.finditer("safe", message, re.IGNORECASE)}
        starts = {0} | {m.end() for m in re.finditer(r" \| ", message)}
        assert hits <= starts

    def test_clean(self):
        assert clean("Safety | unSAFE warning") == "Sa-fety / unSA-FE wa-rning"


class TestParse:
    def test_summary_round_trip(self):
        parsed = parse_status(
            summary(
                snap(
                    status("a", "tripped", "P=5"), status("b", "ok"), status("c", "off")
                )
            )
        )
        assert not parsed.safe and parsed.wire_format == "summary"
        assert [(p.state, p.name, p.reason) for p in parsed.parts] == [
            ("TRIPPED", "a", "P=5"),
            ("OK", "b", ""),
            ("OFF", "c", ""),
        ]

    def test_legacy_round_trip(self):
        parsed = parse_status(
            legacy(snap(status("a", "tripped", "P=5"), status("b", "ok")))
        )
        assert not parsed.safe and parsed.wire_format == "legacy"
        assert [(p.state, p.name, p.reason) for p in parsed.parts] == [
            ("TRIPPED", "a", "P=5"),
            ("OK", "b", ""),
        ]

    def test_legacy_all_clear(self):
        parsed = parse_status("SAFE: a | SAFE: b")
        assert parsed.safe and parsed.wire_format == "legacy"

    def test_legacy_nothing_armed(self):
        parsed = parse_status(LEGACY_NOTHING)
        assert parsed.no_interlocks and not parsed.safe

    def test_unknown_parts_read_as_tripped(self):
        assert parse_status("SAFE | WHAT: x").parts[0].state == "TRIPPED"
        assert not parse_status("SAFE: a | WHAT: x").safe

    def test_no_interlocks(self):
        assert parse_status(f"WARNING! TRIPPED | {NO_INTERLOCKS}").no_interlocks

    def test_only_a_clear_frame_is_safe(self):
        assert parse_status(ALL_SAFE).safe
        assert not parse_status("SAFE-ISH | OK: a").safe
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


def test_the_servers_own_reasons_need_no_cleaning():
    """Reasons the server writes itself never hold a keyword (only names and packets do)."""
    from geecs_interlocks.conditions import Inputs, evaluate
    from geecs_interlocks.external import SignalReading
    from geecs_schemas import ExternalCondition

    condition = ExternalCondition(kind="external", port=1, source="stage")
    alarm = evaluate(
        condition, Inputs(signal=SignalReading(True, 0.0, "x")), tripped=False
    )
    for reason in (alarm.diagnostic, "clear again; OK in 2.0 s"):
        assert clean(reason) == reason
