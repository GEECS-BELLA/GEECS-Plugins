"""screen_rules.js under node: each row one rule of the screen's words (skipped without node)."""

from __future__ import annotations

from typing import Any

import pytest

from tests._node import call_all, needs_node

UP = {"value": 1.0, "connected": True, "severity": "NO_ALARM", "alarmStatus": 0}
VIEW = {"ok": True, "state": "off", "roles": {}}
ROLE = {
    "role": "current",
    "pv": "ps:I",
    "read": "monitor",
    "verdict": "stale",
    "writable": True,
}


def bare(**frame: Any) -> dict[str, Any]:
    return {"hareUp": True, "bare": True, "frame": UP | frame}


def role(**over: Any) -> dict[str, Any]:
    return {
        "hareUp": True,
        "bare": False,
        "frame": UP,
        "view": VIEW,
        "entry": ROLE,
    } | over


ROWS: list[tuple[str, str, list[Any], Any]] = [
    # judge: a bare PV, by Hare's connection and alarm
    ("bare: Hare down is unknown", "judge", [bare() | {"hareUp": False}], "unknown"),
    ("bare: no frame yet is unknown", "judge", [bare() | {"frame": None}], "unknown"),
    ("bare: disconnected is unreachable", "judge", [bare(connected=False)], "unreachable"),
    ("bare: INVALID+UDF is unknown", "judge", [bare(severity="INVALID", alarmStatus=17)], "unknown"),
    ("bare: INVALID+READ is not answering", "judge", [bare(severity="INVALID", alarmStatus=1)], "not answering"),
    ("bare: INVALID+COMM is not answering", "judge", [bare(severity="INVALID", alarmStatus=9)], "not answering"),
    ("bare: INVALID+TIMEOUT is not answering", "judge", [bare(severity="INVALID", alarmStatus=10)], "not answering"),
    ("bare: MAJOR is error", "judge", [bare(severity="MAJOR", alarmStatus=3)], "error"),
    ("bare: other INVALID is error", "judge", [bare(severity="INVALID", alarmStatus=2)], "error"),
    ("bare: MINOR is live", "judge", [bare(severity="MINOR")], "live"),
    # judge: a device role, by the module's verdict
    ("role: module silent is unknown", "judge", [role(view={"ok": False, "roles": {}})], "unknown"),
    ("role: no entry is unknown", "judge", [role(entry=None)], "unknown"),
    ("role: read never is the device's state", "judge", [role(entry=ROLE | {"read": "never"}, hareUp=False)], "off"),
    ("role: Hare down is unknown", "judge", [role(hareUp=False)], "unknown"),
    ("role: disconnected is unreachable", "judge", [role(frame=UP | {"connected": False})], "unreachable"),
    ("role: else the module's verdict", "judge", [role()], "stale"),
    # judge: a deferred widget (a DB device) and a role the module does not judge
    ("deferred: no such role is unknown role", "judge", [role(entry=None, fits=["setpoint"])], "unknown role"),
    ("deferred: a misfit is wrong kind", "judge", [role(entry=ROLE | {"type": "readback"}, fits=["bit", "enum"])], "wrong kind"),
    ("deferred: a fit is the module's verdict", "judge", [role(entry=ROLE | {"type": "readback"}, fits=["readback"])], "stale"),
    ("role: no verdict is the bare PV's rules", "judge", [role(entry=ROLE | {"verdict": None}, frame=UP | {"severity": "MAJOR"})], "error"),
    # show
    ("show: nothing is a dash", "show", [None, {}, None], "—"),
    ("show: a null value is a dash", "show", [{"value": None}, {}, None], "—"),
    ("show: an enum index is its label", "show", [{"value": 1}, {"enumChoices": ["Off", "On"]}, None], "On"),
    ("show: the widget's precision wins", "show", [{"value": 1.23456}, {"precision": 1}, "3"], "1.235"),
    ("show: else the record's precision", "show", [{"value": 1.23456}, {"precision": 1}, None], "1.2"),
    ("show: no precision is the number", "show", [{"value": 2.5}, {}, None], "2.5"),
    ("show: a string is itself", "show", [{"value": "idle"}, {}, None], "idle"),
    # bounds
    ("bounds: both set", "bounds", [{"controlLow": -1, "controlHigh": 5}], "[-1 … 5]"),
    ("bounds: ±1e300 is no limit", "bounds", [{"controlLow": -1e300, "controlHigh": 5}], ""),
    ("bounds: an empty range is none", "bounds", [{"controlLow": 0, "controlHigh": 0}], ""),
    # writable
    ("writable: the module's word", "writable", [ROLE, False, False], True),
    ("writable: never a bare PV", "writable", [ROLE, True, False], False),
    ("writable: never while busy", "writable", [ROLE, False, True], False),
    ("writable: not when the module says no", "writable", [ROLE | {"writable": False}, False, False], False),
    ("writable: never a deferred misfit", "writable", [ROLE | {"type": "readback"}, False, False, ["bit", "enum"]], False),
    # hazard: the screen's mark, or the module's (the policy's hazards)
    ("hazard: the module's mark asks", "hazard", [None, ROLE | {"confirm": "hazard"}], True),
    ("hazard: the screen's note asks", "hazard", ["Are you sure?", ROLE], True),
    ("hazard: else no confirm", "hazard", [None, ROLE], False),
    # differs
    ("differs: -0.000 equals 0.000", "differs", ["-0.000", "0.000", True], False),
    ("differs: two values disagree", "differs", ["1.000", "1.001", True], True),
    ("differs: not unless both are read", "differs", ["1.000", "2.000", False], False),
    # the outcome word as a kit state
    ("outcome: accepted is ok", "outcomeState", ["accepted"], "ok"),
    ("outcome: denied is degraded", "outcomeState", ["denied"], "degraded"),
    ("outcome: refused is failed", "outcomeState", ["refused"], "failed"),
    ("outcome: sending is running", "outcomeState", ["sending"], "running"),
    ("outcome: anything else is unknown", "outcomeState", ["unsent"], "unknown"),
    # the source line
    ("source: live", "sourceLine", ["open", []], {"text": "live values", "kit": "ok", "title": ""}),
    ("source: module slow", "sourceLine", ["open", ["a", "b"]],
     {"text": "live values · module slow", "kit": "degraded", "title": "no answer for: a, b"}),
    ("source: Hare down", "sourceLine", ["closed", []], {"text": "Hare closed", "kit": "unknown", "title": ""}),
    # the device route as the page keeps it
    ("view: roles keyed by role|axis", "deviceView", [True, {"state": "live", "roles": [{"role": "rbv", "axis": "m1"}]}],
     {"ok": True, "state": "live", "roles": {"rbv|m1": {"role": "rbv", "axis": "m1"}}}),
    ("view: a refusal is not ok", "deviceView", [False, {"code": "not_found"}], {"ok": False, "roles": {}}),
]  # fmt: skip


@pytest.fixture(scope="module")
def answers() -> list[Any]:
    return call_all("screen_rules.js", [(fn, args) for _, fn, args, _ in ROWS])


@needs_node
@pytest.mark.parametrize("i", range(len(ROWS)), ids=[r[0] for r in ROWS])
def test_rule(answers: list[Any], i: int) -> None:
    assert answers[i] == ROWS[i][3]
