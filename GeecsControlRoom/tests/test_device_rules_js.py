"""device_rules.js under node: each row one rule of the device pages' words (skipped without node)."""

from __future__ import annotations

from typing import Any

import pytest

from tests._node import call_all, needs_node

UP = {"value": 1.0, "connected": True, "severity": "NO_ALARM", "alarmStatus": 0}
ENTRY = {
    "role": "exposure",
    "pv": "cam:AcquireTime_RBV",
    "read": "monitor",
    "verdict": "live",
}
VIEW = {"ok": True, "state": "live", "roles": {}}
ROWS_IN = [{"name": "b", "state": "live"}, {"name": "a"}]
CH = ["ch_a_enable", "ch_a_range", "ch_a_coupling", "ch_a_offset"]
KNOWN = CH + ["ch_a_present", "acquire"]
AXES = {"ok": True, "roles": {
    "hls|m1": {"role": "hls", "axis": "m1", "label": "high limit switch", "verdict": "error"},
    "hls|m2": {"role": "hls", "axis": "m2", "label": "high limit switch", "verdict": "live"},
    "fault|": {"role": "fault", "axis": None, "label": "fault", "verdict": "error"},
}}  # fmt: skip


def row(**over: Any) -> dict[str, Any]:
    return {"hareUp": True, "frame": UP, "view": VIEW, "entry": ENTRY} | over


ROWS: list[tuple[str, str, list[Any], Any]] = [
    # the picker's options: by name, the roster's word, the current one selected
    ("picker: by name, unknown for no word", "pickerOptions", [ROWS_IN, "", "camera"],
     [{"value": "", "text": "choose a camera…", "selected": True},
      {"value": "a", "text": "a · unknown", "selected": False},
      {"value": "b", "text": "b · live", "selected": False}]),
    ("picker: the current device is selected", "pickerOptions", [ROWS_IN, "b", "motor"],
     [{"value": "", "text": "choose a motor…", "selected": False},
      {"value": "a", "text": "a · unknown", "selected": False},
      {"value": "b", "text": "b · live", "selected": True}]),
    ("picker: an empty registry says so", "pickerOptions", [[], "", "scope"],
     [{"value": "", "text": "no scope in the registry", "selected": True}]),
    ("picker: a typed name the registry lacks stays chosen", "pickerOptions", [[{"name": "a", "state": "live"}], "z", "camera"],
     [{"value": "", "text": "choose a camera…", "selected": False},
      {"value": "a", "text": "a · live", "selected": False},
      {"value": "z", "text": "z · typed", "selected": True}]),
    ("recovered: the device answered", "recovered", [200], True),
    ("recovered: the roster answered without it", "recovered", [404], True),
    ("recovered: a refusal keeps asking, never reloads", "recovered", [401], False),
    ("recovered: still silent", "recovered", [502], False),
    ("picker note: the listing's detail", "pickerNote", [{"detail": "the module did not answer"}], "the module did not answer"),
    ("picker note: none", "pickerNote", [{"detail": None}], ""),
    # which rows Hare reads
    ("watched: a monitored role", "watched", [ENTRY], True),
    ("watched: read once (the frame URL)", "watched", [ENTRY | {"read": "once"}], True),
    ("watched: never an array on request", "watched", [ENTRY | {"read": "on_request"}], False),
    ("watched: never a command", "watched", [ENTRY | {"read": "never"}], False),
    ("watched: never without a PV", "watched", [ENTRY | {"pv": None}], False),
    ("watched: never a role the module lacks", "watched", [None], False),
    # a row's word
    ("row: the module's verdict", "rowWord", [row()], "live"),
    ("row: an array is on request", "rowWord", [row(entry=ENTRY | {"read": "on_request"})], "on request"),
    ("row: the module silent is unknown", "rowWord", [row(view={"ok": False, "roles": {}})], "unknown"),
    ("row: Hare down is unknown", "rowWord", [row(hareUp=False)], "unknown"),
    ("row: a role the module lacks is unknown", "rowWord", [row(entry=None)], "unknown"),
    # a row's units
    ("units: the family's first", "units", ["bit", "A", {"units": "mm"}], "A"),
    ("units: else the record's for a value", "units", ["readback", None, {"units": "mm"}], "mm"),
    ("units: never the record's for a bit", "units", ["bit", None, {"units": "mm"}], ""),
    ("units: none known", "units", ["setpoint", None, {}], ""),
    # the device route as kept, and the header
    ("view: keeps the heartbeat", "deviceView", [True, {"state": "live", "heartbeat": "stale", "roles": []}],
     {"ok": True, "state": "live", "roles": {}, "heartbeat": "stale"}),
    ("view: a refusal is not ok", "deviceView", [False, {"code": "not_found"}], {"ok": False, "roles": {}}),
    ("header: the module's words", "header", [{"ok": True, "state": "error", "heartbeat": "live"}],
     {"state": "error", "heartbeat": "heartbeat live"}),
    ("header: no heartbeat is a dash", "header", [{"ok": True, "state": "live", "heartbeat": None}],
     {"state": "live", "heartbeat": "heartbeat —"}),
    ("header: silent is unknown", "header", [{"ok": False}], {"state": "unknown", "heartbeat": "heartbeat unknown"}),
    # the rail and the readings (device_rail.js)
    ("present: a channel's group, by its present role", "presentRole", [CH, KNOWN], "ch_a_present"),
    ("present: a channel's reading", "presentRole", [["ch_a_total"], KNOWN], "ch_a_present"),
    ("present: a mixed group is never hidden", "presentRole", [["acquire", "ch_a_range"], KNOWN], None),
    ("present: the device names no present role", "presentRole", [CH, CH], None),
    ("hidden: while present reads 0", "hidden", [{"value": 0}], True),
    ("hidden: present, or unknown, shows it", "hidden", [None], False),
    ("write: a checkbox is 1 or 0", "writeValue", [{"type": "checkbox", "tag": "INPUT", "checked": True, "raw": "on"}], 1),
    ("write: a select is its choice's index", "writeValue", [{"type": "select-one", "tag": "SELECT", "raw": "2"}], 2),
    ("write: an empty select sends nothing", "writeValue", [{"type": "select-one", "tag": "SELECT", "raw": ""}], None),
    ("write: a text setting is its text", "writeValue", [{"type": "text", "tag": "INPUT", "raw": "C:/data"}], "C:/data"),
    ("write: a whole-number field", "writeValue", [{"type": "number", "tag": "INPUT", "step": "1", "raw": "7"}], 7),
    ("write: a whole-number field rounds", "writeValue", [{"type": "number", "tag": "INPUT", "step": "1", "raw": "7.5"}], 8),
    ("write: a number", "writeValue", [{"type": "number", "tag": "INPUT", "step": "any", "raw": "0.25"}], 0.25),
    ("write: junk sends nothing", "writeValue", [{"type": "number", "tag": "INPUT", "step": "any", "raw": "x"}], None),
    ("alarms: one axis' error roles, by label", "alarmed", [AXES, "m1"], ["high limit switch"]),
    ("alarms: the other axis has none", "alarmed", [AXES, "m2"], []),
    ("alarms: a device without axes, every error role", "alarmed", [AXES, None], ["high limit switch", "fault"]),
    ("alarms: the module silent, none", "alarmed", [{"ok": False, "roles": {}}, None], []),
]  # fmt: skip


@pytest.fixture(scope="module")
def answers() -> list[Any]:
    return call_all("device_rules.js", [(fn, args) for _, fn, args, _ in ROWS])


@needs_node
@pytest.mark.parametrize("i", range(len(ROWS)), ids=[r[0] for r in ROWS])
def test_rule(answers: list[Any], i: int) -> None:
    assert answers[i] == ROWS[i][3]
