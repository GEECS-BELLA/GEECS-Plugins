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
     {"ok": True, "state": "live", "roles": {}, "heartbeat": "stale", "source": None}),
    ("view: keeps a DB device's source", "deviceView", [True, {"state": "live", "source": "gateway", "roles": []}],
     {"ok": True, "state": "live", "roles": {}, "heartbeat": None, "source": "gateway"}),
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
    ("commit: Enter writes a draft", "commitKey", ["Enter", True], "write"),
    ("commit: Escape drops a draft", "commitKey", ["Escape", True], "revert"),
    ("commit: an arrow key never writes", "commitKey", ["ArrowUp", True], None),
    ("commit: Enter without a draft sends nothing", "commitKey", ["Enter", False], None),
    ("tweak: no draft, the press alone", "tweakList", ["twf", "twv", False, 2.0], [["twf", 1]]),
    ("tweak: a draft step is written first, in the same list", "tweakList", ["twr", "twv", True, 0.5], [["twv", 0.5], ["twr", 1]]),
    ("tweak: a draft that is no number sends nothing", "tweakList", ["twf", "twv", True, None], None),
    # the setpoint control: what the field tracks and what stands beside it
    ("sources: areaDetector's X beside X_RBV: the field tracks X, both shown", "sources",
     [{"role": "exposure", "pv": "c:AcquireTime_RBV", "writePv": "c:AcquireTime"}, "exposure", False], {"field": "write", "set": True}),
    ("sources: a motor's .VAL with its .RBV: the field tracks .VAL, both shown", "sources",
     [{"role": "val", "pv": "m:m1.VAL", "writePv": "m:m1.VAL"}, "rbv", False], {"field": "self", "set": True}),
    ("sources: one record (a speed): one value", "sources",
     [{"role": "velo", "pv": "m:m1.VELO", "writePv": "m:m1.VELO"}, "velo", False], {"field": "self", "set": False}),
    ("sources: a DB device tracks its value, its :SP shown beside", "sources",
     [{"role": "Current", "pv": "u:ps:current", "writePv": "u:ps:current:SP"}, "Current", True], {"field": "self", "set": True}),
    ("sources: no entry, the role's own record", "sources", [None, "x", False], {"field": "self", "set": False}),
    ("field text: a number to six digits", "fieldText", [0.1 + 0.2], "0.3"),
    ("field text: a whole number", "fieldText", [12], "12"),
    ("field text: text as it is", "fieldText", ["Name 1"], "Name 1"),
    ("field text: nothing read leaves the field", "fieldText", [None], None),
    # the draft: kept until accepted
    ("accepted: the module's word", "accepted", [{"outcome": "accepted", "journalId": 3}], True),
    ("accepted: a refusal is not", "accepted", [{"outcome": "denied", "code": "not_permitted"}], False),
    ("accepted: no answer is not", "accepted", [None], False),
    ("draft: cleared on acceptance", "keepsDraft", [True], False),
    ("draft: kept on a refusal", "keepsDraft", [False], True),
    ("draft: kept when unsent (another write in flight)", "keepsDraft", [None], True),
    # the tolerance and the mark
    ("tolerance: the family's number first", "tolerance", [{"hint": 0.05, "by": 0.2, "precision": 3}], 0.05),
    ("tolerance: else its role's value (a motor's RDBD)", "tolerance", [{"by": 0.002, "precision": 3}], 0.002),
    ("tolerance: its role unread judges nothing", "tolerance", [{"by": None, "precision": 3}], None),
    ("tolerance: else one unit in the record's last digit", "tolerance", [{"precision": 2}], 0.01),
    ("tolerance: a whole number is exact", "tolerance", [{"precision": 2, "integer": True}], 0),
    ("tolerance: no precision is exact", "tolerance", [{}], 0),
    ("done: DMOV 1 is done", "done", [{"value": 1}], True),
    ("done: DMOV 0 is moving", "done", [{"value": 0}], False),
    ("done: unread is not done", "done", [None], False),
    ("mark: unarmed, none", "mark", [{"now": 5000, "sp": 1, "mv": 2, "tol": 0}], "none"),
    ("mark: before it settles, waiting", "mark", [{"armedAt": 5000, "now": 5500, "sp": 1, "mv": 2, "tol": 0}], "waiting"),
    ("mark: settled outside the tolerance, differs", "mark", [{"armedAt": 5000, "now": 6000, "sp": 1, "mv": 1.02, "tol": 0.01}], "differs"),
    ("mark: one unit in the last digit agrees (float error)", "mark", [{"armedAt": 5000, "now": 6000, "sp": 1.0, "mv": 1.1, "tol": 0.1}], "agrees"),
    ("mark: within it, agrees", "mark", [{"armedAt": 5000, "now": 6000, "sp": 10, "mv": 10.004, "tol": 0.01}], "agrees"),
    ("mark: a motor still moving waits", "mark", [{"armedAt": 5000, "now": 9000, "isDone": False, "sp": 3, "mv": 1, "tol": 0.001}], "waiting"),
    ("mark: a motor done outside its deadband differs", "mark", [{"armedAt": 5000, "now": 9000, "isDone": True, "sp": 3, "mv": 2.99, "tol": 0.001}], "differs"),
    ("mark: a value unread waits", "mark", [{"armedAt": 5000, "now": 9000, "sp": 3, "mv": None, "tol": 0}], "waiting"),
    ("mark: an unread tolerance waits", "mark", [{"armedAt": 5000, "now": 9000, "sp": 3, "mv": 2, "tol": None}], "waiting"),
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
