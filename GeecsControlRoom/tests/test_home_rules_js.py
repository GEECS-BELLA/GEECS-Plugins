"""home_rules.js under node: each row one rule of a home list's words, a preset row, the baseline chip, a restore plan (skipped without node)."""

from __future__ import annotations

from typing import Any

import pytest

from tests._node import call_all, needs_node

ROW = {"name": "ps1"}
SILENT = {"ok": False, "detail": "the module did not answer"}

# (id, key, body, the fields of listView that row pins)
ROWS: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = [
    ("silent: unknown, never none", "not_live", SILENT,
     {"count": "?", "chipState": "unknown", "chipText": "unknown", "none": False, "note": "the module did not answer"}),
    ("empty and every device read: none", "in_alarm", {"ok": True, "in_alarm": [], "unread": []},
     {"count": "0", "chipState": "ok", "chipText": "none", "none": True, "title": ""}),
    ("empty with an unread device: unknown", "in_alarm", {"ok": True, "in_alarm": [], "unread": [ROW]},
     {"count": "0+?", "chipState": "unknown", "chipText": "unknown", "none": False,
      "title": "no roles read for 1: ps1", "note": "the module gave no roles for 1 device: ps1"}),
    ("rows with an unread device: N+?", "in_alarm", {"ok": True, "in_alarm": [ROW], "unread": [{"name": "m"}]},
     {"count": "1+?", "chipState": "failed", "chipText": "1+?"}),
    ("an alarm row is failed", "in_alarm", {"ok": True, "in_alarm": [ROW], "unread": []},
     {"count": "1", "chipState": "failed", "chipText": "1"}),
    ("a not-live row is degraded", "not_live", {"ok": True, "not_live": [ROW]},
     {"count": "1", "chipState": "degraded", "rows": [ROW]}),
    ("unread never touches the state list", "not_live", {"ok": True, "not_live": [], "unread": [ROW]},
     {"count": "0", "chipText": "none", "none": True}),
]  # fmt: skip


NULL = "<null>"  # the row expects JSON null
PRESET = {"id": "p", "name": "bench-a", "time": "2026-10-29T12:00:00Z", "operator": "ann", "scope": {"devices": ["M"], "screen": []}, "golden": True, "pvCount": 3}  # fmt: skip
DIFF = [{"device": "M", "word": "differs"}, {"device": "S", "word": "differs"}, {"device": "M", "word": "same"}, {"device": "G", "word": "retired"}]  # fmt: skip
# (id, export, args, the fields pinned)
PRESET_ROWS: list[tuple[str, str, list[Any], dict[str, Any]]] = [
    ("a preset row: scope in words, golden", "presetRow", [PRESET], {"by": "ann", "scope": "devices: M", "golden": True, "count": 3}),
    ("no scope reads everything", "presetRow", [PRESET | {"scope": {}, "operator": None}], {"scope": "everything", "by": "—"}),
    ("no golden: the chip hides", "baselineChip", [{"ok": True, "golden": 0, "differs": 0}], {"hidden": True}),
    ("a golden with differences: degraded", "baselineChip", [{"ok": True, "golden": 1, "differs": 2, "incomplete": 0, "presets": [{"name": "a", "differs": 2, "incomplete": 0}]}], {"hidden": False, "text": "2 differ", "state": "degraded", "title": "a: 2 differ"}),
    ("as baseline: ok, every row compared", "baselineChip", [{"ok": True, "golden": 1, "differs": 0, "incomplete": 0, "presets": []}], {"text": "as baseline", "state": "ok"}),
    ("a row not compared: never as baseline", "baselineChip", [{"ok": True, "golden": 1, "differs": 0, "incomplete": 1, "presets": [{"name": "a", "differs": 0, "incomplete": 1}]}], {"text": "1 not compared", "state": "unknown", "title": "a: 0 differ, 1 not compared"}),
    ("differences and a row not compared: unknown, both said", "baselineChip", [{"ok": True, "golden": 1, "differs": 2, "incomplete": 3}], {"text": "2 differ · 3 not compared", "state": "unknown"}),
    ("a silent module: unknown, shown", "baselineChip", [{"ok": False, "detail": "down"}], {"hidden": False, "text": "unknown", "state": "unknown", "title": "down"}),
    ("a restore plan: differing rows of the picked devices", "restorePlan", [DIFF, ["M"]], {"devices": ["M"], "text": "1 row on M"}),
    ("nothing picked: nothing to restore", "restorePlan", [DIFF, []], {"devices": [], "rows": [], "text": "no device picked"}),
    ("a pick with no differing row: nothing", "restorePlan", [DIFF, ["G"]], {"rows": [], "text": "nothing differs on the picked devices"}),
    ("opening the drawer starts a generation", "openSelection", [None, {"id": "a"}], {"generation": 1, "preset": {"id": "a"}, "rows": []}),
    ("a diff for the current generation of an open drawer is taken", "takeDiff", [{"generation": 2, "preset": {"id": "b"}, "rows": []}, 2, [{"device": "M"}], True], {"generation": 2, "preset": {"id": "b"}, "rows": [{"device": "M"}]}),
    ("a late diff for a superseded generation is dropped", "takeDiff", [{"generation": 2, "preset": {"id": "b"}, "rows": []}, 1, [{"device": "M"}], True], NULL),
    ("a diff for a drawer closed meanwhile (Esc, the scrim) is dropped", "takeDiff", [{"generation": 2, "preset": {"id": "b"}, "rows": []}, 2, [{"device": "M"}], False], NULL),
    ("an answer for the open drawer's generation is current", "stillCurrent", [{"generation": 2}, 2, True], True),
    ("an answer for a superseded generation is not", "stillCurrent", [{"generation": 2}, 1, True], False),
    ("an answer for a closed drawer is not (any status)", "stillCurrent", [{"generation": 2}, 2, False], False),
    ("the drawer's summary counts every word but same and differs as not compared", "diffSummary", [[{"word": "differs"}, {"word": "same"}, {"word": "unread"}, {"word": "stale"}]], "4 rows · 1 differ · 2 not compared; tick the devices to restore"),
    ("an unsaved row explains itself", "diffNote", ["unsaved"], None),
]  # fmt: skip


@pytest.fixture(scope="module")
def preset_answers() -> list[Any]:
    return call_all("home_rules.js", [(fn, args) for _, fn, args, _ in PRESET_ROWS])


@needs_node
@pytest.mark.parametrize("i", range(len(PRESET_ROWS)), ids=[r[0] for r in PRESET_ROWS])
def test_preset_rule(preset_answers: list[Any], i: int) -> None:
    want = PRESET_ROWS[i][3]
    if want is None:
        assert preset_answers[i] == "unsaved: unreachable at save time"
    elif want == NULL:
        assert preset_answers[i] is None
    elif not isinstance(want, dict):
        assert preset_answers[i] == want
    else:
        assert {k: preset_answers[i][k] for k in want} == want


@pytest.fixture(scope="module")
def answers() -> list[Any]:
    return call_all("home_rules.js", [("listView", [key, body]) for _, key, body, _ in ROWS])  # fmt: skip


@needs_node
@pytest.mark.parametrize("i", range(len(ROWS)), ids=[r[0] for r in ROWS])
def test_rule(answers: list[Any], i: int) -> None:
    want = ROWS[i][3]
    assert {k: answers[i][k] for k in want} == want
