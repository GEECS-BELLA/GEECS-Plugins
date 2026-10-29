"""home_rules.js under node: each row one rule of a home list's words (skipped without node)."""

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


@pytest.fixture(scope="module")
def answers() -> list[Any]:
    return call_all("home_rules.js", [("listView", [key, body]) for _, key, body, _ in ROWS])  # fmt: skip


@needs_node
@pytest.mark.parametrize("i", range(len(ROWS)), ids=[r[0] for r in ROWS])
def test_rule(answers: list[Any], i: int) -> None:
    want = ROWS[i][3]
    assert {k: answers[i][k] for k in want} == want
