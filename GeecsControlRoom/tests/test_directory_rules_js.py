"""directory_rules.js under node: each row one rule of the home's device directory (the filter, a row's words, the chip); skipped without node."""

from __future__ import annotations

from typing import Any

import pytest

from tests._node import call_all, needs_node

GROUPS = [
    {"group": "camera", "rows": [{"name": "CamTop", "family": "ad_camera_chain"}, {"name": "CamRad", "family": "ad_camera_chain"}]},
    {"group": "trace", "rows": [{"name": "spec1", "family": "spectrometer_contract"}, {"name": "scope1", "family": "scope_contract"}]},
    {"group": "no page yet", "rows": [{"name": "dg1", "family": "srs_dg645"}]},
]  # fmt: skip
WORDS = {
    "ok": True,
    "words": {
        "CamRad": {"state": "live", "kit": "ok", "heartbeat": "live", "heartbeatKit": "ok"},
        "spec1": {"state": "stale", "kit": "degraded", "heartbeat": "stale", "heartbeatKit": "degraded"},
        "ps1": {"state": "live", "kit": "ok", "heartbeat": None, "heartbeatKit": None},
    },
}  # fmt: skip
SILENT = {"ok": False, "detail": "the module did not answer: refused"}
UNKNOWN = {
    "state": "unknown",
    "kit": "unknown",
    "heartbeat": "unknown",
    "heartbeatKit": "unknown",
}

# Each row: its id, the export, its arguments, the answer.
ROWS: list[tuple[str, str, list[Any], Any]] = [
    ("an empty query keeps every row", "matches", ["", "CamRad", "ad_camera_chain"], True),
    ("a substring of the name, any case", "matches", ["rad", "CamRad", "ad_camera_chain"], True),
    ("a substring of the family", "matches", ["scope", "scope1", "scope_contract"], True),
    ("every term must match, name or family", "matches", ["cam camera", "CamRad", "ad_camera_chain"], True),
    ("one term missing: no match", "matches", ["cam scope", "CamRad", "ad_camera_chain"], False),
    ("a family-less row matches on its name", "matches", ["dg", "dg1", None], True),
    ("no query: nothing hidden", "filterView", [GROUPS, "  "],
     {"shown": {"camera": ["CamTop", "CamRad"], "trace": ["spec1", "scope1"], "no page yet": ["dg1"]},
      "hidden": [], "matched": 5, "total": 5, "none": False}),
    ("a filter hides the groups it empties", "filterView", [GROUPS, "contract"],
     {"shown": {"camera": [], "trace": ["spec1", "scope1"], "no page yet": []},
      "hidden": ["camera", "no page yet"], "matched": 2, "total": 5, "none": False}),
    ("nothing matches: none", "filterView", [GROUPS, "nope"],
     {"shown": {"camera": [], "trace": [], "no page yet": []},
      "hidden": ["camera", "trace", "no page yet"], "matched": 0, "total": 5, "none": True}),
    ("a row's words are the module's", "cellWords", [WORDS, "spec1"],
     {"state": "stale", "kit": "degraded", "heartbeat": "stale", "heartbeatKit": "degraded"}),
    ("a null heartbeat is no IOC", "cellWords", [WORDS, "ps1"],
     {"state": "live", "kit": "ok", "heartbeat": "no IOC", "heartbeatKit": "unknown"}),
    ("a silent module: unknown, never a guess", "cellWords", [SILENT, "CamRad"], UNKNOWN),
    ("a name the answer lacks: unknown", "cellWords", [WORDS, "CamTop"], UNKNOWN),
    ("the chip counts the live ones", "chip", [WORDS, ["CamRad", "spec1"]], {"text": "1/2 live", "state": "degraded", "title": ""}),
    ("every one live: ok", "chip", [WORDS, ["CamRad"]], {"text": "1/1 live", "state": "ok", "title": ""}),
    ("a silent module: unknown, with why", "chip", [SILENT, ["CamRad"]],
     {"text": "unknown", "state": "unknown", "title": "the module did not answer: refused"}),
]  # fmt: skip


@pytest.fixture(scope="module")
def answers() -> list[Any]:
    return call_all("directory_rules.js", [(fn, args) for _, fn, args, _ in ROWS])


@needs_node
@pytest.mark.parametrize("i", range(len(ROWS)), ids=[r[0] for r in ROWS])
def test_rule(answers: list[Any], i: int) -> None:
    assert answers[i] == ROWS[i][3]
