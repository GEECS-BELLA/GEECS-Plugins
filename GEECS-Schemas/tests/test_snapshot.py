"""Snapshot: a scope of devices or a panel, one row per role, round-trips."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY, Snapshot, SnapshotScope

DOC = {
    "schema_version": 1,
    "name": "chicane tuned",
    "created": 1761400000.0,
    "author": "S. Barber",
    "scope": {"devices": ["PS1", "Stage"]},
    "rows": [
        {"device": "PS1", "role": "current_sp", "setpoint": 0.25, "readback": 0.249},
        {"device": "PS1", "role": "output_sp", "setpoint": 1},
        {"device": "Stage", "axis": "m1", "role": "val", "setpoint": 1.25},
        {"device": "Stage", "axis": "m2", "role": "val", "setpoint": -3},
        {"device": "Cam", "role": "trigger_mode", "setpoint": "On"},
    ],
}


def test_registered_and_round_trips() -> None:
    assert SCHEMA_REGISTRY["snapshot"] is Snapshot
    snapshot = Snapshot.model_validate(DOC)
    assert Snapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
    assert snapshot.rows[4].setpoint == "On" and snapshot.rows[0].readback == 0.249
    assert not snapshot.golden and snapshot.note == ""


@pytest.mark.parametrize(
    "scope", [{}, {"devices": ["A"], "panel": "p"}, {"devices": []}, {"panel": "a/b"}]
)
def test_a_scope_is_devices_or_a_panel(scope: dict) -> None:
    with pytest.raises(ValidationError):
        SnapshotScope.model_validate(scope)


def test_a_row_is_kept_once() -> None:
    doc = {**DOC, "rows": DOC["rows"] + [DOC["rows"][2]]}
    with pytest.raises(ValidationError, match="rows"):
        Snapshot.model_validate(doc)


def test_a_snapshot_keeps_at_least_one_row_and_a_setpoint_each() -> None:
    with pytest.raises(ValidationError):
        Snapshot.model_validate({**DOC, "rows": []})
    with pytest.raises(ValidationError):
        Snapshot.model_validate({**DOC, "rows": [{"device": "PS1", "role": "x"}]})
