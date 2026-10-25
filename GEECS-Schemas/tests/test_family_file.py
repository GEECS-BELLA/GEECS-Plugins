"""FamilyFile: the installed families load, a tiny family round-trips, every rule bites."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY, FamilyFile, load_families
from geecs_schemas.epics import family as family_module

#: A tiny family touching every link: a counter-judged statistic, a
#: setpoint with its readback, a parameter written beside its _RBV that
#: settles by that readback, a hazard command, a fault bit, an array.
TINY = {
    "schema_version": 1,
    "template": "tiny",
    "kind": "setpoint_device",
    "description": "A family small enough to read.",
    "roles": [
        {"name": "counter", "suffix": "ArrayCounter_RBV", "type": "counter", "label": "frames"},
        {"name": "on", "suffix": "Compute", "type": "bit", "label": "computing"},
        {"name": "mean", "suffix": "Mean_RBV", "type": "readback", "label": "mean",
         "fresh_by": "counter", "computed_by": "on"},
        {"name": "level", "suffix": "level", "type": "readback", "label": "level"},
        {"name": "level_sp", "suffix": "level:SP", "type": "setpoint", "label": "level setpoint",
         "writable": True, "readback": "level", "limits": [0, 10], "archive": True},
        {"name": "acquire", "suffix": "Acquire_RBV", "type": "enum", "label": "acquire",
         "writable": True, "write": "Acquire", "settled_by": "acquire", "locked": True},
        {"name": "fire", "suffix": "fire", "type": "command", "label": "fire",
         "writable": True, "confirm": "hazard"},
        {"name": "stop", "suffix": "stop", "type": "command", "label": "stop", "writable": True},
        {"name": "fault", "suffix": "fault", "type": "bit", "label": "fault", "fault": True},
        {"name": "trace", "suffix": "Trace", "type": "array", "label": "trace"},
    ],
    "actions": {
        "fire": {"label": "fire", "roles": ["fire"], "requires": ["stop"]},
        "stop": {"label": "stop", "roles": ["stop"]},
    },
}  # fmt: skip


def tiny() -> dict:
    """A fresh, mutable copy of :data:`TINY`."""
    return copy.deepcopy(TINY)


def role(doc: dict, name: str) -> dict:
    """The role called *name* in a raw family document."""
    return next(r for r in doc["roles"] if r["name"] == name)


def test_it_is_a_registered_document() -> None:
    assert SCHEMA_REGISTRY["family_file"] is FamilyFile


def test_a_tiny_family_round_trips() -> None:
    family = FamilyFile.model_validate(tiny())
    again = FamilyFile.model_validate_json(family.model_dump_json())
    assert again == family
    assert family.role_names()[:3] == ["counter", "on", "mean"]


def test_how_each_role_is_read_and_written() -> None:
    family = FamilyFile.model_validate(tiny())
    assert family.role("mean").read_mode == "monitor"
    assert family.role("trace").read_mode == "on_request"
    assert (
        family.role("fire").read_mode == "never"
    )  # a command's record is pressed, not read
    assert family.role("acquire").write_suffix == "Acquire"
    assert family.role("level_sp").write_suffix == "level:SP"
    assert family.role("level").write_suffix is None
    with pytest.raises(KeyError):
        family.role("nope")


def _refused(doc: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        FamilyFile.model_validate(doc)


def test_a_role_name_is_unique() -> None:
    doc = tiny()
    doc["roles"].append({**role(doc, "level"), "suffix": "other"})
    _refused(doc, "role names must be unique")


def test_a_read_suffix_is_unique() -> None:
    doc = tiny()
    role(doc, "level")["suffix"] = "Mean_RBV"
    _refused(doc, "read suffixes must be unique")


def test_a_write_suffix_is_unique() -> None:
    doc = tiny()
    role(doc, "level_sp")["suffix"] = "Acquire"
    _refused(doc, "write suffixes must be unique")


def test_settled_by_names_a_role() -> None:
    doc = tiny()
    role(doc, "acquire")["settled_by"] = "acquire_rbv"
    _refused(doc, "settled_by names no role")


def test_settled_by_names_a_monitored_role() -> None:
    doc = tiny()
    role(doc, "acquire")["settled_by"] = "trace"
    _refused(doc, "settled_by 'trace' is not monitored")


def test_readback_names_a_role() -> None:
    doc = tiny()
    role(doc, "level_sp")["readback"] = "levl"
    _refused(doc, "readback names no role")


def test_fresh_by_names_a_counter() -> None:
    doc = tiny()
    role(doc, "mean")["fresh_by"] = "level"
    _refused(doc, "fresh_by names a counter")


def test_computed_by_names_a_role() -> None:
    doc = tiny()
    role(doc, "mean")["computed_by"] = "off"
    _refused(doc, "computed_by names no role")


def test_a_hazard_confirm_is_on_a_writable_role_only() -> None:
    doc = tiny()
    role(doc, "fault")["confirm"] = "hazard"
    _refused(doc, "apply to writable roles only")


@pytest.mark.parametrize(
    "key, value",
    [("locked", True), ("limits", [0, 1]), ("choices", ["a"]), ("integer", True),
     ("settled_by", "level")],
)  # fmt: skip
def test_write_policy_is_on_a_writable_role_only(key: str, value: object) -> None:
    doc = tiny()
    role(doc, "level")[key] = value
    _refused(doc, "apply to writable roles only")


@pytest.mark.parametrize("name", ["level_sp", "stop"])
def test_setpoints_and_commands_are_writable(name: str) -> None:
    doc = tiny()
    role(doc, name)["writable"] = False
    _refused(doc, "is writable")


@pytest.mark.parametrize("name", ["level", "counter", "trace"])
def test_readbacks_counters_and_arrays_are_never_writable(name: str) -> None:
    doc = tiny()
    role(doc, name)["writable"] = True
    _refused(doc, "is never writable")


def test_a_separate_write_record_needs_writable() -> None:
    doc = tiny()
    role(doc, "on").update(write="Compute:SP")
    _refused(doc, "'write' needs writable")


def test_only_a_bit_is_a_fault() -> None:
    doc = tiny()
    role(doc, "level")["fault"] = True
    _refused(doc, "only a bit is a fault")


def test_limits_are_low_then_high() -> None:
    doc = tiny()
    role(doc, "level_sp")["limits"] = [10, 0]
    _refused(doc, "limits are low then high")


def test_an_action_writes_writable_roles() -> None:
    doc = tiny()
    doc["actions"]["peek"] = {"label": "peek", "roles": ["level"]}
    _refused(doc, "'level' is not a writable role")


def test_an_action_requires_known_actions() -> None:
    doc = tiny()
    doc["actions"]["fire"]["requires"] = ["halt"]
    _refused(doc, "requires unknown action")


def test_an_unknown_key_is_refused() -> None:
    doc = tiny()
    role(doc, "level")["egu"] = "A"
    _refused(doc, "Extra inputs are not permitted")


# ---- the installed families -------------------------------------------------


def test_the_installed_families_load_by_template() -> None:
    families = load_families()
    assert set(families) == {
        "ad_camera_chain",
        "easydriver",
        "fastps",
        "motor_record",
        "scope_contract",
    }
    assert families["motor_record"].per_axis
    assert families["ad_camera_chain"].kind == "aravis_camera"


def test_the_installed_families_mark_their_hazards() -> None:
    families = load_families()
    hazards = {
        name: sorted(r.name for r in f.roles if r.confirm == "hazard")
        for name, f in families.items()
    }
    assert hazards["motor_record"] == ["homf", "homr"]
    assert hazards["fastps"] == hazards["easydriver"] == ["output_sp"]


def test_the_loader_refuses_a_file_named_for_another_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "families").mkdir()
    (tmp_path / "families" / "other.json").write_text(json.dumps(TINY))
    monkeypatch.setattr(family_module.resources, "files", lambda _pkg: tmp_path)
    load_families.cache_clear()
    try:
        with pytest.raises(ValueError, match="declares template 'tiny'"):
            load_families()
        (tmp_path / "families" / "other.json").rename(
            tmp_path / "families" / "tiny.json"
        )
        load_families.cache_clear()
        assert list(load_families()) == ["tiny"]
    finally:
        load_families.cache_clear()
