"""PanelDevices: per-device panel data round-trips and is checked against its family."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY, FamilyFile, PanelDevices, load_families
from test_family_file import TINY

#: The tiny family under two names: one with actions, one per-axis without.
FAMILIES = {
    "tiny": FamilyFile.model_validate(TINY),
    "axes": FamilyFile.model_validate(
        {**copy.deepcopy(TINY), "template": "axes", "per_axis": True, "actions": {}}
    ),
}

DOC = {
    "schema_version": 1,
    "devices": {
        "U_Tiny": {
            "family": "tiny",
            "prefix": "bench:tiny1:",
            "roles": {"level_sp": "level_ramp:SP"},
            "labels": {"fault": "chiller flow"},
            "allow": ["fire", "stop"],
            "note": "On the bench.",
            "archive": ["mean"],
        },
        "U_Stage": {"family": "axes", "axes": ["m1", "m2"]},
    },
}


def doc() -> dict:
    """A fresh, mutable copy of :data:`DOC`."""
    return copy.deepcopy(DOC)


def validate(document: dict) -> PanelDevices:
    """Validate against the tiny families."""
    return PanelDevices.model_validate(document, context={"families": FAMILIES})


def test_it_is_a_registered_document() -> None:
    assert SCHEMA_REGISTRY["panel_devices"] is PanelDevices


def test_a_document_round_trips() -> None:
    devices = validate(doc())
    assert validate(devices.model_dump()) == devices
    assert devices.families() == {"U_Tiny": "tiny", "U_Stage": "axes"}


def test_resolve_applies_the_devices_differences() -> None:
    device = validate(doc()).devices["U_Tiny"]
    roles = {r.name: r for r in device.resolve(FAMILIES["tiny"])}
    assert roles["level_sp"].suffix == "level_ramp:SP"
    assert roles["fault"].label == "chiller flow"
    assert roles["mean"].archive and roles["level_sp"].archive
    assert not roles["level"].archive
    assert roles["level"].suffix == "level"  # not overridden: the family's own


def test_allowed_actions_are_the_familys_within_allow() -> None:
    devices = validate(doc())
    assert devices.devices["U_Tiny"].allowed_actions(FAMILIES["tiny"]) == [
        "fire",
        "stop",
    ]
    unrestricted = devices.devices["U_Tiny"].model_copy(update={"allow": None})
    assert unrestricted.allowed_actions(FAMILIES["tiny"]) == ["fire", "stop"]


def _refused(document: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        validate(document)


def test_a_family_must_exist() -> None:
    d = doc()
    d["devices"]["U_Tiny"]["family"] = "tinny"
    _refused(d, "no family 'tinny'")


@pytest.mark.parametrize(
    "key, value",
    [("roles", {"levl": "x"}), ("labels", {"falt": "x"}), ("archive", ["meen"])],
)
def test_an_overridden_relabelled_or_archived_role_exists(
    key: str, value: object
) -> None:
    d = doc()
    d["devices"]["U_Tiny"][key] = value
    _refused(d, f"{key} names role")


def test_an_allowed_action_exists() -> None:
    d = doc()
    d["devices"]["U_Tiny"]["allow"] = ["stop", "launch"]
    _refused(d, r"unknown action\(s\) \['launch'\]")


def test_an_allowed_action_brings_what_it_requires() -> None:
    d = doc()
    d["devices"]["U_Tiny"]["allow"] = ["fire"]
    _refused(d, r"allowing 'fire' needs \['stop'\]")


def test_allow_needs_a_family_with_actions() -> None:
    d = doc()
    d["devices"]["U_Stage"]["allow"] = ["stop"]
    _refused(d, "has no actions to allow")


def test_axes_need_a_per_axis_family() -> None:
    d = doc()
    d["devices"]["U_Tiny"]["axes"] = ["x"]
    _refused(d, "has no axes")


def test_an_override_may_not_make_two_roles_read_one_record() -> None:
    d = doc()
    d["devices"]["U_Tiny"]["roles"] = {"level_sp": "level"}
    _refused(d, "read suffixes must be unique")


@pytest.mark.parametrize("name", ["a/b", "a b", "..", "50%"])
def test_a_device_name_is_one_url_segment(name: str) -> None:
    d = doc()
    d["devices"][name] = d["devices"].pop("U_Stage")
    _refused(d, "is not one URL path segment")


def test_a_prefix_ends_in_a_colon() -> None:
    d = doc()
    d["devices"]["U_Tiny"]["prefix"] = "bench:tiny1"
    _refused(d, "String should match pattern")


def test_a_file_validates_against_the_installed_families(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "panel_devices.yaml"
    path.write_text(
        "devices:\n"
        "  U_ChicaneInner:\n"
        "    family: fastps\n"
        "    prefix: 'undulator:fpstest1:'\n"
        "    labels: {interlock_1: chiller flow}\n"
        "    allow: [current, output, reset]\n"
    )
    devices = PanelDevices.from_path(path)
    fastps = load_families()["fastps"]
    assert devices.devices["U_ChicaneInner"].allowed_actions(fastps) == [
        "current",
        "output",
        "reset",
    ]
    path.write_text("devices:\n  U_X: {family: motor_record, allow: [move]}\n")
    with pytest.raises(ValidationError, match=r"allowing 'move' needs \['stop'\]"):
        PanelDevices.from_path(path)
