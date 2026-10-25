"""ComposedPanel and PanelOverride: closed, round-trip, and checked against the families."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from geecs_schemas import (
    SCHEMA_REGISTRY,
    ComposedPanel,
    FamilyFile,
    PanelOverride,
    load_families,
)
from test_family_file import TINY

FAMILIES = {
    "tiny": FamilyFile.model_validate(TINY),
    "ad_camera_chain": load_families()["ad_camera_chain"],
    "motor_record": load_families()["motor_record"],
}
DEVICES = {"U_Tiny": "tiny", "UC_Cam": "ad_camera_chain", "U_Stage": "motor_record"}
CONTEXT = {"families": FAMILIES, "devices": DEVICES}

PANEL = {
    "schema_version": 1,
    "title": "Bench",
    "sections": [
        {
            "title": "Devices",
            "columns": 3,
            "widgets": [
                {"kind": "device", "device": "U_Tiny", "roles": ["level", "level_sp"]},
                {"kind": "component", "component": "camera_viewer", "device": "UC_Cam"},
                {"kind": "pv", "pv": "bench:flow", "type": "readback", "label": "flow"},
                {
                    "kind": "trend",
                    "span": 3,
                    "series": [
                        {"device": "U_Tiny", "role": "mean"},
                        {"pv": "bench:flow", "y_axis": "right"},
                    ],
                },
            ],
        },
        {
            "title": "Room",
            "show_when": {"pv": "bench:mode", "equals": "Run"},
            "widgets": [
                {"kind": "action", "plan": "park"},
                {"kind": "journal"},
                {"kind": "roster", "show_when": {"pv": "bench:open", "equals": 1}},
                {"kind": "alarms"},
            ],
        },
    ],
}


def panel() -> dict:
    """A fresh, mutable copy of :data:`PANEL`."""
    return copy.deepcopy(PANEL)


def widget(doc: dict, section: int, index: int) -> dict:
    """One widget of a raw panel document."""
    return doc["sections"][section]["widgets"][index]


def _refused(doc: dict, match: str, context: dict | None = CONTEXT) -> None:
    with pytest.raises(ValidationError, match=match):
        ComposedPanel.model_validate(doc, context=context)


def test_they_are_registered_documents() -> None:
    assert SCHEMA_REGISTRY["composed_panel"] is ComposedPanel
    assert SCHEMA_REGISTRY["panel_override"] is PanelOverride


def test_a_panel_round_trips_with_every_widget_kind() -> None:
    p = ComposedPanel.model_validate(panel(), context=CONTEXT)
    kinds = [w.kind for s in p.sections for w in s.widgets]
    assert kinds == [
        "device", "component", "pv", "trend", "action", "journal", "roster", "alarms",
    ]  # fmt: skip
    assert ComposedPanel.model_validate_json(p.model_dump_json(), context=CONTEXT) == p


def test_without_devices_only_the_structure_is_checked() -> None:
    doc = panel()
    widget(doc, 0, 0)["roles"] = ["no_such_role"]
    ComposedPanel.model_validate(
        doc
    )  # the bridge passes its devices; a bare load cannot
    _refused(doc, "has no role")


def test_a_device_widget_names_its_familys_roles() -> None:
    doc = panel()
    widget(doc, 0, 0)["roles"] = ["level", "levle"]
    _refused(doc, r"'U_Tiny' \(tiny\) has no role\(s\) \['levle'\]")


def test_a_trend_names_its_familys_roles() -> None:
    doc = panel()
    widget(doc, 0, 3)["series"][0]["role"] = "median"
    _refused(doc, r"has no role\(s\) \['median'\]")


def test_a_widget_names_a_known_device() -> None:
    doc = panel()
    widget(doc, 0, 0)["device"] = "U_Nope"
    _refused(doc, "no device 'U_Nope'")


def test_a_component_draws_its_own_family() -> None:
    doc = panel()
    widget(doc, 0, 1)["device"] = "U_Tiny"
    _refused(doc, "camera_viewer draws a ad_camera_chain device")


@pytest.mark.parametrize("role", ["trace", "fire"])
def test_a_trend_plots_a_monitored_role(role: str) -> None:
    doc = panel()
    widget(doc, 0, 3)["series"][0]["role"] = role
    _refused(doc, "not trendable")


def test_a_trend_of_a_per_axis_device_names_its_axis() -> None:
    doc = panel()
    widget(doc, 0, 3)["series"][0] = {"device": "U_Stage", "role": "rbv"}
    _refused(doc, "names its axis")
    widget(doc, 0, 3)["series"][0]["axis"] = "m1"
    ComposedPanel.model_validate(doc, context=CONTEXT)
    widget(doc, 0, 3)["series"][0] = {"device": "U_Tiny", "role": "mean", "axis": "m1"}
    _refused(doc, "has no axis")


@pytest.mark.parametrize("kind", ["setpoint", "command", "image", "array"])
def test_a_bare_pv_is_read_only(kind: str) -> None:
    doc = panel()
    widget(doc, 0, 2)["type"] = kind
    _refused(doc, "type", context=None)


def test_a_device_widget_shows_a_role_once() -> None:
    doc = panel()
    widget(doc, 0, 0)["roles"] = ["level", "level"]
    _refused(doc, "roles must be unique", context=None)


def test_a_widget_fits_its_sections_grid() -> None:
    doc = panel()
    widget(doc, 0, 0)["span"] = 4  # the section has 3 columns
    _refused(doc, "span more than its 3 columns", context=None)


def test_a_device_name_is_one_url_segment() -> None:
    doc = panel()
    widget(doc, 0, 0)["device"] = "a/b"
    _refused(doc, "not one URL path segment", context=None)


def test_an_unknown_component_is_refused() -> None:
    doc = panel()
    widget(doc, 0, 1)["component"] = "plotter"
    _refused(doc, "camera_viewer", context=None)


@pytest.mark.parametrize(
    "series",
    [
        {"pv": "a:b", "device": "U_Tiny", "role": "mean"},
        {"device": "U_Tiny"},
        {"role": "mean"},
        {},
    ],
)
def test_a_trend_series_is_a_pv_or_a_device_role(series: dict) -> None:
    doc = panel()
    widget(doc, 0, 3)["series"] = [series]
    _refused(doc, "names a pv, or a device and a role", context=None)


def test_the_spec_is_closed() -> None:
    doc = panel()
    widget(doc, 0, 2)["expr"] = "a + b"
    _refused(doc, "Extra inputs are not permitted", context=None)
    doc = panel()
    doc["sections"][1]["show_when"]["greater_than"] = 3
    _refused(doc, "Extra inputs are not permitted", context=None)
    doc = panel()
    widget(doc, 1, 0)["kind"] = "script"
    _refused(doc, "does not match any of the expected tags", context=None)


@pytest.mark.parametrize("value", [["Run", "Idle"], True])
def test_show_when_takes_a_number_or_a_name(value: object) -> None:
    doc = panel()
    doc["sections"][1]["show_when"]["equals"] = (
        value  # True: YAML's reading of a bare On
    )
    _refused(doc, "show_when.equals", context=None)


@pytest.mark.parametrize("key, value", [("span", 0), ("span", 13)])
def test_a_widget_spans_one_to_twelve_columns(key: str, value: int) -> None:
    doc = panel()
    widget(doc, 0, 0)[key] = value
    _refused(doc, "span", context=None)


def test_a_section_has_a_widget() -> None:
    doc = panel()
    doc["sections"][1]["widgets"] = []
    _refused(doc, "at least 1 item", context=None)


# ---- overrides ---------------------------------------------------------------

OVERRIDE = {"family": "tiny", "hide": ["trace"], "labels": {"mean": "average"}, "order": ["level", "level_sp"]}  # fmt: skip


def _override(doc: dict) -> PanelOverride:
    return PanelOverride.model_validate(doc, context={"families": FAMILIES})


def test_an_override_round_trips() -> None:
    o = _override(copy.deepcopy(OVERRIDE))
    assert _override(o.model_dump()) == o


@pytest.mark.parametrize(
    "key, value",
    [("hide", ["nope"]), ("labels", {"nope": "x"}), ("order", ["nope"])],
)
def test_an_override_names_the_familys_roles(key: str, value: object) -> None:
    with pytest.raises(ValidationError, match=f"{key} names role"):
        _override({**OVERRIDE, key: value})


def test_an_override_orders_a_role_once() -> None:
    with pytest.raises(ValidationError, match="ordered roles must be unique"):
        _override({**OVERRIDE, "order": ["level", "level"]})


def test_an_override_does_not_hide_what_it_shows() -> None:
    with pytest.raises(ValidationError, match="hidden and also relabelled or ordered"):
        _override({**OVERRIDE, "hide": ["level"]})


def test_an_override_names_a_family() -> None:
    with pytest.raises(ValidationError, match="no family 'tinny'"):
        _override({**OVERRIDE, "family": "tinny"})


def test_an_override_adds_nothing() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _override({**OVERRIDE, "add": [{"name": "x", "suffix": "x"}]})


def test_an_override_validates_against_the_installed_families() -> None:
    PanelOverride.model_validate({"family": "motor_record", "hide": ["set"]})
    with pytest.raises(ValidationError, match="does not have"):
        PanelOverride.model_validate({"family": "motor_record", "hide": ["nope"]})
