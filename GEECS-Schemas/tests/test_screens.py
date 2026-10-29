"""Screen: seven widget kinds, kit tokens, one kind of binding; resolve() fits widgets to roles."""

from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError

from geecs_schemas import (
    NativeDevices,
    PanelDevices,
    Screen,
    ScreenError,
    load_families,
    resolve_screen,
)
from geecs_schemas.screens import WidgetKind

PANEL = PanelDevices.model_validate({"devices": {"ps": {"family": "easydriver"}}})
NATIVE = NativeDevices.model_validate({"devices": {"st": {"kind": "epics_motor", "axes": ["m1"]}}})  # fmt: skip


OUT = {"device": "ps", "role": "output_sp"}
FLIP = {
    "kind": "flipper",
    "bind": OUT,
    "indicators": [{"device": "ps", "role": "output"}],
}


def screen(*widgets: dict, tone: str = "trace-1") -> Screen:
    group = {"title": "G", "at": [1, 1], "tone": tone, "widgets": list(widgets)}
    return Screen.model_validate({"title": "S", "groups": [group]})


def test_the_seven_kinds_and_no_eighth() -> None:
    assert set(typing.get_args(WidgetKind)) == {
        "setpoint", "toggle", "flipper", "button", "readout", "annotation", "image"
    }  # fmt: skip
    with pytest.raises(ValidationError):
        screen({"kind": "slider", "bind": {"pv": "X:Y"}})


@pytest.mark.parametrize("tone", ["#ff0000", "red", "ok", "var(--accent)"])
def test_a_group_wears_a_kit_token_never_a_colour(tone: str) -> None:
    with pytest.raises(ValidationError):
        screen({"kind": "annotation", "label": "x"}, tone=tone)


@pytest.mark.parametrize(
    "bind",
    [
        {"device": "ps"},
        {"role": "current"},
        {"device": "ps", "role": "current", "pv": "A:B"},
        {"pv": "A:B", "axis": "m1"},
    ],  # fmt: skip
)
def test_a_binding_is_a_device_and_role_or_a_pv(bind: dict) -> None:
    with pytest.raises(ValidationError):
        screen({"kind": "readout", "bind": bind})


def test_resolve_binds_roles_and_their_readbacks() -> None:
    s = screen(
        {
            "kind": "setpoint",
            "label": "X",
            "bind": {"device": "st", "role": "val", "axis": "m1"},
        },  # fmt: skip
        {
            "kind": "toggle",
            "bind": {"device": "ps", "role": "output_sp"},
            "confirm": "Shutter first.",
        },  # fmt: skip
        {"kind": "readout", "bind": {"pv": "U_Gauge:pressure"}},
        FLIP,
    )
    (row,) = resolve_screen(s, load_families(), PANEL, NATIVE)
    assert [m.name for m in row.pop().indicators] == ["output"]
    assert [(b.role and b.role.name, b.readback and b.readback.name) for b in row] == [
        ("val", "rbv"), ("output_sp", "output"), (None, None)
    ]  # fmt: skip
    assert row[1].role.confirm == "hazard"


def test_a_db_device_draws_its_controls_through_bare_pvs() -> None:
    """A DB device's setpoint, toggle and flipper (in/out lamps) bind bare PVs; the renderer keeps them disabled."""
    s = screen(
        {"kind": "setpoint", "label": "Jet X", "bind": {"pv": "x:u_jet:position_axis_1"}},
        {"kind": "toggle", "label": "Slit", "bind": {"pv": "x:u_plc:do_ch2"}, "confirm": "Shutter first."},
        {"kind": "flipper", "label": "Amp2", "bind": {"pv": "x:u_plc:do_ch3"},
         "indicators": [{"pv": "x:u_plc:di_ch5"}, {"device": "ps", "role": "output"}]},
    )  # fmt: skip
    (row,) = resolve_screen(s, load_families(), PANEL, NATIVE)
    assert [b.role for b in row] == [None, None, None]
    assert [m and m.name for m in row[2].indicators] == [None, "output"]


def test_a_group_lays_its_widgets_in_columns() -> None:
    group = {"title": "G", "at": [1, 1], "columns": 3, "widgets": [{"kind": "annotation", "label": "x"}]}  # fmt: skip
    assert (
        Screen.model_validate({"title": "S", "groups": [group]}).groups[0].columns == 3
    )
    with pytest.raises(ValidationError):
        Screen.model_validate({"title": "S", "groups": [group | {"columns": 0}]})


@pytest.mark.parametrize(
    "widget, why",
    [
        (
            {"kind": "setpoint", "bind": {"device": "ps", "role": "current"}},
            "not for a setpoint",
        ),  # fmt: skip
        ({"kind": "readout", "bind": {"device": "ps", "role": "nope"}}, "no role"),
        (
            {"kind": "readout", "bind": {"device": "gone", "role": "current"}},
            "no device",
        ),
        (
            {"kind": "readout", "bind": {"device": "st", "role": "rbv", "axis": "m9"}},
            "axis",
        ),  # fmt: skip
        (FLIP | {"indicators": [{"device": "ps", "role": "reset"}]}, "not readable"),
    ],
)
def test_resolve_refuses_a_misfit(widget: dict, why: str) -> None:
    with pytest.raises(ScreenError, match=why):
        resolve_screen(screen(widget), load_families(), PANEL, NATIVE)


@pytest.mark.parametrize(
    "panel, why",
    [
        ({"st": {"family": "easydriver"}}, "two families|in the registry"),
        ({"ps": {"family": "motor_record"}}, "needs axes"),
    ],
)
def test_resolve_refuses_a_device_the_two_files_disagree_on(panel: dict, why: str) -> None:  # fmt: skip
    name = next(iter(panel))
    widget = {"kind": "readout", "bind": {"device": name, "role": "desc" if name == "ps" else "current"}}  # fmt: skip
    with pytest.raises(ScreenError, match=why):
        resolve_screen(screen(widget), load_families(), PanelDevices.model_validate({"devices": panel}), NATIVE)  # fmt: skip
