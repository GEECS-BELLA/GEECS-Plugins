"""Screen: nine widget kinds, kit tokens, one kind of binding; resolve() fits widgets to roles and groups."""

from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError

from geecs_schemas import (
    AUTO_GROUP,
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


def test_the_nine_kinds_and_no_tenth() -> None:
    assert set(typing.get_args(WidgetKind)) == {
        "setpoint", "toggle", "flipper", "button", "select", "readout", "annotation", "image", "group"
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
        {"action": "park", "pv": "A:B"},
        {"action": "park", "device": "ps"},
        {"group": "Output"},
        {"device": "ps", "role": "current", "group": "Output"},
        {"pv": "A:B", "group": "Output"},
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
    """A writing kind and a flipper's in/out lamps may bind bare PVs; the renderer keeps them disabled."""
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


def test_a_db_device_the_module_lists_is_bound_deferred_by_its_variable_name() -> None:
    lamp = {"device": "U_PLC", "role": "DI.Ch1"}
    flip = {"kind": "flipper", "bind": {"device": "U_PLC", "role": "DO.Ch1"}, "indicators": [lamp]}  # fmt: skip
    ((bound,),) = resolve_screen(screen(flip), load_families(), PANEL, NATIVE, deferred={"U_PLC", "ps"})  # fmt: skip
    assert (bound.deferred, bound.role, bound.indicators) == (True, None, (None,))
    assert bound.fits == (
        "bit",
        "enum",
    )  # the page checks the module's type against these
    ((native,),) = resolve_screen(screen(FLIP), load_families(), PANEL, NATIVE, deferred={"ps"})  # fmt: skip
    assert not native.deferred and native.role.name == "output_sp"  # the files win


def test_a_device_in_neither_the_files_nor_the_roster_is_refused() -> None:
    widget = {
        "kind": "setpoint",
        "bind": {"device": "U_Gone", "role": "Position.Axis 1"},
    }
    with pytest.raises(ScreenError, match="no device 'U_Gone'"):
        resolve_screen(
            screen(widget), load_families(), PANEL, NATIVE, deferred={"U_PLC"}
        )


def test_only_a_button_runs_an_action_and_only_one_the_library_holds() -> None:
    with pytest.raises(ValidationError, match="only a button"):
        screen({"kind": "toggle", "bind": {"action": "park"}})
    runs = screen({"kind": "button", "label": "Park", "bind": {"action": "park"}})
    assert resolve_screen(runs, load_families(), PANEL, NATIVE, actions={"park"})[0][0].role is None  # fmt: skip
    with pytest.raises(ScreenError, match="no action plan 'park'"):
        resolve_screen(runs, load_families(), PANEL, NATIVE, actions={"other"})


CAM = NativeDevices.model_validate({"devices": {"st": {"kind": "epics_motor", "axes": ["m1"]}, "cam": {"kind": "aravis_camera", "trigger_source": "line0"}}})  # fmt: skip
VALUES = {"on": 1, "off": 0}


def test_a_select_binds_an_enum_and_a_valued_toggle_a_numeric_setpoint() -> None:
    s = screen(
        {"kind": "select", "bind": {"device": "cam", "role": "transform"}},
        {"kind": "toggle", "bind": {"device": "ps", "role": "current_sp"}, "values": VALUES, "confirm": "Sure?"},
        {"kind": "flipper", "bind": {"device": "ps", "role": "current_sp"}, "values": VALUES},
        {"kind": "toggle", "bind": OUT, "values": VALUES},  # a bit still fits
        {"kind": "readout", "bind": {"device": "cam", "role": "crosshair1_name"}},  # a text setpoint reads
    )  # fmt: skip
    (row,) = resolve_screen(s, load_families(), PANEL, CAM)
    assert [b.role.name for b in row] == ["transform", "current_sp", "current_sp", "output_sp", "crosshair1_name"]  # fmt: skip


@pytest.mark.parametrize(
    "widget, why",
    [
        ({"kind": "select", "bind": {"device": "ps", "role": "current_sp"}}, "not for a select"),
        ({"kind": "setpoint", "bind": {"device": "cam", "role": "transform"}}, "not for a setpoint"),
        ({"kind": "toggle", "bind": {"device": "ps", "role": "current_sp"}}, "not for a toggle"),
        ({"kind": "toggle", "bind": {"device": "cam", "role": "crosshair1_name"}, "values": VALUES}, "is text"),
    ],
)  # fmt: skip
def test_a_select_takes_only_an_enum_and_a_toggle_a_setpoint_only_with_values(widget: dict, why: str) -> None:  # fmt: skip
    with pytest.raises(ScreenError, match=why):
        resolve_screen(screen(widget), load_families(), PANEL, CAM)


@pytest.mark.parametrize(
    "widget",
    [
        {"kind": "setpoint", "bind": OUT, "values": VALUES},
        {"kind": "toggle", "bind": OUT, "values": {"on": 1, "off": 1}},
        {"kind": "toggle", "bind": OUT, "values": {"on": 1}},
        {"kind": "toggle", "bind": OUT, "values": {"on": float("nan"), "off": 0}},
    ],
)
def test_values_are_two_numbers_on_a_toggle_or_flipper(widget: dict) -> None:
    with pytest.raises(ValidationError):
        screen(widget)


def test_a_deferred_valued_toggle_fits_a_setpoint_and_a_select_an_enum() -> None:
    s = screen(
        {"kind": "toggle", "bind": {"device": "U_Gaia", "role": "InternalShutterA"}, "values": VALUES},
        {"kind": "select", "bind": {"device": "U_Filters", "role": "pos.Channel0"}},
        {"kind": "toggle", "bind": {"device": "U_Gaia", "role": "InternalShutterB"}},
    )  # fmt: skip
    (row,) = resolve_screen(s, load_families(), PANEL, NATIVE, deferred={"U_Gaia", "U_Filters"})  # fmt: skip
    assert [b.fits for b in row] == [("bit", "enum", "setpoint"), ("enum",), ("bit", "enum")]  # fmt: skip


def test_values_read_from_yaml_take_its_on_and_off_booleans_as_names() -> None:
    import yaml

    widget = yaml.safe_load("{kind: toggle, bind: {device: ps, role: current_sp}, values: {on: 1, off: 0}}")  # fmt: skip
    assert screen(widget).groups[0].widgets[0].values.model_dump() == {"on": 1.0, "off": 0.0}  # fmt: skip


@pytest.mark.integration
def test_every_screen_in_the_configs_repo_validates_and_binds(configs_repo) -> None:
    """The deployed screens (the HTU Control Center among them) against these models, their DB devices deferred."""
    from geecs_schemas import NativeDevices, PanelDevices, load_families, resolve_screen

    experiments = configs_repo / "scanner_configs" / "experiments"
    found = sorted(experiments.glob("*/screens/*.yaml"))
    assert found, "no screen in the configs checkout"
    for path in found:
        folder = path.parent.parent
        screen = Screen.from_path(path)
        panel = PanelDevices.from_path(folder / "panel_devices.yaml") if (folder / "panel_devices.yaml").is_file() else PanelDevices()  # fmt: skip
        native = NativeDevices.from_path(folder / "native_devices.yaml") if (folder / "native_devices.yaml").is_file() else NativeDevices.model_validate({})  # fmt: skip
        bound = {b.device for g in screen.groups for w in g.widgets for b in (w.bind, *w.indicators) if b and b.device}  # fmt: skip
        rows = resolve_screen(screen, load_families(), panel, native, deferred=bound - set(panel.devices) - set(native.devices))  # fmt: skip
        assert sum(map(len, rows)) == sum(len(g.widgets) for g in screen.groups), (
            path.name
        )


GROUP = {"kind": "group", "bind": {"device": "ps", "group": "Output"}}


def test_a_group_binds_a_device_and_a_rail_group_of_its_family_panel() -> None:
    motor = {"kind": "group", "bind": {"device": "st", "group": "Move", "axis": "m1"}}
    db = {"kind": "group", "label": "Plunger", "bind": {"device": "U_PLC", "group": AUTO_GROUP}}  # fmt: skip
    rows = resolve_screen(screen(GROUP, motor, db), load_families(), PANEL, NATIVE, deferred={"U_PLC"})  # fmt: skip
    assert [(b.role, b.deferred) for b in rows[0]] == [(None, False), (None, False), (None, True)]  # fmt: skip


@pytest.mark.parametrize(
    "bind, why",
    [
        ({"device": "ps", "group": "output"}, r"no group 'output': its groups are \['Output', 'Faults'\]"),
        ({"device": "ps", "group": AUTO_GROUP}, "no group 'settings'"),  # a family with a panel draws it
        ({"device": "st", "group": "Move"}, "takes an axis"),
        ({"device": "st", "group": "Move", "axis": "m9"}, "takes an axis"),
        ({"device": "U_PLC", "group": "Output"}, "a DB device: its one group is 'settings'"),
        ({"device": "U_Gone", "group": AUTO_GROUP}, "no device 'U_Gone'"),
    ],
)  # fmt: skip
def test_resolve_refuses_a_group_its_device_does_not_draw(bind: dict, why: str) -> None:
    with pytest.raises(ScreenError, match=why):
        resolve_screen(screen({"kind": "group", "bind": bind}), load_families(), PANEL, NATIVE, deferred={"U_PLC"})  # fmt: skip


def test_a_family_without_a_panel_offers_the_automatic_layout_alone() -> None:
    families = dict(load_families())
    families["easydriver"] = families["easydriver"].model_copy(update={"panel": None})
    ((bound,),) = resolve_screen(screen(GROUP | {"bind": {"device": "ps", "group": AUTO_GROUP}}), families, PANEL, NATIVE)  # fmt: skip
    assert not bound.deferred
    with pytest.raises(ScreenError, match=r"its groups are \['settings'\]"):
        resolve_screen(screen(GROUP), families, PANEL, NATIVE)


@pytest.mark.parametrize(
    "widget, why",
    [
        ({"kind": "readout", "bind": {"device": "ps", "group": "Output"}}, "only a group"),
        ({"kind": "group", "bind": {"device": "ps", "role": "current"}}, "only a group"),
        (FLIP | {"indicators": [{"device": "ps", "group": "Output"}]}, "only a group"),
        (GROUP | {"units": "A"}, "units and precision"),
        (GROUP | {"precision": 2}, "units and precision"),
        (GROUP | {"confirm": "Sure?"}, "nothing to confirm"),
    ],
)  # fmt: skip
def test_only_a_group_binds_a_group_and_takes_nothing_of_a_controls(widget: dict, why: str) -> None:  # fmt: skip
    with pytest.raises(ValidationError, match=why):
        screen(widget)
