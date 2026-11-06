"""FamilyFile: the installed families load, a tiny family round-trips, every rule bites."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY, FamilyFile, NativeDevice, load_families
from geecs_schemas.epics import family as family_module
from geecs_schemas.epics.family import FamilyKind

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
        {"name": "flag", "suffix": "Compute", "type": "bit", "label": "computing"},
        {"name": "mean", "suffix": "Mean_RBV", "type": "readback", "label": "mean",
         "fresh_by": "counter", "computed_by": "flag"},
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
        "level": {"label": "level", "roles": ["level_sp"]},
        "acquire": {"label": "acquire", "roles": ["acquire"]},
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
    assert family.role_names()[:3] == ["counter", "flag", "mean"]


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
    role(doc, "mean")["computed_by"] = "flg"
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
    role(doc, "flag").update(write="Compute:SP")
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


def test_with_actions_every_writable_role_belongs_to_one() -> None:
    doc = tiny()
    del doc["actions"]["level"]
    _refused(doc, r"writable role\(s\) \['level_sp'\] belong to no action")
    doc["actions"] = {}  # without actions, each writable role is its own permission
    FamilyFile.model_validate(doc)


@pytest.mark.parametrize("word", ["off", "on", "yes", "no", "true", "null"])
def test_a_role_name_is_not_a_yaml_word(word: str) -> None:
    doc = tiny()
    role(doc, "level")["name"] = word
    role(doc, "level_sp")["readback"] = word
    _refused(doc, "reads as a boolean or null in YAML")


def test_the_installed_families_cannot_be_changed_in_place() -> None:
    families = load_families()
    with pytest.raises(TypeError):
        families["tiny"] = families["fastps"]  # type: ignore[index]
    with pytest.raises(ValidationError, match="frozen"):
        families["fastps"].role("current").label = "amps"


def test_every_registry_kind_has_a_family_kind() -> None:
    registry_kinds = {
        member.model_fields["kind"].annotation.__args__[0]
        for member in NativeDevice.__args__[0].__args__
    }
    assert registry_kinds <= set(FamilyKind.__args__)


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
        "spectrometer_contract",
    }
    assert families["motor_record"].per_axis
    assert families["ad_camera_chain"].kind == "aravis_camera"
    camera = families["ad_camera_chain"]
    # the raw frame's source is never read by the bridge (its own route reads it); its geometry
    # and counter travel with it on request: none is monitored, judged or archived
    assert camera.role("raw_frame").read_mode == "never" and camera.role("raw_frame").type == "array"  # fmt: skip
    assert {camera.role(r).read_mode for r in ("raw_width", "raw_height", "raw_counter")} == {"on_request"}  # fmt: skip
    assert not any(camera.role(r).archive for r in ("raw_frame", "raw_width", "raw_height", "raw_counter"))  # fmt: skip


def test_the_installed_families_mark_their_hazards() -> None:
    families = load_families()
    hazards = {
        name: sorted(r.name for r in f.roles if r.confirm == "hazard")
        for name, f in families.items()
    }
    assert hazards["motor_record"] == ["homf", "homr"]
    assert hazards["fastps"] == hazards["easydriver"] == ["output_sp"]
    assert hazards["scope_contract"] == hazards["spectrometer_contract"] == []


def test_autosaved_round_trips_and_defaults_false() -> None:
    doc = tiny()
    role(doc, "level_sp")["autosaved"] = True
    family = FamilyFile.model_validate(doc)
    assert family.role("level_sp").autosaved and not family.role("fire").autosaved
    assert FamilyFile.model_validate(family.model_dump(mode="json")) == family


def test_the_installed_families_mark_what_autosave_restores() -> None:
    families = load_families()
    saved = {n: {r.name for r in f.roles if r.autosaved} for n, f in families.items()}
    # The camera pins its trigger mode at boot; motors and supplies run no autosave.
    assert {"exposure", "roi_minx"} <= saved["ad_camera_chain"]
    assert "trigger_mode" not in saved["ad_camera_chain"]
    assert {"trigger_level", "ch_a_range"} <= saved["scope_contract"]
    # the spectrometer IOC writes nothing at boot: its output records take the board's values
    assert saved["motor_record"] == saved["fastps"] == saved["easydriver"] == saved["spectrometer_contract"] == set()  # fmt: skip


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


# ---- an array's axis, the panel ---------------------------------------------


def _with_axis(**axis: object) -> dict:
    doc = tiny()
    doc["roles"].append(
        {"name": "x", "suffix": "X", "type": "array", "label": "x", **axis}
    )
    return doc


def test_an_axis_names_its_array() -> None:
    family = FamilyFile.model_validate(_with_axis(axis_of="trace"))
    assert family.role("x").axis_of == ["trace"]  # one name is a list of one
    assert family.role("trace").axis_of is None


def _with_second_array() -> dict:
    doc = tiny()
    doc["roles"].append({"name": "trace2", "suffix": "Trace2", "type": "array", "label": "trace 2"})  # fmt: skip
    return doc


def test_an_axis_draws_several_arrays() -> None:
    doc = _with_second_array()
    doc["roles"].append({"name": "x", "suffix": "X", "type": "array", "label": "x", "axis_of": ["trace", "trace2"]})  # fmt: skip
    family = FamilyFile.model_validate(doc)
    assert family.role("x").axis_of == ["trace", "trace2"]
    assert FamilyFile.model_validate(family.model_dump(mode="json")) == family
    for bad, match in (([], "at least 1 item"), (["trace", "level"], "axis_of names an array other than itself"),
                       (["trace", "trac"], "axis_of names no role")):  # fmt: skip
        role(doc, "x")["axis_of"] = bad
        _refused(doc, match)


def test_an_array_of_a_list_still_has_one_axis() -> None:
    doc = _with_second_array()
    doc["roles"] += [
        {"name": "x", "suffix": "X", "type": "array", "label": "x", "axis_of": ["trace", "trace2"]},
        {"name": "x2", "suffix": "X2", "type": "array", "label": "x2", "axis_of": "trace2"},
    ]  # fmt: skip
    _refused(doc, r"axes \(one per array\) must be unique; repeated: \['trace2'\]")


def test_an_axis_names_a_role() -> None:
    _refused(_with_axis(axis_of="trac"), "axis_of names no role")


def test_an_axis_is_of_an_array() -> None:
    _refused(_with_axis(axis_of="level"), "axis_of names an array other than itself")
    _refused(_with_axis(axis_of="x"), "axis_of names an array other than itself")


def test_only_an_array_is_an_axis() -> None:
    doc = tiny()
    role(doc, "level")["axis_of"] = "trace"
    _refused(doc, "only an array is an axis")


def test_an_array_has_one_axis() -> None:
    doc = _with_axis(axis_of="trace")
    doc["roles"].append(
        {
            "name": "x2",
            "suffix": "X2",
            "type": "array",
            "label": "x2",
            "axis_of": "trace",
        }
    )
    _refused(doc, r"axes \(one per array\) must be unique; repeated: \['trace'\]")


def test_an_axis_is_no_arrays_trace() -> None:
    doc = _with_axis(axis_of="trace")
    role(doc, "trace")["axis_of"] = "x"  # each the other's axis: which is the trace?
    _refused(doc, "an axis is no array's trace")


def test_a_channel_count_is_an_arrays_and_a_readback() -> None:
    doc = tiny()
    role(doc, "trace")["channels_by"] = "level"
    assert FamilyFile.model_validate(doc).role("trace").channels_by == "level"
    for owner, target in (("trace", "counter"), ("trace", "lvl"), ("level", "level")):
        doc = tiny()
        role(doc, owner)["channels_by"] = target
        _refused(doc, "channels_by is an array's, naming a readback")


PANEL = {
    "rail": [
        {"title": "Run", "roles": ["acquire", "level_sp"]},
        {"title": "Safety", "roles": ["fire", "stop", "fault"]},
    ],
    "readings": ["mean", "counter", "fault"],
}


def test_a_panel_round_trips_and_takes_any_role_type() -> None:
    family = FamilyFile.model_validate({**tiny(), "panel": PANEL})
    assert [g.title for g in family.panel.rail] == ["Run", "Safety"]
    assert family.panel.readings == ["mean", "counter", "fault"]
    assert FamilyFile.model_validate(family.model_dump(mode="json")) == family
    assert FamilyFile.model_validate(tiny()).panel is None


@pytest.mark.parametrize("where", ["rail", "readings"])
def test_a_panel_names_family_roles(where: str) -> None:
    panel = copy.deepcopy(PANEL)
    if where == "rail":
        panel["rail"][1]["roles"].append("halt")
    else:
        panel["readings"].append("halt")
    _refused({**tiny(), "panel": panel}, r"panel names no role \['halt'\]")


def test_a_reading_is_monitored_and_listed_once() -> None:
    for reading, match in (("trace", "panel readings \\['trace'\\] are not monitored"),
                           ("mean", "panel readings must be unique")):  # fmt: skip
        panel = copy.deepcopy(PANEL)
        panel["readings"].append(reading)
        _refused({**tiny(), "panel": panel}, match)


def _buttons(**over: object) -> dict:
    doc = {**tiny(), "panel": {**copy.deepcopy(PANEL), **over}}
    role(doc, "acquire").update(integer=True, limits=[0, 1])
    return doc


def test_a_two_state_rail_role_may_be_drawn_as_a_button_pair() -> None:
    family = FamilyFile.model_validate(_buttons(buttons={"acquire": ["Stop", "Start"]}))
    assert family.panel.buttons == {"acquire": ("Stop", "Start")}
    assert FamilyFile.model_validate(family.model_dump(mode="json")) == family
    assert FamilyFile.model_validate({**tiny(), "panel": PANEL}).panel.buttons == {}


@pytest.mark.parametrize(
    "name, change",
    [
        ("level_sp", None),  # a setpoint limited to [0, 10]: not two-state
        ("fault", None),  # a bit, but read only and not in the rail
        ("acquire", lambda doc: role(doc, "acquire").pop("limits")),  # no [0, 1] limits
        ("acquire", lambda doc: doc["panel"]["rail"][0]["roles"].remove("acquire")),  # not in the rail
    ],
)  # fmt: skip
def test_a_button_pair_draws_a_writable_two_state_rail_role(name, change) -> None:
    doc = _buttons(buttons={name: ["Off", "On"]})
    if change:
        change(doc)
    _refused(doc, f"panel buttons {name!r}: a button pair draws a writable two-state rail role")  # fmt: skip


def test_a_button_pair_has_two_labels() -> None:
    for labels in (["Stop"], ["Stop", "Start", "Pause"], ["", "Start"]):
        _refused(_buttons(buttons={"acquire": labels}), "buttons")
    _refused(
        _buttons(buttons={"halt": ["Off", "On"]}), r"panel names no role \['halt'\]"
    )


def test_a_default_array_is_one_an_axis_draws_while_a_monitored_role_reads_1() -> None:
    doc = _with_second_array()
    doc["roles"].append({"name": "x", "suffix": "X", "type": "array", "label": "x", "axis_of": ["trace", "trace2"]})  # fmt: skip
    family = FamilyFile.model_validate({**doc, "panel": {"show_while": {"trace2": "flag"}}})  # fmt: skip
    assert family.panel.show_while == {"trace2": "flag"}
    _refused({**doc, "panel": {"show_while": {"x": "flag"}}}, "panel show_while 'x': no axis draws it")  # fmt: skip
    _refused({**doc, "panel": {"show_while": {"trace2": "fire"}}}, "'fire' is not monitored")  # fmt: skip
    _refused({**doc, "panel": {"show_while": {"trace2": "flg"}}}, r"panel names no role \['flg'\]")  # fmt: skip


def test_a_rail_role_sits_in_one_group() -> None:
    panel = copy.deepcopy(PANEL)
    panel["rail"][1]["roles"].append("acquire")
    _refused({**tiny(), "panel": panel}, "panel rail roles must be unique")


def _tweak(**over: object) -> dict:
    """TINY with a step setpoint and two step commands beside level_sp in its group."""
    doc = tiny()
    doc["roles"] += [
        {"name": "step", "suffix": "step", "type": "setpoint", "label": "step", "writable": True},
        {"name": "up", "suffix": "up", "type": "command", "label": "up", "writable": True},
        {"name": "down", "suffix": "down", "type": "command", "label": "down", "writable": True},
    ]  # fmt: skip
    doc["actions"]["level"]["roles"] += ["step", "up", "down"]
    panel = copy.deepcopy(PANEL)
    panel["rail"][0]["roles"] += ["step", "up", "down"]
    panel["tweak"] = {"level_sp": {"step": "step", "forward": "up", "reverse": "down"}}
    doc["panel"] = {**panel, **over}
    return doc


def test_a_setpoint_may_be_drawn_with_its_step_and_step_commands() -> None:
    family = FamilyFile.model_validate(_tweak())
    hint = family.panel.tweak["level_sp"]
    assert (hint.step, hint.forward, hint.reverse) == ("step", "up", "down")
    assert FamilyFile.model_validate(family.model_dump(mode="json")) == family
    assert FamilyFile.model_validate({**tiny(), "panel": PANEL}).panel.tweak == {}


@pytest.mark.parametrize(
    "hint, match",
    [
        ({"step": "step", "forward": "up", "reverse": "lvl"}, r"panel names no role \['lvl'\]"),
        ({"step": "level", "forward": "up", "reverse": "down"}, r"'level' must be a rail setpoint \(a readback off the rail\)"),
        ({"step": "step", "forward": "step", "reverse": "down"}, r"'step' must be a rail command \(a setpoint\)"),
        ({"step": "step", "forward": "up", "reverse": "up"}, r"panel tweak roles must be unique; repeated: \['up'\]"),
        ({"step": "step", "forward": "fire", "reverse": "down"}, "'fire' sits in the setpoint's rail group"),
    ],
)  # fmt: skip
def test_a_tweak_names_a_step_setpoint_and_two_commands_of_its_group(
    hint, match
) -> None:
    _refused(_tweak(tweak={"level_sp": hint}), match)


def test_a_tweak_is_one_group_not_one_heading() -> None:
    """Two groups may share a heading: the tweak's group is the group itself, never its title."""
    doc = _tweak()
    doc["panel"]["rail"][0]["roles"].remove("up")
    doc["panel"]["rail"].append(
        {"title": "Run", "roles": ["up"]}
    )  # the setpoint's heading, another group
    _refused(
        doc,
        r"'up' sits in the setpoint's rail group \('Run', group 1\), not \('Run', group 3\)",
    )


def test_a_tweaked_setpoint_and_its_roles_are_on_the_rail() -> None:
    doc = _tweak()
    doc["panel"]["rail"][0]["roles"].remove("down")
    _refused(doc, r"'down' must be a rail command \(a command off the rail\)")
    doc = _tweak(
        tweak={"acquire": {"step": "step", "forward": "up", "reverse": "down"}}
    )
    _refused(
        doc, r"panel tweak 'acquire': 'acquire' must be a rail setpoint \(an enum\)"
    )


def test_a_role_is_drawn_by_one_hint() -> None:
    doc = _tweak(buttons={"up": ["Off", "On"]})
    _refused(
        doc, "a button pair draws a writable two-state rail role"
    )  # a command is no two-state role
    doc = _tweak()
    role(doc, "step").update(integer=True, limits=[0, 1])
    doc["panel"]["buttons"] = {"step": ["Off", "On"]}
    _refused(doc, r"panel roles \['step'\] are drawn by a tweak and a button pair")


def test_a_grid_group_draws_bits_and_enums() -> None:
    panel = copy.deepcopy(PANEL)
    panel["rail"] = [
        {"title": "Run", "roles": ["acquire", "flag"], "layout": "grid"},
        {"title": "Safety", "roles": ["fire", "stop"]},
    ]
    family = FamilyFile.model_validate({**tiny(), "panel": panel})
    assert [g.layout for g in family.panel.rail] == ["grid", "fields"]
    assert FamilyFile.model_validate(family.model_dump(mode="json")) == family
    panel["rail"][1]["layout"] = "grid"
    _refused({**tiny(), "panel": panel}, r"panel group 'Safety' \(group 2\): a grid draws short controls only \(bits, enums, two-state whole numbers\), not \['fire', 'stop'\]")  # fmt: skip
    # a two-state whole number is short, as a button pair counts it; a whole number over a wider span is not
    doc = {
        **tiny(),
        "panel": {"rail": [{"title": "Run", "roles": ["level_sp"], "layout": "grid"}]},
    }
    _refused(doc, r"not \['level_sp'\]")
    role(doc, "level_sp").update(integer=True, limits=[0, 1])
    assert FamilyFile.model_validate(doc).panel.rail[0].layout == "grid"
    panel["rail"][1]["layout"] = "table"
    _refused({**tiny(), "panel": panel}, "layout")


def _railed_is_writable(family: FamilyFile) -> None:
    """The rail is exactly the family's writable roles; every reading is monitored."""
    railed = [r for g in family.panel.rail for r in g.roles]
    assert sorted(railed) == sorted(r.name for r in family.roles if r.writable)
    assert all(family.role(r).read_mode == "monitor" for r in family.panel.readings)


def test_the_motor_record_draws_its_target_with_a_tweak() -> None:
    motor = load_families()["motor_record"]
    _railed_is_writable(motor)
    assert [g.title for g in motor.panel.rail] == ["Move", "Home", "Speed"]
    hint = motor.panel.tweak["val"]
    assert (hint.step, hint.forward, hint.reverse) == ("twv", "twf", "twr")
    # position, done moving, the limit switches and the soft-limit violation
    assert motor.panel.readings == ["rbv", "dmov", "hls", "lls", "lvio"]
    # the action groups stay the write permissions the page's hints do not replace
    assert motor.actions["tweak"].roles == ["twv", "twf", "twr"]


@pytest.mark.parametrize("name", ["easydriver", "fastps"])
def test_the_supplies_draw_output_as_a_button_pair(name: str) -> None:
    supply = load_families()[name]
    _railed_is_writable(supply)
    assert [g.title for g in supply.panel.rail] == ["Output", "Faults"]
    assert supply.panel.buttons == {
        "output_sp": ("Off", "On")
    }  # 0 then 1, behind its hazard confirm
    assert supply.role("output_sp").confirm == "hazard"
    faults = [r.name for r in supply.roles if r.fault]
    assert supply.panel.readings[:3] == ["current", "voltage", "output"]
    assert set(faults) <= set(supply.panel.readings)


def test_the_scope_contract_draws_its_capture_over_its_time_axis() -> None:
    scope = load_families()["scope_contract"]
    assert scope.role("time_axis").axis_of == ["data"]
    # [channels, samples] by arr_channels, held on the ARR plugin's own counter (it publishes after the driver's)
    assert (scope.role("data").channels_by, scope.role("data").fresh_by) == ("arr_channels", "arr_counter")  # fmt: skip
    assert scope.role("arr_counter").suffix == "ARR:ArrayCounter_RBV"
    assert [g.title for g in scope.panel.rail] == [
        "Acquisition", "Trigger", "Channel A", "Channel B", "Channel C", "Channel D",
    ]  # fmt: skip
    assert scope.panel.rail[2].roles == [
        "ch_a_enable", "ch_a_range", "ch_a_coupling", "ch_a_offset",
    ]  # fmt: skip
    # the detector's health beside its Start button: the state word, the line, the driver's reason
    assert scope.panel.readings[:4] == ["state", "connected", "status", "count"]
    assert scope.panel.readings[-1] == "last_error"
    # the rail is exactly the family's writable roles: none left off, no read-only one placed
    railed = [r for g in scope.panel.rail for r in g.roles]
    assert sorted(railed) == sorted(r.name for r in scope.roles if r.writable)
    assert scope.panel.buttons == {"acquire": ("Stop", "Start")}  # 0 then 1


def test_the_spectrometer_contract_draws_its_spectrum_over_its_wavelengths() -> None:
    spec = load_families()["spectrometer_contract"]
    assert spec.kind == "array_detector"
    # the raw spectrum (ARR, on the driver's port) first, then the background-subtracted one (PROC's own ARR)
    assert (
        spec.role("wavelength").axis_of == ["spectrum", "processed"]
        and spec.role("wavelength").units == "nm"
    )
    # one channel ([pixels]) each, held on its own plugin's counter, as the scope's capture is
    assert (spec.role("spectrum").channels_by, spec.role("spectrum").fresh_by) == (None, "arr_counter")  # fmt: skip
    assert (spec.role("processed").channels_by, spec.role("processed").fresh_by) == (None, "proc_counter")  # fmt: skip
    assert spec.role("arr_counter").suffix == "ARR:ArrayCounter_RBV"
    assert (spec.role("processed").suffix, spec.role("proc_counter").suffix) == ("PROC:ARR:ArrayData", "PROC:ARR:ArrayCounter_RBV")  # fmt: skip
    # the page shows the processed spectrum while subtraction is on; acquire is a Stop / Start pair
    assert spec.panel.show_while == {"processed": "background_enable"}
    assert spec.panel.buttons == {"acquire": ("Stop", "Start")}
    assert [g.title for g in spec.panel.rail] == [
        "Acquisition",
        "Trigger",
        "Background",
    ]
    assert spec.panel.readings[:4] == ["state", "connected", "status", "count"]
    assert spec.panel.readings[-1] == "last_error"
    # the rail is exactly the family's writable roles: none left off, no read-only one placed
    railed = [r for g in spec.panel.rail for r in g.roles]
    assert sorted(railed) == sorted(r.name for r in spec.roles if r.writable)
    # saturation is read as the record's own word (OK / Saturated), not a fault bit of the
    # family's; its alarm severity is the IOC's, and the module's verdict follows that
    assert spec.role("saturated").type == "enum" and not spec.role("saturated").fault
    # the statistics are NDStats' over PROC: fresh by that plugin's own counter, as the camera's are
    stats = [r for r in spec.roles if r.suffix.startswith("STAT1:") and r.type not in ("counter", "bit")]  # fmt: skip
    assert len(stats) == 8 and {r.fresh_by for r in stats} == {"stat_counter"}
    assert spec.role("stat_counter").suffix == "STAT1:ArrayCounter_RBV"
    # and each bound to the flag that computes it (off: the value reads 'off', never fresh)
    assert {r.name: r.computed_by for r in stats} == {
        "total": "compute_statistics", "max": "compute_statistics", "min": "compute_statistics",
        "mean": "compute_statistics", "sigma": "compute_statistics", "peak_index": "compute_statistics",
        "centroid_index": "compute_centroid", "width_index": "compute_centroid",
    }  # fmt: skip
    assert (spec.role("compute_statistics").suffix, spec.role("compute_centroid").suffix) == ("STAT1:ComputeStatistics", "STAT1:ComputeCentroid")  # fmt: skip
    # the background is the chain's (NDProcess), never the instrument's dark command
    assert spec.role("background_save").type == "command" and spec.role("background_save").read_mode == "never"  # fmt: skip


def test_the_two_array_detector_families_share_their_vocabulary() -> None:
    """A convention, pinned so a divergence is a decision: where the two families serve the same thing, they name it alike."""
    scope, spec = (
        load_families()["scope_contract"],
        load_families()["spectrometer_contract"],
    )
    shared = {"connected", "state", "status", "last_error", "acquire", "image_mode", "num_images", "trigger_mode", "count", "arr_counter", "arr_uid", "model", "serial", "manufacturer", "firmware"}  # fmt: skip
    assert shared <= {r.name for r in scope.roles} and shared <= {
        r.name for r in spec.roles
    }
    for name in shared:
        assert (scope.role(name).type, scope.role(name).suffix) == (spec.role(name).type, spec.role(name).suffix), name  # fmt: skip
