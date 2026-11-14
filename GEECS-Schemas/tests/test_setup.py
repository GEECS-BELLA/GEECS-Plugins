"""The setup document: steps, nested applies, and the check's comparison."""

from decimal import Decimal

import pytest
import yaml
from pydantic import ValidationError

from geecs_schemas import (
    NativeDevices,
    OperatorPolicy,
    PanelDevices,
    Setup,
    SetupCycleError,
    SetupNotFoundError,
    flatten_setup,
    load_families,
    load_setups,
    setup_digest,
    setup_hazards,
    setup_value_matches,
)


def _setup(name, *steps):
    return Setup.model_validate({"name": name, "steps": list(steps)})


SET = {"do": "set", "device": "U_S1H", "variable": "Current", "value": 1.5}


def test_a_captured_setup_round_trips():
    doc = Setup.model_validate(
        {
            "name": "low_power",
            "steps": [
                SET,
                {"do": "wait", "seconds": 2},
                {
                    "do": "check",
                    "device": "U_S1H",
                    "variable": "Current",
                    "expected": 1.5,
                    "tolerance": 0.01,
                },
                {"do": "apply", "setup": "shutters_closed"},
            ],
            "captured": {
                "by": "A. Operator",
                "at": "2026-11-07T18:02:00-08:00",
                "readbacks": [{"device": "U_S1H", "variable": "Current"}],
            },
        }
    )
    assert Setup.model_validate(doc.model_dump(mode="json")) == doc


@pytest.mark.parametrize(
    "step",
    [
        {"do": "run", "plan": "other"},  # the action library's verb, not a setup's
        {
            "do": "set",
            "device": "U_S1H",
            "variable": "Current",
            "value": 1,
            "wait_for_execution": False,
        },
        {"do": "wait", "seconds": 0},
    ],
)
def test_an_unknown_or_malformed_step_is_refused(step):
    with pytest.raises(ValidationError):
        _setup("s", step)


def test_a_word_check_takes_no_tolerance():
    with pytest.raises(ValidationError, match="matches exactly"):
        _setup(
            "s",
            {
                "do": "check",
                "device": "D",
                "variable": "V",
                "expected": "on",
                "tolerance": 1,
            },
        )


@pytest.mark.parametrize(
    "name", ["", "../x", "a/b", "low power", "x.yaml", "x.yml", "x.YAML"]
)
def test_the_name_is_a_file_stem(name):
    with pytest.raises(ValidationError):
        _setup(name, SET)


def test_an_empty_setup_is_refused():
    with pytest.raises(ValidationError):
        _setup("s")


def test_a_setup_applying_itself_is_refused():
    with pytest.raises(ValidationError, match="loop"):
        _setup("s", {"do": "apply", "setup": "s"})


def test_flatten_inlines_nested_setups_in_order():
    inner = _setup("inner", {"do": "wait", "seconds": 1})
    outer = _setup("outer", SET, {"do": "apply", "setup": "inner"}, SET)
    steps = flatten_setup(outer, setups={"inner": inner, "outer": outer})
    assert [(s.do, origin) for s, origin in steps] == [
        ("set", None),
        ("wait", "inner"),
        ("set", None),
    ]


def test_flatten_refuses_a_cycle():
    a = _setup("a", {"do": "apply", "setup": "b"})
    b = _setup("b", {"do": "apply", "setup": "a"})
    with pytest.raises(SetupCycleError) as caught:
        flatten_setup(a, setups={"a": a, "b": b})
    assert caught.value.chain == ["a", "b", "a"]


def test_flatten_refuses_an_unknown_setup():
    a = _setup("a", {"do": "apply", "setup": "missing"})
    with pytest.raises(SetupNotFoundError):
        flatten_setup(a, setups={"a": a})


@pytest.mark.parametrize(
    ("expected", "actual", "tolerance", "matches"),
    [
        (1.2, 1.15, 0.05, True),  # exactly on tolerance, despite representation error
        (1.2, 1.14, 0.05, False),
        (1.5, "1.5", 0.0, True),  # a numeric string reading parses
        (2, 2.0, 0.0, True),
        (1.0, float("nan"), 1.0, False),
        (1.0, "on", 1.0, False),
        (1, True, 0.0, True),  # floated, as the confirm settable floats it
        (2, Decimal("2.0"), 0.0, True),  # any number float() takes
        (1.0, None, 1.0, False),
        ("on", "on", 0.0, True),
        ("on", "ON", 0.0, False),  # a word matches exactly
        ("25", 25, 0.0, False),  # a numeric-looking word is still a word
        ("25", "25", 0.0, True),
    ],
)
def test_check_comparison(expected, actual, tolerance, matches):
    assert setup_value_matches(expected, actual, tolerance) is matches


@pytest.mark.parametrize("word", ["on", "off", "yes", "no", "true"])
def test_an_unquoted_yaml_word_is_refused_with_the_fix(word):
    """YAML 1.1 reads a bare on/off as a boolean; the step refuses it, never writes 1.0."""
    step = yaml.safe_load(f"{{do: set, device: U_X, variable: State, value: {word}}}")
    with pytest.raises(ValidationError, match="quote the word"):
        _setup("s", step)
    check = yaml.safe_load(
        f"{{do: check, device: U_X, variable: State, expected: {word}}}"
    )
    with pytest.raises(ValidationError, match="quote the word"):
        _setup("s", check)


def test_values_keep_their_type():
    """A quoted word, an int and a float each stay as written."""
    doc = _setup(
        "s",
        {"do": "set", "device": "D", "variable": "V", "value": "on"},
        {"do": "set", "device": "D", "variable": "V", "value": 2},
        {"do": "set", "device": "D", "variable": "V", "value": 2.5},
        {"do": "set", "device": "D", "variable": "V", "value": "2"},
    )
    assert [type(step.value) for step in doc.steps] == [str, int, float, str]


def test_flatten_refuses_a_setup_listed_under_another_name():
    a = _setup("a", SET)
    with pytest.raises(ValueError, match="another name"):
        flatten_setup(a, setups={"b": a})


def _write(folder, filename, name):
    (folder / filename).write_text(f"name: {name}\nsteps: [{{do: wait, seconds: 1}}]\n")


def test_load_setups_reads_both_suffixes(tmp_path):
    folder = tmp_path / "setups"
    folder.mkdir()
    _write(folder, "low_power.yaml", "low_power")
    _write(folder, "high_power.yml", "high_power")
    (folder / "notes.txt").write_text("not a setup")
    assert sorted(load_setups(tmp_path)) == ["high_power", "low_power"]


def test_load_setups_refuses_a_name_that_is_not_the_stem(tmp_path):
    folder = tmp_path / "setups"
    folder.mkdir()
    _write(folder, "low_power.yaml", "lowpower")
    with pytest.raises(ValueError, match="file stem"):
        load_setups(tmp_path)


def test_load_setups_refuses_a_stem_under_both_suffixes(tmp_path):
    folder = tmp_path / "setups"
    folder.mkdir()
    _write(folder, "a.yaml", "a")
    _write(folder, "a.yml", "a")
    with pytest.raises(ValueError, match="two files"):
        load_setups(tmp_path)


def test_a_missing_setups_folder_is_empty_and_never_created(tmp_path):
    assert load_setups(tmp_path) == {}
    assert not (tmp_path / "setups").exists()


# ---------------------------------------------------------------- the digest


def test_the_digest_follows_what_runs_nested_setups_included():
    inner = _setup("inner", SET)
    outer = _setup("outer", {"do": "apply", "setup": "inner"}, {"do": "wait", "seconds": 1})  # fmt: skip
    before = setup_digest(flatten_setup(outer, setups={"inner": inner}))
    assert before == setup_digest(flatten_setup(outer, setups={"inner": inner}))
    changed = _setup("inner", {**SET, "value": 1.6})
    assert before != setup_digest(flatten_setup(outer, setups={"inner": changed}))
    # The same steps written in the outer setup itself are another document.
    flat = _setup("outer", SET, {"do": "wait", "seconds": 1})
    assert before != setup_digest(flatten_setup(flat, setups={}))
    # A value's type is part of it: 1 and 1.0 are different writes.
    one = _setup("x", {**SET, "value": 1})
    one_float = _setup("x", {**SET, "value": 1.0})
    assert setup_digest(flatten_setup(one, setups={})) != setup_digest(
        flatten_setup(one_float, setups={})
    )


# --------------------------------------------------------------- the hazards


def _hazards(*steps, policy=None, native=None, panels=None, db_families=None):
    setup = _setup("s", *steps)
    return setup_hazards(
        flatten_setup(setup, setups={}),
        policy=OperatorPolicy.model_validate(policy or {}),
        families=load_families(),
        native=native,
        panels=panels,
        db_families=db_families,
    )


def _native(devices):
    return NativeDevices.model_validate({"devices": devices})


SUPPLY = {"kind": "setpoint_device", "template": "fastps", "settables": {"current": {"tolerance": 0.01}}}  # fmt: skip


def test_a_family_hazard_role_is_asked_by_role_suffix_or_readback():
    native = _native({"U_PS": SUPPLY})
    for variable in ("output_sp", "enable_output", "output"):
        found = _hazards(
            {"do": "set", "device": "U_PS", "variable": variable, "value": 1},
            native=native,
        )
        assert [h.index for h in found] == [1], variable
        assert "output_sp" in found[0].reasons[0]
    quiet = _hazards({"do": "set", "device": "U_PS", "variable": "current", "value": 2.0}, native=native)  # fmt: skip
    assert quiet == []


def test_a_panel_hazard_adds_to_the_family():
    panels = PanelDevices.model_validate(
        {"devices": {"U_PS": {"family": "fastps", "hazard": ["current_sp"]}}}
    )
    found = _hazards(
        {"do": "wait", "seconds": 1},
        {"do": "set", "device": "U_PS", "variable": "current", "value": 2.0},
        panels=panels,
    )
    assert [(h.index, h.device) for h in found] == [(2, "U_PS")]


def test_the_policy_names_a_db_device_by_name_or_device_type():
    by_name = {"hazards": [{"devices": ["U_Gas"], "roles": ["Valve"]}]}
    step = {"do": "set", "device": "U_Gas", "variable": "Valve", "value": "open"}
    assert len(_hazards(step, policy=by_name)) == 1
    by_type = {"hazards": [{"families": ["GasValve"], "roles": ["Valve"]}]}
    assert len(_hazards(step, policy=by_type, db_families={"U_Gas": "GasValve"})) == 1
    assert _hazards(step, policy=by_type, db_families={"U_Gas": "Other"}) == []
    other_role = {**step, "variable": "Pressure"}
    assert _hazards(other_role, policy=by_type, db_families={"U_Gas": "GasValve"}) == []  # fmt: skip


def test_an_unknown_family_asks_when_the_policy_names_the_role_for_any_family():
    by_type = {"hazards": [{"families": ["GasValve"], "roles": ["Valve"]}]}
    step = {"do": "set", "device": "U_Gas", "variable": "Valve", "value": "open"}
    found = _hazards(step, policy=by_type, db_families=None)
    assert len(found) == 1 and "unknown" in found[0].reasons[0]
    # The DB answered and does not know the device: still unknown, still asked.
    assert len(_hazards(step, policy=by_type, db_families={})) == 1


def test_checks_and_waits_are_never_hazards():
    native = _native({"U_PS": SUPPLY})
    found = _hazards(
        {"do": "check", "device": "U_PS", "variable": "output_sp", "expected": 1},
        {"do": "wait", "seconds": 1},
        native=native,
    )
    assert found == []


@pytest.mark.parametrize(
    ("device", "variable"),
    [("U_PS", "output"), ("U_PS", "Output"), ("u_ps", "output"), ("U_PS", "OUTPUT_SP")],
)
def test_a_family_hazard_is_asked_whatever_the_spelling_the_worker_accepts(device, variable):  # fmt: skip
    """The worker binds a device and a variable case-folded, so the hazard rule does too."""
    native = _native({"U_PS": SUPPLY})
    step = {"do": "set", "device": device, "variable": variable, "value": 1}
    assert [h.index for h in _hazards(step, native=native)] == [1]


@pytest.mark.parametrize(
    ("device", "variable"),
    [
        ("U_Laser", "Shutter"),
        ("U_Laser", "shutter"),
        ("u_laser", "Shutter"),
        ("U_LASER", "SHUTTER"),
    ],  # fmt: skip
)
def test_a_policy_hazard_is_asked_whatever_the_spelling(device, variable):
    by_name = {"hazards": [{"devices": ["U_Laser"], "roles": ["Shutter"]}]}
    by_type = {"hazards": [{"families": ["LaserCtl"], "roles": ["Shutter"]}]}
    step = {"do": "set", "device": device, "variable": variable, "value": "closed"}
    assert len(_hazards(step, policy=by_name)) == 1
    found = _hazards(step, policy=by_type, db_families={"U_Laser": "LaserCtl"})
    assert len(found) == 1 and "unknown" not in found[0].reasons[0]


def test_the_converted_fixture_setups_load_and_flatten():
    """The action library's Amp4_DUMP_HP, converted once: it applies the other two."""
    from pathlib import Path

    setups = load_setups(Path(__file__).parent / "fixtures")
    assert set(setups) == {
        "Amp4_DUMP_HP",
        "close_gaia_internal_shutters",
        "open_gaia_internal_shutters",
    }
    steps = flatten_setup(setups["Amp4_DUMP_HP"], setups=setups)
    origins = {origin for _, origin in steps}
    assert origins == {
        None,
        "close_gaia_internal_shutters",
        "open_gaia_internal_shutters",
    }


def test_the_step_variable_grammar_round_trips():
    """One grammar for a step's variable on a native device: <role>, or <axis>.<role>."""
    from geecs_schemas import split_step_variable, step_variable

    roles = load_families()["motor_record"].roles
    axes = ["x", "y"]
    assert split_step_variable("x", axes=axes, roles=roles) == ("val", "x")
    assert split_step_variable("y.rbv", axes=axes, roles=roles) == ("rbv", "y")
    assert split_step_variable("Current", axes=axes, roles=roles) == ("Current", None)
    for role, axis in (("rbv", "y"), ("Current", None)):
        assert split_step_variable(
            step_variable(role, axis), axes=axes, roles=roles
        ) == (role, axis)


@pytest.mark.parametrize("variable", ["m1", "m1.val", "M1.VAL"])
def test_an_axis_step_finds_the_hazard_on_its_setpoint(variable):
    """A bare axis names its setpoint (the module's grammar), ``<axis>.<role>`` that role."""
    native = _native({"U_Stage": {"kind": "epics_motor", "axes": ["m1", "m2"]}})
    panels = PanelDevices.model_validate(
        {"devices": {"U_Stage": {"family": "motor_record", "hazard": ["val"]}}}
    )
    step = {"do": "set", "device": "U_Stage", "variable": variable, "value": 1.0}
    found = _hazards(step, native=native, panels=panels)
    assert [h.index for h in found] == [1] and "'val'" in found[0].reasons[0]
    quiet = {**step, "variable": "m1.velo"}
    assert _hazards(quiet, native=native, panels=panels) == []
