"""The setup document: steps, nested applies, and the check's comparison."""

from decimal import Decimal

import pytest
import yaml
from pydantic import ValidationError

from geecs_schemas import (
    Setup,
    SetupCycleError,
    SetupNotFoundError,
    flatten_setup,
    load_setups,
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
