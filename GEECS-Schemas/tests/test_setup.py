"""The setup document: steps, nested applies, and the check's comparison."""

import pytest
from pydantic import ValidationError

from geecs_schemas import (
    Setup,
    SetupCycleError,
    SetupNotFoundError,
    flatten_setup,
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


@pytest.mark.parametrize("name", ["", "../x", "a/b", "low power"])
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
        (1.0, True, 1.0, False),  # a bool is not a number reading
        ("on", "on", 0.0, True),
        ("on", "ON", 0.0, False),  # a word matches exactly
        ("25", 25, 0.0, False),  # a numeric-looking word is still a word
        ("25", "25", 0.0, True),
    ],
)
def test_check_comparison(expected, actual, tolerance, matches):
    assert setup_value_matches(expected, actual, tolerance) is matches
