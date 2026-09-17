"""The sweep payload validates independently of numerical and device stacks."""

import json

import pytest
from pydantic import ValidationError

from geecs_schemas import AxisSweep, ListAxis, LogAxis, RangeAxis, Sweep


def line(name="x", num=3, **kwargs):
    return RangeAxis(axis=name, start=-1, stop=1, num=num, **kwargs)


def test_one_axis_and_five_correlated_lists_are_both_ordinary_sweeps():
    assert Sweep(trajectory=AxisSweep(axes=[line()])).n_steps() == 3
    sweep = Sweep(
        trajectory=AxisSweep(
            axes=[
                ListAxis(axis=f"magnet_{i}", positions=[3, 1, 1, 7]) for i in range(5)
            ]
        )
    )
    assert sweep.n_steps() == 4
    assert [axis.axis for axis in sweep.axis_references()] == [
        f"magnet_{i}" for i in range(5)
    ]
    assert sweep.trajectory.axes[0].positions == [3, 1, 1, 7]


def test_mixed_spacing_and_independent_coordinate_frames():
    sweep = Sweep(
        trajectory=AxisSweep(
            axes=[
                line(relative=True),
                ListAxis(axis="y", positions=[3, 5, 9]),
                LogAxis(axis="z", start_exp=-2, stop_exp=1, num=3, relative=True),
            ]
        )
    )
    assert sweep.n_steps() == 3
    assert [a.relative for a in sweep.axis_references()] == [True, False, True]
    assert Sweep.model_validate_json(sweep.model_dump_json()) == sweep


@pytest.mark.parametrize(
    "other",
    [
        RangeAxis(axis="y", start=0, stop=1, num=2),
        ListAxis(axis="y", positions=[1, 2]),
        LogAxis(axis="y", start_exp=0, stop_exp=2, num=2),
    ],
)
def test_all_correlated_spacing_kinds_require_equal_lengths(other):
    with pytest.raises(ValidationError, match="equal point counts.*x: 3, y: 2"):
        AxisSweep(axes=[line(), other])
    grid = AxisSweep(axes=[line(), other], combine="product")
    assert grid.n_steps() == 6


def test_grid_count_does_not_allocate_the_requested_positions():
    sweep = Sweep(
        trajectory=AxisSweep(
            axes=[line("x", 10**12), line("y", 10**12)], combine="product"
        )
    )
    assert sweep.n_steps() == 10**24


def test_single_axis_grid_is_valid_and_can_survive_axis_removal():
    assert AxisSweep(axes=[line()], combine="product", snake=True).n_steps() == 3


@pytest.mark.parametrize("num", [0, -1, True, 1.5, "3"])
def test_point_counts_are_positive_integers(num):
    with pytest.raises(ValidationError):
        line(num=num)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("kind", ["range", "list", "log"])
def test_all_axis_values_are_finite(value, kind):
    payloads = {
        "range": dict(kind=kind, axis="x", start=value, stop=1, num=2),
        "list": dict(kind=kind, axis="x", positions=[0, value]),
        "log": dict(kind=kind, axis="x", start_exp=0, stop_exp=value, num=2),
    }
    with pytest.raises(ValidationError):
        Sweep.model_validate({"trajectory": {"kind": "axes", "axes": [payloads[kind]]}})


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"axes": []}, "at least 1 item"),
        ({"axes": [line(), line(" x ")]}, "only once"),
        ({"axes": [line()], "snake": True}, "Snake applies to a grid"),
        ({"axes": [line()], "snkae": True}, "Extra inputs"),
    ],
)
def test_invalid_axis_sweeps_fail_loudly(payload, message):
    with pytest.raises(ValidationError, match=message):
        AxisSweep.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "list", "axis": "x", "positions": []},
        {"kind": "range", "axis": "   ", "start": 0, "stop": 1, "num": 3},
        {"kind": "log", "axis": "x", "start": 0, "stop": 1, "num": 3},
    ],
)
def test_missing_positions_blank_names_and_ambiguous_log_bounds_are_refused(payload):
    with pytest.raises(ValidationError):
        Sweep.model_validate({"trajectory": {"kind": "axes", "axes": [payload]}})


@pytest.mark.parametrize(
    "kind, fields, steps",
    [
        ("spiral", {"dr": 0.1, "nth": 6}, None),
        ("spiral_fermat", {"dr": 0.1, "factor": 1}, None),
        ("spiral_square", {"x_num": 4, "y_num": 3}, 12),
    ],
)
def test_patterns_round_trip_and_keep_separate_relative_flags(kind, fields, steps):
    sweep = Sweep.model_validate(
        {
            "trajectory": dict(
                kind=kind,
                x={"axis": "x", "relative": True},
                y={"axis": "y"},
                x_center=0,
                y_center=1,
                x_range=4,
                y_range=2,
                **fields,
            )
        }
    )
    assert sweep.n_steps() == steps
    assert [a.relative for a in sweep.axis_references()] == [True, False]
    assert Sweep.model_validate(json.loads(sweep.model_dump_json())) == sweep


def test_x2x_is_always_relative_and_axes_are_distinct():
    payload = dict(
        kind="x2x", x={"axis": "x"}, y={"axis": "y"}, start=-2, stop=2, num=5
    )
    sweep = Sweep.model_validate({"trajectory": payload})
    assert sweep.n_steps() == 5
    assert all(a.relative for a in sweep.axis_references())
    with pytest.raises(ValidationError):
        Sweep.model_validate(
            {"trajectory": {**payload, "x": {"axis": "x", "relative": False}}}
        )
    with pytest.raises(ValidationError, match="distinct"):
        Sweep.model_validate({"trajectory": {**payload, "y": {"axis": "x"}}})


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "spiral", "dr": 0, "nth": 3},
        {"kind": "spiral", "dr": 1, "nth": 0},
        {"kind": "spiral_fermat", "dr": 1, "factor": -1},
        {"kind": "spiral_square", "x_num": 1, "y_num": 4},
    ],
)
def test_pattern_parameters_refuse_undefined_geometry(fields):
    with pytest.raises(ValidationError):
        Sweep.model_validate(
            {
                "trajectory": dict(
                    x={"axis": "x"},
                    y={"axis": "y"},
                    x_center=0,
                    y_center=0,
                    x_range=2,
                    y_range=2,
                    **fields,
                )
            }
        )


def test_json_schema_is_discriminated_and_forbids_unknown_payload_fields():
    schema = Sweep.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["trajectory"]["discriminator"]["propertyName"] == "kind"
    assert (
        schema["$defs"]["AxisSweep"]["properties"]["axes"]["items"]["discriminator"][
            "propertyName"
        ]
        == "kind"
    )
