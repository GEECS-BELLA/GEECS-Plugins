"""Parity with upstream patterns and preview/device key equivalence."""

from dataclasses import dataclass

import numpy as np
import pytest
from bluesky import plan_patterns as patterns
from geecs_schemas import AxisSweep, ListAxis, LogAxis, RangeAxis, Sweep

from geecs_bluesky.trajectory import axis_positions, sweep_to_cycler


@dataclass(frozen=True)
class Motor:
    name: str

    def set(self, value):
        raise AssertionError("Trajectory generation must not touch hardware")

    def read(self):
        raise AssertionError("Trajectory generation must not read hardware")

    def describe(self):
        raise AssertionError("Trajectory generation must not describe hardware")


def assert_same(actual, expected):
    assert actual.keys == expected.keys
    assert len(actual) == len(expected)
    for key in actual.keys:
        np.testing.assert_allclose(
            actual.by_key()[key], expected.by_key()[key], rtol=0, atol=0
        )


@pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
@pytest.mark.parametrize("num", [1, 2, 7])
def test_correlated_ranges_match_stock_inner_product(dimensions, num):
    motors = [Motor(f"axis_{i}") for i in range(dimensions)]
    axes = [
        RangeAxis(axis=m.name, start=i, stop=-i - 2, num=num)
        for i, m in enumerate(motors)
    ]
    sweep = Sweep(trajectory=AxisSweep(axes=axes))
    args = [item for m, a in zip(motors, axes) for item in (m, a.start, a.stop)]
    assert_same(
        sweep_to_cycler(sweep, {m.name: m for m in motors}.__getitem__),
        patterns.inner_product(num, args),
    )


@pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
@pytest.mark.parametrize("snake", [False, True])
def test_range_grid_matches_stock_outer_product(dimensions, snake):
    motors = [Motor(f"axis_{i}") for i in range(dimensions)]
    axes = [
        RangeAxis(axis=m.name, start=i, stop=i + 2, num=2 + i % 2)
        for i, m in enumerate(motors)
    ]
    args = []
    for i, (motor, axis) in enumerate(zip(motors, axes)):
        args.extend([motor, axis.start, axis.stop, axis.num])
        if i:
            args.append(snake)
    sweep = Sweep(trajectory=AxisSweep(axes=axes, combine="product", snake=snake))
    actual = sweep_to_cycler(sweep, {m.name: m for m in motors}.__getitem__)
    assert_same(actual, patterns.outer_product(args))
    assert len(actual) == sweep.n_steps()


def test_five_lists_move_together_without_sorting_or_dropping_repeats():
    axes = [ListAxis(axis=f"x{i}", positions=[i + 3, i + 1, i + 1]) for i in range(5)]
    sweep = Sweep(trajectory=AxisSweep(axes=axes))
    rows = list(sweep_to_cycler(sweep, str))
    assert len(rows) == 3
    assert rows[0] == {f"x{i}": i + 3 for i in range(5)}
    assert rows[1] == rows[2] == {f"x{i}": i + 1 for i in range(5)}


def test_mixed_grid_matches_explicit_upstream_lists_and_device_preview_agree():
    sweep = Sweep(
        trajectory=AxisSweep(
            combine="product",
            snake=True,
            axes=[
                RangeAxis(axis="x", start=3, stop=1, num=3, relative=True),
                ListAxis(axis="y", positions=[8, 6]),
                LogAxis(axis="z", start_exp=-1, stop_exp=1, num=3),
            ],
        )
    )
    expected = patterns.outer_list_product(
        ["x", [3.0, 2.0, 1.0], "y", [8.0, 6.0], "z", np.logspace(-1, 1, 3)], True
    )
    preview = sweep_to_cycler(sweep, str)
    assert_same(preview, expected)
    motors = {name: Motor(name) for name in ("x", "y", "z")}
    execution = sweep_to_cycler(sweep, motors.__getitem__)
    assert [{m.name: v for m, v in row.items()} for row in execution] == list(preview)
    assert sweep.trajectory.axes[0].relative  # expansion did not consume/change it


@pytest.mark.parametrize("start, stop", [(-2, 2), (2, -2), (0, 0)])
def test_log_bounds_are_decade_exponents(start, stop):
    axis = LogAxis(axis="x", start_exp=start, stop_exp=stop, num=5)
    np.testing.assert_array_equal(axis_positions(axis), np.logspace(start, stop, 5))


@pytest.mark.parametrize(
    "axis",
    [
        LogAxis(axis="x", start_exp=400, stop_exp=500, num=2),
        LogAxis(axis="x", start_exp=-500, stop_exp=-400, num=2),
        RangeAxis(axis="x", start=-1e308, stop=1e308, num=3),
    ],
)
def test_numerical_overflow_and_underflow_are_refused(axis):
    with pytest.raises(ValueError, match="unrepresentable"):
        sweep_to_cycler(Sweep(trajectory=AxisSweep(axes=[axis])), str)


def test_aliases_to_one_movable_are_refused_before_numerical_expansion():
    sweep = Sweep(
        trajectory=AxisSweep(
            axes=[
                RangeAxis(axis="alias", start=0, stop=1, num=10**12),
                RangeAxis(axis="device:current", start=0, stop=1, num=10**12),
            ]
        )
    )
    with pytest.raises(ValueError, match="same movable"):
        sweep_to_cycler(sweep, lambda name: "one_movable")


@pytest.mark.parametrize(
    "kind, params, stock",
    [
        ("spiral", dict(dr=0.3, nth=6, dr_y=0.2, tilt=0.2), patterns.spiral),
        (
            "spiral_fermat",
            dict(dr=0.3, factor=1.3, dr_y=0.2, tilt=0.2),
            patterns.spiral_fermat,
        ),
        ("spiral_square", dict(x_num=6, y_num=3), patterns.spiral_square_pattern),
    ],
)
def test_typed_patterns_match_bluesky(kind, params, stock):
    sweep = Sweep.model_validate(
        {
            "trajectory": dict(
                kind=kind,
                x={"axis": "x", "relative": True},
                y={"axis": "y"},
                x_center=1,
                y_center=-2,
                x_range=4,
                y_range=2,
                **params,
            )
        }
    )
    actual = sweep_to_cycler(sweep, str)
    expected = stock("x", "y", 1, -2, 4, 2, **params)
    assert_same(actual, expected)


@pytest.mark.parametrize("x_num", [2, 3, 6, 9])
@pytest.mark.parametrize("y_num", [2, 3, 6, 9])
def test_square_count_matches_bluesky_for_even_odd_and_unequal_sizes(x_num, y_num):
    sweep = Sweep.model_validate(
        {
            "trajectory": dict(
                kind="spiral_square",
                x={"axis": "x"},
                y={"axis": "y"},
                x_center=0,
                y_center=0,
                x_range=4,
                y_range=2,
                x_num=x_num,
                y_num=y_num,
            )
        }
    )
    assert len(sweep_to_cycler(sweep, str)) == sweep.n_steps()


def test_x2x_y_traverses_half_of_x_and_both_are_relative():
    sweep = Sweep.model_validate(
        {
            "trajectory": dict(
                kind="x2x",
                x={"axis": "x"},
                y={"axis": "y"},
                start=-4,
                stop=2,
                num=4,
            )
        }
    )
    assert_same(
        sweep_to_cycler(sweep, str), patterns.inner_product(4, ["x", -4, 2, "y", -2, 1])
    )
    assert all(a.relative for a in sweep.axis_references())
