"""Hardware-free expansion of a Sweep, shared by preview and execution.

Only this module turns the schema into numerical positions. NumPy supplies
range/log spacing, and Bluesky's plan_patterns owns every traversal and
spiral. A browser draws the resulting coordinates; it never expands them.

Relative coordinates remain offsets here. The plan applies the starting
readbacks and restores the relative axes through Bluesky preprocessors;
preview can therefore work without a live readback, namespace or gateway.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from bluesky import plan_patterns
from cycler import Cycler
from geecs_schemas.sweep import (
    Axis,
    AxisSweep,
    FermatSpiralSweep,
    ListAxis,
    LogAxis,
    RangeAxis,
    SpiralSweep,
    SquareSpiralSweep,
    Sweep,
    X2XSweep,
)

AxisKey = TypeVar("AxisKey", bound=Hashable)


@dataclass(frozen=True)
class _Ref:
    """Movable/readable stand-in for Bluesky's pattern type checks."""

    name: str

    def set(self, value: float) -> None:
        raise NotImplementedError("Trajectory expansion never moves an axis.")

    def read(self) -> dict[str, object]:
        return {}

    def describe(self) -> dict[str, object]:
        return {}


def axis_positions(axis: Axis) -> list[float]:
    """Expand one axis, refusing unrepresentable positions before execution.

    A list is copied without sorting or deduplication. Log bounds are decade
    exponents, including when the resulting values are relative offsets.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if isinstance(axis, RangeAxis):
            points = np.linspace(axis.start, axis.stop, axis.num)
        elif isinstance(axis, LogAxis):
            points = np.logspace(axis.start_exp, axis.stop_exp, axis.num)
        elif isinstance(axis, ListAxis):
            points = np.asarray(axis.positions, dtype=float)
        else:
            raise TypeError(f"Unsupported sweep axis: {type(axis).__name__}")
    if not np.isfinite(points).all() or (
        isinstance(axis, LogAxis) and (points <= 0).any()
    ):
        raise ValueError(f"Axis {axis.axis!r} produces unrepresentable positions.")
    return points.tolist()


def sweep_to_cycler(
    sweep: Sweep, resolve: Callable[[str], AxisKey]
) -> Cycler[AxisKey, float]:
    """Build the same ordered trajectory for a preview or a stock scan_nd.

    Parameters
    ----------
    sweep : Sweep
        Validated payload. Relative values remain offsets.
    resolve : callable
        Maps authored axis references to hashable keys. Preview may pass
        ``str``; execution passes its namespace lookup. No device method
        is invoked. Aliases resolving to the same key are refused.

    Returns
    -------
    Cycler
        Positions keyed by the resolved axes, in Bluesky traversal order.

    Notes
    -----
    This function materializes trajectories. Callers serving untrusted or
    interactive input must enforce their resource budget before expansion;
    ``Sweep.n_steps`` provides a cheap exact count for axes and square/x2x
    patterns. Curved patterns require a separate expansion budget.
    """
    axes = sweep.axis_references()
    keys = [resolve(axis.axis) for axis in axes]
    if len(set(keys)) != len(keys):
        raise ValueError("Two sweep axes resolve to the same movable.")
    refs = [_Ref(axis.axis) for axis in axes]
    trajectory = sweep.trajectory
    if isinstance(trajectory, AxisSweep):
        args = []
        for ref, axis in zip(refs, trajectory.axes):
            args.extend((ref, axis_positions(axis)))
        if trajectory.combine == "zip":
            result = plan_patterns.inner_list_product(args)
        else:
            result = plan_patterns.outer_list_product(args, trajectory.snake)
    elif isinstance(trajectory, X2XSweep):
        # The stock x2x_scan is this relative inner product (Y = X / 2).
        result = plan_patterns.inner_product(
            trajectory.num,
            [
                refs[0],
                trajectory.start,
                trajectory.stop,
                refs[1],
                trajectory.start / 2,
                trajectory.stop / 2,
            ],
        )
    else:
        args = [
            *refs,
            trajectory.x_center,
            trajectory.y_center,
            trajectory.x_range,
            trajectory.y_range,
        ]
        if isinstance(trajectory, SquareSpiralSweep):
            result = plan_patterns.spiral_square_pattern(
                *args, trajectory.x_num, trajectory.y_num
            )
        elif isinstance(trajectory, SpiralSweep):
            result = plan_patterns.spiral(
                *args,
                trajectory.dr,
                trajectory.nth,
                dr_y=trajectory.dr_y,
                tilt=trajectory.tilt,
            )
        elif isinstance(trajectory, FermatSpiralSweep):
            result = plan_patterns.spiral_fermat(
                *args,
                trajectory.dr,
                trajectory.factor,
                dr_y=trajectory.dr_y,
                tilt=trajectory.tilt,
            )
        else:
            raise TypeError(f"Unsupported trajectory: {type(trajectory).__name__}")
    if not len(result):
        raise ValueError("The requested pattern produces no positions.")
    for ref, values in result.by_key().items():
        if not np.isfinite(values).all():
            raise ValueError(f"Axis {ref.name!r} produces unrepresentable positions.")
    for ref, key in zip(refs, keys):
        result.change_key(ref, key)
    return result
