"""Serializable trajectories for the predetermined ``sweep`` plan.

A sweep describes positions, not acquisition: detectors, shots, trigger
profile and strict/gated mode remain plan arguments. Motionless acquisition
remains ``count``. Numerical expansion belongs to ``geecs_bluesky.trajectory``;
this module imports neither NumPy nor Bluesky and never reads a device.

``Sweep.trajectory`` is discriminated by ``kind``: ordinary axis sweeps mix
range, list and logarithmic spacing; patterns have their own typed parameters.
All nested models reject unknown fields. The enclosing preset versions the
document, so this payload does not introduce a second version stamp.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, StringConstraints, model_validator

from ._base import SchemaModel

AxisName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PointCount = Annotated[int, Field(strict=True, ge=1)]
PositiveFinite = Annotated[FiniteFloat, Field(gt=0)]


class SweepAxis(SchemaModel):
    """An authored axis reference and its coordinate frame.

    ``axis`` names a catalog variable, a ``Device:Variable`` or an expanded
    namespace binding. The client resolves aliases, not this model. Relative
    values are offsets from the run's starting readback; the execution plan
    must restore these axes on completion or abort. This flag is distinct
    from a pseudo positioner's own relative definition and stage lifecycle.
    """

    axis: AxisName
    relative: bool = False


class RangeAxis(SweepAxis):
    """Inclusive evenly spaced positions; ``num=1`` visits ``start`` only."""

    kind: Literal["range"] = "range"
    start: FiniteFloat
    stop: FiniteFloat
    num: PointCount


class ListAxis(SweepAxis):
    """Positions in the supplied order, retaining deliberate repeats."""

    kind: Literal["list"] = "list"
    positions: list[FiniteFloat] = Field(min_length=1)


class LogAxis(SweepAxis):
    """Logarithmic positions with bounds expressed as decade exponents."""

    kind: Literal["log"] = "log"
    start_exp: FiniteFloat
    stop_exp: FiniteFloat
    num: PointCount


Axis = Annotated[RangeAxis | ListAxis | LogAxis, Field(discriminator="kind")]


class AxisSweep(SchemaModel):
    """One or more axes moving together or across a Cartesian grid.

    Axis order is scan order: the first is outermost in a grid. Snaking
    reverses the inner axes using Bluesky's traversal. A single-axis product
    is allowed (the Cartesian product of one list); there is no axis-count
    ceiling. Every correlated axis must have the same number of positions.
    """

    kind: Literal["axes"] = "axes"
    axes: list[Axis] = Field(min_length=1)
    combine: Literal["zip", "product"] = "zip"
    snake: bool = False

    @model_validator(mode="after")
    def _validate_combination(self) -> Self:
        names = [axis.axis for axis in self.axes]
        if len(set(names)) != len(names):
            raise ValueError("Each axis may appear only once in a sweep.")
        if self.snake and self.combine != "product":
            raise ValueError("Snake applies to a grid; choose combine='product'.")
        counts = [
            len(a.positions) if isinstance(a, ListAxis) else a.num for a in self.axes
        ]
        if self.combine == "zip" and len(set(counts)) != 1:
            detail = ", ".join(f"{name}: {n}" for name, n in zip(names, counts))
            raise ValueError(
                "Axes moving together need equal point counts "
                f"({detail}). Adjust the counts or choose a grid; "
                "positions are never truncated."
            )
        return self

    def n_steps(self) -> int:
        """Count positions without allocating arrays or forming the product."""
        counts = [
            len(a.positions) if isinstance(a, ListAxis) else a.num for a in self.axes
        ]
        return counts[0] if self.combine == "zip" else math.prod(counts)


class _TwoAxisPattern(SchemaModel):
    x: SweepAxis
    y: SweepAxis

    @model_validator(mode="after")
    def _distinct_axes(self) -> Self:
        if self.x.axis == self.y.axis:
            raise ValueError("A two-axis pattern requires two distinct axes.")
        return self


class _SpiralBounds(_TwoAxisPattern):
    x_center: FiniteFloat
    y_center: FiniteFloat
    x_range: PositiveFinite
    y_range: PositiveFinite


class SpiralSweep(_SpiralBounds):
    """Bluesky's Archimedean spiral, with ``tilt`` in radians.

    ``dr`` and optional ``dr_y`` are radial increments; ``nth`` is the
    number of angular steps in the first ring. Expansion and clipping to
    the requested region are entirely Bluesky's.
    """

    kind: Literal["spiral"] = "spiral"
    dr: PositiveFinite
    nth: PointCount
    dr_y: PositiveFinite | None = None
    tilt: FiniteFloat = 0


class FermatSpiralSweep(_SpiralBounds):
    """Bluesky's Fermat spiral; ``factor`` controls angular sampling."""

    kind: Literal["spiral_fermat"] = "spiral_fermat"
    dr: PositiveFinite
    factor: PositiveFinite
    dr_y: PositiveFinite | None = None
    tilt: FiniteFloat = 0


class SquareSpiralSweep(_SpiralBounds):
    """Bluesky's square spiral; each dimension needs at least two points."""

    kind: Literal["spiral_square"] = "spiral_square"
    x_num: Annotated[int, Field(strict=True, ge=2)]
    y_num: Annotated[int, Field(strict=True, ge=2)]


class RelativeSweepAxis(SweepAxis):
    """An axis which is necessarily relative, as in stock ``x2x_scan``."""

    relative: Literal[True] = True


class X2XSweep(_TwoAxisPattern):
    """Two relative axes; Y traverses half of X's requested range."""

    kind: Literal["x2x"] = "x2x"
    x: RelativeSweepAxis
    y: RelativeSweepAxis
    start: FiniteFloat
    stop: FiniteFloat
    num: PointCount


SweepTrajectory = Annotated[
    AxisSweep | SpiralSweep | FermatSpiralSweep | SquareSpiralSweep | X2XSweep,
    Field(discriminator="kind"),
]


class Sweep(SchemaModel):
    """The JSON payload of one predetermined moving scan.

    Examples
    --------
    A five-variable correlated list sweep has five ``ListAxis`` entries
    under ``trajectory={kind: axes, combine: zip, axes: [...]}``. Ten
    values on each axis mean ten steps, not 100,000. ``count`` remains a
    distinct plan; an empty sweep is a validation error.
    """

    trajectory: SweepTrajectory

    def axis_references(self) -> tuple[SweepAxis, ...]:
        """Return every axis in authored order, including its relative flag."""
        trajectory = self.trajectory
        if isinstance(trajectory, AxisSweep):
            return tuple(trajectory.axes)
        return trajectory.x, trajectory.y

    def n_steps(self) -> int | None:
        """Return an algebraic count, or ``None`` when Bluesky must expand it.

        Curved spirals are clipped by Bluesky's geometry. Their exact size
        comes from the resulting cycler; it is never estimated here. Axis
        sweeps can be size-checked without materializing any positions.
        """
        trajectory = self.trajectory
        if isinstance(trajectory, AxisSweep):
            return trajectory.n_steps()
        if isinstance(trajectory, X2XSweep):
            return trajectory.num
        if isinstance(trajectory, SquareSpiralSweep):
            return trajectory.x_num * trajectory.y_num
        return None
