"""Coordinate-preserving region selection in numpy dimension order."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class RoiSpec(StepSpec):
    """Half-open index or inclusive physical bounds, one pair per dimension."""

    step: Literal["roi"] = "roi"
    bounds: tuple[tuple[float | None, float | None], ...] = Field(
        min_length=1,
        max_length=2,
        description=(
            "(low, high) per dimension in numpy order: rows (y) then columns "
            "(x) for an image, one pair for a trace; null leaves that side open."
        ),
    )
    units: Literal["index", "axis"] = Field(
        "index",
        description="'index' = half-open sample indices; 'axis' = inclusive physical coordinates.",
    )

    @model_validator(mode="after")
    def valid_bounds(self) -> Self:
        """Reject reversed bounds and fractional/negative sample indices."""
        for lo, hi in self.bounds:
            if lo is not None and hi is not None and lo > hi:
                raise ValueError("ROI lower bound must not exceed upper bound")
            if self.units == "index" and any(
                v is not None and (v < 0 or not v.is_integer()) for v in (lo, hi)
            ):
                raise ValueError("Index bounds must be nonnegative integers")
        return self


@step(RoiSpec, ndim={1, 2})
def roi(frame: Frame, spec: RoiSpec) -> Frame:
    """Select samples and matching coordinates without resetting their origin."""
    import numpy as np
    from geecs_data_utils.frames import Axis

    if len(spec.bounds) != frame.data.ndim:
        raise ValueError("ROI requires one bounds pair per frame dimension")
    if spec.units == "index":
        return frame.crop(
            tuple(
                slice(None if lo is None else int(lo), None if hi is None else int(hi))
                for lo, hi in spec.bounds
            )
        )
    data = frame.data
    axes = []
    for dim, (axis, (lo, hi)) in enumerate(zip(frame.axes, spec.bounds, strict=True)):
        selected = np.ones(axis.values.size, dtype=bool)
        if lo is not None:
            selected &= axis.values >= lo
        if hi is not None:
            selected &= axis.values <= hi
        indices = np.flatnonzero(selected)
        if not indices.size:
            raise ValueError("ROI selects no samples")
        data = np.take(data, indices, axis=dim)
        axes.append(Axis(axis.values[indices], unit=axis.unit, label=axis.label))
    return frame.replace(data=data, axes=tuple(axes))
