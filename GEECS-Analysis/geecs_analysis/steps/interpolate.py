"""Linear trace resampling with explicit zero padding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class InterpolateSpec(StepSpec):
    """Resample a trace on a uniform axis; omitted bounds use its extrema.

    Uses numpy.interp's sample ordering semantics, as the legacy trace pipeline
    does. Samples are not sorted, deduplicated, or reversed implicitly.
    """

    step: Literal["interpolate"] = "interpolate"
    count: int = Field(
        ge=2, description="Number of uniformly spaced samples on the new axis."
    )
    lower: float | None = Field(
        None, description="Start of the new axis; unset uses the trace's minimum."
    )
    upper: float | None = Field(
        None, description="End of the new axis; unset uses the trace's maximum."
    )

    @model_validator(mode="after")
    def ordered_bounds(self) -> Self:
        """Reject reversed or degenerate explicitly supplied bounds."""
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError("Interpolation upper bound must exceed lower bound")
        return self


@step(InterpolateSpec, ndim={1})
def interpolate(frame: Frame, spec: InterpolateSpec) -> Frame:
    """Resample values and coordinates together, preserving units and provenance."""
    import numpy as np
    from geecs_data_utils.frames import Axis

    axis = frame.axes[0]
    lower = spec.lower if spec.lower is not None else np.min(axis.values)
    upper = spec.upper if spec.upper is not None else np.max(axis.values)
    coordinates = np.linspace(lower, upper, spec.count)
    values = np.interp(coordinates, axis.values, frame.data, left=0.0, right=0.0)
    return frame.replace(
        data=values,
        axes=(Axis(coordinates, unit=axis.unit, label=axis.label),),
    )
