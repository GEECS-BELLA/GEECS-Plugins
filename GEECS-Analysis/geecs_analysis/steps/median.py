"""Median filtering on sample indices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class MedianSpec(StepSpec):
    """Odd-width median kernel, applied along every sample dimension."""

    step: Literal["median"] = "median"
    kernel: int = Field(3, ge=1, strict=True)

    @field_validator("kernel")
    @classmethod
    def odd_kernel(cls, value: int) -> int:
        """Require an odd width to retain the legacy centering convention."""
        if value % 2 != 1:
            raise ValueError("Median kernel must be odd")
        return value


@step(MedianSpec, ndim={1, 2})
def median(frame: Frame, spec: MedianSpec) -> Frame:
    """Filter samples with reflect boundaries, retaining axes and provenance."""
    from scipy.ndimage import median_filter

    return frame.replace(
        data=median_filter(frame.data, size=spec.kernel, mode="reflect")
    )
