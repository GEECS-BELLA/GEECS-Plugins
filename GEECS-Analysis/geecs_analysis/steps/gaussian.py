"""Gaussian smoothing on sample indices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class GaussianSpec(StepSpec):
    """Isotropic Gaussian width in samples, not physical-axis units."""

    step: Literal["gaussian"] = "gaussian"
    sigma: float = Field(1.0, gt=0, description="Gaussian width in samples.")


@step(GaussianSpec, ndim={1, 2})
def gaussian(frame: Frame, spec: GaussianSpec) -> Frame:
    """Smooth with reflect boundaries, retaining axes and provenance."""
    from scipy.ndimage import gaussian_filter

    return frame.replace(
        data=gaussian_filter(frame.data, sigma=spec.sigma, mode="reflect")
    )
