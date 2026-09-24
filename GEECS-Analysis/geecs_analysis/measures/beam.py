"""Legacy beam statistics with global coordinates and declarative overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement


class BeamSpec(MeasureSpec):
    """Beam projections, optional scalar selection and local-index slopes."""

    kind: Literal["beam"] = "beam"
    enabled_stats: tuple[str, ...] | None = Field(
        None, description="Scalar names to keep (unset keeps every statistic)."
    )
    compute_slopes: bool = Field(
        False,
        description="Also fit local slopes of the centroid and peak (image_*_slope_*).",
    )

    def emitted_scalars(self) -> frozenset[str]:
        """Preserve the existing optimizer key-discovery and selection contract."""
        keys = {"image_total", "image_peak_value"} | {
            f"{axis}_{stat}"
            for axis in ("x", "y", "x_45", "y_45")
            for stat in ("CoM", "rms", "fwhm", "peak_location")
        }
        if self.enabled_stats is not None:
            keys.intersection_update(self.enabled_stats)
        if self.compute_slopes:
            keys.update(
                f"image_{stat}_slope_{axis}"
                for stat in ("com", "peak")
                for axis in ("x", "y")
            )
        return frozenset(keys)


@measure(BeamSpec, ndim={2})
def beam(frame: Frame, spec: BeamSpec) -> Measurement:
    """Measure a beam; x/y use frame coordinates, diagonals/slopes local indices."""
    from math import isfinite

    from geecs_data_utils.frames import Frame
    from geecs_analysis.algorithms.basic_beam_stats import (
        beam_profile_stats,
        flatten_beam_stats,
    )
    from geecs_analysis.algorithms.beam_slopes import compute_beam_slopes
    from geecs_analysis.measurement import Marker, Measurement, Projection

    if frame.data.ndim != 2:
        raise ValueError("Beam measurement requires a 2D frame")
    stats = beam_profile_stats(frame.data, tuple(axis.values for axis in frame.axes))
    scalars = flatten_beam_stats(
        stats,
        include=set(spec.enabled_stats) if spec.enabled_stats is not None else None,
    )
    if spec.compute_slopes:
        scalars.update(compute_beam_slopes(frame.data))
    overlays = [
        Projection(
            "projection_x",
            1,
            Frame.from_array(
                frame.data.sum(axis=0),
                axes=(frame.axes[1],),
                shot=frame.shot,
                unit=frame.unit,
            ),
        ),
        Projection(
            "projection_y",
            0,
            Frame.from_array(
                frame.data.sum(axis=1),
                axes=(frame.axes[0],),
                shot=frame.shot,
                unit=frame.unit,
            ),
        ),
    ]
    if isfinite(stats.x.CoM) and isfinite(stats.y.CoM):
        overlays.append(Marker("com", stats.x.CoM, stats.y.CoM))
    return Measurement(scalars=scalars, frame=frame, overlays=tuple(overlays))
