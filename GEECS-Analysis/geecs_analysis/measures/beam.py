"""Legacy beam statistics with global coordinates and declarative overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Mapping

from pydantic import Field

from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement


_PROJECTION = {
    "CoM": "Intensity-weighted centroid of the {p}, in {u}",
    "rms": "RMS width of the {p} about its centroid, in {u}; sensitive to halos and leftover background",
    "fwhm": "Full width at half maximum of the {p}, in {u}; insensitive to faint halos",
    "peak_location": "Position of the maximum of the {p}, in {u}",
}
_AXES = {
    "x": (
        "x projection (the image summed over rows)",
        "x-axis units (pixels unless calibrated)",
    ),
    "y": (
        "y projection (the image summed over columns)",
        "y-axis units (pixels unless calibrated)",
    ),
    "x_45": ("NW-SE diagonal projection", "local pixel index"),
    "y_45": ("NE-SW diagonal projection", "local pixel index"),
}
BEAM_SCALAR_DOCS: dict[str, str] = {
    "image_total": "Sum of every pixel of the processed image, in counts",
    "image_peak_value": "Brightest single pixel of the processed image, in counts",
    **{
        f"{axis}_{stat}": text.format(p=projection, u=units)
        for axis, (projection, units) in _AXES.items()
        for stat, text in _PROJECTION.items()
    },
    "image_com_slope_x": "Tilt: change of each column's vertical centroid per column, px per px",
    "image_com_slope_y": "Shear: change of each row's horizontal centroid per row, px per px",
    "image_peak_slope_x": "As image_com_slope_x, using each column's peak instead of its centroid",
    "image_peak_slope_y": "As image_com_slope_y, using each row's peak instead of its centroid",
}


class BeamSpec(MeasureSpec):
    """Beam projections, optional scalar selection and local-index slopes."""

    kind: Literal["beam"] = "beam"
    scalar_docs: ClassVar[Mapping[str, str]] = BEAM_SCALAR_DOCS
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
