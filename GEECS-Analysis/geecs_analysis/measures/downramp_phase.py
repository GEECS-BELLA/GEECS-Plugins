"""Plasma downramp shock metrics of a 2D phase map.

The legacy ``DownrampPhaseAnalyzer`` (a ``StandardAnalyzer`` whose camera
pipeline is the recipe's steps), ported with its conventions. The input is
a phase map made upstream — a HASO recipe's processed phase, phase files in
a folder, a notebook array — never a device. The map is inverted about its
minimum, its median subtracted, and binarised through a Gaussian mask
(``algorithms.downramp_shock``); the shock is measured on that map rotated
onto the laser axis. The frame is the legacy product: the masked phase
scaled to 16 bits (``(2¹⁶−1)/2π`` counts per radian, truncated), stored as
float64. The per-shot combined PDF is dropped: the shock line, its position
and the plateau bounds are ``Segment`` overlays, mapped back from the
rotated canvas onto the frame, beside the two projections the legacy
renderer drew.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Mapping

from pydantic import Field

from geecs_analysis.measures.beam import BEAM_SCALAR_DOCS
from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Axis, Frame
    from geecs_analysis.algorithms.downramp_shock import ShockAnalysis
    from geecs_analysis.measurement import Measurement, Segment

#: The five legacy scalar names, verbatim: they are s-file columns in use.
SHOCK_ANGLE = "Plasma downramp shock_angle"
SHOCK_SLOPE = "Plasma downramp shock slope (phase/pixel)"
SHOCK_LOCATION = "Plasma downramp shock location (pixel)"
PLATEAU_AVG = "Plasma downramp plateau avg (phase)"
PEAK_TO_PLATEAU = "Plasma downramp peak to plateau (phase)"
SHOCK_SCALARS = (SHOCK_ANGLE, SHOCK_SLOPE, SHOCK_LOCATION, PLATEAU_AVG, PEAK_TO_PLATEAU)
#: The beam slopes the laser axis comes from; the legacy merged them into its results.
SLOPE_SCALARS = (
    "image_com_slope_x",
    "image_com_slope_y",
    "image_peak_slope_x",
    "image_peak_slope_y",
)
#: One count of the stored frame.
PHASE_UNIT = "2π/65535 rad"
#: The overlay ids, for the recipe's ``figure.overlays`` styles.
OVERLAYS = ("shock_fit", "shock_location", "plateau_start", "plateau_end")


class DownrampPhaseSpec(MeasureSpec):
    """Shock angle, gradient, position and plateau of a plasma downramp phase map.

    The map (any 2D phase, in radians) is inverted, median-subtracted and
    binarised through a Gaussian mask about its centre before the shock is
    measured on the laser axis; the parameters are the legacy analyzer's
    internal constants, at their legacy values.
    """

    kind: Literal["downramp_phase"] = "downramp_phase"
    scalar_docs: ClassVar[Mapping[str, str]] = {
        SHOCK_ANGLE: (
            "Angle of the shock front to the laser axis, rad: the line fitted "
            "through each row's steepest window on the rotated map; NaN with "
            "fewer than two such rows"
        ),
        SHOCK_SLOPE: (
            "Steepest local slope of the column-summed phase along the laser "
            "axis, summed phase per px; NaN when no window qualifies"
        ),
        SHOCK_LOCATION: (
            "Column of that steepest window on the rotated (laser-axis) canvas, "
            "px; NaN when no window qualifies"
        ),
        PLATEAU_AVG: (
            "Mean of the column-summed phase over the plateau window, four to "
            "one windows upstream of the shock; NaN when it is undefined"
        ),
        PEAK_TO_PLATEAU: (
            "Maximum of the column-summed phase minus the plateau average; NaN "
            "when the plateau is undefined"
        ),
        **{key: BEAM_SCALAR_DOCS[key] for key in SLOPE_SCALARS},
    }
    window_size: int = Field(
        20,
        ge=2,
        description=(
            "Sliding window, px, of the local linear fits: the row-wise angle "
            "fit, the summed-phase gradient and the plateau window."
        ),
    )
    sigma_x: float = Field(
        100.0,
        gt=0,
        description="Gaussian mask width along x, px, about the map's centre.",
    )
    sigma_y: float = Field(
        20.0,
        gt=0,
        description="Gaussian mask width along y, px, about the map's centre.",
    )
    threshold_factor: float = Field(
        0.01,
        ge=0,
        description="Binarise the masked map at this fraction of its maximum; pixels below are zeroed.",
    )

    def emitted_scalars(self) -> frozenset[str]:
        """The five shock scalars and the four laser-axis slopes."""
        return frozenset(SHOCK_SCALARS + SLOPE_SCALARS)


@measure(DownrampPhaseSpec, ndim={2})
def downramp_phase(frame: Frame, spec: DownrampPhaseSpec) -> Measurement:
    """Measure the shock of one phase map, the legacy analyzer's way.

    A map with no signal after masking has no laser axis: every shock
    scalar is NaN with a note (the legacy class failed there). So are the
    slope, location and plateau when no window of the summed phase
    qualifies, and the plateau values when its window is empty. The
    overlays are omitted wherever their geometry is undefined. Extras:
    the map on the laser axis (``rotated_phase``) and its column sums
    (``summed_phase``), the legacy figure's data.
    """
    import numpy as np
    from geecs_data_utils.frames import Frame

    from geecs_analysis.algorithms.downramp_shock import (
        analyze_shock,
        apply_gaussian_mask,
    )
    from geecs_analysis.measurement import Measurement, Projection

    if frame.data.ndim != 2:
        raise ValueError("downramp_phase requires a 2D phase map")
    # The legacy preprocessing, step for step.
    processed_image = frame.data
    processed_image = -(processed_image - np.nanmin(processed_image))
    processed_image = abs(processed_image - np.nanmin(processed_image))
    processed_image = processed_image - np.nanmedian(processed_image)
    processed_image = apply_gaussian_mask(
        image=processed_image,
        sigma_x=spec.sigma_x,
        sigma_y=spec.sigma_y,
        binarize=True,
        threshold_factor=spec.threshold_factor,
    )
    shock = analyze_shock(processed_image, window_size=spec.window_size)
    scalars, notes = _scalars(shock)

    phase_scale_16bit_scale = (2**16 - 1) / (2 * np.pi)
    phase_converted = (phase_scale_16bit_scale * processed_image).astype(np.uint16)
    stored = Frame.from_array(
        phase_converted,
        axes=frame.axes,
        shot=frame.shot,
        unit=PHASE_UNIT,
        label="phase",
    )

    def projection(name: str, axis: int) -> Projection:
        lineout = Frame.from_array(
            processed_image.sum(axis=1 - axis),
            axes=(frame.axes[axis],),
            shot=frame.shot,
        )
        return Projection(name, axis, lineout)

    overlays = [projection("projection_x", 1), projection("projection_y", 0)]
    extras = {}
    if shock.rotated is not None:
        overlays.extend(_segments(shock, frame))
        extras = {
            "rotated_phase": Frame.from_array(
                shock.rotated, shot=frame.shot, label="rotated phase"
            ),
            "summed_phase": Frame.from_array(
                shock.profile, shot=frame.shot, label="summed phase"
            ),
        }
    return Measurement(
        scalars=scalars,
        frame=stored,
        overlays=tuple(overlays),
        notes=notes,
        extras=extras,
    )


def _scalars(shock: ShockAnalysis) -> tuple[dict[str, float], tuple[str, ...]]:
    """The legacy result dict (the five shock values, then the slopes) and why any is NaN."""
    import numpy as np

    nan = float("nan")
    if shock.rotated is None:
        return dict.fromkeys(SHOCK_SCALARS, nan) | dict(shock.slopes), (
            "No signal after masking: the laser axis is undefined, shock analysis skipped",
        )
    gradient, plateau = shock.gradient, shock.plateau
    notes: tuple[str, ...] = ()
    if gradient.center is None:
        notes = (
            "No steepest window in the summed phase: shock slope, location and plateau undefined",
        )
    elif not np.isfinite(plateau.value):
        notes = ("Empty plateau window: plateau average and peak-to-plateau undefined",)
    values = {
        SHOCK_ANGLE: shock.angle.angle_rad,
        SHOCK_SLOPE: gradient.slope,
        SHOCK_LOCATION: nan if gradient.center is None else gradient.center,
        PLATEAU_AVG: nan if plateau is None else plateau.value,
        PEAK_TO_PLATEAU: nan if plateau is None else plateau.delta,
    }
    return {**values, **shock.slopes}, notes


def _segments(shock: ShockAnalysis, frame: Frame) -> list[Segment]:
    """The legacy figure's shock geometry, mapped from the rotated canvas onto the frame.

    The shock-angle fit line over the rows it was fitted on, the steepest
    column across the canvas, and the plateau window's two bounding
    columns — each a line on the rotated canvas, carried back through the
    rotation to the frame's coordinates (the canvas is larger, so an end
    may lie past the frame; the renderer clips).
    """
    import numpy as np

    from geecs_analysis.algorithms.downramp_shock import unrotate
    from geecs_analysis.measurement import Segment

    height = shock.rotated.shape[0] - 1
    lines: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    angle = shock.angle
    if angle.first_row is not None:
        lines["shock_fit"] = (
            (angle.first_row, angle.slope * angle.first_row + angle.intercept),
            (angle.last_row, angle.slope * angle.last_row + angle.intercept),
        )
    if shock.gradient.center is not None:
        column = shock.gradient.center
        lines["shock_location"] = ((0, column), (height, column))
        plateau = shock.plateau
        if plateau.stop > plateau.start:
            lines["plateau_start"] = ((0, plateau.start), (height, plateau.start))
            lines["plateau_end"] = ((0, plateau.stop), (height, plateau.stop))
    segments = []
    for name, ends in lines.items():
        (r0, c0), (r1, c1) = unrotate(
            np.array(ends), shock.laser_angle_deg, frame.data.shape, shock.rotated.shape
        )
        y, x = frame.axes
        segments.append(Segment(name, _at(x, c0), _at(y, r0), _at(x, c1), _at(y, r1)))
    return segments


def _at(axis: Axis, index: float) -> float:
    """A fractional sample index as the axis's coordinate (affine in the sample spacing)."""
    values = axis.values
    spacing = float(values[1] - values[0]) if values.size > 1 else 1.0
    return float(values[0]) + index * spacing
