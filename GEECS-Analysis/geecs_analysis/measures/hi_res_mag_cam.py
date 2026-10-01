"""HTU HiResMagCam: beam statistics plus a bow-tie fit of the dispersed trace.

The legacy ``HiResMagCamAnalyzer`` (a ``BeamAnalyzer`` with the bow-tie fit
bolted on). The emittance proxy keeps its optimizer contract; the fit's
waist column and companions are new scalars, so a per-bin run yields the
imaged-energy column once and a per-shot recipe can read the beam at it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Mapping

from pydantic import Field

from geecs_analysis.measures.beam import BEAM_SCALAR_DOCS, BeamSpec
from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement

#: The legacy analyzer zeroes the processed frame below this many counts
#: before the fit and the ``total_counts`` sum (a fixed floor, not a setting).
FIT_FLOOR = 10.0
#: The fit's rejection sentinel: the ``emittance_proxy`` the optimizer reads
#: as "no fit" (too few columns, or too little weight near the waist).
REJECTED = 1e6
#: The bow-tie scalars beside the beam statistics, in emission order.
BOWTIE_SCALARS = (
    "emittance_proxy",
    "total_counts",
    "bowtie_x0",
    "bowtie_w0",
    "bowtie_theta",
    "bowtie_r_squared",
)


class HiResMagCamSpec(MeasureSpec):
    """Bow-tie fit parameters; defaults and bounds are the v2 ``hi_res_mag_cam`` analyzer's."""

    kind: Literal["hi_res_mag_cam"] = "hi_res_mag_cam"
    scalar_docs: ClassVar[Mapping[str, str]] = {
        # The beam statistics it runs at their defaults: no slopes.
        **{k: v for k, v in BEAM_SCALAR_DOCS.items() if "_slope_" not in k},
        "emittance_proxy": (
            "w0 times |theta|, the value the optimizer minimises; 1e6 when the "
            "fit is rejected, NaN when it fails"
        ),
        "total_counts": "Sum of the pixels at or above the fit's 10-count floor",
        "bowtie_x0": "Waist position along x, in the frame's x units; NaN when the fit is rejected",
        "bowtie_w0": "Vertical RMS beam size at the waist, in px; NaN when the fit is rejected",
        "bowtie_theta": (
            "Divergence: growth of the vertical RMS size per px along x; NaN when "
            "the fit is rejected"
        ),
        "bowtie_r_squared": "Fit quality, 1 is perfect; NaN when the fit is rejected",
    }
    n_beam_size_clearance: int = Field(
        4,
        ge=0,
        description=(
            "Vertical sigmas of clearance a column's beam needs inside the frame "
            "to enter the fit; columns clipped by the edge are left out."
        ),
    )
    min_total_counts: float = Field(
        2500.0,
        ge=0,
        description="Columns with fewer total counts than this are left out of the fit.",
    )

    def emitted_scalars(self) -> frozenset[str]:
        """The full beam statistics and the bow-tie metrics."""
        return BeamSpec().emitted_scalars() | frozenset(BOWTIE_SCALARS)


@measure(HiResMagCamSpec, ndim={2})
def hi_res_mag_cam(frame: Frame, spec: HiResMagCamSpec) -> Measurement:
    """The ``beam`` measurement of the frame, then the bow-tie fit of its columns.

    The beam statistics and their overlays are the ``beam`` measure's own
    (called, not copied). The fit and ``total_counts`` see the frame floored
    at ``FIT_FLOOR`` counts, as the legacy analyzer did; the measured frame
    itself is unchanged. ``emittance_proxy`` is the fit score with its
    legacy failure values (the ``1e6`` sentinel, NaN on a fit exception).
    The waist column ``bowtie_x0`` is the fit's column mapped through the
    frame's column axis — affinely, so a waist the fit places a few columns
    past the frame (it accepts one within ten columns of data) reads
    beyond the edge as the legacy ``x0 + x_min`` did, never clamped to it.
    It, the waist size ``bowtie_w0``, the divergence ``bowtie_theta`` and
    ``bowtie_r_squared`` are NaN, with a note, whenever the fit is
    rejected — never the sentinel, so averaging them over shots or bins
    excludes failed fits by itself.
    """
    import numpy as np
    from geecs_data_utils.frames import Frame

    from geecs_analysis.algorithms.bowtie_fit import BowtieFitAlgorithm
    from geecs_analysis.measurement import Measurement, Projection
    from geecs_analysis.measures.beam import beam

    if frame.data.ndim != 2:
        raise ValueError("HiResMagCam measurement requires a 2D frame")
    stats = beam(frame, BeamSpec())

    floored = frame.data.copy()
    floored[floored < FIT_FLOOR] = 0
    fit = BowtieFitAlgorithm(
        n_beam_size_clearance=spec.n_beam_size_clearance,
        min_total_counts=spec.min_total_counts,
    ).evaluate(floored)
    accepted = np.isfinite(fit.score) and fit.score != REJECTED
    notes: tuple[str, ...] = ()
    if not accepted:
        notes = (
            "Bow-tie fit rejected: emittance_proxy is the 1e6 sentinel"
            if fit.score == REJECTED
            else "Bow-tie fit failed",
        )
    columns = frame.axes[1].values
    spacing = float(columns[1] - columns[0]) if columns.size > 1 else 1.0
    scalars = dict(stats.scalars)
    scalars.update(
        {
            "emittance_proxy": fit.score,
            "total_counts": float(np.sum(floored)),
            "bowtie_x0": float(columns[0]) + fit.x0 * spacing
            if accepted
            else float("nan"),
            "bowtie_w0": fit.w0 if accepted else float("nan"),
            "bowtie_theta": fit.theta if accepted else float("nan"),
            "bowtie_r_squared": fit.r_squared if accepted else float("nan"),
        }
    )
    # The legacy render overlay: the per-column weights the fit used.
    weights = Projection(
        "bowtie_weights",
        1,
        Frame.from_array(
            np.asarray(fit.weights, dtype=float),
            axes=(frame.axes[1],),
            shot=frame.shot,
            unit=frame.unit,
        ),
    )
    return Measurement(
        scalars=scalars,
        frame=stats.frame,
        overlays=stats.overlays + (weights,),
        notes=notes,
    )
