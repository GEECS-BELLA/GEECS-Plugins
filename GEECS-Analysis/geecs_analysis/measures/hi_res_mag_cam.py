"""HTU HiResMagCam: beam statistics plus a bow-tie fit of the dispersed trace.

The legacy ``HiResMagCamAnalyzer`` (a ``BeamAnalyzer`` with the bow-tie fit
bolted on). The emittance proxy keeps its optimizer contract; the fit's
waist column and companions are new scalars, so a per-bin run yields the
imaged-energy column once and a per-shot recipe can read the beam at it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from geecs_analysis.measures.beam import BeamSpec
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
    """Beam statistics of the frame, then the bow-tie fit of its columns.

    Beam statistics use the frame's coordinates, as the ``beam`` measure
    does. The fit and ``total_counts`` see the frame floored at
    ``FIT_FLOOR`` counts, as the legacy analyzer did; the measured frame
    itself is unchanged. ``emittance_proxy`` is the fit score with its
    legacy failure values (the ``1e6`` sentinel, NaN on a fit exception).
    The waist column ``bowtie_x0`` (in the frame's column coordinates),
    the waist size ``bowtie_w0``, the divergence ``bowtie_theta`` and
    ``bowtie_r_squared`` are NaN, with a note, whenever the fit is
    rejected — never the sentinel, so averaging them over shots or bins
    excludes failed fits by itself.
    """
    import numpy as np
    from geecs_data_utils.frames import Frame

    from geecs_analysis.algorithms.basic_beam_stats import (
        beam_profile_stats,
        flatten_beam_stats,
    )
    from geecs_analysis.algorithms.bowtie_fit import BowtieFitAlgorithm
    from geecs_analysis.measurement import Marker, Measurement, Projection

    if frame.data.ndim != 2:
        raise ValueError("HiResMagCam measurement requires a 2D frame")
    stats = beam_profile_stats(frame.data, tuple(axis.values for axis in frame.axes))
    scalars = flatten_beam_stats(stats)

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
    scalars.update(
        {
            "emittance_proxy": fit.score,
            "total_counts": float(np.sum(floored)),
            "bowtie_x0": float(np.interp(fit.x0, np.arange(columns.size), columns))
            if accepted
            else float("nan"),
            "bowtie_w0": fit.w0 if accepted else float("nan"),
            "bowtie_theta": fit.theta if accepted else float("nan"),
            "bowtie_r_squared": fit.r_squared if accepted else float("nan"),
        }
    )

    overlays: list[Projection | Marker] = [
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
        # The legacy render overlay: the per-column weights the fit used.
        Projection(
            "bowtie_weights",
            1,
            Frame.from_array(
                np.asarray(fit.weights, dtype=float),
                axes=(frame.axes[1],),
                shot=frame.shot,
                unit=frame.unit,
            ),
        ),
    ]
    if np.isfinite(stats.x.CoM) and np.isfinite(stats.y.CoM):
        overlays.append(Marker("com", stats.x.CoM, stats.y.CoM))
    return Measurement(
        scalars=scalars, frame=frame, overlays=tuple(overlays), notes=notes
    )
