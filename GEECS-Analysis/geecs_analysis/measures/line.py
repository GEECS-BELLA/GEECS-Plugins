"""Legacy line statistics over caller-supplied calibrated trace coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement


class LineSpec(MeasureSpec):
    """Six legacy line statistics, with their original bare scalar names."""

    kind: Literal["line"] = "line"

    def emitted_scalars(self) -> frozenset[str]:
        """Expose scalar keys without importing the numerical algorithm."""
        return frozenset(
            {
                "CoM",
                "rms",
                "fwhm",
                "peak_location",
                "integrated_intensity",
                "peak_value",
            }
        )


@measure(LineSpec, ndim={1})
def line(frame: Frame, spec: LineSpec) -> Measurement:
    """Measure a private scratch trace; preserve the processed frame unchanged."""
    from geecs_analysis.algorithms.basic_line_stats import LineBasicStats
    from geecs_analysis.measurement import Measurement

    if frame.data.ndim != 1:
        raise ValueError("Line measurement requires a 1D frame")
    stats = LineBasicStats(line_data=frame.as_trace())
    return Measurement(scalars=stats.to_dict(), frame=frame)
