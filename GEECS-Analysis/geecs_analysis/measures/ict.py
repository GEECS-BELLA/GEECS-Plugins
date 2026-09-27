"""ICT charge from an oscilloscope voltage trace (the legacy ``ICT1DAnalyzer``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from pydantic import Field

from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement

#: The legacy analyzer's bare scalar keys, the space included.
SCALARS = ("charge_pC", "ICT Signal Peak_us")


class IctSpec(MeasureSpec):
    """ICT parameters; fields, defaults and bounds are the v2 ``ict`` analyzer's."""

    kind: Literal["ict"] = "ict"
    butterworth_order: int = Field(
        1, ge=1, description="Order of the low-pass Butterworth filter."
    )
    butterworth_crit_f: float = Field(
        0.125,
        gt=0,
        description="Normalised critical frequency of the low-pass filter (1 = Nyquist).",
    )
    calibration_factor: float = Field(
        0.1, description="ICT calibration factor, V·s per C."
    )
    dt: Optional[float] = Field(
        None,
        description="Sample interval in seconds; unset derives it from the trace's time axis.",
    )

    def emitted_scalars(self) -> frozenset[str]:
        """Every trace emits the charge and the pulse time."""
        return frozenset(SCALARS)


@measure(IctSpec, ndim={1})
def ict(frame: Frame, spec: IctSpec) -> Measurement:
    """Integrate the ICT pulse of the processed trace to a charge in pC.

    ``dt`` unset is the first sample spacing of the trace's time axis. A
    trace the algorithm cannot analyze yields NaN scalars with a note (the
    legacy analyzer wrote 0 pC, indistinguishable from no charge).

    The samples are the stored trace's (float32-rounded by default), filtered
    at float64: the legacy analyzer handed scipy the float32 array itself, so
    its low-pass ran in float32 and its charge differs from this one by about
    1e-7 relative. The algorithm is otherwise the legacy one bit for bit.
    """
    from geecs_analysis.algorithms.ict import apply_ict_analysis
    from geecs_analysis.measurement import Measurement

    if frame.data.ndim != 1:
        raise ValueError("ICT measurement requires a trace")
    dt = spec.dt
    if dt is None:
        times = frame.axes[0].values
        dt = float(times[1] - times[0])
    notes: tuple[str, ...] = ()
    try:
        charge, peak = apply_ict_analysis(
            data=frame.data,
            dt=dt,
            butterworth_order=spec.butterworth_order,
            butterworth_crit_f=spec.butterworth_crit_f,
            calibration_factor=spec.calibration_factor,
        )
    except ValueError as exc:
        charge = peak = float("nan")
        notes = (f"ICT analysis failed: {exc}",)
    return Measurement(
        scalars={"charge_pC": charge, "ICT Signal Peak_us": peak},
        frame=frame,
        notes=notes,
    )
