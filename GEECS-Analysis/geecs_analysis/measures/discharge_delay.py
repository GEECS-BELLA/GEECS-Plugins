"""Discharge current: peak, start, and the laser's arrival against both.

The measured trace is the discharge current probe's voltage on a scope
channel; the laser's photodiode arrives as the line input's ``photodiode``
channel (the loading's ``auxiliary_columns``, e.g. ``{photodiode: 1}`` with
``trace_index: 3`` for a ``tek_scope_hdf5`` scope file). Each trace's time
axis is shifted by its channel's offset (cable length, optical delay)
before anything is compared; the current is the voltage times
``amps_per_volt``. Without a ``photodiode`` channel, or when the
photodiode's peak stays below ``photodiode_min_v`` (no laser on the shot),
the laser quantities are NaN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Mapping, Optional

from pydantic import Field

from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    import numpy as np
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement

#: The auxiliary channel holding the laser photodiode trace.
PHOTODIODE = "photodiode"

#: Nanoseconds per unit of a time axis.
_NS_PER = {"s": 1e9, "ms": 1e6, "us": 1e3, "\u00b5s": 1e3, "\u03bcs": 1e3, "ns": 1.0}

SCALARS: Mapping[str, str] = {
    "peak_current_A": "Maximum of the current trace (volts × amps_per_volt), A",
    "peak_time_ns": "Time of the peak current on the offset current axis, ns",
    "start_time_ns": (
        "Discharge start: where the current last rises through start_fraction "
        "of its peak before the peak (interpolated), ns"
    ),
    "laser_time_ns": (
        "Laser arrival: where the photodiode last rises through laser_fraction "
        "of its peak before the peak (interpolated), on the offset axis, ns"
    ),
    "current_at_laser_A": "Current at the laser's arrival, A",
    "laser_minus_start_ns": "Laser arrival minus the discharge start, ns",
    "laser_minus_peak_ns": "Laser arrival minus the time of the peak current, ns",
}


class DischargeDelaySpec(MeasureSpec):
    """Peak discharge current and the laser's timing against the discharge.

    Reads a scope's current channel as the trace and the laser photodiode
    as the ``photodiode`` channel: declare it in
    ``input.loading.auxiliary_columns`` (e.g. ``{photodiode: 1}``). The
    time axis must carry a time unit (s, ms, us or ns).
    """

    kind: Literal["discharge_delay"] = "discharge_delay"
    scalar_docs: ClassVar[Mapping[str, str]] = SCALARS
    current_offset_ns: float = Field(
        100.0, description="Added to the current channel's time axis, ns."
    )
    photodiode_offset_ns: float = Field(
        15.0, description="Added to the photodiode channel's time axis, ns."
    )
    amps_per_volt: float = Field(
        10.0,
        description="Current per volt on the current channel (probe and attenuators), A/V.",
    )
    start_fraction: float = Field(
        0.1,
        gt=0,
        lt=1,
        description="Fraction of the peak current that marks the discharge's start.",
    )
    laser_fraction: float = Field(
        0.5,
        gt=0,
        lt=1,
        description="Fraction of the photodiode's peak that marks the laser's arrival.",
    )
    photodiode_min_v: Optional[float] = Field(
        1.0,
        description=(
            "The photodiode's peak must reach this for the laser to be timed, V; "
            "below it the laser quantities are NaN. Unset times any peak."
        ),
    )
    median_window_ns: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Median-filter both traces over this window before measuring, ns; "
            "unset measures the raw traces."
        ),
    )

    def emitted_scalars(self) -> frozenset[str]:
        """The seven timing and current scalars."""
        return frozenset(SCALARS)


def _time_ns(frame: Frame, offset_ns: float) -> np.ndarray:
    """The frame's time axis in ns, shifted by *offset_ns*."""
    axis = frame.axes[0]
    scale = _NS_PER.get(axis.unit)
    if scale is None:
        raise ValueError(
            f"discharge_delay needs a time axis in s, ms, us or ns, not {axis.unit!r}"
        )
    return axis.values * scale + offset_ns


def _smoothed(values: np.ndarray, time_ns: np.ndarray, window_ns: float | None):
    """Median-filter *values* over *window_ns* (an odd sample count), or return them."""
    import numpy as np
    from scipy.ndimage import median_filter

    values = np.asarray(values, dtype=float)
    if window_ns is None or values.size < 2:
        return values
    step = float(np.median(np.abs(np.diff(time_ns))))
    size = max(1, int(round(window_ns / step))) | 1
    return median_filter(values, size=size, mode="nearest")


@measure(DischargeDelaySpec, ndim={1}, channels=(PHOTODIODE,))
def discharge_delay(
    frame: Frame, spec: DischargeDelaySpec, channels: Mapping[str, Frame]
) -> Measurement:
    """Measure the discharge current and, with a photodiode, the laser's timing.

    The processed frame is the current in A on the offset time axis in ns
    (smoothed when ``median_window_ns`` is set); the ``photodiode`` extra is
    the photodiode on its own offset axis. Markers: the peak, the start and
    the laser's arrival on the current, each omitted when undefined.
    """
    import numpy as np
    from geecs_data_utils.frames import Axis, Frame

    from geecs_analysis.algorithms.discharge_delay import (
        discharge_delay as measure_shot,
    )
    from geecs_analysis.measurement import Marker, Measurement

    if frame.data.ndim != 1 or frame.data.size < 2:
        raise ValueError("discharge_delay requires a trace of at least two samples")
    time_ns = _time_ns(frame, spec.current_offset_ns)
    current = _smoothed(frame.data, time_ns, spec.median_window_ns) * spec.amps_per_volt
    photodiode = channels.get(PHOTODIODE)
    pd_time = pd = None
    if photodiode is not None:
        pd_time = _time_ns(photodiode, spec.photodiode_offset_ns)
        pd = _smoothed(photodiode.data, pd_time, spec.median_window_ns)

    result = measure_shot(
        time_ns,
        current,
        pd_time,
        pd,
        start_fraction=spec.start_fraction,
        laser_fraction=spec.laser_fraction,
        laser_min_amplitude=spec.photodiode_min_v,
    )
    scalars = result.to_dict()

    def at(t: float) -> float:
        return float(np.interp(t, time_ns, current))

    overlays = [
        Marker(name, t, y)
        for name, t, y in (
            ("peak", result.peak_time_ns, result.peak_current_A),
            ("start", result.start_time_ns, at(result.start_time_ns)),
            ("laser", result.laser_time_ns, result.current_at_laser_A),
        )
        if np.isfinite(t) and np.isfinite(y)
    ]
    if photodiode is None:
        notes: tuple[str, ...] = ("No photodiode channel",)
    elif spec.photodiode_min_v is not None and np.max(pd) < spec.photodiode_min_v:
        notes = (f"Photodiode below {spec.photodiode_min_v} V: no laser timed",)
    elif not np.isfinite(result.laser_time_ns):
        notes = (
            "Photodiode never below laser_fraction of its peak before the peak: "
            "no laser timed",
        )
    else:
        notes = ()
    label = frame.axes[0].label or "Time"
    processed = Frame.from_array(
        current,
        axes=(Axis(time_ns, unit="ns", label=label),),
        shot=frame.shot,
        unit="A",
        label="Discharge current",
    )
    extras = {}
    if photodiode is not None:
        extras[PHOTODIODE] = Frame.from_array(
            pd,
            axes=(Axis(pd_time, unit="ns", label=label),),
            shot=photodiode.shot,
            unit=photodiode.unit,
            label="Photodiode",
        )
    return Measurement(
        scalars=scalars,
        frame=processed,
        overlays=tuple(overlays),
        notes=notes,
        extras=extras,
    )
