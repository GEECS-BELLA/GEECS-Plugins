"""Discharge current and laser timing from a current trace and a photodiode trace.

The LabVIEW "open scope trace file and calc delay" VI's core, without its
cable-charging thresholds. Both traces arrive on their own time axes in ns,
each already shifted by its channel's offset (cable length, optical delay),
so times on the two are comparable. The current is in A, the photodiode in
any unit; no baseline is subtracted.

- The **peak** is the current's maximum and its time.
- The **start** of the discharge is where the current, walking back from
  its peak, last rises through ``start_fraction`` of the peak (the "2nd I
  threshold"), interpolated linearly between samples.
- The laser's **arrival** is where the photodiode, walking back from its
  peak, last rises through ``laser_fraction`` of its peak, interpolated the
  same way — only when the photodiode's peak reaches ``laser_min_amplitude``
  (in the photodiode's unit), so a shot without the laser is not timed on
  noise.

A crossing that does not exist (the trace never falls below the level before
its peak) or a missing photodiode yields NaN for every quantity that depends
on it. An arrival outside the current trace's time span leaves only
``current_at_laser_A`` NaN; the delays are still differences of times.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DischargeDelay:
    """One shot's discharge current and laser timing; times in ns, currents in A."""

    peak_current_A: float
    peak_time_ns: float
    start_time_ns: float
    laser_time_ns: float
    current_at_laser_A: float
    laser_minus_start_ns: float
    laser_minus_peak_ns: float

    def to_dict(self) -> dict[str, float]:
        """The quantities as bare scalar names."""
        return asdict(self)


def rising_crossing_before_peak(
    time: np.ndarray, values: np.ndarray, fraction: float
) -> float:
    """Time at which *values* last rises through ``fraction`` of its peak before it.

    Walks back from the maximum to the last sample below the level and
    interpolates linearly between it and the next sample. NaN when no sample
    before the peak is below the level.
    """
    peak = int(np.argmax(values))
    level = fraction * values[peak]
    below = np.flatnonzero(values[:peak] < level)
    if below.size == 0:
        return float("nan")
    i = int(below[-1])
    v0, v1 = float(values[i]), float(values[i + 1])
    t0, t1 = float(time[i]), float(time[i + 1])
    return t0 + (level - v0) * (t1 - t0) / (v1 - v0)


def discharge_delay(
    time_ns: np.ndarray,
    current_A: np.ndarray,
    photodiode_time_ns: np.ndarray | None = None,
    photodiode: np.ndarray | None = None,
    *,
    start_fraction: float = 0.1,
    laser_fraction: float = 0.5,
    laser_min_amplitude: float | None = None,
) -> DischargeDelay:
    """Measure one shot: peak current, discharge start, laser arrival and delays.

    Parameters
    ----------
    time_ns, current_A : np.ndarray
        The current trace on its offset time axis (ascending), in ns and A.
    photodiode_time_ns, photodiode : np.ndarray or None
        The photodiode trace on its offset time axis; ``None`` leaves every
        laser quantity NaN.
    start_fraction : float
        Fraction of the peak current that marks the discharge's start.
    laser_fraction : float
        Fraction of the photodiode's peak that marks the laser's arrival.
    laser_min_amplitude : float or None
        The photodiode's peak must reach this for a laser to be timed;
        below it every laser quantity is NaN. ``None`` times any peak.
    """
    time_ns = np.asarray(time_ns, dtype=float)
    current_A = np.asarray(current_A, dtype=float)
    if time_ns.shape != current_A.shape or time_ns.ndim != 1 or time_ns.size < 2:
        raise ValueError("the current trace needs matching 1D time and value arrays")
    peak = int(np.argmax(current_A))
    peak_time = float(time_ns[peak])
    start_time = rising_crossing_before_peak(time_ns, current_A, start_fraction)

    laser_time = float("nan")
    if photodiode is not None and photodiode_time_ns is not None:
        pd_time = np.asarray(photodiode_time_ns, dtype=float)
        pd = np.asarray(photodiode, dtype=float)
        if pd_time.shape != pd.shape or pd.ndim != 1 or pd.size < 2:
            raise ValueError(
                "the photodiode trace needs matching 1D time and value arrays"
            )
        if laser_min_amplitude is None or np.max(pd) >= laser_min_amplitude:
            laser_time = rising_crossing_before_peak(pd_time, pd, laser_fraction)

    current_at_laser = (
        float(np.interp(laser_time, time_ns, current_A, left=np.nan, right=np.nan))
        if np.isfinite(laser_time)
        else float("nan")
    )
    return DischargeDelay(
        peak_current_A=float(current_A[peak]),
        peak_time_ns=peak_time,
        start_time_ns=start_time,
        laser_time_ns=laser_time,
        current_at_laser_A=current_at_laser,
        laser_minus_start_ns=laser_time - start_time,
        laser_minus_peak_ns=laser_time - peak_time,
    )
