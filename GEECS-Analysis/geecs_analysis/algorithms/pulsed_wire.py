"""Plateau means and kicks of a pulsed-wire first-field-integral trace.

The trace is the wire deflection versus time (position along the magnet
line): flat in the drift gaps, changing inside each element. An element's
kick is the plateau after it minus the plateau before it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


def plateau_means(
    coordinates: ArrayLike,
    samples: ArrayLike,
    windows: Sequence[tuple[float, float]],
) -> tuple[list[float], list[str]]:
    """Mean the finite samples whose coordinate lies inside each window.

    Parameters
    ----------
    coordinates : array_like
        The trace's axis coordinates, one per sample, in any order.
    samples : array_like
        The trace's values.
    windows : sequence of (start, end)
        Closed coordinate intervals, ``start < end``.

    Returns
    -------
    means : list of float
        One mean per window; NaN when the window holds no finite sample.
    notes : list of str
        Why a plateau is NaN, or that nonfinite samples were left out of it.
    """
    x = np.asarray(coordinates, dtype=np.float64)
    y = np.asarray(samples, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Coordinates and samples must be equal-length 1D arrays")
    means: list[float] = []
    notes: list[str] = []
    for index, (start, end) in enumerate(windows):
        inside = (x >= start) & (x <= end)
        values = y[inside]
        finite = values[np.isfinite(values)]
        label = f"window {index} [{start:g}, {end:g}]"
        if values.size == 0:
            means.append(float("nan"))
            notes.append(f"Pulsed wire: {label} contains no samples")
        elif finite.size == 0:
            means.append(float("nan"))
            notes.append(f"Pulsed wire: {label} contains only nonfinite samples")
        else:
            means.append(float(finite.mean()))
            if finite.size < values.size:
                notes.append(
                    f"Pulsed wire: {label} left out "
                    f"{values.size - finite.size} nonfinite samples"
                )
    return means, notes


def kicks(plateaus: Sequence[float]) -> list[float]:
    """Return each element's kick: the plateau after it minus the one before.

    A NaN plateau makes both kicks touching it NaN (IEEE arithmetic).
    """
    return [after - before for before, after in zip(plateaus[:-1], plateaus[1:])]
