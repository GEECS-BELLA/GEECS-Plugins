"""Ordinary least-squares straight line with standard errors and zero crossing.

The numbers behind the ``scalar_fit`` summary: slope and intercept from the
centred normal equations, their covariance from the residual variance, and
the zero crossing ``-intercept / slope`` with its error propagated to first
order through that covariance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: The keys of the numbers :func:`linear_fit` returns, in a fixed order.
FIT_SUFFIXES = (
    "slope",
    "slope_stderr",
    "intercept",
    "intercept_stderr",
    "zero_crossing",
    "zero_crossing_stderr",
    "r2",
    "points",
)


def linear_fit(
    x: Sequence[float], y: Sequence[float], key: str
) -> tuple[dict[str, float], list[str]]:
    """Fit ``y = slope * x + intercept`` over the finite pairs.

    Three or more points give standard errors from the residual variance;
    two give an exact line with NaN errors; fewer, or positions that do not
    vary, give NaN throughout. A zero or nonfinite slope leaves the zero
    crossing NaN.

    Parameters
    ----------
    x, y : sequence of float
        Paired positions and values; a pair with a nonfinite member is
        dropped.
    key : str
        Name of the fitted quantity, used only in the notes.

    Returns
    -------
    fit : dict of str to float
        One number per :data:`FIT_SUFFIXES` entry (unprefixed); ``points``
        counts the finite pairs.
    notes : list of str
        One line per NaN among the numbers, saying why.
    """
    nan = float("nan")
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    n = int(xs.size)
    out = {suffix: nan for suffix in FIT_SUFFIXES}
    out["points"] = float(n)
    notes: list[str] = []
    if n < 2:
        notes.append(f"{key}: {n} finite point(s); no line is fitted")
        return out, notes
    # Centred normal equations: exact on an exact line (a flat one gives a
    # slope of exactly zero), and the textbook OLS covariance.
    x_mean, y_mean = xs.mean(), ys.mean()
    dx, dy = xs - x_mean, ys - y_mean
    sxx = float(dx @ dx)
    if sxx <= 0:
        notes.append(f"{key}: the positions do not vary; no line is fitted")
        return out, notes
    slope = float(dx @ dy) / sxx
    intercept = float(y_mean - slope * x_mean)
    residual = ys - (slope * xs + intercept)
    ssr = float(residual @ residual)
    sst = float(dy @ dy)
    if n >= 3:
        variance = ssr / (n - 2)
        cov = variance * np.array(
            [[1.0 / sxx, -x_mean / sxx], [-x_mean / sxx, 1.0 / n + x_mean**2 / sxx]]
        )
    else:
        cov = np.full((2, 2), nan)
        notes.append(f"{key}: 2 points; the line is exact and has no standard errors")
    out.update(
        slope=float(slope),
        intercept=float(intercept),
        slope_stderr=float(np.sqrt(cov[0, 0])),
        intercept_stderr=float(np.sqrt(cov[1, 1])),
        r2=1.0 - ssr / sst if sst > 0 else nan,
    )
    if sst <= 0:
        notes.append(f"{key}: the values do not vary; r2 is undefined")
    if slope == 0 or not np.isfinite(slope):
        notes.append(f"{key}: the slope is {slope}; no zero crossing")
        return out, notes
    gradient = np.array([intercept / slope**2, -1.0 / slope])
    out["zero_crossing"] = float(-intercept / slope)
    out["zero_crossing_stderr"] = float(np.sqrt(gradient @ cov @ gradient))
    return out, notes
