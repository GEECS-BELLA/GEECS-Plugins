"""Straight-line fits of measured scalars against the scan position.

For each named scalar the points are ``(positions[i], results[i].scalars[key])``
over the bins; the finite pairs are fitted with :func:`numpy.polyfit` and
its covariance (:func:`fit_line`). The fit's numbers are scan-level results
with no per-shot home, so the layout returns them beside the figure
(:class:`~geecs_analysis.registry.SummaryOutput`) and the sink writes them
as a JSON sidecar. Undefined numbers stay NaN and are explained in the
notes; nothing is replaced by zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from geecs_schemas.analysis.recipe import ScalarFitSummary

from geecs_analysis.registry import SummaryOutput, summary
from geecs_analysis.render import RenderError

if TYPE_CHECKING:
    from geecs_analysis.measurement import Measurement
    from geecs_analysis.render.specs import FigureSpec

#: The numbers written per fitted scalar, as ``{key}_{suffix}``.
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


def fit_line(
    x: Sequence[float], y: Sequence[float], key: str
) -> tuple[dict[str, float], list[str]]:
    """Fit ``y = slope * x + intercept`` over the finite pairs with numpy.

    :func:`numpy.polyfit` gives the line and, from three points on, its
    covariance (residual variance over ``n - 2`` degrees of freedom); two
    points give an exact line with NaN errors; fewer, or positions that
    do not vary, give NaN throughout. The zero crossing is
    ``-intercept / slope`` with its error propagated to first order through
    the covariance; it is NaN when the values do not vary or the slope is
    zero or nonfinite.

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
    import numpy as np

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
    if np.ptp(xs) == 0:
        notes.append(f"{key}: the positions do not vary; no line is fitted")
        return out, notes
    if n >= 3:
        (slope, intercept), cov = np.polyfit(xs, ys, 1, cov=True)
    else:
        slope, intercept = np.polyfit(xs, ys, 1)
        cov = np.full((2, 2), nan)
        notes.append(f"{key}: 2 points; the line is exact and has no standard errors")
    residual = ys - (slope * xs + intercept)
    ssr = float(residual @ residual)
    sst = float(((ys - ys.mean()) ** 2).sum())
    out.update(
        slope=float(slope),
        intercept=float(intercept),
        slope_stderr=float(np.sqrt(cov[0, 0])),
        intercept_stderr=float(np.sqrt(cov[1, 1])),
        r2=1.0 - ssr / sst if sst > 0 else nan,
    )
    if sst <= 0:
        notes.append(f"{key}: the values do not vary; r2 is undefined")
    if sst <= 0 or slope == 0 or not np.isfinite(slope):
        notes.append(f"{key}: the slope is {float(slope):.3g}; no zero crossing")
        return out, notes
    gradient = np.array([intercept / slope**2, -1.0 / slope])
    out["zero_crossing"] = float(-intercept / slope)
    out["zero_crossing_stderr"] = float(np.sqrt(gradient @ cov @ gradient))
    return out, notes


@summary(ScalarFitSummary, consumes="panels", filename="summary_scalar_fit")
def scalar_fit(
    results: Sequence[Measurement],
    positions: Sequence[float | None],
    label: str,
    options: ScalarFitSummary,
    figure: FigureSpec,
) -> SummaryOutput:
    """Fit each named scalar against the position and draw points and lines.

    ``label`` names the scan parameter (the x axis). A key missing from a
    result is a NaN point and one note naming the key; a position that is
    absent is a NaN point. From ``figure`` this layout takes ``fig`` (dpi
    and the rest, not figsize), as the waterfall does. Returns the figure
    with ``{key}_{suffix}`` numbers for every :data:`FIT_SUFFIXES` entry.
    A zero crossing is drawn as a dashed line only inside the scanned range.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if len(results) != len(positions):
        raise RenderError("Scalar fit requires one position per measurement")
    x = np.array([np.nan if p is None else float(p) for p in positions])
    scalars: dict[str, float] = {}
    notes: list[str] = []
    fits = []
    for key in options.scalars:
        missing = sum(key not in r.scalars for r in results)
        if missing:
            notes.append(f"{key}: missing from {missing} of {len(results)} results")
        y = np.array([r.scalars.get(key, np.nan) for r in results], dtype=float)
        fit, fit_notes = fit_line(x, y, key)
        notes.extend(fit_notes)
        scalars.update({f"{key}_{suffix}": value for suffix, value in fit.items()})
        fits.append((key, y, fit))
    try:
        fig = Figure(
            **{
                "figsize": (8, 6),
                "dpi": 150,
                "constrained_layout": True,
                **{k: v for k, v in figure.fig.items() if k != "figsize"},
            }
        )
        ax = fig.subplots()
        finite_x = x[np.isfinite(x)]
        span = (
            np.linspace(finite_x.min(), finite_x.max(), 2)
            if finite_x.size
            else np.array([])
        )
        for key, y, fit in fits:
            legend = f"{key}: slope {fit['slope']:.4g}, zero {fit['zero_crossing']:.4g}"
            (points,) = ax.plot(x, y, "o", label=legend)
            if np.isfinite(fit["slope"]) and span.size:
                ax.plot(
                    span,
                    fit["slope"] * span + fit["intercept"],
                    "-",
                    color=points.get_color(),
                )
            # A crossing outside the scanned range is reported in the legend
            # and the numbers, but not drawn: a marker there would stretch the
            # x axis until the measured points collapse (a skew plane's
            # crossings lie tens of mm away).
            if span.size and span[0] <= fit["zero_crossing"] <= span[-1]:
                ax.axvline(
                    fit["zero_crossing"],
                    linestyle="--",
                    linewidth=1,
                    color=points.get_color(),
                )
        ax.axhline(0, color="0.6", linewidth=0.8)
        ax.set(xlabel=label, ylabel="value", title=f"Scalar fit: {label} scan")
        ax.legend(fontsize="small")
        FigureCanvasAgg(fig).draw()
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"{type(exc).__name__}: {exc}") from exc
    return SummaryOutput(figure=fig, scalars=scalars, notes=tuple(notes))
