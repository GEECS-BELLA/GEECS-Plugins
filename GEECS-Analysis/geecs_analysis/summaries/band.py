"""A centre line with the shot-to-shot spread as a band, per bin or per scan.

The kind consumes every shot's trace (``consumes="shots"``), each at its
bin's position, and groups them by that position: one group per bin on a
parameter scan, one group (every position ``None``) on a noscan. Per
sample, over the shots whose value there is not NaN, ``mean`` draws the
mean ± ``sigma`` sample standard deviations (ddof 1) and ``median`` the
median between two percentiles (linear interpolation, numpy's default).

The figure is a picture, as the waterfall's is: a trace longer than
:data:`~geecs_analysis.summaries.waterfall.MAX_COLUMNS` samples is reduced
to block means by ``display_trace`` before the statistics, so the band is
the spread of those means, not of every recorded sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from geecs_schemas.analysis.recipe import BandSummary

from geecs_analysis.registry import SummaryOutput, summary
from geecs_analysis.render import RenderError
from geecs_analysis.summaries.waterfall import display_trace

if TYPE_CHECKING:
    import numpy as np
    from geecs_data_utils.frames import Frame
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from geecs_analysis.measurement import Measurement
    from geecs_analysis.render.specs import FigureSpec

#: The band colour when the recipe names none (overlay bands excepted).
DEFAULT_BAND_COLOR = "gray"

#: An overlay with more bins than this keys colours with a colorbar, not a legend.
MAX_LEGEND_ENTRIES = 10

#: Panel (width, height) in inches in the panels layout.
PANEL_SIZE = (4.0, 3.0)


@dataclass(frozen=True)
class Band:
    """One group's per-sample statistics: the line, the band edges, the counts.

    ``count`` is the number of shots whose sample is not NaN; a sample no
    shot reached is NaN in every array, and a ``mean`` band is NaN where
    fewer than two shots did (a standard deviation needs two).
    """

    center: np.ndarray
    low: np.ndarray
    high: np.ndarray
    count: np.ndarray


def band_statistics(
    data: np.ndarray,
    statistic: str = "mean",
    *,
    sigma: float = 1.0,
    percentiles: tuple[float, float] = (0.25, 0.75),
) -> Band:
    """Per-sample centre and band over the rows (shots) of ``data``, NaN-aware.

    ``statistic`` ``mean``: mean ± ``sigma`` sample standard deviations;
    ``median``: the median and the ``percentiles`` (fractions in [0, 1]).
    """
    import warnings

    import numpy as np

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("band_statistics takes a (shots, samples) array")
    count = np.sum(~np.isnan(data), axis=0)
    with warnings.catch_warnings():
        # All-NaN columns and single-shot deviations are NaN by design.
        warnings.simplefilter("ignore", RuntimeWarning)
        if statistic == "mean":
            center = np.nanmean(data, axis=0)
            spread = sigma * np.nanstd(data, axis=0, ddof=1)
            spread[count < 2] = np.nan
            return Band(center, center - spread, center + spread, count)
        if statistic == "median":
            low, center, high = np.nanquantile(
                data, [percentiles[0], 0.5, percentiles[1]], axis=0
            )
            return Band(center, low, high, count)
    raise ValueError(f"Unknown band statistic: {statistic!r}")


def group_by_position(
    frames: Sequence[Frame], positions: Sequence[float | None]
) -> list[tuple[float | None, list[Frame]]]:
    """Group the shots' frames by equal position, in order of first appearance."""
    import math

    groups: dict[float | None, list[Frame]] = {}
    for frame, position in zip(frames, positions):
        if position is not None and math.isnan(position):
            position = math.nan  # one key for every NaN (NaN != NaN)
        groups.setdefault(position, []).append(frame)
    return list(groups.items())


def _describe(options: BandSummary) -> str:
    if options.statistic == "mean":
        sigma = f"{options.sigma:g}"
        return f"mean ± {'' if sigma == '1' else sigma}σ"
    low, high = (round(p * 100, 2) for p in options.percentiles)
    return f"median, {low:g}–{high:g}th percentile band"


def _draw(
    ax: Axes,
    frames: list[Frame],
    options: BandSummary,
    *,
    line_style: dict,
    band_color: str,
    label: str | None = None,
) -> None:
    import numpy as np

    stats = band_statistics(
        np.stack([f.data for f in frames]),
        options.statistic,
        sigma=options.sigma,
        percentiles=options.percentiles,
    )
    x = np.asarray(frames[0].axes[0].values, dtype=np.float64)
    ax.fill_between(
        x,
        stats.low,
        stats.high,
        color=band_color,
        alpha=options.band_alpha,
        linewidth=0,
    )
    ax.plot(x, stats.center, label=label, **line_style)


def _labels(ax: Axes, frame: Frame, figure: FigureSpec) -> None:
    axis = frame.axes[0]
    xlabel = axis.label or "x"
    if axis.unit:
        xlabel += f" ({axis.unit})"
    signal = frame.label or "Signal"
    if frame.unit:
        signal += f" ({frame.unit})"
    ax.set(
        xlabel=figure.axes.get("xlabel", xlabel),
        ylabel=figure.axes.get("ylabel", signal),
    )


@summary(BandSummary, consumes="shots", filename="summary_band")
def band(  # noqa: C901, PLR0912, PLR0915
    results: Sequence[Measurement],
    positions: Sequence[float | None],
    label: str,
    options: BandSummary,
    figure: FigureSpec,
) -> Figure | SummaryOutput:
    """Draw each group's centre line and spread band; ``label`` names the parameter.

    Shots group by position (:func:`group_by_position`); a group with fewer
    than ``min_shots`` shots, traces of different lengths or a nonfinite
    position is left out, with a note (none left raises ``RenderError``). One group draws one
    axes; several draw ``options.layout``. From ``figure`` this layout
    takes ``fig`` (dpi, and figsize except in the panels layout),
    ``plot`` (the centre line's keywords; an overlay keeps its colours)
    and ``axes`` (xlabel, ylabel, title).
    """
    import math

    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    if not results or len(results) != len(positions):
        raise RenderError("Band requires one position per measurement")
    frames = [display_trace(r.frame) for r in results]
    if any(f.data.ndim != 1 for f in frames):
        raise RenderError("Band requires 1D traces")
    first = frames[0]
    if any(
        f.unit != first.unit or f.axes[0].unit != first.axes[0].unit for f in frames
    ):
        raise RenderError("Band requires matching trace units")
    grouped = group_by_position(frames, positions)
    notes, groups = [], []
    for position, members in grouped:
        where = "the scan" if position is None else f"{label} = {position:.4g}"
        if position is not None and not math.isfinite(position):
            notes.append(f"{where}: the position is not finite; not drawn")
        elif len({f.data.shape for f in members}) > 1:
            notes.append(f"{where}: traces of different lengths; not drawn")
        elif len(members) < options.min_shots:
            notes.append(
                f"{where}: {len(members)} shot(s), fewer than min_shots "
                f"{options.min_shots}; not drawn"
            )
        else:
            groups.append((position, members))
    if not groups:
        raise RenderError("No bin to draw: " + "; ".join(notes))
    groups.sort(key=lambda g: (g[0] is None, g[0] if g[0] is not None else 0.0))
    gray = options.band_color or DEFAULT_BAND_COLOR
    line = {"color": "C0", **figure.plot}
    described = _describe(options)
    shots = sum(len(m) for _, m in groups)
    panels = options.layout == "panels" and len(groups) > 1
    try:
        settings = {"figsize": (10, 6), "dpi": 150, "constrained_layout": True}
        settings.update(figure.fig)
        if panels:
            columns = min(
                options.columns or math.ceil(math.sqrt(len(groups))), len(groups)
            )
            rows = math.ceil(len(groups) / columns)
            settings["figsize"] = (columns * PANEL_SIZE[0], rows * PANEL_SIZE[1])
            fig = Figure(**settings)
            axes = fig.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
            for ax, (position, members) in zip(axes.flat, groups):
                _draw(ax, members, options, line_style=line, band_color=gray)
                value = "all shots" if position is None else f"{position:.4g}"
                ax.set_title(f"{value} ({len(members)} shots)", fontsize=10)
            for ax in axes.flat[len(groups) :]:
                ax.set_visible(False)
            for ax in axes[-1, :]:
                _labels(ax, first, figure)
            for ax in axes[:, 0]:
                _labels(ax, first, figure)
            for ax in axes.flat:
                if not ax.get_subplotspec().is_last_row():
                    ax.set_xlabel("")
                if not ax.get_subplotspec().is_first_col():
                    ax.set_ylabel("")
            fig.suptitle(
                figure.axes.get("title", f"{described}: {label} scan"), fontsize=12
            )
        else:
            fig = Figure(**settings)
            ax = fig.subplots()
            values = [p for p, _ in groups if p is not None]
            if len(groups) == 1:
                position, members = groups[0]
                _draw(ax, members, options, line_style=line, band_color=gray)
                where = "" if position is None else f", {label} = {position:.4g}"
                title = f"{described} over {len(members)} shots{where}"
            else:
                cmap = colormaps["viridis"]
                norm = Normalize(min(values), max(values))
                if norm.vmin == norm.vmax:
                    norm = Normalize(norm.vmin - 0.5, norm.vmax + 0.5)
                style = {
                    k: v for k, v in figure.plot.items() if k not in ("color", "c")
                }
                for position, members in groups:
                    color = cmap(norm(position)) if position is not None else "k"
                    _draw(
                        ax,
                        members,
                        options,
                        line_style={**style, "color": color},
                        band_color=options.band_color or color,
                        label=("all" if position is None else f"{position:.4g}"),
                    )
                if len(groups) <= MAX_LEGEND_ENTRIES:
                    ax.legend(title=label or None, fontsize=8)
                else:
                    fig.colorbar(ScalarMappable(norm, cmap), ax=ax, label=label)
                title = f"{described}: {label} scan, {shots} shots"
            _labels(ax, first, figure)
            ax.set_title(figure.axes.get("title", title))
        FigureCanvasAgg(fig).draw()
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"{type(exc).__name__}: {exc}") from exc
    if notes:
        return SummaryOutput(figure=fig, notes=tuple(notes))
    return fig
