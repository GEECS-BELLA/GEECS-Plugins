"""Pure figure authoring: columns in, a plotly ``Figure`` out.

The Plot page's figures: up to three Y columns against one X on
GEECSplotter's axis ladder (``_y_axes``), per shot or binned. The palette,
the page serializer, the display cosmetics and the binned trace are the
portal's (``geecs_portal.figures``), imported, so the two cannot drift;
the layout (no legend, the ladder) is this page's own. The fits (:func:`fit_columns`, the ``fits`` of :func:`shots_figure`) are
the plotter's own: GEECSplotter's Fit menu, the numbers from
``geecs_data_utils.data.curve_fit``.

The page asks for :data:`THEMED_PALETTE`, whose placeholder hex
:func:`page_figure` swaps for ``$tok:--name`` sentinels that the browser
resolves against the live theme tokens; a notebook takes the default
:data:`NOTEBOOK_PALETTE` and gets real hex. Everything else is the same
figure either way.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import plotly.graph_objects as go
import plotly.io as pio
from geecs_data_utils.data.curve_fit import CurveFit, fit_curve

# One palette, one serializer, one reading of the display cosmetics. The
# page's palette and serializer are re-exported for this page's routes.
from geecs_portal.figures import (
    NOTEBOOK_PALETTE,
    Palette,
    page_figure as page_figure,
    THEMED_PALETTE as THEMED_PALETTE,
    add_binned_trace,
    apply_display,
    axis_ticks,
    marker_size,
    trace_color,
)


#: GEECSplotter's Y1, Y2 and Y3: the most Y columns one figure draws.
MAX_Y = 3

#: Pixels between the inner right axis's title and the outer axis's ticks.
_OUTER_AXIS_GAP = 20

#: GEECSplotter's Fit menu, in its order and with its names, for the models
#: ``geecs_data_utils.data.curve_fit`` has so far; ``none`` is no fit.
FITS: dict[str, str] = {
    "none": "None",
    "gaussian": "Gaussian",
    "gaussian_no_offset": "GaussianNoOffset",
    "supergaussian": "SuperGaussian",
    "supergaussian_no_offset": "SuperGaussianNoOffset",
    "polynomial": "Polynomial",
}

#: Points along a drawn fit curve.
FIT_POINTS = 400


def base_layout(palette: Palette) -> dict:
    """The layout every figure starts from.

    Parameters
    ----------
    palette : Palette
        The colours.

    Returns
    -------
    dict
        A plotly layout mapping.
    """
    axis = {
        "gridcolor": palette.grid_soft,
        "zerolinecolor": palette.grid,
        "automargin": True,
        **axis_ticks(palette),
    }
    return {
        "template": pio.templates["none"],
        "paper_bgcolor": palette.paper,
        "plot_bgcolor": palette.plot,
        "font": {"color": palette.font, "size": 12},
        "margin": {"t": 24, "r": 56, "b": 44, "l": 56},
        "showlegend": False,
        "xaxis": dict(axis),
        "yaxis": dict(axis),
    }


def _y_axes(
    axes: Sequence[tuple[int, str]],
    palette: Palette,
    display: Optional[Mapping] = None,
) -> dict:
    """The y-axis ladder, one axis per trace, each in its trace's colour.

    After the portal's ``_multi_y_layout``, with the third axis moved:
    axis 1 (left) owns the grid; every other axis is on the right, with a
    coloured title. The first right-hand axis anchors to the plot; a second
    one is free at the plot's edge with ``autoshift``, which places it just
    past the first's tick labels and title (plus ``_OUTER_AXIS_GAP``)
    whatever the plot's width. Its title moves with it on the bundled
    plotly.js 3.1.1 (re-checked in the browser 2026-10-27, left and right
    sides), so the third label shows: the portal's #725 note that a free
    axis's title does not shift does not hold on this same bundle.
    A fixed share of the width, the first cut, left a wide gap on a wide
    window.

    *axes* is ``(slot, title)`` per drawn trace: the slot (0–2, the Y list
    it came from) decides colour and axis id, so Y3 is Y3 with Y2 off. A
    custom colour in *display* colours its axis as it does its trace.
    """
    layout: dict = {
        "showlegend": len(axes) > 1,
        # A row above the plot, clear of the right-hand axis.
        "legend": {"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
    }
    right = 0
    for i, title in axes:
        colour = trace_color(display, i, palette)
        axis: dict = {
            "automargin": True,
            "zerolinecolor": palette.grid,
            "tickfont": {"color": colour},
            "title": {"text": title, "font": {"color": colour}},
            **axis_ticks(palette),
        }
        if i == 0:
            axis["gridcolor"] = palette.grid_soft
        else:
            axis.update(overlaying="y", side="right", gridcolor="rgba(0,0,0,0)")
            if right:
                axis.update(
                    anchor="free", position=1.0, autoshift=True, shift=_OUTER_AXIS_GAP
                )
            right += 1
        layout["yaxis" if i == 0 else f"yaxis{i + 1}"] = axis
    return layout


def _x_values(
    series: Mapping[str, Sequence],
    y0: str,
    x: Optional[str],
    shot: Optional[Sequence],
) -> list:
    """The X the points are drawn (and fitted) against: the column, the shots, or 1..n."""
    if x and x in series:
        return list(series[x])
    if shot is not None:
        return list(shot)
    return list(range(1, len(series[y0]) + 1))


def fit_columns(
    series: Mapping[str, Sequence],
    y: Sequence[Optional[str]],
    models: Sequence[Optional[str]],
    *,
    x: Optional[str] = None,
    shot: Optional[Sequence] = None,
    orders: Sequence[int] = (),
) -> list[Optional[CurveFit]]:
    """Fit each Y slot against X with its model: GEECSplotter's per-plot Fit.

    Parameters
    ----------
    series, x, shot
        As :func:`shots_figure`: the points fitted are the points drawn.
    y : Sequence[str or None]
        The Y slots (``None`` is a list that is off).
    models : Sequence[str or None]
        One :data:`FITS` key per slot; ``None`` or ``"none"`` is no fit.
    orders : Sequence[int], optional
        The polynomial order per slot (default 2).

    Returns
    -------
    list of CurveFit or None
        One per slot, ``None`` where no fit was asked for or the list is off.
        A fit that could not be made has ``error`` set.
    """
    ys = list(y)
    x_values = _x_values(series, ys[0], x, shot)
    picked = list(models) + [None] * (len(ys) - len(models))
    degrees = list(orders) + [2] * (len(ys) - len(orders))
    fits: list[Optional[CurveFit]] = []
    for name, model, order in zip(ys, picked, degrees):
        if not name or model in (None, "none"):
            fits.append(None)
            continue
        fits.append(fit_curve(x_values, list(series[name]), model, order=order))
    return fits


def shots_figure(
    series: Mapping[str, Sequence],
    y: Union[str, Sequence[str]],
    *,
    x: Optional[str] = None,
    shot: Optional[Sequence] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    y_labels: Sequence[Optional[str]] = (),
    fits: Sequence[Optional[CurveFit]] = (),
    display: Optional[Mapping] = None,
    palette: Palette = NOTEBOOK_PALETTE,
) -> go.Figure:
    """The per-shot scatter: one marker trace per Y column, against ``x``.

    Parameters
    ----------
    series : Mapping[str, Sequence]
        Column name → values; a pandas ``DataFrame`` qualifies.
    y : str or Sequence[str or None]
        The plotted column, or GEECSplotter's Y1–Y3 in their slots, each on
        its own y axis in its slot's colour; ``None`` is a list that is off
        (``["a", None, "c"]`` keeps "c" on the third axis, third colour).
    x : str, optional
        The X column; absent (or missing from ``series``) means the shot
        number.
    shot : Sequence, optional
        The shot numbers; defaults to 1..n.
    x_label : str, optional
        The X title, when not the column name (the alias toggle).
    y_label : str, optional
        The first Y axis's title; shorthand for ``y_labels[0]``.
    y_labels : Sequence[str or None], optional
        One title per Y slot, ``None`` for the column name. Per axis, not
        per column: the same column may be on two axes.
    fits : Sequence[CurveFit or None], optional
        One per Y slot (:func:`fit_columns`): each fit made is drawn as a
        dashed line over its fitted range, on its slot's axis, in its
        slot's colour.
    display : Mapping, optional
        The plot cosmetics (``geecs_portal.analysis.parse_display``): log
        axes, X and first-Y ranges, marker size, plot size, a colour per
        slot. Its raw ``layout`` patch is the browser's to merge.
    palette : Palette, optional
        :data:`NOTEBOOK_PALETTE` (default) or :data:`THEMED_PALETTE`.

    Returns
    -------
    plotly.graph_objects.Figure
        Exactly what the page draws, up to palette.
    """
    ys = [y] if isinstance(y, str) else list(y)
    if not 1 <= len(ys) <= MAX_Y or not ys[0]:
        raise ValueError(f"plot 1 to {MAX_Y} Y columns, Y1 first, not {ys!r}")
    labels = list(y_labels) + [None] * (len(ys) - len(y_labels))
    if y_label is not None:
        labels[0] = y_label
    drawn = [(i, name, labels[i] or name) for i, name in enumerate(ys) if name]
    x = x if x and x in series else None
    x_values = _x_values(series, ys[0], x, shot)
    fig = go.Figure(layout={"template": pio.templates["none"]})
    for i, name, title in drawn:
        fig.add_scatter(
            x=x_values,
            y=list(series[name]),
            mode="markers",
            name=title,
            marker={
                "color": trace_color(display, i, palette),
                "size": marker_size(display),
            },
            yaxis="y" if i == 0 else f"y{i + 1}",
        )
    for i, _, title in drawn:
        fit = fits[i] if i < len(fits) else None
        if fit is None or not fit.ok:
            continue
        curve_x, curve_y = fit.curve(FIT_POINTS)
        fig.add_scatter(
            x=curve_x.tolist(),
            y=curve_y.tolist(),
            mode="lines",
            name=f"{title}: {FITS.get(fit.model, fit.model)} fit",
            line={
                "color": trace_color(display, i, palette),
                "dash": "dash",
                "width": 2,
            },
            hoverinfo="skip",
            yaxis="y" if i == 0 else f"y{i + 1}",
        )
    layout = base_layout(palette)
    layout["xaxis"]["title"] = {"text": (x_label or x) if x else "shot #"}
    del layout["yaxis"]
    layout.update(_y_axes([(i, title) for i, _, title in drawn], palette, display))
    apply_display(layout, display, x_is_date=False, y_is_date=False)
    fig.update_layout(layout)
    return fig


def binned_figure(
    bins: Sequence,
    series: Mapping[str, Mapping[str, Sequence]],
    y: Union[str, Sequence[Optional[str]]],
    *,
    bin_col: str = "Bin #",
    x_values: Optional[Sequence] = None,
    x_label: Optional[str] = None,
    y_labels: Sequence[Optional[str]] = (),
    display: Optional[Mapping] = None,
    palette: Palette = NOTEBOOK_PALETTE,
) -> go.Figure:
    """The binned plot: each bin's centre and error bars, Y1–Y3 in their slots.

    The portal's binned view (``geecs_portal.figures.binned_figure``) on
    this page's axis ladder: bins group the shots (``bin_col``), the X
    column's per-bin mean places them.

    Parameters
    ----------
    bins : Sequence
        The bin labels: the X axis when no ``x_values`` are given.
    series : Mapping[str, Mapping[str, Sequence]]
        Column → ``{"center", "err_low", "err_high"}``, one value per bin
        (``geecs_data_utils.data.binning.bin_frame``'s columns).
    y : str or Sequence[str or None]
        As :func:`shots_figure`: the Y slots, ``None`` for one off.
    bin_col : str, optional
        The X title when plotting against the bin labels.
    x_values : Sequence, optional
        Per-bin X positions, aligned one-to-one with ``bins``.
    x_label : str, optional
        The X title for ``x_values``.
    y_labels, display, palette
        As :func:`shots_figure`.

    Returns
    -------
    plotly.graph_objects.Figure
        Exactly what the page draws in the binned view, up to palette.
    """
    ys = [y] if isinstance(y, str) else list(y)
    if not 1 <= len(ys) <= MAX_Y or not ys[0]:
        raise ValueError(f"plot 1 to {MAX_Y} Y columns, Y1 first, not {ys!r}")
    labels = list(y_labels) + [None] * (len(ys) - len(y_labels))
    drawn = [(i, name, labels[i] or name) for i, name in enumerate(ys) if name]
    positions = list(x_values) if x_values is not None else list(bins)
    fig = go.Figure(layout={"template": pio.templates["none"]})
    for i, name, title in drawn:
        add_binned_trace(
            fig,
            positions,
            series.get(name),
            name=title,
            colour=trace_color(display, i, palette),
            display=display,
            yaxis="y" if i == 0 else f"y{i + 1}",
        )
    layout = base_layout(palette)
    layout["xaxis"]["title"] = {
        "text": (x_label or bin_col) if x_values is not None else bin_col
    }
    del layout["yaxis"]
    layout.update(_y_axes([(i, title) for i, _, title in drawn], palette, display))
    apply_display(layout, display, x_is_date=False, y_is_date=False)
    fig.update_layout(layout)
    return fig


def _overlay_layout(
    palette: Palette,
    display: Optional[Mapping],
    x_title: str,
    y_title: str,
) -> dict:
    """The overlay's layout: one Y axis, the scans named in a legend row."""
    layout = base_layout(palette)
    layout["xaxis"]["title"] = {"text": x_title}
    layout["yaxis"]["title"] = {"text": y_title}
    layout["showlegend"] = True
    layout["legend"] = {"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"}
    apply_display(layout, display, x_is_date=False, y_is_date=False)
    return layout


def overlay_figure(
    groups: Sequence[tuple[str, Mapping[str, Sequence], Sequence]],
    y: str,
    *,
    x: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    fits: Sequence[Optional[CurveFit]] = (),
    display: Optional[Mapping] = None,
    palette: Palette = NOTEBOOK_PALETTE,
) -> go.Figure:
    """Several scans' shots on one plot: one Y, a trace and a colour per scan.

    Parameters
    ----------
    groups : Sequence[tuple[str, Mapping, Sequence]]
        ``(legend name, column → values, shot numbers)`` per scan, in
        colour order (``plotter.multi.overlay_groups``).
    y : str
        The one Y column drawn.
    x, x_label, display, palette
        As :func:`shots_figure`; X absent is the shot number (each scan's
        own).
    y_label : str, optional
        The Y title, when not the column name.
    fits : Sequence[CurveFit or None], optional
        One per scan, drawn dashed in its scan's colour.

    Returns
    -------
    plotly.graph_objects.Figure
        Exactly what the page draws in the overlay, up to palette.
    """
    fig = go.Figure(layout={"template": pio.templates["none"]})
    for i, (label, series, shot) in enumerate(groups):
        colour = trace_color(None, i, palette)
        fig.add_scatter(
            x=_x_values(series, y, x, shot),
            y=list(series[y]),
            mode="markers",
            name=label,
            legendgroup=label,
            marker={"color": colour, "size": marker_size(display)},
        )
        fit = fits[i] if i < len(fits) else None
        if fit is not None and fit.ok:
            curve_x, curve_y = fit.curve(FIT_POINTS)
            fig.add_scatter(
                x=curve_x.tolist(),
                y=curve_y.tolist(),
                mode="lines",
                name=f"{label}: {FITS.get(fit.model, fit.model)} fit",
                legendgroup=label,
                showlegend=False,
                line={"color": colour, "dash": "dash", "width": 2},
                hoverinfo="skip",
            )
    x_title = (x_label or x) if x else "shot #"
    fig.update_layout(_overlay_layout(palette, display, x_title, y_label or y))
    return fig


def overlay_binned_figure(
    groups: Sequence[tuple[str, Sequence, Mapping[str, Mapping], Optional[Sequence]]],
    y: str,
    *,
    bin_col: str = "Bin #",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    display: Optional[Mapping] = None,
    palette: Palette = NOTEBOOK_PALETTE,
) -> go.Figure:
    """Several scans binned, each on its own: a centre-and-errors trace per scan.

    Parameters
    ----------
    groups : Sequence[tuple]
        ``(legend name, bin labels, {y: centre and errors}, X positions or
        None)`` per scan, in colour order (``plotter.multi.overlay_bins``).
    y : str
        The one Y column drawn.
    bin_col : str, optional
        The X title when the bins are placed by their labels.
    x_label : str, optional
        The X title when they are placed by an X column.
    y_label, display, palette
        As :func:`overlay_figure`.

    Returns
    -------
    plotly.graph_objects.Figure
        Exactly what the page draws in the binned overlay, up to palette.
    """
    fig = go.Figure(layout={"template": pio.templates["none"]})
    placed = False
    for i, (label, bins, series, x_values) in enumerate(groups):
        placed = placed or x_values is not None
        add_binned_trace(
            fig,
            list(x_values) if x_values is not None else list(bins),
            series.get(y),
            name=label,
            colour=trace_color(None, i, palette),
            display=display,
            yaxis="y",
        )
    x_title = (x_label or bin_col) if placed else bin_col
    fig.update_layout(_overlay_layout(palette, display, x_title, y_label or y))
    return fig
