"""Server-side Plot-tab figure authoring (plotly.py → the vendored renderer).

The Plot tab's traces and layout are authored HERE, in Python, and served
to the page as ready Plotly figure JSON — the page's job shrinks to
``Plotly.react(host, figure.data, figure.layout, PLOT_CONFIG)`` plus the
client-side ``display.layout`` passthrough (which deliberately stays in
the browser: the URL-carried patch is untrusted, and the deep-merge
prototype-pollution guard lives with it).

This module is pure: no FastAPI, no catalog — mappings of column name →
values in, :class:`plotly.graph_objects.Figure` out.  A pandas DataFrame
is a valid ``series`` mapping, so the "show the code" snippets call the
same functions on the reproduced ``frame``/``result`` and get the
*identical figure* the page renders — the reproducibility doctrine,
extended from the numbers to the plot.

The ``display`` mapping is the URL-carried plot-cosmetics JSON
(:func:`geecs_portal.analysis.parse_display` type-checks it at the
boundary).  Value semantics keep the client's historical degrade
behavior: a non-hex color entry or a non-positive marker size falls back
to the default rather than erroring — display state rides shared links,
and a cosmetic value should never make a link fail.  The ``layout`` key
is carried but never applied here (client-side passthrough, above).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import plotly.graph_objects as go
import plotly.io as pio

#: Trace palette — the template injects this into the page so the rail's
#: column chips stay color-matched to the server-authored traces.
TRACE_COLORS: tuple[str, ...] = ("#4cc2b4", "#d6a860", "#6f9fd8", "#c47ab8")

_GRID = "#2c353d"


def _bare_figure() -> go.Figure:
    """A figure with the "none" template pinned.

    plotly.py otherwise stamps its full light-theme default template
    into ``to_plotly_json()`` — kilobytes of styling the page never had
    (the pre-0.10.0 client-built layouts were template-free), and a
    second styling authority fighting the explicit BASE_LAYOUT.  Pinning
    "none" keeps the served JSON template-free in effect AND makes a
    notebook ``fig.show()`` render the same figure as the page.
    """
    return go.Figure(layout={"template": pio.templates["none"]})


#: The shared base layout (ported verbatim from run.html's PLOT_LAYOUT).
BASE_LAYOUT: dict = {
    "paper_bgcolor": "#1a2026",
    "plot_bgcolor": "#12161a",
    "font": {"color": "#dde4ea", "size": 12},
    "margin": {"t": 24, "r": 56, "b": 44, "l": 56},
    "xaxis": {"gridcolor": _GRID, "zerolinecolor": _GRID, "automargin": True},
    "showlegend": True,
    "legend": {"orientation": "h", "y": 1.08},
}


def trace_color(display: Optional[Mapping], i: int) -> str:
    """The i-th trace color: a valid custom hex wins, else the palette.

    Only a ``#rgb``-style hex may come through — the display JSON rides
    shared links, and anything else degrades to the palette (the same
    rule the page applies before colors reach attribute sinks).
    """
    colors = (display or {}).get("colors")
    if isinstance(colors, Sequence) and not isinstance(colors, str) and i < len(colors):
        candidate = colors[i]
        if isinstance(candidate, str) and _is_hex_color(candidate):
            return candidate
    return TRACE_COLORS[i % len(TRACE_COLORS)]


def _is_hex_color(value: str) -> bool:
    body = value[1:]
    return (
        value.startswith("#")
        and len(body) in (3, 4, 6, 8)
        and all(c in "0123456789abcdefABCDEF" for c in body)
    )


def _marker_size(display: Optional[Mapping]) -> float:
    size = (display or {}).get("msize")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        if math.isfinite(size) and size > 0:
            return float(size)
    return 5.0


def _axis_range(lo: Any, hi: Any, log: bool) -> Optional[list]:
    """Build a [lo, hi] axis range (log axes take exponents).

    Bad bounds mean "no explicit range" — autorange stands, mirroring
    the page.
    """
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in (lo, hi)
    ):
        return None
    if log:
        if lo <= 0 or hi <= 0:
            return None
        return [math.log10(lo), math.log10(hi)]
    return [lo, hi]


def _pretty(pretty: Optional[Mapping], name: str) -> str:
    return (pretty or {}).get(name, name)


def _multi_y_layout(
    y: Sequence[str],
    pretty: Optional[Mapping],
    display: Optional[Mapping],
) -> dict:
    """Build the stacked-axis ladder for up to four y columns.

    Axis 1 owns the grid and a colored title; axis 2 anchors right;
    axes 3–4 are free + autoshift with color-matched ticks only (a
    free-anchored axis's rotated title does not shift with it —
    measured, see #725).
    """
    layout: dict = {
        "yaxis": {
            "gridcolor": _GRID,
            "zerolinecolor": _GRID,
            "tickfont": {"color": trace_color(display, 0)},
            "title": {
                "text": _pretty(pretty, y[0]),
                "font": {"color": trace_color(display, 0)},
            },
            "automargin": True,
        },
        # One trace: the axis title says it all.
        "showlegend": len(y) > 1,
    }
    for i in range(1, len(y)):
        axis: dict = {
            "overlaying": "y",
            "side": "right" if i % 2 else "left",
            "automargin": True,
            "gridcolor": "rgba(0,0,0,0)",
            "zerolinecolor": _GRID,
            "tickfont": {"color": trace_color(display, i)},
        }
        if i >= 2:
            axis["anchor"] = "free"
            axis["autoshift"] = True
        else:
            axis["title"] = {
                "text": _pretty(pretty, y[i]),
                "font": {"color": trace_color(display, i)},
            }
        layout[f"yaxis{i + 1}"] = axis
    return layout


def _apply_display(
    layout: dict,
    display: Optional[Mapping],
    *,
    x_is_date: bool,
    y_is_date: bool,
) -> None:
    """Apply display log types and explicit ranges to the layout.

    Date axes take neither log type nor numeric ranges (the page's
    rule, ported).
    """
    d = display or {}
    if d.get("logx") and not x_is_date:
        layout["xaxis"]["type"] = "log"
    if d.get("logy") and not y_is_date:
        layout["yaxis"]["type"] = "log"
    x_range = _axis_range(
        d.get("xmin"), d.get("xmax"), bool(d.get("logx")) and not x_is_date
    )
    if x_range and not x_is_date:
        layout["xaxis"]["range"] = x_range
        layout["xaxis"]["autorange"] = False
    y_range = _axis_range(d.get("ymin"), d.get("ymax"), bool(d.get("logy")))
    if y_range and not y_is_date:
        layout["yaxis"]["range"] = y_range
        layout["yaxis"]["autorange"] = False


def _shot_axis(series: Mapping, y: Sequence[str], shot: Optional[Sequence]) -> Sequence:
    if shot is not None:
        return shot
    if "Shotnumber" in series:
        return series["Shotnumber"]
    if hasattr(series, "index"):
        # A DataFrame without a Shotnumber column: 1-based row labels,
        # matching the endpoint's index+1 shot key (a filtered frame
        # keeps original shot identities — never renumber).
        return [i + 1 for i in series.index]
    return list(range(1, len(series[y[0]]) + 1))


def shots_figure(
    series: Mapping[str, Sequence],
    y: Sequence[str],
    *,
    x: Optional[str] = None,
    shot: Optional[Sequence] = None,
    kinds: Optional[Mapping[str, str]] = None,
    pretty: Optional[Mapping[str, str]] = None,
    display: Optional[Mapping] = None,
) -> go.Figure:
    """The per-shot scatter figure — one markers trace per ``y`` column.

    Parameters
    ----------
    series : Mapping[str, Sequence]
        Column name → values.  The ``/api`` frame payload's ``series``
        dict and a pandas ``DataFrame`` both qualify.
    y : Sequence[str]
        The plotted columns, in trace order (max 4 upstream).
    x : str, optional
        X column name; absent (or missing from ``series``) falls back
        to the shot axis.
    shot : Sequence, optional
        The shot-number axis; defaults to a ``Shotnumber`` column, else
        1..n.
    kinds : Mapping[str, str], optional
        Column → ``"datetime"`` for columns served as ISO strings.
    pretty : Mapping[str, str], optional
        Column → display name for titles and the legend.
    display : Mapping, optional
        The URL-carried plot-cosmetics JSON (already type-checked).

    Returns
    -------
    plotly.graph_objects.Figure
        Exactly what the Plot tab renders.
    """
    # An unservable X falls back to the shot axis WITH its title — the
    # figure never claims an axis it did not draw.
    x = x if x and x in series else None
    x_values = series[x] if x else _shot_axis(series, y, shot)
    x_is_date = bool(x) and (kinds or {}).get(x) == "datetime"
    y_is_date = (kinds or {}).get(y[0]) == "datetime"
    fig = _bare_figure()
    for i, name in enumerate(y):
        fig.add_scatter(
            x=list(x_values),
            y=list(series[name]),
            mode="markers",
            name=_pretty(pretty, name),
            marker={"color": trace_color(display, i), "size": _marker_size(display)},
            yaxis="y" if i == 0 else f"y{i + 1}",
        )
    layout = {**BASE_LAYOUT, "xaxis": dict(BASE_LAYOUT["xaxis"])}
    layout["xaxis"]["title"] = {"text": _pretty(pretty, x) if x else "shot #"}
    if x_is_date:
        layout["xaxis"]["type"] = "date"  # ISO strings from the API
    layout.update(_multi_y_layout(y, pretty, display))
    _apply_display(layout, display, x_is_date=x_is_date, y_is_date=y_is_date)
    fig.update_layout(layout)
    return fig


def binned_figure(
    bins: Sequence,
    series: Mapping[str, Mapping[str, Sequence]],
    y: Sequence[str],
    *,
    bin_col: str = "Bin #",
    pretty: Optional[Mapping[str, str]] = None,
    display: Optional[Mapping] = None,
) -> go.Figure:
    """The binned figure — centers, lines, and asymmetric error bars.

    Parameters
    ----------
    bins : Sequence
        Bin labels (the x axis).
    series : Mapping[str, Mapping[str, Sequence]]
        Column → ``{"center": …, "err_low": …, "err_high": …}`` — the
        ``/api`` binned payload's ``series`` shape.
    y : Sequence[str]
        The plotted columns, in trace order.
    bin_col : str
        The x-axis title (the binning column).
    pretty, display
        As in :func:`shots_figure`.

    Returns
    -------
    plotly.graph_objects.Figure
        Exactly what the Plot tab renders in binned view.
    """
    fig = _bare_figure()
    for i, name in enumerate(y):
        s = series.get(name) or {"center": [], "err_low": [], "err_high": []}
        fig.add_scatter(
            x=list(bins),
            y=list(s["center"]),
            mode="markers+lines",
            name=_pretty(pretty, name),
            marker={
                "color": trace_color(display, i),
                "size": _marker_size(display) + 1,
            },
            line={"color": trace_color(display, i), "width": 1},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": list(s["err_high"]),
                "arrayminus": list(s["err_low"]),
                "color": trace_color(display, i),
                "thickness": 1,
            },
            yaxis="y" if i == 0 else f"y{i + 1}",
        )
    layout = {**BASE_LAYOUT, "xaxis": dict(BASE_LAYOUT["xaxis"])}
    layout["xaxis"]["title"] = {"text": bin_col}
    layout.update(_multi_y_layout(y, pretty, display))
    # Binned serves raw numbers — no date axes on either side.
    _apply_display(layout, display, x_is_date=False, y_is_date=False)
    fig.update_layout(layout)
    return fig
