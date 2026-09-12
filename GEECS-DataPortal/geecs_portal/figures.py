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
*identical figure* the page renders, up to palette — the reproducibility
doctrine, extended from the numbers to the plot.  "Up to palette" is
precise: the page asks for :data:`THEMED_PALETTE`, whose colours are
``$tok:--name`` sentinels the browser resolves against the live theme
tokens in one walk before ``Plotly.react``; a notebook takes the default
:data:`NOTEBOOK_PALETTE` and gets real hex.  Shapes, traces, axes and
every other property are the same object either way.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pandas is runtime-optional here (local imports)
    import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio

#: The notebook trace palette — still injected into the page as the
#: server's list (a pinned contract) and the JS-off / notebook fallback.
#: On the page the chips and traces both come from the theme's
#: ``--trace-1..4`` instead, so they cannot drift apart.
TRACE_COLORS: tuple[str, ...] = ("#4cc2b4", "#d6a860", "#6f9fd8", "#c47ab8")


@dataclass(frozen=True)
class Palette:
    """Every colour a figure needs, in one place.

    Two instances exist. :data:`NOTEBOOK_PALETTE` is real hex, for
    ``fig.show()`` outside the page. :data:`THEMED_PALETTE` is sentinel
    strings of the form ``$tok:--name`` that the page resolves against the
    live theme tokens — one client-side walk covers traces, tick fonts,
    grids, axis titles and the legend alike, so no per-property list here
    or in the template can fall out of step.
    """

    trace: tuple[str, ...]
    grid: str
    grid_soft: str
    font: str
    paper: str
    plot: str


NOTEBOOK_PALETTE = Palette(
    trace=TRACE_COLORS,
    grid="#2c353d",
    # Gridlines one step subtler than the axis furniture — the Vega-Lite
    # look the owner liked in the renderer bake-off (2026-08-31 ruling).
    grid_soft="#232a31",
    font="#dde4ea",
    paper="#1a2026",
    plot="#12161a",
)

#: What the page asks for. plotly.py validates every colour property at
#: figure-build time and rejects a ``$tok:`` string outright, so this
#: palette is *placeholder hex* — nine near-black values no theme uses —
#: and :func:`page_figure` swaps them for ``$tok:--name`` sentinels in
#: the serialized JSON. The browser resolves those against the live
#: tokens. A user display colour that happened to equal a placeholder
#: would be re-themed; ``#010101``–``#010109`` are not colours anyone
#: picks.
THEMED_PALETTE = Palette(
    trace=("#010101", "#010102", "#010103", "#010104"),
    grid="#010105",
    grid_soft="#010106",
    font="#010107",
    paper="#010108",
    plot="#010109",
)

#: Placeholder → sentinel, the substitution :func:`page_figure` applies.
SENTINELS: dict[str, str] = {
    "#010101": "$tok:--trace-1",
    "#010102": "$tok:--trace-2",
    "#010103": "$tok:--trace-3",
    "#010104": "$tok:--trace-4",
    "#010105": "$tok:--rule",
    "#010106": "$tok:--rule-soft",
    "#010107": "$tok:--ink",
    "#010108": "$tok:--surface",
    "#010109": "$tok:--paper",
}


def page_figure(fig: go.Figure) -> dict:
    """Serialize a figure built with :data:`THEMED_PALETTE` for the page.

    ``to_plotly_json()`` then one recursive walk replacing each placeholder
    hex with its ``$tok:`` sentinel — traces, tick fonts, grids, titles and
    the legend alike, so no per-property list here or in the template can
    fall out of step. The browser resolves the sentinels against the live
    theme tokens before ``Plotly.react``.
    """

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return SENTINELS.get(value, value)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return walk(fig.to_plotly_json())


def _ticks(palette: Palette) -> dict:
    """Outside tick marks in the palette's axis colour."""
    return {"ticks": "outside", "ticklen": 4, "tickcolor": palette.grid}


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


def base_layout(palette: Palette) -> dict:
    """The shared base layout in one palette (from run.html's PLOT_LAYOUT).

    Every figure starts here. The legend sits horizontally above the plot
    and the x axis reserves room for its title — both were silently lost
    once when this was rebuilt by hand, so ``tests/test_figures.py`` pins
    what :func:`shots_figure` actually serves, not a constant.
    """
    return {
        "paper_bgcolor": palette.paper,
        "plot_bgcolor": palette.plot,
        "font": {"color": palette.font, "size": 12},
        "margin": {"t": 24, "r": 56, "b": 44, "l": 56},
        "legend": {"orientation": "h", "y": 1.08},
        "showlegend": True,
        "xaxis": {
            "gridcolor": palette.grid_soft,
            "zerolinecolor": palette.grid,
            "automargin": True,
            **_ticks(palette),
        },
    }


def trace_color(
    display: Optional[Mapping], i: int, palette: Palette = NOTEBOOK_PALETTE
) -> str:
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
    return palette.trace[i % len(palette.trace)]


def _is_hex_color(value: str) -> bool:
    body = value[1:]
    return (
        value.startswith("#")
        and len(body) in (3, 4, 6, 8)
        and all(c in "0123456789abcdefABCDEF" for c in body)
    )


#: Default marker size — injected into the page so the display popup's
#: "is this the default?" check cannot drift from the figure builder.
MARKER_SIZE_DEFAULT = 5.0


def _marker_size(display: Optional[Mapping]) -> float:
    size = (display or {}).get("msize")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        if math.isfinite(size) and size > 0:
            return float(size)
    return MARKER_SIZE_DEFAULT


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
    palette: Palette = NOTEBOOK_PALETTE,
) -> dict:
    """Build the stacked-axis ladder for up to four y columns.

    Axis 1 owns the grid and a colored title; axis 2 anchors right;
    axes 3–4 are free + autoshift with color-matched ticks only (a
    free-anchored axis's rotated title does not shift with it —
    measured, see #725).
    """
    layout: dict = {
        "yaxis": {
            "gridcolor": palette.grid_soft,
            "zerolinecolor": palette.grid,
            "tickfont": {"color": trace_color(display, 0, palette)},
            "title": {
                "text": _pretty(pretty, y[0]),
                "font": {"color": trace_color(display, 0, palette)},
            },
            "automargin": True,
            **_ticks(palette),
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
            "zerolinecolor": palette.grid,
            "tickfont": {"color": trace_color(display, i, palette)},
            **_ticks(palette),
        }
        if i >= 2:
            axis["anchor"] = "free"
            axis["autoshift"] = True
        else:
            axis["title"] = {
                "text": _pretty(pretty, y[i]),
                "font": {"color": trace_color(display, i, palette)},
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
    # Explicit plot dimensions (copy-paste graphics sizing): a fixed
    # width/height also fixes the exported image size.  Non-positive
    # values degrade to responsive autosizing, like every cosmetic.
    for key in ("width", "height"):
        value = d.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ):
            layout[key] = float(value)


def shot_axis_for_frame(frame: "pd.DataFrame") -> "pd.Series":
    """The shot axis for a DataFrame — THE one implementation of the rule.

    ``scan_event_index`` when present (1-based already), else 1-based
    row labels; union rows the event side missed carry NA there and are
    coalesced from the s-file's own shot identity (plain, or suffixed
    by scan_frame's collision rename) — the 0.9.1 rule: Plotly silently
    drops points with a null x, so those rows must keep a shot axis.
    The ``/api`` frame endpoint and the "show the code" snippet both go
    through here; a filtered frame keeps original shot identities.
    """
    import pandas as pd

    from geecs_data_utils.tiled_schema import SHOT_INDEX_COLUMN

    if SHOT_INDEX_COLUMN in frame.columns:
        shot = frame[SHOT_INDEX_COLUMN].copy()
    else:
        shot = frame.index.to_series() + 1
    if shot.isna().any():
        for name in ("Shotnumber", "Shotnumber (s-file)"):
            if name in frame.columns:
                shot = shot.fillna(pd.to_numeric(frame[name], errors="coerce"))
    return shot


def _shot_axis(series: Mapping, y: Sequence[str], shot: Optional[Sequence]) -> Sequence:
    if shot is not None:
        return shot
    if hasattr(series, "columns") and hasattr(series, "index"):
        return shot_axis_for_frame(series)
    if "Shotnumber" in series:
        return series["Shotnumber"]
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
    palette: Palette = NOTEBOOK_PALETTE,
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
            marker={
                "color": trace_color(display, i, palette),
                "size": _marker_size(display),
            },
            yaxis="y" if i == 0 else f"y{i + 1}",
        )
    layout = base_layout(palette)
    layout["xaxis"]["title"] = {"text": _pretty(pretty, x) if x else "shot #"}
    if x_is_date:
        layout["xaxis"]["type"] = "date"  # ISO strings from the API
    layout.update(_multi_y_layout(y, pretty, display, palette))
    _apply_display(layout, display, x_is_date=x_is_date, y_is_date=y_is_date)
    fig.update_layout(layout)
    return fig


def binned_figure(
    bins: Sequence,
    series: Mapping[str, Mapping[str, Sequence]],
    y: Sequence[str],
    *,
    bin_col: str = "Bin #",
    x_values: Optional[Sequence] = None,
    x_label: Optional[str] = None,
    pretty: Optional[Mapping[str, str]] = None,
    display: Optional[Mapping] = None,
    palette: Palette = NOTEBOOK_PALETTE,
) -> go.Figure:
    """The binned figure — centers, lines, and asymmetric error bars.

    Parameters
    ----------
    bins : Sequence
        Bin labels — the x axis when no ``x_values`` are given.
    series : Mapping[str, Mapping[str, Sequence]]
        Column → ``{"center": …, "err_low": …, "err_high": …}`` — the
        ``/api`` binned payload's ``series`` shape.
    y : Sequence[str]
        The plotted columns, in trace order.
    bin_col : str
        The x-axis title when plotting against the bin labels.
    x_values : Sequence, optional
        Per-bin x positions (the selected X column's per-bin mean) —
        bins group the data, the X parameter places it.  Must align
        with ``bins`` one-to-one.
    x_label : str, optional
        The x-axis title for ``x_values`` (defaults to ``bin_col``).
    pretty, display
        As in :func:`shots_figure`.

    Returns
    -------
    plotly.graph_objects.Figure
        Exactly what the Plot tab renders in binned view.
    """
    positions = list(x_values) if x_values is not None else list(bins)
    x_title = (x_label or bin_col) if x_values is not None else bin_col
    fig = _bare_figure()
    for i, name in enumerate(y):
        s = series.get(name) or {"center": [], "err_low": [], "err_high": []}
        fig.add_scatter(
            x=positions,
            y=list(s["center"]),
            mode="markers+lines",
            name=_pretty(pretty, name),
            marker={
                "color": trace_color(display, i, palette),
                "size": _marker_size(display) + 1,
            },
            line={"color": trace_color(display, i, palette), "width": 1},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": list(s["err_high"]),
                "arrayminus": list(s["err_low"]),
                "color": trace_color(display, i, palette),
                "thickness": 1,
            },
            yaxis="y" if i == 0 else f"y{i + 1}",
        )
    layout = base_layout(palette)
    layout["xaxis"]["title"] = {"text": x_title}
    layout.update(_multi_y_layout(y, pretty, display, palette))
    # Binned serves raw numbers — no date axes on either side.
    _apply_display(layout, display, x_is_date=False, y_is_date=False)
    fig.update_layout(layout)
    return fig
