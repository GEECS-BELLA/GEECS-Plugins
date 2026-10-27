"""Pure figure authoring: columns in, a plotly ``Figure`` out.

Copied from the portal's ``geecs_portal/figures.py`` (``Palette``,
``page_figure``, ``base_layout``, ``shots_figure``) and cut to what M1
draws — up to three Y columns against one X (the portal's multi-Y ladder,
``_y_axes``). Accepted debt, named in this package's
CLAUDE.md: the two copies may drift until portal/plotter integration.

The page asks for :data:`THEMED_PALETTE`, whose placeholder hex
:func:`page_figure` swaps for ``$tok:--name`` sentinels that the browser
resolves against the live theme tokens; a notebook takes the default
:data:`NOTEBOOK_PALETTE` and gets real hex. Everything else is the same
figure either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import plotly.graph_objects as go
import plotly.io as pio


#: GEECSplotter's Y1, Y2 and Y3: the most Y columns one figure draws.
MAX_Y = 3


@dataclass(frozen=True)
class Palette:
    """Every colour a figure needs, in one place."""

    trace: tuple[str, ...]
    grid: str
    grid_soft: str
    font: str
    paper: str
    plot: str


NOTEBOOK_PALETTE = Palette(
    trace=("#4cc2b4", "#d6a860", "#6f9fd8", "#c47ab8"),
    grid="#2c353d",
    grid_soft="#232a31",
    font="#dde4ea",
    paper="#1a2026",
    plot="#12161a",
)

#: Placeholder hex no theme uses: plotly.py validates colours at build time
#: and rejects a ``$tok:`` string, so :func:`page_figure` substitutes later.
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
    """Serialize a :data:`THEMED_PALETTE` figure for the page.

    One walk replaces each placeholder with its sentinel and each
    non-finite float with ``None`` (a gap, which is what a missing shot
    is) — NaN is not valid JSON.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        A figure built with :data:`THEMED_PALETTE`.

    Returns
    -------
    dict
        The figure JSON the page hands to ``Plotly.react``.
    """

    def walk(value: Any) -> Any:  # Any: arbitrary plotly JSON
        if isinstance(value, str):
            return SENTINELS.get(value, value)
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return walk(fig.to_plotly_json())


def _ticks(palette: Palette) -> dict:
    return {"ticks": "outside", "ticklen": 4, "tickcolor": palette.grid}


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
        **_ticks(palette),
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


def _y_axes(titles: Sequence[str], palette: Palette) -> dict:
    """The y-axis ladder, one axis per trace, each in its trace's colour.

    Copied from the portal's ``_multi_y_layout``: axis 1 (left) owns the
    grid and a coloured title; axis 2 anchors right with its title; axis
    3 is free with ``autoshift`` and coloured ticks only, since a free
    axis's rotated title does not shift with it (portal #725) — the
    legend, on whenever there is more than one trace, names it.
    """
    layout: dict = {
        "showlegend": len(titles) > 1,
        # A row above the plot, clear of the right-hand axis.
        "legend": {"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
    }
    for i, title in enumerate(titles):
        colour = palette.trace[i]
        axis: dict = {
            "automargin": True,
            "zerolinecolor": palette.grid,
            "tickfont": {"color": colour},
            **_ticks(palette),
        }
        if i == 0:
            axis["gridcolor"] = palette.grid_soft
        else:
            axis.update(
                overlaying="y",
                side="right" if i % 2 else "left",
                gridcolor="rgba(0,0,0,0)",
            )
        if i >= 2:
            axis.update(anchor="free", autoshift=True)
        else:
            axis["title"] = {"text": title, "font": {"color": colour}}
        layout["yaxis" if i == 0 else f"yaxis{i + 1}"] = axis
    return layout


def shots_figure(
    series: Mapping[str, Sequence],
    y: Union[str, Sequence[str]],
    *,
    x: Optional[str] = None,
    shot: Optional[Sequence] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    y_labels: Sequence[Optional[str]] = (),
    palette: Palette = NOTEBOOK_PALETTE,
) -> go.Figure:
    """The per-shot scatter: one marker trace per Y column, against ``x``.

    Parameters
    ----------
    series : Mapping[str, Sequence]
        Column name → values; a pandas ``DataFrame`` qualifies.
    y : str or Sequence[str]
        The plotted column, or up to three (GEECSplotter's Y1–Y3), each on
        its own y axis in its trace's colour.
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
        One title per Y column, ``None`` for the column name. Per axis,
        not per column: the same column may be on two axes.
    palette : Palette, optional
        :data:`NOTEBOOK_PALETTE` (default) or :data:`THEMED_PALETTE`.

    Returns
    -------
    plotly.graph_objects.Figure
        Exactly what the page draws, up to palette.
    """
    ys = [y] if isinstance(y, str) else list(y)
    if not 1 <= len(ys) <= MAX_Y:
        raise ValueError(f"plot 1 to {MAX_Y} Y columns, not {len(ys)}")
    labels = list(y_labels) + [None] * (len(ys) - len(y_labels))
    if y_label is not None:
        labels[0] = y_label
    titles = [label or name for label, name in zip(labels, ys)]
    x = x if x and x in series else None
    if x:
        x_values = list(series[x])
    elif shot is not None:
        x_values = list(shot)
    else:
        x_values = list(range(1, len(series[ys[0]]) + 1))
    fig = go.Figure(layout={"template": pio.templates["none"]})
    for i, (name, title) in enumerate(zip(ys, titles)):
        fig.add_scatter(
            x=x_values,
            y=list(series[name]),
            mode="markers",
            name=title,
            marker={"color": palette.trace[i], "size": 5.0},
            yaxis="y" if i == 0 else f"y{i + 1}",
        )
    layout = base_layout(palette)
    layout["xaxis"]["title"] = {"text": (x_label or x) if x else "shot #"}
    del layout["yaxis"]
    layout.update(_y_axes(titles, palette))
    fig.update_layout(layout)
    return fig
