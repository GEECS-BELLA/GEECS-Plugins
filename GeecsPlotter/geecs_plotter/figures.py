"""Pure figure authoring: columns in, a plotly ``Figure`` out.

Copied from the portal's ``geecs_portal/figures.py`` (``Palette``,
``page_figure``, ``base_layout``, ``shots_figure``) and cut to what M1
draws — one Y column against one X. Accepted debt, named in this package's
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
from typing import Any, Mapping, Optional, Sequence

import plotly.graph_objects as go
import plotly.io as pio


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


def shots_figure(
    series: Mapping[str, Sequence],
    y: str,
    *,
    x: Optional[str] = None,
    shot: Optional[Sequence] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    palette: Palette = NOTEBOOK_PALETTE,
) -> go.Figure:
    """The per-shot scatter: one marker per shot, ``y`` against ``x``.

    Parameters
    ----------
    series : Mapping[str, Sequence]
        Column name → values; a pandas ``DataFrame`` qualifies.
    y : str
        The plotted column.
    x : str, optional
        The X column; absent (or missing from ``series``) means the shot
        number.
    shot : Sequence, optional
        The shot numbers; defaults to 1..n.
    x_label, y_label : str, optional
        The axis titles, when not the column names (the alias toggles).
        One per axis, not per column: X and Y may be the same column.
    palette : Palette, optional
        :data:`NOTEBOOK_PALETTE` (default) or :data:`THEMED_PALETTE`.

    Returns
    -------
    plotly.graph_objects.Figure
        Exactly what the page draws, up to palette.
    """
    x = x if x and x in series else None
    y_values = list(series[y])
    if x:
        x_values = list(series[x])
    elif shot is not None:
        x_values = list(shot)
    else:
        x_values = list(range(1, len(y_values) + 1))
    fig = go.Figure(layout={"template": pio.templates["none"]})
    fig.add_scatter(
        x=x_values,
        y=y_values,
        mode="markers",
        name=y_label or y,
        marker={"color": palette.trace[0], "size": 5.0},
    )
    layout = base_layout(palette)
    layout["xaxis"]["title"] = {"text": (x_label or x) if x else "shot #"}
    layout["yaxis"]["title"] = {"text": y_label or y}
    fig.update_layout(layout)
    return fig
