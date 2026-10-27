"""The figure builders: pure, so pinned on hand-built columns."""

from __future__ import annotations

import math

import plotly.graph_objects as go

from geecs_plotter import figures


def test_y_against_x() -> None:
    fig = figures.shots_figure({"a": [1.0, 2.0], "b": [3.0, 4.0]}, "b", x="a")
    (trace,) = fig.data
    assert list(trace.x) == [1.0, 2.0]
    assert list(trace.y) == [3.0, 4.0]
    assert trace.mode == "markers"
    assert fig.layout.xaxis.title.text == "a"
    assert fig.layout.yaxis.title.text == "b"


def test_no_x_is_the_shot_axis() -> None:
    fig = figures.shots_figure({"b": [3.0, 4.0]}, "b", shot=[7, 8])
    assert list(fig.data[0].x) == [7, 8]
    assert fig.layout.xaxis.title.text == "shot #"


def test_an_x_not_in_the_series_falls_back_to_shots() -> None:
    fig = figures.shots_figure({"b": [3.0, 4.0]}, "b", x="missing")
    assert list(fig.data[0].x) == [1, 2]
    assert fig.layout.xaxis.title.text == "shot #"


def test_page_figure_swaps_placeholders_and_drops_nan() -> None:
    fig = figures.shots_figure(
        {"b": [1.0, math.nan]}, "b", palette=figures.THEMED_PALETTE
    )
    page = figures.page_figure(fig)
    assert page["data"][0]["marker"]["color"] == "$tok:--trace-1"
    assert page["layout"]["paper_bgcolor"] == "$tok:--surface"
    assert page["data"][0]["y"] == [1.0, None]


def test_the_notebook_palette_is_real_hex() -> None:
    fig = figures.shots_figure({"b": [1.0]}, "b")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].marker.color == figures.NOTEBOOK_PALETTE.trace[0]


def test_pretty_names_title_the_axes() -> None:
    fig = figures.shots_figure(
        {"a": [1.0], "b A": [2.0]}, "b A", x="a", pretty={"b A": "B"}
    )
    assert fig.layout.yaxis.title.text == "B"
    assert fig.layout.xaxis.title.text == "a"
    assert fig.data[0].name == "B"
