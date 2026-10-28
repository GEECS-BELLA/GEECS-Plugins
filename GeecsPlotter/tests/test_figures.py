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


def test_labels_title_the_axes() -> None:
    fig = figures.shots_figure({"a": [1.0], "b A": [2.0]}, "b A", x="a", y_label="B")
    assert fig.layout.yaxis.title.text == "B"
    assert fig.layout.xaxis.title.text == "a"
    assert fig.data[0].name == "B"


def test_three_ys_get_three_axes_in_their_trace_colours() -> None:
    series = {"x": [1.0, 2.0], "a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]}
    fig = figures.shots_figure(series, ["a", "b", "c"], x="x")
    assert [t.yaxis for t in fig.data] == ["y", "y2", "y3"]
    trace = figures.NOTEBOOK_PALETTE.trace
    assert [t.marker.color for t in fig.data] == list(trace[:3])
    layout = fig.layout
    assert layout.yaxis.title.text == "a"
    assert layout.yaxis.tickfont.color == trace[0]
    assert layout.yaxis.title.font.color == trace[0]
    assert (layout.yaxis2.side, layout.yaxis2.overlaying) == ("right", "y")
    assert layout.yaxis2.title.font.color == trace[1]
    # Y3 on the right, outside Y2, with its own title (the operators' ask).
    assert (layout.yaxis3.side, layout.yaxis3.anchor) == ("right", "free")
    assert layout.yaxis3.position == 1.0
    assert layout.yaxis3.title.text == "c"
    assert layout.yaxis3.title.font.color == trace[2]
    assert tuple(layout.xaxis.domain) == (0.0, figures._RIGHT_DOMAIN)
    assert layout.showlegend


def test_two_ys_keep_the_full_width() -> None:
    fig = figures.shots_figure({"a": [1.0], "b": [2.0]}, ["a", "b"])
    assert fig.layout.xaxis.domain is None


def test_one_y_has_no_legend() -> None:
    assert not figures.shots_figure({"a": [1.0]}, "a").layout.showlegend


def test_labels_per_axis() -> None:
    series = {"a": [1.0], "b": [2.0]}
    fig = figures.shots_figure(series, ["a", "b"], y_labels=[None, "B!"])
    assert fig.layout.yaxis.title.text == "a"
    assert fig.layout.yaxis2.title.text == "B!"
    assert [t.name for t in fig.data] == ["a", "B!"]


def test_more_than_three_ys_is_refused() -> None:
    import pytest

    with pytest.raises(ValueError, match="1 to 3"):
        figures.shots_figure({"a": [1.0]}, ["a", "a", "a", "a"])


def test_themed_axes_resolve_to_trace_tokens() -> None:
    fig = figures.shots_figure(
        {"a": [1.0], "b": [2.0]}, ["a", "b"], palette=figures.THEMED_PALETTE
    )
    page = figures.page_figure(fig)
    assert page["layout"]["yaxis2"]["tickfont"]["color"] == "$tok:--trace-2"


def test_y3_keeps_its_slot_with_y2_off() -> None:
    """Y3 is the third colour, titled on the right, whether or not Y2 is on."""
    fig = figures.shots_figure({"a": [1.0], "c": [2.0]}, ["a", None, "c"])
    assert [t.yaxis for t in fig.data] == ["y", "y3"]
    assert fig.data[1].marker.color == figures.NOTEBOOK_PALETTE.trace[2]
    assert fig.layout.yaxis3.side == "right"
    assert fig.layout.yaxis3.anchor is None  # the only right axis: on the plot
    assert fig.layout.yaxis3.title.text == "c"
    assert "yaxis2" not in fig.layout.to_plotly_json()  # no Y2 axis drawn
