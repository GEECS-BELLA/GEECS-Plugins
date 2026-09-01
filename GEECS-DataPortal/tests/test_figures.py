"""Unit tests for the server-side figure authoring (geecs_portal.figures).

These pin the ported page behavior trace by trace: the multi-axis
ladder, display semantics (types checked upstream, values degrade), the
shot-axis fallbacks, and the DataFrame entry point the "show the code"
snippets rely on.
"""

import pandas as pd

from geecs_portal import figures
from geecs_portal.figures import TRACE_COLORS, binned_figure, shots_figure

SERIES = {
    "a": [1.0, 2.0, 3.0],
    "b": [10.0, 20.0, 30.0],
    "c": [0.1, 0.2, 0.3],
    "d": [5.0, 6.0, 7.0],
}


def _fig_dict(fig):
    return fig.to_plotly_json()


class TestShotsFigure:
    def test_traces_wire_to_stacked_axes_in_order(self):
        fig = _fig_dict(shots_figure(SERIES, ["a", "b", "c", "d"]))
        assert [t["yaxis"] for t in fig["data"]] == ["y", "y2", "y3", "y4"]
        assert [t["marker"]["color"] for t in fig["data"]] == list(TRACE_COLORS)
        assert all(t["mode"] == "markers" for t in fig["data"])

    def test_axis_ladder_matches_the_page(self):
        layout = _fig_dict(shots_figure(SERIES, ["a", "b", "c", "d"]))["layout"]
        assert layout["yaxis"]["title"]["text"] == "a"
        # The bake-off aesthetics rider: subtler grid, outside ticks.
        assert layout["yaxis"]["gridcolor"] == "#232a31"
        assert layout["yaxis"]["ticks"] == "outside"
        assert layout["yaxis2"]["side"] == "right"
        assert layout["yaxis2"]["title"]["text"] == "b"
        assert layout["yaxis2"]["overlaying"] == "y"
        # Axes 3-4: free + autoshift, ticks only (no rotated title —
        # Plotly does not shift a free-anchored axis's title, #725).
        for name, side in (("yaxis3", "left"), ("yaxis4", "right")):
            assert layout[name]["anchor"] == "free"
            assert layout[name]["autoshift"] is True
            assert layout[name]["side"] == side
            assert "title" not in layout[name]
            assert layout[name]["gridcolor"] == "rgba(0,0,0,0)"

    def test_single_trace_hides_the_legend(self):
        assert _fig_dict(shots_figure(SERIES, ["a"]))["layout"]["showlegend"] is False
        assert (
            _fig_dict(shots_figure(SERIES, ["a", "b"]))["layout"]["showlegend"] is True
        )

    def test_x_column_used_when_servable_else_shot_axis(self):
        fig = _fig_dict(shots_figure(SERIES, ["a"], x="b"))
        assert fig["data"][0]["x"] == [10.0, 20.0, 30.0]
        assert fig["layout"]["xaxis"]["title"]["text"] == "b"
        fallback = _fig_dict(shots_figure(SERIES, ["a"], x="missing", shot=[7, 8, 9]))
        assert fallback["data"][0]["x"] == [7, 8, 9]
        assert fallback["layout"]["xaxis"]["title"]["text"] == "shot #"

    def test_datetime_x_sets_date_type_and_blocks_log_and_range(self):
        series = {"a": [1.0, 2.0], "t": ["2026-08-21 07:44:49", "2026-08-21 07:44:57"]}
        layout = _fig_dict(
            shots_figure(
                series,
                ["a"],
                x="t",
                kinds={"t": "datetime"},
                display={"logx": True, "xmin": 0.0, "xmax": 1.0},
            )
        )["layout"]
        assert layout["xaxis"]["type"] == "date"
        assert "range" not in layout["xaxis"]

    def test_display_log_and_ranges(self):
        layout = _fig_dict(
            shots_figure(
                SERIES,
                ["a"],
                display={"logy": True, "ymin": 0.1, "ymax": 100.0},
            )
        )["layout"]
        assert layout["yaxis"]["type"] == "log"
        # Log ranges are exponents.
        assert layout["yaxis"]["range"] == [-1.0, 2.0]
        assert layout["yaxis"]["autorange"] is False

    def test_bad_display_values_degrade_not_error(self):
        # Non-positive log bound → no explicit range; non-hex color →
        # palette; non-positive marker size → default.  Shared-link
        # cosmetics never make a figure fail.
        fig = _fig_dict(
            shots_figure(
                SERIES,
                ["a"],
                display={
                    "logy": True,
                    "ymin": -5.0,
                    "ymax": 100.0,
                    "colors": ["javascript:alert(1)"],
                    "msize": -3,
                },
            )
        )
        assert "range" not in fig["layout"]["yaxis"]
        assert fig["data"][0]["marker"]["color"] == TRACE_COLORS[0]
        assert fig["data"][0]["marker"]["size"] == 5.0

    def test_custom_hex_colors_apply_to_trace_axis_and_title(self):
        fig = _fig_dict(
            shots_figure(SERIES, ["a", "b"], display={"colors": ["#123456"]})
        )
        assert fig["data"][0]["marker"]["color"] == "#123456"
        assert fig["layout"]["yaxis"]["tickfont"]["color"] == "#123456"
        assert fig["layout"]["yaxis"]["title"]["font"]["color"] == "#123456"
        assert fig["data"][1]["marker"]["color"] == TRACE_COLORS[1]

    def test_pretty_names_reach_titles_and_legend(self):
        fig = _fig_dict(
            shots_figure(SERIES, ["a", "b"], pretty={"a": "Alpha [A]", "b": "Beta"})
        )
        assert fig["data"][0]["name"] == "Alpha [A]"
        assert fig["layout"]["yaxis"]["title"]["text"] == "Alpha [A]"
        assert fig["layout"]["yaxis2"]["title"]["text"] == "Beta"

    def test_dataframe_input_is_the_snippet_contract(self):
        # The "show the code" snippet passes the reproduced frame
        # directly; a filtered frame keeps original shot identities.
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[0, 4, 9])
        fig = _fig_dict(shots_figure(frame, ["a"]))
        assert fig["data"][0]["x"] == [1, 5, 10]
        # A DataFrame follows the ENDPOINT's rule (scan_event_index
        # else index+1; Shotnumber only coalesces NA cells) — a frame
        # with Shotnumber but no event index gets row labels, exactly
        # as /api serves it.  Plain mappings (the /api payload shape)
        # still honor an explicit Shotnumber key.
        with_shot = pd.DataFrame({"a": [1.0, 2.0], "Shotnumber": [11, 12]})
        assert _fig_dict(shots_figure(with_shot, ["a"]))["data"][0]["x"] == [1, 2]
        as_mapping = {"a": [1.0, 2.0], "Shotnumber": [11, 12]}
        assert _fig_dict(shots_figure(as_mapping, ["a"]))["data"][0]["x"] == [11, 12]

    def test_frame_shot_axis_is_the_endpoint_rule(self):
        # ONE implementation of the shot-axis contract: the notebook
        # path must coalesce union NA rows exactly like /api does
        # (0.9.1's fix — a null x is a silently dropped point).
        frame = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "scan_event_index": [1.0, 2.0, float("nan")],
                "Shotnumber": [float("nan"), 2.0, 3.0],
            }
        )
        fig = _fig_dict(shots_figure(frame, ["a"]))
        assert fig["data"][0]["x"] == [1.0, 2.0, 3.0]


class TestBinnedFigure:
    BINS = [1.0, 2.0, 3.0]
    SERIES = {
        "a": {
            "center": [10.0, 20.0, 30.0],
            "err_low": [1.0, 2.0, 3.0],
            "err_high": [4.0, 5.0, 6.0],
        }
    }

    def test_centers_lines_and_asymmetric_error_bars(self):
        fig = _fig_dict(binned_figure(self.BINS, self.SERIES, ["a"]))
        trace = fig["data"][0]
        assert trace["mode"] == "markers+lines"
        assert trace["x"] == self.BINS
        assert trace["y"] == [10.0, 20.0, 30.0]
        assert trace["error_y"]["symmetric"] is False
        assert trace["error_y"]["array"] == [4.0, 5.0, 6.0]
        assert trace["error_y"]["arrayminus"] == [1.0, 2.0, 3.0]
        # Binned markers render one point larger than per-shot.
        assert trace["marker"]["size"] == 6.0

    def test_bin_col_titles_the_x_axis(self):
        fig = _fig_dict(
            binned_figure(self.BINS, self.SERIES, ["a"], bin_col="U_S1H Current")
        )
        assert fig["layout"]["xaxis"]["title"]["text"] == "U_S1H Current"

    def test_missing_column_renders_empty_not_error(self):
        fig = _fig_dict(binned_figure(self.BINS, self.SERIES, ["a", "ghost"]))
        assert fig["data"][1]["y"] == []

    def test_display_ranges_apply_raw(self):
        layout = _fig_dict(
            binned_figure(
                self.BINS,
                self.SERIES,
                ["a"],
                display={"ymin": 0.0, "ymax": 50.0},
            )
        )["layout"]
        assert layout["yaxis"]["range"] == [0.0, 50.0]


class TestTemplateStaysOut:
    def test_no_default_template_styling_in_the_json(self):
        # plotly.py's default template is kilobytes of light-theme
        # styling the page never had (pre-0.10.0 layouts were built in
        # the browser, template-free) — the "none" pin keeps the served
        # figure the only styling authority.
        import json

        layout = _fig_dict(shots_figure(SERIES, ["a"]))["layout"]
        blob = json.dumps(layout.get("template", {}))
        assert "colorscale" not in blob and "#E5ECF6" not in blob
        assert len(blob) < 200


class TestPalette:
    def test_palette_is_the_committed_contract(self):
        # The template injects this tuple so rail chips match traces;
        # changing it is a deliberate cosmetic release, not drift.
        assert TRACE_COLORS == ("#4cc2b4", "#d6a860", "#6f9fd8", "#c47ab8")
        assert figures.BASE_LAYOUT["legend"] == {"orientation": "h", "y": 1.08}
