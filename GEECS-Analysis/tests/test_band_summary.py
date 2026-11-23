"""The band summary: per-sample statistics, grouping by bin, and the layouts."""

import numpy as np
import pytest
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import BandSummary
from pydantic import ValidationError

from geecs_analysis.measurement import Measurement
from geecs_analysis.registry import summary_definition, summary_output
from geecs_analysis.render import RenderError
from geecs_analysis.render.specs import FigureSpec
from geecs_analysis.summaries.band import band_statistics, group_by_position

FIGURE = FigureSpec(fig={"dpi": 30})


def shot(y, x=None):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if x is None else x
    return Measurement({}, Frame.from_trace(np.column_stack([x, y]), x_unit="ns"))


def draw(results, positions, label="motor", **options):
    spec = BandSummary(**options)
    return summary_output(
        summary_definition(spec).function(results, positions, label, spec, FIGURE)
    )


# ------------------------------------------------------------ statistics


def test_mean_band_is_sample_std_times_sigma():
    data = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 60.0]])
    stats = band_statistics(data, "mean", sigma=2.0)
    np.testing.assert_allclose(stats.center, [3.0, 30.0])
    spread = 2.0 * data.std(axis=0, ddof=1)
    np.testing.assert_allclose(stats.low, stats.center - spread)
    np.testing.assert_allclose(stats.high, stats.center + spread)
    np.testing.assert_array_equal(stats.count, [3, 3])


def test_median_band_is_the_chosen_percentiles():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(101, 4))
    stats = band_statistics(data, "median", percentiles=(0.1, 0.9))
    np.testing.assert_allclose(stats.center, np.median(data, axis=0))
    np.testing.assert_allclose(stats.low, np.quantile(data, 0.1, axis=0))
    np.testing.assert_allclose(stats.high, np.quantile(data, 0.9, axis=0))


def test_nan_samples_are_left_out_per_sample_not_per_shot():
    data = np.array(
        [
            [1.0, np.nan, np.nan, 4.0],
            [3.0, 5.0, np.nan, 4.0],
            [5.0, 7.0, 2.0, np.nan],
        ]
    )
    stats = band_statistics(data, "mean")
    np.testing.assert_array_equal(stats.count, [3, 2, 1, 2])
    np.testing.assert_allclose(stats.center, [3.0, 6.0, 2.0, 4.0])
    # one shot reached sample 2: a centre, but no standard deviation
    assert np.isnan(stats.low[2]) and np.isnan(stats.high[2])
    np.testing.assert_allclose(stats.high[1] - stats.center[1], np.std([5, 7], ddof=1))
    median = band_statistics(data, "median")
    np.testing.assert_allclose(median.center, [3.0, 6.0, 2.0, 4.0])
    assert median.low[2] == median.high[2] == 2.0


def test_an_all_nan_sample_is_nan_everywhere_without_warnings():
    data = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    with np.errstate(all="raise"):
        for statistic in ("mean", "median"):
            stats = band_statistics(data, statistic)
            assert np.isnan(stats.center[0]) and np.isnan(stats.low[0])
            assert stats.count[0] == 0


def test_statistics_refuse_unknown_and_non_2d_input():
    with pytest.raises(ValueError):
        band_statistics(np.zeros((2, 3)), "mode")
    with pytest.raises(ValueError):
        band_statistics(np.zeros(3), "mean")


def test_percentiles_are_ordered_fractions():
    assert BandSummary(statistic="median", percentiles=(0.1, 0.9)).percentiles == (
        0.1,
        0.9,
    )
    for bad in ((0.9, 0.1), (0.5, 0.5), (-0.1, 0.5), (0.2, 1.5)):
        with pytest.raises(ValidationError):
            BandSummary(percentiles=bad)


# ------------------------------------------------------------ grouping


def test_shots_group_by_position_in_order_of_first_appearance():
    frames = [shot([i]).frame for i in range(5)]
    groups = group_by_position(frames, [2.0, 1.0, 2.0, None, 1.0])
    assert [(p, [int(f.data[0]) for f in m]) for p, m in groups] == [
        (2.0, [0, 2]),
        (1.0, [1, 4]),
        (None, [3]),
    ]


def test_a_noscan_draws_one_line_and_one_gray_band():
    results = [shot([1, 2, 3]), shot([3, 4, 5]), shot([5, 6, 7])]
    out = draw(results, [None] * 3, "Shot Number")
    ax = out.figure.axes[0]
    assert len(ax.lines) == 1 and len(ax.collections) == 1
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [3, 4, 5])
    band = ax.collections[0]
    assert tuple(band.get_facecolor()[0][:3]) == pytest.approx(
        (0.502, 0.502, 0.502), abs=1e-3
    )
    assert "over 3 shots" in ax.get_title()
    assert ax.get_xlabel() == "x (ns)"
    assert not out.notes


def test_overlay_draws_one_line_per_bin_in_position_order_with_its_own_colour():
    results = [shot([1, 1]), shot([3, 3]), shot([10, 10]), shot([12, 12])]
    out = draw(results, [2.0, 2.0, 1.0, 1.0])
    ax = out.figure.axes[0]
    assert len(ax.lines) == 2 and len(ax.collections) == 2
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [11, 11])  # position 1.0
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [2, 2])
    colours = [line.get_color() for line in ax.lines]
    assert colours[0] != colours[1]
    band_colour = ax.collections[1].get_facecolor()[0][:3]
    np.testing.assert_allclose(band_colour, colours[1][:3])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["1", "2"]


def test_band_color_overrides_the_overlay_colours():
    results = [shot([1]), shot([3]), shot([10]), shot([12])]
    out = draw(results, [1.0, 1.0, 2.0, 2.0], band_color="red")
    for band in out.figure.axes[0].collections:
        assert tuple(band.get_facecolor()[0][:3]) == (1.0, 0.0, 0.0)


def test_many_bins_key_the_colours_with_a_colorbar():
    results = [shot([i, i]) for i in range(24)]
    out = draw(results, [float(i // 2) for i in range(24)])
    assert out.figure.axes[0].get_legend() is None
    assert len(out.figure.axes) == 2  # the colorbar's axes


def test_panels_draw_one_panel_per_bin_with_a_shared_y_axis():
    results = [
        shot([1, 2]),
        shot([3, 4]),
        shot([5, 6]),
        shot([7, 8]),
        shot([9, 9]),
        shot([9, 9]),
    ]
    out = draw(results, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], layout="panels", columns=2)
    visible = [ax for ax in out.figure.axes if ax.get_visible()]
    assert len(visible) == 3 and len(out.figure.axes) == 4
    assert [ax.get_title() for ax in visible] == [
        "1 (2 shots)",
        "2 (2 shots)",
        "3 (2 shots)",
    ]
    assert visible[0].get_shared_y_axes().joined(visible[0], visible[2])


def test_bins_below_min_shots_are_left_out_with_a_note():
    results = [shot([1]), shot([2]), shot([3]), shot([9])]
    out = draw(results, [1.0, 1.0, 1.0, 2.0], min_shots=2)
    assert len(out.figure.axes[0].lines) == 1
    assert out.notes == ("motor = 2: 1 shot(s), fewer than min_shots 2; not drawn",)
    with pytest.raises(RenderError, match="min_shots"):
        draw(results, [1.0, 1.0, 1.0, 2.0], min_shots=4)


def test_unequal_traces_in_a_bin_and_mismatched_units_are_refused():
    with pytest.raises(RenderError, match="equal-length"):
        draw([shot([1, 2]), shot([1, 2, 3])], [1.0, 1.0])
    other = Measurement(
        {}, Frame.from_trace(np.column_stack([[0.0, 1.0], [1.0, 2.0]]), x_unit="us")
    )
    with pytest.raises(RenderError, match="units"):
        draw([shot([1, 2]), other], [1.0, 1.0])
    with pytest.raises(RenderError):
        draw([shot([1])], [1.0, 2.0])


def test_a_long_trace_is_drawn_from_display_block_means():
    from geecs_analysis.summaries.waterfall import MAX_COLUMNS

    size = MAX_COLUMNS * 2
    results = [shot(np.full(size, v)) for v in (1.0, 3.0)]
    out = draw(results, [None, None])
    line = out.figure.axes[0].lines[0]
    assert line.get_ydata().shape == (MAX_COLUMNS,)
    np.testing.assert_allclose(line.get_ydata(), 2.0)
