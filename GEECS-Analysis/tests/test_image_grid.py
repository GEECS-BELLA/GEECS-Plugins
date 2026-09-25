"""Image grids preserve independent geometry and a truthful shared palette."""

from io import BytesIO

import numpy as np
import pytest
from geecs_data_utils.frames import Axis, Frame
from matplotlib.colors import LogNorm, Normalize

from geecs_analysis.measurement import Marker, Measurement
from geecs_analysis.render import RenderError, image_grid
from geecs_analysis.render.specs import FigureSpec


def result(data, **kwargs):
    return Measurement({}, Frame.from_array(data, **kwargs), (Marker("com", 1, 1),))


def test_grid_preserves_arrays_extents_titles_and_global_limits():
    results = [result(np.full((2, 3), value)) for value in (2, 7, 11)]
    fig = image_grid(results, titles=["low", "middle", "high"], columns=2)
    assert len(fig.axes) == 5  # Four panel slots, one shared colorbar.
    for ax, measurement, title in zip(fig.axes, results, ["low", "middle", "high"]):
        np.testing.assert_array_equal(ax.images[0].get_array(), measurement.frame.data)
        assert ax.images[0].get_clim() == (2, 11)
        assert ax.get_title() == title
        assert list(ax.images[0].get_extent()) == [-0.5, 2.5, 1.5, -0.5]
        assert len(ax.lines) == 1
    assert not fig.axes[3].get_visible()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    assert buffer.getvalue().startswith(b"\x89PNG")


def test_nonuniform_and_uniform_panels_share_style_and_keep_their_own_geometry():
    a = result([[1, 2, 3], [4, 5, 6]])
    b = result([[7, 8, 9], [10, 11, 12]], axes=(Axis([2, 5]), Axis([1, 3, 9])))
    style = FigureSpec(
        imshow={"cmap": "viridis", "vmin": 0, "vmax": 20}, colorbar={"label": "signal"}
    )
    fig = image_grid([a, b], style=style)
    assert (
        fig.axes[0].images[0].get_clim()
        == fig.axes[1].collections[0].get_clim()
        == (0, 20)
    )
    assert fig.axes[1].collections[0].cmap.name == "viridis"
    np.testing.assert_array_equal(fig.axes[1].collections[0].get_array(), b.frame.data)
    assert fig.axes[1].get_xlim() == (0, 12)
    assert fig.axes[-1].get_ylabel() == "signal"


def test_mutable_normalizer_is_owned_and_scales_over_every_panel():
    norm = Normalize()
    style = FigureSpec(imshow={"norm": norm}, colorbar={"show": False})
    fig = image_grid([result([[1, 2]]), result([[10, 20]])], style=style)
    assert style.imshow["norm"].vmin is None and style.imshow["norm"].vmax is None
    assert len(fig.axes) == 2
    assert all(ax.images[0].get_clim() == (1, 20) for ax in fig.axes)


@pytest.mark.parametrize("norm", [LogNorm(), "log"])
def test_log_scale_uses_positive_samples_despite_nonpositive_background(norm):
    style = FigureSpec(imshow={"norm": norm})
    fig = image_grid(
        [result([[-10, 0, 1], [0, 2, 10]]), result([[0, 4, 20], [-1, 50, 100]])],
        style=style,
    )
    assert fig.axes[0].images[0].get_clim() == (1, 100)
    assert fig.axes[1].images[0].get_clim() == (1, 100)
    np.testing.assert_allclose(fig.axes[0].images[0].norm([1, 10, 100]), [0, 0.5, 1])
    if isinstance(norm, LogNorm):
        assert style.imshow["norm"].vmin is None and style.imshow["norm"].vmax is None


def test_string_normalizer_retains_explicit_limits():
    fig = image_grid(
        [result([[0, 1, 100]])],
        style=FigureSpec(imshow={"norm": "log", "vmin": 0.1, "vmax": 1000}),
    )
    assert fig.axes[0].images[0].get_clim() == (0.1, 1000)


@pytest.mark.parametrize("value", [0, 5, np.nan])
def test_constant_or_all_nan_panels_keep_identical_color_scales(value):
    fig = image_grid([result(np.full((2, 2), value)) for _ in range(2)])
    assert fig.axes[0].images[0].get_clim() == fig.axes[1].images[0].get_clim()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"columns": 0},
        {"columns": True},
        {"titles": []},
        {
            "style": FigureSpec(
                imshow={"cmap": "plasma"}, pcolormesh={"cmap": "viridis"}
            )
        },
        {"style": FigureSpec(imshow={"vmin": 4, "vmax": 1})},
    ],
)
def test_invalid_layout_or_palette_is_a_render_error(kwargs):
    with pytest.raises(RenderError):
        image_grid([result([[1, 2]])], **kwargs)


def test_empty_traces_and_mixed_units_are_refused():
    for results in (
        [],
        [Measurement({}, Frame.from_array([1, 2]))],
        [result([[1]], unit="V"), result([[2]], unit="A")],
    ):
        with pytest.raises(RenderError):
            image_grid(results)


def test_shared_colorbar_spans_the_drawn_panels_not_the_grid_slots():
    fig = image_grid([result(np.ones((40, 160)))] * 5, columns=3)
    fig.savefig(BytesIO(), format="png")
    panels = [ax.get_position(original=False) for ax in fig.axes[:5]]
    cax = fig.axes[-1].get_position(original=False)
    assert cax.y0 == pytest.approx(min(p.y0 for p in panels), abs=1e-6)
    assert cax.y1 == pytest.approx(max(p.y1 for p in panels), abs=1e-6)
    assert cax.x0 > max(p.x1 for p in panels)
