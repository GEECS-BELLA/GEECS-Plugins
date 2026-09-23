"""V2 plotting preserves waterfall samples/geometry and explicit palette choices."""

import numpy as np
import pytest
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis.renderer import RendererOptions

from geecs_analysis.compat.v2_render import image_grid_v2, single_v2, waterfall_v2
from geecs_analysis.measurement import Measurement
from geecs_analysis.render import RenderError


def trace(x, y):
    return Measurement(
        {},
        Frame.from_trace(
            np.column_stack([x, y]), x_unit="eV", y_unit="counts", y_label="Signal"
        ),
    )


@pytest.mark.parametrize("mode", ["auto", "sequential", "diverging", "custom"])
def test_waterfall_palette_matches_legacy_for_finite_data(mode):
    from scan_analysis.analyzers.renderers.line_1d_renderer import Line1DRenderer
    from scan_analysis.analyzers.renderers.config import Line1DRendererConfig

    options = RendererOptions(colormap_mode=mode, vmin=-3, vmax=9, dpi=30)
    data = np.array([[-1, 0, 8], [2, 3, 4]])
    expected = Line1DRenderer()._get_colormap_params_1d(
        data, Line1DRendererConfig(**options.as_kwargs())
    )
    fig = waterfall_v2(
        [trace([1, 2, 3], row) for row in data], [1, 2], "motor", options
    )
    artist = fig.axes[0].collections[0]
    assert artist.get_clim() == expected[:2]
    assert artist.cmap.name == expected[2]
    if expected[3] is not None:
        assert artist.norm(0) == 0.5
    np.testing.assert_array_equal(artist.get_array().reshape(data.shape), data)
    fig.clear()


def test_waterfall_uses_first_x_grid_preserves_zero_and_ascending_y():
    results = [trace([1, 2, 4], [2, 3, 4]), trace([10, 20, 40], [5, 6, 7])]
    fig = waterfall_v2(results, [0, 3], "motor", RendererOptions(dpi=30))
    ax = fig.axes[0]
    coordinates = ax.collections[0].get_coordinates()
    np.testing.assert_array_equal(coordinates[0, :, 0], [0.5, 1.5, 3, 5])
    np.testing.assert_array_equal(coordinates[:, 0, 1], [-1.5, 1.5, 4.5])
    assert ax.get_ylim()[0] < ax.get_ylim()[1]
    assert ax.get_xlabel() == "x (eV)"
    assert fig.axes[1].get_ylabel() == "Signal (counts)"
    fig.clear()


@pytest.mark.parametrize("explicit", [None, False, True])
def test_waterfall_sort_implies_even_spacing_unless_explicitly_overridden(explicit):
    results = [trace([1, 2], [1, 2])] * 3
    fig = waterfall_v2(
        results,
        [1, 10, 50],
        "charge",
        RendererOptions(
            waterfall_sort_key="charge", waterfall_even_y_spacing=explicit, dpi=30
        ),
    )
    expected = [1, 10, 50] if explicit is False else [0, 1, 2]
    np.testing.assert_array_equal(fig.axes[0].get_yticks(), expected)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
        "1.000",
        "10.000",
        "50.000",
    ]
    fig.clear()


def test_singleton_waterfall_and_repeated_nonmonotonic_positions():
    item = trace([2], [5])
    fig = waterfall_v2([item], [0], "motor", RendererOptions(dpi=30))
    np.testing.assert_array_equal(
        fig.axes[0].collections[0].get_coordinates()[:, 0, 1], [-0.5, 0.5]
    )
    fig.clear()
    fig = waterfall_v2([item] * 4, [1, 1, 3, 2], "motor", RendererOptions(dpi=30))
    np.testing.assert_array_equal(
        fig.axes[0].collections[0].get_array().reshape(-1), [5] * 4
    )
    fig.clear()


def test_bad_waterfall_shapes_or_positions_are_explicit():
    item = trace([1, 2], [3, 4])
    with pytest.raises(RenderError, match="equal-length"):
        waterfall_v2([item, trace([1], [2])], [1, 2], "motor", RendererOptions())
    with pytest.raises(RenderError, match="finite"):
        waterfall_v2([item], [np.nan], "motor", RendererOptions())


def test_grid_palette_is_shared_and_labels_are_honored():
    results = [Measurement({}, Frame.from_array([[-1, 1], [0, n]])) for n in [5, 10]]
    options = RendererOptions(
        colormap_mode="diverging",
        cmap="RdBu_r",
        xlabel="horizontal",
        colorbar_label="Signal",
        dpi=30,
        figsize=(2, 2),
    )
    original = options.model_dump()
    fig = image_grid_v2(results, [0, 1], options, label="motor")
    assert fig._suptitle.get_text() == "Scan parameter: motor"
    assert [ax.images[0].get_clim() for ax in fig.axes[:2]] == [(-10, 10)] * 2
    assert [ax.get_title() for ax in fig.axes[:2]] == ["0.00", "1.00"]
    assert fig.axes[0].get_xlabel() == "horizontal"
    assert fig.axes[-1].get_ylabel() == "Signal"
    assert options.model_dump() == original
    fig.clear()
    assert image_grid_v2(results, [0, 1], options)._suptitle is None


def test_single_line_retains_coordinates_samples_and_title():
    item = trace([1, 3, 9], [2, 8, 4])
    fig = single_v2(
        item, RendererOptions(dpi=30, ylabel="custom"), title="motor = 0.000"
    )
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [1, 3, 9])
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [2, 8, 4])
    assert fig.axes[0].get_title() == "motor = 0.000"
    assert fig.axes[0].get_ylabel() == "custom"
    fig.clear()
