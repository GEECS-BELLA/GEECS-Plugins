"""Object-API rendering, coordinate fidelity and explicit waterfall geometry."""

import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from matplotlib.figure import Figure

from geecs_analysis.measurement import Measurement
from geecs_analysis.render import RenderError, single, waterfall
from geecs_analysis.render.specs import FigureSpec
from geecs_analysis.run import analyze
from geecs_analysis.specs import Analysis
from geecs_data_utils.frames import Axis, Frame


def beam():
    y, x = np.mgrid[:9, :11]
    return analyze(
        Frame.from_array(
            np.exp(-((x - 5) ** 2 + (y - 4) ** 2) / 8),
            axes=(
                Axis(np.arange(9) * 0.5 + 10, "mm", "y"),
                Axis(np.arange(11) * 0.25 + 20, "mm", "x"),
            ),
        ),
        Analysis(measure={"kind": "beam"}),
    )


def test_calibrated_image_extent_marker_and_projection_coordinates():
    result = beam()
    figure = single(result)
    ax = figure.axes[0]
    assert isinstance(figure, Figure)
    assert ax.images[0].get_extent() == [19.875, 22.625, 14.25, 9.75]
    assert ax.get_xlabel() == "x (mm)" and ax.get_ylabel() == "y (mm)"
    assert len(figure.axes) == 2  # explicit image colorbar
    x_projection, y_projection, marker = ax.lines
    np.testing.assert_array_equal(x_projection.get_xdata(), result.frame.axes[1].values)
    np.testing.assert_array_equal(y_projection.get_ydata(), result.frame.axes[0].values)
    np.testing.assert_array_equal(marker.get_xdata(), [result.scalars["x_CoM"]])
    np.testing.assert_array_equal(marker.get_ydata(), [result.scalars["y_CoM"]])


def test_passthrough_styles_and_overlay_hiding_do_not_modify_recipe():
    style = FigureSpec(
        imshow={"cmap": "viridis", "vmin": 0, "vmax": 1},
        colorbar={"show": False},
        axes={"title": "Beam"},
        fig={"figsize": [6, 3], "dpi": 80},
        overlays={
            "projection_x": {"hidden": True},
            "projection_y": {"scale": 0.1, "color": "white"},
            "com": {"color": "red", "marker": "x"},
        },
    )
    before = style.model_dump()
    fig = single(beam(), style)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.images[0].get_cmap().name == "viridis"
    assert ax.get_title() == "Beam"
    assert len(ax.lines) == 2 and ax.lines[-1].get_marker() == "x"
    assert fig.dpi == 80
    assert style.model_dump() == before


def test_trace_uses_calibrated_x_and_no_colorbar():
    frame = Frame.from_trace(
        [[80, 1], [70, 3], [60, 2]],
        x_unit="MeV",
        y_unit="pC",
        x_label="energy",
        y_label="charge",
    )
    fig = single(
        Measurement({}, frame), FigureSpec(plot={"color": "red", "marker": "o"})
    )
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [80, 70, 60])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1, 3, 2])
    assert ax.get_xlabel() == "energy (MeV)" and ax.get_ylabel() == "charge (pC)"


def test_waterfall_preserves_nonuniform_bin_positions_without_resampling():
    frames = [
        Frame.from_trace(
            [[60, i], [70, i + 1], [90, i + 2]], x_unit="MeV", y_unit="counts"
        )
        for i in range(3)
    ]
    positions = Axis([1, 2, 5], "mm", "scan position")
    fig = waterfall(
        frames,
        positions,
        FigureSpec(pcolormesh={"cmap": "viridis"}, colorbar={"label": "counts"}),
    )
    ax = fig.axes[0]
    assert len(ax.images) == 0 and len(ax.collections) == 1
    mesh = ax.collections[0]
    np.testing.assert_array_equal(mesh.get_array(), [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    coordinates = mesh.get_coordinates()
    np.testing.assert_array_equal(coordinates[0, :, 0], [55, 65, 80, 100])
    np.testing.assert_array_equal(coordinates[:, 0, 1], [0.5, 1.5, 3.5, 6.5])
    assert ax.get_ylabel() == "scan position (mm)"


@pytest.mark.parametrize("centers", [[1, 1, 2], [1, 3, 2]])
def test_ambiguous_image_coordinates_are_refused(centers):
    frame = Frame.from_array(np.ones((3, 3)), axes=(Axis(centers), Axis(np.arange(3))))
    with pytest.raises(RenderError, match="strictly monotonic"):
        single(Measurement({}, frame))


def test_descending_image_axes_preserve_sample_orientation():
    frame = Frame.from_array([[1, 2], [3, 4]], axes=(Axis([5, 3]), Axis([20, 10])))
    ax = single(Measurement({}, frame)).axes[0]
    assert ax.images[0].get_extent() == [25, 5, 2, 6]
    np.testing.assert_array_equal(ax.images[0].get_array(), frame.data)


def test_waterfall_rejects_grid_or_unit_mismatches_and_empty_inputs():
    first = Frame.from_trace([[1, 2], [2, 3]])
    for other in [
        Frame.from_trace([[1, 2], [3, 3]]),
        Frame.from_trace([[1, 2], [2, 3]], x_unit="MeV"),
    ]:
        with pytest.raises(RenderError, match="share an x grid"):
            waterfall([first, other], Axis([1, 2]))
    with pytest.raises(RenderError, match="one position"):
        waterfall([], Axis([1]))


@pytest.mark.parametrize(
    "style",
    [
        FigureSpec(imshow={"extent": [0, 1, 0, 1]}),
        FigureSpec(imshow={"not_a_kwarg": True}),
        FigureSpec(axes={"not_a_property": True}),
        FigureSpec(overlays={"projection_x": {"scale": -1}}),
    ],
)
def test_invalid_preview_styles_raise_render_error(style):
    with pytest.raises(RenderError):
        single(beam(), style)


def test_figures_render_to_independent_pngs_without_writing_files():
    result = beam()

    def draw(_):
        figure = single(result)
        buf = io.BytesIO()
        figure.savefig(buf, format="png")
        return figure, buf.getvalue()

    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(draw, range(3)))
    assert len({id(fig) for fig, _ in outputs}) == 3
    assert all(data.startswith(b"\x89PNG") for _, data in outputs)


def test_renderer_import_and_execution_never_load_pyplot():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from geecs_analysis.render import single
from geecs_analysis.measurement import Measurement
from geecs_data_utils.frames import Frame
figure = single(Measurement({}, Frame.from_array([1, 2, 3])))
assert "matplotlib.pyplot" not in sys.modules
assert "image_analysis" not in sys.modules
""",
        ],
        check=True,
    )


def test_projection_anchor_tracks_lower_origin_without_moving_data():
    result = beam()
    ax = single(result, FigureSpec(imshow={"origin": "lower"})).axes[0]
    bottom, top = ax.get_ylim()
    assert bottom < top
    projection_y = ax.lines[0].get_ydata()
    assert np.min(projection_y) >= bottom
    assert np.max(projection_y) == pytest.approx(bottom + (top - bottom) * 0.2)
    np.testing.assert_array_equal(ax.images[0].get_array(), result.frame.data)


def test_uniform_waterfall_uses_readable_automatic_aspect():
    frames = [
        Frame.from_trace(np.column_stack((np.arange(200), np.ones(200))))
        for _ in range(3)
    ]
    ax = waterfall(frames, Axis([1, 2, 3])).axes[0]
    assert ax.get_aspect() == "auto"


def test_deferred_canvas_validation_is_inside_render_error_boundary():
    style = FigureSpec(imshow={"vmin": 4, "vmax": 1}, colorbar={"show": False})
    with pytest.raises(RenderError, match="minvalue must be less"):
        single(beam(), style)


def _drawn(fig):
    fig.savefig(io.BytesIO(), format="png")  # layout reruns on every save
    return [ax.get_position(original=False) for ax in fig.axes]


def test_colorbar_spans_the_drawn_image_not_its_layout_slot():
    # A wide fixed-aspect image leaves its slot short; the colorbar follows it.
    wide = Measurement({}, Frame.from_array(np.ones((40, 160))), ())
    image, cax = _drawn(single(wide))
    assert image.height < 0.6  # the image really is shorter than its slot
    assert (cax.y0, cax.y1) == pytest.approx((image.y0, image.y1), abs=1e-6)
    assert 0 < cax.x0 - image.x1 < 0.1


def test_colorbar_placement_keywords_leave_layout_to_the_recipe():
    wide = Measurement({}, Frame.from_array(np.ones((40, 160))), ())
    image, cax = _drawn(single(wide, FigureSpec(colorbar={"shrink": 0.9})))
    assert cax.height > image.height + 0.1
