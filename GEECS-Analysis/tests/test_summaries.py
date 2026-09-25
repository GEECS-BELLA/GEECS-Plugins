"""The frozen summary kinds: registered once each, drawn from the per-frame figure."""

import subprocess
import sys

import numpy as np
import pytest
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import AverageSummary, ImageGridSummary, WaterfallSummary

from geecs_analysis.measurement import Measurement
from geecs_analysis.registry import summary_definition, summary_definitions
from geecs_analysis.render import RenderError
from geecs_analysis.render.specs import FigureSpec
from geecs_analysis.summaries.image_grid import DEFAULT_PANEL_SIZE


def image(value):
    return Measurement({}, Frame.from_array(np.full((4, 6), float(value))))


def trace(y):
    return Measurement(
        {}, Frame.from_trace(np.column_stack([[1, 2, 3], y]), x_unit="eV")
    )


def test_registry_lists_the_three_kinds_with_their_contracts():
    definitions = {d.spec: d for d in summary_definitions()}
    assert set(definitions) == {ImageGridSummary, WaterfallSummary, AverageSummary}
    assert definitions[ImageGridSummary].consumes == "panels"
    assert definitions[ImageGridSummary].filename == "averaged_image_grid"
    assert definitions[ImageGridSummary].ndim == {2}
    assert definitions[WaterfallSummary].consumes == "panels"
    assert definitions[WaterfallSummary].filename == "summary_waterfall"
    assert definitions[WaterfallSummary].ndim == {1}
    assert definitions[AverageSummary].consumes == "average"
    assert definitions[AverageSummary].filename == "average_processed_visual"
    assert definitions[AverageSummary].ndim == {1, 2}


def test_image_grid_sizes_the_canvas_from_the_panel_size_and_keeps_the_figure_palette():
    figure = FigureSpec(imshow={"vmin": 0, "cmap": "gray"}, fig={"dpi": 30})
    grid = ImageGridSummary(columns=2, panel_size=(3, 2))
    draw = summary_definition(grid).function
    fig = draw([image(1), image(5), image(9)], [1.0, 2.0, 3.0], "motor", grid, figure)
    assert tuple(fig.get_size_inches()) == (6, 4) and fig.dpi == 30
    assert fig.axes[0].images[0].get_clim() == (0, 9)
    assert fig.axes[0].images[0].cmap.name == "gray"
    assert [ax.get_title() for ax in fig.axes[:3]] == ["1.00", "2.00", "3.00"]
    assert fig._suptitle.get_text() == "Scan parameter: motor"
    default = draw(
        [image(1)], [None], "", ImageGridSummary(), FigureSpec(fig={"dpi": 30})
    )
    assert tuple(default.get_size_inches()) == DEFAULT_PANEL_SIZE
    assert default._suptitle is None


def test_average_draws_exactly_one_measurement_through_single():
    draw = summary_definition(AverageSummary()).function
    fig = draw([image(2)], [None], "", AverageSummary(), FigureSpec(fig={"dpi": 30}))
    assert len(fig.axes[0].images) == 1
    with pytest.raises(RenderError):
        draw([image(1), image(2)], [None, None], "", AverageSummary(), FigureSpec())


def test_waterfall_takes_its_palette_from_the_kind_and_labels_from_the_figure():
    stack = WaterfallSummary(
        scale="custom", vmin=-2, vmax=8, cmap="viridis", sort_key="k"
    )
    draw = summary_definition(stack).function
    figure = FigureSpec(
        axes={"xlabel": "energy", "ylabel": "ignored"},
        colorbar={"label": "Q"},
        fig={"dpi": 30, "figsize": (1, 1)},
    )
    fig = draw(
        [trace([1, 2, 3]), trace([4, 5, 6])], [10.0, 30.0], "charge", stack, figure
    )
    ax = fig.axes[0]
    artist = ax.collections[0]
    assert artist.get_clim() == (-2, 8) and artist.cmap.name == "viridis"
    assert ax.get_xlabel() == "energy" and ax.get_ylabel() == "charge"
    assert fig.axes[1].get_ylabel() == "Q"
    assert tuple(fig.get_size_inches()) == (10, 8) and fig.dpi == 30
    # sorting by a key spaces rows evenly with the values as tick labels
    np.testing.assert_array_equal(ax.get_yticks(), [0, 1])
    with pytest.raises(RenderError, match="finite"):
        draw([trace([1, 2, 3])], [float("nan")], "", stack, figure)


def test_kinds_are_registered_by_the_recipe_module_alone():
    """A process whose first document is a recipe must still resolve every kind."""
    script = (
        "from geecs_analysis.recipe import summaries_of\n"
        "from geecs_analysis.registry import summary_definition\n"
        "from geecs_schemas.analysis import AnalysisRecipe\n"
        "r = AnalysisRecipe.model_validate({'device': 'D', 'input': {'kind': 'camera'},"
        " 'summaries': [{'kind': 'image_grid'}, {'kind': 'average'}]})\n"
        "print(sorted(summary_definition(s).filename for s in summaries_of(r)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "['average_processed_visual', 'averaged_image_grid']"
