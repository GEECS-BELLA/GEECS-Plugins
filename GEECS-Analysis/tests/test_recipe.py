"""A recipe (v3) compiles to the same in-memory recipe as the v2 diagnostic it came from."""

import pytest
from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisRecipe
from geecs_schemas.analysis.canonical import canonical_document
from pydantic import ValidationError

from geecs_analysis.compat.convert import to_v3
from geecs_analysis.compat.v2 import UnsupportedRecipe, compile_v2
from geecs_analysis.recipe import (
    RecipeError,
    compile_document,
    compile_recipe,
    figure_of,
    is_line,
    summaries_of,
)


def diagnostic(kind="beam", *, scan=None, **image):
    if kind in {"line", "trace"}:
        image = {"type": "line", "data_loading": {"data_type": "tsv"}, **image}
    else:
        image = {"type": "camera", "bit_depth": 12, **image}
    return AnalysisDiagnostic.model_validate(
        {
            "name": "Device",
            "output_name": "Output",
            "metric_suffix": "_left",
            "analyzer": {
                "kind": kind,
                **({"compute_slopes": True} if kind == "beam" else {}),
            },
            "image": image,
            "scan": scan or {},
        }
    )


CAMERA = dict(
    pipeline=["background", "roi", "crosshair_masking", "filtering", "thresholding"],
    background={
        "method": "from_file",
        "file_path": "{scan_dir}/bg.npy",
        "constant_level": 3,
    },
    roi={"x_min": 5, "x_max": 20, "y_min": 2, "y_max": 12},
    crosshair_masking={
        "crosshairs": [{"center": [4, 6], "width": 3, "height": 3, "thickness": 1}]
    },
    filtering={"gaussian_sigma": 1.5, "median_kernel_size": 3},
    thresholding={"method": "constant", "mode": "to_zero", "value": 7},
)
LINE = dict(
    pipeline=["background", "interpolation", "roi", "filtering", "thresholding"],
    background={"method": "constant", "constant_level": 0.5},
    interpolation={"num_points": 20, "x_min": 0, "x_max": 10},
    roi={"x_min": 1, "x_max": 9},
    filtering={"method": "median", "kernel_size": 3},
    thresholding={"method": "absolute", "threshold_value": -1, "clip_below": True},
    x_scale_factor=1000.0,
    x_units="MeV",
    label="Charge",
    storage_dtype="float32",
)


@pytest.mark.parametrize(
    "doc",
    [
        diagnostic(
            "beam", **CAMERA, scan={"mode": "per_bin", "priority": 3, "save": False}
        ),
        diagnostic(
            "line", **LINE, scan={"device": "Device-interp", "file_tail": ".txt"}
        ),
        diagnostic("standard"),
        diagnostic("trace", storage_dtype="float64"),
        diagnostic(
            "beam",
            scan={
                "renderer": {
                    "cmap": "viridis",
                    "vmax": 9,
                    "xlabel": "x",
                    "dpi": 40,
                    "figsize": [3, 2],
                }
            },
        ),
        diagnostic(
            "line",
            scan={
                "renderer": {
                    "waterfall_sort_key": "U_Charge",
                    "waterfall_sort_bounds": [1, 2],
                    "colormap_mode": "diverging",
                    "cmap": "RdBu",
                }
            },
        ),
    ],
    ids=[
        "camera-full",
        "line-full",
        "standard",
        "trace-float64",
        "camera-renderer",
        "line-renderer",
    ],
)
def test_converted_recipe_compiles_and_serializes_identically(doc):
    source = compile_v2(doc, allow_file_backgrounds=True)
    conversion = to_v3(doc)
    recipe = conversion.recipe
    assert compile_recipe(recipe, allow_file_backgrounds=True) == source
    # through YAML-ready form and back, still the same
    reread = AnalysisRecipe.model_validate(canonical_document(recipe))
    assert compile_recipe(reread, allow_file_backgrounds=True) == source
    assert compile_document(reread, allow_file_backgrounds=True) == source
    assert is_line(reread) == is_line(doc)
    assert recipe.effective_output_name == "Output" and recipe.scalar_suffix == "_left"
    assert not any("coordinates" in note for note in conversion.notes)


def test_conversion_carries_runtime_input_and_rendering_facts():
    doc = diagnostic(
        "beam",
        **CAMERA,
        scan={"mode": "per_bin", "priority": 3, "save": False, "gdoc_slot": 1},
    )
    conversion = to_v3(doc)
    recipe = conversion.recipe
    assert (
        recipe.scan.average_frames_first
        and recipe.scan.priority == 3
        and not recipe.scan.save
    )
    # constant_level on a from_file background is the fallback, not a step
    assert [s.step for s in recipe.steps] == [
        "background_frame",
        "roi",
        "crosshair_mask",
        "gaussian",
        "median",
        "zero_below",
    ]
    assert recipe.inputs == {
        "camera_background": {"path": "{scan_dir}/bg.npy", "fallback_level": 3.0}
    } or (
        recipe.inputs["camera_background"].path == "{scan_dir}/bg.npy"
        and recipe.inputs["camera_background"].fallback_level == 3.0
    )
    assert recipe.figure.imshow == {"vmin": 0} and recipe.figure.fig == {"dpi": 150}
    assert [s.kind for s in recipe.summaries] == ["image_grid", "average"]
    # the v2 grid's panel size and resolution are written out, not defaulted
    assert recipe.summaries[0].panel_size == (6, 6)
    assert "scan.gdoc_slot dropped (retired)" in conversion.notes
    assert "image.bit_depth dropped: the core does not use it" in conversion.notes

    line = to_v3(
        diagnostic(
            "line",
            **LINE,
            scan={
                "device": "Device-interp",
                "file_tail": ".txt",
                "renderer": {"waterfall_sort_key": "U_Charge", "vmin": -1},
            },
        )
    ).recipe
    assert line.input.folder == "Device-interp" and line.input.file_tail == ".txt"
    assert line.input.x_scale == 1000.0 and line.input.x_unit == "MeV"
    assert line.figure.imshow == {} and line.figure.fig == {"dpi": 150}
    stack, average = line.summaries
    assert (
        stack.kind == "waterfall" and stack.sort_key == "U_Charge" and stack.vmin == -1
    )
    assert average.kind == "average"


def test_inactive_roi_origin_is_reported_not_reproduced():
    doc = diagnostic("beam", roi={"x_min": 5, "x_max": 20, "y_min": 2, "y_max": 12})
    assert compile_v2(doc).camera_origin == (2, 5)
    conversion = to_v3(doc)
    assert compile_recipe(conversion.recipe).camera_origin == (0, 0)
    assert any("coordinates started at (2, 5)" in note for note in conversion.notes)


def test_unported_recipes_are_not_converted():
    with pytest.raises(UnsupportedRecipe):
        to_v3(
            diagnostic(
                "beam", pipeline=["transforms"], transforms={"flip_horizontal": True}
            )
        )


def recipe(**patch):
    return AnalysisRecipe.model_validate(
        {
            "device": "D",
            "input": {"kind": "camera"},
            "measure": {"kind": "beam"},
            **patch,
        }
    )


@pytest.mark.parametrize(
    "patch,needle",
    [
        ({"steps": [{"step": "sharpen"}]}, "does not bind"),
        ({"steps": [{"step": "median", "kernel": 3, "bogus": 1}]}, "does not bind"),
        ({"steps": [{"step": "interpolate", "count": 5}]}, "does not process camera"),
        ({"measure": {"kind": "line"}}, "does not measure camera"),
        ({"steps": [{"step": "background_frame", "source": "bg"}]}, "does not declare"),
        ({"inputs": {"bg": {"path": "x.npy"}}}, "no step uses"),
    ],
)
def test_binding_errors_are_recipe_errors(patch, needle):
    with pytest.raises(RecipeError, match=needle):
        compile_recipe(recipe(**patch), allow_file_backgrounds=True)


def test_frame_inputs_need_a_loading_host_and_camera_frames():
    bound = recipe(
        steps=[{"step": "background_frame", "source": "bg", "alignment": "samples"}],
        inputs={"bg": {"path": "x.npy"}},
    )
    with pytest.raises(RecipeError, match="source host"):
        compile_recipe(bound)
    compiled = compile_recipe(bound, allow_file_backgrounds=True)
    assert compiled.file_backgrounds[0].fallback_level is None
    line = AnalysisRecipe.model_validate(
        {
            "device": "D",
            "input": {"kind": "line", "loading": {"data_type": "tsv"}},
            "steps": [{"step": "background_frame", "source": "bg"}],
            "inputs": {"bg": {"path": "x.npy"}},
        }
    )
    with pytest.raises(RecipeError, match="camera recipes only"):
        compile_recipe(line, allow_file_backgrounds=True)


def test_v3_figure_and_summaries_are_as_declared_and_v2_are_translated():
    declared = recipe(
        figure={"imshow": {"cmap": "gray"}, "overlays": {"com": {"hidden": True}}},
        summaries=[{"kind": "average"}],
    )
    assert figure_of(declared).imshow == {"cmap": "gray"}
    assert figure_of(declared).overlays == {"com": {"hidden": True}}
    assert [s.kind for s in summaries_of(declared)] == ["average"]
    v2 = diagnostic("beam", scan={"renderer": {"cmap": "gray", "figsize": [2, 3]}})
    assert figure_of(v2).imshow == {"cmap": "gray", "vmin": 0}
    assert figure_of(v2).fig == {"figsize": (4, 4), "dpi": 150}
    grid, average = summaries_of(v2)
    assert grid.kind == "image_grid" and grid.panel_size == (2, 3)
    assert average.kind == "average"


def test_schema_refuses_what_the_core_cannot_see():
    with pytest.raises(ValidationError):
        AnalysisRecipe.model_validate(
            {
                "device": "D",
                "input": {"kind": "camera"},
                "summaries": [{"kind": "waterfall"}],
            }
        )
