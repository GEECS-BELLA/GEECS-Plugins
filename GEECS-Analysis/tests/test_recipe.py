"""A v3 recipe compiles, serializes canonically, and refuses what it cannot carry."""

import pytest
from geecs_schemas.analysis import AnalysisRecipe
from pydantic import ValidationError

from geecs_analysis.recipe import (
    RecipeError,
    compile_recipe,
    figure_of,
    recipe_schema,
    summaries_of,
)
from geecs_analysis.registry import definitions


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
        ({"inputs": {"bg": {"path": "x.npy"}}}, "no step or measure uses"),
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


def test_v3_figure_and_summaries_are_as_declared():
    declared = recipe(
        figure={"imshow": {"cmap": "gray"}, "overlays": {"com": {"hidden": True}}},
        summaries=[{"kind": "average"}],
    )
    assert figure_of(declared).imshow == {"cmap": "gray"}
    assert figure_of(declared).overlays == {"com": {"hidden": True}}
    assert [s.kind for s in summaries_of(declared)] == ["average"]


def test_schema_refuses_what_the_core_cannot_see():
    with pytest.raises(ValidationError):
        AnalysisRecipe.model_validate(
            {
                "device": "D",
                "input": {"kind": "camera"},
                "summaries": [{"kind": "waterfall"}],
            }
        )


def test_recipe_schema_binds_the_registry_vocabulary():
    """The form's schema lists every registered step, measure and summary, typed."""
    schema = recipe_schema()
    steps = schema["properties"]["steps"]["items"]["discriminator"]["mapping"]
    assert set(steps) == {d.spec.model_fields["step"].default for d in definitions()}
    assert "roi" in steps and "StepRef" not in schema["$defs"]
    assert "MeasureRef" not in schema["$defs"]
    measures = schema["properties"]["measure"]["discriminator"]["mapping"]
    assert set(measures) == {
        "beam",
        "downramp_phase",
        "frog",
        "frog_spectral_phase",
        "haso",
        "hi_res_mag_cam",
        "ict",
        "line",
        "magspec",
        "none",
        "pulsed_wire",
    }
    assert schema["$defs"]["RoiSpec"]["x-ndim"] == [1, 2]
    assert schema["$defs"]["BeamSpec"]["x-ndim"] == [2]
    assert schema["$defs"]["FrogSpec"]["x-ndim"] == [2]
    assert schema["$defs"]["HasoSpec"]["x-ndim"] == [2]
    assert schema["$defs"]["InterpolateSpec"]["x-ndim"] == [1]
    assert schema["$defs"]["WaterfallSummary"]["x-ndim"] == [1]
    assert schema["$defs"]["ImageGridSummary"]["x-ndim"] == [2]
    # the document's frame is the schema package's, untouched
    assert schema["properties"]["input"]["discriminator"]["propertyName"] == "kind"
    assert "FigureStyle" in schema["$defs"]
    # every parameter is described: the form shows the description as help
    for name, definition in schema["$defs"].items():
        if not name.endswith("Spec"):
            continue
        for field, prop in definition["properties"].items():
            if field not in {"step", "kind"}:
                assert prop.get("description"), (name, field)
    # the schema is plain JSON (the router serves it)
    import json

    json.dumps(schema)
