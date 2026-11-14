"""A v3 recipe compiles, serializes canonically, and refuses what it cannot carry."""

import pytest
from geecs_schemas.analysis import AnalysisRecipe
from pydantic import ValidationError

from geecs_analysis.recipe import (
    RecipeError,
    binding_errors,
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


@pytest.mark.parametrize(
    "patch,loc",
    [
        ({"steps": [{"step": "roi", "bounds": [[0, 4], [0, 4]]}, {"step": "sharpen"}]}, ["steps", 1]),
        ({"steps": [{"step": "median", "kernel": 3, "bogus": 1}]}, ["steps", 0, "bogus"]),
        ({"steps": [{"step": "median", "kernel": 3}, {"step": "interpolate", "count": 5}]}, ["steps", 1]),
        ({"measure": {"kind": "line"}}, ["measure"]),
        ({"inputs": {"bg": {"path": "x.npy"}}}, ["inputs"]),
    ],
)  # fmt: skip
def test_binding_errors_are_at_the_documents_field(patch, loc):
    """The editor's view of compile_recipe: the union variant dropped, a worded error at the field it names."""
    from geecs_analysis.recipe import binding_errors

    errors = binding_errors(recipe(**patch))
    assert errors and errors[0]["loc"][: len(loc)] == loc, errors
    assert binding_errors(recipe(steps=[{"step": "median", "kernel": 3}])) == []


def _blank_input_fields(kind: str) -> dict:
    """What an input kind needs beyond its kind, read off the schema (a trace's loading format)."""
    from geecs_schemas.analysis.processing_1d import Data1DType

    return {"loading": {"data_type": next(iter(Data1DType)).value}} if kind == "line" else {}  # fmt: skip


def test_every_frame_kind_has_one_default_measure_and_its_blank_binds():
    """A new recipe opens on a draft that compiles: for each input kind, its frames' default measure, no steps."""
    from geecs_analysis.recipe import blank_recipe, input_kinds
    from geecs_analysis.registry import default_measure, measure_definitions

    kinds = input_kinds()
    assert set(kinds.values()) == {1, 2}
    for ndim in {1, 2}:
        assert sum(1 for d in measure_definitions() if d.default and ndim in d.ndim) == 1  # fmt: skip
    for kind, ndim in kinds.items():
        doc = blank_recipe(kind, name="New_one", input_fields=_blank_input_fields(kind))
        recipe = AnalysisRecipe.model_validate(doc)
        assert binding_errors(recipe) == [] and recipe.steps == []
        assert (doc["output_name"], doc["device"], recipe.measure.kind) == ("New_one", "New_one", default_measure(ndim).spec.model_fields["kind"].default)  # fmt: skip
        compile_recipe(
            recipe
        )  # no frame inputs: it compiles even for a host that loads none


def test_a_blank_names_what_its_input_lacks_and_refuses_an_unknown_kind():
    from geecs_analysis.recipe import blank_recipe

    with pytest.raises(RecipeError, match="loading"):
        blank_recipe("line", name="L")
    with pytest.raises(RecipeError, match="no input kind"):
        blank_recipe("telescope", name="T")


def test_a_second_default_measure_for_a_dimensionality_is_refused():
    from geecs_analysis.registry import measure

    with pytest.raises(ValueError, match="second default"):

        @measure(AnalysisRecipe, ndim={2}, default=True)  # type: ignore[type-var]
        def _(frame, spec):  # pragma: no cover - never registered
            return None


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
