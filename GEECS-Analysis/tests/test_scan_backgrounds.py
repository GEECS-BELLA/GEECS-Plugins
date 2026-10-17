"""Scan backgrounds compile to host requests; the core never reads a scan."""

from __future__ import annotations

import pytest
from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisRecipe

from geecs_analysis.compat.convert import to_v3
from geecs_analysis.compat.v2 import ScanBackground, UnsupportedRecipe, compile_v2
from geecs_analysis.recipe import compile_recipe
from geecs_analysis.steps.background_constant import BackgroundConstantSpec
from geecs_analysis.steps.background_frame import BackgroundFrameSpec


def document(source, *, pipeline=("background",), image_type="camera", **background):
    image = (
        {"type": "line", "data_loading": {"data_type": "tsv"}}
        if image_type == "line"
        else {
            "type": "camera",
            "background": {"method": "constant", "constant_level": 2.0, **background},
            "pipeline": list(pipeline),
        }
    )
    return AnalysisDiagnostic.model_validate(
        {
            "name": "Cam",
            "analyzer": {"kind": "beam" if image_type == "camera" else "line"},
            "image": image,
            "scan": {"background_source": source},
        }
    )


def recipe(from_scan=None, steps=()):
    """A v3 camera recipe binding ``camera_background`` to a scan when given."""
    document = {
        "schema_version": 3,
        "device": "Cam",
        "input": {"kind": "camera"},
        "steps": list(steps),
        "measure": {"kind": "beam"},
    }
    if from_scan is not None:
        document["inputs"] = {"camera_background": {"from_scan": from_scan}}
    return compile_recipe(
        AnalysisRecipe.model_validate(document), allow_file_backgrounds=True
    )


@pytest.mark.parametrize(
    "source,expected",
    [
        ({"scan": 4}, ScanBackground("camera_background", 4, "mean")),
        (
            {"statistic": "median"},
            ScanBackground("camera_background", None, "median"),
        ),
        (
            {"statistic": "percentile", "percentile": 12.5},
            ScanBackground("camera_background", None, "percentile", 12.5),
        ),
    ],
)
def test_a_scan_background_compiles_to_a_request_and_a_frame_step(source, expected):
    """The legacy wrapper made the section a from_file background on the computed frame."""
    compiled = recipe(
        source,
        steps=[
            {
                "step": "background_frame",
                "source": "camera_background",
                "alignment": "samples",
            },
            {"step": "background_constant", "level": 1.5},
        ],
    )
    assert compiled.scan_backgrounds == (expected,)
    assert compiled.file_backgrounds == ()
    assert compiled.analysis.steps == (
        BackgroundFrameSpec(source="camera_background", alignment="samples"),
        BackgroundConstantSpec(level=1.5),
    )


def test_scan_backgrounds_need_a_host_and_autodetect_stays_unported():
    with pytest.raises(UnsupportedRecipe, match="resolved by a source"):
        compile_v2(document({"scan_number": 1}))
    with pytest.raises(UnsupportedRecipe, match="autodetect"):
        compile_v2(document({"autodetect": {}}), allow_file_backgrounds=True)
    with pytest.raises(UnsupportedRecipe, match="camera recipes only"):
        compile_v2(
            document({"scan_number": 1}, image_type="line"),
            allow_file_backgrounds=True,
        )


def test_no_background_step_means_no_request():
    """A background section outside the pipeline was never applied, as before."""
    compiled = recipe()
    assert compiled.scan_backgrounds == () and compiled.analysis.steps == ()


@pytest.mark.parametrize(
    "source",
    [
        {"scan_number": 3},
        {"from_current_scan": {"method": "percentile", "percentile": 20}},
    ],
)
def test_a_scan_background_converts_to_a_from_scan_input(source):
    document_v2 = document(source)
    conversion = to_v3(document_v2)
    binding = conversion.recipe.inputs["camera_background"]
    assert binding.path is None and binding.from_scan is not None
    assert compile_recipe(conversion.recipe, allow_file_backgrounds=True) == compile_v2(
        document_v2, allow_file_backgrounds=True
    )


@pytest.mark.parametrize(
    "binding,message",
    [
        ({}, "exactly one"),
        ({"path": "x.png", "from_scan": {"statistic": "mean"}}, "exactly one"),
        ({"from_scan": {"statistic": "percentile"}}, "percentile is required"),
        (
            {"from_scan": {"statistic": "median", "percentile": 5}},
            "percentile is required",
        ),
        ({"from_scan": {"statistic": "mean"}, "fallback_level": 1.0}, "fallback_level"),
    ],
)
def test_a_frame_input_names_exactly_one_source(binding, message):
    with pytest.raises(ValueError, match=message):
        AnalysisRecipe.model_validate(
            {
                "schema_version": 3,
                "device": "Cam",
                "input": {"kind": "camera"},
                "inputs": {"camera_background": binding},
                "steps": [{"step": "background_frame", "source": "camera_background"}],
            }
        )
