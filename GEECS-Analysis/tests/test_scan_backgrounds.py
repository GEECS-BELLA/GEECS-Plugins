"""Scan backgrounds compile to host requests; the core never reads a scan."""

from __future__ import annotations

import pytest
from geecs_schemas.analysis import AnalysisRecipe

from geecs_analysis.engine.v2 import ScanBackground
from geecs_analysis.recipe import RecipeError, compile_recipe
from geecs_analysis.steps.background_constant import BackgroundConstantSpec
from geecs_analysis.steps.background_frame import BackgroundFrameSpec


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


def test_scan_backgrounds_need_a_host():
    with pytest.raises(RecipeError, match="need a source host"):
        compile_recipe(
            AnalysisRecipe.model_validate(
                {
                    "schema_version": 3,
                    "device": "Cam",
                    "input": {"kind": "camera"},
                    "inputs": {"camera_background": {"from_scan": {"scan": 4}}},
                    "steps": [
                        {"step": "background_frame", "source": "camera_background"}
                    ],
                    "measure": {"kind": "beam"},
                }
            )
        )


def test_no_background_step_means_no_request():
    """A background section outside the pipeline was never applied, as before."""
    compiled = recipe()
    assert compiled.scan_backgrounds == () and compiled.analysis.steps == ()


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
