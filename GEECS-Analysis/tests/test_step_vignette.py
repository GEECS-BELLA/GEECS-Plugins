"""The vignette step: the legacy radial map bit for bit, on sensor pixels."""

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisRecipe

from geecs_analysis.compat.v2 import UnsupportedRecipe, analyze_v2, compile_v2
from geecs_analysis.pipeline import apply_pipeline
from geecs_analysis.recipe import compile_recipe
from geecs_analysis.specs import Pipeline
from geecs_data_utils.frames import Axis, Frame, ShotMeta

MODEL = dict(
    full_width=1400, full_height=1100, vgnt4=-2.0e-13, vgnt2=3.0e-7, vgnt0=0.98
)


def run(frame, **params):
    return apply_pipeline(
        frame, Pipeline.model_validate({"steps": [{"step": "vignette", **params}]})
    )


@pytest.mark.parametrize(
    "shape, offsets, model",
    [
        ((7, 9), (0, 0), {"full_width": 9, "full_height": 7}),
        ((40, 64), (25, 170), MODEL),
        ((33, 50), (3, 11), {**MODEL, "vgnt4": 4e-9, "vgnt2": -5e-4}),
        ((20, 30), (0, 0), {**MODEL, "vgnt0": -1.0, "min_model_value": 1e-3}),
    ],
)
def test_equals_the_legacy_map_bit_for_bit(shape, offsets, model, expected, request):
    raw = np.random.default_rng(3).normal(100, 20, shape)
    x_offset, y_offset = offsets
    frame = Frame.from_array(raw, shot=ShotMeta("camera", 1, 0.0))
    result = run(frame, x_offset=x_offset, y_offset=y_offset, **model)
    expected(
        f"vignette_{request.node.callspec.id}",
        actual=result.data,
    )
    assert result.axes is frame.axes and result.shot is frame.shot
    np.testing.assert_array_equal(frame.data, raw)


def test_after_a_crop_is_the_crop_of_the_corrected_frame():
    raw = np.random.default_rng(4).normal(100, 20, (40, 60))
    frame = Frame.from_array(raw)
    crop = {"step": "roi", "bounds": [[5, 31], [12, 47]]}
    vignette = {"step": "vignette", "x_offset": 25, "y_offset": 170, **MODEL}
    before = apply_pipeline(frame, Pipeline(steps=[vignette, crop]))
    after = apply_pipeline(frame, Pipeline(steps=[crop, vignette]))
    np.testing.assert_array_equal(after.data, before.data)


def test_the_floor_bounds_the_model_before_it_is_inverted():
    result = run(Frame.from_array(np.ones((4, 4))), **{**MODEL, "vgnt0": -1.0})
    np.testing.assert_array_equal(result.data, np.full((4, 4), 1.0 / 1e-9))


def test_non_pixel_axes_are_refused():
    frame = Frame.from_array(
        np.ones((3, 4)),
        axes=(Axis(np.arange(3.0)), Axis(np.linspace(0, 1, 4), unit="mm")),
    )
    with pytest.raises(ValueError, match="px"):
        run(frame, **MODEL)


def camera(pipeline, **sections):
    return AnalysisDiagnostic.model_validate(
        {
            "name": "diagnostic",
            "output_name": "output",
            "analyzer": {"kind": "standard"},
            "image": {"type": "camera", "pipeline": pipeline, **sections},
        }
    )


SECTIONS = dict(
    background={"method": "constant", "constant_level": 3, "additional_constant": 1},
    vignette={"x_offset": 25, "y_offset": 170, **MODEL},
    roi={"x_min": 4, "x_max": 40, "y_min": 2, "y_max": 27},
    thresholding={"method": "constant", "value": 0, "mode": "to_zero"},
    filtering={"median_kernel_size": 5},
    transforms={"rotation_angle": 0.2},
)


def test_the_htt_magcam_chain_matches_the_legacy_pipeline(expected):
    # The HTT-MagCam documents' order: background, vignette, roi, ...
    steps = [
        {"step": "background_constant", "level": 3},
        {"step": "background_constant", "level": 1},
        {"step": "vignette", "x_offset": 25, "y_offset": 170, **MODEL},
        {"step": "roi", "bounds": [[2, 27], [4, 40]]},
        {"step": "zero_below", "level": 0},
        {"step": "median", "kernel": 5},
        {"step": "rotate", "angle": 0.2},
    ]
    document = {
        "schema_version": 3,
        "device": "diagnostic",
        "output_name": "output",
        "input": {"kind": "camera"},
        "steps": steps,
    }
    data = np.random.default_rng(5).integers(0, 4000, (31, 45), dtype=np.uint16)
    recipe = compile_recipe(AnalysisRecipe.model_validate(document))
    assert [s.step for s in recipe.analysis.steps][:3] == [
        "background_constant",
        "background_constant",
        "vignette",
    ]
    result = analyze_v2(data, recipe, shot=ShotMeta("diagnostic", 1, 0.0))
    expected(
        "htt_magcam_chain",
        actual=result.frame.data,
    )


def test_a_vignette_after_an_active_roi_is_refused():
    for pipeline in (["roi", "vignette"], ["vignette", "roi", "vignette"]):
        with pytest.raises(UnsupportedRecipe, match="after roi"):
            compile_v2(camera(pipeline, **SECTIONS))
    # An inactive roi is no crop: the order compiles.
    sections = {**SECTIONS, "roi": None}
    compile_v2(camera(["roi", "vignette"], **sections))


def test_a_map_file_vignette_is_refused():
    vignette = {"method": "map_file", "map_file_path": "never-opened.npy"}
    with pytest.raises(UnsupportedRecipe, match="map_file"):
        compile_v2(camera(["vignette"], vignette=vignette))
