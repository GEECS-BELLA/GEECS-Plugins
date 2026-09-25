"""Loaded backgrounds obey explicit geometry, ownership and binding contracts."""

import numpy as np
import pytest
from pydantic import ValidationError

from geecs_analysis.pipeline import apply_pipeline, bind_inputs
from geecs_analysis.run import analyze
from geecs_analysis.specs import Analysis, Pipeline
from geecs_data_utils.frames import Axis, Frame, ShotMeta


def background_pipeline(**options):
    return Pipeline(steps=[{"step": "background_frame", "source": "dark", **options}])


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_background_subtraction_retains_negative_values_and_ownership(shape):
    raw = np.arange(np.prod(shape)).reshape(shape)
    frame = Frame.from_array(raw, unit="counts", shot=ShotMeta("camera", 7, 1.5))
    background = Frame.from_array(np.full(shape, 2), unit="counts")
    result = analyze(
        frame,
        Analysis(steps=background_pipeline().steps, measure={"kind": "none"}),
        inputs={"dark": background},
    ).frame
    np.testing.assert_array_equal(result.data, raw - 2)
    np.testing.assert_array_equal(frame.data, raw)
    np.testing.assert_array_equal(background.data, np.full(shape, 2))
    assert result.shot is frame.shot
    assert result.axes is frame.axes
    assert result.unit == "counts"
    assert not result.data.flags.writeable


def test_background_after_crop_requires_the_cropped_grid_and_preserves_order():
    frame = Frame.from_array([10, 20, 30, 40])
    dark = Frame.from_array([2, 50], axes=(Axis([1, 2]),))
    pipeline = Pipeline(
        steps=[
            {"step": "roi", "bounds": [[1, 3]]},
            *background_pipeline().steps,
            {"step": "clip_below", "level": 0},
            *background_pipeline().steps,
        ]
    )
    result = apply_pipeline(frame, pipeline, inputs={"dark": dark})
    np.testing.assert_array_equal(result.data, [16, -50])
    np.testing.assert_array_equal(result.axes[0].values, [1, 2])
    # The same samples on another grid are not an aligned background.
    with pytest.raises(ValueError, match="axes"):
        apply_pipeline(frame, pipeline, inputs={"dark": Frame.from_array([2, 50])})


@pytest.mark.parametrize(
    "background, message",
    [
        (Frame.from_array([[1, 2, 3]]), "shape"),
        (Frame.from_array([1, 2, 3], unit="counts"), "signal units"),
        (Frame.from_array([1, 2, 3], axes=(Axis([0, 1, 2], "mm"),)), "axes"),
        (Frame.from_array([1, 2, 3], axes=(Axis([2, 1, 0]),)), "axes"),
    ],
)
def test_default_alignment_rejects_geometry_and_unit_mismatches(background, message):
    with pytest.raises(ValueError, match=message):
        apply_pipeline(
            Frame.from_array([10, 20, 30]),
            background_pipeline(),
            inputs={"dark": background},
        )


def test_sample_alignment_is_explicit_and_never_broadcasts():
    frame = Frame.from_array([10, 20, 30], unit="counts")
    dark = Frame.from_array([1, 2, 3], axes=(Axis([30, 20, 10], "mm"),))
    pipeline = background_pipeline(alignment="samples")
    np.testing.assert_array_equal(
        apply_pipeline(frame, pipeline, inputs={"dark": dark}).data, [9, 18, 27]
    )
    with pytest.raises(ValueError, match="shape"):
        apply_pipeline(frame, pipeline, inputs={"dark": Frame.from_array([1])})


@pytest.mark.parametrize(
    "inputs, error", [(None, ValueError), ({"dark": [1, 2]}, TypeError)]
)
def test_all_bindings_are_validated_before_any_step_runs(inputs, error):
    pipeline = Pipeline(
        steps=[{"step": "roi", "bounds": [[99, 100]]}, *background_pipeline().steps]
    )
    # Executing the ROI first would raise an empty-selection error instead.
    with pytest.raises(error, match="[Ff]rame input"):
        apply_pipeline(Frame.from_array([1, 2]), pipeline, inputs=inputs)


def test_binding_snapshot_cannot_be_rebound_by_caller():
    dark = Frame.from_array([2, 3])
    supplied = {"dark": dark, "unused": object()}
    pipeline = background_pipeline()
    bound = bind_inputs(pipeline.steps, supplied)
    supplied["dark"] = Frame.from_array([100, 100])
    assert dict(bound) == {"dark": dark}
    with pytest.raises(TypeError):
        bound["dark"] = supplied["dark"]
    result = apply_pipeline(Frame.from_array([5, 5]), pipeline, inputs=bound)
    np.testing.assert_array_equal(result.data, [3, 2])


def test_binding_specs_serialize_only_names_and_reject_empty_names():
    pipeline = background_pipeline()
    assert Pipeline.model_validate_json(pipeline.model_dump_json()) == pipeline
    with pytest.raises(ValidationError):
        background_pipeline(source="")
