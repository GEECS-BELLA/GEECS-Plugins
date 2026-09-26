"""Results and compiled recipes round-trip through pickle for process-pool workers."""

import pickle

import numpy as np
import pytest
from geecs_data_utils.frames import Axis, Frame, ShotMeta
from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisRecipe

from geecs_analysis.compat.v2 import analyze_v2, compile_v2
from geecs_analysis.measurement import Marker, Measurement, Projection
from geecs_analysis.recipe import compile_document


def same_measurement(copy: Measurement, original: Measurement) -> None:
    assert list(copy.scalars) == list(original.scalars)
    for key, value in original.scalars.items():
        assert copy.scalars[key] == value or (
            np.isnan(value) and np.isnan(copy.scalars[key])
        )
    assert type(copy.scalars) is type(original.scalars)
    np.testing.assert_array_equal(copy.frame.data, original.frame.data)
    assert copy.frame.shot == original.frame.shot
    assert (copy.frame.unit, copy.frame.label) == (
        original.frame.unit,
        original.frame.label,
    )
    for a, b in zip(copy.frame.axes, original.frame.axes, strict=True):
        np.testing.assert_array_equal(a.values, b.values)
        assert (a.unit, a.label) == (b.unit, b.label)
    assert copy.notes == original.notes
    assert len(copy.overlays) == len(original.overlays)
    for a, b in zip(copy.overlays, original.overlays, strict=True):
        assert type(a) is type(b) and a.id == b.id
        if isinstance(a, Projection):
            assert a.axis == b.axis
            np.testing.assert_array_equal(a.frame.data, b.frame.data)
        else:
            assert (a.x, a.y) == (b.x, b.y)


def test_camera_measurement_round_trips_with_overlays_and_notes():
    frame = Frame.from_array(
        np.arange(12.0).reshape(3, 4),
        axes=(Axis([0, 1, 2], "px", "y"), Axis([3, 4, 5, 6], "px", "x")),
        shot=ShotMeta("cam", 7, 99.5),
        unit="counts",
        label="beam",
    )
    original = Measurement(
        {"x_CoM": 1.5, "x_rms": float("nan")},
        frame,
        (
            Projection("projection_x", 1, Frame.from_array([1.0, 2.0, 3.0, 4.0])),
            Marker("centroid", 1.0, 2.0),
        ),
        ("custom note",),
    )
    assert original.notes == ("custom note", "Nonfinite scalar: x_rms")
    copy = pickle.loads(pickle.dumps(original))
    same_measurement(copy, original)
    # The proxy view is restored, and the notes are not re-annotated.
    with pytest.raises(TypeError):
        copy.scalars["x_CoM"] = 2.0
    assert not copy.frame.data.flags.writeable


def test_line_measurement_from_the_evaluator_round_trips():
    doc = AnalysisDiagnostic.model_validate(
        {
            "name": "Spectrum",
            "analyzer": {"kind": "line"},
            "image": {
                "type": "line",
                "data_loading": {"data_type": "npy"},
                "x_scale_factor": 2.0,
                "storage_dtype": "float32",
            },
        }
    )
    compiled = compile_v2(doc)
    x = np.linspace(0.0, 10.0, 40)
    trace = np.column_stack([x, np.exp(-((x - 4) ** 2))])
    original = analyze_v2(trace, compiled, shot=ShotMeta("Spectrum", 2))
    copy = pickle.loads(pickle.dumps(original))
    same_measurement(copy, original)
    assert copy.frame.shot == ShotMeta("Spectrum", 2)


def test_compiled_recipes_of_both_formats_round_trip():
    v2 = compile_v2(
        AnalysisDiagnostic.model_validate(
            {
                "name": "Camera",
                "analyzer": {"kind": "beam", "compute_slopes": True},
                "image": {
                    "type": "camera",
                    "pipeline": ["roi", "filtering"],
                    "roi": {"x_min": 1, "x_max": 5, "y_min": 0, "y_max": 4},
                    "filtering": {"gaussian_sigma": 1.5},
                },
            }
        )
    )
    v3 = compile_document(
        AnalysisRecipe.model_validate(
            {
                "device": "Camera",
                "input": {"kind": "camera"},
                "steps": [{"step": "gaussian", "sigma": 1.5}],
                "measure": {"kind": "beam"},
            }
        )
    )
    for compiled in (v2, v3):
        assert pickle.loads(pickle.dumps(compiled)) == compiled
