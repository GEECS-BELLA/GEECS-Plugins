"""Saved averages retain legacy dtype, NaN and post-analysis semantics."""

import numpy as np
import pytest
from geecs_data_utils.frames import Axis, Frame, ShotMeta
from geecs_schemas.analysis import AnalysisDiagnostic

from geecs_analysis.compat.v2 import analyze_v2, compile_v2
from geecs_analysis.compat.v2_average import average_results
from geecs_analysis.measurement import Measurement, Projection


def recipe(kind="camera", storage="float32"):
    data = {"type": kind, "pipeline": []}
    if kind == "line":
        data.update(data_loading={"data_type": "npy"}, storage_dtype=storage)
    doc = AnalysisDiagnostic.model_validate(
        {
            "name": "Detector",
            "analyzer": {"kind": "beam" if kind == "camera" else "line"},
            "image": data,
        }
    )
    return doc, compile_v2(doc)


@pytest.mark.parametrize(
    "kind,storage", [("camera", "float64"), ("line", "float32"), ("line", "float64")]
)
@pytest.mark.parametrize("mode", ["noscan", "bin"])
def test_saved_averages_match_old_results_without_reanalysis(kind, storage, mode):
    from image_analysis.ephemeral import run_document_ephemeral
    from image_analysis.types import ImageAnalyzerResult

    doc, compiled = recipe(kind, storage)
    raw = []
    for shift in (0.2, 0.9, 1.7):
        if kind == "camera":
            y, x = np.mgrid[:15, :18]
            array = np.exp(-((x - 7 - shift) ** 2 + (y - 8) ** 2) / 9)
        else:
            x = np.linspace(0.03, 20.07, 101, dtype=np.float32)
            array = np.column_stack((x + shift, np.exp(-((x - 7 - shift) ** 2) / 4)))
        raw.append(array)
    old = run_document_ephemeral(doc, raw)
    new = [
        analyze_v2(array, compiled, shot=ShotMeta("Detector", n))
        for n, array in enumerate(raw, 1)
    ]
    result = average_results(new, compiled, mode=mode)
    if mode == "bin":
        expected = ImageAnalyzerResult.average(old)
        expected_data = expected.get_primary_data()
        scalars = expected.scalars
    else:
        expected_data = np.mean([r.get_primary_data() for r in old], axis=0)
        scalars = {
            key: np.mean([r.scalars[key] for r in old]) for key in old[0].scalars
        }
    actual_data = result.frame.data if kind == "camera" else result.frame.as_trace()
    np.testing.assert_array_equal(actual_data, expected_data)
    assert all(np.isfinite(v) for v in scalars.values())
    assert dict(result.scalars) == scalars
    assert result.frame.shot is None
    if mode == "noscan":
        assert not result.overlays
    elif kind == "camera":
        projections = {o.id: o for o in result.overlays if isinstance(o, Projection)}
        np.testing.assert_array_equal(
            projections["projection_x"].frame.data,
            expected.render_data["horizontal_projection"],
        )
        np.testing.assert_array_equal(
            projections["projection_y"].frame.data,
            expected.render_data["vertical_projection"],
        )
        assert all(o.frame.shot is None for o in projections.values())
    # Summary means are statistics of measurements, not a fresh nonlinear fit.
    rerun = analyze_v2(np.mean(raw, axis=0), compiled)
    width = "x_rms" if kind == "camera" else "rms"
    assert result.scalars[width] != pytest.approx(rerun.scalars[width])


def test_noscan_propagates_nan_while_bin_ignores_it():
    _, compiled = recipe()
    results = [
        Measurement(
            {"value": np.nan, "always_bad": np.nan},
            Frame.from_array([[1, np.nan], [3, 4]]),
        ),
        Measurement(
            {"value": 6, "always_bad": np.nan}, Frame.from_array([[5, 6], [7, 8]])
        ),
    ]
    no = average_results(results, compiled, mode="noscan")
    bin_result = average_results(results, compiled, mode="bin")
    assert np.isnan(no.scalars["value"])
    assert np.isnan(no.frame.data[0, 1])
    assert bin_result.scalars["value"] == 6
    assert bin_result.frame.data[0, 1] == 6
    assert np.isnan(bin_result.scalars["always_bad"])
    assert "Nonfinite scalar: always_bad" in bin_result.notes
    assert np.isnan(results[0].frame.data[0, 1])


def test_empty_or_mixed_shapes_skip_the_average():
    _, compiled = recipe()
    assert average_results([], compiled, mode="bin") is None
    results = [
        Measurement({}, Frame.from_array(np.zeros(shape))) for shape in ((2, 2), (3, 2))
    ]
    assert average_results(results, compiled, mode="noscan") is None


def test_trace_axes_average_index_wise_at_storage_precision():
    _, compiled = recipe("line", "float32")
    arrays = [
        np.column_stack(
            (np.array([0.1, 0.3, 0.9], np.float32) + shift, [1, 2, 3])
        ).astype(np.float32)
        for shift in (0, 0.01, 0.03)
    ]
    results = [Measurement({}, Frame.from_trace(array)) for array in arrays]
    average = average_results(results, compiled, mode="bin")
    np.testing.assert_array_equal(average.frame.as_trace(), np.nanmean(arrays, axis=0))


@pytest.mark.parametrize("mismatch", ["unit", "axis", "rank"])
def test_incompatible_physical_metadata_is_not_silently_discarded(mismatch):
    _, compiled = recipe()
    first = Frame.from_array(np.ones((2, 2)))
    if mismatch == "unit":
        other = Frame.from_array(np.ones((2, 2)), unit="V")
    elif mismatch == "axis":
        other = first.replace(data=first.data, axes=(Axis([1, 2]), Axis([0, 1])))
    else:
        other = Frame.from_array([1, 2])
    with pytest.raises(ValueError):
        average_results(
            [Measurement({}, first), Measurement({}, other)], compiled, mode="bin"
        )


def test_scalar_key_policy_matches_each_legacy_summary():
    _, compiled = recipe()
    frame = Frame.from_array(np.ones((2, 2)))
    results = [
        Measurement({}, frame),
        Measurement({"x": 2}, frame),
        Measurement({"x": 4, "y": 8}, frame),
    ]
    assert not average_results(results, compiled, mode="noscan").scalars
    assert dict(average_results(results, compiled, mode="bin").scalars) == {"x": 3}
    assert dict(average_results(results[1:], compiled, mode="noscan").scalars) == {
        "x": 3,
        "y": 8,
    }
