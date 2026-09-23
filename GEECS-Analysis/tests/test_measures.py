"""Scientific compatibility, source ownership and scalar discovery."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from geecs_analysis.measurement import Marker, Measurement, Projection
from geecs_analysis.measures.beam import BeamSpec
from geecs_analysis.measures.line import LineSpec
from geecs_analysis.run import analyze
from geecs_analysis.specs import Analysis
from geecs_data_utils.frames import Axis, Frame, ShotMeta


def recipe(kind, **options):
    return Analysis.model_validate({"measure": {"kind": kind, **options}})


def assert_finite_scalars_equal(actual, expected):
    assert actual.keys() == expected.keys()
    assert all(np.isfinite(list(expected.values())))
    assert dict(actual) == expected


@pytest.mark.parametrize("shape", [(11, 13), (23, 17)])
@pytest.mark.parametrize("slopes", [False, True])
def test_beam_matches_legacy_in_global_pixel_coordinates(shape, slopes):
    from image_analysis.algorithms.basic_beam_stats import (
        beam_profile_stats,
        flatten_beam_stats,
    )
    from image_analysis.algorithms.beam_slopes import compute_beam_slopes

    samples = np.random.default_rng(7).uniform(1, 100, shape)
    expected = flatten_beam_stats(beam_profile_stats(samples, roi_offset=(30, 20)))
    if slopes:
        expected.update(compute_beam_slopes(samples))
    frame = Frame.from_array(
        samples,
        axes=(
            Axis(np.arange(shape[0]) + 20, label="y"),
            Axis(np.arange(shape[1]) + 30, label="x"),
        ),
        shot=ShotMeta("camera", 3, 45),
    )
    result = analyze(frame, recipe("beam", compute_slopes=slopes))
    assert_finite_scalars_equal(result.scalars, expected)
    assert result.frame is frame
    assert not result.notes
    np.testing.assert_array_equal(frame.data, samples)
    projections = {o.id: o for o in result.overlays if isinstance(o, Projection)}
    np.testing.assert_array_equal(
        projections["projection_x"].frame.data, samples.sum(axis=0)
    )
    assert projections["projection_x"].frame.axes[0] is frame.axes[1]
    assert projections["projection_x"].frame.shot is frame.shot
    marker = next(o for o in result.overlays if isinstance(o, Marker))
    assert (marker.x, marker.y) == (expected["x_CoM"], expected["y_CoM"])


def test_beam_affine_coordinates_transform_centroids_and_widths():
    samples = np.random.default_rng(8).uniform(1, 10, (9, 11))
    local = analyze(Frame.from_array(samples), recipe("beam"))
    physical = analyze(
        Frame.from_array(
            samples,
            axes=(
                Axis(np.arange(9) * 0.25 + 4, "mm", "y"),
                Axis(np.arange(11) * 0.5 + 10, "mm", "x"),
            ),
        ),
        recipe("beam"),
    )
    for dim, scale, offset in [("x", 0.5, 10), ("y", 0.25, 4)]:
        assert physical.scalars[f"{dim}_CoM"] == pytest.approx(
            local.scalars[f"{dim}_CoM"] * scale + offset
        )
        for width in ["rms", "fwhm"]:
            assert physical.scalars[f"{dim}_{width}"] == pytest.approx(
                local.scalars[f"{dim}_{width}"] * scale
            )
    for key in local.scalars:
        if key.startswith(("x_45", "y_45", "image")):
            assert physical.scalars[key] == local.scalars[key]


@pytest.mark.parametrize(
    "coordinates",
    [
        np.arange(9),
        np.linspace(60, 160, 9),
        np.array([2, 3, 6, 7, 10, 12, 15, 19, 20]),
        np.arange(9)[::-1],
    ],
)
def test_line_matches_legacy_coordinate_conversion(coordinates):
    from image_analysis.algorithms.basic_line_stats import LineBasicStats

    trace = np.column_stack((coordinates, [1, 2, 4, 8, 12, 9, 4, 2, 1]))
    expected = LineBasicStats(line_data=trace.copy()).to_dict()
    result = analyze(Frame.from_trace(trace, x_unit="MeV"), recipe("line"))
    assert_finite_scalars_equal(result.scalars, expected)
    assert not result.notes


def test_negative_line_samples_preserve_legacy_scalars_without_mutating_frame():
    from image_analysis.algorithms.basic_line_stats import LineBasicStats

    trace = np.column_stack((np.arange(9), [-1, 2, 4, 8, 12, 9, 4, 2, -2]))
    expected = LineBasicStats(line_data=trace.copy()).to_dict()
    frame = Frame.from_trace(trace)
    result = analyze(frame, recipe("line"))
    assert_finite_scalars_equal(result.scalars, expected)
    np.testing.assert_array_equal(frame.as_trace(), trace)
    assert result.frame is frame
    # Deliberately retain the legacy sum after internal negative clipping.
    assert result.scalars["integrated_intensity"] == 41


@pytest.mark.parametrize(
    "kind,frame",
    [
        ("line", Frame.from_array(np.zeros(9))),
        ("beam", Frame.from_array(np.zeros((5, 7)))),
    ],
)
def test_zero_signal_marks_undefined_metrics_without_claiming_parity(kind, frame):
    result = analyze(frame, recipe(kind))
    assert result.notes
    for key, value in result.scalars.items():
        assert (f"Nonfinite scalar: {key}" in result.notes) == (not np.isfinite(value))
    if kind == "line":
        assert result.scalars["integrated_intensity"] == 0
        assert result.scalars["peak_location"] == 0
        assert np.isnan(result.scalars["CoM"])
    else:
        assert result.scalars["image_total"] == 0
        assert not any(isinstance(o, Marker) for o in result.overlays)


@pytest.mark.parametrize("enabled", [None, [], ["x_CoM", "image_total"], ["typo"]])
@pytest.mark.parametrize("slopes", [False, True])
def test_scalar_discovery_preserves_v2_selection_contract(enabled, slopes):
    from geecs_schemas.analysis import BeamAnalyzerSpec

    spec = BeamSpec(enabled_stats=enabled, compute_slopes=slopes)
    old = BeamAnalyzerSpec(enabled_stats=enabled, compute_slopes=slopes)
    assert spec.emitted_scalars() == old.emitted_scalars()
    result = analyze(Frame.from_array(np.ones((3, 4))), Analysis(measure=spec))
    assert set(result.scalars) == spec.emitted_scalars()
    assert LineSpec().emitted_scalars() == set(
        analyze(Frame.from_array([1, 2, 1]), recipe("line")).scalars
    )


def test_immutable_measurement_owns_the_scalar_mapping():
    scalars = {"total": 3}
    result = Measurement(scalars, Frame.from_array([1, 2]))
    scalars["total"] = 99
    assert result.scalars["total"] == 3
    with pytest.raises(TypeError):
        result.scalars["total"] = 6
    with pytest.raises(TypeError):
        Measurement({"bad": "a label"}, result.frame)
    with pytest.raises(ValueError):
        Measurement({}, result.frame, (Marker("com", 1, 2), Marker("com", 3, 4)))


@pytest.mark.parametrize("kind,shape", [("beam", (5,)), ("line", (5, 5))])
def test_wrong_dimensionality_is_rejected_before_processing(kind, shape):
    with pytest.raises(ValueError, match="does not support"):
        analyze(Frame.from_array(np.ones(shape)), recipe(kind))


def test_process_and_measure_runs_in_order_and_is_safe_to_reuse_concurrently():
    frame = Frame.from_array(np.arange(81).reshape(9, 9))
    doc = Analysis.model_validate(
        {
            "steps": [
                {"step": "roi", "bounds": [[2, 8], [3, 9]]},
                {"step": "background_constant", "level": 20},
                {"step": "zero_below", "level": 0},
            ],
            "measure": {"kind": "beam"},
        }
    )
    expected = analyze(frame, doc)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: analyze(frame, doc), range(12)))
    for result in results:
        assert result.scalars == expected.scalars
        np.testing.assert_array_equal(result.frame.data, expected.frame.data)
    np.testing.assert_array_equal(expected.frame.axes[1].values, np.arange(3, 9))
    np.testing.assert_array_equal(frame.data, np.arange(81).reshape(9, 9))


def test_preprocessing_only_none_measure():
    doc = Analysis.model_validate(
        {"steps": [{"step": "background_constant", "level": 1}]}
    )
    result = analyze(Frame.from_array([2, 3]), doc)
    np.testing.assert_array_equal(result.frame.data, [1, 2])
    assert result.scalars == {} and result.overlays == ()
    assert doc.measure.emitted_scalars() == frozenset()


def test_complete_recipe_schema_and_discovery_have_no_numerical_imports():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from geecs_analysis.specs import Analysis
for kind, count in (("beam", 18), ("line", 6), ("none", 0)):
    recipe = Analysis.model_validate({"measure": {"kind": kind}})
    assert len(recipe.measure.emitted_scalars()) == count
Analysis.model_json_schema()
for module in ("numpy", "scipy", "matplotlib", "geecs_data_utils", "image_analysis"):
    assert module not in sys.modules, module
""",
        ],
        check=True,
    )


def test_unresolved_fwhm_is_reported_for_a_single_sample_peak():
    result = analyze(Frame.from_array([1, 2, 4, 8, 20, 9, 4, 2, 1]), recipe("line"))
    assert np.isnan(result.scalars["fwhm"])
    assert result.notes == ("Nonfinite scalar: fwhm",)
