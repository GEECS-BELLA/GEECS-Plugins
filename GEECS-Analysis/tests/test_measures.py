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
        np.arange(9)[::-1],
        # Evenly spaced to single precision only: still the legacy arithmetic.
        np.linspace(0.1, 0.9, 9).astype(np.float32).astype(float),
    ],
)
def test_line_matches_legacy_on_evenly_spaced_axes(coordinates):
    from image_analysis.algorithms.basic_line_stats import LineBasicStats

    trace = np.column_stack((coordinates, [1, 2, 4, 8, 12, 9, 4, 2, 1]))
    expected = LineBasicStats(line_data=trace.copy()).to_dict()
    result = analyze(Frame.from_trace(trace, x_unit="MeV"), recipe("line"))
    assert_finite_scalars_equal(result.scalars, expected)
    assert not result.notes


def test_line_widths_on_an_uneven_axis_are_moments_over_x():
    """#1029: not index-space widths times the one spacing at the centroid."""
    from image_analysis.algorithms.basic_line_stats import LineBasicStats

    x = np.array([2, 3, 6, 7, 10, 12, 15, 19, 20], dtype=float)
    y = np.array([1, 2, 4, 8, 12, 9, 4, 2, 1], dtype=float)
    legacy = LineBasicStats(line_data=np.column_stack((x, y))).to_dict()
    result = analyze(
        Frame.from_trace(np.column_stack((x, y)), x_unit="MeV"), recipe("line")
    )

    # rms is the Δx-weighted moment over x — the trapezoid integral, with
    # weights by hand: each sample owns half the interval to each neighbour
    # (the one at 6 owns (6−3)/2 + (7−6)/2 = 2), the ends half of their one.
    w = np.array([0.5, 2, 2, 2, 2.5, 2.5, 3.5, 2.5, 0.5])
    centroid = np.sum(w * y * x) / np.sum(w * y)
    assert result.scalars["rms"] == pytest.approx(
        np.sqrt(np.sum(w * y * (x - centroid) ** 2) / np.sum(w * y)), rel=1e-12
    )
    # Not a per-sample mean over x (4.02), which would weigh a densely
    # sampled stretch more than a sparse one.
    assert result.scalars["rms"] == pytest.approx(3.6882, abs=1e-4)
    # Baseline-shifted half maximum 5.5 is crossed between samples 3→7 at
    # x 6→7 (6.625 MeV) and 8→3 at x 12→15 (13.5 MeV).
    assert result.scalars["fwhm"] == pytest.approx(13.5 - 6.625, rel=1e-12)
    # Legacy scaled by the 2 MeV spacing at the centroid: 2.875 × 2 = 5.75.
    assert legacy["fwhm"] == 5.75
    assert legacy["rms"] == pytest.approx(3.2987, abs=1e-4)
    for key in ("CoM", "peak_location", "integrated_intensity", "peak_value"):
        assert result.scalars[key] == legacy[key]
    assert not result.notes


def test_line_widths_on_a_descending_uneven_axis_keep_the_sign_convention():
    x = np.array([2, 3, 6, 7, 10, 12, 15, 19, 20], dtype=float)
    y = np.array([1, 2, 4, 8, 12, 9, 4, 2, 1], dtype=float)
    ascending = analyze(Frame.from_trace(np.column_stack((x, y))), recipe("line"))
    descending = analyze(
        Frame.from_trace(np.column_stack((x[::-1], y[::-1]))), recipe("line")
    )
    # Legacy's local dx made widths negative on a descending axis; the
    # coordinate path keeps that one convention rather than mixing two.
    for width in ("rms", "fwhm"):
        assert descending.scalars[width] == pytest.approx(-ascending.scalars[width])
    assert descending.scalars["CoM"] == pytest.approx(ascending.scalars["CoM"])


def stitched_repro(samples=(260, 200, 180), gains=(1.0, 0.92, 1.08), noise=0.015):
    """The #1029 repro: three cameras' segments joined as the scan host joins siblings."""
    rng = np.random.default_rng(7)

    def spectrum(e):
        return np.exp(-0.5 * ((e - 128) / 22) ** 2) + 0.25 * np.exp(-(e - 40) / 25)

    segments = []
    for (lo, hi), n, gain in zip([(40, 105), (100, 170), (165, 260)], samples, gains):
        e = np.linspace(lo, hi, n)
        y = gain * spectrum(e) + (rng.normal(0, noise, n) if noise else 0.0)
        segments.append(np.column_stack([e, y]))
    joined = np.concatenate(segments)
    # scan_analysis.core_source's join: concatenate, then sort by x.
    return joined[joined[:, 0].argsort()]


def interval_weights(x):
    """Trapezoid weights: half the interval to each neighbour, the ends half of one."""
    return np.concatenate(
        [[(x[1] - x[0]) / 2], (x[2:] - x[:-2]) / 2, [(x[-1] - x[-2]) / 2]]
    )


def test_line_widths_on_a_stitched_trace_are_measured_in_x():
    from image_analysis.algorithms.basic_line_stats import LineBasicStats

    joined = stitched_repro()
    x, y = joined[:, 0], joined[:, 1]
    frame = Frame.from_array(y, axes=(Axis(values=x, unit="MeV"),))
    result = analyze(frame, recipe("line"))
    legacy = LineBasicStats(line_data=joined.copy()).to_dict()

    # Measured directly in MeV on the joined trace: the span of the samples
    # at or above half maximum brackets the interpolated crossings to within
    # one sample spacing (≤ 0.35 MeV). The measure clips negatives first
    # (legacy), so its baseline is the clipped minimum.
    clipped = np.clip(y, 0, None)
    shifted = clipped - clipped.min()
    above = x[shifted >= shifted.max() / 2]
    direct_fwhm = above.max() - above.min()
    assert result.scalars["fwhm"] == pytest.approx(direct_fwhm, abs=0.5)
    true_fwhm = 2 * np.sqrt(2 * np.log(2)) * 22  # 51.8 MeV
    assert result.scalars["fwhm"] == pytest.approx(true_fwhm, abs=2.0)
    # The old value (index-space width × the spacing at the centroid) read
    # ~60 MeV here, 13 % wide.
    assert legacy["fwhm"] > true_fwhm + 5
    assert abs(result.scalars["fwhm"] - legacy["fwhm"]) > 5

    # rms: the Δx-weighted second moment over x — the trapezoid integral,
    # with the legacy normalisation (clipped weights over the unclipped
    # total, here the unclipped trace's integral).
    w = interval_weights(x)
    clipped = np.clip(y, 0, None)
    centroid = np.sum(w * clipped * x) / np.sum(w * clipped)
    expected_rms = np.sqrt(np.sum(w * clipped * (x - centroid) ** 2) / np.sum(w * y))
    assert result.scalars["rms"] == pytest.approx(expected_rms, rel=1e-9)
    assert abs(result.scalars["rms"] - legacy["rms"]) > 5

    for key in ("CoM", "peak_location", "integrated_intensity", "peak_value"):
        assert result.scalars[key] == legacy[key]
    assert not result.notes


@pytest.mark.parametrize("samples", [(260, 200, 720), (1040, 200, 180)])
def test_line_rms_on_a_stitched_trace_does_not_depend_on_sampling_density(samples):
    """#1044 review: a per-sample mean over x read 8 % apart between these samplings."""

    def widths(samples):
        joined = stitched_repro(samples, gains=(1.0, 1.0, 1.0), noise=0.0)
        frame = Frame.from_array(joined[:, 1], axes=(Axis(joined[:, 0], unit="MeV"),))
        return analyze(frame, recipe("line")).scalars

    reference, other = widths((260, 200, 180)), widths(samples)
    # The same analytic spectrum, one camera sampled four times as densely:
    # the Δx-weighted moment is the integral over x either way.
    assert other["rms"] == pytest.approx(reference["rms"], rel=1e-4)
    assert reference["rms"] == pytest.approx(29.34, abs=0.01)
    assert other["fwhm"] == pytest.approx(reference["fwhm"], abs=0.05)


def test_even_spacing_is_judged_against_the_step():
    """#1044 review: an offset must not hide a gap or jitter; single precision must pass."""
    from geecs_analysis.algorithms.basic_line_stats import is_evenly_spaced

    assert is_evenly_spaced(np.linspace(0.1, 0.9, 9).astype(np.float32).astype(float))
    assert is_evenly_spaced(
        np.linspace(1000, 5096, 4096).astype(np.float32).astype(float)
    )
    assert is_evenly_spaced(np.arange(9)[::-1])
    # A Unix-time axis with a gap: a tolerance of max|x| was 1760 s wide.
    assert not is_evenly_spaced(1.76e9 + np.array([0, 1, 2, 3, 4, 5, 6, 7, 1000.0]))
    # Jitter of 30 % of the step at an offset of 400 000 steps.
    i = np.arange(50)
    jitter = np.random.default_rng(1).uniform(-0.3, 0.3, i.size)
    assert not is_evenly_spaced(799 + 0.002 * (i + jitter))
    assert not is_evenly_spaced(stitched_repro()[:, 0])


def test_beam_widths_on_an_uneven_camera_axis_follow_the_line_measure():
    from image_analysis.algorithms.basic_beam_stats import (
        beam_profile_stats,
        flatten_beam_stats,
    )

    samples = np.random.default_rng(9).uniform(1, 10, (9, 11))
    x = np.array([0, 1, 3, 4, 6, 9, 10, 13, 14, 18, 20], dtype=float)
    frame = Frame.from_array(
        samples, axes=(Axis(np.arange(9) + 4, label="y"), Axis(x, "MeV", "x"))
    )
    beam = analyze(frame, recipe("beam")).scalars
    projection = analyze(
        Frame.from_trace(np.column_stack((x, samples.sum(axis=0))), x_unit="MeV"),
        recipe("line"),
    ).scalars
    for key in ("CoM", "rms", "fwhm", "peak_location"):
        assert beam[f"x_{key}"] == projection[key]
    # And those widths are measured over the uneven axis: x_rms is the
    # Δx-weighted moment of the projection (weights by hand: half of each
    # neighbouring interval), not a per-sample mean (6.47) and not the
    # legacy index width times the one spacing at the centroid.
    w = np.array([0.5, 1.5, 1.5, 1.5, 2.5, 2, 2, 2, 2.5, 3, 1])
    p = samples.sum(axis=0)
    centroid = np.sum(w * p * x) / np.sum(w * p)
    assert beam["x_rms"] == pytest.approx(
        np.sqrt(np.sum(w * p * (x - centroid) ** 2) / np.sum(w * p)), rel=1e-12
    )
    assert beam["x_rms"] == pytest.approx(5.8563, abs=1e-4)
    # The evenly spaced y axis, the diagonals and the image totals are legacy.
    legacy = flatten_beam_stats(beam_profile_stats(samples, roi_offset=(0, 4)))
    for key, value in legacy.items():
        if key.startswith(("y_", "x_45", "y_45", "image")):
            assert beam[key] == value


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
for kind, count in (
    ("beam", 18), ("hi_res_mag_cam", 24), ("line", 6), ("none", 0), ("pulsed_wire", 10)
):
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
