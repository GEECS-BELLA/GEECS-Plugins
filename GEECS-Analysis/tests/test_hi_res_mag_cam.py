"""The ``hi_res_mag_cam`` measure: the legacy fit bit for bit, the v2 route, rejected fits made visible."""

from __future__ import annotations

import math

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisDiagnostic, HiResMagCamSpec as V2Spec

from geecs_analysis.algorithms.bowtie_fit import BowtieFitAlgorithm
from geecs_analysis.compat.convert import to_v3
from geecs_analysis.compat.v2 import analyze_v2, compile_v2
from geecs_analysis.measures.beam import BeamSpec
from geecs_analysis.measures.hi_res_mag_cam import (
    BOWTIE_SCALARS,
    REJECTED,
    HiResMagCamSpec,
)
from geecs_analysis.recipe import compile_recipe
from geecs_analysis.render import single

FIT_FIELDS = ("score", "w0", "theta", "x0", "r_squared")


def bowtie(seed: int, **kwargs) -> np.ndarray:
    """A synthetic bow-tie frame from the legacy generator (the tests' oracle input)."""
    from image_analysis.tools.synthetic_generators import generate_bowtie_image

    return generate_bowtie_image(
        **{
            "shape": (64, 128),
            "noise_level": 10.0,
            "background_level": 0,
            "seed": seed,
            **kwargs,
        }
    )


def frame(case: str, seed: int) -> np.ndarray:
    """Named inputs covering every fit outcome; the generator itself only makes accepted fits."""
    if case == "bowtie":
        return bowtie(seed)
    if case == "off-centre":
        return bowtie(
            seed, min_sigma=3.0, divergence=0.1, energy_center=40, vertical_offset=-8
        )
    if case == "left arm":
        # Only the converging arm: the fit puts the waist past the data,
        # where no fit column carries weight -> the waist-weight rejection.
        image = bowtie(seed, energy_center=30, energy_spread=15.0, divergence=0.15)
        image[:, 45:] = 0
        return image
    if case == "waist blanked":
        # Fewer than five valid columns survive -> the few-columns rejection.
        image = bowtie(seed)
        image[:, 55:73] = 0
        return image
    raise KeyError(case)


CASES = ("bowtie", "off-centre", "left arm", "waist blanked")


def diagnostic(analyzer: dict | None = None, **image) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": "UC_HiResMagCam",
            "analyzer": {"kind": "hi_res_mag_cam", **(analyzer or {})},
            "image": {"type": "camera", "pipeline": [], **image},
        }
    )


def assert_same_value(actual: float, expected: float, key: str) -> None:
    """Equal wherever legacy has a value (finite or infinite); NaN only where legacy is NaN."""
    if math.isnan(expected):
        assert math.isnan(actual), key
    else:
        assert actual == expected, key


# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("clearance,min_counts", [(4, 2500.0), (2, 500.0)])
def test_the_algorithm_is_the_legacy_one_bit_for_bit(seed, case, clearance, min_counts):
    from image_analysis.algorithms.bowtie_fit import BowtieFitAlgorithm as Legacy

    image = frame(case, seed).astype(float)
    image[image < 10] = 0
    args = dict(n_beam_size_clearance=clearance, min_total_counts=min_counts)
    ours, theirs = BowtieFitAlgorithm(**args), Legacy(**args)
    result, expected = ours.evaluate(image), theirs.evaluate(image)
    for name in FIT_FIELDS:
        assert_same_value(getattr(result, name), getattr(expected, name), name)
    for got, want in zip(result.param_errors, expected.param_errors, strict=True):
        assert_same_value(got, want, "param_errors")
    np.testing.assert_array_equal(result.weights, expected.weights)
    assert np.array_equal(result.sizes, expected.sizes, equal_nan=True)
    assert all(
        np.array_equal(a, b, equal_nan=True)
        for a, b in zip(ours.get_last_profile(), theirs.get_last_profile(), strict=True)
    )


def test_the_differential_cases_cover_an_accepted_fit_and_both_rejections():
    """The comparison above is only evidence if it saw every outcome."""
    outcomes = set()
    for case in CASES:
        image = frame(case, 2).astype(float)
        image[image < 10] = 0
        fit = BowtieFitAlgorithm(n_beam_size_clearance=4, min_total_counts=2500.0)
        result = fit.evaluate(image)
        if result.score == REJECTED:
            outcomes.add("few columns" if math.isnan(result.x0) else "waist weight")
        elif math.isfinite(result.score):
            outcomes.add("accepted")
            assert 40 < result.x0 < 90 and result.w0 > 0 and result.theta > 0
    assert outcomes == {"accepted", "few columns", "waist weight"}


# ---------------------------------------------------------------------------
# The measure through the v2 route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "image",
    [
        {},
        {
            "pipeline": ["roi"],
            "roi": {"x_min": 10, "x_max": 120, "y_min": 4, "y_max": 60},
        },
        {
            "pipeline": ["background", "filtering"],
            "background": {"method": "constant", "constant_level": 20},
            "filtering": {"median_kernel_size": 3},
        },
        # The waist (column 64) lies 4 columns PAST the crop on either side:
        # the fit still accepts it, and legacy reports it beyond the edge.
        {
            "pipeline": ["roi"],
            "roi": {"x_min": 0, "x_max": 60, "y_min": 0, "y_max": 64},
        },
        {
            "pipeline": ["roi"],
            "roi": {"x_min": 68, "x_max": 128, "y_min": 0, "y_max": 64},
        },
    ],
    ids=[
        "plain",
        "roi",
        "background+median",
        "waist past right edge",
        "waist past left edge",
    ],
)
def test_the_v2_route_matches_the_legacy_analyzer(seed, image):
    """Same frame and config through HiResMagCamAnalyzer and through the core."""
    from image_analysis.analyzers.Undulator.hi_res_mag_cam_analyzer import (
        HiResMagCamAnalyzer,
    )

    document = diagnostic(
        {"n_beam_size_clearance": 3, "min_total_counts": 1500}, **image
    )
    raw = bowtie(seed, energy_center=70, vertical_offset=3)
    legacy = HiResMagCamAnalyzer(document.image, spec=document.analyzer).analyze_image(
        raw
    )
    core = analyze_v2(raw, compile_v2(document))
    assert set(core.scalars) == set(legacy.scalars) == V2Spec().emitted_scalars()
    assert set(core.scalars) == HiResMagCamSpec().emitted_scalars()
    assert legacy.scalars["emittance_proxy"] != REJECTED
    for key, value in legacy.scalars.items():
        assert_same_value(core.scalars[key], float(value), key)
    assert core.notes == ()
    assert math.isfinite(core.scalars["bowtie_x0"])
    weights = {o.id: o for o in core.overlays}["bowtie_weights"]
    np.testing.assert_array_equal(
        weights.frame.data, legacy.render_data["bowtie_weights"]
    )
    np.testing.assert_array_equal(
        weights.frame.axes[0].values, core.frame.axes[1].values
    )


@pytest.mark.parametrize(
    "roi,side",
    [
        ({"x_min": 0, "x_max": 60, "y_min": 0, "y_max": 64}, "right"),
        ({"x_min": 68, "x_max": 128, "y_min": 0, "y_max": 64}, "left"),
    ],
)
def test_a_waist_past_the_crop_reads_past_the_edge_not_clamped_to_it(roi, side):
    """The fit accepts a waist within ten columns of data; x0 must say where it is."""
    core = analyze_v2(bowtie(1), compile_v2(diagnostic(pipeline=["roi"], roi=roi)))
    assert core.scalars["emittance_proxy"] != REJECTED
    x0 = core.scalars["bowtie_x0"]
    assert x0 == pytest.approx(64, abs=4)
    # A clamped value would sit exactly on the last (first) column.
    assert x0 > roi["x_max"] - 1 if side == "right" else x0 < roi["x_min"]


def test_the_waist_column_is_in_sensor_pixels_like_the_centroid():
    document = diagnostic(
        pipeline=["roi"], roi={"x_min": 20, "x_max": 128, "y_min": 0, "y_max": 64}
    )
    raw = bowtie(1, energy_center=70)
    cropped = analyze_v2(raw, compile_v2(document))
    local = analyze_v2(raw[:, 20:], compile_v2(diagnostic()))
    assert cropped.scalars["bowtie_x0"] == pytest.approx(
        local.scalars["bowtie_x0"] + 20
    )
    assert cropped.scalars["x_CoM"] == pytest.approx(local.scalars["x_CoM"] + 20)


@pytest.mark.parametrize(
    "raw,reason",
    [
        (np.zeros((64, 128), dtype=np.uint16), "few columns"),
        (frame("waist blanked", 2), "few columns"),
        (frame("left arm", 2), "waist weight"),
    ],
)
def test_a_rejected_fit_keeps_the_sentinel_and_blanks_the_fit_parameters(raw, reason):
    core = analyze_v2(raw, compile_v2(diagnostic()))
    assert core.scalars["emittance_proxy"] == REJECTED
    for key in ("bowtie_x0", "bowtie_w0", "bowtie_theta", "bowtie_r_squared"):
        assert math.isnan(core.scalars[key]), key
    assert any(note.startswith("Bow-tie fit rejected") for note in core.notes)
    assert math.isfinite(core.scalars["total_counts"])


def test_the_measure_never_mutates_the_frame_it_floors():
    """The fit sees the frame floored at 10 counts; the measured frame does not."""
    document = diagnostic(
        pipeline=["background"],
        background={"method": "constant", "constant_level": 0},
    )
    raw = bowtie(3)
    core = analyze_v2(raw, compile_v2(document))
    assert (core.frame.data > 0).sum() > (core.frame.data >= 10).sum()
    assert core.scalars["total_counts"] == core.frame.data[core.frame.data >= 10].sum()
    assert core.scalars["image_total"] == core.frame.data.sum()


def test_overlays_render_the_fit_weights_as_a_lineout():
    core = analyze_v2(bowtie(4), compile_v2(diagnostic()))
    assert {o.id for o in core.overlays} == {
        "projection_x",
        "projection_y",
        "bowtie_weights",
        "com",
    }
    figure = single(core)
    assert figure.axes
    import matplotlib.pyplot as plt

    plt.close(figure)


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------


def test_the_spec_mirrors_the_v2_spec_except_the_unused_threshold_factor():
    v2, core = V2Spec.model_fields, HiResMagCamSpec.model_fields
    assert set(v2) - {"kind"} - set(core) == {"threshold_factor"}
    assert set(core) - {"kind"} <= set(v2)
    for name in set(core) - {"kind"}:
        assert core[name].default == v2[name].default, name
        assert core[name].metadata == v2[name].metadata, name
    assert HiResMagCamSpec().emitted_scalars() == BeamSpec().emitted_scalars() | set(
        BOWTIE_SCALARS
    )


def test_a_diagnostic_converts_to_a_recipe_that_compiles_identically():
    document = diagnostic({"n_beam_size_clearance": 5})
    conversion = to_v3(document)
    assert conversion.recipe.measure.model_dump()["kind"] == "hi_res_mag_cam"
    assert compile_recipe(conversion.recipe) == compile_v2(document)
    assert not any("threshold_factor" in note for note in conversion.notes)
    spoken = to_v3(diagnostic({"threshold_factor": 5.0}))
    assert any(
        note.startswith("analyzer.threshold_factor dropped") for note in spoken.notes
    )
    assert compile_recipe(spoken.recipe) == compile_v2(diagnostic())
