"""The ``hi_res_mag_cam`` measure: the legacy fit bit for bit, the v2 route, rejected fits made visible."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from geecs_schemas.analysis import (
    AnalysisRecipe,
)

from geecs_analysis.algorithms.bowtie_fit import BowtieFitAlgorithm
from geecs_analysis.engine.v2 import V2Recipe, analyze_v2
from geecs_analysis.measures.hi_res_mag_cam import (
    REJECTED,
    HiResMagCamSpec,
)
from geecs_analysis.recipe import compile_recipe
from geecs_analysis.render import single

_spec = importlib.util.spec_from_file_location(
    "synthetic_bowtie", Path(__file__).with_name("synthetic_bowtie.py")
)
_bowtie = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bowtie)

FIT_FIELDS = ("score", "w0", "theta", "x0", "r_squared")


def bowtie(seed: int, **kwargs) -> np.ndarray:
    """A synthetic bow-tie frame from the legacy generator (``synthetic_bowtie.py``)."""
    return _bowtie.generate_bowtie_image(
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


def recipe(*steps: dict, **measure) -> V2Recipe:
    return compile_recipe(
        AnalysisRecipe.model_validate(
            {
                "schema_version": 3,
                "device": "UC_HiResMagCam",
                "input": {"kind": "camera"},
                "steps": list(steps),
                "measure": {"kind": "hi_res_mag_cam", **measure},
            }
        )
    )


# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------


def fit_values(algorithm, image: np.ndarray) -> dict:
    """What a bowtie fit reports, flattened for the stored-value check."""
    result = algorithm.evaluate(image)
    values = {name: float(getattr(result, name)) for name in FIT_FIELDS}
    values["param_errors"] = np.asarray(result.param_errors, dtype=float)
    values["weights"] = np.asarray(result.weights)
    values["sizes"] = np.asarray(result.sizes, dtype=float)
    for n, profile in enumerate(algorithm.get_last_profile()):
        values[f"profile_{n}"] = np.asarray(profile)
    return values


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("clearance,min_counts", [(4, 2500.0), (2, 500.0)])
def test_the_algorithm_is_the_legacy_one_bit_for_bit(
    seed, case, clearance, min_counts, expected, request
):
    image = frame(case, seed).astype(float)
    image[image < 10] = 0
    args = dict(n_beam_size_clearance=clearance, min_total_counts=min_counts)
    expected(
        f"algorithm_{request.node.callspec.id}",
        actual=fit_values(BowtieFitAlgorithm(**args), image),
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
# The measure through the core route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "steps",
    [
        [],
        [{"step": "roi", "bounds": [[4, 60], [10, 120]]}],
        [{"step": "background_constant", "level": 20}, {"step": "median"}],
        # The waist (column 64) lies 4 columns PAST the crop on either side:
        # the fit still accepts it, and legacy reports it beyond the edge.
        [{"step": "roi", "bounds": [[0, 64], [0, 60]]}],
        [{"step": "roi", "bounds": [[0, 64], [68, 128]]}],
    ],
    ids=[
        "plain",
        "roi",
        "background+median",
        "waist past right edge",
        "waist past left edge",
    ],
)
def test_the_core_route_matches_the_legacy_analyzer(seed, steps, expected, request):
    """Same frame and config through HiResMagCamAnalyzer and through the core."""

    compiled = recipe(*steps, n_beam_size_clearance=3, min_total_counts=1500)
    raw = bowtie(seed, energy_center=70, vertical_offset=3)

    core = analyze_v2(raw, compiled)
    weights = {o.id: o for o in core.overlays}["bowtie_weights"]
    stored = expected(
        f"route_{request.node.callspec.id}",
        actual={
            **{f"scalar_{k}": v for k, v in core.scalars.items()},
            "bowtie_weights": weights.frame.data,
        },
    )
    assert set(core.scalars) == HiResMagCamSpec().emitted_scalars()
    assert float(stored["scalar_emittance_proxy"]) != REJECTED
    assert core.notes == ()
    assert math.isfinite(core.scalars["bowtie_x0"])
    np.testing.assert_array_equal(
        weights.frame.axes[0].values, core.frame.axes[1].values
    )


@pytest.mark.parametrize(
    "columns,side",
    [
        ((0, 60), "right"),
        ((68, 128), "left"),
    ],
)
def test_a_waist_past_the_crop_reads_past_the_edge_not_clamped_to_it(columns, side):
    """The fit accepts a waist within ten columns of data; x0 must say where it is."""
    roi = {"step": "roi", "bounds": [[0, 64], list(columns)]}
    core = analyze_v2(bowtie(1), recipe(roi))
    assert core.scalars["emittance_proxy"] != REJECTED
    x0 = core.scalars["bowtie_x0"]
    assert x0 == pytest.approx(64, abs=4)
    # A clamped value would sit exactly on the last (first) column.
    assert x0 > columns[1] - 1 if side == "right" else x0 < columns[0]


def test_the_waist_column_is_in_sensor_pixels_like_the_centroid():
    raw = bowtie(1, energy_center=70)
    cropped = analyze_v2(raw, recipe({"step": "roi", "bounds": [[0, 64], [20, 128]]}))
    local = analyze_v2(raw[:, 20:], recipe())
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
    core = analyze_v2(raw, recipe())
    assert core.scalars["emittance_proxy"] == REJECTED
    for key in ("bowtie_x0", "bowtie_w0", "bowtie_theta", "bowtie_r_squared"):
        assert math.isnan(core.scalars[key]), key
    assert any(note.startswith("Bow-tie fit rejected") for note in core.notes)
    assert math.isfinite(core.scalars["total_counts"])


def test_the_measure_never_mutates_the_frame_it_floors():
    """The fit sees the frame floored at 10 counts; the measured frame does not."""
    raw = bowtie(3)
    core = analyze_v2(raw, recipe())
    assert (core.frame.data > 0).sum() > (core.frame.data >= 10).sum()
    assert core.scalars["total_counts"] == core.frame.data[core.frame.data >= 10].sum()
    assert core.scalars["image_total"] == core.frame.data.sum()


def test_overlays_render_the_fit_weights_as_a_lineout():
    core = analyze_v2(bowtie(4), recipe())
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
