"""The ``downramp_phase`` measure: the legacy shock analysis bit for bit, its PDF as overlays."""

from __future__ import annotations

import math
import warnings
from functools import partial

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisDiagnostic, DownrampPhaseSpec as V2Spec
from scipy import ndimage

from geecs_analysis.algorithms import downramp_shock as ported
from geecs_analysis.compat.convert import to_v3
from geecs_analysis.compat.v2 import analyze_v2, compile_v2
from geecs_analysis.measurement import Measurement, Segment
from geecs_analysis.measures.downramp_phase import (
    OVERLAYS,
    PLATEAU_AVG,
    SHOCK_ANGLE,
    SHOCK_LOCATION,
    SHOCK_SCALARS,
    SLOPE_SCALARS,
    DownrampPhaseSpec,
)
from geecs_analysis.recipe import compile_recipe
from geecs_analysis.render import single
from geecs_analysis.run import analyze
from geecs_analysis.specs import Analysis
from geecs_data_utils.frames import Frame

SHOCK_COLUMN = 60
CASES = ("shock", "angled", "tilted", "ramp", "plateau", "noise")


def phase_map(case: str, seed: int, shape: tuple[int, int] = (64, 96)) -> np.ndarray:
    """A phase map in radians: a plasma channel (a phase dip) with or without a shock.

    ``shock``: the channel's phase steps at ``SHOCK_COLUMN``; ``angled``:
    that step tilted 0.3 columns per row; ``tilted``: the channel itself
    tilted 0.1 rows per column (a laser axis the analysis must rotate onto);
    ``ramp``, ``plateau`` and ``noise`` have no shock.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]
    axis = H / 2 + (0.1 if case == "tilted" else 0.0) * (x - W / 2)
    edge = SHOCK_COLUMN + (0.3 if case == "angled" else 0.0) * (y - H / 2)
    along = {
        "ramp": 1.0 + 0.02 * x,
        "plateau": np.full_like(x, 1.5, dtype=float),
        "noise": np.zeros_like(x, dtype=float),
    }.get(case, 1.0 + 1.5 / (1 + np.exp(-(x - edge) / 2.0)))
    dip = along * np.exp(-0.5 * ((y - axis) / 8) ** 2)
    return 3.0 - dip + 0.02 * rng.normal(size=(H, W))


def document(**image) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": "U_HasoLift",
            "analyzer": {"kind": "downramp_phase"},
            "image": {"type": "camera", "pipeline": [], **image},
        }
    )


def legacy(doc: AnalysisDiagnostic, image: np.ndarray, analyzer_class=None):
    from image_analysis.analyzers.downramp_phase_analyzer import DownrampPhaseAnalyzer

    cls = analyzer_class or DownrampPhaseAnalyzer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the oracle's mean-of-empty / fit warnings
        return cls(camera_config=doc.image).analyze_image(image.copy())


def assert_equal_scalars(result: Measurement, expected: dict) -> None:
    """Every legacy key, equal — and the oracle finite, so matched NaNs never pass."""
    assert all(math.isfinite(float(v)) for v in expected.values()), "oracle undefined"
    assert set(result.scalars) == set(expected)
    for key, value in expected.items():
        assert result.scalars[key] == float(value), key


def overlay(result: Measurement, name: str):
    return {o.id: o for o in result.overlays}[name]


def distance_to_segment(segment: Segment, x: float, y: float) -> float:
    """Distance from a point to the segment's line."""
    dx, dy = segment.x1 - segment.x0, segment.y1 - segment.y0
    return abs(dy * (x - segment.x0) - dx * (y - segment.y0)) / math.hypot(dx, dy)


# ----------------------------------------------------------- the v2 route


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("case", CASES)
def test_the_v2_route_equals_the_legacy_analyzer(case, seed):
    doc = document()
    image = phase_map(case, seed)
    old = legacy(doc, image)
    new = analyze_v2(image, compile_v2(doc))
    assert_equal_scalars(new, old.scalars)
    assert new.notes == ()
    # The frame is the legacy 16-bit phase product, stored as float64.
    assert old.processed_image.dtype == np.uint16
    np.testing.assert_array_equal(new.frame.data, old.processed_image)
    for name, key in (("projection_x", "horizontal"), ("projection_y", "vertical")):
        np.testing.assert_array_equal(
            overlay(new, name).frame.data, old.render_data[f"{key}_projection"]
        )
    assert {o.id for o in new.overlays} == {"projection_x", "projection_y", *OVERLAYS}


@pytest.mark.parametrize(
    "image",
    [
        {
            "pipeline": ["roi"],
            "roi": {"x_min": 8, "x_max": 90, "y_min": 10, "y_max": 58},
        },
        {
            "pipeline": ["background", "filtering"],
            "background": {"method": "constant", "constant_level": 0.5},
            "filtering": {"median_kernel_size": 3},
        },
    ],
    ids=["roi", "background+median"],
)
def test_the_recipe_steps_precede_the_shock_analysis_as_the_pipeline_did(image):
    doc = document(**image)
    data = phase_map("angled", 3)
    old = legacy(doc, data)
    new = analyze_v2(data, compile_v2(doc))
    assert_equal_scalars(new, old.scalars)
    np.testing.assert_array_equal(new.frame.data, old.processed_image)


def test_the_measured_shock_is_where_it_was_planted():
    plain = analyze_v2(phase_map("shock", 0), compile_v2(document()))
    assert plain.scalars[SHOCK_LOCATION] == pytest.approx(SHOCK_COLUMN, abs=3)
    assert plain.scalars[SHOCK_ANGLE] == pytest.approx(0.0, abs=0.05)
    angled = analyze_v2(phase_map("angled", 0), compile_v2(document()))
    assert angled.scalars[SHOCK_ANGLE] == pytest.approx(math.atan(0.3), abs=0.05)
    assert plain.scalars[PLATEAU_AVG] > 0


# ------------------------------------------------------------ the options


def test_options_reach_the_legacy_functions_at_the_same_values(monkeypatch):
    """The spec's knobs are the legacy constants; at other values both agree too."""
    import image_analysis.analyzers.downramp_phase_analyzer as oracle

    options = dict(window_size=12, sigma_x=30.0, sigma_y=10.0, threshold_factor=0.05)
    monkeypatch.setattr(
        oracle,
        "apply_gaussian_mask",
        partial(
            oracle.apply_gaussian_mask,
            sigma_x=30.0,
            sigma_y=10.0,
            threshold_factor=0.05,
        ),
    )

    class Legacy(oracle.DownrampPhaseAnalyzer):
        # The legacy composite passed its window to the gradient and plateau
        # steps but not to the angle fit (always 20); the spec's one window
        # reaches all three, so the oracle gets it there too.
        def compile_shock_analysis(self, phase_array, window_size=12):
            return super().compile_shock_analysis(phase_array, window_size=12)

        def get_shock_angle(self, image, ax, window_size=12):
            return super().get_shock_angle(image, ax, window_size=12)

    image = phase_map("angled", 5)
    old = legacy(document(), image, Legacy)
    new = analyze(
        Frame.from_array(image),
        Analysis(steps=(), measure=DownrampPhaseSpec(**options)),
    )
    assert_equal_scalars(new, old.scalars)
    np.testing.assert_array_equal(new.frame.data, old.processed_image)
    defaults = analyze_v2(image, compile_v2(document()))
    assert defaults.scalars[SHOCK_LOCATION] != new.scalars[SHOCK_LOCATION]


def test_the_mask_and_the_window_search_are_the_legacy_functions_bit_for_bit():
    import image_analysis.analyzers.downramp_phase_analyzer as oracle
    from matplotlib.figure import Figure

    image = phase_map("tilted", 2)
    for kwargs in ({}, {"sigma_x": 25, "sigma_y": 6, "threshold_factor": 0.1}):
        np.testing.assert_array_equal(
            ported.apply_gaussian_mask(image, binarize=True, **kwargs),
            oracle.apply_gaussian_mask(image, binarize=True, **kwargs),
        )
    masked = ported.apply_gaussian_mask(image, binarize=True)
    analyzer = oracle.DownrampPhaseAnalyzer(camera_config=document().image)
    for window in (20, 7):
        for profile in (masked.sum(axis=0), masked[20], -masked[20]):
            theirs = analyzer.find_max_local_slope(profile, window)
            ours = ported.find_max_local_slope(profile, window)
            assert (ours.slope, ours.center) == (theirs[0], theirs[1]) or (
                math.isnan(ours.slope) and math.isnan(theirs[0]) and theirs[1] is None
            )
        ax = Figure().subplots()
        assert ported.shock_angle(masked, window).angle_rad == analyzer.get_shock_angle(
            masked, ax, window
        )


# ----------------------------------------------------------- degenerate


def test_a_flat_map_is_nan_with_a_note_where_the_legacy_class_fails():
    flat = np.full((40, 60), 2.5)
    with pytest.raises(ValueError):
        legacy(document(), flat)
    result = analyze_v2(flat, compile_v2(document()))
    assert all(math.isnan(result.scalars[key]) for key in SHOCK_SCALARS + SLOPE_SCALARS)
    assert result.notes[0].startswith("No signal after masking")
    assert np.all(result.frame.data == 0)
    assert {o.id for o in result.overlays} == {"projection_x", "projection_y"}
    assert dict(result.extras) == {}


def test_a_profile_with_no_steepest_window_leaves_the_plateau_undefined():
    """The legacy plateau step failed on ``None``; the measure says why instead."""
    from geecs_analysis.algorithms.downramp_shock import LocalSlope, ShockAnalysis
    from geecs_analysis.measures.downramp_phase import _scalars

    shock = ShockAnalysis(
        slopes=dict.fromkeys(SLOPE_SCALARS, 0.0),
        laser_angle_deg=0.0,
        rotated=np.zeros((4, 4)),
        angle=ported.ShockAngle(math.nan, math.nan, math.nan, None, None),
        gradient=LocalSlope(math.nan, None),
        plateau=None,
        profile=np.zeros(4),
    )
    scalars, notes = _scalars(shock)
    assert all(math.isnan(scalars[key]) for key in SHOCK_SCALARS)
    assert notes == (
        "No steepest window in the summed phase: shock slope, location and plateau undefined",
    )


# ------------------------------------------------------------- overlays


@pytest.mark.parametrize("angle", [11.3, -37.0, 95.0])
def test_unrotate_inverts_the_rotation_of_the_laser_axis(angle):
    """Linear interpolation reproduces a plane exactly, so the inverse map is pinned exactly."""
    rows, cols = np.mgrid[0:40, 0:70]
    plane = 0.7 * rows - 0.3 * cols + 5.0
    rotated = ndimage.rotate(plane, angle=angle, reshape=True, order=1)
    canvas = np.argwhere(np.ones(rotated.shape, dtype=bool))
    back = ported.unrotate(canvas, angle, plane.shape, rotated.shape)
    # Only canvas points whose sample lies inside the input (no padding).
    inside = (
        (back[:, 0] >= 0) & (back[:, 0] <= 39) & (back[:, 1] >= 0) & (back[:, 1] <= 69)
    )
    assert inside.sum() > 1000
    expected = 0.7 * back[inside, 0] - 0.3 * back[inside, 1] + 5.0
    np.testing.assert_allclose(rotated[tuple(canvas[inside].T)], expected, atol=1e-9)


def test_the_shock_overlays_sit_on_the_frame_where_the_shock_is():
    """The rotated-canvas geometry lands back on the planted shock, tilted channel included."""
    axis_row = 32  # the channel crosses the map's centre
    for case in ("shock", "angled", "tilted"):
        result = analyze_v2(phase_map(case, 1), compile_v2(document()))
        location = overlay(result, "shock_location")
        assert distance_to_segment(location, SHOCK_COLUMN, axis_row) < 3.0, case
        start, end = overlay(result, "plateau_start"), overlay(result, "plateau_end")
        assert start.x0 < end.x0 < location.x0, case
        if case != "tilted":
            # The legacy angle fit on a tilted channel follows the rotated
            # canvas's edges, not the shock; its line is drawn as fitted.
            fit = overlay(result, "shock_fit")
            assert distance_to_segment(fit, SHOCK_COLUMN, axis_row) < 4.0, case
        assert set(result.extras) == {"summed_phase"}
        assert result.extras["summed_phase"].data.size >= result.frame.data.shape[1]


def test_overlays_follow_the_frame_coordinates_after_a_crop():
    roi = {"x_min": 10, "x_max": 96, "y_min": 4, "y_max": 60}
    cropped = analyze_v2(
        phase_map("shock", 1), compile_v2(document(pipeline=["roi"], roi=roi))
    )
    local = analyze_v2(phase_map("shock", 1)[4:60, 10:], compile_v2(document()))
    for name in OVERLAYS:
        a, b = overlay(cropped, name), overlay(local, name)
        assert (a.x0, a.y0, a.x1, a.y1) == pytest.approx(
            (b.x0 + 10, b.y0 + 4, b.x1 + 10, b.y1 + 4)
        )


def test_segments_render_and_can_be_hidden_by_id():
    from geecs_analysis.render.specs import FigureSpec
    import matplotlib.pyplot as plt

    result = analyze_v2(phase_map("tilted", 0), compile_v2(document()))
    segments = [o for o in result.overlays if isinstance(o, Segment)]
    assert min(min(s.y0, s.y1) for s in segments) < 0  # past the frame's edge
    drawn = single(result)
    lines = {line.get_linestyle() for line in drawn.axes[0].lines}
    assert "--" in lines
    # The product draw keeps the image's limits; a segment past the edge is clipped.
    left, right, bottom, top = drawn.axes[0].images[0].get_extent()
    assert drawn.axes[0].get_xlim() == (left, right)
    assert drawn.axes[0].get_ylim() == (bottom, top)
    plt.close(drawn)
    hidden = single(
        result,
        FigureSpec(overlays={name: {"hidden": True} for name in OVERLAYS}),
    )
    assert len(hidden.axes[0].lines) == len(drawn.axes[0].lines) - len(OVERLAYS)
    plt.close(hidden)


# ------------------------------------------------------------ vocabulary


def test_the_scalars_are_the_legacy_names_plus_the_slopes_it_merged_in():
    assert set(SHOCK_SCALARS) == V2Spec.scalar_keys
    assert DownrampPhaseSpec().emitted_scalars() == set(SHOCK_SCALARS) | set(
        SLOPE_SCALARS
    )
    assert set(V2Spec.model_fields) == {"kind"}
    assert DownrampPhaseSpec() == DownrampPhaseSpec(
        window_size=20, sigma_x=100, sigma_y=20, threshold_factor=0.01
    )


def test_compiles_to_the_measure_and_converts_to_v3_identically():
    doc = document()
    recipe = compile_v2(doc)
    assert recipe.analysis.measure == DownrampPhaseSpec()
    assert recipe.input_kind == "camera"
    conversion = to_v3(doc)
    assert conversion.notes == ()
    assert conversion.recipe.measure.model_dump() == {"kind": "downramp_phase"}
    recompiled = compile_recipe(conversion.recipe, allow_file_backgrounds=True)
    assert recompiled == compile_v2(doc, allow_file_backgrounds=True)
    image = phase_map("shock", 0)
    assert dict(analyze_v2(image, recompiled).scalars) == dict(
        analyze_v2(image, recipe).scalars
    )
