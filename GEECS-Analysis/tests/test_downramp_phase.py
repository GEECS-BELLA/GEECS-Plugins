"""The ``downramp_phase`` measure: the legacy shock analysis bit for bit, its PDF as overlays."""

from __future__ import annotations

import math

import numpy as np
import pytest
from geecs_schemas.analysis import (
    AnalysisRecipe,
)
from scipy import ndimage

from geecs_analysis.algorithms import downramp_shock as ported
from geecs_analysis.engine.compiled import CompiledRecipe, analyze_compiled
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


def recipe(*steps: dict) -> CompiledRecipe:
    return compile_recipe(
        AnalysisRecipe.model_validate(
            {
                "schema_version": 3,
                "device": "U_HasoLift",
                "input": {"kind": "camera"},
                "steps": list(steps),
                "measure": {"kind": "downramp_phase"},
            }
        )
    )


def core_values(result: Measurement) -> dict:
    """The core's scalars, frame and projections."""
    return {
        **{f"scalar_{k}": v for k, v in result.scalars.items()},
        "frame": result.frame.data,
        "projection_x": overlay(result, "projection_x").frame.data,
        "projection_y": overlay(result, "projection_y").frame.data,
    }


def assert_defined(stored: dict) -> None:
    """The stored scalars are finite, so matched NaNs never pass."""
    scalars = [v for k, v in stored.items() if k.startswith("scalar_")]
    assert scalars and all(math.isfinite(float(v)) for v in scalars)
    # The frame is the legacy 16-bit phase product, stored as float64.
    assert stored["frame"].dtype == np.uint16


def overlay(result: Measurement, name: str):
    return {o.id: o for o in result.overlays}[name]


def distance_to_segment(segment: Segment, x: float, y: float) -> float:
    """Distance from a point to the segment's line."""
    dx, dy = segment.x1 - segment.x0, segment.y1 - segment.y0
    return abs(dy * (x - segment.x0) - dx * (y - segment.y0)) / math.hypot(dx, dy)


# --------------------------------------------------------- the core route


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("case", CASES)
def test_the_core_route_equals_the_legacy_analyzer(case, seed, expected, request):
    image = phase_map(case, seed)
    new = analyze_compiled(image, recipe())
    stored = expected(
        f"route_{request.node.callspec.id}",
        actual=core_values(new),
    )
    assert_defined(stored)
    assert new.notes == ()
    assert {o.id for o in new.overlays} == {"projection_x", "projection_y", *OVERLAYS}


@pytest.mark.parametrize(
    "steps",
    [
        [{"step": "roi", "bounds": [[10, 58], [8, 90]]}],
        [{"step": "background_constant", "level": 0.5}, {"step": "median"}],
    ],
    ids=["roi", "background+median"],
)
def test_the_recipe_steps_precede_the_shock_analysis_as_the_pipeline_did(
    steps, expected, request
):
    data = phase_map("angled", 3)
    new = analyze_compiled(data, recipe(*steps))
    stored = expected(
        f"steps_{request.node.callspec.id}",
        actual=core_values(new),
    )
    assert_defined(stored)


def test_the_measured_shock_is_where_it_was_planted():
    plain = analyze_compiled(phase_map("shock", 0), recipe())
    assert plain.scalars[SHOCK_LOCATION] == pytest.approx(SHOCK_COLUMN, abs=3)
    assert plain.scalars[SHOCK_ANGLE] == pytest.approx(0.0, abs=0.05)
    angled = analyze_compiled(phase_map("angled", 0), recipe())
    assert angled.scalars[SHOCK_ANGLE] == pytest.approx(math.atan(0.3), abs=0.05)
    assert plain.scalars[PLATEAU_AVG] > 0


# ------------------------------------------------------------ the options


def test_options_reach_the_legacy_functions_at_the_same_values(expected):
    """The spec's knobs are the legacy constants; at other values both agree too."""
    options = dict(window_size=12, sigma_x=30.0, sigma_y=10.0, threshold_factor=0.05)
    image = phase_map("angled", 5)

    new = analyze(
        Frame.from_array(image),
        Analysis(steps=(), measure=DownrampPhaseSpec(**options)),
    )
    stored = expected(
        "options",
        actual=core_values(new),
    )
    assert_defined(stored)
    defaults = analyze_compiled(image, recipe())
    assert defaults.scalars[SHOCK_LOCATION] != new.scalars[SHOCK_LOCATION]


def search_values(module, slope, angle, image: np.ndarray) -> dict:
    """The mask, window search and angle fit of one implementation."""
    values = {}
    for n, kwargs in enumerate(
        ({}, {"sigma_x": 25, "sigma_y": 6, "threshold_factor": 0.1})
    ):
        values[f"mask_{n}"] = module.apply_gaussian_mask(image, binarize=True, **kwargs)
    masked = ported.apply_gaussian_mask(image, binarize=True)
    for window in (20, 7):
        for n, profile in enumerate((masked.sum(axis=0), masked[20], -masked[20])):
            found_slope, center = slope(profile, window)
            values[f"slope_{window}_{n}"] = float(found_slope)
            values[f"center_{window}_{n}"] = np.nan if center is None else float(center)
        values[f"angle_{window}"] = float(angle(masked, window))
    return values


def test_the_mask_and_the_window_search_are_the_legacy_functions_bit_for_bit(expected):
    image = phase_map("tilted", 2)

    def ported_slope(profile, window):
        found = ported.find_max_local_slope(profile, window)
        center = None if math.isnan(found.slope) else found.center
        return found.slope, center

    expected(
        "search",
        actual=search_values(
            ported,
            ported_slope,
            lambda m, w: ported.shock_angle(m, w).angle_rad,
            image,
        ),
    )


# ----------------------------------------------------------- degenerate


def test_a_flat_map_is_nan_with_a_note_where_the_legacy_class_fails():
    flat = np.full((40, 60), 2.5)
    result = analyze_compiled(flat, recipe())
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
        result = analyze_compiled(phase_map(case, 1), recipe())
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
    roi = {"step": "roi", "bounds": [[4, 60], [10, 96]]}
    cropped = analyze_compiled(phase_map("shock", 1), recipe(roi))
    local = analyze_compiled(phase_map("shock", 1)[4:60, 10:], recipe())
    for name in OVERLAYS:
        a, b = overlay(cropped, name), overlay(local, name)
        assert (a.x0, a.y0, a.x1, a.y1) == pytest.approx(
            (b.x0 + 10, b.y0 + 4, b.x1 + 10, b.y1 + 4)
        )


def test_overlays_follow_uniform_axes_and_are_omitted_on_a_nonuniform_one():
    """Segments follow a scaled, offset axis; a nonuniform one gets none, with a note."""
    from geecs_data_utils.frames import Axis

    image = phase_map("shock", 1)
    compiled = recipe()
    reference = analyze_compiled(image, compiled)
    # Uniform axes: offset, scaled, and one with a computed axis's float noise.
    scaled = analyze(
        Frame.from_array(
            image,
            axes=(Axis(np.linspace(0.0, 1.0, 64)), Axis(-9.6 + 0.2 * np.arange(96))),
        ),
        compiled.analysis,
    )
    for name in OVERLAYS:
        a, b = overlay(scaled, name), overlay(reference, name)
        assert (a.x0, a.y0, a.x1, a.y1) == pytest.approx(
            (-9.6 + 0.2 * b.x0, b.y0 / 63, -9.6 + 0.2 * b.x1, b.y1 / 63)
        )
    assert scaled.notes == reference.notes == ()
    # A nonuniform axis: the scalars live on the sample grid and are
    # untouched; the segments are gone and a note says why.
    for axes, names in (
        ((Axis(np.arange(64.0)), Axis(np.arange(96.0) ** 2)), "x"),
        ((Axis(np.arange(64.0) ** 2), Axis(np.arange(96.0) ** 2)), "x, y"),
    ):
        result = analyze(Frame.from_array(image, axes=axes), compiled.analysis)
        assert dict(result.scalars) == dict(reference.scalars)
        assert {o.id for o in result.overlays} == {"projection_x", "projection_y"}
        assert result.notes == (
            f"Shock overlays omitted on a nonuniform axis ({names}): "
            "a line on the sample grid is not a line in its coordinates",
        )


def test_segments_render_and_can_be_hidden_by_id():
    from geecs_analysis.render.specs import FigureSpec
    import matplotlib.pyplot as plt

    result = analyze_compiled(phase_map("tilted", 0), recipe())
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
    assert DownrampPhaseSpec().emitted_scalars() == set(SHOCK_SCALARS) | set(
        SLOPE_SCALARS
    )
    assert DownrampPhaseSpec() == DownrampPhaseSpec(
        window_size=20, sigma_x=100, sigma_y=20, threshold_factor=0.01
    )
