"""Step and runtime parity: recipes run on raw arrays against the stored legacy outputs.

Each case is the v3 form of a retired v2 differential test (#1039); the
stored ``.npz`` values were captured from the legacy backend and are
unchanged, so these pin trace ROI, interpolation (including descending and
duplicate x), circular-mask local centres, crosshair + rotation
composition, camera ROI clamping and its empty-fallback origin, the
threshold variants, trace storage rounding and negative clipping, and the
camera step sequences.
"""

import numpy as np
import pytest
from geecs_data_utils.frames import ShotMeta
from geecs_schemas.analysis import AnalysisRecipe

from geecs_analysis.engine.v2 import analyze_v2
from geecs_analysis.recipe import compile_recipe

LINE = {"kind": "line", "loading": {"data_type": "tsv"}, "label": "x vs y"}


def recipe(steps=(), measure="beam", **line):
    source = {**LINE, **line} if line or measure in {"line", "none_line"} else None
    return AnalysisRecipe.model_validate(
        {
            "schema_version": 3,
            "device": "diagnostic",
            "output_name": "output",
            "scalar_suffix": "_left",
            "input": source or {"kind": "camera"},
            "steps": list(steps),
            "measure": {"kind": "none" if measure == "none_line" else measure},
        }
    )


def trace_recipe(steps=(), measured=True, **line):
    """A line recipe: ``measured`` is the legacy ``line`` kind, else ``trace``."""
    return recipe(steps, "line" if measured else "none_line", **line)


@pytest.fixture
def compare(expected, request):
    """Run ``doc`` on ``data`` through the core and check it against the stored output."""
    calls = iter(range(100))

    def run(doc, data):
        line = doc.input_kind == "line"
        original = data.copy()
        before = doc.model_dump_json()
        compiled = compile_recipe(doc)
        result = analyze_v2(data, compiled, shot=ShotMeta("diagnostic", 7, 123))
        actual = {
            "data": result.frame.as_trace() if line else result.frame.data,
            **result.scalars,
        }
        stored = expected(f"{request.node.name}-{next(calls)}", actual)
        for key, value in stored.items():
            if key != "data" and not np.isfinite(value):
                # Undefined outcomes are validity assertions, not finite parity.
                assert f"Nonfinite scalar: {key}" in result.notes
        np.testing.assert_array_equal(data, original)
        assert doc.model_dump_json() == before
        assert compiled.output_name == "output" and compiled.metric_suffix == "_left"
        assert result.frame.shot.shot_number == 7
        return result

    return run


BG = [
    {"step": "background_constant", "level": 3},
    {"step": "background_constant", "level": -1},
]
ROI = {"step": "roi", "bounds": [[2, 17], [1, 18]]}
FILTER = [{"step": "gaussian", "sigma": 1.2}, {"step": "median"}]
ZERO = {"step": "zero_below", "level": 5}


SEQUENCES = [
    [],
    [*BG, ROI, *FILTER, ZERO],
    [ZERO, ROI, *BG, *FILTER, ZERO],
    [ROI, ROI],
    [*BG, *BG],
]


@pytest.mark.parametrize(
    "kind,pipeline",
    [
        pytest.param(kind, steps, id=f"{kind}-pipeline{i}")
        for kind in ("beam", "standard")
        for i, steps in enumerate(SEQUENCES)
        # A beam without a crop: the legacy coordinates started at an inactive
        # roi section's origin, which a recipe has no way to say (#1039).
        if kind == "standard" or any(step is ROI for step in steps)
    ],
)
def test_camera_sequences_match_old_backend(kind, pipeline, compare):
    doc = recipe(pipeline, "beam" if kind == "beam" else "none")
    data = np.random.default_rng(1).integers(0, 50, (21, 23), dtype=np.uint16)
    compare(doc, data)


@pytest.mark.parametrize(
    "step",
    [
        pytest.param("zero_below", id="to_zero"),
        pytest.param("clip_above", id="truncate"),
        pytest.param("clip_below", id="truncate_inv"),
    ],
)
def test_camera_threshold_variants(step, compare):
    compare(recipe([{"step": step, "level": 10}]), np.arange(99).reshape(9, 11))


@pytest.mark.parametrize("bounds", [[[1, 300], [1, 200]], [[70, 80], [50, 60]]])
def test_camera_roi_clamping_and_empty_fallback_keep_legacy_origin(bounds, compare):
    compare(recipe([{"step": "roi", "bounds": bounds}]), np.arange(99).reshape(9, 11))


@pytest.mark.parametrize("storage", ["float32", "float64"])
@pytest.mark.parametrize("native_dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["line", "trace"])
def test_trace_scaling_storage_rounding_and_private_negative_clipping(
    kind, native_dtype, storage, compare
):
    data = np.column_stack(
        (np.linspace(0.050, 0.180, 131), np.sin(np.linspace(0, np.pi, 131)) * 9 - 0.5)
    ).astype(native_dtype)
    roi = [{"step": "roi", "bounds": [[55, 160]], "units": "axis"}]
    steps = [
        {"step": "background_constant", "level": 1.2},
        *(roi if kind == "line" else []),
        {"step": "median"},
        {"step": "clip_below", "level": -10},
    ]
    doc = trace_recipe(
        steps,
        kind == "line",
        x_scale=1000.0,
        y_scale=2.3,
        x_unit="MeV",
        y_unit="a.u.",
        storage_dtype=storage,
    )
    result = compare(doc, data)
    assert result.frame.axes[0].unit == "MeV" and result.frame.unit == "a.u."


@pytest.mark.parametrize("below", [True, False])
def test_gaussian_trace_and_clipping(below, compare):
    clip = {"step": "clip_below" if below else "clip_above", "level": 2}
    compare(
        trace_recipe([{"step": "gaussian", "sigma": 1.1}, clip]),
        np.column_stack((np.arange(20), np.random.default_rng(2).uniform(1, 4, 20))),
    )


@pytest.mark.parametrize(
    "bounds",
    [(1.5, 3.5), (0.0, 4.0), (-5.0, 2.0), (3.0, 99.0), (2.0, 2.0)],
)
@pytest.mark.parametrize("storage", ["float32", "float64"])
def test_trace_roi_matches_the_legacy_trace_analyzer(bounds, storage, compare):
    x = np.linspace(0.0, 4.0, 9)
    data = np.column_stack((x, np.sin(x) + 2))
    doc = trace_recipe(
        [{"step": "roi", "bounds": [list(bounds)], "units": "axis"}],
        False,
        storage_dtype=storage,
    )
    compare(doc, data)


def test_a_trace_roi_that_selects_nothing_fails_the_shot_explicitly():
    """Legacy returned an empty trace; the core fails that shot with a reason."""
    doc = trace_recipe([{"step": "roi", "bounds": [[20, 30]], "units": "axis"}], False)
    data = np.column_stack((np.arange(5.0), np.ones(5)))
    with pytest.raises(ValueError, match="ROI selects no samples"):
        analyze_v2(data, compile_recipe(doc))


MASK = {"step": "circular_mask", "center": [17, 21], "radius": 12, "value": 0.5}
CROP = {"step": "roi", "bounds": [[2, 36], [3, 44]]}


@pytest.mark.parametrize("mask_outside", [True, False])
@pytest.mark.parametrize("pipeline", [[CROP, MASK], [MASK, CROP], [MASK, CROP, MASK]])
def test_circular_mask_keeps_legacy_local_center(mask_outside, pipeline, compare):
    inside = {} if mask_outside else {"mask_outside": False}
    steps = [{**step, **inside} if step is MASK else step for step in pipeline]
    compare(recipe(steps), np.random.default_rng(7).uniform(1, 100, (41, 51)))


@pytest.mark.parametrize("kind", ["line", "trace"])
@pytest.mark.parametrize("storage_dtype", ["float32", "float64"])
@pytest.mark.parametrize("axis_order", ["ascending", "descending", "duplicate"])
def test_interpolation_matches_legacy_samples_and_measures(
    kind, storage_dtype, axis_order, compare
):
    x = np.array([1, 2, 3.5, 5, 8, 9], dtype=np.float32)
    if axis_order == "descending":
        x = x[::-1]
    elif axis_order == "duplicate":
        x[2] = x[1]
    data = np.column_stack([x, [1, 2, 9, 6, 2, 1]]).astype(np.float32)
    doc = trace_recipe(
        [{"step": "interpolate", "count": 71, "lower": 0, "upper": 30}],
        kind == "line",
        x_scale=2.5,
        storage_dtype=storage_dtype,
    )
    compare(doc, data)


def _crosshairs(angle):
    extra = {"angle": angle} if angle else {}
    return [
        {"step": "crosshair_mask", "center": [5, 7], "width": 9, "height": 7,
         "thickness": 3, **extra, "value": 2},
        {"step": "crosshair_mask", "center": [8, 9], "width": 7, "height": 11,
         "thickness": 4, **extra, "value": 2},
    ]  # fmt: skip


@pytest.mark.parametrize("angle", [0, -3.5, 2.5, -8])
@pytest.mark.parametrize("pipeline", [[0], [1]])
def test_camera_crosshairs_and_rotation_match_legacy_composition(
    angle, pipeline, compare
):
    roi = {"step": "roi", "bounds": [[3, 17], [2, 18]]}
    rotate = [{"step": "rotate", "angle": angle}] if angle else []
    masks = _crosshairs(angle)
    if pipeline == [0]:
        steps = [*masks, roi, *rotate]
    else:
        steps = [roi, *masks, *rotate, *masks, *rotate]
    compare(
        recipe(steps),
        np.random.default_rng(19).integers(100, 200, (21, 23), dtype=np.uint16),
    )
