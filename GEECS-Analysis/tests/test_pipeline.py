"""Order, coordinates, schema isolation and parity with the existing pipeline."""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from geecs_analysis.pipeline import apply_pipeline
from geecs_analysis.specs import Pipeline
from geecs_data_utils.frames import Axis, Frame, ShotMeta


def run(frame, *steps):
    return apply_pipeline(frame, Pipeline.model_validate({"steps": steps}))


def test_order_duplicates_and_input_ownership():
    data = np.array([-3.0, 2.0, 10.0])
    frame = Frame.from_array(data, shot=ShotMeta("camera", 7, 12.3), unit="counts")
    result = run(
        frame,
        {"step": "background_constant", "level": 1},
        {"step": "clip_below", "level": 0},
        {"step": "background_constant", "level": 1},
    )
    np.testing.assert_array_equal(result.data, [-1, 0, 8])
    np.testing.assert_array_equal(frame.data, data)
    assert result.shot is frame.shot
    assert result.unit == "counts"
    assert result.axes[0] is frame.axes[0]
    assert not result.data.flags.writeable
    assert apply_pipeline(frame, Pipeline()) is frame


def test_calibrated_image_nested_crops_keep_global_coordinates():
    frame = Frame.from_array(
        np.arange(30).reshape(5, 6),
        axes=(
            Axis(np.arange(5) * 0.25 + 10, "mm", "y"),
            Axis(np.arange(6) * 0.5 + 20, "mm", "x"),
        ),
    )
    result = run(
        frame,
        {"step": "roi", "bounds": [[1, 5], [1, 6]]},
        {"step": "roi", "bounds": [[1, 3], [2, 4]]},
    )
    np.testing.assert_array_equal(result.data, frame.data[2:4, 3:5])
    np.testing.assert_array_equal(result.axes[0].values, [10.5, 10.75])
    np.testing.assert_array_equal(result.axes[1].values, [21.5, 22])
    assert [a.unit for a in result.axes] == ["mm", "mm"]


def test_physical_roi_preserves_nonmonotonic_samples_and_both_endpoints():
    frame = Frame.from_trace(
        [[8, 80], [3, 30], [5, 50], [2, 20], [4, 40]], x_unit="MeV"
    )
    result = run(frame, {"step": "roi", "bounds": [[3, 5]], "units": "axis"})
    np.testing.assert_array_equal(result.as_trace(), [[3, 30], [5, 50], [4, 40]])
    assert result.axes[0].unit == "MeV"


def test_physical_roi_selects_each_image_dimension_independently():
    frame = Frame.from_array(
        np.arange(12).reshape(3, 4),
        axes=(Axis([5, 4, 2], "mm", "y"), Axis([1, 3, 8, 6], "mm", "x")),
    )
    result = run(frame, {"step": "roi", "bounds": [[4, None], [3, 6]], "units": "axis"})
    np.testing.assert_array_equal(result.data, [[1, 3], [5, 7]])
    np.testing.assert_array_equal(result.axes[0].values, [5, 4])
    np.testing.assert_array_equal(result.axes[1].values, [3, 6])


@pytest.mark.parametrize(
    "step",
    [
        {"step": "median", "kernel": 0},
        {"step": "median", "kernel": 2},
        {"step": "median", "kernel": True},
        {"step": "median", "kernel": 3.5},
        {"step": "gaussian", "sigma": 0},
        {"step": "gaussian", "sigma": float("nan")},
        {"step": "clip_below", "level": float("inf")},
        {"step": "background_constant", "level": 1, "typo": 2},
        {"step": "roi", "bounds": [[-1, 3]]},
        {"step": "roi", "bounds": [[1.5, 3]]},
        {"step": "roi", "bounds": [[5, 3]], "units": "axis"},
        {"step": "roi", "bounds": [[0, float("inf")]], "units": "axis"},
        {"step": "roi", "bounds": []},
        {"step": "not_registered"},
    ],
)
def test_invalid_specs_are_rejected(step):
    with pytest.raises(ValidationError):
        Pipeline.model_validate({"steps": [step]})


@pytest.mark.parametrize(
    "step",
    [
        {"step": "roi", "bounds": [[0, 2], [0, 2]]},
        {"step": "roi", "bounds": [[2, 2]]},
        {"step": "roi", "bounds": [[10, 12]], "units": "axis"},
    ],
)
def test_invalid_or_empty_geometry_is_explicit(step):
    with pytest.raises(ValueError):
        run(Frame.from_array([1, 2, 3]), step)


def test_schema_round_trip_retains_duplicate_steps():
    pipeline = Pipeline.model_validate(
        {"steps": [{"step": "median", "kernel": 3}, {"step": "median", "kernel": 5}]}
    )
    assert Pipeline.model_validate_json(pipeline.model_dump_json()) == pipeline
    mapping = Pipeline.model_json_schema()["properties"]["steps"]["items"][
        "discriminator"
    ]["mapping"]
    assert set(mapping) == {
        "background_constant",
        "background_frame",
        "circular_mask",
        "crosshair_mask",
        "derivative",
        "interpolate",
        "lowpass",
        "clip_below",
        "clip_above",
        "zero_below",
        "median",
        "gaussian",
        "roi",
        "rotate",
        "vignette",
    }
    with pytest.raises(ValidationError):
        pipeline.steps[0].kernel = 7


def test_schema_import_is_numerical_and_io_dependency_free():
    code = """
import sys
from geecs_analysis.specs import Pipeline
Pipeline.model_json_schema()
for name in ("numpy", "scipy", "cv2", "matplotlib", "geecs_data_utils"):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_processing_imports_only_the_data_value_model():
    root = Path(__file__).parents[1] / "geecs_analysis"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert name not in {"pathlib", "os", "io", "matplotlib.pyplot"}, path
                if name.startswith("geecs_data_utils"):
                    # The value models only (frames, and the tables a
                    # measure binds); never a reader.
                    assert name in {
                        "geecs_data_utils.frames",
                        "geecs_data_utils.tables",
                    }, path
                assert not name.startswith("scan_analysis"), path


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("kind", ["median", "gaussian"])
def test_filters_match_legacy_including_reflected_edges(ndim, kind, expected, request):
    samples = np.random.default_rng(13).normal(size=(19,) if ndim == 1 else (7, 9))
    # Deliberate edge impulses distinguish reflect from constant/wrap padding.
    samples.flat[0] = 100
    frame = Frame.from_array(samples)
    parameter = 3 if kind == "median" else 1.3
    actual = run(
        frame, {"step": kind, "kernel" if kind == "median" else "sigma": parameter}
    )
    expected(
        f"filter_{request.node.callspec.id}",
        actual=actual.data,
    )


def test_trace_processing_matches_legacy_composition(expected):
    trace = np.column_stack(
        (np.linspace(50, 180, 131), np.random.default_rng(14).normal(size=131))
    )
    result = run(
        Frame.from_trace(trace),
        {"step": "background_constant", "level": 0.2},
        {"step": "roi", "bounds": [[60, 160]], "units": "axis"},
        {"step": "median", "kernel": 3},
        {"step": "clip_below", "level": -0.1},
    )
    expected(
        "trace_composition",
        actual=result.as_trace(),
    )


def test_thresholds_match_distinct_legacy_semantics(expected):
    data = np.array([[-2.0, 0.0, 1.0, 3.0, 10.0]])
    image = Frame.from_array(data)
    zeroed = run(image, {"step": "zero_below", "level": 3})
    expected(
        "threshold_zero_below",
        actual=zeroed.data,
    )
    trace = Frame.from_array(data[0])
    for kind in ("clip_below", "clip_above"):
        expected(
            f"threshold_{kind}",
            actual=run(trace, {"step": kind, "level": 3}).as_trace(),
        )
    np.testing.assert_array_equal(zeroed.data, [[0, 0, 0, 3, 10]])


def test_beam_processing_matches_legacy_composition(expected):
    image = np.random.default_rng(15).integers(0, 200, (15, 17), dtype=np.uint16)
    result = run(
        Frame.from_array(image),
        {"step": "background_constant", "level": 90},
        {"step": "roi", "bounds": [[1, 12], [2, 14]]},
        {"step": "zero_below", "level": 0},
    )
    expected(
        "beam_composition",
        actual=result.data,
    )
    np.testing.assert_array_equal(result.axes[0].values, np.arange(1, 12))
    np.testing.assert_array_equal(result.axes[1].values, np.arange(2, 14))
