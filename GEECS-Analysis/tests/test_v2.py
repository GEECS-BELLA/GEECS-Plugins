"""Differential v2 tests run the old and new backends on the same raw arrays."""

import builtins
import subprocess
import sys

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisDiagnostic

from geecs_analysis.compat.v2 import UnsupportedRecipe, analyze_v2, compile_v2
from geecs_data_utils.frames import ShotMeta


def document(kind="beam", **image):
    if kind in {"line", "trace"}:
        image = {"type": "line", "data_loading": {"data_type": "tsv"}, **image}
    else:
        image = {"type": "camera", **image}
    return AnalysisDiagnostic.model_validate(
        {
            "name": "diagnostic",
            "output_name": "output",
            "metric_suffix": "_left",
            "analyzer": {"kind": kind},
            "image": image,
        }
    )


def compare(doc, data):
    from image_analysis.ephemeral import run_document_ephemeral

    original = data.copy()
    before = doc.model_dump_json()
    old = run_document_ephemeral(doc, [data])[0]
    recipe = compile_v2(doc)
    result = analyze_v2(data, recipe, shot=ShotMeta("diagnostic", 7, 123))
    expected_data = old.line_data if doc.image.type == "line" else old.processed_image
    actual_data = (
        result.frame.as_trace() if doc.image.type == "line" else result.frame.data
    )
    np.testing.assert_array_equal(actual_data, expected_data)
    assert result.scalars.keys() == old.scalars.keys()
    for key, value in old.scalars.items():
        if np.isfinite(value):
            assert result.scalars[key] == value, key
        else:
            # Undefined outcomes are validity assertions, not finite parity.
            assert not np.isfinite(result.scalars[key])
            assert f"Nonfinite scalar: {key}" in result.notes
    np.testing.assert_array_equal(data, original)
    assert doc.model_dump_json() == before
    assert recipe.output_name == "output" and recipe.metric_suffix == "_left"
    assert result.frame.shot.shot_number == 7
    return result


@pytest.mark.parametrize(
    "pipeline",
    [
        [],
        ["background", "roi", "filtering", "thresholding"],
        ["thresholding", "roi", "background", "filtering", "thresholding"],
        ["roi", "roi"],
        ["background", "background"],
    ],
)
@pytest.mark.parametrize("kind", ["beam", "standard"])
def test_camera_sequences_match_old_backend(kind, pipeline):
    doc = document(
        kind,
        pipeline=pipeline,
        background={
            "method": "constant",
            "constant_level": 3,
            "additional_constant": -1,
        },
        roi={"x_min": 1, "x_max": 18, "y_min": 2, "y_max": 17},
        filtering={"gaussian_sigma": 1.2, "median_kernel_size": 3},
        thresholding={"method": "constant", "value": 5, "mode": "to_zero"},
    )
    data = np.random.default_rng(1).integers(0, 50, (21, 23), dtype=np.uint16)
    compare(doc, data)


@pytest.mark.parametrize("mode", ["to_zero", "truncate", "truncate_inv"])
def test_camera_threshold_variants(mode):
    compare(
        document(
            pipeline=["thresholding"],
            thresholding={"method": "constant", "value": 10, "mode": mode},
        ),
        np.arange(99).reshape(9, 11),
    )


@pytest.mark.parametrize("bounds", [(1, 200, 1, 300), (50, 60, 70, 80)])
def test_camera_roi_clamping_and_empty_fallback_keep_v2_origin(bounds):
    x0, x1, y0, y1 = bounds
    doc = document(
        pipeline=["roi"], roi={"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1}
    )
    compare(doc, np.arange(99).reshape(9, 11))


@pytest.mark.parametrize("storage", ["float32", "float64"])
@pytest.mark.parametrize("native_dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["line", "trace"])
def test_trace_scaling_storage_rounding_and_private_negative_clipping(
    kind, native_dtype, storage
):
    data = np.column_stack(
        (np.linspace(0.050, 0.180, 131), np.sin(np.linspace(0, np.pi, 131)) * 9 - 0.5)
    ).astype(native_dtype)
    doc = document(
        kind,
        pipeline=(
            ["background", "roi", "filtering", "thresholding"]
            if kind == "line"
            else ["background", "filtering", "thresholding"]
        ),
        x_scale_factor=1000,
        y_scale_factor=2.3,
        storage_dtype=storage,
        x_units="MeV",
        y_units="a.u.",
        background={"method": "constant", "constant_level": 1.2},
        roi={"x_min": 55, "x_max": 160},
        filtering={"method": "median", "kernel_size": 3},
        thresholding={"method": "absolute", "threshold_value": -10, "clip_below": True},
    )
    result = compare(doc, data)
    assert result.frame.axes[0].unit == "MeV" and result.frame.unit == "a.u."


@pytest.mark.parametrize("below", [True, False])
def test_gaussian_trace_and_clipping(below):
    compare(
        document(
            "line",
            pipeline=["filtering", "thresholding"],
            filtering={"method": "gaussian", "sigma": 1.1},
            thresholding={
                "method": "absolute",
                "threshold_value": 2,
                "clip_below": below,
            },
        ),
        np.column_stack((np.arange(20), np.random.default_rng(2).uniform(1, 4, 20))),
    )


def test_none_sections_and_disabled_line_steps_do_not_run():
    compare(
        document(
            "trace",
            pipeline=["filtering", "roi", "background", "thresholding"],
            filtering={"method": "none"},
            thresholding={"method": "none"},
            background={"method": "none"},
        ),
        np.column_stack((np.arange(20), np.ones(20))),
    )


@pytest.mark.parametrize(
    "kind,image",
    [
        ("beam", {"pipeline": ["transforms"], "transforms": {"rotation_angle": 45}}),
        (
            "beam",
            {
                "pipeline": ["thresholding"],
                "thresholding": {"method": "percentage_max", "value": 5},
            },
        ),
        (
            "beam",
            {
                "pipeline": ["thresholding"],
                "thresholding": {"method": "constant", "value": 5, "invert": True},
            },
        ),
        (
            "beam",
            {
                "pipeline": ["thresholding"],
                "thresholding": {"method": "constant", "value": 5, "mode": "binary"},
            },
        ),
        ("beam", {"pipeline": ["background"], "background": {"method": "edge"}}),
        (
            "beam",
            {
                "pipeline": ["background"],
                "background": {"method": "from_file", "file_path": "missing.npy"},
            },
        ),
        ("line", {"processing_dtype": "float32"}),
        ("line", {"storage_dtype": "int16"}),
        ("line", {"pipeline": ["filtering"], "filtering": {"method": "bilateral"}}),
        (
            "line",
            {
                "pipeline": ["thresholding"],
                "thresholding": {"method": "percentile", "percentile": 10},
            },
        ),
    ],
)
def test_unported_active_features_are_refused_before_execution(kind, image):
    with pytest.raises(UnsupportedRecipe):
        compile_v2(document(kind, **image))


def test_inactive_unsupported_features_are_not_executed():
    doc = document(
        pipeline=[], background={"method": "from_file", "file_path": "never-opened.npy"}
    )
    compare(doc, np.ones((9, 11)))


def test_compiled_recipe_owns_config_state_and_runs_without_files(monkeypatch):
    doc = document(
        pipeline=["background"], background={"method": "constant", "constant_level": 3}
    )
    compiled = compile_v2(doc)
    doc.image.background.constant_level = 20
    doc.image.pipeline.clear()

    def refuse(*args, **kwargs):
        raise AssertionError("Analysis attempted filesystem I/O")

    monkeypatch.setattr(builtins, "open", refuse)
    result = analyze_v2(np.full((9, 11), 10), compiled)
    np.testing.assert_array_equal(result.frame.data, np.full((9, 11), 7))


def test_v2_compilation_imports_no_numerical_backend():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from geecs_analysis.compat.v2 import compile_v2
from geecs_schemas.analysis import AnalysisDiagnostic
recipe = compile_v2(AnalysisDiagnostic.model_validate({"name": "camera", "analyzer": {"kind": "beam"}, "image": {"type": "camera"}}))
assert len(recipe.analysis.measure.emitted_scalars()) == 18
for module in ("numpy", "scipy", "matplotlib", "image_analysis", "geecs_data_utils"):
    assert module not in sys.modules, module
""",
        ],
        check=True,
    )


def test_trace_roi_empty_result_stays_on_legacy_route():
    from image_analysis.ephemeral import run_document_ephemeral

    doc = document("trace", pipeline=["roi"], roi={"x_min": 20, "x_max": 30})
    data = np.column_stack((np.arange(5.0), np.ones(5)))
    (legacy,) = run_document_ephemeral(doc, [data])
    assert legacy.line_data.shape == (0, 2)
    with pytest.raises(UnsupportedRecipe, match="empty result"):
        compile_v2(doc)
