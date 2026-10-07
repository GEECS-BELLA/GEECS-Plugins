"""The derivative step: values, axes, units and registration."""

import subprocess
import sys

import numpy as np
import pytest

from geecs_analysis.pipeline import apply_pipeline
from geecs_analysis.registry import definitions
from geecs_analysis.specs import Pipeline
from geecs_data_utils.frames import Axis, Frame, ShotMeta


def run(frame, **params):
    return apply_pipeline(
        frame, Pipeline.model_validate({"steps": [{"step": "derivative", **params}]})
    )


def trace(x, y, *, unit="", x_unit="", label=""):
    return Frame.from_array(
        np.asarray(y, float),
        axes=(Axis(np.asarray(x, float), x_unit, "t"),),
        shot=ShotMeta("scope", 3, 1.5),
        unit=unit,
        label=label,
    )


@pytest.mark.parametrize(
    "x",
    [
        np.linspace(-2, 5, 40),
        np.array([0.0, 0.1, 0.15, 0.7, 1.0, 2.5, 2.6, 4.0]),
    ],
    ids=["uniform", "nonuniform"],
)
def test_a_line_has_its_slope_everywhere(x):
    result = run(trace(x, 3.5 * x - 2.0))
    np.testing.assert_allclose(result.data, 3.5, rtol=0, atol=1e-12)


def test_sine_becomes_cosine_on_a_fine_grid():
    x = np.linspace(0, 2 * np.pi, 2001)
    result = run(trace(x, np.sin(x)))
    np.testing.assert_allclose(result.data, np.cos(x), atol=1e-5)


def test_a_descending_axis_gives_the_right_sign():
    x = np.linspace(4, -4, 50)
    result = run(trace(x, 2.0 * x))
    np.testing.assert_allclose(result.data, 2.0, atol=1e-12)
    rising = run(trace(x, -(x**2)))
    np.testing.assert_allclose(rising.data[1:-1], -2 * x[1:-1], atol=1e-12)


@pytest.mark.parametrize(
    ("unit", "x_unit", "expected"),
    [("V", "us", "V/us"), ("", "us", ""), ("V", "", ""), ("", "", "")],
)
def test_unit_composition(unit, x_unit, expected):
    x = np.arange(5.0)
    assert run(trace(x, x, unit=unit, x_unit=x_unit)).unit == expected


def test_axes_label_and_provenance_are_untouched():
    x = np.array([0.0, 1.0, 3.0, 6.0])
    frame = trace(x, x**2, unit="V", x_unit="us", label="deflection")
    result = run(frame)
    assert result.axes[0] is frame.axes[0]
    assert result.label == "deflection"
    assert result.shot is frame.shot
    np.testing.assert_array_equal(frame.data, x**2)
    assert not result.data.flags.writeable


def test_nonfinite_samples_stay_visible():
    x = np.arange(6.0)
    y = x.copy()
    y[3] = np.nan
    result = run(trace(x, y))
    assert np.isnan(result.data[[2, 4]]).all()
    np.testing.assert_allclose(result.data[[0, 1, 5]], 1.0)


def test_a_single_sample_is_refused():
    with pytest.raises(ValueError, match="two samples"):
        run(trace([1.0], [2.0]))


def test_registered_for_traces_only():
    found = {d.spec.model_fields["step"].default: d.ndim for d in definitions()}
    assert found["derivative"] == {1}
    with pytest.raises(ValueError):
        run(Frame.from_array(np.ones((3, 4))))


def test_spec_module_imports_without_numerical_packages():
    code = """
import sys
import geecs_analysis.steps.derivative
for name in ("numpy", "scipy", "matplotlib"):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True)
