"""The lowpass step: DC, stop band, pass band, phase, edges and registration."""

import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from geecs_analysis.pipeline import apply_pipeline
from geecs_analysis.registry import definitions
from geecs_analysis.specs import Pipeline
from geecs_data_utils.frames import Axis, Frame, ShotMeta

N = 2000
INDEX = np.arange(N, dtype=float)


def run(frame, **params):
    return apply_pipeline(
        frame, Pipeline.model_validate({"steps": [{"step": "lowpass", **params}]})
    )


def trace(y, x=None):
    x = INDEX[: len(y)] if x is None else x
    return Frame.from_array(
        np.asarray(y, float),
        axes=(Axis(np.asarray(x, float), "us", "t"),),
        shot=ShotMeta("scope", 2, 0.5),
        unit="V",
        label="deflection",
    )


def tone(fraction_of_nyquist):
    # Nyquist is 0.5 cycles per sample.
    return np.sin(2 * np.pi * 0.5 * fraction_of_nyquist * INDEX)


def amplitude(samples):
    middle = samples[N // 4 : 3 * N // 4]
    return np.sqrt(2 * np.mean(middle**2))


def test_dc_is_preserved():
    result = run(trace(np.full(N, 7.25)), order=4, critical_frequency=0.05)
    np.testing.assert_allclose(result.data, 7.25, rtol=1e-9)


def test_a_tone_above_the_cutoff_loses_more_than_20_db():
    result = run(trace(tone(0.4)), order=2, critical_frequency=0.1)
    assert 20 * np.log10(amplitude(result.data) / amplitude(tone(0.4))) < -20


def test_a_tone_well_below_the_cutoff_keeps_its_amplitude():
    result = run(trace(tone(0.01)), order=2, critical_frequency=0.1)
    np.testing.assert_allclose(amplitude(result.data), amplitude(tone(0.01)), rtol=0.01)


def test_zero_phase_keeps_a_symmetric_input_symmetric():
    rng = np.random.default_rng(5)
    half = rng.normal(size=N // 2)
    symmetric = np.concatenate([half, half[::-1]])
    result = run(trace(symmetric), order=3, critical_frequency=0.2)
    # Away from the edges, whose transients start from different samples.
    interior = result.data[200:-200]
    np.testing.assert_allclose(interior, interior[::-1], atol=1e-9)
    # A step edge is not delayed: its half-height crossing stays put.
    edge = run(trace((INDEX >= N / 2).astype(float)), critical_frequency=0.05)
    assert abs(edge.data[N // 2 - 1] + edge.data[N // 2] - 1.0) < 1e-6


def test_axes_unit_label_and_provenance_are_untouched():
    x = np.cumsum(np.linspace(1, 2, 50))  # nonuniform: filtered by index anyway
    frame = trace(np.sin(x), x)
    result = run(frame)
    assert result.axes[0] is frame.axes[0]
    assert (result.unit, result.label, result.shot) == ("V", "deflection", frame.shot)
    assert not result.data.flags.writeable


# sosfiltfilt's default padlen + 1: order 1 has one section with a zero,
# order 2 one section, order 4 two sections.
MINIMUM = {1: 7, 2: 10, 4: 16}


@pytest.mark.parametrize("order", sorted(MINIMUM))
@pytest.mark.parametrize("values", ["constant", "ramp"])
def test_a_trace_shorter_than_the_padding_is_refused(order, values):
    for size in (1, 3, MINIMUM[order] - 1):
        y = np.full(size, 3.0) if values == "constant" else np.arange(size, dtype=float)
        with pytest.raises(ValueError, match=f"at least {MINIMUM[order]} samples"):
            run(trace(y), order=order)


@pytest.mark.parametrize("order", sorted(MINIMUM))
def test_the_minimum_length_is_filtered_as_sosfiltfilt_filters_it(order):
    from scipy.signal import butter, sosfiltfilt

    y = np.arange(MINIMUM[order], dtype=float) ** 1.5
    result = run(trace(y), order=order, critical_frequency=0.3)
    sos = butter(order, 0.3, btype="low", output="sos")
    np.testing.assert_array_equal(result.data, sosfiltfilt(sos, y))


def test_a_nonfinite_sample_stays_visible():
    y = np.ones(100)
    y[40] = np.nan
    assert not np.isfinite(run(trace(y)).data).any()


@pytest.mark.parametrize(
    "params",
    [
        {"order": 0},
        {"critical_frequency": 0},
        {"critical_frequency": 1},
        {"critical_frequency": float("nan")},
    ],
)
def test_invalid_parameters_are_refused(params):
    with pytest.raises(ValidationError):
        Pipeline.model_validate({"steps": [{"step": "lowpass", **params}]})


def test_registered_for_traces_only():
    found = {d.spec.model_fields["step"].default: d.ndim for d in definitions()}
    assert found["lowpass"] == {1}
    with pytest.raises(ValueError):
        run(Frame.from_array(np.ones((3, 4))))


def test_spec_module_imports_without_numerical_packages():
    code = """
import sys
import geecs_analysis.steps.lowpass
for name in ("numpy", "scipy", "matplotlib"):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True)
