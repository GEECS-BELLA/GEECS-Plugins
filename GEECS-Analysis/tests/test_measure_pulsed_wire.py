"""The ``pulsed_wire`` measure: drift plateaus, kicks, and visible failures."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from geecs_data_utils.frames import Axis, Frame
from pydantic import ValidationError

from geecs_analysis.measures.pulsed_wire import PulsedWireSpec
from geecs_analysis.registry import measure_definition, measure_definitions
from geecs_analysis.run import analyze
from geecs_analysis.specs import Analysis

DT = 1e-6  # one sample per microsecond


def synthetic_trace(levels: list[float], *, drift: int = 400, element: int = 600):
    """Flat plateaus at ``levels`` joined by linear ramps inside each element.

    Returns the frame and the drift windows (strictly inside each plateau).
    """
    pieces, windows, t0 = [], [], 0
    for i, level in enumerate(levels):
        pieces.append(np.full(drift, level))
        windows.append({"start": (t0 + 50) * DT, "end": (t0 + drift - 50) * DT})
        t0 += drift
        if i + 1 < len(levels):
            pieces.append(np.linspace(level, levels[i + 1], element + 2)[1:-1])
            t0 += element
    y = np.concatenate(pieces)
    t = np.arange(y.size) * DT
    return Frame.from_array(y, axes=(Axis(values=t, unit="s"),)), windows


def spec(windows, lengths=None) -> PulsedWireSpec:
    lengths = lengths or [None] * (len(windows) - 1)
    return PulsedWireSpec.model_validate(
        {
            "windows": windows,
            "elements": [
                {"name": f"Q{i + 1}", "length": length}
                for i, length in enumerate(lengths)
            ],
        }
    )


def measure(frame, s):
    return analyze(frame, Analysis(measure=s))


@pytest.mark.parametrize(
    "levels",
    [[0.0, 2.5], [1.0, -3.0, 4.0], [0.0, 1.5, -0.25, 0.75]],
    ids=["one", "two", "triplet"],
)
def test_kicks_are_the_plateau_differences(levels):
    frame, windows = synthetic_trace(levels)
    result = measure(frame, spec(windows))
    n = len(levels)
    assert set(result.scalars) == spec(windows).emitted_scalars()
    for i, level in enumerate(levels):
        assert result.scalars[f"plateau_{i}"] == pytest.approx(level, abs=1e-12)
    for i in range(1, n):
        assert result.scalars[f"kick_{i}"] == pytest.approx(
            levels[i] - levels[i - 1], abs=1e-12
        )
    assert result.notes == ()
    assert result.frame is frame or np.array_equal(result.frame.data, frame.data)


def test_lengths_give_per_length_kicks_only_where_set():
    frame, windows = synthetic_trace([0.0, 2.0, 5.0])
    s = spec(windows, lengths=[0.5, None])
    result = measure(frame, s)
    assert s.emitted_scalars() == {
        "plateau_0",
        "plateau_1",
        "plateau_2",
        "kick_1",
        "kick_2",
        "kick_per_length_1",
    }
    assert set(result.scalars) == s.emitted_scalars()
    assert result.scalars["kick_per_length_1"] == pytest.approx(4.0)


def test_window_outside_the_axis_is_nan_with_a_note_never_zero():
    frame, windows = synthetic_trace([0.0, 2.0, 5.0])
    windows[2] = {"start": 1.0, "end": 2.0}  # far beyond the trace's end
    result = measure(frame, spec(windows, lengths=[1.0, 1.0]))
    assert result.scalars["plateau_1"] == pytest.approx(2.0)
    assert result.scalars["kick_1"] == pytest.approx(2.0)
    for key in ("plateau_2", "kick_2", "kick_per_length_2"):
        assert np.isnan(result.scalars[key]), key
    assert any("window 2" in note and "no samples" in note for note in result.notes)
    assert "Nonfinite scalar: kick_2" in result.notes


def test_all_nonfinite_window_is_nan_and_partly_nonfinite_is_noted():
    frame, windows = synthetic_trace([0.0, 2.0, 5.0])
    y = frame.data.copy()
    t = frame.axes[0].values
    first = (t >= windows[0]["start"]) & (t <= windows[0]["end"])
    y[first] = np.nan
    y[np.flatnonzero((t >= windows[1]["start"]) & (t <= windows[1]["end"]))[:3]] = (
        np.inf
    )
    result = measure(Frame.from_array(y, axes=frame.axes), spec(windows))
    assert np.isnan(result.scalars["plateau_0"]) and np.isnan(result.scalars["kick_1"])
    assert result.scalars["plateau_1"] == pytest.approx(2.0)
    assert result.scalars["kick_2"] == pytest.approx(3.0)
    assert any("window 0" in n and "only nonfinite" in n for n in result.notes)
    assert any("window 1" in n and "left out 3" in n for n in result.notes)


def test_processed_frame_is_returned_unchanged():
    frame, windows = synthetic_trace([0.0, 1.0])
    before = frame.data.copy()
    result = measure(frame, spec(windows))
    np.testing.assert_array_equal(result.frame.data, before)
    assert result.overlays == ()


def test_defaults_are_a_documented_triplet():
    s = PulsedWireSpec()
    assert len(s.windows) == 4 and len(s.elements) == 3
    assert s.emitted_scalars() == set(PulsedWireSpec.scalar_docs)


W = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]


@pytest.mark.parametrize(
    "doc",
    [
        {"windows": [W[0]], "elements": []},
        {"windows": [W[0], {"start": 0.5, "end": 3.0}], "elements": [{"name": "a"}]},
        {"windows": [W[1], W[0]], "elements": [{"name": "a"}]},
        {"windows": [W[0], {"start": 1.0, "end": 3.0}], "elements": [{"name": "a"}]},
        {"windows": [{"start": 1.0, "end": 0.0}, W[1]], "elements": [{"name": "a"}]},
        {"windows": W, "elements": []},
        {"windows": W, "elements": [{"name": "a"}, {"name": "b"}]},
        {
            "windows": W + [{"start": 4.0, "end": 5.0}],
            "elements": [{"name": "a"}, {"name": "a"}],
        },
        {"windows": W, "elements": [{"name": ""}]},
        {"windows": W, "elements": [{"name": "a", "length": 0.0}]},
    ],
    ids=[
        "one-window",
        "overlap",
        "disorder",
        "touching",
        "start-after-end",
        "too-few-elements",
        "too-many-elements",
        "duplicate-names",
        "empty-name",
        "zero-length",
    ],
)
def test_validation_refusals(doc):
    with pytest.raises(ValidationError):
        PulsedWireSpec.model_validate(doc)


def test_registered_for_traces_only():
    (item,) = [d for d in measure_definitions() if d.spec is PulsedWireSpec]
    assert item.ndim == frozenset({1})
    assert measure_definition(PulsedWireSpec()) is item
    with pytest.raises(ValueError):
        item.function(Frame.from_array(np.zeros((3, 3))), PulsedWireSpec())


def test_spec_imports_without_numerical_libraries():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from geecs_analysis.specs import Analysis
recipe = Analysis.model_validate({"measure": {"kind": "pulsed_wire"}})
assert len(recipe.measure.emitted_scalars()) == 10
Analysis.model_json_schema()
for module in ("numpy", "scipy", "matplotlib", "geecs_data_utils"):
    assert module not in sys.modules, module
""",
        ],
        check=True,
    )


def test_schema_describes_the_nested_window_and_element_fields():
    from geecs_analysis.recipe import recipe_schema

    defs = recipe_schema()["$defs"]
    assert defs["PulsedWireSpec"]["x-ndim"] == [1]
    for name in ("PulsedWireWindow", "PulsedWireElement"):
        for field, prop in defs[name]["properties"].items():
            assert prop.get("description"), (name, field)
