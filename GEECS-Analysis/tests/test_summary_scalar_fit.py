"""The scalar_fit summary: scan-level fit numbers beside the figure, and the output contract."""

import math
import subprocess
import sys

import numpy as np
import pytest
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import (
    AverageSummary,
    ImageGridSummary,
    ScalarFitSummary,
    WaterfallSummary,
)
from matplotlib.figure import Figure

from geecs_analysis.measurement import Measurement
from geecs_analysis.registry import (
    SummaryOutput,
    summary_definition,
    summary_output,
)
from geecs_analysis.render import RenderError
from geecs_analysis.render.specs import FigureSpec
from geecs_analysis.algorithms.linear_fit import FIT_SUFFIXES, linear_fit

FIGURE = FigureSpec(fig={"dpi": 30})


def shot(**scalars):
    return Measurement(
        scalars, Frame.from_trace(np.column_stack([[1, 2, 3], [0, 1, 0]]))
    )


def draw(results, positions, keys, label="hexapod x (mm)", figure=FIGURE):
    options = ScalarFitSummary(scalars=keys)
    return summary_definition(options).function(
        results, positions, label, options, figure
    )


def test_an_exact_line_recovers_slope_intercept_and_zero_crossing():
    positions = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    results = [shot(kick_1=0.5 * p - 0.25, kick_2=-3.0 * p + 1.5) for p in positions]
    out = draw(results, positions, ["kick_1", "kick_2"])
    assert isinstance(out, SummaryOutput) and isinstance(out.figure, Figure)
    s = out.scalars
    assert set(s) == {f"{k}_{x}" for k in ("kick_1", "kick_2") for x in FIT_SUFFIXES}
    assert s["kick_1_slope"] == pytest.approx(0.5, abs=1e-12)
    assert s["kick_1_intercept"] == pytest.approx(-0.25, abs=1e-12)
    assert s["kick_1_zero_crossing"] == pytest.approx(0.5, abs=1e-12)
    assert s["kick_2_slope"] == pytest.approx(-3.0, abs=1e-12)
    assert s["kick_2_zero_crossing"] == pytest.approx(0.5, abs=1e-12)
    assert s["kick_1_r2"] == pytest.approx(1.0) and s["kick_1_points"] == 6
    for key in ("slope", "intercept", "zero_crossing"):
        assert s[f"kick_1_{key}_stderr"] == pytest.approx(0.0, abs=1e-9)
    assert out.notes == ()
    ax = out.figure.axes[0]
    assert ax.get_xlabel() == "hexapod x (mm)" and out.figure.dpi == 30
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend[0].startswith("kick_1: slope 0.5, zero 0.5")
    assert legend[1].startswith("kick_2: slope -3, zero 0.5")
    # one zero-crossing marker per key (a vertical line at x = 0.5)
    verticals = [
        line
        for line in ax.lines
        if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), 0.5)
    ]
    assert len(verticals) == 2


def test_a_crossing_outside_the_scan_is_reported_but_not_drawn():
    """A skew plane's far crossing must not stretch the axis over the points."""
    positions = [-6.0, -4.0, -2.0, 0.0]
    results = [shot(kick_1=-0.01 * p + 0.2) for p in positions]
    out = draw(results, positions, ["kick_1"])
    assert out.scalars["kick_1_zero_crossing"] == pytest.approx(20.0)
    ax = out.figure.axes[0]
    assert not [
        line
        for line in ax.lines
        if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), 20.0)
    ]
    assert ax.get_xlim()[1] < 1.0


def test_standard_errors_match_ordinary_least_squares():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.1, 2.9, 5.2, 6.8, 9.1])
    fit, notes = linear_fit(x, y, "k")
    n = len(x)
    design = np.column_stack([x, np.ones(n)])
    beta, ssr, *_ = np.linalg.lstsq(design, y, rcond=None)
    cov = ssr[0] / (n - 2) * np.linalg.inv(design.T @ design)
    assert fit["slope"] == pytest.approx(beta[0])
    assert fit["slope_stderr"] == pytest.approx(math.sqrt(cov[0, 0]))
    assert fit["intercept_stderr"] == pytest.approx(math.sqrt(cov[1, 1]))
    m, b = beta
    grad = np.array([b / m**2, -1 / m])
    assert fit["zero_crossing"] == pytest.approx(-b / m)
    assert fit["zero_crossing_stderr"] == pytest.approx(math.sqrt(grad @ cov @ grad))
    assert fit["r2"] == pytest.approx(1 - ssr[0] / ((y - y.mean()) ** 2).sum())
    assert notes == []


def test_two_points_give_the_line_with_nan_errors_and_a_note():
    fit, notes = linear_fit([1.0, 3.0], [2.0, 6.0], "k")
    assert (fit["slope"], fit["intercept"], fit["zero_crossing"]) == (2.0, 0.0, 0.0)
    for key in ("slope_stderr", "intercept_stderr", "zero_crossing_stderr"):
        assert math.isnan(fit[key])
    assert fit["points"] == 2 and len(notes) == 1 and "k" in notes[0]


@pytest.mark.parametrize("y", [[], [1.0], [1.0, float("nan")]])
def test_fewer_than_two_finite_points_are_all_nan_with_a_note(y):
    fit, notes = linear_fit([0.0, 1.0][: len(y)], y, "k")
    assert all(math.isnan(fit[s]) for s in FIT_SUFFIXES if s != "points")
    assert fit["points"] == sum(math.isfinite(v) for v in y)
    assert notes and "no line" in notes[0]


def test_a_flat_line_has_no_zero_crossing_and_says_so():
    fit, notes = linear_fit([0.0, 1.0, 2.0], [4.0, 4.0, 4.0], "k")
    assert fit["slope"] == 0.0 and fit["intercept"] == pytest.approx(4.0)
    assert math.isnan(fit["zero_crossing"]) and math.isnan(fit["zero_crossing_stderr"])
    assert math.isnan(fit["r2"])
    assert any("zero crossing" in n for n in notes)


def test_positions_that_do_not_vary_fit_nothing():
    fit, notes = linear_fit([1.0, 1.0, 1.0], [0.0, 1.0, 2.0], "k")
    assert math.isnan(fit["slope"]) and fit["points"] == 3
    assert notes and "do not vary" in notes[0]


def test_a_missing_key_is_a_nan_point_and_one_note_never_an_exception():
    positions = [0.0, 1.0, 2.0, 3.0, None]
    results = [
        shot(kick_1=2.0 * p - 2.0) if p != 2.0 and p is not None else shot(other=1.0)
        for p in positions
    ]
    results[1] = shot(kick_1=float("nan"))
    out = draw(results, positions, ["kick_1", "absent"])
    s = out.scalars
    # points at 0 and 3 survive: the missing key, the NaN value and the None
    # position are each dropped
    assert s["kick_1_points"] == 2
    assert s["kick_1_slope"] == pytest.approx(2.0)
    assert s["kick_1_zero_crossing"] == pytest.approx(1.0)
    assert s["absent_points"] == 0 and math.isnan(s["absent_slope"])
    missing = [n for n in out.notes if "missing from" in n]
    assert missing == [
        "kick_1: missing from 2 of 5 results",
        "absent: missing from 5 of 5 results",
    ]
    assert any(n.startswith("absent:") and "no line" in n for n in out.notes)


def test_mismatched_positions_are_a_render_error():
    with pytest.raises(RenderError):
        draw([shot(k=1.0)], [1.0, 2.0], ["k"])


def test_summary_output_passes_a_figure_kind_through_unchanged():
    image = Measurement({}, Frame.from_array(np.ones((4, 6))))
    trace = shot()
    for options, results, positions in (
        (ImageGridSummary(), [image, image], [1.0, 2.0]),
        (WaterfallSummary(), [trace, trace], [1.0, 2.0]),
        (AverageSummary(), [image], [None]),
    ):
        raw = summary_definition(options).function(
            results, positions, "x", options, FIGURE
        )
        assert isinstance(raw, Figure)
        out = summary_output(raw)
        assert out.figure is raw and dict(out.scalars) == {} and out.notes == ()
    fit = draw([shot(k=1.0), shot(k=2.0)], [0.0, 1.0], ["k"])
    assert summary_output(fit) is fit


def test_summary_output_owns_its_numbers():
    out = SummaryOutput(figure=None, scalars={"a": 1}, notes=["n"])
    assert out.scalars == {"a": 1.0} and out.notes == ("n",)
    with pytest.raises(TypeError):
        out.scalars["a"] = 2.0


def test_the_registry_and_the_contract_import_without_numpy():
    code = (
        "import sys\n"
        "from geecs_analysis.registry import SummaryOutput, summary_output\n"
        "for name in ('numpy', 'matplotlib', 'scipy'):\n"
        "    assert name not in sys.modules, name\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
