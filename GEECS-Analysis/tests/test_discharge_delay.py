"""The ``discharge_delay`` measure: peak current, discharge start, laser timing."""

from __future__ import annotations

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisRecipe

from geecs_analysis.algorithms.discharge_delay import (
    discharge_delay,
    rising_crossing_before_peak,
)
from geecs_analysis.engine.compiled import analyze_compiled
from geecs_analysis.recipe import compile_recipe

DT_S = 5e-11
N = 40_000  # 2 µs at 50 ps


def recipe(*, photodiode: bool = True, measure=None) -> AnalysisRecipe:
    """The LeCroy recipe: current on trace 3, photodiode on trace 1, axis in s."""
    loading = {"data_type": "tek_scope_hdf5", "trace_index": 3}
    if photodiode:
        loading["auxiliary_columns"] = {"photodiode": 1}
    return AnalysisRecipe.model_validate(
        {
            "device": "SCOPE-AA-LeCroy8254M",
            "input": {"kind": "line", "loading": loading, "x_unit": "s"},
            "measure": {"kind": "discharge_delay", **(measure or {})},
        }
    )


def scope_shot(
    *, peak_v=50.0, rise_start_s=0.5e-6, peak_s=0.9e-6, laser_s=1.2e-6, noise=0.0
):
    """(time s, current V, photodiode V): a half-sine current and a Gaussian laser."""
    t = np.arange(N) * DT_S
    half = peak_s - rise_start_s
    current = np.where(
        (t > rise_start_s) & (t < rise_start_s + 2 * half),
        peak_v * np.sin(np.pi * (t - rise_start_s) / (2 * half)),
        0.0,
    )
    pd = 5.0 * np.exp(-0.5 * ((t - laser_s) / 2e-9) ** 2)
    rng = np.random.default_rng(0)
    current = current + noise * rng.standard_normal(N)
    return t, current, pd


def table(t, current, pd=None) -> np.ndarray:
    return np.column_stack([t, current] if pd is None else [t, current, pd])


# --------------------------------------------------------------- algorithm


def test_the_crossing_is_interpolated_between_samples():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    v = np.array([0.0, 0.0, 4.0, 10.0, 2.0])
    # 10 % of 10 is 1: a quarter of the way from 0 (t=1) to 4 (t=2).
    assert rising_crossing_before_peak(t, v, 0.1) == pytest.approx(1.25)
    # A trace already above the level before its peak has no crossing.
    assert np.isnan(rising_crossing_before_peak(t, v + 5.0, 0.1))


def test_the_crossing_is_the_last_one_before_the_peak():
    t = np.arange(8.0)
    v = np.array([0.0, 5.0, 0.0, 0.0, 2.0, 6.0, 10.0, 0.0])  # an early bump at t=1
    assert rising_crossing_before_peak(t, v, 0.1) == pytest.approx(3.5)


def test_a_half_sine_and_a_gaussian():
    t_ns = np.linspace(0.0, 1000.0, 100_001)
    current = np.where(t_ns < 400.0, 500.0 * np.sin(np.pi * t_ns / 400.0), 0.0)
    pd = np.exp(-0.5 * ((t_ns - 300.0) / 2.0) ** 2)
    r = discharge_delay(t_ns, current, t_ns, pd, start_fraction=0.1, laser_fraction=0.5)
    start = 400.0 / np.pi * np.arcsin(0.1)
    laser = 300.0 - 2.0 * np.sqrt(2 * np.log(2))
    assert r.peak_current_A == pytest.approx(500.0)
    assert r.peak_time_ns == pytest.approx(200.0)
    assert r.start_time_ns == pytest.approx(start, abs=1e-3)
    assert r.laser_time_ns == pytest.approx(laser, abs=1e-3)
    assert r.current_at_laser_A == pytest.approx(
        500.0 * np.sin(np.pi * laser / 400.0), rel=1e-6
    )
    assert r.laser_minus_start_ns == pytest.approx(laser - start, abs=1e-3)
    assert r.laser_minus_peak_ns == pytest.approx(laser - 200.0, abs=1e-3)


def test_no_photodiode_leaves_the_laser_quantities_nan():
    t_ns = np.linspace(0.0, 10.0, 11)
    r = discharge_delay(t_ns, np.sin(t_ns / 3.0))
    assert np.isfinite(r.peak_current_A) and np.isfinite(r.start_time_ns)
    for name in (
        "laser_time_ns",
        "current_at_laser_A",
        "laser_minus_start_ns",
        "laser_minus_peak_ns",
    ):
        assert np.isnan(getattr(r, name)), name


def test_a_laser_outside_the_current_trace_has_no_current():
    t_ns = np.linspace(0.0, 10.0, 11)
    pd_t = t_ns + 100.0
    pd = np.exp(-0.5 * (pd_t - 105.0) ** 2)
    r = discharge_delay(t_ns, np.sin(t_ns / 3.0), pd_t, pd)
    assert np.isfinite(r.laser_time_ns) and np.isnan(r.current_at_laser_A)


# ----------------------------------------------------------------- measure


def test_offsets_shift_each_channel_and_amps_per_volt_scales():
    t, current, pd = scope_shot()
    base = analyze_compiled(
        table(t, current, pd),
        compile_recipe(
            recipe(
                measure={
                    "current_offset_ns": 0.0,
                    "photodiode_offset_ns": 0.0,
                    "amps_per_volt": 1.0,
                }
            )
        ),
    ).scalars
    nominal = analyze_compiled(table(t, current, pd), compile_recipe(recipe())).scalars
    assert base["peak_time_ns"] == pytest.approx(900.0, abs=0.05)
    assert base["laser_time_ns"] == pytest.approx(
        1200.0 - 2.0 * np.sqrt(2 * np.log(2)), abs=0.01
    )
    # Defaults: current +100 ns, photodiode +15 ns, 10 A/V.
    assert nominal["peak_current_A"] == pytest.approx(10 * base["peak_current_A"])
    assert nominal["peak_time_ns"] == pytest.approx(base["peak_time_ns"] + 100.0)
    assert nominal["start_time_ns"] == pytest.approx(base["start_time_ns"] + 100.0)
    assert nominal["laser_time_ns"] == pytest.approx(base["laser_time_ns"] + 15.0)
    assert nominal["laser_minus_peak_ns"] == pytest.approx(
        base["laser_minus_peak_ns"] - 85.0
    )


def test_the_measure_reports_the_current_frame_markers_and_photodiode():
    t, current, pd = scope_shot()
    result = analyze_compiled(table(t, current, pd), compile_recipe(recipe()))
    assert (
        set(result.scalars)
        == compile_recipe(recipe()).analysis.measure.emitted_scalars()
    )
    assert result.frame.unit == "A" and result.frame.axes[0].unit == "ns"
    assert result.frame.axes[0].values[0] == pytest.approx(100.0)
    assert result.extras["photodiode"].axes[0].values[0] == pytest.approx(15.0)
    assert {o.id for o in result.overlays} == {"peak", "start", "laser"}


def test_without_a_photodiode_channel_the_laser_is_nan():
    t, current, _ = scope_shot()
    compiled = compile_recipe(recipe(photodiode=False))
    assert compiled.channels == ()
    result = analyze_compiled(table(t, current), compiled)
    assert np.isnan(result.scalars["laser_time_ns"])
    assert np.isfinite(result.scalars["peak_current_A"])
    assert "No photodiode channel" in result.notes
    assert {o.id for o in result.overlays} == {"peak", "start"}


def test_the_median_window_smooths_spiky_noise():
    t, current, pd = scope_shot(noise=2.0)
    raw = analyze_compiled(table(t, current, pd), compile_recipe(recipe())).scalars
    smooth = analyze_compiled(
        table(t, current, pd),
        compile_recipe(recipe(measure={"median_window_ns": 5.0})),
    ).scalars
    assert abs(smooth["peak_current_A"] - 500.0) < abs(raw["peak_current_A"] - 500.0)


def test_an_axis_without_a_time_unit_is_refused():
    t, current, pd = scope_shot()
    doc = recipe().model_dump()
    doc["input"]["x_unit"] = ""
    with pytest.raises(ValueError, match="time axis"):
        analyze_compiled(
            table(t, current, pd),
            compile_recipe(AnalysisRecipe.model_validate(doc)),
        )


def test_a_photodiode_below_the_threshold_times_no_laser():
    t_ns = np.linspace(0.0, 10.0, 11)
    pd = 0.1 * np.exp(-0.5 * (t_ns - 5.0) ** 2)
    gated = discharge_delay(t_ns, np.sin(t_ns / 3.0), t_ns, pd, laser_min_amplitude=0.5)
    assert np.isnan(gated.laser_time_ns) and np.isnan(gated.current_at_laser_A)
    ungated = discharge_delay(t_ns, np.sin(t_ns / 3.0), t_ns, pd)
    assert np.isfinite(ungated.laser_time_ns)


def test_the_measure_gates_on_photodiode_min_v():
    t, current, pd = scope_shot()
    noise = 0.1 * pd / pd.max()  # a 0.1 V photodiode: no laser
    result = analyze_compiled(table(t, current, noise), compile_recipe(recipe()))
    assert np.isnan(result.scalars["laser_time_ns"])
    assert np.isfinite(result.scalars["peak_current_A"])
    assert any("Photodiode below 1.0 V" in note for note in result.notes)
    ungated = analyze_compiled(
        table(t, current, noise),
        compile_recipe(recipe(measure={"photodiode_min_v": None})),
    )
    assert np.isfinite(ungated.scalars["laser_time_ns"])


def test_a_photodiode_with_no_crossing_is_not_reported_as_below_threshold():
    t, current, pd = scope_shot()
    offset = pd + 6.0  # 6 V DC under a 5 V pulse: never below half its 11 V peak
    result = analyze_compiled(table(t, current, offset), compile_recipe(recipe()))
    assert np.isnan(result.scalars["laser_time_ns"])
    assert not any("Photodiode below" in note for note in result.notes)
    assert any("never below laser_fraction" in note for note in result.notes)


def test_a_microsecond_axis_reads_with_either_mu():
    t, current, pd = scope_shot()
    base = analyze_compiled(table(t, current, pd), compile_recipe(recipe())).scalars
    for unit in ("µs", "μs"):
        doc = recipe().model_dump()
        doc["input"]["x_unit"] = unit
        doc["input"]["x_scale"] = 1e6
        got = analyze_compiled(
            table(t, current, pd), compile_recipe(AnalysisRecipe.model_validate(doc))
        ).scalars
        assert got["laser_minus_peak_ns"] == pytest.approx(base["laser_minus_peak_ns"])
