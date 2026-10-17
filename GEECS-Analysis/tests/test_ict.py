"""The ``ict`` measure: the legacy algorithm bit for bit, the v2 route, and failures made visible."""

from __future__ import annotations

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisRecipe

from geecs_analysis.algorithms.ict import apply_ict_analysis
from geecs_analysis.engine.compiled import CompiledRecipe, analyze_compiled
from geecs_analysis.measures.ict import SCALARS, IctSpec
from geecs_analysis.recipe import compile_recipe

DT = 4e-9


def scope_trace(seed: int, *, n: int = 2500, pulse_at: int | None = None) -> np.ndarray:
    """An ICT-like voltage trace: a negative pulse, a sinusoidal pickup, noise."""
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    at = pulse_at if pulse_at is not None else int(rng.integers(300, n - 800))
    pulse = -0.05 * rng.uniform(0.5, 2.0) * np.exp(-(((i - at) / 25.0) ** 2))
    pickup = 0.004 * np.sin(2 * np.pi * i / rng.uniform(150, 400) + rng.uniform(0, 6))
    return pulse + pickup + rng.normal(0, 0.0015, n) + 0.0005


def recipe(dt: float | None = DT) -> CompiledRecipe:
    measure = {"kind": "ict", "calibration_factor": 0.2, "butterworth_crit_f": 0.1}
    if dt is not None:
        measure["dt"] = dt
    return compile_recipe(
        AnalysisRecipe.model_validate(
            {
                "schema_version": 3,
                "device": "U_BCaveICT",
                "input": {
                    "kind": "line",
                    "loading": {"data_type": "npy"},
                    "label": "volts (V) vs time (s)",
                },
                "measure": measure,
            }
        )
    )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("pulse_at", [None, 40, 2300])
def test_the_algorithm_is_the_legacy_one_bit_for_bit(seed, pulse_at, expected, request):
    data = scope_trace(seed, pulse_at=pulse_at)
    args = dict(
        dt=DT, butterworth_order=2, butterworth_crit_f=0.1, calibration_factor=0.2
    )
    expected(
        f"algorithm_{request.node.callspec.id}",
        actual=apply_ict_analysis(data, **args),
    )


@pytest.mark.parametrize("dt", [DT, None])
@pytest.mark.parametrize("seed", range(4))
def test_the_core_route_matches_the_legacy_ict_analyzer(seed, dt, expected, request):
    """Same trace and config through ICT1DAnalyzer and through the core."""

    time = np.arange(2500) * DT
    raw = np.column_stack([time, scope_trace(seed)])
    legacy = expected(
        f"v2_route_{request.node.callspec.id}",
    )
    core = analyze_compiled(raw, recipe(dt=dt))
    # Legacy low-pass-filtered the float32 stored trace in float32; the core
    # filters the same samples at float64. Stated tolerance, not equality.
    assert set(core.scalars) == set(legacy)
    for key, value in legacy.items():
        assert core.scalars[key] == pytest.approx(float(value), rel=1e-6), key
    assert set(core.scalars) == IctSpec().emitted_scalars() == set(SCALARS)
    assert core.notes == ()


def test_a_trace_the_algorithm_cannot_analyze_is_nan_with_a_note():
    """Legacy wrote 0 pC here; the core keeps the failure visible."""
    raw = np.column_stack([np.arange(5) * DT, -np.ones(5)])
    result = analyze_compiled(raw, recipe())
    assert all(np.isnan(v) for v in result.scalars.values())
    assert any(n.startswith("ICT analysis failed") for n in result.notes)
