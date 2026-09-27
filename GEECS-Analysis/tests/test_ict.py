"""The ``ict`` measure: the legacy algorithm bit for bit, the v2 route, and failures made visible."""

from __future__ import annotations

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisDiagnostic, IctAnalyzerSpec

from geecs_analysis.algorithms.ict import apply_ict_analysis
from geecs_analysis.compat.convert import to_v3
from geecs_analysis.compat.v2 import analyze_v2, compile_v2
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


def diagnostic(dt: float | None = DT, **image) -> AnalysisDiagnostic:
    analyzer = {"kind": "ict", "calibration_factor": 0.2, "butterworth_crit_f": 0.1}
    if dt is not None:
        analyzer["dt"] = dt
    return AnalysisDiagnostic.model_validate(
        {
            "name": "U_BCaveICT",
            "analyzer": analyzer,
            "image": {
                "type": "line",
                "data_loading": {"data_type": "npy"},
                "storage_dtype": "float32",
                "label": "volts (V) vs time (s)",
                "pipeline": [],
                **image,
            },
        }
    )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("pulse_at", [None, 40, 2300])
def test_the_algorithm_is_the_legacy_one_bit_for_bit(seed, pulse_at):
    from image_analysis.algorithms.ict_algorithms import (
        apply_ict_analysis as legacy,
    )

    data = scope_trace(seed, pulse_at=pulse_at)
    args = dict(
        dt=DT, butterworth_order=2, butterworth_crit_f=0.1, calibration_factor=0.2
    )
    assert apply_ict_analysis(data, **args) == legacy(data, **args)


@pytest.mark.parametrize("dt", [DT, None])
@pytest.mark.parametrize("seed", range(4))
def test_the_v2_route_matches_the_legacy_ict_analyzer(seed, dt):
    """Same trace and config through ICT1DAnalyzer and through the core."""
    from image_analysis.analyzers.ict_1d_analyzer import ICT1DAnalyzer

    document = diagnostic(dt=dt)
    time = np.arange(2500) * DT
    raw = np.column_stack([time, scope_trace(seed)])
    legacy = ICT1DAnalyzer(document.image, spec=document.analyzer).analyze_image(raw)
    core = analyze_v2(raw, compile_v2(document))
    # Legacy low-pass-filtered the float32 stored trace in float32; the core
    # filters the same samples at float64. Stated tolerance, not equality.
    assert set(core.scalars) == set(legacy.scalars)
    for key, value in legacy.scalars.items():
        assert core.scalars[key] == pytest.approx(float(value), rel=1e-6), key
    assert set(core.scalars) == IctSpec().emitted_scalars() == set(SCALARS)
    assert core.notes == ()


def test_a_trace_the_algorithm_cannot_analyze_is_nan_with_a_note():
    """Legacy wrote 0 pC here; the core keeps the failure visible."""
    raw = np.column_stack([np.arange(5) * DT, -np.ones(5)])
    result = analyze_v2(raw, compile_v2(diagnostic()))
    assert all(np.isnan(v) for v in result.scalars.values())
    assert any(n.startswith("ICT analysis failed") for n in result.notes)


def test_ict_spec_mirrors_the_v2_spec():
    v2, core = IctAnalyzerSpec.model_fields, IctSpec.model_fields
    assert set(core) - {"kind"} == set(v2) - {"kind"}
    for name in set(core) - {"kind"}:
        assert core[name].default == v2[name].default, name
        assert core[name].metadata == v2[name].metadata, name


def test_an_ict_diagnostic_converts_to_a_recipe_that_compiles_identically():
    document = diagnostic()
    recipe = to_v3(document).recipe
    assert recipe.measure.model_dump()["kind"] == "ict"
    assert compile_recipe(recipe) == compile_v2(document)
