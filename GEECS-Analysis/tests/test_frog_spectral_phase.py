"""The ``frog_spectral_phase`` measure: the legacy fit bit for bit, the weights as a channel."""

from __future__ import annotations

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisRecipe
from geecs_schemas.analysis import FrogSpectralPhaseSpec as V2Spec

from geecs_analysis.algorithms import polynomial_fit as ported
from geecs_analysis.compat.convert import to_v3
from geecs_analysis.compat.v2 import UnsupportedRecipe, analyze_v2, compile_v2
from geecs_analysis.measures.frog_spectral_phase import (
    FrogSpectralPhaseSpec,
    wavelength_nm_to_omega_rad_per_fs,
)
from geecs_analysis.recipe import RecipeError, compile_recipe
from geecs_analysis.registry import measure

LOADING = {
    "data_type": "tsv",
    "delimiter": "\t",
    "x_column": 3,
    "y_column": 5,
    "auxiliary_columns": {"weights": 4},
}


def document(
    *, storage="float64", loading=None, analyzer=None, image=None
) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": "U_FROG_Grenouille-Temporal",
            "analyzer": {"kind": "frog_spectral_phase", **(analyzer or {})},
            "image": {
                "type": "line",
                "storage_dtype": storage,
                "data_loading": LOADING if loading is None else loading,
                "roi": {"x_min": 750.0, "x_max": 850.0},
                "pipeline": ["roi"],
                **(image or {}),
            },
            "scan": {"mode": "per_shot", "file_tail": "_retrieved_lineouts.tsv"},
        }
    )


def recipe(*, storage="float64", loading=None, measure=None) -> AnalysisRecipe:
    """The v3 recipe of `document`, as the v2 converter wrote it."""
    return AnalysisRecipe.model_validate(
        {
            "device": "U_FROG_Grenouille-Temporal",
            "input": {
                "kind": "line",
                "file_tail": "_retrieved_lineouts.tsv",
                "loading": LOADING if loading is None else loading,
                "label": "x vs y",
                "storage_dtype": storage,
            },
            "steps": [{"step": "roi", "bounds": [[750, 850]], "units": "axis"}],
            "measure": {"kind": "frog_spectral_phase", **(measure or {})},
        }
    )


def compiled(**kwargs):
    return compile_recipe(recipe(**kwargs))


def lineout(seed: int, *, gdd: float = 300.0, n: int = 256) -> np.ndarray:
    """A retrieved spectrum over 700-900 nm: (wavelength, phase, intensity)."""
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(700.0, 900.0, n)
    detuning = wavelength_nm_to_omega_rad_per_fs(
        wavelength
    ) - wavelength_nm_to_omega_rad_per_fs(np.asarray([800.0 + seed]))
    phase = (
        0.3 * seed
        - 10.0 * detuning
        + 0.5 * gdd * detuning**2
        + 1000.0 / 6.0 * detuning**3
        + rng.normal(0.0, 0.05, n)
    )
    intensity = np.exp(-(((wavelength - 795.0 - seed) / 15.0) ** 2))
    intensity += rng.normal(0.0, 0.01, n)
    return np.column_stack([wavelength, phase, intensity])


def core_values(result) -> dict:
    """The core's scalars, trace, phase and fit reference."""
    return {
        **{f"scalar_{k}": v for k, v in result.scalars.items()},
        "trace": result.frame.as_trace(),
        "spectral_phase": result.extras["spectral_phase"].data,
        "fit_reference": result.extras["fit_reference"].data,
    }


# ----------------------------------------------------------------- algorithm


def fit_values(module, data: np.ndarray, weighted: bool, threshold) -> dict:
    """One implementation's fit and sign canonicalisations, flattened."""
    fit = module.fit_polynomial(
        y=data[:, 1],
        x=data[:, 0] - 800.0,
        weights=data[:, 2] if weighted else None,
        config=module.PolynomialFitConfig(
            order=3, mask_threshold=threshold, fit_num_points=50
        ),
    )
    values = {
        field: getattr(fit, field)
        for field in (
            "coefficients",
            "fit_x",
            "fit_y",
            "valid_mask",
            "normalized_reference",
        )
    }
    values["order"], values["num_fit_points"] = fit.order, fit.num_fit_points
    for n, (order, sign, epsilon) in enumerate(
        ((2, 1.0, 0.0), (2, -1.0, 0.0), (3, 1.0, 1e9))
    ):
        coefficients, flipped = module.canonicalize_polynomial_sign(
            fit.coefficients, order, sign, epsilon
        )
        values[f"canonical_{n}"] = coefficients
        values[f"flipped_{n}"] = flipped
    return values


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("threshold", [None, 0.2, 0.5])
def test_the_fit_is_the_legacy_fit_bit_for_bit(
    seed, weighted, threshold, expected, request
):
    data = lineout(seed)
    expected(
        f"fit_{request.node.callspec.id}",
        actual=fit_values(ported, data, weighted, threshold),
    )


def test_the_fit_refuses_what_the_legacy_fit_refuses():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    for module in (ported,):
        with pytest.raises(ValueError, match="requires at least 4 valid points; got 3"):
            module.fit_polynomial(y, config=module.PolynomialFitConfig(order=3))
        with pytest.raises(ValueError, match="no positive finite"):
            module.fit_polynomial(
                y, weights=np.zeros(4), config=module.PolynomialFitConfig(order=1)
            )


# ---------------------------------------------------------- the core route


@pytest.mark.parametrize("storage", ["float64", "float32"])
@pytest.mark.parametrize("seed", range(3))
def test_the_core_route_equals_the_legacy_analyzer(storage, seed, expected, request):
    data = lineout(seed)
    result = analyze_v2(data, compiled(storage=storage))
    # The fit is float64 on both routes whatever the storage dtype: the
    # stored input is rounded identically, the fit computed in float64.
    expected(
        f"route_{request.node.callspec.id}",
        actual=core_values(result),
    )
    assert set(result.scalars) == FrogSpectralPhaseSpec().emitted_scalars()


def test_a_sign_flip_matches_the_legacy_flip(expected):
    data = lineout(1, gdd=-300.0)
    result = analyze_v2(data, compiled())
    stored = expected(
        "sign_flip",
        actual=core_values(result),
    )
    assert result.scalars["flipped"] == 1.0 == stored["scalar_flipped"]


@pytest.mark.parametrize(
    "analyzer",
    [
        {"fit_order": 5, "sign_reference_order": None},
        {"mask_threshold": None, "min_points": 20, "fit_num_points": 64},
        {"reference_wavelength_nm": 790.0, "sign_reference": -1.0},
    ],
)
def test_options_match_the_legacy_analyzer(analyzer, expected, request):
    data = lineout(2)
    result = analyze_v2(data, compiled(measure=analyzer))
    expected(
        f"options_{request.node.callspec.id}",
        actual=core_values(result),
    )


def test_the_weights_channel_weights_the_fit_and_follows_the_roi(expected):
    data = lineout(0)
    weighted = analyze_v2(data, compiled())
    # Same trace, no channel declared: the unweighted legacy fit.
    unweighted = compiled(loading={**LOADING, "auxiliary_columns": {}})
    plain = analyze_v2(data[:, :2], unweighted)
    expected(
        "unweighted",
        actual=core_values(plain),
    )
    assert plain.scalars["gdd_fs2"] != weighted.scalars["gdd_fs2"]
    # The channel was cut by the roi: its reference spans the ROI samples.
    inside = (data[:, 0] >= 750.0) & (data[:, 0] <= 850.0)
    assert weighted.extras["fit_reference"].data.size == inside.sum()
    np.testing.assert_array_equal(
        weighted.extras["fit_reference"].axes[0].values, data[inside, 0]
    )


def test_the_channel_is_the_declared_column_not_a_neighbour():
    core = compiled()
    data = lineout(0)
    swapped = data.copy()
    swapped[:, 2] = np.roll(data[:, 2], 7)
    assert dict(analyze_v2(swapped, core).scalars) != dict(
        analyze_v2(data, core).scalars
    )


def test_a_trace_without_its_declared_channel_is_refused():
    with pytest.raises(ValueError, match=r"Nx3 \(x, y, weights\)"):
        analyze_v2(lineout(0)[:, :2], compiled())


def test_too_few_points_above_the_mask_fails_the_shot():
    data = lineout(0)
    data[:, 2] = 0.0
    data[128, 2] = 1.0
    with pytest.raises(ValueError, match="valid points"):
        analyze_v2(data, compiled())


# --------------------------------------------------------------- compiling


def test_compiles_to_the_measure_with_its_parameters_and_channel():
    core = compiled(measure={"fit_order": 4, "mask_threshold": 0.3})
    assert core.analysis.measure == FrogSpectralPhaseSpec(
        fit_order=4, mask_threshold=0.3
    )
    assert core.channels == ("weights",)
    assert core.input_kind == "line"


def test_a_column_no_measure_reads_is_not_a_channel():
    loading = {**LOADING, "auxiliary_columns": {"other": 4}}
    line = AnalysisRecipe.model_validate(
        {
            "device": "Dev",
            "input": {"kind": "line", "loading": loading, "label": "x vs y"},
            "measure": {"kind": "line"},
        }
    )
    assert compile_recipe(line).channels == ()
    assert compiled(loading=loading).channels == ()


def test_interpolation_with_a_channel_is_refused_as_legacy_refused_it():
    interpolated = {
        "pipeline": ["roi", "interpolation"],
        "interpolation": {"num_points": 100, "x_min": 750.0, "x_max": 850.0},
    }
    with pytest.raises(UnsupportedRecipe, match="interpolation"):
        compile_v2(document(image=interpolated))
    no_channel = document(
        image=interpolated, loading={**LOADING, "auxiliary_columns": {}}
    )
    assert compile_v2(no_channel).channels == ()


def test_the_spec_mirrors_the_v2_analyzer_spec():
    assert set(FrogSpectralPhaseSpec.model_fields) == set(V2Spec.model_fields)
    for name, field in V2Spec.model_fields.items():
        if name != "kind":
            assert FrogSpectralPhaseSpec.model_fields[name].default == field.default
    for order in (0, 3, 6):
        assert (
            FrogSpectralPhaseSpec(fit_order=order).emitted_scalars()
            == V2Spec(fit_order=order).emitted_scalars()
        )


def test_converts_to_v3_and_recompiles_identically():
    doc = document(analyzer={"fit_order": 4})
    conversion = to_v3(doc)
    assert conversion.notes == ()
    assert conversion.recipe.input.loading.auxiliary_columns == {"weights": 4}
    recompiled = compile_recipe(conversion.recipe, allow_file_backgrounds=True)
    assert recompiled == compile_v2(doc, allow_file_backgrounds=True)
    data = lineout(0)
    assert dict(analyze_v2(data, recompiled).scalars) == dict(
        analyze_v2(data, compile_v2(doc)).scalars
    )


def _v3(**input_fields) -> AnalysisRecipe:
    return AnalysisRecipe.model_validate(
        {
            "schema_version": 3,
            "device": "Dev",
            "input": {"kind": "line", "loading": LOADING, **input_fields},
            "steps": [{"step": "interpolate", "count": 10}],
            "measure": {"kind": "frog_spectral_phase"},
        }
    )


def test_a_v3_recipe_refuses_a_channel_it_cannot_carry():
    with pytest.raises(RecipeError, match="interpolation"):
        compile_recipe(_v3())
    stitched = _v3(siblings=["Other"]).model_copy(update={"steps": []})
    with pytest.raises(RecipeError, match="stitched"):
        compile_recipe(stitched)


def test_channels_belong_to_trace_measures():
    from geecs_analysis.registry import MeasureSpec

    class Probe(MeasureSpec):
        kind: str = "probe"

    for ndim, channels in (({2}, ("w",)), ({1}, ("w", "w")), ({1}, ("",))):
        with pytest.raises(ValueError, match="channels"):
            measure(Probe, ndim=ndim, channels=channels)
