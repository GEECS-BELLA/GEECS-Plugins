"""The ``frog`` measure: a host-bound retriever, legacy parity, the v2 route and the pool."""

from __future__ import annotations

import importlib
import pickle
import textwrap
from dataclasses import dataclass

import numpy as np
import pytest
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import AnalysisDiagnostic

from geecs_analysis.compat.convert import to_v3
from geecs_analysis.compat.v2 import analyze_v2, compile_v2
from geecs_analysis.compat.v2_run import ShotGroup, run_units
from geecs_analysis.measures.frog import SCALARS, FrogSpec
from geecs_analysis.recipe import compile_recipe
from geecs_analysis.run import analyze
from geecs_analysis.specs import Analysis

PARAMETERS = dict(
    delt=0.966601,
    dellam=-0.0787696,
    lam0=400.0,
    N=512,
    target_error=0.0001,
    max_time_seconds=15.0,
    max_iterations=300,
    noise_subtype=3,
    noise_rad=2.0,
)


@dataclass
class FakeResult:
    """The attributes of ``FrogRetrievalResult`` the measure reads."""

    temporal_fwhm: float
    spectral_fwhm: float
    frog_error: float
    num_iterations: int
    retrieved_trace: np.ndarray
    time: np.ndarray
    temporal_intensity: np.ndarray
    temporal_phase: np.ndarray
    wavelength: np.ndarray
    spectral_intensity: np.ndarray
    spectral_phase: np.ndarray

    @property
    def tw_per_joule(self) -> float:
        dt = self.time[1] - self.time[0]
        return 1000.0 / (np.sum(self.temporal_intensity) * dt)


def fake_result(trace: np.ndarray, iterations: int = 7) -> FakeResult:
    """A deterministic function of the trace, standing in for FROG.dll."""
    total = float(trace.sum())
    time = np.linspace(-50.0, 50.0, 16)
    wave = np.linspace(390.0, 410.0, 24)
    return FakeResult(
        temporal_fwhm=total % 97.0 + 1.0,
        spectral_fwhm=float(trace.max()) / 3.0,
        frog_error=1.0 / (1.0 + total),
        num_iterations=iterations,
        retrieved_trace=trace[:8, :8] * 0.5,
        time=time,
        temporal_intensity=np.exp(-((time / 20.0) ** 2)),
        temporal_phase=time * 0.01,
        wavelength=wave,
        spectral_intensity=np.exp(-(((wave - 400.0) / 4.0) ** 2)),
        spectral_phase=(wave - 400.0) ** 2 * 0.001,
    )


class FakeRetriever:
    """Records every call; returns :func:`fake_result`."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, dict]] = []

    def retrieve_pulse(self, trace, **parameters):
        self.calls.append((np.array(trace), parameters))
        return fake_result(trace)


def trace(seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 400, size=(12, 10)).astype(np.uint16)


def diagnostic(**image) -> AnalysisDiagnostic:
    return AnalysisDiagnostic.model_validate(
        {
            "name": "U_FROG_Grenouille-Temporal",
            "analyzer": {"kind": "frog_retrieval", **PARAMETERS},
            "image": {"type": "camera", **image},
        }
    )


def test_measure_calls_the_bound_retriever_and_packages_its_result():
    retriever = FakeRetriever()
    data = trace()
    result = analyze(
        Frame.from_array(data),
        Analysis(measure=FrogSpec(**PARAMETERS)),
        inputs={"frog": retriever},
    )
    (sent, parameters), *rest = retriever.calls
    assert not rest
    assert sent.dtype == np.float64
    np.testing.assert_array_equal(sent, data)
    assert parameters == PARAMETERS
    expected = fake_result(data.astype(np.float64))
    assert result.scalars == {
        "temporal_fwhm": expected.temporal_fwhm,
        "spectral_fwhm": expected.spectral_fwhm,
        "frog_error": expected.frog_error,
        "frog_iterations": 7.0,
        "tw_per_joule": expected.tw_per_joule,
    }
    assert set(result.scalars) == FrogSpec().emitted_scalars() == set(SCALARS)
    np.testing.assert_array_equal(result.frame.data, expected.retrieved_trace)
    projections = {o.id: o for o in result.overlays}
    np.testing.assert_array_equal(
        projections["projection_x"].frame.data, expected.retrieved_trace.sum(axis=0)
    )
    np.testing.assert_array_equal(
        projections["projection_y"].frame.data, expected.retrieved_trace.sum(axis=1)
    )
    assert list(result.extras) == [
        "temporal_intensity",
        "temporal_phase",
        "spectral_intensity",
        "spectral_phase",
    ]
    temporal = result.extras["temporal_intensity"]
    np.testing.assert_array_equal(temporal.axes[0].values, expected.time)
    assert (temporal.axes[0].label, temporal.axes[0].unit) == ("time", "fs")
    spectral = result.extras["spectral_phase"]
    np.testing.assert_array_equal(spectral.data, expected.spectral_phase)
    assert (spectral.axes[0].label, spectral.axes[0].unit) == ("wavelength", "nm")


def test_a_missing_retriever_is_an_error_naming_the_service():
    with pytest.raises(ValueError, match="Missing service: frog"):
        analyze(Frame.from_array(trace()), Analysis(measure=FrogSpec()))
    with pytest.raises(TypeError, match="collaborator"):
        analyze(
            Frame.from_array(trace()),
            Analysis(measure=FrogSpec()),
            inputs={"frog": Frame.from_array(trace())},
        )


def test_v2_frog_diagnostic_compiles_to_the_frog_measure_with_its_parameters():
    recipe = compile_v2(diagnostic(pipeline=[]))
    assert recipe.analysis.measure == FrogSpec(**PARAMETERS)
    assert recipe.analysis.steps == ()


def test_the_v2_route_matches_the_legacy_grenouille_analyzer(monkeypatch):
    """The legacy analyzer, on the same fake DLL, is the oracle."""
    from geecs_schemas.analysis import FrogRetrievalSpec
    from image_analysis.algorithms.frog_dll_retrieval import FrogDllRetrieval
    from image_analysis.analyzers.grenouille_analyzer import GrenouilleAnalyzer

    image = {
        "bit_depth": 16,
        "background": {"method": "constant", "constant_level": 1.0},
        "thresholding": {
            "method": "constant",
            "value": 0.0,
            "mode": "to_zero",
            "invert": False,
        },
        "filtering": {"median_kernel_size": 5},
        "pipeline": ["background", "filtering", "thresholding"],
    }
    document = diagnostic(**image)
    legacy_retriever = FakeRetriever()
    monkeypatch.setattr(
        FrogDllRetrieval, "from_config", classmethod(lambda cls: legacy_retriever)
    )
    legacy = GrenouilleAnalyzer(
        document.image, spec=FrogRetrievalSpec(**PARAMETERS)
    ).analyze_image(trace())

    retriever = FakeRetriever()
    result = analyze_v2(trace(), compile_v2(document), inputs={"frog": retriever})

    np.testing.assert_array_equal(retriever.calls[0][0], legacy_retriever.calls[0][0])
    assert retriever.calls[0][1] == legacy_retriever.calls[0][1]
    assert result.scalars == {k: float(v) for k, v in legacy.scalars.items()}
    np.testing.assert_array_equal(result.frame.data, legacy.processed_image)
    projections = {o.id: o.frame.data for o in result.overlays}
    np.testing.assert_array_equal(
        projections["projection_x"], legacy.render_data["horizontal_projection"]
    )
    np.testing.assert_array_equal(
        projections["projection_y"], legacy.render_data["vertical_projection"]
    )


def test_a_frog_diagnostic_converts_to_a_v3_recipe_that_compiles_identically():
    document = diagnostic(pipeline=[])
    converted = to_v3(document).recipe
    assert converted.measure.model_dump()["kind"] == "frog"
    assert compile_recipe(converted).analysis == compile_v2(document).analysis


def test_a_frog_measurement_pickles_with_its_extras():
    result = analyze(
        Frame.from_array(trace()),
        Analysis(measure=FrogSpec()),
        inputs={"frog": FakeRetriever()},
    )
    copy = pickle.loads(pickle.dumps(result))
    assert copy.scalars == result.scalars
    assert list(copy.extras) == list(result.extras)
    for key, frame in result.extras.items():
        np.testing.assert_array_equal(copy.extras[key].data, frame.data)
        np.testing.assert_array_equal(
            copy.extras[key].axes[0].values, frame.axes[0].values
        )


# A spawned worker imports the retriever's module by name (see test_v2_pool).
HELPER = textwrap.dedent(
    '''
    """A picklable retriever and loader for the pooled FROG run."""
    import numpy as np


    class Result:
        def __init__(self, trace):
            total = float(trace.sum())
            self.temporal_fwhm = total % 97.0
            self.spectral_fwhm = float(trace.max())
            self.frog_error = 1.0 / (1.0 + total)
            self.num_iterations = 3
            self.retrieved_trace = trace[:4, :4] * 2.0
            self.time = np.linspace(-1.0, 1.0, 5)
            self.temporal_intensity = np.ones(5)
            self.temporal_phase = np.zeros(5)
            self.wavelength = np.linspace(399.0, 401.0, 6)
            self.spectral_intensity = np.ones(6)
            self.spectral_phase = np.zeros(6)
            self.tw_per_joule = 1.0


    class Retriever:
        def retrieve_pulse(self, trace, **parameters):
            return Result(trace)


    def load(shot):
        return np.full((6, 6), shot, dtype=np.uint16)
    '''
)


def test_a_pooled_frog_run_equals_the_serial_run(tmp_path, monkeypatch):
    (tmp_path / "frog_pool_helper.py").write_text(HELPER)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    helper = importlib.import_module("frog_pool_helper")
    recipe = compile_v2(diagnostic(pipeline=[]))
    groups = [ShotGroup(n, (n,)) for n in range(1, 9)]
    inputs = {"frog": helper.Retriever()}

    def outcomes(workers):
        return [
            (o.group.key, dict(o.measurement.scalars), o.measurement.frame.data)
            for o in run_units(
                recipe, groups, helper.load, inputs=inputs, workers=workers
            )
        ]

    serial, pooled = outcomes(1), outcomes(2)
    assert [s[:2] for s in serial] == [p[:2] for p in pooled]
    for (_, _, a), (_, _, b) in zip(serial, pooled, strict=True):
        np.testing.assert_array_equal(a, b)
    monkeypatch.delitem(importlib.sys.modules, "frog_pool_helper", raising=False)


def test_frog_spec_mirrors_the_v2_spec():
    """Same fields and defaults, and no bound the v2 schema lacks: every v2
    config the legacy route ran must compile (target_error 0 = never stop)."""
    from geecs_schemas.analysis import FrogRetrievalSpec

    v2 = FrogRetrievalSpec.model_fields
    core = FrogSpec.model_fields
    assert set(core) - {"kind"} == set(v2) - {"kind"}
    for name in set(core) - {"kind"}:
        assert core[name].default == v2[name].default, name
    edge = {"target_error": 0.0, "max_time_seconds": 0.0, "max_iterations": 0}
    document = AnalysisDiagnostic.model_validate(
        {
            "name": "Dev",
            "analyzer": {"kind": "frog_retrieval", **edge},
            "image": {"type": "camera"},
        }
    )
    assert compile_v2(document).analysis.measure == FrogSpec(**edge)
