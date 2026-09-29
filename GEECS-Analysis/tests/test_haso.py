"""The ``haso`` measure: a host-bound WaveKit engine, pixel handling, packaging, the pool."""

from __future__ import annotations

import importlib
import pickle
import textwrap
from dataclasses import dataclass

import numpy as np
import pytest
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import AnalysisRecipe
from pydantic import ValidationError

from geecs_analysis.compat.v2_run import ShotGroup, run_units
from geecs_analysis.measures.haso import (
    EXTRAS,
    SCALARS,
    SHOT_STORE,
    HasoFilters,
    HasoMask,
    HasoSpec,
    sensor_pixels,
)
from geecs_analysis.recipe import compile_recipe
from geecs_analysis.registry import measure_definition
from geecs_analysis.run import analyze
from geecs_analysis.specs import Analysis

SENSOR = "WFS_HASO4_LIFT_680_8244_gain_enabled.dat"
ROWS, COLS = 6, 8


@dataclass
class FakeResult:
    """The attributes of ``HasoWaveKitResult`` the measure reads."""

    processed_phase: np.ndarray
    raw_phase: np.ndarray
    intensity: np.ndarray
    slopes_x: np.ndarray
    slopes_y: np.ndarray
    pupil: np.ndarray


def fake_result(pixels: np.ndarray, mask) -> FakeResult:
    """A deterministic function of the pixels, standing in for WaveKit.

    The slopes grid is ``(ROWS, COLS)`` whatever the pixel frame's shape,
    as the SDK's is; the pupil is the mask (or everything); the phase is
    NaN outside it, with one NaN inside to exercise the finite filter.
    """
    total = float(pixels.astype(np.float64).sum())
    grid = np.arange(ROWS * COLS, dtype=np.float32).reshape(ROWS, COLS)
    pupil = np.zeros((ROWS, COLS), dtype=bool)
    if mask is None:
        pupil[:] = True
    else:
        top, bottom, left, right = mask
        pupil[top:bottom, left:right] = True
    processed = np.where(pupil, grid * 0.01 + total % 7.0, np.nan).astype(np.float32)
    processed[np.argwhere(pupil)[0][0], np.argwhere(pupil)[0][1]] = np.nan
    return FakeResult(
        processed_phase=processed,
        raw_phase=(grid * 0.02).astype(np.float32),
        intensity=(grid + total).astype(np.float32),
        slopes_x=(grid * 0.5).astype(np.float32),
        slopes_y=(grid * -0.5).astype(np.float32),
        pupil=pupil,
    )


class FakeEngine:
    """Records every call; returns :func:`fake_result`."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, dict]] = []

    def compute(self, pixels, **parameters):
        self.calls.append((np.array(pixels), parameters))
        return fake_result(pixels, parameters["mask"])


def pixels(seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(10, 12)).astype(np.uint16)


def spec(**overrides) -> HasoSpec:
    return HasoSpec(**{"sensor_config": SENSOR, **overrides})


def test_the_spec_keeps_the_legacy_defaults_and_names_the_sensor_by_file():
    s = spec()
    assert s.wavelength_nm == 800.0
    assert s.start_subpupil == (87, 64) and s.zonal_prefs == (100, 500, 1e-6)
    assert s.filters.flags() == (True, True, True, True, True, False)
    assert s.mask is None
    assert s.emitted_scalars() == frozenset(SCALARS)
    definition = measure_definition(s)
    assert definition.service == "haso"
    assert definition.sidecar is None and definition.shot_store == SHOT_STORE
    assert definition.ndim == frozenset({2})


@pytest.mark.parametrize(
    "name", ["/abs/sensor.dat", "dir/sensor.dat", "..", "a\\b.dat"]
)
def test_a_sensor_config_path_is_refused(name):
    with pytest.raises(ValidationError, match="file name"):
        spec(sensor_config=name)


def test_mask_and_preference_bounds_are_validated():
    with pytest.raises(ValidationError, match="top < bottom"):
        HasoMask(top=5, bottom=5, left=0, right=3)
    with pytest.raises(ValidationError, match="non-negative"):
        spec(start_subpupil=(-1, 4))
    with pytest.raises(ValidationError, match="positive"):
        spec(zonal_prefs=(0, 500, 1e-6))
    with pytest.raises(ValidationError):
        spec(unknown=1)


def test_sensor_pixels_round_and_clip_a_processed_frame():
    frame = Frame.from_array(np.array([[-3.0, 0.4], [0.5, 70000.2]]))
    out = sensor_pixels(frame)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, [[0, 0], [0, 65535]])


def test_measure_calls_the_bound_engine_and_packages_its_result():
    engine = FakeEngine()
    data = pixels()
    mask = HasoMask(top=1, bottom=5, left=2, right=7)
    result = analyze(
        Frame.from_array(data),
        Analysis(measure=spec(mask=mask, wavelength_nm=532.0)),
        inputs={"haso": engine},
    )
    (sent, parameters), *rest = engine.calls
    assert not rest
    assert sent.dtype == np.uint16
    np.testing.assert_array_equal(sent, data)
    assert parameters == {
        "sensor_config": SENSOR,
        "mask": (1, 5, 2, 7),
        "filters": (True, True, True, True, True, False),
        "wavelength_nm": 532.0,
        "start_subpupil": (87, 64),
        "zonal_prefs": (100, 500, 1e-6),
    }
    expected = fake_result(data, (1, 5, 2, 7))
    np.testing.assert_array_equal(result.frame.data, expected.processed_phase)
    assert result.frame.data.dtype == np.float64
    assert (result.frame.unit, result.frame.label) == ("um", "phase")
    inside = expected.processed_phase.astype(np.float64)[expected.pupil]
    inside = inside[np.isfinite(inside)]
    assert result.scalars == {
        "phase_rms": float(inside.std()),
        "phase_pv": float(inside.max() - inside.min()),
    }
    assert set(result.scalars) == set(SCALARS)
    assert tuple(result.extras) == EXTRAS
    np.testing.assert_array_equal(result.extras["raw_phase"].data, expected.raw_phase)
    np.testing.assert_array_equal(result.extras["intensity"].data, expected.intensity)
    np.testing.assert_array_equal(result.extras["slopes_x"].data, expected.slopes_x)
    np.testing.assert_array_equal(result.extras["slopes_y"].data, expected.slopes_y)
    np.testing.assert_array_equal(result.extras["pupil"].data, expected.pupil)
    assert result.extras["slopes_x"].unit == "mrad"
    assert result.overlays == () and result.notes == ()


def test_no_mask_keeps_the_sensor_pupil_and_an_empty_pupil_is_nan():
    engine = FakeEngine()
    result = analyze(
        Frame.from_array(pixels()), Analysis(measure=spec()), inputs={"haso": engine}
    )
    assert engine.calls[0][1]["mask"] is None
    assert result.extras["pupil"].data.all()

    class Empty:
        def compute(self, pixels, **parameters):
            out = fake_result(pixels, None)
            out.pupil = np.zeros_like(out.pupil)
            return out

    result = analyze(
        Frame.from_array(pixels()), Analysis(measure=spec()), inputs={"haso": Empty()}
    )
    assert all(np.isnan(v) for v in result.scalars.values())
    assert result.notes == ("Nonfinite scalar: phase_rms", "Nonfinite scalar: phase_pv")


def test_a_missing_engine_is_an_error_naming_the_service():
    with pytest.raises(ValueError, match="Missing service: haso"):
        analyze(Frame.from_array(pixels()), Analysis(measure=spec()))
    with pytest.raises(TypeError, match="collaborator"):
        analyze(
            Frame.from_array(pixels()),
            Analysis(measure=spec()),
            inputs={"haso": Frame.from_array(pixels())},
        )


def test_a_v3_recipe_binds_the_measure_with_its_parameters():
    recipe = AnalysisRecipe.model_validate(
        {
            "device": "U_HasoLift",
            "input": {"kind": "camera", "file_tail": ".himg", "format": "device_hdf5"},
            "measure": {
                "kind": "haso",
                "sensor_config": SENSOR,
                "mask": {"top": 125, "bottom": 300, "left": 10, "right": 670},
                "filters": {"others": True},
            },
        }
    )
    compiled = compile_recipe(recipe)
    assert compiled.analysis.measure == spec(
        mask=HasoMask(top=125, bottom=300, left=10, right=670),
        filters=HasoFilters(others=True),
    )
    line = AnalysisRecipe.model_validate(
        {
            "device": "Dev",
            "input": {"kind": "line", "loading": {"data_type": "tsv"}},
            "measure": {"kind": "haso", "sensor_config": SENSOR},
        }
    )
    with pytest.raises(ValueError, match="does not measure line frames"):
        compile_recipe(line)


def test_a_haso_measurement_pickles_with_its_extras():
    result = analyze(
        Frame.from_array(pixels()),
        Analysis(measure=spec()),
        inputs={"haso": FakeEngine()},
    )
    copy = pickle.loads(pickle.dumps(result))
    assert copy.scalars == result.scalars
    assert tuple(copy.extras) == EXTRAS
    for key, frame in result.extras.items():
        np.testing.assert_array_equal(copy.extras[key].data, frame.data)


# A spawned worker imports the engine's module by name (see test_v2_pool).
HELPER = textwrap.dedent(
    '''
    """A picklable engine and loader for the pooled HASO run."""
    import numpy as np


    class Result:
        def __init__(self, pixels):
            total = float(pixels.sum())
            grid = np.arange(12, dtype=np.float32).reshape(3, 4)
            self.processed_phase = grid + total
            self.raw_phase = grid
            self.intensity = grid * 2
            self.slopes_x = grid * 3
            self.slopes_y = grid * 4
            self.pupil = np.ones((3, 4), dtype=bool)


    class Engine:
        def compute(self, pixels, **parameters):
            return Result(pixels)


    def load(shot):
        return np.full((6, 6), shot, dtype=np.uint16)
    '''
)


def test_a_pooled_haso_run_equals_the_serial_run(tmp_path, monkeypatch):
    (tmp_path / "haso_pool_helper.py").write_text(HELPER)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    helper = importlib.import_module("haso_pool_helper")
    recipe = compile_recipe(
        AnalysisRecipe.model_validate(
            {
                "device": "U_HasoLift",
                "input": {"kind": "camera"},
                "measure": {"kind": "haso", "sensor_config": SENSOR},
            }
        )
    )
    groups = [ShotGroup(n, (n,)) for n in range(1, 9)]
    inputs = {"haso": helper.Engine()}

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
    monkeypatch.delitem(importlib.sys.modules, "haso_pool_helper", raising=False)
