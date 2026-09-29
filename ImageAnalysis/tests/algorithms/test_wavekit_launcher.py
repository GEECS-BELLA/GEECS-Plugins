"""The WaveKit worker command and result: run directly on Windows, behind a launcher elsewhere."""

from __future__ import annotations

import json
import pickle
import struct
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from image_analysis.algorithms import haso_wavekit as wavekit
from image_analysis.algorithms.haso_wavekit import (
    HasoWaveKit,
    WaveKitError,
    WaveKitSensorMismatch,
)

SENSOR = "WFS_HASO4_LIFT_680_8244_gain_enabled.dat"
HEIGHT, WIDTH = 4, 6


def header() -> bytes:
    blob = b"\x07" * 12
    return b"\x00" + struct.pack("<4I", 2, WIDTH, HEIGHT, len(blob)) + blob


@pytest.fixture
def tree(tmp_path):
    sdk = tmp_path / "wavekit_43"
    (sdk / "wavekit_py").mkdir(parents=True)
    (sdk / "dlls" / "x64").mkdir(parents=True)
    python = tmp_path / "py38" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / SENSOR).write_bytes(b"licence")
    return sdk, python, configs


def engine(tree, **kwargs) -> HasoWaveKit:
    sdk, python, configs = tree
    return HasoWaveKit(sdk, python, configs, header(), **kwargs)


def pixels() -> np.ndarray:
    return np.arange(HEIGHT * WIDTH, dtype=np.uint16).reshape(HEIGHT, WIDTH)


def fake_worker(behaviour):
    """A ``subprocess.run`` stand-in that plays the worker for one call."""
    seen = []

    def run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        workdir = Path(cmd[-1])
        return behaviour(workdir, cmd, kwargs)

    return seen, run


def good_worker(workdir: Path, cmd, kwargs):
    grid = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez(
        workdir / "output.npz",
        raw_phase=grid,
        processed_phase=grid * 2,
        intensity=grid * 3,
        slopes_x=grid * 4,
        slopes_y=grid * 5,
        pupil=grid > 2,
    )
    (workdir / "result.json").write_text(
        json.dumps(
            {
                "image_serial": "8244",
                "config_serial": "8244",
                "timings": {"slopes": 6.1},
            }
        )
    )
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="[  5.90 s] slopes")


def test_the_tree_is_checked_at_construction(tree, tmp_path):
    sdk, python, configs = tree
    with pytest.raises(FileNotFoundError, match="wavekit_py"):
        HasoWaveKit(tmp_path / "nowhere", python, configs, header())
    with pytest.raises(FileNotFoundError, match="Windows Python"):
        HasoWaveKit(sdk, tmp_path / "no.exe", configs, header())
    with pytest.raises(FileNotFoundError, match="sensor configurations"):
        HasoWaveKit(sdk, python, tmp_path / "noconfigs", header())
    with pytest.raises(ValueError, match="header"):
        HasoWaveKit(sdk, python, configs, b"")


def test_the_worker_gets_a_rebuilt_himg_and_the_parameters(tree, monkeypatch):
    seen, run = fake_worker(good_worker)
    monkeypatch.setattr(wavekit.subprocess, "run", run)
    captured = {}

    def capture(cmd, **kwargs):
        workdir = Path(cmd[-1])
        captured["himg"] = (workdir / "input.himg").read_bytes()
        captured["params"] = json.loads((workdir / "params.json").read_text())
        return run(cmd, **kwargs)

    monkeypatch.setattr(wavekit.subprocess, "run", capture)
    service = engine(tree)
    result = service.compute(
        pixels(),
        sensor_config=SENSOR,
        mask=(1, 3, 0, 4),
        filters=(True, False, True, False, True, False),
        wavelength_nm=532.0,
        start_subpupil=(3, 2),
        zonal_prefs=(10, 50, 1e-5),
    )
    cmd, kwargs = seen[0]
    assert cmd[0] == str(tree[1]) and cmd[1] == str(wavekit._WORKER_SCRIPT)
    assert kwargs["timeout"] == 300.0
    assert "MKL_NUM_THREADS" not in kwargs["env"]
    assert captured["himg"] == header() + pixels().astype("<u2").tobytes()
    assert captured["params"] == {
        "sdk_path": str(tree[0]),
        "sensor_config": str(tree[2] / SENSOR),
        "lift": True,
        "wavelength_nm": 532.0,
        "start_subpupil": [3, 2],
        "denoising_strength": 0.0,
        "zonal_prefs": [10, 50, 1e-5],
        "mask": [1, 3, 0, 4],
        "filters": [True, False, True, False, True, False],
    }
    grid = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(result.processed_phase, grid * 2)
    np.testing.assert_array_equal(result.slopes_y, grid * 5)
    np.testing.assert_array_equal(result.pupil, grid > 2)
    assert result.pupil.dtype == bool and result.raw_phase.dtype == np.float32
    assert (result.image_serial, result.config_serial) == ("8244", "8244")
    assert result.timings == {"slopes": 6.1}
    # The work directory is gone after the call.
    assert not Path(cmd[-1]).exists()


def test_a_launcher_prefixes_the_interpreter_and_shares_cores(tree, monkeypatch):
    seen, run = fake_worker(good_worker)
    monkeypatch.setattr(wavekit.subprocess, "run", run)
    monkeypatch.setattr(wavekit.os, "cpu_count", lambda: 4)
    service = engine(tree, launcher=("env", "WINEPREFIX=/p", "wine"))
    service.share_cores(3)
    service.compute(pixels(), sensor_config=SENSOR)
    cmd, kwargs = seen[0]
    assert cmd[:4] == ["env", "WINEPREFIX=/p", "wine", str(tree[1])]
    assert kwargs["env"]["MKL_NUM_THREADS"] == "1"
    service.share_cores(1)
    assert service.threads == 4


def test_a_serial_mismatch_and_a_crash_are_distinct_errors(tree, monkeypatch):
    def mismatch(workdir, cmd, kwargs):
        (workdir / "result.json").write_text(
            json.dumps({"error": "sensor mismatch: image 7784, config 8244"})
        )
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="")

    _, run = fake_worker(mismatch)
    monkeypatch.setattr(wavekit.subprocess, "run", run)
    with pytest.raises(WaveKitSensorMismatch, match="7784"):
        engine(tree).compute(pixels(), sensor_config=SENSOR)

    def crash(workdir, cmd, kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    _, run = fake_worker(crash)
    monkeypatch.setattr(wavekit.subprocess, "run", run)
    with pytest.raises(WaveKitError, match="boom"):
        engine(tree).compute(pixels(), sensor_config=SENSOR)

    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(wavekit.subprocess, "run", timeout)
    with pytest.raises(WaveKitError, match="timed out"):
        engine(tree, timeout=7).compute(pixels(), sensor_config=SENSOR)


def test_unknown_sensor_or_wrong_frame_shape_never_starts_the_worker(tree, monkeypatch):
    seen, run = fake_worker(good_worker)
    monkeypatch.setattr(wavekit.subprocess, "run", run)
    service = engine(tree)
    with pytest.raises(FileNotFoundError, match="not found in"):
        service.compute(pixels(), sensor_config="other.dat")
    with pytest.raises(ValueError, match="file name"):
        service.compute(pixels(), sensor_config="../x.dat")
    with pytest.raises(Exception, match="does not match the header"):
        service.compute(np.zeros((2, 2), dtype=np.uint16), sensor_config=SENSOR)
    assert seen == []


def test_from_config_reads_the_keys_and_splits_the_launcher(tree, monkeypatch):
    sdk, python, configs = tree
    config = types.SimpleNamespace(
        wavekit_sdk_path=sdk,
        wavekit_python_path=python,
        wavekit_configs_path=configs,
        wavekit_launcher="env WINEDEBUG=-all WINEPREFIX=/var/lib/geecs/wine64 wine",
    )
    import geecs_data_utils

    monkeypatch.setattr(geecs_data_utils, "GeecsPathsConfig", lambda: config)
    service = HasoWaveKit.from_config(header())
    assert service.launcher == (
        "env",
        "WINEDEBUG=-all",
        "WINEPREFIX=/var/lib/geecs/wine64",
        "wine",
    )
    assert (service.sdk_path, service.python_path, service.configs_path) == (
        sdk,
        python,
        configs,
    )
    config.wavekit_launcher = None
    assert HasoWaveKit.from_config(header()).launcher == ()
    config.wavekit_configs_path = None
    with pytest.raises(FileNotFoundError, match="wavekit_configs_path"):
        HasoWaveKit.from_config(header())


def test_the_engine_pickles_for_pool_workers(tree):
    service = engine(tree, launcher=["wine"])
    service.share_cores(2)
    copy = pickle.loads(pickle.dumps(service))
    assert (copy.sdk_path, copy.python_path, copy.configs_path) == tree
    assert copy.header == header() and copy.launcher == ("wine",)
    assert copy.threads == service.threads


def test_the_worker_script_is_python_38(tmp_path):
    """The worker runs in the SDK's Python 3.8: no newer syntax may creep in."""
    import ast

    source = wavekit._WORKER_SCRIPT.read_text()
    ast.parse(source, feature_version=(3, 8))
    assert "from image_analysis" not in source and "geecs_" not in source
