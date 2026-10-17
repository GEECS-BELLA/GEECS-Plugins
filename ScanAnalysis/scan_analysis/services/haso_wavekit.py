r"""HASO WaveKit — the ``haso`` measure's engine, run out of process.

The wavefront reconstruction is Imagine Optic's WaveKit 4.3, a licensed
64-bit Windows SDK. This module is the host side the analysis core binds
as the ``haso`` service: it hands one shot's pixels to
``_wavekit_worker.py`` running in the SDK's own Windows Python — natively
on Windows, or on Linux under 64-bit Wine, where the results are
bit-identical to Windows (26_0310 Scan012, 2026-09-28) — and reads back
the phases, intensity, slopes and pupil. The SDK reads a real ``.himg``
and nothing else, so every call rebuilds one in a temporary directory
from the pixels and *a header of the same sensor* (the per-shot header
differs only in its timestamp; the scan's stack keeps them all).

A reference (the ``haso`` measure's ``reference``) costs one extra worker
per process, not per shot: its raw slopes are computed once, saved as an
SDK ``.has`` in a private temporary directory, and every later shot's
worker loads that file and subtracts it. The cache is keyed by the
reference pixels and the settings that shape slopes, lives as long as the
engine object, and never travels when the engine is pickled to a pool
worker (each process computes its own).

Facility values come from ``~/.config/geecs_python_api/config.ini``::

    [Paths]
    wavekit_sdk_path = /mnt/share/software/WaveKit/wavekit_43        # wavekit_py/ + dlls/x64/
    wavekit_python_path = /mnt/share/software/WaveKit/python-3.8.10-embed-amd64/python.exe
    wavekit_configs_path = /mnt/share/software/WaveKit/configs        # the sensors' .dat / .lift
    # Linux only: the command that runs the Windows Python in a 64-bit Wine
    # prefix (Wine's default, or WINEPREFIX=<dir> for one of its own)
    wavekit_launcher = env WINEDEBUG=-all wine

The runbook (``docs/analysis/haso.md``) and ``geecs-wavekit-doctor`` cover
the share layout, the Wine prefix and the self-tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from geecs_data_utils.io.himg import himg_bytes
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["HasoWaveKit", "HasoWaveKitResult", "WaveKitError", "WaveKitSensorMismatch"]

#: The worker script, beside this file.
_WORKER_SCRIPT = Path(__file__).parent / "_wavekit_worker.py"

#: The worker's exit code for an image whose serial is not the config's.
_EXIT_SERIAL_MISMATCH = 3

#: The config keys, in the order the error messages list them.
CONFIG_KEYS = (
    "wavekit_sdk_path",
    "wavekit_python_path",
    "wavekit_configs_path",
    "wavekit_launcher",
)


class WaveKitError(RuntimeError):
    """The worker did not produce a result (a crash, a timeout, a bad install)."""


class WaveKitSensorMismatch(WaveKitError):
    """The image's embedded serial is not the named sensor configuration's."""


@dataclass
class HasoWaveKitResult:
    """One shot's wavefront products on the sub-aperture grid.

    Attributes
    ----------
    processed_phase, raw_phase : float32 arrays
        Zonal phase after and before the pupil mask and filters; NaN
        outside the pupil.
    intensity : float32 array
        The sub-aperture intensities.
    slopes_x, slopes_y : float32 arrays
        The processed slopes.
    pupil : bool array
        The processed pupil.
    image_serial, config_serial : str
        The sensor serial in the image and in the configuration (equal).
    timings : dict
        Seconds from the worker's start to each step.
    """

    processed_phase: NDArray[np.float32]
    raw_phase: NDArray[np.float32]
    intensity: NDArray[np.float32]
    slopes_x: NDArray[np.float32]
    slopes_y: NDArray[np.float32]
    pupil: NDArray[np.bool_]
    image_serial: str
    config_serial: str
    timings: dict


class HasoWaveKit:
    """The WaveKit engine as a picklable service: paths, a header, a launcher.

    Parameters
    ----------
    sdk_path : Path
        The SDK directory holding ``wavekit_py/`` and ``dlls/x64/``.
    python_path : Path
        The 64-bit Windows Python that imports the SDK (its ``python.exe``).
    configs_path : Path
        The directory of sensor configurations; a recipe's ``sensor_config``
        names a file in it.
    header : bytes
        Any ``.himg`` header of the sensor whose pixels will be computed
        (a scan's stack keeps one per frame; the first serves every shot).
    launcher : sequence of str
        Command prefix that runs the interpreter (``("env", "WINEDEBUG=-all",
        "wine")`` on Linux, in a 64-bit prefix); empty runs it directly, as
        on Windows.
    timeout : float
        Seconds one shot may take end to end (a shot takes ~20 s on a
        4-core host; the first after a Wine start a few more).

    Raises
    ------
    FileNotFoundError
        A path does not exist, or the SDK directory lacks its bindings or
        its 64-bit DLLs.
    """

    def __init__(
        self,
        sdk_path: str | Path,
        python_path: str | Path,
        configs_path: str | Path,
        header: bytes,
        launcher: Sequence[str] = (),
        timeout: float = 300.0,
    ) -> None:
        self.sdk_path = Path(sdk_path)
        self.python_path = Path(python_path)
        self.configs_path = Path(configs_path)
        self.header = bytes(header)
        self.launcher: tuple[str, ...] = tuple(launcher)
        self.timeout = float(timeout)
        #: MKL threads per worker process; ``None`` leaves MKL its default
        #: (every core). A pooled scan run sets it through :meth:`share_cores`.
        self.threads: Optional[int] = None
        #: The first sensor mismatch this engine met: every later call fails
        #: at once, so a recipe naming the wrong sensor costs one worker
        #: process, not one per shot.
        self._refused: Optional[str] = None
        self._init_reference_cache()
        for path, what in (
            (self.sdk_path / "wavekit_py", "the SDK's Python bindings (wavekit_py/)"),
            (self.sdk_path / "dlls" / "x64", "the SDK's 64-bit DLLs (dlls/x64/)"),
            (self.python_path, "the 64-bit Windows Python (wavekit_python_path)"),
            (self.configs_path, "the sensor configurations (wavekit_configs_path)"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{what} not found at {path}")
        if not self.header:
            raise ValueError("HasoWaveKit needs a .himg header of the sensor")

    def _init_reference_cache(self) -> None:
        """An empty per-process cache of reference slopes (``.has`` files)."""
        #: Reference key → saved slopes; see :meth:`_reference_slopes`.
        self._references: dict[str, Path] = {}
        self._reference_dir: Optional[Path] = None

    def __getstate__(self) -> dict:
        """Pickle the configuration; the reference cache stays in this process."""
        state = dict(self.__dict__)
        state.pop("_references", None)
        state.pop("_reference_dir", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the configuration with an empty reference cache."""
        self.__dict__.update(state)
        self._init_reference_cache()

    def share_cores(self, workers: int) -> None:
        """Give each of ``workers`` concurrent shots an equal share of the cores.

        MKL takes every core by default; a pooled run of N shots at once
        would otherwise oversubscribe the host N times.
        """
        cores = os.cpu_count() or 1
        self.threads = max(1, cores // max(1, int(workers)))

    def sensor_file(self, sensor_config: str) -> Path:
        """The sensor configuration ``sensor_config`` names, under ``configs_path``."""
        if (
            not sensor_config
            or sensor_config in {".", ".."}
            or "/" in sensor_config
            or "\\" in sensor_config
        ):
            raise ValueError(
                "sensor_config must be a file name under wavekit_configs_path"
            )
        path = self.configs_path / sensor_config
        if not path.is_file():
            raise FileNotFoundError(
                f"sensor configuration {sensor_config!r} not found in {self.configs_path}"
            )
        return path

    def compute(
        self,
        pixels: NDArray[np.uint16],
        *,
        sensor_config: str,
        mask: Optional[tuple[int, int, int, int]] = None,
        filters: Sequence[bool] = (True, True, True, True, True, False),
        wavelength_nm: float = 800.0,
        start_subpupil: tuple[int, int] = (87, 64),
        zonal_prefs: tuple[int, int, float] = (100, 500, 1e-6),
        lift: bool = True,
        denoising_strength: float = 0.0,
        reference: Optional[NDArray[np.uint16]] = None,
    ) -> HasoWaveKitResult:
        """Reconstruct one frame's wavefront in a fresh worker process.

        Parameters
        ----------
        pixels : uint16 array
            The sensor frame, ``(height, width)`` as the header says.
        sensor_config : str
            The sensor's ``.dat`` file name under ``configs_path``.
        mask : tuple of int, optional
            ``(top, bottom, left, right)`` numpy slice bounds of the pupil
            on the slopes grid; ``None`` keeps the sensor's pupil.
        filters : sequence of bool
            The six SDK filter flags: tilt x, tilt y, curvature, 0° and 45°
            astigmatism, others.
        wavelength_nm, start_subpupil, zonal_prefs, lift, denoising_strength
            The engine settings (see the ``haso`` measure's spec).
        reference : uint16 array, optional
            A frame of the same sensor whose raw slopes are subtracted from
            this frame's before the mask and the filters; the processed
            phase, slopes and pupil then describe the difference. Computed
            once per distinct reference and settings (see the module notes).

        Raises
        ------
        WaveKitSensorMismatch
            The pixels' header carries another sensor's serial — on this
            call, and on every later call of this engine without starting
            a worker (a wrong ``sensor_config`` is a whole run's mistake).
        WaveKitError
            The worker failed or timed out.
        """
        if self._refused is not None:
            raise WaveKitSensorMismatch(self._refused)
        config = self.sensor_file(sensor_config)
        frame = np.asarray(pixels)
        params = {
            "sdk_path": str(self.sdk_path),
            "sensor_config": str(config),
            "lift": bool(lift),
            "wavelength_nm": float(wavelength_nm),
            "start_subpupil": [int(v) for v in start_subpupil],
            "denoising_strength": float(denoising_strength),
            "zonal_prefs": [
                int(zonal_prefs[0]),
                int(zonal_prefs[1]),
                float(zonal_prefs[2]),
            ],
            "mask": None if mask is None else [int(v) for v in mask],
            "filters": [bool(v) for v in filters],
            "task": "shot",
            "reference_slopes": None,
        }
        if reference is not None:
            params["reference_slopes"] = str(
                self._reference_slopes(np.asarray(reference), params)
            )
        workdir = Path(tempfile.mkdtemp(prefix="wavekit_"))
        try:
            report = self._run_worker(workdir, frame, params)
            return self._read_output(workdir / "output.npz", report)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _reference_slopes(self, pixels: NDArray, params: dict) -> Path:
        """The reference's raw slopes as a ``.has`` file, computed once per key.

        The key hashes the pixels, their shape and every setting that shapes
        raw slopes (the sensor file, LIFT, wavelength, start sub-pupil,
        denoising); the mask and the filters act after the subtraction and
        are not part of it.
        """
        shaping = {
            key: params[key]
            for key in (
                "sensor_config",
                "lift",
                "wavelength_nm",
                "start_subpupil",
                "denoising_strength",
            )
        }
        frame = np.ascontiguousarray(pixels, dtype=np.uint16)
        digest = hashlib.sha256(json.dumps(shaping, sort_keys=True).encode())
        digest.update(repr(frame.shape).encode())
        digest.update(frame.tobytes())
        key = digest.hexdigest()
        cached = self._references.get(key)
        if cached is not None:
            return cached
        if self._reference_dir is None:
            directory = Path(tempfile.mkdtemp(prefix="wavekit_reference_"))
            weakref.finalize(self, shutil.rmtree, directory, True)
            self._reference_dir = directory
        target = self._reference_dir / f"{key[:16]}.has"
        workdir = Path(tempfile.mkdtemp(prefix="wavekit_"))
        try:
            self._run_worker(
                workdir,
                frame,
                {**params, "task": "reference", "save_slopes": str(target)},
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if not target.is_file():
            raise WaveKitError("WaveKit worker saved no reference slopes")
        logger.info("WaveKit reference slopes computed once for this process")
        self._references[key] = target
        return target

    def _run_worker(self, workdir: Path, frame: NDArray, params: dict) -> dict:
        """Run one worker task in ``workdir``; its ``result.json`` report.

        Raises :class:`WaveKitSensorMismatch` (remembered for the engine's
        life) or :class:`WaveKitError` as :meth:`compute` documents.
        """
        (workdir / "input.himg").write_bytes(himg_bytes(self.header, frame))
        (workdir / "params.json").write_text(json.dumps(params))
        cmd = [
            *self.launcher,
            str(self.python_path),
            str(_WORKER_SCRIPT),
            str(workdir),
        ]
        env = dict(os.environ)
        if self.threads is not None:
            env["MKL_NUM_THREADS"] = str(self.threads)
        logger.debug("Running WaveKit worker: %s", " ".join(cmd))
        try:
            run = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout, env=env
            )
        except subprocess.TimeoutExpired as exc:
            raise WaveKitError(
                f"WaveKit worker timed out after {self.timeout:.0f} s"
            ) from exc
        result_path = workdir / "result.json"
        report = json.loads(result_path.read_text()) if result_path.is_file() else {}
        if run.returncode == _EXIT_SERIAL_MISMATCH:
            self._refused = report.get("error") or run.stderr.strip()
            raise WaveKitSensorMismatch(self._refused)
        if run.returncode != 0:
            raise WaveKitError(
                f"WaveKit worker failed (return code {run.returncode}).\n"
                f"stdout: {run.stdout}\nstderr: {run.stderr}"
            )
        for line in run.stderr.strip().splitlines():
            logger.info("WaveKit worker: %s", line)
        return report

    @staticmethod
    def _read_output(path: Path, report: dict) -> HasoWaveKitResult:
        """The worker's arrays and serials, checked for shape agreement."""
        if not path.is_file():
            raise WaveKitError("WaveKit worker produced no output.npz")
        with np.load(path) as data:
            arrays = {key: np.array(data[key]) for key in data.files}
        expected = (
            "processed_phase",
            "raw_phase",
            "intensity",
            "slopes_x",
            "slopes_y",
            "pupil",
        )
        missing = [key for key in expected if key not in arrays]
        if missing:
            raise WaveKitError(f"WaveKit worker output lacks {missing}")
        shape = arrays["processed_phase"].shape
        if any(arrays[key].shape != shape for key in expected):
            raise WaveKitError("WaveKit worker arrays disagree in shape")
        return HasoWaveKitResult(
            processed_phase=arrays["processed_phase"].astype(np.float32, copy=False),
            raw_phase=arrays["raw_phase"].astype(np.float32, copy=False),
            intensity=arrays["intensity"].astype(np.float32, copy=False),
            slopes_x=arrays["slopes_x"].astype(np.float32, copy=False),
            slopes_y=arrays["slopes_y"].astype(np.float32, copy=False),
            pupil=arrays["pupil"].astype(bool, copy=False),
            image_serial=str(report.get("image_serial", "")),
            config_serial=str(report.get("config_serial", "")),
            timings=dict(report.get("timings", {})),
        )

    @classmethod
    def from_config(
        cls,
        header: bytes,
        *,
        sdk_path: Optional[str | Path] = None,
        python_path: Optional[str | Path] = None,
        configs_path: Optional[str | Path] = None,
        launcher: Optional[Sequence[str]] = None,
        timeout: float = 300.0,
    ) -> HasoWaveKit:
        """Build the engine from ``[Paths] wavekit_*`` in the client ``config.ini``.

        Every path not given is read from ``GeecsPathsConfig``; the launcher
        from ``wavekit_launcher``, split like a shell command line (absent:
        the interpreter runs directly). A key that is missing — or names a
        path that does not exist, which the config reader reports as
        missing — is a ``FileNotFoundError`` naming it.
        """
        if (
            sdk_path is None
            or python_path is None
            or configs_path is None
            or launcher is None
        ):
            from geecs_data_utils import GeecsPathsConfig

            config = GeecsPathsConfig()
            if launcher is None:
                launcher = shlex.split(getattr(config, "wavekit_launcher", None) or "")
            values = {
                "sdk_path": sdk_path,
                "python_path": python_path,
                "configs_path": configs_path,
            }
            for name, key in (
                ("sdk_path", "wavekit_sdk_path"),
                ("python_path", "wavekit_python_path"),
                ("configs_path", "wavekit_configs_path"),
            ):
                if values[name] is None:
                    values[name] = getattr(config, key, None)
                    if values[name] is None:
                        raise FileNotFoundError(
                            f"{key} is not set in config.ini (or names a path that "
                            "does not exist). Add it under [Paths]; see "
                            "docs/analysis/haso.md and run geecs-wavekit-doctor."
                        )
            sdk_path, python_path, configs_path = (
                values["sdk_path"],
                values["python_path"],
                values["configs_path"],
            )
        return cls(
            sdk_path,
            python_path,
            configs_path,
            header,
            launcher=launcher or (),
            timeout=timeout,
        )
