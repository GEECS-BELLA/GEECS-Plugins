r"""Standalone HASO WaveKit worker: pixels in, wavefront out, one shot per process.

Run by the 64-bit Windows Python the WaveKit SDK ships with (Python 3.8 +
numpy 1.19 — under Wine 6.0.3 a newer numpy crashes at import), natively
on Windows or under Wine on Linux. This file must therefore stay Python
3.8 syntax with numpy 1.19's API and import nothing from this repository.

Usage::

    python.exe _wavekit_worker.py <workdir>

``<workdir>`` holds ``params.json`` and ``input.himg`` (a real ``.himg``:
the SDK reads files, never memory); the worker writes ``output.npz`` and
``result.json`` there. A ``"reference"`` task instead saves the image's
raw slopes as an SDK ``.has`` file at ``save_slopes`` (and writes only
``result.json``); a later ``"shot"`` task naming that file as
``reference_slopes`` subtracts it from the shot's slopes before the mask
and the filters — the SDK's own ``apply_substractor``, the legacy order. Each process makes a fresh engine: the SDK's spot
tracker carries state between ``compute_slopes`` calls, so a shot must
never follow a differently processed image in the same engine.

``params.json``::

    {
        "sdk_path": "/mnt/.../WaveKit/wavekit_43",      # holds wavekit_py/ and dlls/x64/
        "sensor_config": "/mnt/.../configs/WFS_....dat", # the sensor's licence + calibration
        "lift": true,                                    # LIFT reconstruction at wavelength_nm
        "wavelength_nm": 800.0,
        "start_subpupil": [87, 64],
        "denoising_strength": 0.0,
        "zonal_prefs": [100, 500, 1e-6],
        "mask": [top, bottom, left, right] or null,      # numpy slice bounds on the slopes grid
        "filters": [tilt_x, tilt_y, curvature, astig_0, astig_45, others],
        "task": "shot",                                  # or "reference"; absent = "shot"
        "reference_slopes": "/tmp/.../reference.has",    # shot task: slopes to subtract, or null
        "save_slopes": "/tmp/.../reference.has"          # reference task: where the .has goes
    }

``output.npz``: ``raw_phase``, ``processed_phase``, ``intensity``,
``slopes_x``, ``slopes_y`` (float32, the SDK's precision) and ``pupil``
(bool), all on the sub-aperture grid. ``result.json``: the image's and
the config's serial numbers and the step timings; on a serial mismatch
only ``result.json`` is written, with ``error``, and the exit code is 3.
"""

import json
import os
import sys
import time

EXIT_USAGE = 2
EXIT_SERIAL_MISMATCH = 3


def _windows_path(path):
    r"""A path the Windows Python and the SDK can open.

    Under Wine the host hands over Linux paths; Wine's ``Z:`` drive is the
    Unix root, so ``/mnt/share/x`` is ``Z:\mnt\share\x``. The DLLs accept
    a Unix path as it is (Wine's file API does), but Python's import system
    does not, hence the mapping for ``sys.path``; every path is mapped for
    uniformity. A Windows path, or a path on a real Windows host, is
    returned unchanged.
    """
    if sys.platform == "win32" and path.startswith("/"):
        return "Z:" + path.replace("/", "\\")
    return path


def _load_params(workdir):
    with open(os.path.join(workdir, "params.json"), "r") as f:
        return json.load(f)


def _write_result(workdir, payload):
    with open(os.path.join(workdir, "result.json"), "w") as f:
        json.dump(payload, f)


def main():
    """Compute one shot's wavefront from the work directory's inputs."""
    if len(sys.argv) != 2:
        print("Usage: python.exe _wavekit_worker.py <workdir>", file=sys.stderr)
        sys.exit(EXIT_USAGE)
    workdir = sys.argv[1]
    params = _load_params(workdir)
    timings = {}
    started = time.time()

    def stamp(name):
        timings[name] = round(time.time() - started, 3)

    sys.path.insert(0, _windows_path(params["sdk_path"]))
    import numpy as np

    import wavekit_py as wkpy

    stamp("sdk_loaded")

    config = _windows_path(params["sensor_config"])
    image_path = _windows_path(os.path.join(workdir, "input.himg"))
    image = wkpy.Image(image_file_path=image_path)
    image_serial = image.get_haso_serial_number()
    config_serial = wkpy.HasoConfig.get_serial_number(config)
    stamp("image_loaded")
    if image_serial != config_serial:
        message = (
            "sensor mismatch: the image was taken by HASO serial %r but the "
            "configuration %s is for serial %r" % (image_serial, config, config_serial)
        )
        print(message, file=sys.stderr)
        _write_result(
            workdir,
            {
                "error": message,
                "image_serial": image_serial,
                "config_serial": config_serial,
                "timings": timings,
            },
        )
        sys.exit(EXIT_SERIAL_MISMATCH)

    engine = wkpy.HasoEngine(config_file_path=config)
    wavelength = float(params["wavelength_nm"])
    if params.get("lift", True):
        engine.set_lift_enabled(True, wavelength)
        engine.set_lift_option(True, wavelength)
    x, y = params["start_subpupil"]
    engine.set_preferences(
        wkpy.uint2D(int(x), int(y)), float(params.get("denoising_strength", 0.0)), False
    )
    phase_set = wkpy.ComputePhaseSet(type_phase=wkpy.E_COMPUTEPHASESET.ZONAL)
    weak, most, residual = params["zonal_prefs"]
    phase_set.set_zonal_prefs(int(weak), int(most), float(residual))
    post = wkpy.SlopesPostProcessor()
    stamp("engine_ready")

    _, raw = engine.compute_slopes(image, False)
    stamp("slopes")

    if params.get("task", "shot") == "reference":
        raw.save_to_file(_windows_path(params["save_slopes"]), "", "")
        stamp("saved")
        _write_result(
            workdir,
            {
                "image_serial": image_serial,
                "config_serial": config_serial,
                "timings": timings,
            },
        )
        return

    def phase_of(slopes):
        data = wkpy.HasoData(hasoslopes=slopes)
        return np.array(wkpy.Compute.phase_zonal(phase_set, data).get_data()[0])

    intensity = np.array(wkpy.Intensity(hasoslopes=raw).get_data()[0])
    raw_phase = phase_of(raw)
    stamp("raw_phase")

    processed = raw
    if params.get("reference_slopes"):
        reference = wkpy.HasoSlopes(
            has_file_path=_windows_path(params["reference_slopes"])
        )
        processed = post.apply_substractor(processed, reference)
        stamp("reference_subtracted")
    mask = params.get("mask")
    if mask is not None:
        top, bottom, left, right = (int(v) for v in mask)
        pupil = wkpy.Pupil(hasoslopes=processed)
        buffer = np.asarray(pupil.get_data(), dtype=bool)
        buffer.fill(False)
        rows, cols = buffer.shape
        buffer[max(0, top) : min(rows, bottom), max(0, left) : min(cols, right)] = True
        pupil.set_data(datas=buffer)
        processed = post.apply_pupil(processed, pupil)
    flags = [bool(v) for v in params["filters"]]
    processed = post.apply_filter(processed, *flags)
    processed_phase = phase_of(processed)
    slopes_x, slopes_y = processed.get_slopes()
    pupil_buffer = np.asarray(processed.get_pupil_buffer(), dtype=bool)
    stamp("processed_phase")

    np.savez(
        os.path.join(workdir, "output.npz"),
        raw_phase=np.asarray(raw_phase, dtype=np.float32),
        processed_phase=np.asarray(processed_phase, dtype=np.float32),
        intensity=np.asarray(intensity, dtype=np.float32),
        slopes_x=np.asarray(slopes_x, dtype=np.float32),
        slopes_y=np.asarray(slopes_y, dtype=np.float32),
        pupil=pupil_buffer,
    )
    stamp("written")
    _write_result(
        workdir,
        {
            "image_serial": image_serial,
            "config_serial": config_serial,
            "shape": list(pupil_buffer.shape),
            "timings": timings,
        },
    )


if __name__ == "__main__":
    main()
