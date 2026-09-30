"""Regenerate the example data embedded in this page's ``index.html``.

Every number and processed image on the page comes from running a real
recipe through ``geecs_analysis.run.analyze`` on a synthetic input made
here. The page stays self-contained: this script rewrites the JSON inside
its ``<script id="data">`` element and nothing else. Run it from any env
that has GEECS-Analysis installed (the portal's does)::

    python docs/sites/analysis_docs_preview/make_examples.py

FROG and HASO need vendor libraries (FROG.dll, WaveKit) that only the
Linux analysis host carries, so their inputs are generated here but their
measures are not run; the page labels those figures as illustrations.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import numpy as np
from matplotlib import cm
from PIL import Image

from geecs_analysis.pipeline import apply_pipeline
from geecs_analysis.run import analyze
from geecs_analysis.specs import Analysis, Pipeline
from geecs_data_utils.frames import Axis, Frame

PAGE = Path(__file__).with_name("index.html")
rng = np.random.default_rng(7)


def png(a: np.ndarray, vmax: float, cmap=cm.magma) -> str:
    """Encode an image as a colormapped PNG data URI."""
    a = np.clip(np.asarray(a, float) / vmax, 0, 1)
    rgba = (cmap(a) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def floats(d: dict) -> dict:
    """Scalars as plain floats (NaN becomes None for JSON)."""
    return {k: (None if not np.isfinite(v) else float(v)) for k, v in d.items()}


def frame_info(f: Frame, vmax: float) -> dict:
    """PNG, shape and first/last axis coordinates of a processed image."""
    return {
        "png": png(f.data, vmax),
        "shape": list(f.data.shape),
        "x": [float(f.axes[1].values[0]), float(f.axes[1].values[-1])],
        "y": [float(f.axes[0].values[0]), float(f.axes[0].values[-1])],
    }


def overlays(res) -> dict:
    """Projections as sample lists, markers as (x, y)."""
    out = {}
    for o in res.overlays:
        if hasattr(o, "frame"):
            out[o.id] = [float(v) for v in o.frame.data]
        else:
            out[o.id] = [float(o.x), float(o.y)]
    return out


def beam_example() -> dict:
    """A tilted elliptical spot with a halo, hot pixels and a pedestal."""
    H, W = 240, 320
    y, x = np.mgrid[0:H, 0:W]
    xc, yc, sx, sy, th = 185, 112, 22, 12, np.deg2rad(25)
    xr = (x - xc) * np.cos(th) + (y - yc) * np.sin(th)
    yr = -(x - xc) * np.sin(th) + (y - yc) * np.cos(th)
    img = 3200 * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    img += 250 * np.exp(-0.5 * (((x - 150) / 70) ** 2 + ((y - 130) / 60) ** 2))
    img += 90 + rng.normal(0, 18, img.shape)
    for _ in range(25):
        img[rng.integers(0, H), rng.integers(0, W)] = 4000
    raw = np.clip(img, 0, 4095).astype(np.uint16)
    steps = [
        {"step": "background_constant", "level": 90},
        {"step": "roi", "bounds": [[40, 200], [90, 290]], "units": "index"},
        {"step": "median", "kernel": 3},
        {"step": "zero_below", "level": 40},
    ]
    frame = Frame.from_array(raw.astype(float))
    stages = [{"png": png(raw, 3300), "shape": list(raw.shape)}]
    for i in range(1, len(steps) + 1):
        f = apply_pipeline(frame, Pipeline.model_validate({"steps": steps[:i]}))
        stages.append({"png": png(f.data, 3200), "shape": list(f.data.shape)})
    res = analyze(
        frame, Analysis.model_validate({"steps": steps, "measure": {"kind": "beam"}})
    )
    return {
        "steps": steps,
        "stages": stages,
        "scalars": floats(res.scalars),
        "overlays": overlays(res),
        "final": frame_info(res.frame, 3200),
    }


def line_example() -> dict:
    """A spectrum with a main peak and a weaker satellite."""
    t = np.linspace(400, 900, 500)
    tr = np.exp(-0.5 * ((t - 640) / 28) ** 2) + 0.35 * np.exp(
        -0.5 * ((t - 735) / 15) ** 2
    )
    tr += rng.normal(0, 0.02, t.size)
    f = Frame.from_array(tr, axes=(Axis(values=t, unit="nm"),))
    res = analyze(
        f, Analysis.model_validate({"steps": [], "measure": {"kind": "line"}})
    )
    return {
        "t": t[::2].round(2).tolist(),
        "y": tr[::2].round(4).tolist(),
        "scalars": floats(res.scalars),
    }


def stitch_example() -> dict:
    """Three spectrometer segments joined as the scan host joins siblings."""
    edges = [(40.0, 105.0, 260), (100.0, 170.0, 200), (165.0, 260.0, 180)]
    gain = [1.0, 0.92, 1.08]  # per-camera response, uncorrected by the join

    def spectrum(e: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((e - 128) / 22) ** 2) + 0.25 * np.exp(-(e - 40) / 25)

    segments = []
    for (lo, hi, n), g in zip(edges, gain):
        e = np.linspace(lo, hi, n)
        segments.append(np.column_stack([e, g * spectrum(e) + rng.normal(0, 0.015, n)]))
    # The join in scan_analysis.core_source: concatenate, then sort by x.
    combined = np.concatenate(segments, axis=0)
    combined = combined[combined[:, 0].argsort()]
    f = Frame.from_array(
        combined[:, 1], axes=(Axis(values=combined[:, 0], unit="MeV"),)
    )
    res = analyze(
        f, Analysis.model_validate({"steps": [], "measure": {"kind": "line"}})
    )
    # FWHM measured directly in MeV, to compare with the index-space value.
    y = combined[:, 1]
    above = combined[y >= y.max() / 2, 0]
    return {
        "direct_fwhm": float(above.max() - above.min()),
        "true_fwhm": float(2 * np.sqrt(2 * np.log(2)) * 22),
        "segments": [
            {"x": seg[::2, 0].round(2).tolist(), "y": seg[::2, 1].round(4).tolist()}
            for seg in segments
        ],
        "joined": {
            "x": combined[::2, 0].round(2).tolist(),
            "y": combined[::2, 1].round(4).tolist(),
        },
        "scalars": floats(res.scalars),
    }


def ict_example() -> dict:
    """A negative ICT pulse riding on sinusoidal pickup, sampled at 1 ns."""
    dt = 1e-9
    t = np.arange(2000) * dt
    v = -50e-3 * np.exp(-0.5 * ((t - 1.0e-6) / 3e-9) ** 2)
    v += 5e-3 * np.sin(2 * np.pi * 4e6 * t + 0.6) + 1e-3 + rng.normal(0, 8e-4, t.size)
    f = Frame.from_array(v, axes=(Axis(values=t, unit="s"),))
    res = analyze(f, Analysis.model_validate({"steps": [], "measure": {"kind": "ict"}}))
    keep = slice(None, None, 4)
    return {
        "t_us": (t[keep] * 1e6).round(4).tolist(),
        "v_mV": (v[keep] * 1e3).round(4).tolist(),
        "scalars": floats(res.scalars),
        "truth_pC": float(50e-3 * 3e-9 * np.sqrt(2 * np.pi) * 0.1 * 1e12),
        "notes": list(res.notes),
    }


def hires_example() -> dict:
    """A dispersed beam whose vertical size narrows to a waist (a bow tie)."""
    H, W = 160, 360
    y, x = np.mgrid[0:H, 0:W]
    w0, theta, x0, yc = 3.5, 0.09, 205.0, 80.0
    sigma = np.sqrt(w0**2 + ((x - x0) * theta) ** 2)
    envelope = 900 * np.exp(-0.5 * ((x - 190) / 110) ** 2) + 60
    img = envelope * (w0 / sigma) * np.exp(-0.5 * ((y - yc) / sigma) ** 2)
    img += 30 + rng.normal(0, 6, img.shape)
    raw = np.clip(img, 0, 4095)
    steps = [
        {"step": "background_constant", "level": 30},
        {"step": "zero_below", "level": 20},
    ]
    res = analyze(
        Frame.from_array(raw),
        Analysis.model_validate(
            {"steps": steps, "measure": {"kind": "hi_res_mag_cam"}}
        ),
    )
    ov = overlays(res)
    return {
        "steps": steps,
        "scalars": floats(res.scalars),
        "weights": ov.get("bowtie_weights", []),
        "final": frame_info(res.frame, 700),
        "yc": yc,
        "truth": {"bowtie_w0": w0, "bowtie_theta": theta, "bowtie_x0": x0},
        "notes": list(res.notes),
    }


def none_example() -> dict:
    """Preprocessing only: raw and processed frames, no scalars."""
    H, W = 120, 160
    y, x = np.mgrid[0:H, 0:W]
    img = 1500 * np.exp(-0.5 * (((x - 80) / 18) ** 2 + ((y - 60) / 14) ** 2)) + 120
    img += rng.normal(0, 25, img.shape)
    steps = [
        {"step": "background_constant", "level": 120},
        {"step": "gaussian", "sigma": 2},
        {"step": "circular_mask", "center": [60, 80], "radius": 45},
    ]
    res = analyze(
        Frame.from_array(img),
        Analysis.model_validate({"steps": steps, "measure": {"kind": "none"}}),
    )
    return {
        "steps": steps,
        "raw": png(img, 1600),
        "processed": png(res.frame.data, 1500),
        "nscalars": len(res.scalars),
    }


def frog_illustration() -> dict:
    """An SHG FROG trace of a known chirped Gaussian pulse (the input only)."""
    n = 128
    t = np.linspace(-150, 150, n)  # fs
    tau, chirp = 30.0, 0.0012
    field = np.exp(-2 * np.log(2) * (t / tau) ** 2) * np.exp(1j * chirp * t**2)
    trace = np.empty((n, n))
    for i, d in enumerate(range(-n // 2, n // 2)):
        gate = np.roll(field, d)
        trace[:, i] = np.abs(np.fft.fftshift(np.fft.fft(field * gate))) ** 2
    trace /= trace.max()
    intensity = np.abs(field) ** 2
    above = t[intensity >= 0.5]
    return {
        "png": png(trace, 1.0, cm.viridis),
        "true_fwhm_fs": float(above[-1] - above[0]),
    }


def haso_illustration() -> dict:
    """A Shack-Hartmann spot grid and a made-up phase map (not WaveKit)."""
    n, pitch = 22, 10
    size = n * pitch
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = np.mgrid[0:n, 0:n]
    u, v = (cx - n / 2) / (n / 2), (cy - n / 2) / (n / 2)
    phase = 0.35 * (2 * (u**2 + v**2) - 1) + 0.18 * (u**2 - v**2) + 0.08 * u
    pupil = (u**2 + v**2) <= 0.9
    gy, gx = np.gradient(phase)
    img = np.zeros((size, size))
    for j in range(n):
        for i in range(n):
            if not pupil[j, i]:
                continue
            sx = (i + 0.5) * pitch + 12 * gx[j, i]
            sy = (j + 0.5) * pitch + 12 * gy[j, i]
            img += np.exp(-0.5 * (((xx - sx) / 1.3) ** 2 + ((yy - sy) / 1.3) ** 2))
    p = np.where(pupil, phase - phase[pupil].mean(), np.nan)
    shown = np.nan_to_num((p - np.nanmin(p)) / (np.nanmax(p) - np.nanmin(p)), nan=0.0)
    return {
        "spots": png(img, 1.0, cm.gray),
        "phase": png(shown, 1.0, cm.RdBu_r),
        "rms": float(np.nanstd(p)),
        "pv": float(np.nanmax(p) - np.nanmin(p)),
    }


def reference() -> dict:
    """Each measure's parameters (from the recipe schema) and scalar names."""
    from geecs_analysis.recipe import recipe_schema
    from geecs_analysis.registry import measure_definitions

    defs = recipe_schema()["$defs"]
    out = {}
    for entry in measure_definitions():
        spec = entry.spec
        props = defs.get(spec.__name__, {}).get("properties", {})
        out[spec.model_fields["kind"].default] = {
            "params": [
                {
                    "name": name,
                    "default": json.dumps(field["default"])
                    if "default" in field
                    else "required",
                    "description": field.get("description", ""),
                }
                for name, field in props.items()
                if name != "kind"
            ],
            # model_construct: a spec with required fields (haso) still lists its keys.
            "scalars": sorted(spec.model_construct().emitted_scalars()),
            "ndim": sorted(entry.ndim),
            "service": entry.service,
        }
    siblings = defs["LineInput"]["properties"]["siblings"]
    out["stitch"] = {
        "params": [
            {
                "name": "input.siblings",
                "default": "unset",
                "description": siblings.get("description", ""),
            }
        ],
        "scalars": out["line"]["scalars"],
        "ndim": [1],
        "service": None,
    }
    return out


def main() -> None:
    """Build every example and write the JSON into the page."""
    data = {
        "reference": reference(),
        "beam": beam_example(),
        "line": line_example(),
        "ict": ict_example(),
        "hires": hires_example(),
        "none": none_example(),
        "frog": frog_illustration(),
        "haso": haso_illustration(),
    }
    # Last, so adding it left every earlier example's random draws unchanged.
    data["stitch"] = stitch_example()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = PAGE.read_text()
    html, n = re.subn(
        r'(<script id="data" type="application/json">)(.*?)(</script>)',
        lambda m: m.group(1) + blob + m.group(3),
        html,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise SystemExit("data block not found in index.html")
    PAGE.write_text(html)
    for k in ("ict", "hires"):
        print(k, data[k]["scalars"], data[k].get("notes"))
    print(f"{len(blob) // 1024} KB of example data")


if __name__ == "__main__":
    main()
