"""Regenerate the example data embedded in the recipe reference page.

Every number and processed image on the page comes from running a real
recipe through ``geecs_analysis.run.analyze`` on a synthetic input made
here. The page stays self-contained: this script rewrites the JSON inside
its ``<script id="data">`` element and nothing else. Run it from any env
that has GEECS-Analysis installed (the portal's does)::

    PYTHONPATH=GEECS-Analysis python docs/sites/analysis_recipes/make_examples.py

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
    rgb = (cmap(a)[..., :3] * 255).astype(np.uint8)
    buf = io.BytesIO()
    # A 64-colour palette keeps the page small; colormaps survive it intact.
    Image.fromarray(rgb).quantize(64).save(buf, "PNG", optimize=True)
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
    # FWHM measured directly in MeV (the span of the samples at or above half
    # maximum), to compare with the measure's interpolated crossings.
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


def _params(defn: dict, skip: str) -> list[dict]:
    """A variant's fields as rows: name, default (or "required"), description."""
    return [
        {
            "name": name,
            "default": json.dumps(field["default"])
            if "default" in field
            else "required",
            "description": field.get("description", ""),
        }
        for name, field in defn.get("properties", {}).items()
        if name != skip
    ]


def reference() -> dict:
    """Every step, measure and summary as the recipe schema describes it.

    Parameters, descriptions and dimensionality come from ``recipe_schema()``;
    a measure's scalar meanings from its ``scalar_docs``. Nothing here is
    written by hand, so the page cannot drift from the code.
    """
    from geecs_analysis.recipe import recipe_schema
    from geecs_analysis.registry import (
        definitions,
        measure_definitions,
        summary_definitions,
    )

    defs = recipe_schema()["$defs"]

    def entry(item, field: str) -> tuple[str, dict]:
        defn = defs[item.spec.__name__]
        return item.spec.model_fields[field].default, {
            "description": defn.get("description", ""),
            "params": _params(defn, field),
            "ndim": sorted(item.ndim),
        }

    measures = {}
    for item in measure_definitions():
        kind, row = entry(item, "kind")
        # model_construct: a spec with required fields (haso) still lists its keys.
        row["scalars"] = sorted(item.spec.model_construct().emitted_scalars())
        row["scalar_docs"] = dict(item.spec.scalar_docs)
        row["service"] = item.service
        measures[kind] = row
    siblings = defs["LineInput"]["properties"]["siblings"]
    return {
        "measures": measures,
        "steps": dict(entry(item, "step") for item in definitions()),
        "summaries": dict(entry(item, "kind") for item in summary_definitions()),
        "siblings": {
            "name": "input.siblings",
            "default": "unset",
            "description": siblings.get("description", ""),
        },
    }


def _small_beam() -> np.ndarray:
    """A 120 x 160 camera frame: spot, pedestal, noise and a few hot pixels."""
    y, x = np.mgrid[0:120, 0:160]
    img = 1800 * np.exp(-0.5 * (((x - 88) / 14) ** 2 + ((y - 56) / 9) ** 2)) + 100
    img = img + rng.normal(0, 30, img.shape)
    for _ in range(12):
        img[rng.integers(0, 120), rng.integers(0, 160)] = 3500
    return img


# One example per step: its parameters, and "use" (when to reach for it).
STEP_EXAMPLES = {
    "background_constant": ({"level": 100}, "Remove a camera's dark pedestal."),
    "background_frame": (
        {"source": "dark"},
        "Subtract a recorded background image (a dark or laser-off frame) bound under inputs.",
    ),
    "circular_mask": (
        {"center": [56, 88], "radius": 40},
        "Keep a round aperture or screen; blank everything outside it.",
    ),
    "clip_above": (
        {"level": 1200},
        "Cap saturated or hot pixels before a measurement.",
    ),
    "clip_below": (
        {"level": 150},
        "Floor the noise at a level (the level stays, not zero).",
    ),
    "crosshair_mask": (
        {"center": [56, 88], "width": 30, "height": 30, "thickness": 2},
        "Blank a fiducial crosshair printed on a screen.",
    ),
    "derivative": (
        {},
        "Differentiate a trace along its axis (e.g. a field integral into the field).",
    ),
    "gaussian": ({"sigma": 2.0}, "Smooth noise before widths or peaks are measured."),
    "interpolate": (
        {"count": 120, "lower": 450, "upper": 850},
        "Resample a trace onto an even axis (e.g. before a waterfall).",
    ),
    "lowpass": (
        {"order": 2, "critical_frequency": 0.1},
        "Zero-phase Butterworth smoothing of a trace (cutoff as a fraction of Nyquist).",
    ),
    "median": ({"kernel": 3}, "Remove isolated hot pixels without blurring edges."),
    "roi": (
        {"bounds": [[20, 95], [45, 135]], "units": "index"},
        "Crop to the region that matters.",
    ),
    "rotate": ({"angle": 20}, "Straighten a tilted image before projections."),
    "zero_below": (
        {"level": 150},
        "Zero pixels below a noise floor (unlike clip_below).",
    ),
}


def step_examples() -> dict:
    """Each step applied alone to a small frame: before and after."""
    from geecs_analysis.registry import definitions

    raw = _small_beam()
    t = np.linspace(400, 900, 250)
    trace = np.exp(-0.5 * ((t - 640) / 40) ** 2) + rng.normal(0, 0.03, t.size)
    line = Frame.from_array(trace, axes=(Axis(values=t, unit="nm"),))
    image = Frame.from_array(raw)
    dark = Frame.from_array(np.full(raw.shape, 100.0) + rng.normal(0, 30, raw.shape))
    out = {"_input": png(raw, 1900)}  # every image step starts from this frame
    for item in definitions():
        name = item.spec.model_fields["step"].default
        params, use = STEP_EXAMPLES[name]
        spec = {"step": name, **params}
        if 2 in item.ndim:
            after = apply_pipeline(
                image, Pipeline.model_validate({"steps": [spec]}), inputs={"dark": dark}
            )
            out[name] = {
                "spec": spec,
                "use": use,
                "after": png(
                    after.data,
                    1900
                    if name not in {"background_constant", "background_frame"}
                    else 1800,
                ),
                "shape": [list(raw.shape), list(after.data.shape)],
            }
        else:
            after = apply_pipeline(line, Pipeline.model_validate({"steps": [spec]}))
            out[name] = {
                "spec": spec,
                "use": use,
                "trace": {"x": t.round(2).tolist(), "y": trace.round(4).tolist()},
                "after_trace": {
                    "x": np.asarray(after.axes[0].values).round(2).tolist(),
                    "y": np.asarray(after.data).round(4).tolist(),
                },
            }
    return out


def fig_png(fig) -> str:
    """A matplotlib Figure as a PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def summary_examples() -> dict:
    """Each summary kind drawn by its registered layout on a synthetic scan."""
    from geecs_schemas.analysis.recipe import (
        AverageSummary,
        ImageGridSummary,
        ScalarFitSummary,
        WaterfallSummary,
    )

    from geecs_analysis.measurement import Measurement
    from geecs_analysis.registry import summary_definitions
    from geecs_analysis.render.specs import FigureSpec

    layouts = {item.spec: item.function for item in summary_definitions()}
    y, x = np.mgrid[0:90, 0:120]
    positions = [-2.0, -1.0, 0.0, 1.0, 2.0]
    beams = []
    for p in positions:
        img = 1000 * np.exp(
            -0.5 * (((x - 60 - 12 * p) / (10 + 3 * abs(p))) ** 2 + ((y - 45) / 7) ** 2)
        )
        beams.append(
            analyze(
                Frame.from_array(img),
                Analysis.model_validate({"steps": [], "measure": {"kind": "none"}}),
            )
        )
    t = np.linspace(400, 900, 200)
    traces = []
    scan = np.linspace(0, 10, 11)
    for p in scan:
        tr = np.exp(-0.5 * ((t - 560 - 18 * p) / 30) ** 2)
        f = Frame.from_array(tr, axes=(Axis(values=t, unit="nm"),))
        traces.append(
            analyze(
                f, Analysis.model_validate({"steps": [], "measure": {"kind": "none"}})
            )
        )
    style = FigureSpec(imshow={"cmap": "magma"})
    grid = layouts[ImageGridSummary](
        beams, positions, "quad current (A)", ImageGridSummary(), style
    )
    fall = layouts[WaterfallSummary](
        traces,
        list(scan),
        "delay (ps)",
        WaterfallSummary(),
        FigureSpec(axes={"xlabel": "wavelength (nm)"}),
    )
    avg = layouts[AverageSummary]([beams[2]], [None], "", AverageSummary(), style)
    # Two magnets' kicks across a transverse scan: each is linear in the
    # position, its zero crossing the magnet's centre. Seeded noise, drawn
    # last so the earlier examples keep their draws.
    rng = np.random.default_rng(7)
    hexapod = np.linspace(-1.0, 1.0, 9)
    kicks = [
        Measurement(
            {
                "kick_1": 2.0 * (p - 0.15) + rng.normal(0, 0.05),
                "kick_2": -1.4 * (p + 0.3) + rng.normal(0, 0.05),
            },
            Frame.from_array(np.zeros(4)),
        )
        for p in hexapod
    ]
    fit = layouts[ScalarFitSummary](
        kicks,
        list(hexapod),
        "hexapod x (mm)",
        ScalarFitSummary(scalars=["kick_1", "kick_2"]),
        FigureSpec(),
    )
    return {
        "image_grid": fig_png(grid),
        "waterfall": fig_png(fall),
        "average": fig_png(avg),
        "scalar_fit": fig_png(fit.figure),
    }


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
    # Last, so adding them left every earlier example's random draws unchanged.
    data["stitch"] = stitch_example()
    data["steps"] = step_examples()
    data["summaries"] = summary_examples()
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
