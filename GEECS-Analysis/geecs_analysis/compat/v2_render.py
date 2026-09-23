"""Translate v2 figure options and legacy waterfall geometry; never write files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from geecs_analysis.render import RenderError, image_grid, single
from geecs_analysis.render.specs import FigureSpec

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure
    from geecs_schemas.analysis.renderer import RendererOptions
    from geecs_analysis.measurement import Measurement


def _palette(data: np.ndarray, options: RendererOptions, *, line: bool) -> dict:
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    finite = data[np.isfinite(data)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0, 1)
    mode = options.colormap_mode or ("auto" if line else "sequential")
    cmap = options.cmap or "plasma"
    if mode == "auto" and line and low < 0 < high:
        return {
            "cmap": options.cmap or "RdBu_r",
            "norm": TwoSlopeNorm(vmin=low, vcenter=0, vmax=high),
        }
    if mode == "diverging":
        limit = max(abs(low), abs(high))
        return {
            "vmin": -limit,
            "vmax": limit,
            "cmap": options.cmap or ("RdBu_r" if line else "plasma"),
        }
    if mode == "sequential" or (mode == "auto" and line):
        low = min(low, 0) if mode == "auto" else 0
    else:
        low = options.vmin if options.vmin is not None else low
    return {
        "vmin": low,
        "vmax": options.vmax if options.vmax is not None else high,
        "cmap": cmap,
    }


def _axes(options: RendererOptions) -> dict[str, str]:
    return {
        key: getattr(options, key)
        for key in ("xlabel", "ylabel")
        if getattr(options, key) is not None
    }


def single_v2(
    result: Measurement, options: RendererOptions, *, title: str | None = None
) -> Figure:
    """Render a v2 single product with configured labels and color limits."""
    line = result.frame.data.ndim == 1
    side = options.figsize_inches or 4
    palette = {} if line else _palette(result.frame.data, options, line=False)
    style = FigureSpec(
        imshow=palette,
        pcolormesh=palette,
        axes={**_axes(options), **({"title": title} if title else {})},
        colorbar={"label": options.colorbar_label or "Intensity"},
        fig={"figsize": (8, 6) if line else (side, side), "dpi": options.dpi or 150},
    )
    return single(result, style)


def image_grid_v2(
    results: Sequence[Measurement],
    positions: Sequence[float | None],
    options: RendererOptions,
) -> Figure:
    """Draw v2 bin images with one palette computed across all panels."""
    import math
    import numpy as np

    if not results or len(results) != len(positions):
        raise RenderError("Image grid requires one position per measurement")
    columns = math.ceil(math.sqrt(len(results)))
    rows = math.ceil(len(results) / columns)
    width, height = options.figsize or (6, 6)
    palette = _palette(
        np.concatenate([r.frame.data.ravel() for r in results]), options, line=False
    )
    return image_grid(
        results,
        titles=[
            f"{p:.2f}" if p is not None else str(i + 1) for i, p in enumerate(positions)
        ],
        style=FigureSpec(
            imshow=palette,
            pcolormesh=palette,
            axes=_axes(options),
            colorbar={"label": options.colorbar_label or "Intensity"},
            fig={
                "figsize": (columns * width, rows * height),
                "dpi": options.dpi or 150,
            },
        ),
    )


def _legacy_edges(values: np.ndarray) -> np.ndarray:
    import numpy as np

    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    return np.concatenate(
        (
            [values[0] - (values[1] - values[0]) / 2],
            (values[:-1] + values[1:]) / 2,
            [values[-1] + (values[-1] - values[-2]) / 2],
        )
    )


def waterfall_v2(
    results: Sequence[Measurement],
    positions: Sequence[float],
    label: str,
    options: RendererOptions,
) -> Figure:
    """Draw the v2 waterfall's index-wise stack, using the first trace's x axis.

    This deliberately preserves legacy geometry without weakening the general
    renderer's same-grid requirement. No interpolation or sorting occurs here.
    Repeated/nonmonotonic scan values retain their v2 midpoint cells; optional
    even spacing uses row indices with physical values as tick labels. A real
    zero position is retained (the old renderer replaced it with the bin id).
    """
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not results or len(results) != len(positions):
        raise RenderError("Waterfall requires one position per measurement")
    frames = [r.frame for r in results]
    if any(f.data.ndim != 1 or f.data.shape != frames[0].data.shape for f in frames):
        raise RenderError("Waterfall requires equal-length traces")
    if any(
        f.unit != frames[0].unit or f.axes[0].unit != frames[0].axes[0].unit
        for f in frames
    ):
        raise RenderError("Waterfall requires matching trace units")
    values = np.asarray(positions, dtype=float)
    if not np.all(np.isfinite(values)):
        raise RenderError("Waterfall positions must be finite")
    even = options.waterfall_even_y_spacing
    if even is None:
        even = bool(options.waterfall_sort_key)
    centers = np.arange(len(values)) if even else values
    data = np.stack([f.data for f in frames])
    try:
        fig = Figure(figsize=(10, 8), dpi=options.dpi or 150, constrained_layout=True)
        ax = fig.subplots()
        artist = ax.pcolormesh(
            _legacy_edges(frames[0].axes[0].values),
            _legacy_edges(centers),
            data,
            shading="flat",
            **_palette(data, options, line=True),
        )
        indices = np.linspace(0, len(values) - 1, min(40, len(values)), dtype=int)
        ax.set_yticks(centers[indices], [f"{values[i]:.3f}" for i in indices])
        axis = frames[0].axes[0]
        xlabel = axis.label or "x"
        if axis.unit:
            xlabel += f" ({axis.unit})"
        ax.set(xlabel=xlabel, ylabel=label, title=f"Waterfall Plot: {label} Scan")
        ax.set(**_axes(options))
        signal = frames[0].label or "Intensity"
        if frames[0].unit:
            signal += f" ({frames[0].unit})"
        fig.colorbar(artist, ax=ax, label=options.colorbar_label or signal)
        FigureCanvasAgg(fig).draw()
        return fig
    except Exception as exc:
        raise RenderError(f"{type(exc).__name__}: {exc}") from exc
