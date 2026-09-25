"""Every bin's (or shot's) trace as one row of a heat map, legacy geometry.

The rows stack index-wise on the first trace's x grid with midpoint cell
edges, exactly as the ScanAnalysis waterfall drew them, rather than through
the general renderer's same-grid ``waterfall``: repeated or nonmonotonic
scan values keep their cells, and sorting by a key spaces rows evenly with
the physical values as tick labels. No interpolation or sorting happens
here; the product planner orders and filters the rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from geecs_schemas.analysis.recipe import WaterfallSummary

from geecs_analysis.registry import summary
from geecs_analysis.render import RenderError

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure
    from geecs_analysis.measurement import Measurement
    from geecs_analysis.render.specs import FigureSpec


def palette(
    data: np.ndarray,
    scale: str,
    *,
    cmap: str | None,
    vmin: float | None,
    vmax: float | None,
    line: bool,
) -> dict:
    """Resolve a colour scale rule against the data, as the legacy renderers did.

    ``auto`` on a trace centres a diverging map on zero when the data crosses
    it; ``sequential`` runs from zero; ``diverging`` is symmetric about zero;
    ``custom`` uses the limits as given (missing ones autoscale).
    """
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    finite = data[np.isfinite(data)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0, 1)
    if scale == "auto" and line and low < 0 < high:
        return {
            "cmap": cmap or "RdBu_r",
            "norm": TwoSlopeNorm(vmin=low, vcenter=0, vmax=high),
        }
    if scale == "diverging":
        limit = max(abs(low), abs(high))
        return {
            "vmin": -limit,
            "vmax": limit,
            "cmap": cmap or ("RdBu_r" if line else "plasma"),
        }
    if scale == "sequential" or (scale == "auto" and line):
        low = min(low, 0) if scale == "auto" else 0
    else:
        low = vmin if vmin is not None else low
    return {
        "vmin": low,
        "vmax": vmax if vmax is not None else high,
        "cmap": cmap or "plasma",
    }


def legacy_edges(values: np.ndarray) -> np.ndarray:
    """Midpoint cell edges around sample centres; a lone sample gets unit width."""
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


@summary(WaterfallSummary, consumes="panels", filename="summary_waterfall")
def waterfall(
    results: Sequence[Measurement],
    positions: Sequence[float | None],
    label: str,
    options: WaterfallSummary,
    figure: FigureSpec,
) -> Figure:
    """Stack the traces at their positions; ``label`` names the scan parameter.

    From ``figure`` this layout takes ``fig`` (dpi), ``axes.xlabel`` (the
    same physical axis as the trace) and ``colorbar.label``; the y axis and
    title are the scan's. A real zero position is retained.
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
    if any(p is None for p in positions):
        raise RenderError("Waterfall positions must be given for every row")
    values = np.asarray(positions, dtype=float)
    if not np.all(np.isfinite(values)):
        raise RenderError("Waterfall positions must be finite")
    even = options.even_spacing
    if even is None:
        even = bool(options.sort_key)
    centers = np.arange(len(values)) if even else values
    data = np.stack([f.data for f in frames])
    try:
        fig = Figure(
            **{
                "figsize": (10, 8),
                "dpi": 150,
                "constrained_layout": True,
                **{k: v for k, v in figure.fig.items() if k != "figsize"},
            }
        )
        ax = fig.subplots()
        artist = ax.pcolormesh(
            legacy_edges(frames[0].axes[0].values),
            legacy_edges(centers),
            data,
            shading="flat",
            **palette(
                data,
                options.scale,
                cmap=options.cmap,
                vmin=options.vmin,
                vmax=options.vmax,
                line=True,
            ),
        )
        indices = np.linspace(0, len(values) - 1, min(40, len(values)), dtype=int)
        ax.set_yticks(centers[indices], [f"{values[i]:.3f}" for i in indices])
        axis = frames[0].axes[0]
        xlabel = axis.label or "x"
        if axis.unit:
            xlabel += f" ({axis.unit})"
        ax.set(xlabel=xlabel, ylabel=label, title=f"Waterfall Plot: {label} Scan")
        if "xlabel" in figure.axes:
            ax.set_xlabel(figure.axes["xlabel"])
        signal = frames[0].label or "Intensity"
        if frames[0].unit:
            signal += f" ({frames[0].unit})"
        fig.colorbar(artist, ax=ax, label=figure.colorbar.get("label", signal))
        FigureCanvasAgg(fig).draw()
        return fig
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"{type(exc).__name__}: {exc}") from exc
