"""Object-API rendering of frames and measurements; no pyplot or file writes."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Sequence

from geecs_analysis.render.specs import FigureSpec

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from geecs_data_utils.frames import Axis, Frame
    from geecs_analysis.measurement import Measurement


class RenderError(ValueError):
    """A frame or user-supplied style could not be drawn."""


def _edges(axis: Axis) -> np.ndarray:
    """Infer pixel edges from monotonic sample centers without resampling data."""
    import numpy as np

    values = axis.values
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    delta = np.diff(values)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise RenderError("Image axes must be strictly monotonic")
    return np.concatenate(
        (
            [values[0] - delta[0] / 2],
            values[:-1] + delta / 2,
            [values[-1] + delta[-1] / 2],
        )
    )


def _label(name: str, unit: str) -> str:
    return f"{name} ({unit})" if unit else name


def draw_frame(ax: Axes, frame: Frame, style: FigureSpec | None = None) -> None:
    """Draw a 1D trace or calibrated image into supplied object-API axes."""
    import numpy as np

    style = style or FigureSpec()
    if frame.data.ndim == 1:
        ax.plot(frame.axes[0].values, frame.data, **deepcopy(style.plot))
        ax.set_xlabel(_label(frame.axes[0].label or "x", frame.axes[0].unit))
        ax.set_ylabel(_label(frame.label or "signal", frame.unit))
        return
    y, x = frame.axes
    xe, ye = _edges(x), _edges(y)
    uniform = all(
        len(axis.values) < 3
        or np.allclose(
            np.diff(axis.values), np.diff(axis.values)[0], rtol=1e-10, atol=0
        )
        for axis in (x, y)
    )
    if "extent" in style.imshow:
        raise RenderError("Image extent is derived from Frame coordinates")
    if uniform:
        kwargs = {"cmap": "plasma", "origin": "upper", **deepcopy(style.imshow)}
        origin = kwargs.get("origin")
        extent = (
            (xe[0], xe[-1], ye[-1], ye[0])
            if origin == "upper"
            else (xe[0], xe[-1], ye[0], ye[-1])
        )
        artist = ax.imshow(frame.data, extent=extent, **kwargs)
    else:
        kwargs = {"cmap": "plasma", "shading": "flat", **deepcopy(style.pcolormesh)}
        artist = ax.pcolormesh(xe, ye, frame.data, **kwargs)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[-1], ye[0])
    ax.set_xlabel(_label(x.label or "x", x.unit))
    ax.set_ylabel(_label(y.label or "y", y.unit))
    colorbar = deepcopy(style.colorbar)
    if colorbar.pop("show", True):
        ax.figure.colorbar(artist, ax=ax, **colorbar)


def draw_overlays(
    ax: Axes, result: Measurement, style: FigureSpec | None = None
) -> None:
    """Draw typed overlays using their frame coordinates and stable styling ids."""
    import numpy as np
    from geecs_analysis.measurement import Marker, Projection

    style = style or FigureSpec()
    for overlay in result.overlays:
        kwargs = deepcopy(style.overlays.get(overlay.id, {}))
        if kwargs.pop("hidden", False):
            continue
        if isinstance(overlay, Marker):
            ax.plot(
                [overlay.x],
                [overlay.y],
                **{
                    "marker": "+",
                    "markersize": 12,
                    "color": "cyan",
                    "linestyle": "none",
                    **kwargs,
                },
            )
        elif isinstance(overlay, Projection):
            if result.frame.data.ndim != 2:
                raise RenderError("Projection overlays require an image measurement")
            scale = float(kwargs.pop("scale", 0.2))
            if not np.isfinite(scale) or scale < 0:
                raise RenderError("Projection scale must be finite and nonnegative")
            values = np.maximum(overlay.frame.data, 0)
            finite = np.isfinite(values)
            peak = np.max(values[finite]) if np.any(finite) else 0
            if peak <= 0:
                continue
            values = np.where(finite, values / peak, np.nan)
            coordinates = overlay.frame.axes[0].values
            if overlay.axis == 1:
                bottom, top = ax.get_ylim()
                ax.plot(
                    coordinates,
                    bottom + values * (top - bottom) * scale,
                    **{"color": "cyan", "linewidth": 1, **kwargs},
                )
            else:
                left, right = ax.get_xlim()
                ax.plot(
                    left + values * (right - left) * scale,
                    coordinates,
                    **{"color": "magenta", "linewidth": 1, **kwargs},
                )


def single(result: Measurement, style: FigureSpec | None = None) -> Figure:
    """Render one measurement in a fresh Figure; caller decides if/where to save."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    style = style or FigureSpec()
    try:
        fig = Figure(
            **{
                "figsize": (5.0, 4.2),
                "dpi": 110,
                "constrained_layout": True,
                **deepcopy(style.fig),
            }
        )
        ax = fig.subplots()
        draw_frame(ax, result.frame, style)
        if result.frame.data.ndim == 2:
            ax.set_autoscale_on(False)
        draw_overlays(ax, result, style)
        ax.set(**deepcopy(style.axes))
        # Many matplotlib checks (normalization, text, layout) are deferred until
        # drawing. Validate here so preview failures stay inside RenderError.
        FigureCanvasAgg(fig).draw()
        return fig
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"{type(exc).__name__}: {exc}") from exc


def waterfall(
    frames: Sequence[Frame], positions: Axis, style: FigureSpec | None = None
) -> Figure:
    """Stack same-grid traces at explicit scan/bin coordinates without resampling.

    Positions follow the supplied frame order and must be strictly monotonic.
    Sources/runners own grouping and averaging; this layout only draws them.
    Different x grids are rejected so no interpolation is hidden in rendering.
    """
    import numpy as np
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement

    if not frames or len(frames) != len(positions.values):
        raise RenderError("Waterfall requires one position per nonempty trace list")
    first = frames[0]
    if any(frame.data.ndim != 1 for frame in frames):
        raise RenderError("Waterfall requires 1D frames")
    if any(
        not np.array_equal(frame.axes[0].values, first.axes[0].values)
        or frame.axes[0].unit != first.axes[0].unit
        or frame.unit != first.unit
        for frame in frames
    ):
        raise RenderError("Waterfall traces must share an x grid and units")
    frame = Frame.from_array(
        np.stack([frame.data for frame in frames]),
        axes=(positions, first.axes[0]),
        unit=first.unit,
    )
    style = style or FigureSpec()
    style = style.model_copy(update={"imshow": {"aspect": "auto", **style.imshow}})
    return single(Measurement({}, frame), style)
