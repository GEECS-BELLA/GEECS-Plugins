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


# Keywords that place the colorbar themselves; when a recipe passes any of them
# the layout is the user's, and the colorbar is not refitted to the image.
_COLORBAR_PLACEMENT_KEYS = frozenset(
    {"location", "orientation", "shrink", "anchor", "panchor"}
)


def _add_colorbar(fig: Figure, artist, axes: list[Axes], kwargs: dict) -> None:
    """Add a colorbar whose long side spans the drawn images, not their slots.

    Fixed-aspect images shrink inside the slot the layout engine gives their
    axes, while a colorbar placed with ``ax=`` keeps the full slot height. A
    locator re-derives the colorbar box at every draw (layout reruns on each
    save): it keeps the layout's pad and width rules but maps the slot's
    vertical extent onto the union of the axes as drawn, so a colorbar on an
    aspect-free axes is unchanged.
    """
    from matplotlib.transforms import Bbox

    colorbar = fig.colorbar(artist, ax=axes, **kwargs)
    if _COLORBAR_PLACEMENT_KEYS & kwargs.keys():
        return
    base = colorbar.ax.get_axes_locator()

    def locate(cax: Axes, renderer) -> Bbox:
        slot = cax.get_position(original=True)
        pos = base(cax, renderer) if base else slot
        laid_out = Bbox.union([ax.get_position(original=True) for ax in axes])
        for ax in axes:
            ax.apply_aspect()
        drawn = Bbox.union([ax.get_position(original=False) for ax in axes])
        scale = drawn.height / slot.height
        return Bbox.from_bounds(
            drawn.x1 + (pos.x0 - laid_out.x1),
            drawn.y0 + (pos.y0 - slot.y0) * scale,
            pos.width,
            pos.height * scale,
        )

    colorbar.ax.set_axes_locator(locate)


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
        _add_colorbar(ax.figure, artist, [ax], colorbar)


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


def image_grid(
    results: Sequence[Measurement],
    *,
    titles: Sequence[str] | None = None,
    columns: int | None = None,
    style: FigureSpec | None = None,
) -> Figure:
    """Draw image measurements with one shared color scale and colorbar.

    Each panel retains its own coordinate axes and typed overlays. A palette
    supplied in imshow or pcolormesh applies to both artist types; conflicting
    palettes are refused so the shared colorbar remains truthful. Missing
    limits autoscale across all finite image samples, never just the first
    panel. No averaging, resampling, pyplot state or file writes occur here.
    """
    import math
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not results or any(r.frame.data.ndim != 2 for r in results):
        raise RenderError("Image grid requires nonempty 2D measurements")
    if titles is not None and len(titles) != len(results):
        raise RenderError("Image grid requires one title per panel")
    if columns is not None and (type(columns) is not int or columns < 1):
        raise RenderError("Image grid columns must be a positive integer")
    if len({r.frame.unit for r in results}) != 1:
        raise RenderError("A shared image scale requires matching signal units")
    style = style or FigureSpec()
    try:
        palette = {}
        for key in ("cmap", "norm", "vmin", "vmax"):
            if (
                key in style.imshow
                and key in style.pcolormesh
                and style.imshow[key] != style.pcolormesh[key]
            ):
                raise RenderError(f"Conflicting grid palette option: {key}")
            if key in style.imshow or key in style.pcolormesh:
                palette[key] = deepcopy(
                    style.imshow[key] if key in style.imshow else style.pcolormesh[key]
                )
        from matplotlib.cm import ScalarMappable

        requested_norm = palette.get("norm")
        if (
            requested_norm is not None
            and not isinstance(requested_norm, str)
            and any(palette.get(key) is not None for key in ("vmin", "vmax"))
        ):
            raise RenderError("Set limits on the Normalize object instead of vmin/vmax")
        norm = ScalarMappable(norm=requested_norm).norm
        for key in ("vmin", "vmax"):
            if palette.get(key) is not None:
                setattr(norm, key, palette[key])
        # Logarithmic/custom normalizers need the samples in their valid domain,
        # not just the numerical extrema (which may include a zero background).
        samples = np.concatenate(
            [r.frame.data[np.isfinite(r.frame.data)] for r in results]
        )
        norm.autoscale_None(samples if samples.size else np.array([0.0, 1.0]))
        del samples
        # A colorbar expands degenerate limits in place. Set them consistently
        # before draw_frame makes independent copies for each panel.
        from matplotlib.transforms import nonsingular

        if norm.vmin > norm.vmax:
            raise RenderError("Grid color minimum exceeds maximum")
        if norm.vmin == norm.vmax:
            norm.vmin, norm.vmax = nonsingular(norm.vmin, norm.vmax, expander=0.1)
        palette = {
            "norm": norm,
            **({"cmap": palette["cmap"]} if "cmap" in palette else {}),
        }
        panel_style = style.model_copy(
            update={
                "imshow": {
                    **{
                        k: v
                        for k, v in style.imshow.items()
                        if k not in ("norm", "vmin", "vmax", "cmap")
                    },
                    **palette,
                },
                "pcolormesh": {
                    **{
                        k: v
                        for k, v in style.pcolormesh.items()
                        if k not in ("norm", "vmin", "vmax", "cmap")
                    },
                    **palette,
                },
                "colorbar": {"show": False},
            }
        )
        columns = min(columns or math.ceil(math.sqrt(len(results))), len(results))
        rows = math.ceil(len(results) / columns)
        fig = Figure(
            **{
                "figsize": (columns * 4.0, rows * 3.5),
                "dpi": 110,
                "constrained_layout": True,
                **deepcopy(style.fig),
            }
        )
        axes = fig.subplots(rows, columns, squeeze=False).ravel()
        artists = []
        for index, (ax, result) in enumerate(zip(axes, results)):
            draw_frame(ax, result.frame, panel_style)
            ax.set_autoscale_on(False)
            draw_overlays(ax, result, panel_style)
            ax.set(**deepcopy(style.axes))
            if titles is not None:
                ax.set_title(titles[index])
            artists.append(ax.images[0] if ax.images else ax.collections[0])
        for ax in axes[len(results) :]:
            ax.set_visible(False)
        colorbar = deepcopy(style.colorbar)
        if colorbar.pop("show", True):
            _add_colorbar(fig, artists[0], list(axes[: len(results)]), colorbar)
        FigureCanvasAgg(fig).draw()
        return fig
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"{type(exc).__name__}: {exc}") from exc
