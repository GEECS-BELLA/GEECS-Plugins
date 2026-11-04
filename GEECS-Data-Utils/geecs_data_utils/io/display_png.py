"""A camera frame → 8-bit PNG for a browser, with a percentile window.

Raw 16-bit frames render near-black in a browser, so the window maps the
camera's range into display range: the ``plo``–``phi`` percentiles of the
finite pixels (1 / 99.7 by default, robust to hot pixels) become black and
white. Display values ride shared links and saved settings, so a bad value
degrades to the default, never fails: an unsane window falls back to the
defaults and an unknown colour-map name renders greyscale.

Moved from the Data Portal's ``resources.py`` so every web surface renders
a frame the same way.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np

#: Display normalization percentiles (robust to hot pixels).
DEFAULT_WINDOW = (1.0, 99.7)


def window_percentiles(plo, phi) -> tuple[float, float]:
    """The display window's percentile pair — value-degrade semantics.

    Anything that isn't a sane ``0 <= lo < hi <= 100`` pair falls back to
    :data:`DEFAULT_WINDOW`. Each side defaults on its own, so a one-sided
    override (``plo=5`` alone) applies rather than no-opping the pair.

    Parameters
    ----------
    plo, phi : float or None
        Low and high percentiles; ``None`` keeps that side's default.

    Returns
    -------
    tuple of float
        ``(low, high)`` percentiles.
    """
    default_lo, default_hi = DEFAULT_WINDOW
    try:
        lo = default_lo if plo is None else float(plo)
        hi = default_hi if phi is None else float(phi)
    except (TypeError, ValueError):
        return DEFAULT_WINDOW
    if not (0.0 <= lo < hi <= 100.0):
        return DEFAULT_WINDOW
    return lo, hi


def safe_cmap(cmap: Optional[str]) -> Optional[str]:
    """A known colour-map name, else ``None`` (greyscale).

    Known: matplotlib's maps and :mod:`colour_tables`' (GEECSplotter's
    LabVIEW tables, registered with matplotlib here).

    matplotlib is imported only here, and only for a named map; without
    matplotlib installed every name degrades to greyscale.

    Parameters
    ----------
    cmap : str or None
        A matplotlib colour-map name (``viridis``, ``magma``, …).

    Returns
    -------
    str or None
        The name when matplotlib knows it, else ``None``.
    """
    if not cmap:
        return None
    try:
        import matplotlib as mpl
    except ImportError:
        return None
    from geecs_data_utils.io import colour_tables

    colour_tables.register()  # GEECSplotter's LabVIEW tables, by name
    return str(cmap) if str(cmap) in mpl.colormaps else None


def window_levels(
    array: np.ndarray, plo: Optional[float] = None, phi: Optional[float] = None
) -> tuple[float, float]:
    """The pixel values the percentile window maps to black and white.

    Parameters
    ----------
    array : numpy.ndarray
        The image data; non-finite pixels are ignored.
    plo, phi : float, optional
        The window's percentiles (:func:`window_percentiles`; the defaults
        when ``None``).

    Returns
    -------
    tuple of float
        ``(black, white)``; ``(0.0, 0.0)`` for an image with no finite pixel.
    """
    data = np.asarray(array, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if not finite.size:
        return 0.0, 0.0
    lo, hi = np.percentile(finite, window_percentiles(plo, phi))
    return float(lo), float(hi)


def _colourise(norm: np.ndarray, cmap: Optional[str]) -> np.ndarray:
    """0..1 values as 8-bit pixels: through *cmap* (RGB), else greyscale."""
    known = safe_cmap(cmap)  # unknown name: greyscale, never a failure
    if known and norm.ndim == 2:
        import matplotlib as mpl

        # bytes=True skips the H×W×4 float64 RGBA intermediate — same
        # output bytes at ~1/8 the transient memory on full frames.
        return mpl.colormaps[known](norm, bytes=True)[..., :3]
    return (norm * 255).astype(np.uint8)


def _label(value: float) -> str:
    return f"{value:.4g}"


def colour_bar_image(
    cmap: Optional[str],
    levels: tuple[float, float],
    *,
    length: int,
    thickness: Optional[int] = None,
    vertical: bool = True,
):
    """A colour bar for *levels*: the map's ramp and its low, middle and high values.

    Dark text on white, so it reads on any page and pastes cleanly.
    Vertical: the ramp *length* pixels tall (high at the top), the values to
    its right. Horizontal: *length* wide (low at the left), the values
    under it.

    Parameters
    ----------
    cmap : str or None
        The colour map (:func:`safe_cmap`; greyscale when unknown).
    levels : tuple of float
        The values at the ramp's two ends (black and white).
    length : int
        The ramp's length in pixels.
    thickness : int, optional
        The ramp's width across; a share of *length* by default.
    vertical : bool
        Upright beside a picture, or lying under a grid.

    Returns
    -------
    PIL.Image.Image
        An RGB image.
    """
    from PIL import Image, ImageDraw, ImageFont

    length = max(int(length), 16)
    thick = int(thickness) if thickness else max(8, length // 22)
    # Upright beside a picture, the text scales with it (a tile is drawn
    # small, so it must stay readable shrunk); lying under a grid it is
    # shown near its own size.
    font_px = max(10, length // 9) if vertical else max(10, min(28, length // 30))
    font = ImageFont.load_default(size=font_px)
    ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
    lo, hi = levels
    labels = [(0.0, lo), (0.5, (lo + hi) / 2), (1.0, hi)]
    texts = [_label(v) for _, v in labels]
    widest = max(int(font.getlength(t)) for t in texts)
    pad = max(3, font_px // 3)
    if vertical:
        strip = np.repeat(ramp[::-1, None], thick, axis=1)
        width, height = thick + pad + widest + pad, length + font_px
        image = Image.new("RGB", (width, height), "white")
        image.paste(
            Image.fromarray(_colourise(strip, cmap)).convert("RGB"), (0, font_px // 2)
        )
        draw = ImageDraw.Draw(image)
        for (at, _), text in zip(labels, texts):
            y = font_px // 2 + round((1.0 - at) * (length - 1))
            draw.line([(thick, y), (thick + pad // 2, y)], fill=(40, 40, 40))
            draw.text((thick + pad, y), text, fill=(20, 20, 20), font=font, anchor="lm")
    else:
        strip = np.repeat(ramp[None, :], thick, axis=0)
        side = widest // 2 + pad
        width, height = length + 2 * side, thick + pad + font_px + pad
        image = Image.new("RGB", (width, height), "white")
        image.paste(Image.fromarray(_colourise(strip, cmap)).convert("RGB"), (side, 0))
        draw = ImageDraw.Draw(image)
        for (at, _), text in zip(labels, texts):
            x = side + round(at * (length - 1))
            draw.line([(x, thick), (x, thick + pad // 2)], fill=(40, 40, 40))
            draw.text((x, thick + pad), text, fill=(20, 20, 20), font=font, anchor="ma")
    return image


def colour_bar_png(
    cmap: Optional[str], levels: tuple[float, float], *, length: int = 480
) -> bytes:
    """:func:`colour_bar_image`, horizontal, as PNG bytes (one bar for a grid)."""
    buffer = io.BytesIO()
    colour_bar_image(cmap, levels, length=length, vertical=False).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def to_display_png(
    array: np.ndarray,
    *,
    cmap: Optional[str] = None,
    plo: Optional[float] = None,
    phi: Optional[float] = None,
    levels: Optional[tuple[float, float]] = None,
    colour_bar: bool = False,
) -> bytes:
    """Render a 2D (or RGB) array to 8-bit PNG bytes with robust scaling.

    A flat image renders at the scale's floor rather than dividing by zero
    (black in greyscale; the colour map's lowest colour under a ``cmap``).
    ``cmap`` applies to 2D input only — an RGB input keeps its own colours.

    Parameters
    ----------
    array : numpy.ndarray
        The image data.
    cmap : str, optional
        Matplotlib colour-map name; ``None`` or an unknown name renders
        greyscale (:func:`safe_cmap`).
    plo, phi : float, optional
        Percentile window overrides (:func:`window_percentiles`).
    levels : tuple of float, optional
        The pixel values drawn black and white, in place of the percentile
        window: a fixed z scale, or one shared by several images (each
        image's :func:`window_levels`, combined). Non-finite values, or
        white not above black, draw the image at the floor, as a flat
        image does.
    colour_bar : bool
        A slim colour bar to the right of a 2D image, labelled with the
        values its black and white stand for (:func:`colour_bar_image`).

    Returns
    -------
    bytes
        PNG-encoded 8-bit image (RGB when a colour map or bar applied).
    """
    # Pillow arrives with imageio (a declared dependency), which needs it.
    from PIL import Image

    data = np.asarray(array, dtype=np.float32)
    lo, hi = window_levels(data, plo, phi) if levels is None else levels
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        norm = np.zeros(data.shape, dtype=np.float32)
    else:
        norm = np.nan_to_num(np.clip((data - lo) / (hi - lo), 0.0, 1.0))
    picture = Image.fromarray(
        _colourise(norm, cmap) if data.ndim == 2 else (norm * 255).astype(np.uint8)
    )
    if colour_bar and data.ndim == 2 and np.isfinite(lo) and np.isfinite(hi):
        height, width = data.shape
        bar = colour_bar_image(
            cmap, (float(lo), float(hi)), length=height, thickness=max(6, width // 20)
        )
        gap = max(3, width // 60)
        whole = Image.new(
            "RGB", (width + gap + bar.width, max(height, bar.height)), "white"
        )
        whole.paste(picture.convert("RGB"), (0, (whole.height - height) // 2))
        whole.paste(bar, (width + gap, (whole.height - bar.height) // 2))
        picture = whole
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()
