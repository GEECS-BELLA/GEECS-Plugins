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
    """A known matplotlib colour-map name, else ``None`` (greyscale).

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
    return str(cmap) if str(cmap) in mpl.colormaps else None


def to_display_png(
    array: np.ndarray,
    *,
    cmap: Optional[str] = None,
    plo: Optional[float] = None,
    phi: Optional[float] = None,
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

    Returns
    -------
    bytes
        PNG-encoded 8-bit image (RGB when a colour map applied).
    """
    # Pillow arrives with imageio (a declared dependency), which needs it.
    from PIL import Image

    data = np.asarray(array, dtype=np.float32)
    lo_pct, hi_pct = (
        window_percentiles(plo, phi)
        if (plo is not None or phi is not None)
        else DEFAULT_WINDOW
    )
    finite = data[np.isfinite(data)]
    if finite.size:
        lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    else:
        lo, hi = 0.0, 0.0
    if hi <= lo:
        norm = np.zeros(data.shape, dtype=np.float32)
    else:
        norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    colormap = None
    known = safe_cmap(cmap)  # unknown name: greyscale, never a failure
    if known and data.ndim == 2:
        import matplotlib as mpl

        colormap = mpl.colormaps[known]
    if colormap is not None:
        # bytes=True skips the H×W×4 float64 RGBA intermediate — same
        # output bytes at ~1/8 the transient memory on full frames.
        scaled = colormap(norm, bytes=True)[..., :3]
    else:
        scaled = (norm * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(scaled).save(buffer, format="PNG")
    return buffer.getvalue()
