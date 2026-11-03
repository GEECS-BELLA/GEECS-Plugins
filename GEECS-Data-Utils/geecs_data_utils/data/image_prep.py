"""Clean-ups applied to one camera image before it is shown or averaged.

Three steps, each pure numpy/scipy on a 2-D array:

* :func:`remove_hits` — X-ray hits (single bright pixels or small clusters
  from stray radiation on the sensor): a median filter, or a gentler
  filter that replaces only the pixels far above their neighbours.
* :func:`beam_centre` — the centre of mass of the pixels above a
  threshold: counts when the threshold is 1 or more, a fraction of the
  peak below 1.
* :func:`centre_on` — the image shifted (whole pixels) so a point lands
  at its middle.

:class:`ImagePrep` bundles the choices and :meth:`ImagePrep.apply` runs
them in that order (hits out first, so a hit does not pull the centre).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

#: The X-ray-hit filters: none, a 3×3 or 5×5 median, or the hot-pixel
#: filter (only isolated pixels far above all their neighbours change).
HitFilter = Literal["none", "median3", "median5", "hot"]
HIT_FILTERS: tuple[str, ...] = ("none", "median3", "median5", "hot")

#: How far above its brightest neighbour a pixel must be for the
#: hot-pixel filter to replace it, in robust standard deviations of the noise.
HOT_SIGMAS = 5.0

#: The most a hit's brightest neighbour may stand above the local
#: background, as a share of the hit's own height above it. Real light is
#: spread by the optics: even a focal spot of 0.5 px sigma puts 13% of its
#: peak on the next pixel, while an X-ray hit leaves its neighbours at the
#: background.
HOT_NEIGHBOUR_SHARE = 0.05


def remove_hits(image: np.ndarray, method: str) -> np.ndarray:
    """*image* with X-ray hits removed; the same array for ``"none"``.

    Parameters
    ----------
    image : numpy.ndarray
        A 2-D image.
    method : {"none", "median3", "median5", "hot"}
        ``"median3"``/``"median5"`` replace every pixel by its
        neighbourhood's median (blurs a little, removes clusters up to
        about the kernel's half size). ``"hot"`` replaces only single-pixel
        hits — a pixel above its brightest neighbour by more than
        ``HOT_SIGMAS`` robust standard deviations of the noise, whose
        brightest neighbour is itself barely above the local background
        (``HOT_NEIGHBOUR_SHARE`` of the pixel's height, plus the noise) —
        with its 3×3 median, leaving the rest of the image untouched: a
        beam's peak, however sharp, lights its neighbours too. The price:
        a hit on the bright part of a narrow beam, whose neighbours the
        beam lights by more than ``HOT_NEIGHBOUR_SHARE`` of the hit's
        height, is left in (within one 3×3 neighbourhood it looks like a
        sharp spot on a halo), as is a hit spread over two pixels — the
        medians take both.

    Returns
    -------
    numpy.ndarray
        A float array (or *image* itself for ``"none"``).

    Raises
    ------
    ValueError
        An unknown method.
    """
    if method == "none":
        return image
    if method not in HIT_FILTERS:
        raise ValueError(f"unknown hit filter {method!r}: one of {HIT_FILTERS}")
    from scipy.ndimage import maximum_filter, median_filter

    data = np.asarray(image, dtype=float)
    if method in ("median3", "median5"):
        return median_filter(data, size=int(method[-1]), mode="reflect")
    smooth = median_filter(data, size=3, mode="reflect")
    background = median_filter(data, size=5, mode="reflect")
    residual = data - smooth
    noise = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False  # the eight neighbours, not the pixel itself
    # "mirror", not "reflect": reflect copies an edge pixel into its own
    # neighbourhood, so a hit on the edge would hide behind itself.
    brightest = maximum_filter(data, footprint=ring, mode="mirror")
    height = data - background
    # A hit stands far above its brightest neighbour, and that neighbour
    # is at the background: no optics puts all of a beam into one pixel.
    hot = (data - brightest > HOT_SIGMAS * noise) & (
        brightest - background < HOT_NEIGHBOUR_SHARE * height + 3.0 * noise
    )
    return np.where(hot, smooth, data)


def threshold_level(image: np.ndarray, threshold: float) -> float:
    """The pixel value a threshold names: counts at 1 or more, else a fraction of the peak."""
    if threshold >= 1.0:
        return float(threshold)
    return float(threshold) * float(np.nanmax(image))


def beam_centre(image: np.ndarray, threshold: float) -> Optional[tuple[float, float]]:
    """The centre of mass ``(row, column)`` of the pixels above *threshold*.

    Each pixel above the level weighs its value minus the level, so the
    background under the beam does not drag the centre towards the
    image's middle.

    Parameters
    ----------
    image : numpy.ndarray
        A 2-D image.
    threshold : float
        Counts when 1 or more; below 1, a fraction of the image's peak
        (0.5: half the brightest pixel).

    Returns
    -------
    tuple of float or None
        ``None`` when no pixel is above the level.
    """
    data = np.asarray(image, dtype=float)
    weights = data - threshold_level(data, threshold)
    weights[~(weights > 0)] = 0.0  # below the level, and NaN
    total = float(weights.sum())
    if total <= 0.0:
        return None
    rows, cols = np.indices(data.shape[:2])
    return float((rows * weights).sum() / total), float((cols * weights).sum() / total)


def centre_on(image: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
    """*image* shifted by whole pixels so *centre* lands at its middle.

    Pixels shifted in from past the edge take the image's median (its
    background, for a beam image), so they neither darken nor brighten
    the levels. The shape is kept.
    """
    data = np.asarray(image, dtype=float)
    height, width = data.shape[:2]
    dy = int(round((height - 1) / 2 - centre[0]))
    dx = int(round((width - 1) / 2 - centre[1]))
    out = np.full_like(data, float(np.nanmedian(data)))
    if abs(dy) >= height or abs(dx) >= width:
        return out
    src_y = slice(max(0, -dy), height - max(0, dy))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    out[dst_y, dst_x] = data[src_y, src_x]
    return out


@dataclass(frozen=True)
class ImagePrep:
    """The clean-ups to apply to each shot's image, in order.

    Attributes
    ----------
    hits : str
        One of ``HIT_FILTERS``.
    centre : float or None
        The centring threshold (see :func:`beam_centre`); ``None`` leaves
        the image where it is.
    """

    hits: str = "none"
    centre: Optional[float] = None

    @property
    def active(self) -> bool:
        """Whether this changes an image at all."""
        return self.hits != "none" or self.centre is not None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """*image* with hits removed, then centred on its beam.

        An image with nothing above the centring threshold is left
        where it is.
        """
        out = remove_hits(image, self.hits)
        if self.centre is not None and out.ndim == 2:
            found = beam_centre(out, self.centre)
            if found is not None:
                out = centre_on(out, found)
        return out
