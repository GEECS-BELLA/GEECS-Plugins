"""Single-pixel X-ray hits out of a camera image, the rest left untouched.

A hit (stray radiation on the sensor) lights one pixel and leaves its
neighbours at the background; real light is spread by the optics, so even
a sharp focal spot lights the pixels beside its peak.
:func:`remove_hot_pixels` replaces only the former.
"""

from __future__ import annotations

import numpy as np

#: How far above its brightest neighbour a pixel must be for the filter
#: to replace it, in standard deviations of the noise.
HOT_SIGMAS = 5.0

#: The most a hit's brightest neighbour may stand above the local
#: background, as a share of the hit's own height above it. Even a focal
#: spot of 0.5 px sigma puts 13% of its peak on the next pixel, while an
#: X-ray hit leaves its neighbours at the background.
HOT_NEIGHBOUR_SHARE = 0.05

#: The least the noise is taken to be, in the image's units: one count,
#: a camera's digitisation step. The noise is measured as a median
#: deviation, which is zero when more than half the pixels sit on one
#: value (a dark frame) and far under a count once an averaged background
#: is taken off such a frame; without the floor every single count
#: standing alone there would be a hit.
NOISE_FLOOR = 1.0


def remove_hot_pixels(
    image: np.ndarray, *, noise_floor: float = NOISE_FLOOR
) -> np.ndarray:
    """*image* with single-pixel hits replaced by their 3×3 median, as floats.

    A pixel is a hit when it stands above its brightest neighbour by more
    than ``HOT_SIGMAS`` times the noise, and that neighbour is itself
    barely above the local background (``HOT_NEIGHBOUR_SHARE`` of the
    pixel's height, plus three times the noise). The noise is the robust
    standard deviation of the image less its 3×3 median, and never less
    than *noise_floor*. Every other pixel is returned as it was: a beam's
    peak, however sharp, lights its neighbours too.

    The price: a hit on the bright part of a narrow beam, whose
    neighbours the beam lights by more than ``HOT_NEIGHBOUR_SHARE`` of the
    hit's height, is left in (within one 3×3 neighbourhood it looks like a
    sharp spot on a halo), as is a hit spread over two pixels, and a hit
    of fewer than ``HOT_SIGMAS`` × *noise_floor* above its neighbours.

    Parameters
    ----------
    image : numpy.ndarray
        A 2-D image.
    noise_floor : float
        The least the noise is taken to be, in the image's units (0 or
        more). The default is one camera count; an image in other units
        (normalised, calibrated) needs its own.

    Raises
    ------
    ValueError
        A negative or nonfinite *noise_floor*.
    """
    if not noise_floor >= 0.0 or not np.isfinite(noise_floor):
        raise ValueError(f"the noise floor must be 0 or more, got {noise_floor!r}")
    from scipy.ndimage import maximum_filter, median_filter

    data = np.asarray(image, dtype=float)
    smooth = median_filter(data, size=3, mode="reflect")
    background = median_filter(data, size=5, mode="reflect")
    residual = data - smooth
    measured = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    noise = max(measured, float(noise_floor))
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
