"""Clean-ups applied to one camera image before it is shown or averaged.

:class:`ImagePrep` bundles two choices and :meth:`ImagePrep.apply` runs
them in order, on a plain 2-D array (a background, when there is one,
comes off first with the ``background_frame`` step):

* X-ray hits out (:func:`remove_hits`): single bright pixels or small
  clusters from stray radiation on the sensor. The ``median`` step at a
  3×3 or 5×5 kernel, or the gentler
  :func:`~geecs_analysis.algorithms.hot_pixels.remove_hot_pixels`, which
  replaces only the pixels far above their neighbours.
* The image centred on its beam:
  :func:`~geecs_analysis.algorithms.beam_centre.beam_centre` (the centre
  of mass above a threshold), then
  :func:`~geecs_analysis.algorithms.beam_centre.centre_on` (a whole-pixel
  shift).

Hits go first, so a hit does not pull the centre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from geecs_analysis.algorithms.beam_centre import beam_centre, centre_on
from geecs_analysis.algorithms.hot_pixels import NOISE_FLOOR, remove_hot_pixels

#: The X-ray-hit filters: none, a 3×3 or 5×5 median, or the hot-pixel
#: filter (only isolated pixels far above all their neighbours change).
HitFilter = Literal["none", "median3", "median5", "hot"]
HIT_FILTERS: tuple[str, ...] = ("none", "median3", "median5", "hot")


def remove_hits(
    image: np.ndarray, method: str, *, noise_floor: float = NOISE_FLOOR
) -> np.ndarray:
    """*image* with X-ray hits removed; the same array for ``"none"``.

    Parameters
    ----------
    image : numpy.ndarray
        A 2-D image.
    method : {"none", "median3", "median5", "hot"}
        ``"median3"``/``"median5"`` are the ``median`` step at that
        kernel: every pixel replaced by its neighbourhood's median (blurs
        a little, removes clusters up to about the kernel's half size).
        ``"hot"`` is
        :func:`~geecs_analysis.algorithms.hot_pixels.remove_hot_pixels`:
        only single-pixel hits change.
    noise_floor : float
        ``"hot"`` only: the least the noise is taken to be, in the
        image's units. The default is one count, right for a camera
        image whose values step by one; an image that steps by more (a
        12-bit sensor left-aligned in 16 bits steps by 16) or is
        normalised needs its own.

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
    if method == "hot":
        return remove_hot_pixels(image, noise_floor=noise_floor)
    from geecs_data_utils.frames import Frame

    from geecs_analysis.steps.median import MedianSpec, median

    filtered = median(Frame.from_array(image), MedianSpec(kernel=int(method[-1])))
    return np.array(
        filtered.data
    )  # the frame's samples are read-only: the caller's own copy


@dataclass(frozen=True)
class ImagePrep:
    """The clean-ups to apply to each shot's image, in order.

    Attributes
    ----------
    hits : str
        One of ``HIT_FILTERS``.
    centre : float or None
        The centring threshold
        (:func:`~geecs_analysis.algorithms.beam_centre.beam_centre`);
        ``None`` leaves the image where it is.
    noise_floor : float
        The ``"hot"`` filter's least noise, in the image's units
        (:func:`remove_hits`); one count unless set.
    """

    hits: str = "none"
    centre: Optional[float] = None
    noise_floor: float = NOISE_FLOOR

    @property
    def active(self) -> bool:
        """Whether this changes an image at all."""
        return self.hits != "none" or self.centre is not None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """*image* with hits removed, then centred on its beam.

        An image with nothing above the centring threshold is left
        where it is.
        """
        out = remove_hits(image, self.hits, noise_floor=self.noise_floor)
        if self.centre is not None and out.ndim == 2:
            found = beam_centre(out, self.centre)
            if found is not None:
                out = centre_on(out, found)
        return out
