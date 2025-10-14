"""Utilities for Gaussian fitting of LCLS beam images.

This module provides a thin wrapper around :class:`lcls_tools.common.image.fit.ImageProjectionFit`
to fit a 2‑D image with Gaussian profiles along the x‑ and y‑axes. The
resulting parameters are returned in a flattened dictionary with a prefix
indicating the axis.

The implementation follows NumPy‑style docstring conventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np
from lcls_tools.common.image.fit import ImageProjectionFit


def gauss_fit(img: np.ndarray) -> dict[str, float]:
    """Fit Gaussian projections to a beam image.

    Parameters
    ----------
    img : np.ndarray
        2‑D array representing the beam image to be fitted. The array is
        expected to contain intensity values; no preprocessing is performed
        inside this function.

    Returns
    -------
    dict[str, float]
        Dictionary containing the fitted parameters for the x‑ and y‑axis
        projections. Keys are prefixed with ``'gaussian_fit_x_'`` or
        ``'gaussian_fit_y_'`` respectively, e.g. ``gaussian_fit_x_amplitude``.
        All values are converted to Python ``float`` for JSON‑compatibility.

    Notes
    -----
    The function uses :class:`lcls_tools.common.image.fit.ImageProjectionFit`,
    which internally performs the following steps:

    1. Extracts one‑dimensional projections of the image along each axis.
    2. Fits each projection with a Gaussian model.
    3. Stores the fit parameters (amplitude, mean, sigma, and offset) for
       both axes.

    The returned dictionary merges the two sets of parameters into a single
    flat mapping.

    Examples
    --------
    >>> import numpy as np
    >>> from image_analysis.tools.lcls_tools_gauss_fit import gauss_fit
    >>> # Create a synthetic Gaussian image
    >>> x = np.linspace(-5, 5, 100)
    >>> y = np.linspace(-5, 5, 100)
    >>> xv, yv = np.meshgrid(x, y)
    >>> img = np.exp(-(xv**2 + yv**2))
    >>> params = gauss_fit(img)
    >>> sorted(params.keys())[:3]
    ['gaussian_fit_x_amplitude', 'gaussian_fit_x_mean', 'gaussian_fit_x_offset']
    """
    # Perform the fit using the LCLS tools library
    result = ImageProjectionFit().fit_image(img)

    # Prefixes for the dictionary keys
    prefix_x = "gaussian_fit_x_"
    prefix_y = "gaussian_fit_y_"

    # Prefix the individual parameter dictionaries
    prefixed_dict_x = {
        f"{prefix_x}{k}": float(v)
        for k, v in result.x_projection_fit_parameters.items()
    }
    prefixed_dict_y = {
        f"{prefix_y}{k}": float(v)
        for k, v in result.y_projection_fit_parameters.items()
    }

    # Combine both axes into a single result dictionary
    combined_dict = {**prefixed_dict_x, **prefixed_dict_y}
    return combined_dict
