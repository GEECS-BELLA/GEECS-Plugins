from __future__ import annotations

import math
import numpy as np
from scipy.signal import savgol_filter

def fwhm(sd):
    """ Full-width half-maximum of a normal distribution with standard deviation sd
    """
    return 2 * np.sqrt(2 * np.log(2)) * sd

def n_sigma_window(dist: np.ndarray, n_sigmas: float = 5) -> np.ndarray:
    # mu: float = np.sum(np.arange(dist.size) * dist) / np.sum(dist)
    sig: float = np.sqrt(np.sum(((np.arange(dist.size) * dist)**2) / np.sum(dist)**2))
    dist = savgol_filter(dist, max(int(min(len(dist) / 6., sig / 2.)), 4), 3)

    center = np.argmax(dist)
    mid_level = (dist[center] + np.min(dist)) / 2.

    left = np.where(dist[:center] <= mid_level)[0]
    if left.any():
        left = left[-1]
    else:
        left = 0

    right = np.where(dist[center:] <= mid_level)[0]
    if right.any():
        right = right[0] + center
    else:
        right = dist.size - 1

    n_sig = n_sigmas * (right - left) / (2 * math.sqrt(2 * math.log(2.)))

    return np.array([max(0, center - n_sig), min(dist.size - 1, center + n_sig)])
