"""Bowtie beam-profile fitting utilities.

This module provides:
- `BowtieFitResult`: Dataclass container for fitted parameters and diagnostics.
- `BowtieFitAlgorithm`: Column-wise size extraction and Gaussian-beam model fit
  for "bowtie"-shaped beam profiles in 2D images.
"""

import numpy as np

import logging
import warnings

from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import curve_fit, OptimizeWarning

logger = logging.getLogger(__name__)


@dataclass
class BowtieFitResult:
    """
    Encapsulates the result of a bowtie beam profile fit.

    Attributes
    ----------
    score : float
        Quality metric defined as ``w0 * |theta|`` (smaller is typically better).
    w0 : float
        Minimum beam size (waist) in pixels (or calibrated units if input was scaled).
    theta : float
        Angular divergence in ``pixels^-1`` (magnitude only in final result).
    x0 : float
        Horizontal location (column index) of the beam waist.
    y0 : float
        Vertical location (row index) of the beam at the waist column ``x0``,
        interpolated from `centers`; NaN when the fit failed or ``x0`` falls
        outside the span of columns that entered the fit. This is a row index
        in the image handed to `evaluate`, so any ROI crop or transform the
        caller applied beforehand is already baked into it.
    r_squared : float
        Coefficient of determination for the fit; ``-inf`` if undefined.
    param_errors : Tuple[float, float, float]
        One-sigma uncertainties for ``(w0, theta, x0)`` derived from covariance.
    sizes : numpy.ndarray
        Column-wise vertical size estimates used in the fit (NaN where invalid).
    weights : numpy.ndarray
        Column-wise total intensities used as weights.
    centers : numpy.ndarray
        Column-wise intensity-weighted vertical centers (NaN where invalid) —
        the beam's vertical track across the image, from which `y0` is read.
    """

    score: float
    w0: float
    theta: float
    x0: float
    y0: float
    r_squared: float
    param_errors: Tuple[float, float, float]
    sizes: np.array
    weights: np.array
    centers: np.array


class BowtieFitAlgorithm:
    """
    Evaluate bowtie-shaped profile.

    Achieved by extracting column-wise vertical sizes and fitting a Gaussian-beam-like divergence model.

    The model is:
        ``sigma(x) = sqrt(w0^2 + ((x - x0) * theta)^2)``

    Parameters
    ----------
    threshold_factor : float, default=0.0
        Reserved for external preprocessing/thresholding conventions (not used internally here).
    n_beam_size_clearance : float, default=4.0
        Number of vertical sigmas required to be fully within the image for a column to be valid.
    min_total_counts : float, default=0.0
        Minimum total column intensity required to attempt a size estimate.
    beam_size_func : callable, optional
        Custom function ``f(col: np.ndarray) -> float`` that returns a vertical size for one column.
        If not provided, an RMS size estimator with intensity weights is used.

    Notes
    -----
    - The algorithm stores the last extracted profile via `get_last_profile()`.
    - Weights are used to derive per-point uncertainties for the fit (``sigma_err ~ 1/sqrt(weight)``).
    """

    def __init__(
        self,
        threshold_factor: float = 0.0,
        n_beam_size_clearance: float = 4.0,
        min_total_counts: float = 0.0,
        beam_size_func: Optional[Callable[[np.ndarray], float]] = None,
    ):
        self.threshold_factor = threshold_factor
        self.n_beam_size_clearance = n_beam_size_clearance
        self.min_total_counts = min_total_counts
        self.beam_size_func = beam_size_func
        self._last_profile: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = None

    def get_last_profile(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Return the last ``(sizes, weights, valid_mask, centers)`` extracted by `evaluate()`, or None.

        ``centers`` was appended to the tuple when `BowtieFitResult.y0` was
        added; positional access to the first three entries is unchanged.
        """
        return self._last_profile

    @staticmethod
    def gaussian_fit_beam_size(col: np.ndarray) -> float:
        """Estimate column size via a Gaussian fit and return the fitted sigma."""
        y = np.arange(len(col))
        total = col.sum()
        if total <= 0:
            return np.nan
        mu0 = float(np.average(y, weights=col))
        sigma0 = float(np.sqrt(np.average((y - mu0) ** 2, weights=col)))

        def _gaussian(
            x: np.ndarray, amplitude: float, mu: float, sigma: float, offset: float
        ) -> np.ndarray:
            return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + offset

        try:
            popt, _ = curve_fit(
                _gaussian,
                y,
                col.astype(float),
                p0=[float(col.max()), mu0, sigma0, 0.0],
                maxfev=2000,
            )
            return abs(popt[2])
        except Exception:
            return np.nan

    def _default_beam_size_func(self, y: np.ndarray) -> Callable[[np.ndarray], float]:
        """Return a weighted RMS size estimator over a fixed vertical axis."""

        def rms_beam_size(col: np.ndarray) -> float:
            if col.sum() <= 0:
                return np.nan
            y_center = np.average(y, weights=col)
            return np.sqrt(np.average((y - y_center) ** 2, weights=col))

        return rms_beam_size

    def extract_profile(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute column-wise sizes, weights and centers, with validity checks.

        Parameters
        ----------
        image : numpy.ndarray
            2D input image of shape ``(H, W)``.

        Returns
        -------
        sizes : numpy.ndarray, shape (W,)
            Estimated vertical size per column; NaN where invalid or below `min_total_counts`.
        weights : numpy.ndarray, shape (W,)
            Total intensity per column.
        valid_mask : numpy.ndarray of bool, shape (W,)
            Mask of columns that (a) met count threshold and (b) had spread fully within the image
            by ``n_beam_size_clearance``; further cleaned to be monotonic away from the weighted center.
        centers : numpy.ndarray, shape (W,)
            Intensity-weighted vertical center per column; NaN where invalid or
            below `min_total_counts`. Already computed for the clearance check
            — returned so callers can locate the beam vertically, not only size it.

        Notes
        -----
        - Validity requires that the interval ``[y_center ± n_beam_size_clearance * spread]`` lies within image bounds.
        - The final mask is post-processed by `enforce_monotonic_validity()` to avoid isolated valid islands.
        """
        h, w = image.shape
        y = np.arange(h)
        sizes, weights, valid_mask, centers = [], [], [], []

        beam_size_func = self.beam_size_func or self._default_beam_size_func(y)

        for x in range(w):
            col = image[:, x]
            total = col.sum()
            weights.append(total)

            if total < self.min_total_counts:
                sizes.append(np.nan)
                centers.append(np.nan)
                valid_mask.append(False)
                continue

            y_center = np.nan
            try:
                spread = beam_size_func(col)
                y_center = np.average(y, weights=col)
                ymin, ymax = (
                    y_center - self.n_beam_size_clearance * spread,
                    y_center + self.n_beam_size_clearance * spread,
                )
                is_valid = (ymin >= 0) and (ymax < h) and np.isfinite(spread)
            except Exception:
                spread = np.nan
                y_center = np.nan
                is_valid = False

            sizes.append(spread)
            centers.append(y_center)
            valid_mask.append(is_valid)

        sizes = np.array(sizes)
        weights = np.array(weights)
        centers = np.array(centers, dtype=float)
        valid_mask = self.enforce_monotonic_validity(np.array(valid_mask), weights)

        return sizes, weights, valid_mask, centers

    @staticmethod
    def center_at(x0: float, centers: np.ndarray, valid_mask: np.ndarray) -> float:
        """Interpolate the vertical beam center at the (fractional) waist column ``x0``.

        The waist column the fit returns is a float, so the vertical position
        there is read by linear interpolation of the per-column centers over
        the columns that entered the fit.

        Parameters
        ----------
        x0 : float
            Waist column from the bow-tie fit.
        centers : numpy.ndarray, shape (W,)
            Per-column vertical centers, NaN where the column was not measured.
        valid_mask : numpy.ndarray of bool, shape (W,)
            Columns that entered the fit.

        Returns
        -------
        float
            Row index of the beam at ``x0``, or NaN if ``x0`` is not finite or
            falls outside the span of valid columns. Deliberately does not
            extrapolate: outside that span the centroid track is unmeasured,
            and a projected value would look like a measurement.
        """
        if not np.isfinite(x0):
            return float("nan")

        usable = np.asarray(valid_mask, dtype=bool) & np.isfinite(centers)
        if not usable.any():
            return float("nan")

        x_known = np.flatnonzero(usable).astype(float)
        if x0 < x_known[0] or x0 > x_known[-1]:
            return float("nan")

        return float(np.interp(x0, x_known, np.asarray(centers)[usable]))

    def evaluate(self, image: np.ndarray) -> BowtieFitResult:
        """Extract sizes and fit the divergence model, returning fit parameters and diagnostics.

        Parameters
        ----------
        image : numpy.ndarray
            2D input image of shape ``(H, W)``. Any external preprocessing (thresholding, background
            removal, etc.) should be performed by the caller before this method if desired.

        Returns
        -------
        BowtieFitResult
            Result object with model parameters, R², uncertainties, and the raw profiles used.

        Fit details
        -----------
        - Points are selected where both ``valid_mask`` is True and ``weights > 0``.
        - Per-point uncertainties scale as ``1/sqrt(weight)`` for `curve_fit`.
        - Initial guess defaults to ``[min(sigma_fit), 0.1, median(x_fit)]``.
        - A guard rejects fits if there is insufficient weight within ±10 px of the waist (``x0``).
        """
        preprocessed = image
        sizes, weights, valid_mask, centers = self.extract_profile(preprocessed)
        self._last_profile = (sizes, weights, valid_mask, centers)

        x_vals = np.arange(image.shape[1])
        x_fit = x_vals[valid_mask & (weights > 0)]
        sigma_fit = sizes[valid_mask & (weights > 0)]
        weights_fit = weights[valid_mask & (weights > 0)]

        if len(x_fit) < 5:
            return BowtieFitResult(
                score=1e6,
                w0=np.nan,
                theta=np.nan,
                x0=np.nan,
                y0=np.nan,
                r_squared=-np.inf,
                param_errors=(np.nan, np.nan, np.nan),
                sizes=sizes,
                weights=weights,
                centers=centers,
            )
        try:
            weights_fit_safe = np.clip(weights_fit, 1e-6, None)
            sigma_err = 1.0 / np.sqrt(weights_fit_safe)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(
                    self._beam_model,
                    x_fit,
                    sigma_fit,
                    sigma=sigma_err,
                    absolute_sigma=True,
                    p0=[np.min(sigma_fit), 0.1, np.median(x_fit)],
                )

                w0, theta, x0 = popt
                residuals = sigma_fit - self._beam_model(x_fit, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((sigma_fit - np.mean(sigma_fit)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("-inf")

                perr = (
                    np.sqrt(np.diag(pcov))
                    if np.all(np.isfinite(pcov))
                    else (np.nan, np.nan, np.nan)
                )

                # Additional check: ensure enough weight around the waist.
                waist_window = 10  # pixels
                in_waist_region = (x_fit >= (x0 - waist_window)) & (
                    x_fit <= (x0 + waist_window)
                )
                total_weight_waist = np.sum(weights_fit[in_waist_region])
                total_weight_all = np.sum(weights_fit)

                if total_weight_waist < 0.004 * total_weight_all:
                    logger.warning(
                        f"⚠️ Insufficient weight near waist: x0={x0:.1f}, waist weight = {total_weight_waist:.1f}, total = {total_weight_all:.1f}"
                    )
                    return BowtieFitResult(
                        score=1e6,
                        w0=1e6,
                        theta=1e6,
                        x0=1e6,
                        y0=np.nan,
                        r_squared=-np.inf,
                        param_errors=(np.nan,) * 3,
                        sizes=sizes,
                        weights=weights,
                        centers=centers,
                    )

            return BowtieFitResult(
                score=w0 * abs(theta),
                w0=w0,
                theta=abs(theta),
                x0=x0,
                y0=self.center_at(x0, centers, valid_mask),
                r_squared=r_squared,
                param_errors=perr,
                sizes=sizes,
                weights=weights,
                centers=centers,
            )

        except Exception as e:
            logger.warning(f"[EXCEPTION] Fit failed: {e}")
            return BowtieFitResult(
                score=np.nan,
                w0=np.nan,
                theta=np.nan,
                x0=np.nan,
                y0=np.nan,
                r_squared=-np.inf,
                param_errors=(np.nan, np.nan, np.nan),
                sizes=sizes,
                weights=weights,
                centers=centers,
            )

    @staticmethod
    def _beam_model(x: np.ndarray, w0: float, theta: float, x0: float) -> np.ndarray:
        """Return model: ``sigma(x) = sqrt(w0**2 + ((x - x0) * theta)**2)``."""
        return np.sqrt(w0**2 + ((x - x0) * theta) ** 2)

    @staticmethod
    def _initial_guess(
        x_vals: np.ndarray, sizes: np.ndarray
    ) -> Tuple[float, float, float]:
        """Provide initial parameters ``(w0, theta, x0)`` for the model fit."""
        idx_min = np.nanargmin(sizes)
        x0 = x_vals[idx_min]
        w0 = sizes[idx_min]
        theta = 0.05
        return w0, theta, x0

    def enforce_monotonic_validity(
        self, valid_mask: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Force the valid-mask to be contiguous away from the weighted center.

        Parameters
        ----------
        valid_mask : numpy.ndarray of bool, shape (W,)
            Initial column validity mask.
        weights : numpy.ndarray, shape (W,)
            Column totals used to compute a weighted center.

        Returns
        -------
        numpy.ndarray
            Cleaned boolean mask where invalid regions propagate outward from the center.
        """
        x_vals = np.arange(len(valid_mask))
        center_x = np.average(x_vals, weights=weights + 1e-8)
        center_idx = int(np.round(center_x))
        mask = valid_mask.copy()

        for i in range(center_idx - 1, -1, -1):
            if not mask[i + 1]:
                mask[i] = False
        for i in range(center_idx + 1, len(mask)):
            if not mask[i - 1]:
                mask[i] = False

        return mask
