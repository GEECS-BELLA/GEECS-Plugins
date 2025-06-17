# # File: algorithms/bowtie_fit.py

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import curve_fit
from lcls_tools.common.data.fit.methods import GaussianModel
from lcls_tools.common.data.fit.projection import ProjectionFit


@dataclass
class BowtieFitResult:
    """
    Encapsulates the result of a bowtie beam profile fit.

    Attributes:
        score (float): Quality metric equal to w0 * |theta|.
        w0 (float): Minimum beam size (waist).
        theta (float): Angular divergence in pixels^-1.
        x0 (float): Horizontal location of the beam waist.
        r_squared (float): R-squared value of the fit.
        param_errors (Tuple[float, float, float]): Uncertainties of the fit parameters (w0, theta, x0).
    """
    score: float
    w0: float
    theta: float
    x0: float
    r_squared: float
    param_errors: Tuple[float, float, float]


class BowtieFitAlgorithm:
    """
    Algorithm for evaluating synthetic or experimental bowtie-shaped beam profiles in 2D images.
    Provides preprocessing, beam size extraction, and model fitting utilities.
    """

    def __init__(
        self,
        background_method: str = 'median',
        threshold_factor: float = 0.0,
        n_beam_size_clearance: float = 4.0,
        min_total_counts: float = 0.0,
        beam_size_func: Optional[Callable[[np.ndarray], float]] = None
    ):
        self.background_method = background_method
        self.threshold_factor = threshold_factor
        self.n_beam_size_clearance = n_beam_size_clearance
        self.min_total_counts = min_total_counts
        self.beam_size_func = beam_size_func
        self._last_profile: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def get_last_profile(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns the last (sizes, weights, valid_mask) profile extracted in evaluate().

        Returns:
            Tuple of arrays or None if evaluate() has not been called.
        """
        return self._last_profile

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Subtracts background using the configured method and thresholds low values.

        Args:
            image (np.ndarray): Input 2D image.

        Returns:
            np.ndarray: Processed image.
        """
        img = image.astype(np.float32)
        if self.background_method == 'median':
            background = np.median(img) * 0
        elif self.background_method == 'mode':
            from scipy.stats import mode
            background = mode(img, axis=None, keepdims=False).mode
        else:
            background = 0

        img -= background
        img[img < self.threshold_factor] = 0
        return img

    @staticmethod
    def gaussian_fit_beam_size(col: np.ndarray) -> float:
        projection_fit = ProjectionFit(model=GaussianModel())
        result = projection_fit.fit_projection(col)
        return result['sigma']

    def _default_beam_size_func(self, y: np.ndarray) -> Callable[[np.ndarray], float]:
        """
        Returns RMS beam size estimator with fixed y-axis.

        Args:
            y (np.ndarray): Vertical axis.

        Returns:
            Callable: Beam size function.
        """
        def rms_beam_size(col: np.ndarray) -> float:
            if col.sum() <= 0:
                return np.nan
            y_center = np.average(y, weights=col)
            return np.sqrt(np.average((y - y_center) ** 2, weights=col))
        return rms_beam_size

    def extract_profile(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts vertical beam size (RMS) and total intensity for each column.
        Applies clipping rejection based on vertical spread.

        Args:
            image (np.ndarray): 2D input image.

        Returns:
            Tuple: (sizes, weights, valid_mask)
        """
        h, w = image.shape
        y = np.arange(h)
        sizes, weights, valid_mask = [], [], []

        beam_size_func = self.beam_size_func or self._default_beam_size_func(y)

        for x in range(w):
            col = image[:, x]
            total = col.sum()
            weights.append(total)

            if total < self.min_total_counts:
                sizes.append(np.nan)
                valid_mask.append(False)
                continue

            try:
                spread = beam_size_func(col)
                y_center = np.average(y, weights=col)
                ymin, ymax = y_center - self.n_beam_size_clearance * spread, y_center + self.n_beam_size_clearance * spread
                is_valid = (ymin >= 0) and (ymax < h) and np.isfinite(spread)
            except Exception:
                spread = np.nan
                is_valid = False

            sizes.append(spread)
            valid_mask.append(is_valid)

        sizes = np.array(sizes)
        weights = np.array(weights)
        valid_mask = self.enforce_monotonic_validity(np.array(valid_mask), weights)

        return sizes, weights, valid_mask

    def evaluate(self, image: np.ndarray) -> BowtieFitResult:
        """
        Evaluate the beam profile by extracting vertical sizes column-wise and
        fitting the Gaussian beam divergence model.

        Stores the extracted profile for optional reuse.

        Args:
            image (np.ndarray): Input 2D image.

        Returns:
            BowtieFitResult: Result object containing fitted parameters and diagnostics.
        """
        preprocessed = self.preprocess(image)
        sizes, weights, valid_mask = self.extract_profile(preprocessed)
        self._last_profile = (sizes, weights, valid_mask)

        x_vals = np.arange(image.shape[1])
        x_fit = x_vals[valid_mask & (weights > 0)]
        sigma_fit = sizes[valid_mask & (weights > 0)]
        weights_fit = weights[valid_mask & (weights > 0)]

        if len(x_fit) < 5:
            return BowtieFitResult(score=np.nan, w0=np.nan, theta=np.nan, x0=np.nan, r_squared=-np.inf, param_errors=(np.nan, np.nan, np.nan))

        try:
            weights_fit_safe = np.clip(weights_fit, 1e-6, None)
            sigma_err = 1.0 / np.sqrt(weights_fit_safe)

            popt, pcov = curve_fit(
                self._beam_model,
                x_fit,
                sigma_fit,
                sigma=sigma_err,
                absolute_sigma=True,
                p0=[np.min(sigma_fit), 0.1, np.median(x_fit)]
            )

            w0, theta, x0 = popt
            residuals = sigma_fit - self._beam_model(x_fit, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((sigma_fit - np.mean(sigma_fit)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float('-inf')

            perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else (np.nan, np.nan, np.nan)

            # Additional check: ensure enough weight around the waist
            waist_window = 10  # pixels
            in_waist_region = (x_fit >= (x0 - waist_window)) & (x_fit <= (x0 + waist_window))
            total_weight_waist = np.sum(weights_fit[in_waist_region])
            total_weight_all = np.sum(weights_fit)

            if total_weight_waist < 0.004 * total_weight_all:
                print(
                    f"⚠️ Insufficient weight near waist: x0={x0:.1f}, waist weight = {total_weight_waist:.1f}, total = {total_weight_all:.1f}")
                return BowtieFitResult(score=1e6, w0=1e6, theta=1e6, x0=1e6, r_squared=-np.inf,
                                       param_errors=(np.nan,) * 3)

            return BowtieFitResult(score=w0 * abs(theta), w0=w0, theta=abs(theta), x0=x0, r_squared=r_squared,
                                   param_errors=perr)


        except Exception as e:
            print(f"[EXCEPTION] Fit failed: {e}")
            return BowtieFitResult(score=np.nan, w0=np.nan, theta=np.nan, x0=np.nan, r_squared=-np.inf, param_errors=(np.nan, np.nan, np.nan))

    @staticmethod
    def _beam_model(x: np.ndarray, w0: float, theta: float, x0: float) -> np.ndarray:
        """Returns model: sigma(x) = sqrt(w0^2 + ((x - x0) * theta)^2)"""
        return np.sqrt(w0**2 + ((x - x0) * theta)**2)

    @staticmethod
    def _initial_guess(x_vals: np.ndarray, sizes: np.ndarray) -> Tuple[float, float, float]:
        """Provides initial parameters (w0, theta, x0) for model fit."""
        idx_min = np.nanargmin(sizes)
        x0 = x_vals[idx_min]
        w0 = sizes[idx_min]
        theta = 0.05
        return w0, theta, x0

    def enforce_monotonic_validity(self, valid_mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Ensures beam profile validity mask is monotonic from center.

        Args:
            valid_mask (np.ndarray): Boolean mask.
            weights (np.ndarray): Column totals.

        Returns:
            np.ndarray: Cleaned mask.
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




# import numpy as np
# from typing import Tuple, Optional, Callable
# from dataclasses import dataclass
# from scipy.optimize import curve_fit
# from lcls_tools.common.data.fit.methods import GaussianModel
# from lcls_tools.common.data.fit.projection import ProjectionFit
#
#
# @dataclass
# class BowtieFitResult:
#     """
#     Encapsulates the result of a bowtie beam profile fit.
#
#     Attributes:
#         score (float): Quality metric equal to w0 * |theta|.
#         w0 (float): Minimum beam size (waist).
#         theta (float): Angular divergence in pixels^-1.
#         x0 (float): Horizontal location of the beam waist.
#         r_squared (float): R-squared value of the fit.
#         param_errors (Tuple[float, float, float]): Uncertainties of the fit parameters (w0, theta, x0).
#     """
#     score: float
#     w0: float
#     theta: float
#     x0: float
#     r_squared: float
#     param_errors: Tuple[float, float, float]
#
#
# class BowtieFitAlgorithm:
#     """
#     Algorithm for evaluating synthetic or experimental bowtie-shaped beam profiles in 2D images.
#     Provides preprocessing, beam size extraction, and model fitting utilities.
#     """
#     def __init__(self):
#         self._last_profile: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
#
#     def get_last_profile(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
#         """
#         Returns the last (sizes, weights, valid_mask) profile extracted in evaluate().
#
#         Returns:
#             Tuple of arrays or None if evaluate() has not been called.
#         """
#         return self._last_profile
#
#     def preprocess(self, image: np.ndarray, method: str = 'median', threshold_factor: float = 0.0) -> np.ndarray:
#         """
#         Subtracts background using the given method and thresholds low values.
#
#         Args:
#             image (np.ndarray): Input 2D image.
#             method (str): 'median' or 'mode'.
#             threshold_factor (float): Zero pixels below threshold_factor * std.
#
#         Returns:
#             np.ndarray: Processed image.
#         """
#         img = image.astype(np.float32)
#         if method == 'median':
#             background = np.median(img)*0
#         elif method == 'mode':
#             from scipy.stats import mode
#             background = mode(img, axis=None, keepdims=False).mode
#         else:
#             raise ValueError("method must be 'median' or 'mode'")
#
#         img -= background*0
#         img[img < threshold_factor * np.std(img)] = 0
#         return img
#
#     @staticmethod
#     def gaussian_fit_beam_size(col: np.ndarray) -> float:
#         projection_fit = ProjectionFit(model=GaussianModel())
#         result = projection_fit.fit_projection(col)
#         return result['sigma']
#
#     def extract_profile(
#             self,
#             image: np.ndarray,
#             n_beam_size_clearance: float = 5,
#             min_total_counts: float = 400.0,
#             beam_size_func: Optional[Callable[[np.ndarray], float]] = None
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Extracts vertical beam size (RMS) and total intensity for each column.
#         Applies clipping rejection based on vertical spread.
#
#         Args:
#             image (np.ndarray): 2D input image.
#             n_beam_size_clearance (float): Sigma margin for detecting cropping.
#             min_total_counts (float): Minimum total intensity per column.
#             beam_size_func (Callable): Function that takes column intensity and returns spread.
#
#         Returns:
#             Tuple: (sizes, weights, valid_mask)
#         """
#         h, w = image.shape
#         y = np.arange(h)
#         sizes, weights, valid_mask = [], [], []
#
#         # Default beam size function (RMS of vertical slice, weighted)
#         if beam_size_func is None:
#             def beam_size_func(col: np.ndarray) -> float:
#                 y_center = np.average(y, weights=col)
#                 return np.sqrt(np.average((y - y_center) ** 2, weights=col))
#
#         for x in range(w):
#             col = image[:, x]
#             total = col.sum()
#             weights.append(total)
#
#             if total < min_total_counts:
#                 sizes.append(np.nan)
#                 valid_mask.append(False)
#                 continue
#
#             try:
#                 spread = beam_size_func(col)
#                 y_center = np.average(y, weights=col)
#                 ymin, ymax = y_center - n_beam_size_clearance * spread, y_center + n_beam_size_clearance * spread
#                 is_valid = (ymin >= 0) and (ymax < h) and np.isfinite(spread)
#             except Exception:
#                 spread = np.nan
#                 is_valid = False
#
#             sizes.append(spread)
#             valid_mask.append(is_valid)
#
#         sizes = np.array(sizes)
#         weights = np.array(weights)
#         valid_mask = self.enforce_monotonic_validity(np.array(valid_mask), weights)
#
#         return sizes, weights, valid_mask
#     def _default_beam_size_func(self, y: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
#         """
#         Returns RMS beam size estimator with fixed y-axis.
#
#         Args:
#             y (np.ndarray): Vertical axis.
#
#         Returns:
#             Callable: Beam size function.
#         """
#         def rms_beam_size(y_vals: np.ndarray, weights: np.ndarray) -> float:
#             if weights.sum() <= 0:
#                 return np.nan
#             y_center = np.average(y_vals, weights=weights)
#             return np.sqrt(np.average((y_vals - y_center) ** 2, weights=weights))
#         return rms_beam_size
#
#     def evaluate(
#             self,
#             image: np.ndarray,
#             n_beam_size_clearance: float = 4.0,
#             min_total_counts: float = 0.0,
#             beam_size_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
#     ) -> BowtieFitResult:
#         """
#         Evaluate the beam profile by extracting vertical sizes column-wise and
#         fitting the Gaussian beam divergence model.
#
#         Stores the extracted profile for optional reuse.
#
#         Args:
#             image (np.ndarray): Input 2D image.
#             n_beam_size_clearance (float): Rejection margin for clipping detection (in σ).
#             min_total_counts (float): Minimum intensity to consider a column valid.
#             beam_size_func (Callable, optional): Function to compute beam size for a column.
#
#         Returns:
#             BowtieFitResult: Result object containing fitted parameters and diagnostics.
#         """
#         # Step 1: Preprocess image
#         preprocessed = self.preprocess(image)
#
#         # Step 2: Extract beam profile
#         sizes, weights, valid_mask = self.extract_profile(
#             preprocessed,
#             n_beam_size_clearance=n_beam_size_clearance,
#             min_total_counts=min_total_counts,
#             beam_size_func=beam_size_func
#         )
#
#         self._last_profile = (sizes, weights, valid_mask)
#
#         # Step 3: Prepare data for fitting
#         x_vals = np.arange(image.shape[1])
#         x_fit = x_vals[valid_mask & (weights > 0)]
#         sigma_fit = sizes[valid_mask & (weights > 0)]
#         weights_fit = weights[valid_mask & (weights > 0)]
#
#         if len(x_fit) < 5:
#             return BowtieFitResult.failed()
#
#         try:
#             # Avoid zero weights
#             weights_fit_safe = np.clip(weights_fit, 1e-6, None)
#             sigma_err = 1.0 / np.sqrt(weights_fit_safe)
#
#             popt, pcov = curve_fit(
#                 self._beam_model,
#                 x_fit,
#                 sigma_fit,
#                 sigma=sigma_err,
#                 absolute_sigma=True,
#                 p0=[np.min(sigma_fit), 0.1, np.median(x_fit)]
#             )
#
#             w0, theta, x0 = popt
#
#             residuals = sigma_fit - self._beam_model(x_fit, *popt)
#             ss_res = np.sum(residuals ** 2)
#             ss_tot = np.sum((sigma_fit - np.mean(sigma_fit)) ** 2)
#             r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float('-inf')
#
#             perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else (np.nan, np.nan, np.nan)
#
#             return BowtieFitResult(
#                 score=ss_res,
#                 w0=w0,
#                 theta=theta,
#                 x0=x0,
#                 r_squared=r_squared,
#                 param_errors=perr
#             )
#
#         except Exception as e:
#             print(f"[EXCEPTION] Fit failed: {e}")
#             return BowtieFitResult.failed()
#
#     @staticmethod
#     def _beam_model(x: np.ndarray, w0: float, theta: float, x0: float) -> np.ndarray:
#         """Returns model: sigma(x) = sqrt(w0^2 + ((x - x0) * theta)^2)"""
#         return np.sqrt(w0**2 + ((x - x0) * theta)**2)
#
#     @staticmethod
#     def _initial_guess(x_vals: np.ndarray, sizes: np.ndarray) -> Tuple[float, float, float]:
#         """Provides initial parameters (w0, theta, x0) for model fit."""
#         idx_min = np.nanargmin(sizes)
#         x0 = x_vals[idx_min]
#         w0 = sizes[idx_min]
#         theta = 0.05
#         return w0, theta, x0
#
#     def enforce_monotonic_validity(self, valid_mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
#         """
#         Ensures beam profile validity mask is monotonic from center.
#
#         Args:
#             valid_mask (np.ndarray): Boolean mask.
#             weights (np.ndarray): Column totals.
#
#         Returns:
#             np.ndarray: Cleaned mask.
#         """
#         x_vals = np.arange(len(valid_mask))
#         center_x = np.average(x_vals, weights=weights + 1e-8)
#         center_idx = int(np.round(center_x))
#         mask = valid_mask.copy()
#
#         for i in range(center_idx - 1, -1, -1):
#             if not mask[i + 1]:
#                 mask[i] = False
#         for i in range(center_idx + 1, len(mask)):
#             if not mask[i - 1]:
#                 mask[i] = False
#
#         return mask
