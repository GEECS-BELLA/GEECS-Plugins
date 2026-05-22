"""Polynomial fitting helpers for one-dimensional analysis.

This module keeps reusable fitting mechanics independent from any diagnostic:
finite-value filtering, optional threshold masking, optional weighting, and
coefficient bookkeeping around :func:`numpy.polyfit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class PolynomialFitConfig(BaseModel):
    """Configuration for weighted polynomial fitting.

    Attributes
    ----------
    order : int
        Polynomial order passed to :func:`numpy.polyfit`.
    mask_threshold : float, optional
        If provided, points with normalized reference below this value are
        excluded from the fit. The reference is ``weights`` when provided,
        otherwise ``abs(y)``.
    min_points : int, optional
        Minimum number of points required after filtering. If omitted, the
        minimum is ``order + 1``.
    fit_num_points : int
        Number of points in the smooth fitted curve returned in the result.
    """

    order: int = Field(default=3, ge=0)
    mask_threshold: Optional[float] = Field(default=None, ge=0)
    min_points: Optional[int] = Field(default=None, ge=1)
    fit_num_points: int = Field(default=300, ge=2)


@dataclass(frozen=True)
class PolynomialFitResult:
    """Result from :func:`fit_polynomial`.

    Attributes
    ----------
    coefficients : np.ndarray
        Polynomial coefficients in NumPy order, highest power first.
    fit_x : np.ndarray
        Smooth x-axis for plotting the fitted polynomial.
    fit_y : np.ndarray
        Fitted polynomial evaluated on ``fit_x``.
    valid_mask : np.ndarray
        Boolean mask over the original input arrays showing points used for
        the fit.
    normalized_reference : np.ndarray
        Threshold/weight reference normalized by max absolute value. Entries
        are NaN where no reference is defined.
    order : int
        Polynomial order used for fitting.
    num_fit_points : int
        Number of input samples used after finite filtering and masking.
    """

    coefficients: np.ndarray
    fit_x: np.ndarray
    fit_y: np.ndarray
    valid_mask: np.ndarray
    normalized_reference: np.ndarray
    order: int
    num_fit_points: int

    @property
    def coefficients_by_order(self) -> dict[int, float]:
        """Return coefficients keyed by polynomial order.

        Unlike ``numpy.polyfit`` order, ``0`` is the constant term, ``1`` the
        linear term, and so on.
        """
        return coefficients_by_order(self.coefficients)


def coefficients_by_order(coefficients: np.ndarray) -> dict[int, float]:
    """Return polynomial coefficients keyed by ascending polynomial order."""
    max_order = len(coefficients) - 1
    return {
        max_order - idx: float(coefficient)
        for idx, coefficient in enumerate(coefficients)
    }


def canonicalize_polynomial_sign(
    coefficients: np.ndarray,
    reference_order: int,
    reference_sign: float = 1.0,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """Flip all coefficients to enforce a sign convention.

    Parameters
    ----------
    coefficients : np.ndarray
        Polynomial coefficients in NumPy order, highest power first.
    reference_order : int
        Polynomial order whose sign controls whether the fit is flipped.
    reference_sign : float, default=1.0
        Desired sign of the selected coefficient. Positive values enforce a
        positive coefficient; negative values enforce a negative coefficient.
    epsilon : float, default=0.0
        Coefficients with absolute value less than or equal to this tolerance
        are treated as sign-ambiguous and are not flipped.

    Returns
    -------
    tuple[np.ndarray, bool]
        Possibly sign-flipped coefficients and a flag indicating whether a
        flip was applied.
    """
    by_order = coefficients_by_order(np.asarray(coefficients, dtype=float))
    if reference_order not in by_order:
        raise ValueError(
            f"reference_order={reference_order} is not available for "
            f"polynomial order {len(coefficients) - 1}"
        )
    if reference_sign == 0:
        raise ValueError("reference_sign must be nonzero")

    reference_value = by_order[reference_order]
    if abs(reference_value) <= epsilon:
        return np.asarray(coefficients, dtype=float).copy(), False

    desired_sign = np.sign(reference_sign)
    if np.sign(reference_value) == desired_sign:
        return np.asarray(coefficients, dtype=float).copy(), False

    return -np.asarray(coefficients, dtype=float), True


def fit_polynomial(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    config: Optional[PolynomialFitConfig] = None,
) -> PolynomialFitResult:
    """Fit a polynomial to one-dimensional data.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable values.
    x : np.ndarray, optional
        Independent variable values. If omitted, sample indices are used.
    weights : np.ndarray, optional
        Optional weights used for thresholding and/or weighted fitting.
    config : PolynomialFitConfig, optional
        Fit configuration. Defaults to :class:`PolynomialFitConfig`.

    Returns
    -------
    PolynomialFitResult
        Coefficients, smooth fitted curve, and masks describing which samples
        were used.

    Raises
    ------
    ValueError
        If shapes are incompatible, too few valid points remain, or a
        requested threshold/weight source is unavailable.
    """
    fit_config = config or PolynomialFitConfig()

    y_data = np.asarray(y, dtype=float).reshape(-1)
    if x is None:
        x_data = np.arange(y_data.size, dtype=float)
    else:
        x_data = np.asarray(x, dtype=float).reshape(-1)

    if x_data.shape != y_data.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x_data.shape} and {y_data.shape}"
        )

    weight_data: Optional[np.ndarray]
    if weights is None:
        weight_data = None
    else:
        weight_data = np.asarray(weights, dtype=float).reshape(-1)
        if weight_data.shape != y_data.shape:
            raise ValueError(
                "weights must have the same shape as y, got "
                f"{weight_data.shape} and {y_data.shape}"
            )

    finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
    if weight_data is not None:
        finite_mask &= np.isfinite(weight_data)

    normalized_reference = np.full(y_data.shape, np.nan, dtype=float)
    threshold_mask = np.ones(y_data.shape, dtype=bool)

    reference_data = None
    if weight_data is not None:
        reference_data = weight_data
    elif fit_config.mask_threshold is not None:
        reference_data = y_data
    if reference_data is not None:
        normalized_reference = _normalize_abs(reference_data, finite_mask)
        if fit_config.mask_threshold is not None:
            threshold_mask = normalized_reference >= fit_config.mask_threshold

    valid_mask = finite_mask & threshold_mask
    num_points = int(np.count_nonzero(valid_mask))
    min_points = fit_config.min_points or fit_config.order + 1
    required_points = max(min_points, fit_config.order + 1)
    if num_points < required_points:
        raise ValueError(
            f"Polynomial order {fit_config.order} requires at least "
            f"{required_points} valid points; got {num_points}"
        )

    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]

    fit_weights = None
    if weight_data is not None:
        fit_weights = _normalize_abs(weight_data, finite_mask)[valid_mask]

    coefficients = np.polyfit(
        x_valid,
        y_valid,
        deg=fit_config.order,
        w=fit_weights,
    )
    fit_x = np.linspace(
        float(np.min(x_valid)), float(np.max(x_valid)), fit_config.fit_num_points
    )
    fit_y = np.polyval(coefficients, fit_x)

    return PolynomialFitResult(
        coefficients=coefficients,
        fit_x=fit_x,
        fit_y=fit_y,
        valid_mask=valid_mask,
        normalized_reference=normalized_reference,
        order=fit_config.order,
        num_fit_points=num_points,
    )


def _normalize_abs(values: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
    """Normalize finite values by their max absolute value."""
    normalized = np.full(values.shape, np.nan, dtype=float)
    finite_values = np.abs(values[finite_mask])
    if finite_values.size == 0:
        raise ValueError("No finite values available for normalization")

    scale = float(np.max(finite_values))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization reference has no positive finite values")

    normalized[finite_mask] = np.abs(values[finite_mask]) / scale
    return normalized
