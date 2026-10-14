"""Weighted, threshold-masked polynomial fitting of one trace.

Ported from ImageAnalysis' ``algorithms.polynomial_fit`` with its numerical
conventions unchanged — finite filtering, the normalised-reference mask,
the normalised weights handed to :func:`numpy.polyfit`, the evaluation
grid spanning the fitted samples — so the ``frog_spectral_phase`` measure
reproduces the legacy ``FrogSpectralPhaseAnalyzer`` bit for bit (pinned by
a differential test that uses ImageAnalysis as the oracle). The only change
is shape: the legacy ``fit_polynomial`` body is split into helpers, and its
pydantic config is a frozen dataclass of the same fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PolynomialFitConfig:
    """Fit order, the reference mask, the point floor and the evaluation grid size.

    ``mask_threshold`` excludes points whose normalised reference (the
    weights when given, else ``abs(y)``) is below it; ``min_points``
    unset means ``order + 1``.
    """

    order: int = 3
    mask_threshold: Optional[float] = None
    min_points: Optional[int] = None
    fit_num_points: int = 300


@dataclass(frozen=True)
class PolynomialFitResult:
    """Coefficients (highest power first), the evaluated fit, and which samples it used.

    ``normalized_reference`` is the threshold/weight reference normalised by
    its finite maximum absolute value, NaN where undefined; ``valid_mask``
    marks the input samples the fit used.
    """

    coefficients: np.ndarray
    fit_x: np.ndarray
    fit_y: np.ndarray
    valid_mask: np.ndarray
    normalized_reference: np.ndarray
    order: int
    num_fit_points: int


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
    """Flip all coefficients so the ``reference_order`` term has ``reference_sign``.

    A reference coefficient within ``epsilon`` of zero is sign-ambiguous and
    never flipped. Returns the (possibly negated) coefficients and whether
    they were flipped.
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
        Independent variable values; sample indices when omitted.
    weights : np.ndarray, optional
        Reference for the threshold mask and, normalised, the fit weights.
    config : PolynomialFitConfig, optional
        Fit configuration; the defaults when omitted.

    Returns
    -------
    PolynomialFitResult
        Coefficients, the fit on ``fit_num_points`` evenly spaced x values
        spanning the fitted samples, and the masks.

    Raises
    ------
    ValueError
        Incompatible shapes, too few valid points, or a reference with no
        positive finite value.
    """
    fit_config = config or PolynomialFitConfig()
    x_data, y_data, weight_data = _arrays(y, x, weights)

    finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
    if weight_data is not None:
        finite_mask &= np.isfinite(weight_data)

    normalized_reference, threshold_mask = _threshold(
        y_data, weight_data, finite_mask, fit_config.mask_threshold
    )
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

    coefficients = np.polyfit(x_valid, y_valid, deg=fit_config.order, w=fit_weights)
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


def _arrays(
    y: np.ndarray, x: Optional[np.ndarray], weights: Optional[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Flatten the inputs to float64 and check that their shapes agree."""
    y_data = np.asarray(y, dtype=float).reshape(-1)
    if x is None:
        x_data = np.arange(y_data.size, dtype=float)
    else:
        x_data = np.asarray(x, dtype=float).reshape(-1)
    if x_data.shape != y_data.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x_data.shape} and {y_data.shape}"
        )
    if weights is None:
        return x_data, y_data, None
    weight_data = np.asarray(weights, dtype=float).reshape(-1)
    if weight_data.shape != y_data.shape:
        raise ValueError(
            "weights must have the same shape as y, got "
            f"{weight_data.shape} and {y_data.shape}"
        )
    return x_data, y_data, weight_data


def _threshold(
    y_data: np.ndarray,
    weight_data: Optional[np.ndarray],
    finite_mask: np.ndarray,
    mask_threshold: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """The normalised reference (weights, else ``y`` when masking) and its mask."""
    normalized_reference = np.full(y_data.shape, np.nan, dtype=float)
    threshold_mask = np.ones(y_data.shape, dtype=bool)
    reference_data = None
    if weight_data is not None:
        reference_data = weight_data
    elif mask_threshold is not None:
        reference_data = y_data
    if reference_data is not None:
        normalized_reference = _normalize_abs(reference_data, finite_mask)
        if mask_threshold is not None:
            threshold_mask = normalized_reference >= mask_threshold
    return normalized_reference, threshold_mask


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
