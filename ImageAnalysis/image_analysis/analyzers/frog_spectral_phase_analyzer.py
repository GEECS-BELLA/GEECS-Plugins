"""FROG spectral phase analyzer using the 1D analyzer framework.

This analyzer consumes retrieved Grenouille/FROG lineout TSV files written by
``GrenouilleAnalyzer``. It fits spectral phase as a polynomial in angular
frequency detuning and reports physical dispersion terms such as GD, GDD, and
TOD.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from image_analysis.algorithms.polynomial_fit import (
    PolynomialFitConfig,
    canonicalize_polynomial_sign,
    coefficients_by_order,
    fit_polynomial,
)
from image_analysis.analyzers.standard_1d_analyzer import Standard1DAnalyzer
from image_analysis.config.array1d_processing import Line1DConfig
from image_analysis.types import Array1D, ImageAnalyzerResult

logger = logging.getLogger(__name__)

_SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
_RAD_PER_SECOND_TO_RAD_PER_FS = 1e-15


class FrogSpectralPhaseConfig(BaseModel):
    """Typed configuration for :class:`FrogSpectralPhaseAnalyzer`.

    Attributes
    ----------
    fit_order : int
        Polynomial fit order.
    mask_threshold : float, optional
        Threshold applied to normalized weights. If no ``weights`` auxiliary
        column is loaded, the threshold is applied to ``abs(phase)`` instead.
    reference_wavelength_nm : float
        Reference wavelength used to build angular-frequency detuning.
    sign_reference_order : int, optional
        Polynomial order used to resolve the FROG ``phi`` versus ``-phi`` sign
        ambiguity. ``2`` corresponds to the GDD-like term.
    sign_reference : float
        Desired sign of ``sign_reference_order``.
    sign_epsilon : float
        Tolerance below which the sign reference is treated as ambiguous.
    """

    fit_order: int = Field(default=3, ge=0)
    mask_threshold: Optional[float] = Field(default=0.5, ge=0)
    min_points: Optional[int] = Field(default=None, ge=1)
    fit_num_points: int = Field(default=300, ge=2)
    reference_wavelength_nm: float = Field(default=800.0, gt=0)
    sign_reference_order: Optional[int] = Field(default=2, ge=0)
    sign_reference: float = 1.0
    sign_epsilon: float = Field(default=0.0, ge=0)


class FrogSpectralPhaseAnalyzer(Standard1DAnalyzer):
    """Analyze retrieved FROG spectral phase lineouts.

    The configured ``data_loading`` block selects the spectral phase line from
    a retrieved lineout TSV. A named ``weights`` auxiliary column, when
    configured, is used for thresholding and weighted polynomial fitting.
    """

    def __init__(
        self,
        line_config_name: Union[str, Line1DConfig],
        metric_suffix: Optional[str] = None,
        metric_prefix: Optional[str] = None,
    ):
        """Initialize the analyzer with a line configuration."""
        super().__init__(line_config_name)
        self.analysis_config = FrogSpectralPhaseConfig.model_validate(
            self.line_config.analysis or {}
        )
        self.metric_suffix = _normalize_metric_suffix(metric_suffix)
        self.metric_prefix = metric_prefix

        logger.info(
            "Initialized FrogSpectralPhaseAnalyzer for line config '%s'",
            self.line_config.name,
        )

    def analyze_image(
        self, image: Array1D, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """Fit spectral phase and report polynomial/dispersion scalars."""
        initial_result = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )
        if initial_result.line_data is None:
            raise ValueError("FrogSpectralPhaseAnalyzer requires 1D line_data")

        line_data = np.asarray(initial_result.line_data, dtype=float).copy()
        spectral_axis = line_data[:, 0]
        spectral_phase = line_data[:, 1]
        weights = initial_result.line_auxiliary_column_data.get("weights")

        reference_omega = float(
            wavelength_nm_to_omega_rad_per_fs(
                np.asarray([self.analysis_config.reference_wavelength_nm])
            )[0]
        )
        fit_axis = wavelength_nm_to_omega_rad_per_fs(spectral_axis) - reference_omega

        fit_result = fit_polynomial(
            y=spectral_phase,
            x=fit_axis,
            weights=weights,
            config=PolynomialFitConfig(
                order=self.analysis_config.fit_order,
                mask_threshold=self.analysis_config.mask_threshold,
                min_points=self.analysis_config.min_points,
                fit_num_points=self.analysis_config.fit_num_points,
            ),
        )

        coefficients = fit_result.coefficients
        fit_y = fit_result.fit_y
        flipped = False
        if self.analysis_config.sign_reference_order is not None:
            coefficients, flipped = canonicalize_polynomial_sign(
                coefficients=coefficients,
                reference_order=self.analysis_config.sign_reference_order,
                reference_sign=self.analysis_config.sign_reference,
                epsilon=self.analysis_config.sign_epsilon,
            )
            if flipped:
                line_data[:, 1] *= -1
                fit_y = -fit_y

        prefix = self.metric_prefix or self.line_config.name
        scalars = self._build_scalars(
            prefix=prefix,
            coefficients=coefficients,
            flipped=flipped,
        )

        if self.metric_suffix:
            scalars = {
                f"{key}{self.metric_suffix}": value for key, value in scalars.items()
            }

        metadata = dict(initial_result.metadata or {})
        metadata.update(
            {
                "fit_order": self.analysis_config.fit_order,
                "mask_threshold": self.analysis_config.mask_threshold,
                "reference_wavelength_nm": self.analysis_config.reference_wavelength_nm,
            }
        )

        render_data = {
            "fit_omega_detuning_rad_per_fs": fit_result.fit_x,
            "fit_phase": fit_y,
            "fit_valid_mask": fit_result.valid_mask,
            "fit_normalized_reference": fit_result.normalized_reference,
        }
        render_data["fit_wavelength_nm"] = omega_rad_per_fs_to_wavelength_nm(
            fit_result.fit_x + reference_omega
        )

        return ImageAnalyzerResult(
            data_type="1d",
            line_data=line_data,
            line_auxiliary_column_data=initial_result.line_auxiliary_column_data,
            scalars=scalars,
            metadata=metadata,
            render_data=render_data,
        )

    def _build_scalars(
        self,
        prefix: str,
        coefficients: np.ndarray,
        flipped: bool,
    ) -> dict[str, float]:
        """Build scalar output dictionary."""
        scalars: dict[str, float] = {}
        by_order = coefficients_by_order(coefficients)

        scalars.update(_physical_dispersion_scalars(prefix, by_order))

        scalars[f"{prefix}_flipped"] = float(flipped)
        return scalars


def wavelength_nm_to_omega_rad_per_fs(wavelength_nm: np.ndarray) -> np.ndarray:
    """Convert wavelength in nm to angular frequency in rad/fs."""
    wavelength_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    return (
        2.0
        * np.pi
        * _SPEED_OF_LIGHT_M_PER_S
        / wavelength_m
        * _RAD_PER_SECOND_TO_RAD_PER_FS
    )


def omega_rad_per_fs_to_wavelength_nm(omega_rad_per_fs: np.ndarray) -> np.ndarray:
    """Convert angular frequency in rad/fs to wavelength in nm."""
    omega_rad_per_s = (
        np.asarray(omega_rad_per_fs, dtype=float) / _RAD_PER_SECOND_TO_RAD_PER_FS
    )
    wavelength_m = 2.0 * np.pi * _SPEED_OF_LIGHT_M_PER_S / omega_rad_per_s
    return wavelength_m / 1e-9


def _physical_dispersion_scalars(
    prefix: str,
    coefficients: dict[int, float],
) -> dict[str, float]:
    """Convert phase polynomial coefficients into physical derivatives."""
    scalars: dict[str, float] = {}
    names = {
        0: "phi0_rad",
        1: "gd_fs",
        2: "gdd_fs2",
        3: "tod_fs3",
        4: "fod_fs4",
    }
    for order, value in coefficients.items():
        if order == 0:
            converted = value
        else:
            converted = math.factorial(order) * value

        name = names.get(order, f"order{order}_fs{order}")
        scalars[f"{prefix}_{name}"] = float(converted)
    return scalars


def _normalize_metric_suffix(metric_suffix: Optional[str]) -> Optional[str]:
    """Normalize metric suffix to match LineAnalyzer behavior."""
    if not metric_suffix:
        return None
    if metric_suffix.startswith("_"):
        return metric_suffix
    return f"_{metric_suffix}"
