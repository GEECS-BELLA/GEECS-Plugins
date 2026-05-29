"""FROG spectral phase analyzer using the 1D analyzer framework.

This analyzer consumes retrieved Grenouille/FROG lineout TSV files written by
``GrenouilleAnalyzer``. It fits spectral phase as a polynomial in angular
frequency detuning and reports physical dispersion terms such as GD, GDD, and
TOD.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

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
        Polynomial fit order. ``3`` (default) covers GD / GDD / TOD.
    mask_threshold : float, optional
        Threshold applied to normalized weights. Samples below this
        threshold are excluded from the fit. If no ``weights`` auxiliary
        column is loaded, the threshold is applied to ``abs(phase)``
        instead. Set to ``None`` to disable masking.
    min_points : int, optional
        Minimum number of valid samples (after masking) required to
        attempt the fit. The fit raises if fewer survive. ``None``
        (default) uses the polynomial-fit machinery's own lower bound
        (``fit_order + 1``).
    fit_num_points : int
        Number of points on the output fit grid that
        :meth:`analyze_image` returns as ``result.line_data``. Default
        ``300``. This is also the per-shot ``line_data`` row count seen
        by ``Array1DScanAnalyzer`` for waterfall + averaging.
    reference_wavelength_nm : float
        Reference wavelength (nm) used to build angular-frequency
        detuning. Conventionally the central wavelength of the pulse
        (800 nm by default for Ti:Sa).
    sign_reference_order : int, optional
        Polynomial order used to resolve the FROG ``phi`` vs ``-phi``
        sign ambiguity by comparing the sign of one coefficient against
        ``sign_reference``. ``2`` corresponds to the GDD-like term.
        **Set to ``None`` to skip sign canonicalization entirely** —
        useful when you know the physical sign already or when the
        canonical reference is unreliable (e.g. near-zero GDD where the
        sign of the reference coefficient is noise-dominated).
    sign_reference : float
        Desired sign of the ``sign_reference_order`` coefficient.
        ``+1.0`` (default) means "flip the polynomial if needed so the
        reference coefficient comes out positive." Ignored when
        ``sign_reference_order`` is ``None``.
    sign_epsilon : float
        Tolerance below which the sign reference is treated as
        ambiguous (no flip applied). Ignored when
        ``sign_reference_order`` is ``None``.
    """

    fit_order: int = Field(default=3, ge=0)
    mask_threshold: Optional[float] = Field(default=0.5, ge=0)
    min_points: Optional[int] = Field(default=None, ge=1)
    fit_num_points: int = Field(default=300, ge=2)
    reference_wavelength_nm: float = Field(default=800.0, gt=0)
    sign_reference_order: Optional[int] = Field(
        default=2,
        ge=0,
        description=(
            "Polynomial order used to canonicalize the FROG phi vs -phi "
            "sign. Set to null to skip sign canonicalization."
        ),
    )
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
        line_config: Line1DConfig,
        metric_suffix: Optional[str] = None,
        metric_prefix: Optional[str] = None,
    ):
        """Initialize the analyzer with a typed line configuration.

        Matches the post-PR-E ``Standard1DAnalyzer`` contract — the
        constructor takes a validated ``Line1DConfig`` only. String-by-name
        resolution moved to the loader layer; call
        ``image_analysis.config.load_line_config(name)`` to obtain the
        ``Line1DConfig`` first, then pass it here.

        Parameters
        ----------
        line_config : Line1DConfig
            Typed configuration. The ``analysis`` block is validated
            into a :class:`FrogSpectralPhaseConfig` and exposed as
            ``self.analysis_config``.
        metric_suffix, metric_prefix : str, optional
            Override the scalar-key prefix / suffix. Default behavior:
            prefix from ``line_config.name``; no suffix.
        """
        super().__init__(line_config=line_config)
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
        """Fit spectral phase and report polynomial/dispersion scalars.

        Bypasses :meth:`Standard1DAnalyzer.analyze_image` and calls
        :meth:`_preprocess_line_data` directly so the ROI-filtered
        ``weights`` aux column is in hand at the same call boundary as
        the ROI-filtered line data. Reads the raw weights from
        ``auxiliary_data["_aux_columns"]`` (populated by
        ``Standard1DAnalyzer.analyze_image_file`` when the configured
        ``data_loading.auxiliary_columns`` mapping declares a
        ``weights`` column).

        ``result.line_data`` carries the **fitted** phase curve on a
        wavelength grid of length ``fit_num_points`` (default 300) — a
        fixed-length, wavelength-sorted Nx2 array. Every shot in a scan
        therefore emits ``line_data`` of identical shape, which is what
        ``Array1DScanAnalyzer``'s waterfall and per-bin averaging
        require to aggregate across shots. The raw masked phase samples
        and the intensity weight curve at those samples live in
        ``result.render_data`` for single-shot diagnostic plots
        (``raw_wavelength_nm``, ``raw_spectral_phase``,
        ``fit_normalized_reference``).
        """
        raw_aux = (auxiliary_data or {}).get("_aux_columns", {})
        processed, filtered_aux = self._preprocess_line_data(
            image, auxiliary_column_data=raw_aux
        )

        processed = np.asarray(processed, dtype=float).copy()
        if processed.size == 0 or processed.shape[1] < 2:
            raise ValueError("FrogSpectralPhaseAnalyzer requires 1D line_data")

        spectral_axis = processed[:, 0]
        spectral_phase = processed[:, 1]
        weights = filtered_aux.get("weights")

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
                spectral_phase = -spectral_phase
                fit_y = -fit_y

        # Subtract the constant phi0 term from both the fit and the raw
        # scatter before storing them. phi0 is an unphysical integration
        # constant — spectral phase is only defined up to a constant —
        # and it varies wildly shot-to-shot, which makes waterfall y-
        # scales look chaotic. The scalar ``phi0_rad`` is still emitted
        # via ``_build_scalars`` for downstream drift diagnostics.
        phi0 = coefficients_by_order(coefficients)[0]
        fit_y = fit_y - phi0
        spectral_phase = spectral_phase - phi0

        # Build the fixed-length line_data from the fit, sorted by
        # wavelength so downstream renderers can treat it as a monotonic
        # spectrum without reordering.
        fit_wavelength_nm = omega_rad_per_fs_to_wavelength_nm(
            fit_result.fit_x + reference_omega
        )
        order = np.argsort(fit_wavelength_nm)
        line_data = np.column_stack([fit_wavelength_nm[order], fit_y[order]])

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

        metadata = self._build_input_parameters(auxiliary_data)
        metadata.update(
            {
                "fit_order": self.analysis_config.fit_order,
                "mask_threshold": self.analysis_config.mask_threshold,
                "reference_wavelength_nm": self.analysis_config.reference_wavelength_nm,
            }
        )

        # Raw samples + diagnostic curves go in render_data. Single-shot
        # plots can overlay the variable-length scatter
        # (``raw_wavelength_nm`` vs ``raw_spectral_phase``) on the
        # fixed-length fit in ``line_data``.
        render_data = {
            "raw_wavelength_nm": spectral_axis,
            "raw_spectral_phase": spectral_phase,
            "fit_normalized_reference": fit_result.normalized_reference,
            "fit_valid_mask": fit_result.valid_mask,
            "fit_omega_detuning_rad_per_fs": fit_result.fit_x,
        }

        return ImageAnalyzerResult(
            data_type="1d",
            line_data=line_data,
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
