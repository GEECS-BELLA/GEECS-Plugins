"""FROG spectral phase: a polynomial in angular-frequency detuning, as GD/GDD/TOD.

The legacy ``FrogSpectralPhaseAnalyzer``, ported with its conventions: the
retrieved spectral phase (a ``frog`` measure's ``*_retrieved_lineouts.tsv``
read as a trace over wavelength) is fitted about ``reference_wavelength_nm``,
the coefficients' sign is canonicalised on ``sign_reference_order``, the
constant ``phi0`` is subtracted from the stored curves (it is still a
scalar), and the processed frame is the fit resampled on
``fit_num_points`` wavelengths, sorted — one fixed length per shot, so the
traces of a scan average and stack.

The spectral intensity weights the fit and its threshold mask. It arrives
as the line input's ``weights`` channel (the loading's
``auxiliary_columns``), cut by the trace's own sample selection; without
one the fit is unweighted, as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Mapping, Optional

from pydantic import Field

from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    import numpy as np
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement

#: The auxiliary channel the fit is weighted by (the spectral intensity).
WEIGHTS = "weights"
#: Bare scalar names by polynomial order; higher orders are ``order{n}_fs{n}``.
ORDER_NAMES = {0: "phi0_rad", 1: "gd_fs", 2: "gdd_fs2", 3: "tod_fs3", 4: "fod_fs4"}

_SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
_RAD_PER_SECOND_TO_RAD_PER_FS = 1e-15


class FrogSpectralPhaseSpec(MeasureSpec):
    """Fit parameters; fields, defaults and bounds are the v2 ``frog_spectral_phase`` analyzer's."""

    kind: Literal["frog_spectral_phase"] = "frog_spectral_phase"
    scalar_docs: ClassVar[Mapping[str, str]] = {
        "phi0_rad": "Constant spectral phase at the reference wavelength, rad (shot-to-shot drift only)",
        "gd_fs": "Group delay: first derivative of the phase in angular frequency, fs",
        "gdd_fs2": "Group-delay dispersion: second derivative of the phase, fs²",
        "tod_fs3": "Third-order dispersion: third derivative of the phase, fs³",
        "flipped": "1 when the fit's sign was flipped to put sign_reference_order's term on sign_reference, else 0",
    }
    fit_order: int = Field(3, ge=0, description="Polynomial order of the phase fit.")
    mask_threshold: Optional[float] = Field(
        0.5,
        ge=0,
        description="Fit only where the spectral intensity exceeds this fraction of its peak.",
    )
    min_points: Optional[int] = Field(
        None, ge=1, description="Minimum number of points required for a fit."
    )
    fit_num_points: int = Field(
        300, ge=2, description="Number of points the fitted phase is evaluated on."
    )
    reference_wavelength_nm: float = Field(
        800.0, gt=0, description="Wavelength the phase expansion is taken about, nm."
    )
    sign_reference_order: Optional[int] = Field(
        2,
        ge=0,
        description=(
            "Polynomial order whose sign is forced to sign_reference (2 = the "
            "GDD term, the default); null leaves the fit's sign as is."
        ),
    )
    sign_reference: float = Field(
        1.0, description="Sign (+1 / -1) imposed on sign_reference_order."
    )
    sign_epsilon: float = Field(
        0.0,
        ge=0,
        description="Dead band around zero within which the sign is not flipped.",
    )

    def emitted_scalars(self) -> frozenset[str]:
        """The dispersion coefficients through ``fit_order`` and the flip flag."""
        return frozenset(
            {"flipped"}
            | {ORDER_NAMES.get(i, f"order{i}_fs{i}") for i in range(self.fit_order + 1)}
        )


@measure(FrogSpectralPhaseSpec, ndim={1}, channels=(WEIGHTS,))
def frog_spectral_phase(
    frame: Frame, spec: FrogSpectralPhaseSpec, channels: Mapping[str, Frame]
) -> Measurement:
    """Fit the trace's spectral phase over its wavelength axis (nm).

    The processed frame is the fitted phase (``phi0`` removed, the sign
    canonicalised) on ``fit_num_points`` wavelengths spanning the fitted
    samples, ascending. Extras: the measured phase on the trace's own
    wavelengths, treated the same way (``spectral_phase``), and the
    normalised fit reference (``fit_reference``, NaN where undefined). A
    trace the fit cannot use (too few points above the mask) raises, as the
    legacy analyzer did: the shot fails explicitly.
    """
    import numpy as np
    from geecs_data_utils.frames import Axis, Frame

    from geecs_analysis.algorithms.polynomial_fit import coefficients_by_order
    from geecs_analysis.measurement import Measurement

    if frame.data.ndim != 1 or frame.data.size == 0:
        raise ValueError("frog_spectral_phase requires a nonempty trace")
    wavelength = frame.axes[0].values
    weights = channels.get(WEIGHTS)
    fitted = _fit(
        wavelength,
        np.array(frame.data, dtype=float),
        None if weights is None else weights.data,
        spec,
    )
    by_order = coefficients_by_order(fitted.coefficients)
    axis = frame.axes[0]

    def trace(samples: np.ndarray, at: np.ndarray, label: str) -> Frame:
        return Frame.from_array(
            samples,
            axes=(Axis(at, unit=axis.unit, label=axis.label),),
            shot=frame.shot,
            unit=frame.unit,
            label=label,
        )

    return Measurement(
        scalars={**_dispersion_scalars(by_order), "flipped": float(fitted.flipped)},
        frame=trace(fitted.curve, fitted.curve_nm, frame.label),
        extras={
            "spectral_phase": trace(fitted.phase, wavelength, "spectral_phase"),
            "fit_reference": trace(fitted.reference, wavelength, "fit_reference"),
        },
    )


@dataclass(frozen=True)
class _Fitted:
    """One shot's fit, already sign-canonicalised and with ``phi0`` removed."""

    coefficients: np.ndarray
    flipped: bool
    curve_nm: np.ndarray
    curve: np.ndarray
    phase: np.ndarray
    reference: np.ndarray


def _fit(
    wavelength_nm: np.ndarray,
    phase: np.ndarray,
    weights: Optional[np.ndarray],
    spec: FrogSpectralPhaseSpec,
) -> _Fitted:
    """The legacy analyzer's fit, sign convention, phi0 removal and resample."""
    import numpy as np

    from geecs_analysis.algorithms.polynomial_fit import (
        PolynomialFitConfig,
        canonicalize_polynomial_sign,
        coefficients_by_order,
        fit_polynomial,
    )

    reference_omega = float(
        wavelength_nm_to_omega_rad_per_fs(np.asarray([spec.reference_wavelength_nm]))[0]
    )
    result = fit_polynomial(
        y=phase,
        x=wavelength_nm_to_omega_rad_per_fs(wavelength_nm) - reference_omega,
        weights=weights,
        config=PolynomialFitConfig(
            order=spec.fit_order,
            mask_threshold=spec.mask_threshold,
            min_points=spec.min_points,
            fit_num_points=spec.fit_num_points,
        ),
    )
    coefficients, fit_y, flipped = result.coefficients, result.fit_y, False
    if spec.sign_reference_order is not None:
        coefficients, flipped = canonicalize_polynomial_sign(
            coefficients=coefficients,
            reference_order=spec.sign_reference_order,
            reference_sign=spec.sign_reference,
            epsilon=spec.sign_epsilon,
        )
        if flipped:
            phase = -phase
            fit_y = -fit_y
    # phi0 is an integration constant that wanders shot to shot: removed from
    # the stored curves (waterfalls stay readable), kept as a scalar.
    phi0 = coefficients_by_order(coefficients)[0]
    curve_nm = omega_rad_per_fs_to_wavelength_nm(result.fit_x + reference_omega)
    order = np.argsort(curve_nm)
    return _Fitted(
        coefficients=coefficients,
        flipped=flipped,
        curve_nm=curve_nm[order],
        curve=(fit_y - phi0)[order],
        phase=phase - phi0,
        reference=result.normalized_reference,
    )


def wavelength_nm_to_omega_rad_per_fs(wavelength_nm: np.ndarray) -> np.ndarray:
    """Convert wavelength in nm to angular frequency in rad/fs."""
    import numpy as np

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
    import numpy as np

    omega_rad_per_s = (
        np.asarray(omega_rad_per_fs, dtype=float) / _RAD_PER_SECOND_TO_RAD_PER_FS
    )
    wavelength_m = 2.0 * np.pi * _SPEED_OF_LIGHT_M_PER_S / omega_rad_per_s
    return wavelength_m / 1e-9


def _dispersion_scalars(coefficients: dict[int, float]) -> dict[str, float]:
    """Polynomial coefficients as physical derivatives: ``n! * c_n`` for n ≥ 1."""
    import math

    return {
        ORDER_NAMES.get(order, f"order{order}_fs{order}"): float(
            value if order == 0 else math.factorial(order) * value
        )
        for order, value in coefficients.items()
    }
