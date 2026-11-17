"""Relative delay and differential GDD of two pulses from one spectral interferogram.

Two collinear pulses ``E1(ω) = A1 e^{iφ1}`` and ``E2(ω) = A2 e^{iφ2}`` on one
spectrometer give

    S(ω) = A1² + A2² + 2 A1 A2 cos Φ(ω),    Φ = φ_late − φ_early.

Fourier-transforming ``S`` over ω gives a central peak at zero delay (the
two spectra, ``A1² + A2²``) and two conjugate sidebands at ``±τ``. One
sideband, windowed and transformed back, is the complex interference term
``A1 A2 e^{iΦ(ω)}``. Its unwrapped phase is fitted around ``ω0`` as

    Φ(ω) = Φ0 + τ0 (ω − ω0) + ½ ΔGDD (ω − ω0)²    (ω in rad/fs),

weighted by the sideband's magnitude: each residual counts in proportion to
``|A1 A2|``, so the least-squares weights are ``|A1 A2|²`` (the inverse
variance of the phase under additive noise).

**Sign convention.** One intensity spectrum cannot tell the two arms apart:
``cos Φ`` is even, so ``(τ0, ΔGDD)`` and ``(−τ0, −ΔGDD)`` fit the data
equally well. This module picks the positive-delay sideband, so ``τ0 ≥ 0``
is the delay of the *later* pulse behind the *earlier* one at the
reference wavelength, and ``ΔGDD = GDD_late − GDD_early``, where
``GDD = dτ/dω`` (positive when the blue arrives after the red). Which
compressor arm is the later one has to come from outside the spectrum,
for example by moving one arm's delay and watching which way ``τ0`` goes.
Choosing ``sideband="negative"`` returns the conjugate solution.

The Jacobian ``dλ/dω`` is not applied when resampling the counts onto ω:
it changes the envelope ``A1 A2`` (so the weights only slightly) and never
the phase.

Pure numerics: no file reading, no pyplot. The diagnostic figure is
:func:`geecs_analysis.render.spectral_interferometry.diagnostic_figure`, the
per-shot measure ``spectral_interferometry``, and the one-file command
``python -m scan_analysis.tools.spectral_interferometry FILE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

#: Speed of light, nm/fs.
C_NM_PER_FS = 299.792458

SIGN_CONVENTION = (
    "tau0 >= 0 is the delay of the later pulse behind the earlier one at the "
    "reference wavelength (positive-delay sideband); delta_gdd = GDD_late - "
    "GDD_early with GDD = d(group delay)/d(omega). One intensity spectrum "
    "cannot tell the arms apart: (-tau0, -delta_gdd) fits equally well, so "
    "which arm is late must come from outside the spectrum."
)

#: Every reliability flag, in the order the messages report them.
FLAG_NAMES = (
    "sideband_not_found",
    "sideband_overlaps_central",
    "sideband_broadened",
    "window_sensitive_tau0",
    "window_sensitive_gdd",
    "fringes_undersampled",
    "fringes_unresolved",
    "narrow_bandwidth",
)

#: The flags that leave τ0 standing when it is reported near the fit band's
#: centre: there the slope is pinned whatever the curvature. Away from the
#: centre a loose ΔGDD enters τ0 with lever |ω0 − ω_c|, so every flag
#: counts. Every flag bears on ΔGDD.
GDD_ONLY_FLAGS = frozenset({"narrow_bandwidth", "window_sensitive_gdd"})


def tau0_flags(reference_central: bool) -> frozenset[str]:
    """The flags that make τ0 unreliable, given where it is reported."""
    exempt = GDD_ONLY_FLAGS if reference_central else frozenset()
    return frozenset(FLAG_NAMES) - exempt


class SpectralInterferometryConfig(BaseModel):
    """Overrides and the thresholds behind each reliability flag.

    Every override is optional; unset means "infer from the data". Ranges
    are ``(low, high)`` pairs; the sideband window is given in ``|delay|``
    and applied to the selected sideband.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolution_nm: Optional[float] = Field(
        None, gt=0, description="Spectrometer optical resolution (FWHM), nm."
    )
    fit_range_nm: Optional[tuple[float, float]] = Field(
        None, description="Wavelength range the phase is fitted over, nm."
    )
    reference_wavelength_nm: Optional[float] = Field(
        None,
        gt=0,
        description="Wavelength tau0 is reported at; default the fit band's centre.",
    )
    sideband_window_fs: Optional[tuple[float, float]] = Field(
        None, description="Sideband window in |delay|, fs; default found automatically."
    )
    spectral_region_nm: Optional[tuple[float, float]] = Field(
        None,
        description="Wavelength region transformed, nm; default where the signal is.",
    )
    sideband: Literal["positive", "negative"] = Field(
        "positive", description="Which conjugate sideband to recover the phase from."
    )
    fit_threshold: float = Field(
        0.15,
        gt=0,
        lt=1,
        description="Auto fit band: sideband magnitude above this fraction of its peak.",
    )
    edge_level: float = Field(
        0.05,
        gt=0,
        lt=1,
        description="Sideband edges: magnitude below this fraction of the sideband peak.",
    )
    window_pad: float = Field(
        0.25, ge=0, description="Auto window padding, fraction of the sideband width."
    )
    min_sideband_snr: float = Field(
        10.0,
        gt=0,
        description="A sideband peak must exceed the delay-domain noise floor this many times.",
    )
    min_sideband_ratio: float = Field(
        0.01,
        gt=0,
        lt=1,
        description=(
            "A sideband peak must reach this fraction of the central peak: "
            "below it, it is the central peak's sidelobe, not fringes."
        ),
    )
    min_signal_snr: float = Field(
        10.0,
        gt=0,
        description=(
            "The smoothed spectrum's peak must stand this many noise units above its "
            "mean; below it the shot holds no spectrum."
        ),
    )
    fit_floor: float = Field(
        0.05,
        gt=0,
        lt=1,
        description=(
            "A user fit range is limited to where the sideband magnitude exceeds "
            "this fraction of its peak: below it the phase is noise and unwraps badly."
        ),
    )
    min_valley_dip: float = Field(
        0.5,
        gt=0,
        lt=1,
        description="A sideband must dip below this fraction of its peak on the way to zero delay.",
    )
    broadening_limit: float = Field(
        3.0,
        gt=1,
        description="Flag when the sideband is this many times its unchirped width.",
    )
    broadening_reach: float = Field(
        0.5,
        gt=0,
        lt=1,
        description="...and its window reaches below this fraction of the sideband's delay.",
    )
    sensitivity_step: float = Field(
        0.1,
        gt=0,
        lt=1,
        description="Window-sensitivity test: move each edge by this fraction of the width.",
    )
    sensitivity_tau_fs: float = Field(
        1.0, gt=0, description="Smallest material tau0 change, fs."
    )
    sensitivity_gdd_fs2: float = Field(
        20.0, gt=0, description="Smallest material delta GDD change, fs²."
    )
    sensitivity_gdd_rel: float = Field(
        0.1, gt=0, description="Material delta GDD change relative to its value."
    )
    min_samples_per_fringe: float = Field(
        3.0,
        gt=0,
        description="Detector pixels per fringe below which fringes are undersampled.",
    )
    min_resolution_visibility: float = Field(
        0.5,
        gt=0,
        lt=1,
        description="Fringe visibility left by the optical resolution below which it is flagged.",
    )
    min_fit_pixels: int = Field(
        20, ge=4, description="Detector pixels the fit band must span."
    )
    gdd_sigma_floor_fs2: float = Field(
        50.0, gt=0, description="GDD uncertainty always tolerated, fs²."
    )
    gdd_sigma_rel: float = Field(
        0.25, gt=0, description="GDD uncertainty tolerated relative to |delta GDD|."
    )
    taper: float = Field(
        0.2, ge=0, le=1, description="Tukey taper of the transformed region (fraction)."
    )
    window_taper: float = Field(
        0.25, ge=0, le=1, description="Tukey taper of the sideband window (fraction)."
    )
    pad_factor: int = Field(
        4,
        ge=1,
        description="Zero padding of the transform (delay-domain interpolation).",
    )


class SpectralInterferometryFlags(BaseModel):
    """One boolean per reliability check.

    Any True makes ΔGDD unreliable. For τ0, see :func:`tau0_flags`: with the
    reference in the middle half of the fit band, ``GDD_ONLY_FLAGS`` (a band
    too narrow for the curvature, a window-sensitive ΔGDD) leave it standing.
    """

    sideband_not_found: bool = Field(
        description="No sideband stands clear of the central peak and the noise floor; no fit."
    )
    sideband_overlaps_central: bool = Field(
        description="The sideband does not fall to edge_level before the central peak, or the window reaches into it."
    )
    sideband_broadened: bool = Field(
        description=(
            "The sideband is broadening_limit times its unchirped width and reaches below "
            "broadening_reach of its delay: strong ΔGDD, clean isolation doubtful."
        )
    )
    window_sensitive_tau0: bool = Field(
        description="Moving the window edges changed tau0 materially."
    )
    window_sensitive_gdd: bool = Field(
        description="Moving the window edges changed ΔGDD materially (ΔGDD only)."
    )
    fringes_undersampled: bool = Field(
        description="Fewer than min_samples_per_fringe detector pixels per fringe."
    )
    fringes_unresolved: bool = Field(
        description="The optical resolution leaves less than min_resolution_visibility of the fringe contrast."
    )
    narrow_bandwidth: bool = Field(
        description="The fit band is too few pixels, or too narrow for the GDD to be constrained."
    )


class SpectralInterferometryResult(BaseModel):
    """The machine-readable result: fit values, uncertainties, windows and flags.

    Values the routine could not determine are ``None`` (``null`` in JSON),
    never zero: a merged peak is not evidence of zero delay.
    """

    tau0_fs: Optional[float] = Field(
        description="Relative group delay at the reference wavelength, fs."
    )
    tau0_err_fs: Optional[float] = Field(description="1σ uncertainty of tau0, fs.")
    delta_gdd_fs2: Optional[float] = Field(
        description="Differential GDD (late − early), fs²."
    )
    delta_gdd_err_fs2: Optional[float] = Field(
        description="1σ uncertainty of delta_gdd, fs²."
    )
    phi0_rad: Optional[float] = Field(
        description="Interference phase at the reference, rad (mod 2π)."
    )
    reference_wavelength_nm: float = Field(
        description="Wavelength tau0 is reported at, nm."
    )
    reference_omega_rad_per_fs: float = Field(
        description="The reference as angular frequency, rad/fs."
    )
    reference_source: Literal["band_centre", "user"]
    sideband: Literal["positive", "negative"]
    spectral_region_nm: tuple[float, float] = Field(
        description="Wavelength region transformed, nm."
    )
    fit_range_nm: Optional[tuple[float, float]] = Field(
        description="Wavelength band the phase was fitted over, nm."
    )
    fit_range_source: Literal["auto", "user"]
    sideband_window_fs: Optional[tuple[float, float]] = Field(
        description="Sideband window in |delay|, fs."
    )
    window_source: Literal["auto", "user"]
    sideband_peak_fs: Optional[float] = Field(
        description="Delay of the sideband's maximum, fs."
    )
    central_edge_fs: Optional[float] = Field(
        description="Where the central peak falls below edge_level × the sideband peak, fs."
    )
    signal_snr: float = Field(
        description="The smoothed spectrum's peak above its mean, in pixel-noise units."
    )
    sideband_snr: Optional[float] = Field(
        description="Sideband peak over the delay-domain noise floor."
    )
    sideband_broadening: Optional[float] = Field(
        description="Sideband RMS width over its unchirped RMS width."
    )
    delay_sweep_fs: Optional[float] = Field(
        description="|ΔGDD| × fit-band width: the spread of local delays, fs."
    )
    rms_bandwidth_rad_per_fs: Optional[float] = Field(
        description="Intensity-weighted RMS bandwidth σ_ω of the measured spectrum (I1 + I2), rad/fs."
    )
    tl_rms_fs: Optional[float] = Field(
        description="RMS duration σ_t of the transform-limited pulse with that spectrum, fs."
    )
    delta_duration_rms_fs: Optional[float] = Field(
        description=(
            "How much ΔGDD lengthens the RMS duration of a transform-limited arm, fs: "
            "√(σ_TL² + (ΔGDD σ_ω)²) − σ_TL, exact for any spectral shape (≥ 0, even in "
            "ΔGDD; assumes the other arm is at transform limit)."
        )
    )
    delta_duration_rms_err_fs: Optional[float] = Field(
        description=(
            "1σ of delta_duration_rms from σ(ΔGDD), fs: the larger change over "
            "ΔGDD ± σ, upward only near ΔGDD = 0."
        )
    )
    window_tau_shift_fs: Optional[float] = Field(
        description="Largest tau0 change in the window test, fs."
    )
    window_gdd_shift_fs2: Optional[float] = Field(
        description="Largest ΔGDD change in the window test, fs²."
    )
    min_samples_per_fringe: Optional[float] = Field(
        description="Fewest detector pixels per fringe in the band."
    )
    min_resolution_visibility: Optional[float] = Field(
        description="Lowest fringe visibility the optical resolution allows in the band (null: resolution not given)."
    )
    rms_residual_rad: Optional[float] = Field(
        description="Weighted RMS of the phase residual, rad."
    )
    n_fit_pixels: int = Field(description="Detector pixels inside the fit band.")
    flags: SpectralInterferometryFlags
    reference_central: bool = Field(
        description="The reference lies in the middle half of the fit band (in ω)."
    )
    tau0_reliable: bool = Field(
        description=(
            "True when no flag bearing on tau0 is set: all but GDD_ONLY_FLAGS with a "
            "central reference, every flag otherwise."
        )
    )
    delta_gdd_reliable: bool = Field(description="True when no flag is set.")
    reliable: bool = Field(description="Both tau0 and ΔGDD reliable (no flag set).")
    messages: list[str] = Field(description="One sentence per raised flag.")
    sign_convention: str = SIGN_CONVENTION


@dataclass(frozen=True)
class SpectralInterferometryTraces:
    """The arrays behind the diagnostic figure (not part of the JSON result)."""

    wavelength_nm: np.ndarray
    counts: np.ndarray
    baseline: float
    omega: np.ndarray
    delay_fs: np.ndarray
    delay_magnitude: np.ndarray
    band_omega: np.ndarray
    band_magnitude: np.ndarray
    band_phase: np.ndarray
    fitted_phase: np.ndarray


@dataclass(frozen=True)
class SpectralInterferometryAnalysis:
    """The result and the traces the diagnostic figure draws."""

    result: SpectralInterferometryResult
    traces: SpectralInterferometryTraces


def wavelength_to_omega(wavelength_nm: np.ndarray | float) -> np.ndarray:
    """Wavelength (nm) to angular frequency (rad/fs)."""
    return 2.0 * np.pi * C_NM_PER_FS / np.asarray(wavelength_nm, dtype=float)


def omega_to_wavelength(omega: np.ndarray | float) -> np.ndarray:
    """Angular frequency (rad/fs) to wavelength (nm)."""
    return 2.0 * np.pi * C_NM_PER_FS / np.asarray(omega, dtype=float)


def _nm_range_to_omega(range_nm: tuple[float, float]) -> tuple[float, float]:
    low, high = sorted(float(v) for v in range_nm)
    if low <= 0:
        raise ValueError(f"wavelength range must be positive: {range_nm}")
    return float(wavelength_to_omega(high)), float(wavelength_to_omega(low))


def _omega_range_to_nm(lo: float, hi: float) -> tuple[float, float]:
    return float(omega_to_wavelength(hi)), float(omega_to_wavelength(lo))


def _tukey(n: int, alpha: float) -> np.ndarray:
    """Tukey (tapered cosine) window of ``n`` samples."""
    if n <= 0:
        return np.zeros(0)
    if alpha <= 0 or n == 1:
        return np.ones(n)
    x = np.linspace(0.0, 1.0, n)
    w = np.ones(n)
    edge = alpha / 2.0
    left = x < edge
    right = x > 1.0 - edge
    w[left] = 0.5 * (1 + np.cos(np.pi * (x[left] / edge - 1)))
    w[right] = 0.5 * (1 + np.cos(np.pi * ((x[right] - 1) / edge + 1)))
    return w


#: Fewest pixels an inferred signal region spans.
_MIN_REGION = 64
#: Median over 25th percentile of a Rayleigh distribution: √(ln 2 / ln(4/3)).
_RAYLEIGH_Q25_TO_MEDIAN = float(np.sqrt(np.log(2.0) / np.log(4.0 / 3.0)))


def _rayleigh_median(magnitudes: np.ndarray) -> float:
    """The noise median from the 25th percentile: robust to signal in up to ~70 % of bins."""
    return float(np.percentile(magnitudes, 25)) * _RAYLEIGH_Q25_TO_MEDIAN


def _signal_region(omega: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """Where the spectrum stands above its background, padded by 15 % each side."""
    n = counts.size
    k = _smoothing(n)
    smooth = np.convolve(counts, np.ones(k) / k, mode="same")
    baseline = float(np.median(counts))
    peak = float(np.max(smooth))
    if not peak > baseline:
        # Nothing stands up: the dark check downstream reports it.
        return float(omega[0]), float(omega[-1])
    above = np.flatnonzero(smooth > baseline + 0.05 * (peak - baseline))
    first, last = _cluster_around(above, int(np.argmax(smooth)), max(n // 20, 1))
    # At least _MIN_REGION pixels, so a lone hot pixel is analysed (and
    # found wanting), not refused.
    short = _MIN_REGION - (last - first + 1)
    if short > 0:
        first = max(0, first - short // 2 - 1)
        last = min(n - 1, first + _MIN_REGION - 1)
        first = max(0, last - _MIN_REGION + 1)
    lo, hi = float(omega[first]), float(omega[last])
    pad = 0.15 * (hi - lo)
    return max(lo - pad, float(omega[0])), min(hi + pad, float(omega[-1]))


def _cluster_around(indices: np.ndarray, at: int, gap: int) -> tuple[int, int]:
    """The span of ``indices`` around ``at``, bridging gaps shorter than ``gap``.

    Fringe minima open short gaps in the spectrum and are bridged; an
    isolated noise excursion far from the spectrum is not, so it cannot
    stretch the region (and with it σ_ω) to the detector's edge.
    """
    first = last = int(indices[np.argmin(np.abs(indices - at))])
    for index in indices[indices > at]:
        if index - last > gap:
            break
        last = int(index)
    for index in indices[indices < at][::-1]:
        if first - index > gap:
            break
        first = int(index)
    return first, last


def _contiguous_run(mask: np.ndarray, at: int) -> np.ndarray:
    """Indices of the run of True in ``mask`` containing ``at``."""
    if not mask[at]:
        return np.zeros(0, dtype=int)
    lo = at
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = at
    while hi < mask.size - 1 and mask[hi + 1]:
        hi += 1
    return np.arange(lo, hi + 1)


def _rms_width(t: np.ndarray, power: np.ndarray) -> float:
    total = float(np.sum(power))
    if total <= 0:
        return float("nan")
    centre = float(np.sum(t * power)) / total
    return float(np.sqrt(np.sum((t - centre) ** 2 * power) / total))


@dataclass(frozen=True)
class _Fit:
    phi0: float
    tau0: float
    gdd: float
    tau0_err: float
    gdd_err: float
    rms: float
    phase: np.ndarray
    fitted: np.ndarray


class _Transform:
    """The resampled spectrum's delay-domain transform and sideband filtering."""

    def __init__(self, omega_grid: np.ndarray, signal: np.ndarray, pad_factor: int):
        self.omega = omega_grid
        self.n = omega_grid.size
        self.d_omega = float(omega_grid[1] - omega_grid[0])
        m = 1 << int(np.ceil(np.log2(self.n * pad_factor)))
        self.m = m
        self.spectrum = np.fft.fft(signal, m)
        self.delay = 2.0 * np.pi * np.fft.fftfreq(m, self.d_omega)
        self.magnitude = np.abs(self.spectrum)
        half = m // 2
        self.positive_delay = self.delay[:half]
        self.positive_magnitude = self.magnitude[:half]

    def index(self, delay_fs: float) -> int:
        step = 2.0 * np.pi / (self.m * self.d_omega)
        return int(np.clip(round(delay_fs / step), 1, self.m // 2 - 1))

    def filtered(self, lo: int, hi: int, side: str, taper: float) -> np.ndarray:
        """The complex interference term on the ω grid from window ``[lo, hi]``."""
        return np.fft.ifft(self.spectrum * self.window(lo, hi, side, taper))[: self.n]

    def window(self, lo: int, hi: int, side: str, taper: float) -> np.ndarray:
        """The Tukey window over delay bins ``[lo, hi]`` of the chosen side."""
        window = np.zeros(self.m)
        idx = np.arange(lo, hi + 1)
        shape = _tukey(idx.size, taper)
        if side == "positive":
            window[idx] = shape
        else:
            window[self.m - idx] = shape
        return window

    def noise_covariance(self, window: np.ndarray, noise_power: float) -> np.ndarray:
        """Covariance C(d) of the filtered term's noise at grid lag ``d`` (mod M).

        White noise of total power ``noise_power`` (``E|F|²``) spread over
        the ``n`` grid samples, passed through ``window``.
        """
        return noise_power / self.n * np.fft.ifft(window**2)


def _fit_phase(
    term: np.ndarray,
    omega: np.ndarray,
    band: np.ndarray,
    omega0: float,
    demod: float,
    noise_cov: np.ndarray,
) -> Optional[_Fit]:
    """Unwrap the term's phase over ``band`` and fit the quadratic about ``omega0``."""
    if band.size < 4:
        return None
    w_omega = omega[band]
    z = term[band]
    phase = np.unwrap(np.angle(z * np.exp(-1j * demod * w_omega))) + demod * w_omega
    x = w_omega - omega0
    weight = np.abs(z) ** 2
    if not np.any(weight > 0):
        return None
    weight = weight / weight.max()
    design = np.stack([np.ones_like(x), x, x * x], axis=1)
    root = np.sqrt(weight)
    coeffs, *_ = np.linalg.lstsq(design * root[:, None], phase * root, rcond=None)
    fitted = design @ coeffs
    residual = phase - fitted
    # Uncertainty: the weighted least-squares estimator's covariance under
    # the noise actually in the term (the "sandwich" A⁻¹ B A⁻¹). The window
    # band-limits that noise, so neighbouring grid points are correlated
    # over 2π / (window width): B carries the full covariance, not a
    # diagonal. Phase noise is Im(δa e^{-iφ}) / |a|, and for circular δa
    # Cov(φ_i, φ_j) = Re(C(i−j) e^{-i(φ_i−φ_j)}) / (2 |a_i| |a_j|), so with
    # weights |a|² every factor folds into the term ``a`` itself.
    nb = band.size
    try:
        a_inv = np.linalg.inv(design.T @ (design * weight[:, None]))
    except np.linalg.LinAlgError:
        return None
    wmax = float(np.max(np.abs(z) ** 2))
    kernel = noise_cov[np.arange(-(nb - 1), nb) % noise_cov.size]
    left = design * (np.conj(z) / wmax)[:, None]
    right = design * (z / wmax)[:, None]
    b = np.empty((3, 3))
    for col in range(3):
        folded = np.convolve(kernel, right[:, col])[nb - 1 : 2 * nb - 1]
        b[:, col] = 0.5 * np.real(left.T @ folded)
    cov = a_inv @ b @ a_inv
    rms = float(np.sqrt(np.sum(weight * residual**2) / np.sum(weight)))
    return _Fit(
        phi0=float(np.angle(np.exp(1j * coeffs[0]))),
        tau0=float(coeffs[1]),
        gdd=float(2.0 * coeffs[2]),
        tau0_err=float(np.sqrt(max(cov[1, 1], 0.0))),
        gdd_err=float(2.0 * np.sqrt(max(cov[2, 2], 0.0))),
        rms=rms,
        phase=phase,
        fitted=fitted,
    )


def _find_sideband(
    transform: _Transform, cfg: SpectralInterferometryConfig, noise: float
) -> Optional[int]:
    """The most prominent local maximum clear of the central peak and the noise."""
    mag = transform.positive_magnitude
    valley = np.minimum.accumulate(np.concatenate([[np.inf], mag[1:]]))
    best: Optional[int] = None
    for i in range(2, mag.size - 1):
        if mag[i] < mag[i - 1] or mag[i] < mag[i + 1]:
            continue
        if mag[i] < max(cfg.min_sideband_snr * noise, cfg.min_sideband_ratio * mag[0]):
            continue
        if valley[i - 1] > cfg.min_valley_dip * mag[i]:
            continue
        if best is None or mag[i] > mag[best]:
            best = i
    return best


_MESSAGES = {
    "sideband_not_found": (
        "No interference sideband stands clear of the central peak and the noise floor "
        "(or the shot holds no spectrum); no delay is reported (a merged peak is not "
        "evidence of zero delay)."
    ),
    "sideband_overlaps_central": "The selected sideband overlaps the central peak.",
    "sideband_broadened": (
        "Differential dispersion broadens the sideband; clean isolation is doubtful."
    ),
    "window_sensitive_tau0": "Changing the sideband window changes tau0 materially.",
    "window_sensitive_gdd": "Changing the sideband window changes delta GDD materially.",
    "fringes_undersampled": (
        "The fringes are too fine for the detector's pixel sampling."
    ),
    "fringes_unresolved": "The fringes are too fine for the stated optical resolution.",
    "narrow_bandwidth": (
        "The usable bandwidth is too narrow for a reliable GDD fit (tau0 is unaffected)."
    ),
}


@dataclass(frozen=True)
class _Spectrum:
    """The finite input, the region transformed and its uniform-ω resampling."""

    wavelength_nm: np.ndarray
    counts: np.ndarray
    region: tuple[float, float]
    baseline: float
    pixel_omega: np.ndarray
    grid: np.ndarray
    signal: np.ndarray
    signal_snr: float
    pixel_sigma: float


@dataclass(frozen=True)
class _Sideband:
    """Delay bins (positive side) of the sideband, its window and the central edge."""

    peak: int
    lo: int
    hi: int
    edge_lo: int
    central: int
    overlaps: bool


@dataclass
class _Recovery:
    """What the phase recovery found; ``None`` where it could not get that far."""

    flags: dict[str, bool]
    term: Optional[np.ndarray] = None
    band: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    omega0: Optional[float] = None
    fit: Optional[_Fit] = None
    broadening: Optional[float] = None
    sweep: Optional[float] = None
    tau_shift: Optional[float] = None
    gdd_shift: Optional[float] = None
    spf_min: Optional[float] = None
    vis_min: Optional[float] = None
    n_fit_pixels: int = 0
    rms_bandwidth: Optional[float] = None
    tl_rms: Optional[float] = None
    delta_rms: Optional[float] = None
    delta_rms_err: Optional[float] = None


def _prepare(
    wavelength_nm: np.ndarray, counts: np.ndarray, cfg: SpectralInterferometryConfig
) -> _Spectrum:
    """Drop nonfinite samples, find the region and resample it onto uniform ω."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float).ravel()
    counts = np.asarray(counts, dtype=float).ravel()
    if wavelength_nm.shape != counts.shape:
        raise ValueError("wavelength and counts must have the same length")
    keep = np.isfinite(wavelength_nm) & np.isfinite(counts) & (wavelength_nm > 0)
    if keep.sum() < 16:
        raise ValueError("a spectrum needs at least 16 finite samples")
    wl, ct = wavelength_nm[keep], counts[keep]
    omega = wavelength_to_omega(wl)
    order = np.argsort(omega)
    omega, sorted_counts = omega[order], ct[order]
    pixel_sigma = _pixel_noise(sorted_counts)
    signal_snr = _signal_snr(sorted_counts, pixel_sigma)

    # The region transformed: where the signal is, or the user's choice,
    # always covering a user fit range.
    if cfg.spectral_region_nm is not None:
        region = _nm_range_to_omega(cfg.spectral_region_nm)
    else:
        region = _signal_region(omega, sorted_counts)
    if cfg.fit_range_nm is not None:
        f_lo, f_hi = _nm_range_to_omega(cfg.fit_range_nm)
        region = (min(region[0], f_lo), max(region[1], f_hi))
    inside = (omega >= region[0]) & (omega <= region[1])
    if inside.sum() < 16:
        raise ValueError("the spectral region holds fewer than 16 samples")
    outside = sorted_counts[~inside]
    baseline = float(
        np.median(outside) if outside.size >= 10 else np.percentile(sorted_counts, 5)
    )
    pixel_omega = omega[inside]
    steps = np.diff(pixel_omega)
    if not np.any(steps > 0):
        raise ValueError("the spectral region has no wavelength spread")
    # The finest pixel spacing, so the resampling loses nothing.
    d_omega = float(steps[steps > 0].min())
    n = min(int((pixel_omega[-1] - pixel_omega[0]) / d_omega) + 1, 1 << 16)
    grid = np.linspace(pixel_omega[0], pixel_omega[-1], n)
    resampled = np.interp(grid, pixel_omega, sorted_counts[inside]) - baseline
    return _Spectrum(
        wavelength_nm=wl,
        counts=ct,
        region=region,
        baseline=baseline,
        pixel_omega=pixel_omega,
        grid=grid,
        signal=resampled * _tukey(n, cfg.taper),
        signal_snr=signal_snr,
        pixel_sigma=pixel_sigma,
    )


def _pixel_noise(counts: np.ndarray) -> float:
    """The detector's white pixel noise σ, from the raw pixels' own transform.

    The 25th-percentile (Rayleigh-scaled) magnitude over the top quarter of
    pixel frequencies: fringes coarser than 2.7 pixels each stay below that
    band however much of the detector they fill, and finer ones (beyond what
    the analysis can use) may fill most of it without raising the estimate.
    ``σ = median / √(N ln 2)``; integer counts keep it above zero.
    """
    centred = counts - np.mean(counts)
    floor = np.abs(np.fft.rfft(centred))[3 * counts.size // 8 :]
    return _rayleigh_median(floor) / np.sqrt(counts.size * np.log(2.0))


def _signal_snr(counts: np.ndarray, sigma: float) -> float:
    """How far the smoothed spectrum's peak stands above its mean, in noise units.

    The mean, not the median, is the reference, so a background clipped at
    zero does not stand above itself.
    """
    k = _smoothing(counts.size)
    smooth = np.convolve(counts - np.mean(counts), np.ones(k) / k, mode="valid")
    height = float(np.max(smooth))
    if not sigma > 0:
        return float("inf") if height > 0 else 0.0
    return height / (sigma / np.sqrt(k))


def _smoothing(n: int) -> int:
    """Odd moving-average length for finding the signal: ~0.25 % of the trace."""
    return max(5, (n // 400) | 1)


def _shape_window(
    mag: np.ndarray,
    peak: int,
    user: Optional[tuple[int, int]],
    cfg: SpectralInterferometryConfig,
) -> _Sideband:
    """The central peak's edge at the sideband's scale, and the window."""
    level = cfg.edge_level * mag[peak]
    below = np.flatnonzero(mag[1:peak] < level)
    # No dip below the level before the sideband: the two are merged.
    merged = below.size == 0
    central = 1 + int(
        below[0] if below.size else np.argmin(mag[1:peak]) if peak > 1 else 0
    )
    if user is not None:
        lo, hi = user
        return _Sideband(peak, lo, hi, lo, central, merged or lo < central)
    lo = peak
    while lo > central and mag[lo] >= level:
        lo -= 1
    hi = peak
    while hi < mag.size - 2 and mag[hi] >= level:
        hi += 1
    pad = int(round(cfg.window_pad * (hi - lo)))
    return _Sideband(
        peak, max(lo - pad, central), min(hi + pad, mag.size - 2), lo, central, merged
    )


def _locate_sideband(
    transform: _Transform, cfg: SpectralInterferometryConfig, noise: float
) -> Optional[_Sideband]:
    """The sideband automatically, or the maximum inside the user's window."""
    mag = transform.positive_magnitude
    user: Optional[tuple[int, int]] = None
    if cfg.sideband_window_fs is not None:
        u_lo, u_hi = sorted(abs(float(v)) for v in cfg.sideband_window_fs)
        user = (transform.index(u_lo), transform.index(u_hi))
        if user[1] <= user[0]:
            raise ValueError(f"sideband window too narrow: {cfg.sideband_window_fs}")
        peak = user[0] + int(np.argmax(mag[user[0] : user[1] + 1]))
        if mag[peak] < max(
            cfg.min_sideband_snr * noise, cfg.min_sideband_ratio * mag[0]
        ):
            return None
    else:
        located = _find_sideband(transform, cfg, noise)
        if located is None:
            return None
        peak = located
    return _shape_window(mag, peak, user, cfg)


def _fit_band(
    magnitude: np.ndarray, grid: np.ndarray, cfg: SpectralInterferometryConfig
) -> np.ndarray:
    """The user's fit range (where the fringes are measurable), or the strongest run."""
    if cfg.fit_range_nm is not None:
        f_lo, f_hi = _nm_range_to_omega(cfg.fit_range_nm)
        inside = (grid >= f_lo) & (grid <= f_hi)
        if not inside.any():
            return np.zeros(0, dtype=int)
        held = np.where(inside, magnitude, 0.0)
        return _contiguous_run(held >= cfg.fit_floor * held.max(), int(np.argmax(held)))
    return _contiguous_run(
        magnitude >= cfg.fit_threshold * magnitude.max(), int(np.argmax(magnitude))
    )


def _window_shifts(
    transform: _Transform,
    sb: _Sideband,
    cfg: SpectralInterferometryConfig,
    fit: _Fit,
    refit: Callable[[int, int], Optional[_Fit]],
) -> tuple[Optional[float], Optional[float], bool, bool]:
    """Move each window edge in and out: the largest τ0 and ΔGDD changes, and their flags."""
    step = max(1, int(round(cfg.sensitivity_step * (sb.hi - sb.lo))))
    top = transform.positive_magnitude.size - 2
    shifts = []
    for v_lo, v_hi in (
        (sb.lo - step, sb.hi),
        (sb.lo + step, sb.hi),
        (sb.lo, sb.hi - step),
        (sb.lo, sb.hi + step),
    ):
        v_lo, v_hi = max(v_lo, 1), min(v_hi, top)
        variant = refit(v_lo, v_hi) if v_hi - v_lo >= 2 else None
        if variant is not None:
            shifts.append((abs(variant.tau0 - fit.tau0), abs(variant.gdd - fit.gdd)))
    if not shifts:
        return None, None, False, False
    tau_shift = max(s[0] for s in shifts)
    gdd_shift = max(s[1] for s in shifts)
    tau_limit = max(cfg.sensitivity_tau_fs, 3.0 * fit.tau0_err)
    gdd_limit = max(
        cfg.sensitivity_gdd_fs2,
        cfg.sensitivity_gdd_rel * abs(fit.gdd),
        3.0 * fit.gdd_err,
    )
    return tau_shift, gdd_shift, tau_shift > tau_limit, gdd_shift > gdd_limit


def _broadening(
    transform: _Transform,
    sb: _Sideband,
    magnitude: np.ndarray,
    cfg: SpectralInterferometryConfig,
) -> tuple[Optional[float], bool]:
    """The sideband's RMS width over its unchirped self (the FT of ``|a(ω)|``)."""
    tpos, mag = transform.positive_delay, transform.positive_magnitude
    sel = np.arange(sb.lo, sb.hi + 1)
    measured = _rms_width(tpos[sel], mag[sel] ** 2)
    unchirped = np.abs(np.fft.fft(magnitude, transform.m)) ** 2
    reference = _rms_width(np.fft.fftshift(transform.delay), np.fft.fftshift(unchirped))
    if not reference > 0:
        return None, False
    ratio = measured / reference
    # Broad *and* reaching back toward zero delay: the chirp smears the
    # sideband across the gap that isolates it.
    reaches = tpos[sb.edge_lo] < cfg.broadening_reach * tpos[sb.peak]
    return ratio, bool(ratio > cfg.broadening_limit and reaches)


def _fringe_checks(
    spectrum: _Spectrum,
    band_omega: np.ndarray,
    local_delay: np.ndarray,
    cfg: SpectralInterferometryConfig,
) -> tuple[float, Optional[float]]:
    """Fewest detector pixels per fringe, and the lowest resolution-limited visibility."""
    pixels = spectrum.pixel_omega
    mids = 0.5 * (pixels[1:] + pixels[:-1])
    pixel_step = np.interp(band_omega, mids, np.diff(pixels))
    with np.errstate(divide="ignore"):
        spf_min = float(np.min(2.0 * np.pi / (local_delay * pixel_step)))
    if cfg.resolution_nm is None:
        return spf_min, None
    # A Gaussian instrument function of FWHM δω scales the fringe contrast
    # by exp(−(τ δω)² / (16 ln 2)).
    res_omega = (
        2.0
        * np.pi
        * C_NM_PER_FS
        * cfg.resolution_nm
        / omega_to_wavelength(band_omega) ** 2
    )
    vis = np.exp(-((local_delay * res_omega) ** 2) / (16.0 * np.log(2.0)))
    return spf_min, float(np.min(vis))


def _recover(
    spectrum: _Spectrum,
    transform: _Transform,
    sb: _Sideband,
    cfg: SpectralInterferometryConfig,
    noise_power: float,
) -> _Recovery:
    """Window, inverse-transform, fit and check one located sideband."""
    rec = _Recovery(flags=dict.fromkeys(FLAG_NAMES, False))
    rec.flags["sideband_overlaps_central"] = sb.overlaps
    side, grid = cfg.sideband, spectrum.grid
    tpos, mag = transform.positive_delay, transform.positive_magnitude
    rec.term = transform.filtered(sb.lo, sb.hi, side, cfg.window_taper)
    magnitude = np.abs(rec.term)
    rec.band = band = _fit_band(magnitude, grid, cfg)
    rec.broadening, rec.flags["sideband_broadened"] = _broadening(
        transform, sb, magnitude, cfg
    )
    if band.size == 0:
        rec.flags["narrow_bandwidth"] = True
        return rec
    band_omega = grid[band]
    rec.omega0 = omega0 = (
        float(wavelength_to_omega(cfg.reference_wavelength_nm))
        if cfg.reference_wavelength_nm is not None
        else 0.5 * float(band_omega[0] + band_omega[-1])
    )
    demod = (1.0 if side == "positive" else -1.0) * float(tpos[sb.peak])
    # The noise under the sideband, from the delays just beyond it: the
    # resampling's interpolation attenuates the far delays, so the outer
    # floor would understate it.
    beyond = mag[sb.hi + 1 : sb.hi + 1 + 2 * (sb.hi - sb.lo)]
    if beyond.size >= 16:
        noise_power = float(np.median(beyond)) ** 2 / np.log(2.0)
    noise_cov = transform.noise_covariance(
        transform.window(sb.lo, sb.hi, side, cfg.window_taper), noise_power
    )
    rec.fit = fit = _fit_phase(rec.term, grid, band, omega0, demod, noise_cov)
    rec.n_fit_pixels = int(
        np.sum(
            (spectrum.pixel_omega >= band_omega[0])
            & (spectrum.pixel_omega <= band_omega[-1])
        )
    )
    if fit is None:
        local_delay = np.full(band.size, float(tpos[sb.hi]))
    else:
        local_delay = np.abs(fit.tau0 + fit.gdd * (band_omega - omega0))
        rec.sweep = abs(fit.gdd) * float(band_omega[-1] - band_omega[0])
        (
            rec.tau_shift,
            rec.gdd_shift,
            rec.flags["window_sensitive_tau0"],
            rec.flags["window_sensitive_gdd"],
        ) = _window_shifts(
            transform,
            sb,
            cfg,
            fit,
            lambda lo, hi: _fit_phase(
                transform.filtered(lo, hi, side, cfg.window_taper),
                grid,
                band,
                omega0,
                demod,
                noise_cov,
            ),
        )
    rec.spf_min, rec.vis_min = _fringe_checks(spectrum, band_omega, local_delay, cfg)
    rec.flags["fringes_undersampled"] = rec.spf_min < cfg.min_samples_per_fringe
    rec.flags["fringes_unresolved"] = (
        rec.vis_min is not None and rec.vis_min < cfg.min_resolution_visibility
    )
    loose = fit is not None and fit.gdd_err > max(
        cfg.gdd_sigma_floor_fs2, cfg.gdd_sigma_rel * abs(fit.gdd)
    )
    rec.flags["narrow_bandwidth"] = (
        fit is None or rec.n_fit_pixels < cfg.min_fit_pixels or loose
    )
    if fit is not None:
        spectrum_shape = _arm_spectrum(transform, sb, cfg)
        rec.rms_bandwidth, rec.tl_rms, rec.delta_rms, rec.delta_rms_err = (
            _rms_durations(grid, spectrum_shape, fit)
        )
    return rec


#: The transform-limited RMS duration is taken within this many of its own
#: FWHMs of the centroid: noise and the region's truncation put a faint
#: pedestal far out in time that the t² weight would otherwise inflate.
_RMS_TIME_REACH = 5.0


def _arm_spectrum(
    transform: _Transform, sb: _Sideband, cfg: SpectralInterferometryConfig
) -> np.ndarray:
    """The measured spectrum without its fringes: the central term, I1 + I2.

    The transform's central peak, low-passed out to the sideband window's
    inner edge (a half-Tukey roll-off) and transformed back, is the two
    arms' summed spectrum, smooth and nearly free of noise. The region's
    taper is divided back out where it is above 10 %; below, the edge is
    dropped.
    """
    reach = np.arange(sb.lo + 1)
    roll = _tukey(2 * reach.size - 1, 0.5)[reach.size - 1 :]
    window = np.zeros(transform.m)
    window[reach] = roll
    window[(transform.m - reach[1:]) % transform.m] = roll[1:]
    central = np.real(np.fft.ifft(transform.spectrum * window))[: transform.n]
    taper = _tukey(transform.n, cfg.taper)
    shape = np.where(taper >= 0.1, central / np.maximum(taper, 0.1), 0.0)
    return np.maximum(shape, 0.0)


def _rms_bandwidth(omega: np.ndarray, intensity: np.ndarray) -> float:
    centre = float(np.sum(omega * intensity) / np.sum(intensity))
    return float(np.sqrt(np.sum((omega - centre) ** 2 * intensity) / np.sum(intensity)))


def _tl_rms_duration(omega: np.ndarray, intensity: np.ndarray) -> float:
    """RMS duration of the flat-phase pulse with this intensity spectrum, fs."""
    m = 1 << int(np.ceil(np.log2(omega.size * 8)))
    field = np.fft.fftshift(np.fft.fft(np.sqrt(intensity), m))
    power = np.abs(field) ** 2
    t = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(m, float(omega[1] - omega[0])))
    centre = float(np.sum(t * power) / np.sum(power))
    near = np.abs(t - centre) <= _RMS_TIME_REACH * _intensity_fwhm(t, power)
    t, power = t[near], power[near]
    centre = float(np.sum(t * power) / np.sum(power))
    return float(np.sqrt(np.sum((t - centre) ** 2 * power) / np.sum(power)))


def _intensity_fwhm(t: np.ndarray, power: np.ndarray) -> float:
    """FWHM of a single-peaked trace, crossings interpolated linearly."""
    top = int(np.argmax(power))
    half = 0.5 * float(power[top])
    above = np.flatnonzero(power >= half)
    lo, hi = int(above[0]), int(above[-1])

    def crossing(below: int, inside: int) -> float:
        span = power[inside] - power[below]
        return float(t[below] + (half - power[below]) * (t[inside] - t[below]) / span)

    left = crossing(lo - 1, lo) if lo > 0 else float(t[0])
    right = crossing(hi + 1, hi) if hi < power.size - 1 else float(t[-1])
    return right - left


def _rms_durations(
    omega: np.ndarray, intensity: np.ndarray, fit: _Fit
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """σ_ω, the transform-limited σ_t, the RMS lengthening ΔGDD gives, and its σ.

    For any spectrum with a quadratic phase, the group delay is
    ``ΔGDD (ω − ω0)`` and ``σ_t² = σ_TL² + ΔGDD² σ_ω²`` exactly (the
    flat-phase pulse's own duration plus the spread of group delays,
    weighted by the spectrum). Assuming one arm is at transform limit, the
    other is longer by ``√(σ_TL² + (ΔGDD σ_ω)²) − σ_TL``. It is even in
    ΔGDD and quadratic near zero, so its σ is the larger change over
    ``ΔGDD ± σ(ΔGDD)``, upward only there.
    """
    if not np.any(intensity > 0):
        return None, None, None, None
    sigma_w = _rms_bandwidth(omega, intensity)
    tl = _tl_rms_duration(omega, intensity)

    def lengthening(gdd: float) -> float:
        return float(np.hypot(tl, gdd * sigma_w)) - tl

    delta = lengthening(fit.gdd)
    spread = max(
        abs(lengthening(fit.gdd + s * fit.gdd_err) - delta) for s in (-1.0, 1.0)
    )
    return sigma_w, tl, delta, spread


def _detection_floor(spectrum: _Spectrum, transform: _Transform, taper: float) -> float:
    """The delay-domain noise median a sideband candidate is judged against.

    Predicted from the pixel noise σ rather than read off the transform, so
    neither a sideband in the band read nor the resampling's attenuation of
    the far delays (which made a read floor several times low, letting noise
    peaks pass) can move it. Each pixel's noise enters the resampled sum
    about ``g`` times (``g`` = its spacing over the grid step) and scaled by
    the region taper, so ``E|F|² = σ² Σ (g·taper)²`` at the delays a
    sideband occupies; white noise's median is ``√(E|F|² ln 2)``.
    """
    pixels = spectrum.pixel_omega
    grid = spectrum.grid
    spacing = np.gradient(pixels) / (grid[1] - grid[0])
    shape = np.interp(pixels, grid, _tukey(grid.size, taper))
    power = spectrum.pixel_sigma**2 * float(np.sum((spacing * shape) ** 2))
    return float(np.sqrt(power * np.log(2.0)))


def analyze_interferogram(
    wavelength_nm: np.ndarray,
    counts: np.ndarray,
    config: Optional[SpectralInterferometryConfig] = None,
) -> SpectralInterferometryAnalysis:
    """Recover τ0 and ΔGDD from one spectral interferogram.

    Parameters
    ----------
    wavelength_nm, counts
        The calibrated spectrometer trace, any order; nonfinite samples are
        dropped.
    config
        Overrides and flag thresholds; defaults infer everything.

    Returns
    -------
    SpectralInterferometryAnalysis
        ``result`` (JSON-serialisable) and the ``traces`` the diagnostic
        figure draws. A sideband that cannot be found leaves the fit values
        ``None`` and raises ``sideband_not_found``; it never reports zero.

    Raises
    ------
    ValueError
        Fewer than 16 finite samples, or no signal at all.
    """
    cfg = config or SpectralInterferometryConfig()
    spectrum = _prepare(wavelength_nm, counts, cfg)
    transform = _Transform(spectrum.grid, spectrum.signal, cfg.pad_factor)
    noise = _detection_floor(spectrum, transform, cfg.taper) or float(
        np.finfo(float).tiny
    )
    # No spectrum (a dark or misfired shot): nothing to look for. Without
    # this, the tapered residual's low-delay structure passes as a sideband.
    dark = spectrum.signal_snr < cfg.min_signal_snr
    sb = None if dark else _locate_sideband(transform, cfg, noise)
    if sb is None:
        rec = _Recovery(flags=dict.fromkeys(FLAG_NAMES, False))
        rec.flags["sideband_not_found"] = True
    else:
        rec = _recover(spectrum, transform, sb, cfg, noise**2 / np.log(2.0))
    return SpectralInterferometryAnalysis(
        result=_result(spectrum, transform, sb, rec, cfg, noise),
        traces=_traces(spectrum, transform, rec),
    )


def _result(
    spectrum: _Spectrum,
    transform: _Transform,
    sb: Optional[_Sideband],
    rec: _Recovery,
    cfg: SpectralInterferometryConfig,
    noise: float,
) -> SpectralInterferometryResult:
    """Assemble the JSON result."""
    tpos, fit, band, grid = transform.positive_delay, rec.fit, rec.band, spectrum.grid
    central = False
    if band.size and rec.omega0 is not None:
        lo, hi = float(grid[band[0]]), float(grid[band[-1]])
        central = abs(rec.omega0 - 0.5 * (lo + hi)) <= 0.25 * (hi - lo)
    omega0 = rec.omega0
    if omega0 is None:
        omega0 = (
            float(wavelength_to_omega(cfg.reference_wavelength_nm))
            if cfg.reference_wavelength_nm is not None
            else 0.5 * (spectrum.region[0] + spectrum.region[1])
        )
    return SpectralInterferometryResult(
        tau0_fs=None if fit is None else fit.tau0,
        tau0_err_fs=None if fit is None else fit.tau0_err,
        delta_gdd_fs2=None if fit is None else fit.gdd,
        delta_gdd_err_fs2=None if fit is None else fit.gdd_err,
        phi0_rad=None if fit is None else fit.phi0,
        reference_wavelength_nm=float(omega_to_wavelength(omega0)),
        reference_omega_rad_per_fs=omega0,
        reference_source="user"
        if cfg.reference_wavelength_nm is not None
        else "band_centre",
        sideband=cfg.sideband,
        spectral_region_nm=_omega_range_to_nm(*spectrum.region),
        fit_range_nm=(
            _omega_range_to_nm(float(grid[band[0]]), float(grid[band[-1]]))
            if band.size
            else None
        ),
        fit_range_source="user" if cfg.fit_range_nm is not None else "auto",
        sideband_window_fs=None
        if sb is None
        else (float(tpos[sb.lo]), float(tpos[sb.hi])),
        window_source="user" if cfg.sideband_window_fs is not None else "auto",
        sideband_peak_fs=None if sb is None else float(tpos[sb.peak]),
        central_edge_fs=None if sb is None else float(tpos[sb.central]),
        signal_snr=spectrum.signal_snr,
        sideband_snr=(
            None if sb is None else float(transform.positive_magnitude[sb.peak] / noise)
        ),
        sideband_broadening=rec.broadening,
        delay_sweep_fs=rec.sweep,
        rms_bandwidth_rad_per_fs=rec.rms_bandwidth,
        tl_rms_fs=rec.tl_rms,
        delta_duration_rms_fs=rec.delta_rms,
        delta_duration_rms_err_fs=rec.delta_rms_err,
        window_tau_shift_fs=rec.tau_shift,
        window_gdd_shift_fs2=rec.gdd_shift,
        min_samples_per_fringe=rec.spf_min,
        min_resolution_visibility=rec.vis_min,
        rms_residual_rad=None if fit is None else fit.rms,
        n_fit_pixels=rec.n_fit_pixels,
        flags=SpectralInterferometryFlags(**rec.flags),
        reference_central=central,
        tau0_reliable=fit is not None
        and not any(rec.flags[k] for k in tau0_flags(central)),
        delta_gdd_reliable=fit is not None and not any(rec.flags.values()),
        reliable=fit is not None and not any(rec.flags.values()),
        messages=[_MESSAGES[name] for name in FLAG_NAMES if rec.flags[name]],
    )


def _traces(
    spectrum: _Spectrum, transform: _Transform, rec: _Recovery
) -> SpectralInterferometryTraces:
    """The arrays the diagnostic figure draws."""
    empty = np.zeros(0)
    band, fit = rec.band, rec.fit
    return SpectralInterferometryTraces(
        wavelength_nm=spectrum.wavelength_nm,
        counts=spectrum.counts,
        baseline=spectrum.baseline,
        omega=spectrum.grid,
        delay_fs=np.fft.fftshift(transform.delay),
        delay_magnitude=np.fft.fftshift(transform.magnitude),
        band_omega=spectrum.grid[band] if band.size else empty,
        band_magnitude=np.abs(rec.term[band])
        if band.size and rec.term is not None
        else empty,
        band_phase=fit.phase if fit is not None else empty,
        fitted_phase=fit.fitted if fit is not None else empty,
    )
