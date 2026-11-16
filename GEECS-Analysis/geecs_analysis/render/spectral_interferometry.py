"""The spectral-interferometry diagnostic figure: spectrum, delay domain, phase.

Three panels from :class:`~geecs_analysis.algorithms.spectral_interferometry.SpectralInterferometryAnalysis`:
the measured spectrum with the fit band and reference wavelength; the
delay-domain magnitude with the central peak, both conjugate sidebands
and the chosen window; and the recovered phase with its quadratic fit
(the linear part removed so the curvature shows) over the residuals.
Returns a matplotlib ``Figure``; no pyplot, no file writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from geecs_analysis.algorithms.spectral_interferometry import (
        SpectralInterferometryAnalysis,
        SpectralInterferometryResult,
        SpectralInterferometryTraces,
    )


def _fmt(value: float | None, err: float | None, unit: str) -> str:
    if value is None:
        return "not determined"
    if err is None:
        return f"{value:.4g} {unit}"
    return f"{value:.4g} ± {err:.2g} {unit}"


def _spectrum_panel(
    ax: Axes, result: SpectralInterferometryResult, tr: SpectralInterferometryTraces
) -> None:
    """The measured spectrum, its background, the fit band and the reference."""
    import numpy as np

    order = np.argsort(tr.wavelength_nm)
    ax.plot(tr.wavelength_nm[order], tr.counts[order], lw=0.8, color="C0")
    ax.axhline(tr.baseline, color="0.6", lw=0.6, ls=":")
    low, high = result.spectral_region_nm
    ax.set_xlim(low - 0.1 * (high - low), high + 0.1 * (high - low))
    if result.fit_range_nm is not None:
        ax.axvspan(*result.fit_range_nm, color="C2", alpha=0.15, label="fit band")
    ax.axvline(
        result.reference_wavelength_nm,
        color="C3",
        lw=0.8,
        ls="--",
        label=f"reference {result.reference_wavelength_nm:.2f} nm",
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Counts")
    ax.set_title("Measured spectrum")
    ax.legend(loc="upper right", fontsize="small")


def _delay_panel(
    ax: Axes, result: SpectralInterferometryResult, tr: SpectralInterferometryTraces
) -> None:
    """|FFT| over delay: central peak, both sidebands, the chosen window."""
    import numpy as np

    mag = tr.delay_magnitude
    floor = max(float(np.median(mag)) * 0.3, float(mag.max()) * 1e-7)
    ax.semilogy(tr.delay_fs, np.maximum(mag, floor), lw=0.8, color="C0")
    reach = 4.0 * (result.central_edge_fs or 0.0)
    if result.sideband_window_fs is not None:
        lo, hi = result.sideband_window_fs
        s = 1.0 if result.sideband == "positive" else -1.0
        ax.axvspan(s * lo, s * hi, color="C2", alpha=0.2, label="chosen window")
        ax.axvspan(-s * hi, -s * lo, color="0.5", alpha=0.08, label="conjugate")
        reach = max(reach, 1.5 * hi)
    if result.sideband_peak_fs is not None:
        for sign, color in ((1, "C2"), (-1, "0.5")):
            ax.axvline(sign * result.sideband_peak_fs, color=color, lw=0.6, ls="--")
    ax.axvline(0.0, color="C3", lw=0.6, ls=":", label="central peak")
    if result.central_edge_fs is not None:
        ax.axvspan(
            -result.central_edge_fs, result.central_edge_fs, color="C3", alpha=0.08
        )
    if reach > 0:
        ax.set_xlim(-reach, reach)
    ax.set_ylim(bottom=floor)
    ax.set_xlabel("Delay (fs)")
    ax.set_ylabel("|FFT| (arb.)")
    ax.set_title("Delay domain")
    ax.legend(loc="upper right", fontsize="small")


def _phase_panels(
    ax: Axes,
    ax_res: Axes,
    result: SpectralInterferometryResult,
    tr: SpectralInterferometryTraces,
) -> None:
    """The phase with Φ0 + τ0Δω removed, its quadratic fit, and the residuals."""
    import numpy as np

    from geecs_analysis.algorithms.spectral_interferometry import omega_to_wavelength

    ax.set_ylabel("Phase (rad)")
    ax.set_title(
        f"τ0 = {_fmt(result.tau0_fs, result.tau0_err_fs, 'fs')} "
        f"@ {result.reference_wavelength_nm:.2f} nm\n"
        f"ΔGDD = {_fmt(result.delta_gdd_fs2, result.delta_gdd_err_fs2, 'fs²')}",
        fontsize="medium",
    )
    ax_res.set_xlabel("ω (rad/fs)")
    ax_res.set_ylabel("Residual (rad)")
    if not tr.band_phase.size or result.delta_gdd_fs2 is None:
        ax.text(
            0.5,
            0.5,
            "no isolated sideband:\nno phase recovered",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    x = tr.band_omega - result.reference_omega_rad_per_fs
    # Φ0 + τ0 Δω, taken from the fit itself (Φ0 is unwrapped there).
    linear = tr.fitted_phase - 0.5 * result.delta_gdd_fs2 * x**2
    ax.scatter(
        tr.band_omega,
        tr.band_phase - linear,
        s=4,
        c=tr.band_magnitude / tr.band_magnitude.max(),
        cmap="Greys",
        vmin=-0.3,
        vmax=1.0,
        label="measured − (Φ0 + τ0 Δω)",
    )
    ax.plot(
        tr.band_omega, tr.fitted_phase - linear, color="C3", lw=1.2, label="½ ΔGDD Δω²"
    )
    ax.axvline(result.reference_omega_rad_per_fs, color="C3", lw=0.6, ls="--")
    ax.legend(loc="best", fontsize="small")
    ax_res.plot(tr.band_omega, tr.band_phase - tr.fitted_phase, color="C0", lw=0.8)
    ax_res.axhline(0.0, color="0.5", lw=0.6)

    def to_nm(w):
        return omega_to_wavelength(np.maximum(w, 1e-6))

    ax.secondary_xaxis("top", functions=(to_nm, to_nm)).set_xlabel("Wavelength (nm)")


def diagnostic_figure(
    analysis: SpectralInterferometryAnalysis, title: str = ""
) -> Figure:
    """Draw the three-panel diagnostic for one analysed interferogram.

    The heading says whether the result is reliable and names every raised
    flag; it is red when any is.
    """
    from matplotlib.figure import Figure

    result, tr = analysis.result, analysis.traces
    fig = Figure(figsize=(15, 5.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(3, 1))
    phase = fig.add_subplot(grid[0, 2])
    _spectrum_panel(fig.add_subplot(grid[:, 0]), result, tr)
    _delay_panel(fig.add_subplot(grid[:, 1]), result, tr)
    _phase_panels(phase, fig.add_subplot(grid[1, 2], sharex=phase), result, tr)
    raised = [name for name, value in result.flags.model_dump().items() if value]
    status = "reliable" if result.reliable else "UNRELIABLE: " + ", ".join(raised)
    fig.suptitle(
        f"{title}  —  {status}" if title else status,
        color="black" if result.reliable else "C3",
        fontsize="medium",
    )
    return fig
