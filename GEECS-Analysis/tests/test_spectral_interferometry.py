"""Spectral interferometry on synthetic interferograms of two pulses.

Each spectrum is ``I1 + I2 + 2√(I1 I2) cos Φ(ω)`` with a known
``Φ = Φ0 + τ (ω − ωc) + ½ ΔGDD (ω − ωc)²``, sampled like the Thorlabs CCS
(3648 pixels, 497.7–1103.2 nm) on a 100-count background. Tolerances are
stated per test; numpy alone (no scipy).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from geecs_analysis.algorithms.spectral_interferometry import (
    C_NM_PER_FS,
    FLAG_NAMES,
    SpectralInterferometryConfig,
    SpectralInterferometryResult,
    analyze_interferogram,
    wavelength_to_omega,
)

CENTRE_NM = 800.0
OMEGA_C = float(wavelength_to_omega(CENTRE_NM))


def interferogram(
    tau_fs: float,
    gdd_fs2: float,
    *,
    ratio: float = 1.0,
    noise: float = 0.0,
    fwhm_nm: float = 40.0,
    phi0: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Counts over wavelength for two pulses, the second ``ratio`` times as intense."""
    wavelength = np.linspace(497.7, 1103.2, 3648)
    omega = wavelength_to_omega(wavelength)
    width = 2 * np.pi * C_NM_PER_FS * fwhm_nm / CENTRE_NM**2
    i1 = 1e4 * np.exp(-4 * np.log(2) * ((omega - OMEGA_C) / width) ** 2)
    i2 = ratio * i1
    phase = phi0 + tau_fs * (omega - OMEGA_C) + 0.5 * gdd_fs2 * (omega - OMEGA_C) ** 2
    counts = i1 + i2 + 2 * np.sqrt(i1 * i2) * np.cos(phase) + 100.0
    rng = np.random.default_rng(seed)
    return wavelength, counts + rng.normal(0.0, noise * counts.max(), counts.size)


def analyse(*args, config: SpectralInterferometryConfig | None = None, **kwargs):
    return analyze_interferogram(*interferogram(*args, **kwargs), config).result


def raised(result: SpectralInterferometryResult) -> set[str]:
    return {name for name, value in result.flags.model_dump().items() if value}


def expected_tau(
    tau_fs: float, gdd_fs2: float, result: SpectralInterferometryResult
) -> float:
    """The true delay at the result's reference wavelength."""
    return tau_fs + gdd_fs2 * (result.reference_omega_rad_per_fs - OMEGA_C)


class TestKnownValues:
    def test_recovers_tau_and_gdd_at_the_stated_reference(self):
        result = analyse(
            600.0,
            300.0,
            config=SpectralInterferometryConfig(reference_wavelength_nm=800.0),
        )
        assert result.reliable, result.messages
        assert result.reference_source == "user"
        assert result.reference_wavelength_nm == pytest.approx(800.0)
        assert result.tau0_fs == pytest.approx(600.0, abs=0.1)
        assert result.delta_gdd_fs2 == pytest.approx(300.0, rel=0.01)

    def test_default_reference_is_the_fit_band_centre_and_is_reported(self):
        result = analyse(600.0, 1000.0)
        assert result.reference_source == "band_centre"
        band = wavelength_to_omega(np.array(result.fit_range_nm))
        assert result.reference_omega_rad_per_fs == pytest.approx(band.mean(), rel=1e-3)
        assert result.tau0_fs == pytest.approx(
            expected_tau(600.0, 1000.0, result), abs=0.2
        )
        assert result.delta_gdd_fs2 == pytest.approx(1000.0, rel=0.01)

    @pytest.mark.parametrize(
        ("tau", "gdd"), [(250.0, 0.0), (1200.0, -800.0), (3000.0, 1000.0)]
    )
    def test_other_delays_and_signs(self, tau, gdd):
        result = analyse(tau, gdd)
        assert result.reliable, result.messages
        assert result.tau0_fs == pytest.approx(expected_tau(tau, gdd, result), abs=0.2)
        assert result.delta_gdd_fs2 == pytest.approx(gdd, abs=max(5.0, 0.01 * abs(gdd)))

    def test_unequal_pulse_strengths(self):
        result = analyse(600.0, 300.0, ratio=0.05)
        assert result.reliable, result.messages
        assert result.tau0_fs == pytest.approx(
            expected_tau(600.0, 300.0, result), abs=0.2
        )
        assert result.delta_gdd_fs2 == pytest.approx(300.0, rel=0.02)

    def test_noise_stays_within_the_reported_uncertainty(self):
        pulls_tau, pulls_gdd = [], []
        for seed in range(12):
            result = analyse(600.0, 300.0, ratio=0.3, noise=0.02, seed=seed)
            assert result.reliable, result.messages
            pulls_tau.append(
                (result.tau0_fs - expected_tau(600.0, 300.0, result))
                / result.tau0_err_fs
            )
            pulls_gdd.append((result.delta_gdd_fs2 - 300.0) / result.delta_gdd_err_fs2)
            assert abs(result.tau0_fs - expected_tau(600.0, 300.0, result)) < 1.0
            assert abs(result.delta_gdd_fs2 - 300.0) < 30.0
        # The σ are calibrated (the window-correlated noise sandwich): the
        # pulls' spread is near one, neither tiny nor several.
        assert 0.5 < np.std(pulls_tau) < 2.0
        assert 0.5 < np.std(pulls_gdd) < 2.0

    def test_the_wavelength_order_does_not_matter(self):
        wavelength, counts = interferogram(600.0, 300.0)
        forward = analyze_interferogram(wavelength, counts).result
        backward = analyze_interferogram(wavelength[::-1], counts[::-1]).result
        assert backward.tau0_fs == pytest.approx(forward.tau0_fs, abs=1e-6)
        assert backward.delta_gdd_fs2 == pytest.approx(forward.delta_gdd_fs2, abs=1e-4)


class TestSignAmbiguity:
    def test_swapping_the_arms_gives_the_same_spectrum_and_answer(self):
        wl, a = interferogram(600.0, 300.0, phi0=0.3)
        _, b = interferogram(-600.0, -300.0, phi0=-0.3)
        np.testing.assert_allclose(a, b)
        result = analyze_interferogram(wl, b).result
        assert result.tau0_fs > 0  # the later pulse's delay, whichever arm it is
        assert result.delta_gdd_fs2 == pytest.approx(300.0, rel=0.01)
        assert "cannot tell the arms apart" in result.sign_convention

    def test_the_conjugate_sideband_returns_the_mirror_solution(self):
        config = SpectralInterferometryConfig(reference_wavelength_nm=800.0)
        positive = analyse(600.0, 300.0, config=config)
        negative = analyse(
            600.0, 300.0, config=config.model_copy(update={"sideband": "negative"})
        )
        assert negative.sideband == "negative"
        assert negative.tau0_fs == pytest.approx(-positive.tau0_fs, abs=1e-6)
        assert negative.delta_gdd_fs2 == pytest.approx(
            -positive.delta_gdd_fs2, abs=1e-4
        )
        assert negative.tau0_err_fs == pytest.approx(positive.tau0_err_fs)


class TestUnreliable:
    @pytest.mark.parametrize("tau", [0.0, 40.0])
    def test_a_merged_peak_reports_no_delay_never_zero(self, tau):
        result = analyse(tau, 0.0)
        assert not result.reliable
        assert result.flags.sideband_not_found
        assert result.tau0_fs is None and result.delta_gdd_fs2 is None
        assert "not evidence of zero delay" in result.messages[0]
        assert json.loads(result.model_dump_json())["tau0_fs"] is None

    @pytest.mark.parametrize(
        "wavelength",
        [
            np.linspace(497.7, 1103.2, 3648),
            np.linspace(750.0, 850.0, 2048),
            np.linspace(700.0, 900.0, 2048),
        ],
    )
    def test_a_dark_shot_reports_no_delay(self, wavelength):
        # A misfire: background and noise only. The tapered residual's
        # low-delay structure must not pass as a few-fs sideband.
        for seed in range(10):
            counts = 100.0 + np.random.default_rng(seed).normal(
                0.0, 3.0, wavelength.size
            )
            result = analyze_interferogram(wavelength, counts).result
            assert result.flags.sideband_not_found, (seed, result.tau0_fs)
            assert result.tau0_fs is None and not result.reliable

    def test_a_near_overlap_is_flagged(self):
        result = analyse(100.0, 0.0)
        assert not result.reliable
        assert result.flags.sideband_overlaps_central
        assert result.tau0_fs != 0.0

    def test_strong_chirp_broadening_is_flagged(self):
        result = analyse(1500.0, 8000.0)
        assert result.flags.sideband_broadened
        assert not result.reliable
        assert result.sideband_broadening > 3.0
        assert result.delay_sweep_fs > 1000.0
        # Still recovered here: the flag says isolation is doubtful, not wrong.
        assert result.delta_gdd_fs2 == pytest.approx(8000.0, rel=0.01)

    def test_moderate_chirp_is_not_flagged(self):
        result = analyse(600.0, -1000.0)
        assert not result.flags.sideband_broadened

    def test_fringes_too_fine_for_the_pixels(self):
        result = analyse(5500.0, 0.0)
        assert result.flags.fringes_undersampled
        assert result.min_samples_per_fringe < 3.0

    def test_fringes_too_fine_for_the_optical_resolution(self):
        coarse = analyse(
            1500.0, 0.0, config=SpectralInterferometryConfig(resolution_nm=2.0)
        )
        fine = analyse(
            1500.0, 0.0, config=SpectralInterferometryConfig(resolution_nm=0.2)
        )
        assert (
            coarse.flags.fringes_unresolved and coarse.min_resolution_visibility < 0.5
        )
        assert (
            not fine.flags.fringes_unresolved and fine.min_resolution_visibility > 0.9
        )
        assert analyse(1500.0, 0.0).min_resolution_visibility is None

    def test_narrow_bandwidth_is_flagged(self):
        result = analyse(3000.0, 0.0, fwhm_nm=3.0, noise=0.01)
        assert result.flags.narrow_bandwidth
        assert not result.reliable

    def test_a_partially_overlapping_sideband_is_window_sensitive(self):
        result = analyse(100.0, 0.0)
        assert result.flags.window_sensitive
        assert result.window_tau_shift_fs > 1.0 or result.window_gdd_shift_fs2 > 20.0
        assert analyse(600.0, 300.0).window_tau_shift_fs < 0.1


class TestOverridesAndResult:
    def test_manual_window_and_fit_range(self):
        config = SpectralInterferometryConfig(
            sideband_window_fs=(450.0, 750.0), fit_range_nm=(780.0, 820.0)
        )
        result = analyse(600.0, 300.0, config=config)
        assert result.window_source == "user" and result.fit_range_source == "user"
        assert result.sideband_window_fs == pytest.approx((450.0, 750.0), abs=2.0)
        assert result.fit_range_nm == pytest.approx((780.0, 820.0), abs=0.1)
        assert result.tau0_fs == pytest.approx(
            expected_tau(600.0, 300.0, result), abs=0.5
        )
        assert result.delta_gdd_fs2 == pytest.approx(300.0, rel=0.02)

    def test_a_user_fit_range_into_the_dark_wings_is_held_to_the_fringes(self):
        # 700–900 nm reaches far past a 40 nm spectrum: the phase there is
        # noise and unwrapped badly, biasing ΔGDD by several σ (review #1364).
        config = SpectralInterferometryConfig(fit_range_nm=(700.0, 900.0))
        pulls = []
        for seed in range(12):
            result = analyse(600.0, 300.0, noise=0.01, seed=seed, config=config)
            assert result.fit_range_source == "user"
            assert 740.0 < result.fit_range_nm[0] and result.fit_range_nm[1] < 870.0
            pulls.append((result.delta_gdd_fs2 - 300.0) / result.delta_gdd_err_fs2)
        assert abs(np.mean(pulls)) < 1.0 and np.std(pulls) < 2.0

    def test_a_manual_window_on_the_negative_side_is_mirrored(self):
        config = SpectralInterferometryConfig(sideband_window_fs=(-750.0, -450.0))
        assert analyse(600.0, 300.0, config=config).sideband_window_fs == pytest.approx(
            (450.0, 750.0), abs=2.0
        )

    def test_the_result_round_trips_through_json(self):
        result = analyse(600.0, 300.0)
        again = SpectralInterferometryResult.model_validate_json(
            result.model_dump_json()
        )
        assert again == result
        assert set(json.loads(result.model_dump_json())["flags"]) == set(FLAG_NAMES)

    def test_every_raised_flag_has_a_message(self):
        result = analyse(100.0, 0.0)
        assert len(result.messages) == len(raised(result))

    def test_too_few_samples_raise(self):
        with pytest.raises(ValueError):
            analyze_interferogram(np.arange(10.0) + 700, np.ones(10))


class TestFigure:
    @pytest.mark.parametrize(("tau", "gdd"), [(600.0, 300.0), (0.0, 0.0)])
    def test_three_panels_and_residuals_render(self, tau, gdd):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        from geecs_analysis.render.spectral_interferometry import diagnostic_figure

        figure = diagnostic_figure(
            analyze_interferogram(*interferogram(tau, gdd)), title="synthetic"
        )
        FigureCanvasAgg(figure).draw()
        titles = [ax.get_title() for ax in figure.axes]
        assert "Measured spectrum" in titles and "Delay domain" in titles
        assert any("τ0" in title for title in titles)
        assert ("UNRELIABLE" in figure._suptitle.get_text()) == (tau == 0.0)


class TestMeasure:
    def _frame(self, tau, gdd):
        from geecs_data_utils.frames import Axis, Frame

        wavelength, counts = interferogram(tau, gdd)
        return Frame.from_array(counts, axes=(Axis(wavelength, unit="nm"),))

    def test_the_registered_measure_emits_its_scalars(self):
        from geecs_analysis.run import analyze
        from geecs_analysis.specs import Analysis

        analysis = Analysis.model_validate(
            {
                "steps": [],
                "measure": {
                    "kind": "spectral_interferometry",
                    "reference_wavelength_nm": 800.0,
                },
            }
        )
        result = analyze(self._frame(600.0, 300.0), analysis)
        assert set(result.scalars) == analysis.measure.emitted_scalars()
        assert result.scalars["tau0_fs"] == pytest.approx(600.0, abs=0.1)
        assert result.scalars["delta_gdd_fs2"] == pytest.approx(300.0, rel=0.01)
        assert result.scalars["reliable"] == 1.0
        assert set(result.extras) == {"interference_phase", "phase_residual"}

    def test_undetermined_values_are_nan_with_notes(self):
        from geecs_analysis.run import analyze
        from geecs_analysis.specs import Analysis

        analysis = Analysis.model_validate(
            {"steps": [], "measure": {"kind": "spectral_interferometry"}}
        )
        result = analyze(self._frame(0.0, 0.0), analysis)
        assert math.isnan(result.scalars["tau0_fs"])
        assert result.scalars["reliable"] == 0.0
        assert any("not evidence of zero delay" in note for note in result.notes)
