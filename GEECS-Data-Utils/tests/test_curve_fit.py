"""Curve fits on synthetic data, the expected numbers worked by hand."""

import math

import numpy as np
import pytest

from geecs_data_utils.data.curve_fit import (
    FWHM_PER_SIGMA,
    _initial_guess,
    fit_curve,
    gaussian,
    supergaussian,
)

X = np.linspace(-2.0, 4.0, 61)


def params(fit):
    return {p.name: p.value for p in fit.parameters}


class TestPolynomial:
    def test_a_line_through_three_points(self):
        # y = 0, 1, 1 at x = 0, 1, 2: slope 1/2, intercept 1/6; residuals
        # -1/6, 1/3, -1/6 give s² = (1/6)/1, Sxx = 2, so
        # se(slope) = sqrt(s²/Sxx) = sqrt(1/12),
        # se(intercept) = sqrt(s²(1/n + x̄²/Sxx)) = sqrt(5/36), r2 = 1 - (1/6)/(2/3).
        fit = fit_curve([0, 1, 2], [0, 1, 1], "polynomial", order=1)
        assert fit.ok and fit.order == 1 and fit.points == 3
        c0, c1 = fit.parameters
        assert (c0.name, c1.name) == ("c0", "c1")
        assert c0.value == pytest.approx(1 / 6)
        assert c1.value == pytest.approx(0.5)
        assert c0.stderr == pytest.approx(math.sqrt(5 / 36))
        assert c1.stderr == pytest.approx(math.sqrt(1 / 12))
        assert fit.r2 == pytest.approx(0.75)

    def test_a_quadratic_is_recovered_by_ascending_power(self):
        x = np.arange(6.0)
        fit = fit_curve(x, 1 - 2 * x + 0.5 * x**2, "polynomial", order=2)
        assert [p.value for p in fit.parameters] == pytest.approx([1, -2, 0.5])
        assert fit.r2 == pytest.approx(1.0)
        assert fit.evaluate([10.0]) == pytest.approx([31.0])

    def test_exactly_enough_points_is_exact_without_errors(self):
        fit = fit_curve([0, 1, 2], [1, 3, 7], "polynomial", order=2)
        assert [p.value for p in fit.parameters] == pytest.approx([1, 1, 1])
        assert all(math.isnan(p.stderr) for p in fit.parameters)
        assert any("exact" in note for note in fit.notes)

    def test_too_few_distinct_x_is_an_error_not_an_exception(self):
        fit = fit_curve([1, 1, 2, 2], [0, 1, 2, 3], "polynomial", order=2)
        assert not fit.ok and fit.parameters == []
        assert "3 distinct x" in fit.error and "has 2" in fit.error
        with pytest.raises(ValueError, match="no fit"):
            fit.evaluate([0.0])

    def test_nonfinite_pairs_are_dropped(self):
        fit = fit_curve(
            [0, 1, np.nan, 2, 3], [0, 1, 5, 1, np.inf], "polynomial", order=1
        )
        assert fit.points == 3
        assert params(fit)["c1"] == pytest.approx(0.5)
        assert (fit.x_min, fit.x_max) == (0.0, 2.0)

    def test_a_negative_order_is_refused(self):
        with pytest.raises(ValueError, match="order"):
            fit_curve([0, 1], [0, 1], "polynomial", order=-1)


class TestGaussian:
    def test_a_peak_on_a_baseline(self):
        fit = fit_curve(X, gaussian(X, 3.0, 1.5, 0.4, 0.5), "gaussian")
        assert fit.ok and fit.order is None
        assert params(fit) == pytest.approx(
            {"amplitude": 3.0, "center": 1.5, "sigma": 0.4, "offset": 0.5}
        )
        assert fit.value("fwhm") == pytest.approx(0.4 * 2 * math.sqrt(2 * math.log(2)))
        assert fit.r2 == pytest.approx(1.0)

    def test_a_dip_below_a_baseline(self):
        fit = fit_curve(X, gaussian(X, -2.0, 0.0, 0.7, 10.0), "gaussian")
        assert params(fit) == pytest.approx(
            {"amplitude": -2.0, "center": 0.0, "sigma": 0.7, "offset": 10.0}
        )

    def test_no_offset(self):
        fit = fit_curve(X, gaussian(X, 5.0, -0.5, 0.3), "gaussian_no_offset")
        assert params(fit) == pytest.approx(
            {"amplitude": 5.0, "center": -0.5, "sigma": 0.3}
        )

    def test_errors_are_the_residual_scaled_covariance(self):
        # Independent of curve_fit: cov = s²·(JᵀJ)⁻¹ with J the analytic
        # Jacobian at the fitted point and s² = SSR / (n - 4).
        rng = np.random.default_rng(7)
        y = gaussian(X, 3.0, 1.5, 0.4, 0.5) + rng.normal(0, 0.05, X.size)
        fit = fit_curve(X, y, "gaussian")
        a, c, s, _ = (p.value for p in fit.parameters)
        g = np.exp(-((X - c) ** 2) / (2 * s**2))
        jac = np.column_stack(
            [g, a * g * (X - c) / s**2, a * g * (X - c) ** 2 / s**3, np.ones_like(X)]
        )
        residual = y - fit.evaluate(X)
        s2 = residual @ residual / (X.size - 4)
        cov = s2 * np.linalg.inv(jac.T @ jac)
        assert [p.stderr for p in fit.parameters] == pytest.approx(
            np.sqrt(np.diag(cov)), rel=1e-4
        )
        assert fit.derived[0].stderr == pytest.approx(
            FWHM_PER_SIGMA * math.sqrt(cov[2, 2]), rel=1e-4
        )

    def test_repeated_shots_per_position(self):
        # A scan: several shots per step, the fit over every shot.
        x = np.repeat(np.linspace(0, 10, 11), 5)
        rng = np.random.default_rng(1)
        y = gaussian(x, 100.0, 4.0, 1.5, 20.0) + rng.normal(0, 2.0, x.size)
        fit = fit_curve(x, y, "gaussian")
        assert fit.points == 55
        assert params(fit)["center"] == pytest.approx(4.0, abs=0.1)
        assert params(fit)["sigma"] == pytest.approx(1.5, abs=0.1)

    def test_stray_low_shots_do_not_turn_a_peak_into_a_dip(self):
        # A focus scan's shape (2026-10-26 Scan006): a broad peak that never
        # reaches its baseline, most shots near the top, a few misfires far
        # below. Against the overall median the misfires were the bigger
        # excursion, so the first guess was a dip at the scan's edge
        # (amplitude -1523 at x = 15 here), from which the real scan's fit
        # converged onto that dip; the guess is pinned, then the fit.
        x = np.repeat(np.linspace(15, 45, 13), 5)
        rng = np.random.default_rng(0)
        y = gaussian(x, 1700.0, 23.0, 13.0) + rng.normal(0, 100, x.size)
        y[[3, 21, 40, 52]] = 280.0
        for model in ("gaussian", "supergaussian"):
            amplitude, center, *_ = _initial_guess(model, x, y)
            assert amplitude > 0 and center == pytest.approx(23.0, abs=2.5), model
            fit = fit_curve(x, y, model)
            assert params(fit)["amplitude"] > 0, model
            assert params(fit)["center"] == pytest.approx(23.0, abs=1.5), model

    def test_too_few_distinct_x(self):
        fit = fit_curve([0, 1, 2], [0, 1, 0], "gaussian")
        assert not fit.ok and "4 distinct x" in fit.error

    def test_a_spent_budget_is_an_error(self):
        fit = fit_curve(
            X, gaussian(X, 3.0, 1.5, 0.4, 0.5), "gaussian", max_evaluations=1
        )
        assert not fit.ok and "did not converge" in fit.error


class TestSuperGaussian:
    def test_a_flat_top(self):
        fit = fit_curve(X, supergaussian(X, 2.0, 1.0, 1.2, 4.0, 0.3), "supergaussian")
        assert params(fit) == pytest.approx(
            {
                "amplitude": 2.0,
                "center": 1.0,
                "width": 1.2,
                "power": 4.0,
                "offset": 0.3,
            },
            rel=1e-5,
        )
        # Half height where (|x - x0|/w)^p = ln 2: FWHM = 2w·(ln 2)^(1/p).
        assert fit.value("fwhm") == pytest.approx(2 * 1.2 * math.log(2) ** 0.25)

    def test_power_two_is_the_gaussian(self):
        y = gaussian(X, 3.0, 1.5, 0.4, 0.5)
        fit = fit_curve(X, y, "supergaussian")
        assert params(fit)["power"] == pytest.approx(2.0, rel=1e-5)
        assert params(fit)["width"] == pytest.approx(0.4 * math.sqrt(2), rel=1e-5)
        assert fit.value("fwhm") == pytest.approx(FWHM_PER_SIGMA * 0.4, rel=1e-5)

    def test_no_offset(self):
        y = supergaussian(X, 4.0, 0.5, 0.8, 6.0)
        fit = fit_curve(X, y, "supergaussian_no_offset")
        assert [p.name for p in fit.parameters] == [
            "amplitude",
            "center",
            "width",
            "power",
        ]
        assert params(fit)["power"] == pytest.approx(6.0, rel=1e-4)

    def test_fwhm_error_propagates_width_and_power(self):
        rng = np.random.default_rng(3)
        y = supergaussian(X, 2.0, 1.0, 1.2, 4.0, 0.3) + rng.normal(0, 0.02, X.size)
        fit = fit_curve(X, y, "supergaussian")
        stderr = {p.name: p.stderr for p in fit.parameters}
        assert 0 < fit.derived[0].stderr < 2 * (stderr["width"] + stderr["power"])


def test_the_curve_spans_the_fitted_range():
    fit = fit_curve(X, gaussian(X, 1.0, 0.0, 1.0, 0.0), "gaussian")
    xs, ys = fit.curve(points=5)
    assert xs == pytest.approx([-2.0, -0.5, 1.0, 2.5, 4.0])
    assert ys == pytest.approx(gaussian(xs, 1.0, 0.0, 1.0, 0.0), abs=1e-9)


def test_an_undefined_error_is_json_null():
    fit = fit_curve([0, 1, 2], [1, 3, 7], "polynomial", order=2)
    assert '"stderr":null' in fit.model_dump_json()


class TestFarFromZero:
    """A timestamp-like X (review of #1174): solved about the range's middle."""

    T = 1.7e9 + np.arange(30.0)

    def test_a_polynomial_is_fitted_about_the_middle(self):
        u = self.T - self.T.mean()
        fit = fit_curve(self.T, 2 + 0.5 * u + 0.01 * u**2, "polynomial", order=2)
        assert fit.ok and fit.x_shift == pytest.approx(1.7e9 + 14.5)
        assert [p.value for p in fit.parameters] == pytest.approx([2, 0.5, 0.01])
        assert fit.evaluate([self.T[3]]) == pytest.approx(
            [2 + 0.5 * u[3] + 0.01 * u[3] ** 2]
        )

    def test_an_ordinary_range_keeps_plain_powers_of_x(self):
        x = np.linspace(15, 45, 13)
        assert fit_curve(x, x**2, "polynomial", order=2).x_shift == 0.0

    def test_a_gaussian_far_from_zero_keeps_its_width(self):
        x = 1e9 + X
        fit = fit_curve(x, gaussian(X, 3.0, 1.5, 0.4, 0.5), "gaussian")
        assert params(fit)["center"] == pytest.approx(1e9 + 1.5, abs=1e-6)
        assert params(fit)["sigma"] == pytest.approx(0.4, rel=1e-6)
        assert fit.r2 == pytest.approx(1.0)
        assert fit.parameters[1].stderr < 1e-3


def test_flat_data_is_no_peak():
    fit = fit_curve(X, np.ones_like(X), "gaussian")
    assert not fit.ok and "do not vary" in fit.error
    assert params(
        fit_curve(X, np.ones_like(X), "polynomial", order=1)
    ) == pytest.approx({"c0": 1.0, "c1": 0.0}, abs=1e-12)


def test_a_power_on_its_bound_is_noted():
    # A top hat: the steepest edge the power can make, which is its bound.
    fit = fit_curve(X, np.where(np.abs(X - 1) < 1, 1.0, 0.0), "supergaussian")
    assert any("power sits on its bound" in note for note in fit.notes)


@pytest.mark.parametrize(
    "fit",
    [
        fit_curve([0, 1, 2], [1, 3, 7], "polynomial", order=2),  # NaN errors
        fit_curve([0, 0], [1, 2], "gaussian"),  # an error: NaN r2 and range
    ],
    ids=["exact", "error"],
)
def test_a_result_survives_its_json_round_trip(fit):
    again = type(fit).model_validate_json(fit.model_dump_json())
    assert again.model_dump_json() == fit.model_dump_json()
