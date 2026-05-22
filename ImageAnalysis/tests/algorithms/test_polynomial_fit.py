"""Tests for reusable polynomial fitting helpers."""

import numpy as np
import pytest

from image_analysis.algorithms.polynomial_fit import (
    PolynomialFitConfig,
    canonicalize_polynomial_sign,
    fit_polynomial,
)


class TestPolynomialFit:
    """Polynomial fit behavior for common 1D inputs."""

    def test_y_only_uses_sample_index(self):
        x = np.arange(20, dtype=float)
        y = 2.0 * x**2 + 3.0 * x + 1.0

        result = fit_polynomial(y, config=PolynomialFitConfig(order=2))

        coeffs = result.coefficients_by_order
        assert coeffs[0] == pytest.approx(1.0)
        assert coeffs[1] == pytest.approx(3.0)
        assert coeffs[2] == pytest.approx(2.0)

    def test_x_y_fit_recovers_coefficients(self):
        x = np.linspace(-2.0, 2.0, 50)
        y = -0.5 * x**3 + 2.0 * x - 4.0

        result = fit_polynomial(y, x=x, config=PolynomialFitConfig(order=3))

        coeffs = result.coefficients_by_order
        assert coeffs[0] == pytest.approx(-4.0)
        assert coeffs[1] == pytest.approx(2.0)
        assert coeffs[2] == pytest.approx(0.0, abs=1e-12)
        assert coeffs[3] == pytest.approx(-0.5)

    def test_weight_threshold_masks_outliers(self):
        x = np.linspace(-5.0, 5.0, 101)
        weights = np.exp(-(x**2) / 2.0)
        y = 1.5 * x + 2.0
        y[weights < 0.2] += 100.0

        result = fit_polynomial(
            y,
            x=x,
            weights=weights,
            config=PolynomialFitConfig(
                order=1,
                mask_threshold=0.2,
            ),
        )

        coeffs = result.coefficients_by_order
        assert coeffs[0] == pytest.approx(2.0)
        assert coeffs[1] == pytest.approx(1.5)
        assert result.num_fit_points < x.size

    def test_nan_padded_rows_are_ignored(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, np.nan, np.nan])
        y = np.array([1.0, 3.0, 5.0, 7.0, np.nan, np.nan])
        weights = np.array([1.0, 1.0, 1.0, 1.0, np.nan, np.nan])

        result = fit_polynomial(
            y,
            x=x,
            weights=weights,
            config=PolynomialFitConfig(order=1),
        )

        coeffs = result.coefficients_by_order
        assert coeffs[0] == pytest.approx(1.0)
        assert coeffs[1] == pytest.approx(2.0)
        assert result.num_fit_points == 4

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="requires at least"):
            fit_polynomial(
                np.array([1.0, np.nan]),
                x=np.array([0.0, np.nan]),
                config=PolynomialFitConfig(order=1),
            )


class TestPolynomialSignCanonicalization:
    """Sign convention helper tests."""

    def test_flips_to_reference_sign(self):
        coeffs = np.array([-2.0, 0.0, 1.0])

        flipped, did_flip = canonicalize_polynomial_sign(
            coeffs,
            reference_order=2,
            reference_sign=1.0,
        )

        assert did_flip
        assert flipped[0] == pytest.approx(2.0)

    def test_keeps_matching_reference_sign(self):
        coeffs = np.array([2.0, 0.0, 1.0])

        kept, did_flip = canonicalize_polynomial_sign(
            coeffs,
            reference_order=2,
            reference_sign=1.0,
        )

        assert not did_flip
        assert kept[0] == pytest.approx(2.0)
