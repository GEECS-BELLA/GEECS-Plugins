"""Unit tests for MagSpecManualCalibAnalyzer energy calibration classes.

Tests PolynomialCalibration and ArrayCalibration in isolation — no YAML
config files or real camera images required.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from image_analysis.analyzers.magspec_manual_calib_analyzer import (
    ArrayCalibration,
    MagSpecAnalyzerConfig,
    MagSpecManualCalibAnalyzer,
    PolynomialCalibration,
)
from image_analysis.types import ImageAnalyzerResult


class TestPolynomialCalibration:
    """Tests for PolynomialCalibration.build_axis()."""

    def test_constant_polynomial_returns_constant_axis(self):
        """Single coefficient → flat energy axis."""
        cal = PolynomialCalibration(coeffs=[100.0])
        axis = cal.build_axis(image_width=50)
        assert axis.shape == (50,)
        assert np.allclose(axis, 100.0)

    def test_linear_polynomial_increases_monotonically(self):
        """Positive linear coefficient → monotonically increasing axis."""
        cal = PolynomialCalibration(coeffs=[0.0, 1.0])
        axis = cal.build_axis(image_width=100)
        assert axis[0] == pytest.approx(0.0)
        assert axis[-1] == pytest.approx(99.0)
        assert np.all(np.diff(axis) > 0)

    def test_quadratic_polynomial_shape(self):
        """Two-term quadratic: E(x) = 50 + 0.1*x + 0.001*x^2."""
        coeffs = [50.0, 0.1, 0.001]
        cal = PolynomialCalibration(coeffs=coeffs)
        axis = cal.build_axis(image_width=10)
        expected = np.array([50.0 + 0.1 * x + 0.001 * x**2 for x in range(10)])
        assert np.allclose(axis, expected)

    def test_axis_length_matches_image_width(self):
        cal = PolynomialCalibration(coeffs=[1.0, 0.5])
        for width in [64, 128, 512]:
            assert cal.build_axis(image_width=width).shape == (width,)

    def test_empty_coeffs_raises(self):
        with pytest.raises(Exception):
            PolynomialCalibration(coeffs=[])

    def test_kind_is_polynomial(self):
        cal = PolynomialCalibration(coeffs=[1.0])
        assert cal.kind == "polynomial"

    def test_get_charge_factor_returns_none(self):
        cal = PolynomialCalibration(coeffs=[1.0])
        assert cal.get_charge_factor() is None


class TestArrayCalibration:
    """Tests for ArrayCalibration.build_axis()."""

    def test_inline_values_returned_as_axis(self):
        values = list(range(50))
        cal = ArrayCalibration(values=values)
        axis = cal.build_axis(image_width=50)
        assert axis.shape == (50,)
        assert np.allclose(axis, values)

    def test_width_mismatch_raises(self):
        cal = ArrayCalibration(values=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="length"):
            cal.build_axis(image_width=10)

    def test_no_source_raises(self):
        with pytest.raises(Exception):
            ArrayCalibration()

    def test_kind_is_array(self):
        cal = ArrayCalibration(values=[1.0])
        assert cal.kind == "array"

    def test_get_charge_factor_returns_none(self):
        cal = ArrayCalibration(values=[1.0, 2.0])
        assert cal.get_charge_factor() is None


class TestMagSpecAnalyzerConfig:
    """Tests for MagSpecAnalyzerConfig validation."""

    def test_valid_config_constructs(self):
        cfg = MagSpecAnalyzerConfig(
            calibration={"kind": "polynomial", "coeffs": [50.0, 0.1]},
            energy_range=(20.0, 200.0),
            num_energy_points=500,
        )
        assert isinstance(cfg.calibration, PolynomialCalibration)
        assert cfg.energy_range == (20.0, 200.0)

    def test_energy_range_max_less_than_min_raises(self):
        with pytest.raises(Exception):
            MagSpecAnalyzerConfig(
                calibration={"kind": "polynomial", "coeffs": [1.0]},
                energy_range=(200.0, 20.0),
            )

    def test_energy_range_equal_raises(self):
        with pytest.raises(Exception):
            MagSpecAnalyzerConfig(
                calibration={"kind": "polynomial", "coeffs": [1.0]},
                energy_range=(100.0, 100.0),
            )


class TestScanFolderInvariant:
    """Pin: ``_save_calibrated_outputs`` must never create a missing scan folder.

    Regression guard. The previous implementation used ``mkdir(parents=True,
    exist_ok=True)`` to bring ``ScanXXX/<camera>-interp/`` into existence,
    which silently re-created the scan folder during transient SMB/NetApp
    visibility blips — converting a recoverable blip into permanent data loss
    by planting an empty directory over the real scan contents.
    """

    def _stale_image_file(self, tmp_path: Path) -> Path:
        """Return a path under a non-existent scan folder (the failure mode)."""
        return (
            tmp_path / "scans" / "Scan015" / "UC_TestCam" / "Scan015_UC_TestCam_001.png"
        )

    def _minimal_result(self) -> ImageAnalyzerResult:
        return ImageAnalyzerResult(
            data_type="2d",
            processed_image=np.zeros((4, 4), dtype=np.float64),
            render_data={"energy_axis": np.linspace(0.0, 1.0, 4)},
        )

    def test_save_refuses_when_scan_folder_missing(self, tmp_path):
        stale = self._stale_image_file(tmp_path)
        assert not stale.parent.parent.exists(), "precondition: scan_dir missing"

        # Bind the unbound method to a minimal stub: only ``camera_name`` is read.
        fake_self = SimpleNamespace(camera_name="UC_TestCam")

        with pytest.raises(FileNotFoundError, match="not visible"):
            MagSpecManualCalibAnalyzer._save_calibrated_outputs(
                fake_self, self._minimal_result(), stale
            )

        # The scan folder must not have been created as a side effect.
        assert not stale.parent.parent.exists()
        assert not (stale.parent.parent / "UC_TestCam-interp").exists()
        assert not (stale.parent.parent / "UC_TestCam-interpSpec").exists()
