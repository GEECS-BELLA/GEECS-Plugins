"""Unit tests for BeamAnalyzer using synthetic data.

All tests use gaussian_beam_2d — no real data or YAML config files required.
Replaces the broken test_beamanalyzer.py (old API, config-path dependency).
"""

import math

import pytest

from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
from image_analysis.processing.array2d.config_models import (
    BackgroundConfig,
    CameraConfig,
    ROIConfig,
)
from image_analysis.tools.synthetic_generators import gaussian_beam_2d


def _make_config(name: str = "test_cam") -> CameraConfig:
    """Minimal CameraConfig with no processing steps."""
    return CameraConfig(
        name=name,
        bit_depth=16,
        background=BackgroundConfig(method="constant", constant_level=0),
    )


@pytest.fixture
def analyzer():
    """BeamAnalyzer instance built from a programmatic config."""
    return BeamAnalyzer(_make_config())


class TestBeamAnalyzerInstantiation:
    """Construction tests."""

    def test_accepts_config_object(self):
        """BeamAnalyzer can be constructed from a CameraConfig directly."""
        analyzer = BeamAnalyzer(_make_config("my_cam"))
        assert analyzer.camera_config.name == "my_cam"

    def test_accepts_analysis_config(self):
        """BeamAnalyzer exposes a typed analysis_config after construction."""
        analyzer = BeamAnalyzer(_make_config())
        assert analyzer.analysis_config is not None

    def test_camera_name_property(self):
        analyzer = BeamAnalyzer(_make_config("cam_x"))
        assert analyzer.camera_name == "cam_x"


class TestBeamAnalyzerScalars:
    """Tests that scalars are present and correct for a synthetic Gaussian beam."""

    SCALARS_ALWAYS_PRESENT = [
        "x_CoM",
        "y_CoM",
        "image_total",
        "image_peak_value",
    ]

    def test_all_expected_scalars_present(self, analyzer):
        img = gaussian_beam_2d(shape=(128, 128), center=(64.0, 64.0), seed=0)
        result = analyzer.analyze_image(img)
        for key in self.SCALARS_ALWAYS_PRESENT:
            assert f"test_cam_{key}" in result.scalars, f"Missing: test_cam_{key}"

    def test_all_scalars_finite(self, analyzer):
        img = gaussian_beam_2d(shape=(128, 128), center=(64.0, 64.0), seed=0)
        result = analyzer.analyze_image(img)
        for key in self.SCALARS_ALWAYS_PRESENT:
            prefixed = f"test_cam_{key}"
            assert math.isfinite(result.scalars[prefixed]), f"Non-finite: {prefixed}"


class TestBeamAnalyzerCentroidAccuracy:
    """Tests that the centroid estimate is close to the known ground truth."""

    @pytest.mark.parametrize(
        "center",
        [
            (64.0, 64.0),
            (40.0, 90.0),
            (100.0, 30.0),
        ],
    )
    def test_centroid_matches_ground_truth(self, center):
        """CoM should recover the known beam centre to within 1 pixel."""
        row, col = center
        img = gaussian_beam_2d(
            shape=(128, 128), center=(row, col), sigma=(10.0, 10.0), seed=1
        )
        analyzer = BeamAnalyzer(_make_config())
        result = analyzer.analyze_image(img)

        # BeamAnalyzer reports x=column, y=row
        assert abs(result.scalars["test_cam_x_CoM"] - col) < 1.0
        assert abs(result.scalars["test_cam_y_CoM"] - row) < 1.0

    def test_image_total_scales_with_peak(self):
        """Higher peak_value → higher image_total."""
        low = gaussian_beam_2d(
            shape=(64, 64), center=(32.0, 32.0), peak_value=500, seed=2
        )
        high = gaussian_beam_2d(
            shape=(64, 64), center=(32.0, 32.0), peak_value=5000, seed=2
        )
        analyzer = BeamAnalyzer(_make_config())
        r_low = analyzer.analyze_image(low)
        r_high = analyzer.analyze_image(high)
        assert (
            r_high.scalars["test_cam_image_total"]
            > r_low.scalars["test_cam_image_total"]
        )


class TestBeamAnalyzerResult:
    """Tests on the ImageAnalyzerResult structure."""

    def test_processed_image_is_2d(self, analyzer):
        img = gaussian_beam_2d(shape=(64, 64), seed=3)
        result = analyzer.analyze_image(img)
        assert result.processed_image is not None
        assert result.processed_image.ndim == 2

    def test_data_type_is_2d(self, analyzer):
        img = gaussian_beam_2d(shape=(64, 64), seed=3)
        result = analyzer.analyze_image(img)
        assert result.data_type == "2d"

    def test_render_data_has_projections(self, analyzer):
        img = gaussian_beam_2d(shape=(64, 64), seed=3)
        result = analyzer.analyze_image(img)
        assert "horizontal_projection" in result.render_data
        assert "vertical_projection" in result.render_data


class TestBeamAnalyzerROICoordinates:
    """CoM and peak_location are expressed in full-image coordinates when ROI is set."""

    def test_roi_offset_applied_to_x_com(self):
        """x_CoM reflects the beam position in the full sensor frame, not the crop."""
        # Beam centred at global pixel (col=120, row=80) in a 200x200 image
        global_col, global_row = 120, 80
        img = gaussian_beam_2d(
            shape=(200, 200), center=(global_row, global_col), sigma=(8.0, 8.0), seed=10
        )

        # ROI that contains the beam; origin at (x_min=100, y_min=60)
        roi = ROIConfig(x_min=100, x_max=200, y_min=60, y_max=140)
        config = CameraConfig(name="roi_cam", bit_depth=16, roi=roi)
        analyzer = BeamAnalyzer(config)
        result = analyzer.analyze_image(img)

        # CoM should be close to the global position, not local (20, 20)
        assert abs(result.scalars["roi_cam_x_CoM"] - global_col) < 1.5
        assert abs(result.scalars["roi_cam_y_CoM"] - global_row) < 1.5

    def test_roi_offset_not_applied_without_roi(self):
        """Without ROI, CoM is in the full-image frame (offset is trivially 0)."""
        global_col, global_row = 64, 64
        img = gaussian_beam_2d(
            shape=(128, 128), center=(global_row, global_col), sigma=(8.0, 8.0), seed=11
        )
        config = CameraConfig(name="no_roi_cam", bit_depth=16)
        analyzer = BeamAnalyzer(config)
        result = analyzer.analyze_image(img)

        assert abs(result.scalars["no_roi_cam_x_CoM"] - global_col) < 1.5
        assert abs(result.scalars["no_roi_cam_y_CoM"] - global_row) < 1.5

    def test_roi_width_stats_unchanged_by_offset(self):
        """rms and fwhm are identical with or without a non-zero ROI origin."""
        img = gaussian_beam_2d(
            shape=(100, 100), center=(50.0, 50.0), sigma=(6.0, 6.0), seed=12
        )

        # No ROI — full image
        cfg_full = CameraConfig(name="full", bit_depth=16)
        result_full = BeamAnalyzer(cfg_full).analyze_image(img)

        # ROI that covers the same area but with a non-zero origin
        # (by putting the beam inside a larger image with an offset ROI)
        img_large = gaussian_beam_2d(
            shape=(200, 200), center=(150.0, 150.0), sigma=(6.0, 6.0), seed=12
        )
        roi = ROIConfig(x_min=100, x_max=200, y_min=100, y_max=200)
        cfg_roi = CameraConfig(name="roi", bit_depth=16, roi=roi)
        result_roi = BeamAnalyzer(cfg_roi).analyze_image(img_large)

        assert result_roi.scalars["roi_x_rms"] == pytest.approx(
            result_full.scalars["full_x_rms"], rel=0.01
        )
        assert result_roi.scalars["roi_y_rms"] == pytest.approx(
            result_full.scalars["full_y_rms"], rel=0.01
        )

    def test_oversized_roi_is_clamped_not_raised(self):
        """BeamAnalyzer does not raise when the configured ROI exceeds the image."""
        img = gaussian_beam_2d(shape=(50, 50), center=(25.0, 25.0), seed=13)

        # ROI claims a 200x200 region on a 50x50 image — should be clamped
        roi = ROIConfig(x_min=0, x_max=200, y_min=0, y_max=200)
        config = CameraConfig(name="big_roi_cam", bit_depth=16, roi=roi)
        analyzer = BeamAnalyzer(config)

        result = analyzer.analyze_image(img)  # must not raise
        assert result.processed_image is not None
        assert result.processed_image.shape == (50, 50)  # clamped to actual image


class TestBeamAnalyzerUpdateConfig:
    """Tests for update_config()."""

    def test_update_background_does_not_raise(self, analyzer):
        from image_analysis.processing.array2d import config_models as cfg

        new_bkg = cfg.BackgroundConfig(method="constant", constant_level=200)
        analyzer.update_config(background=new_bkg)
        img = gaussian_beam_2d(shape=(64, 64), peak_value=3000, seed=4)
        result = analyzer.analyze_image(img)
        assert result.processed_image is not None
