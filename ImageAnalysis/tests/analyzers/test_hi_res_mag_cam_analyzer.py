"""Unit tests for HiResMagCamAnalyzer using synthetic bowtie image data."""

import math

import pytest

from image_analysis.offline_analyzers.Undulator.hi_res_mag_cam_analyzer import (
    HiResMagCamAnalyzer,
)
from image_analysis.processing.array2d.config_models import (
    BackgroundConfig,
    CameraConfig,
)
from image_analysis.tools.synthetic_generators import generate_bowtie_image


def _make_config(name: str = "test_hires") -> CameraConfig:
    """Minimal CameraConfig with no processing steps."""
    return CameraConfig(
        name=name,
        bit_depth=16,
        background=BackgroundConfig(method="constant", constant_level=0),
    )


@pytest.fixture
def analyzer():
    return HiResMagCamAnalyzer(_make_config())


@pytest.fixture
def bowtie_image():
    return generate_bowtie_image(
        shape=(64, 128),
        total_charge=1.0,
        noise_level=10.0,
        background_level=0,
        seed=42,
    )


class TestHiResMagCamAnalyzerInstantiation:
    """Construction tests."""

    def test_accepts_camera_config_object(self):
        analyzer = HiResMagCamAnalyzer(_make_config("my_hires"))
        assert analyzer.camera_config.name == "my_hires"

    def test_bowtie_algorithm_initialized(self):
        analyzer = HiResMagCamAnalyzer(_make_config())
        assert analyzer.algo is not None

    def test_custom_parameters_stored(self):
        analyzer = HiResMagCamAnalyzer(
            _make_config(), n_beam_size_clearance=6, min_total_counts=5000.0
        )
        assert analyzer.n_beam_size_clearance == 6
        assert analyzer.min_total_counts == 5000.0


class TestHiResMagCamAnalyzerScalars:
    """Scalar presence tests on bowtie synthetic image."""

    BEAM_SCALARS = ["x_CoM", "y_CoM", "image_total", "image_peak_value"]
    BOWTIE_SCALARS = ["emittance_proxy", "total_counts"]

    def test_beam_scalars_present(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        for key in self.BEAM_SCALARS:
            prefixed = f"test_hires_{key}"
            assert prefixed in result.scalars, f"Missing: {prefixed}"

    def test_bowtie_scalars_present(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        for key in self.BOWTIE_SCALARS:
            assert key in result.scalars, f"Missing: {key}"

    def test_beam_scalars_finite(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        for key in self.BEAM_SCALARS:
            prefixed = f"test_hires_{key}"
            assert math.isfinite(result.scalars[prefixed]), f"Non-finite: {prefixed}"

    def test_total_counts_non_negative(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        assert result.scalars["total_counts"] >= 0.0


class TestHiResMagCamAnalyzerResult:
    """Result structure tests."""

    def test_processed_image_is_2d(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        assert result.processed_image is not None
        assert result.processed_image.ndim == 2

    def test_data_type_is_2d(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        assert result.data_type == "2d"

    def test_render_data_has_projections(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        assert "horizontal_projection" in result.render_data
        assert "vertical_projection" in result.render_data

    def test_render_data_has_bowtie_weights(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        assert "bowtie_weights" in result.render_data
