"""Unit tests for BCaveMagSpecCam1Analyzer using synthetic bowtie image data."""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from geecs_schemas.analysis import BCaveMagSpecCam1Spec

from image_analysis.analyzers.Undulator.bcave_magspec_cam1_analyzer import (
    BCaveMagSpecCam1Analyzer,
)
from geecs_schemas.analysis.processing_2d import (
    BackgroundConfig,
    CameraConfig,
)
from image_analysis.tools.synthetic_generators import generate_bowtie_image


def _make_config() -> CameraConfig:
    """Minimal CameraConfig with no processing steps."""
    return CameraConfig(
        bit_depth=16,
        background=BackgroundConfig(method="constant", constant_level=0),
    )


@pytest.fixture
def analyzer():
    return BCaveMagSpecCam1Analyzer(_make_config())


@pytest.fixture
def bowtie_image():
    return generate_bowtie_image(
        shape=(64, 128),
        total_charge=1.0,
        noise_level=10.0,
        background_level=0,
        seed=42,
    )


class TestBCaveMagSpecCam1AnalyzerInstantiation:
    """Construction tests."""

    def test_accepts_camera_config_object(self):
        analyzer = BCaveMagSpecCam1Analyzer(_make_config())
        # Standalone Mode-1 construction: output_name defaults to None.
        assert analyzer.output_name is None

    def test_bowtie_algorithm_initialized(self):
        analyzer = BCaveMagSpecCam1Analyzer(_make_config())
        assert analyzer.algo is not None

    def test_custom_parameters_stored(self):
        analyzer = BCaveMagSpecCam1Analyzer(
            _make_config(),
            spec=BCaveMagSpecCam1Spec(
                n_beam_size_clearance=6,
                min_total_counts=5000.0,
                count_threshold=25.0,
            ),
        )
        assert analyzer.n_beam_size_clearance == 6
        assert analyzer.min_total_counts == 5000.0
        assert analyzer.count_threshold == 25.0


class TestBCaveMagSpecCam1AnalyzerScalars:
    """Scalar presence tests on bowtie synthetic image."""

    BEAM_SCALARS = ["x_CoM", "y_CoM", "image_total", "image_peak_value"]
    BOWTIE_SCALARS = [
        "emittance_proxy",
        "total_counts",
        "bowtie_w0",
        "bowtie_theta",
        "bowtie_x0",
        "bowtie_r_squared",
    ]

    def test_beam_scalars_present(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        for key in self.BEAM_SCALARS:
            assert key in result.scalars, f"Missing: {key}"

    def test_bowtie_scalars_present(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        for key in self.BOWTIE_SCALARS:
            assert key in result.scalars, f"Missing: {key}"

    def test_beam_scalars_finite(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        for key in self.BEAM_SCALARS:
            assert math.isfinite(result.scalars[key]), f"Non-finite: {key}"

    def test_total_counts_non_negative(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        assert result.scalars["total_counts"] >= 0.0

    def test_declared_scalars_are_emitted(self, analyzer, bowtie_image):
        """The spec's emitted_scalars() promise holds against a real run."""
        result = analyzer.analyze_image(bowtie_image)
        assert BCaveMagSpecCam1Spec().emitted_scalars() <= set(result.scalars)


class TestBCaveMagSpecCam1CountThreshold:
    """The configurable noise floor actually reaches the bowtie fit."""

    def test_high_threshold_zeroes_the_fit_input(self, bowtie_image):
        """A threshold above every pixel leaves no counts for the fit."""
        analyzer = BCaveMagSpecCam1Analyzer(
            _make_config(),
            spec=BCaveMagSpecCam1Spec(count_threshold=1e9),
        )
        result = analyzer.analyze_image(bowtie_image)
        assert result.scalars["total_counts"] == 0.0

    def test_zero_threshold_keeps_more_counts_than_a_high_one(self, bowtie_image):
        low = BCaveMagSpecCam1Analyzer(
            _make_config(), spec=BCaveMagSpecCam1Spec(count_threshold=0.0)
        ).analyze_image(bowtie_image)
        high = BCaveMagSpecCam1Analyzer(
            _make_config(), spec=BCaveMagSpecCam1Spec(count_threshold=50.0)
        ).analyze_image(bowtie_image)
        assert low.scalars["total_counts"] > high.scalars["total_counts"]


class TestBCaveMagSpecCam1AnalyzerResult:
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


class TestBCaveMagSpecCam1Rendering:
    """The renderer honours a supplied axes (the ephemeral seam contract)."""

    def test_render_image_draws_into_supplied_ax(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        fig, ax = plt.subplots()
        try:
            _, returned_ax = analyzer.render_image(result, ax=ax)
            assert returned_ax is ax
            assert len(ax.images) == 1
            assert ax.lines, "the bowtie weight overlay must reach the given axes"
        finally:
            plt.close(fig)
