"""Unit tests for BCaveMagSpecCam1Analyzer using synthetic bowtie image data."""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from geecs_schemas.analysis import BCaveMagSpecCam1Spec

from image_analysis.algorithms.bowtie_fit import BowtieFitAlgorithm
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
        "bowtie_y0",
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

    def test_render_data_has_bowtie_centers(self, analyzer, bowtie_image):
        result = analyzer.analyze_image(bowtie_image)
        assert "bowtie_centers" in result.render_data


class TestBCaveMagSpecCam1WaistPosition:
    """The vertical position of the beam at the fitted waist column."""

    def test_waist_position_is_inside_the_frame(self, analyzer, bowtie_image):
        """y0 is a row index of the processed image, so it lies within its height."""
        result = analyzer.analyze_image(bowtie_image)
        y0 = result.scalars["bowtie_y0"]
        assert math.isfinite(y0), "synthetic bowtie should yield a locatable waist"
        assert 0.0 <= y0 <= result.processed_image.shape[0] - 1

    def test_waist_position_tracks_a_vertical_shift(self, analyzer):
        """Shifting the beam down the frame moves y0 by the same amount."""
        rows = 12
        base = generate_bowtie_image(
            shape=(64, 128),
            total_charge=1.0,
            noise_level=0.0,
            background_level=0,
            seed=7,
        )
        shifted = np.roll(base, rows, axis=0)

        y0_base = analyzer.analyze_image(base).scalars["bowtie_y0"]
        y0_shifted = analyzer.analyze_image(shifted).scalars["bowtie_y0"]

        assert math.isfinite(y0_base) and math.isfinite(y0_shifted)
        assert y0_shifted == pytest.approx(y0_base + rows, abs=1.0)

    def test_waist_position_is_nan_when_the_fit_finds_nothing(self, analyzer):
        """An empty frame has no measurable waist — NaN, not a fabricated row."""
        blank = np.zeros((64, 128), dtype=float)
        result = analyzer.analyze_image(blank)
        assert math.isnan(result.scalars["bowtie_y0"])


class TestBowtieCenterAt:
    """The x0 -> y0 interpolation helper on BowtieFitAlgorithm."""

    def test_interpolates_between_measured_columns(self):
        centers = np.array([np.nan, 10.0, 20.0, 30.0, np.nan])
        valid = np.array([False, True, True, True, False])
        # x0 = 1.5 sits halfway between the centers at columns 1 and 2.
        assert BowtieFitAlgorithm.center_at(1.5, centers, valid) == pytest.approx(15.0)

    def test_returns_measured_value_at_an_exact_column(self):
        centers = np.array([np.nan, 10.0, 20.0, 30.0, np.nan])
        valid = np.array([False, True, True, True, False])
        assert BowtieFitAlgorithm.center_at(2.0, centers, valid) == pytest.approx(20.0)

    def test_does_not_extrapolate_outside_the_valid_span(self):
        """Outside the measured columns the track is unknown; NaN beats a projection."""
        centers = np.array([np.nan, 10.0, 20.0, 30.0, np.nan])
        valid = np.array([False, True, True, True, False])
        assert math.isnan(BowtieFitAlgorithm.center_at(0.0, centers, valid))
        assert math.isnan(BowtieFitAlgorithm.center_at(4.0, centers, valid))

    def test_nan_waist_column_gives_nan(self):
        centers = np.array([10.0, 20.0])
        valid = np.array([True, True])
        assert math.isnan(BowtieFitAlgorithm.center_at(float("nan"), centers, valid))

    def test_no_valid_columns_gives_nan(self):
        centers = np.array([np.nan, np.nan])
        valid = np.array([False, False])
        assert math.isnan(BowtieFitAlgorithm.center_at(1.0, centers, valid))


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
