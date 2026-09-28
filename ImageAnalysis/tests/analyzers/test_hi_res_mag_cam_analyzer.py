"""Unit tests for HiResMagCamAnalyzer using synthetic bowtie image data."""

import math

import numpy as np
import pytest

from geecs_schemas.analysis import HiResMagCamSpec

from image_analysis.analyzers.Undulator.hi_res_mag_cam_analyzer import (
    HiResMagCamAnalyzer,
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
        analyzer = HiResMagCamAnalyzer(_make_config())
        # Standalone Mode-1 construction: output_name defaults to None.
        assert analyzer.output_name is None

    def test_bowtie_algorithm_initialized(self):
        analyzer = HiResMagCamAnalyzer(_make_config())
        assert analyzer.algo is not None

    def test_custom_parameters_stored(self):
        analyzer = HiResMagCamAnalyzer(
            _make_config(),
            spec=HiResMagCamSpec(n_beam_size_clearance=6, min_total_counts=5000.0),
        )
        assert analyzer.n_beam_size_clearance == 6
        assert analyzer.min_total_counts == 5000.0


class TestHiResMagCamAnalyzerScalars:
    """Scalar presence tests on bowtie synthetic image."""

    BEAM_SCALARS = ["x_CoM", "y_CoM", "image_total", "image_peak_value"]
    BOWTIE_SCALARS = ["emittance_proxy", "total_counts", "bowtie_y0"]

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


class TestHiResMagCamWaistPosition:
    """``bowtie_y0``: the beam's vertical position at the fitted waist column."""

    @pytest.mark.parametrize("vertical_offset", [-10, 0, 7])
    def test_y0_tracks_beam_height(self, analyzer, vertical_offset):
        image = generate_bowtie_image(
            shape=(64, 128),
            total_charge=1.0,
            noise_level=10.0,
            background_level=0,
            vertical_offset=vertical_offset,
            seed=42,
        )
        result = analyzer.analyze_image(image)
        assert result.scalars["bowtie_y0"] == pytest.approx(
            32 + vertical_offset, abs=0.25
        )

    def test_y0_reads_the_track_at_the_waist_not_the_image_centroid(self, analyzer):
        # Shear the bow-tie about column 64 (0.5 rows per column) and put the
        # charge off-centre: the whole-image centroid then sits on the track
        # where the charge is, the waist reading on the track at x0.
        shear = 0.5
        image = generate_bowtie_image(
            shape=(128, 128),
            total_charge=1.0,
            noise_level=10.0,
            background_level=0,
            energy_center=90,
            energy_spread=30,
            seed=42,
        )
        image = np.stack(
            [np.roll(image[:, c], round(shear * (c - 64))) for c in range(128)],
            axis=1,
        )
        result = analyzer.analyze_image(image)
        x0 = analyzer.algo.evaluate(np.where(image < 10, 0, image).astype(float)).x0
        track_at_waist = 64 + shear * (x0 - 64)
        # The test only discriminates if the two readings differ.
        assert abs(result.scalars["y_CoM"] - track_at_waist) > 1.0
        assert result.scalars["bowtie_y0"] == pytest.approx(track_at_waist, abs=0.75)

    def test_y0_nan_on_blank_frame(self, analyzer):
        result = analyzer.analyze_image(np.zeros((64, 128), dtype=np.uint16))
        assert math.isnan(result.scalars["bowtie_y0"])

    def test_center_at_waist_interpolates_tilted_track(self):
        # A single-pixel track that climbs one row every 4 columns: the
        # whole-image centroid is the track's mean, the waist reading is not.
        image = np.zeros((40, 40))
        for col in range(40):
            image[5 + col // 4, col] = 100.0
        fit_columns = np.ones(40, dtype=bool)
        # Columns 11 and 12 sit on rows 7 and 8; x0 = 11.5 is halfway between.
        y0 = HiResMagCamAnalyzer.center_at_waist(image, 11.5, fit_columns)
        assert y0 == pytest.approx(7.5)

    @pytest.mark.parametrize("x0", [np.nan, 1e6, 2.0, 30.0])
    def test_center_at_waist_never_extrapolates(self, x0):
        image = np.ones((20, 40))
        fit_columns = np.zeros(40, dtype=bool)
        fit_columns[5:25] = True
        assert math.isnan(HiResMagCamAnalyzer.center_at_waist(image, x0, fit_columns))
