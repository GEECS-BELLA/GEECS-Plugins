"""Integration tests for ImageAnalysis BeamAnalyzer.

Requires network-mounted GEECS data.  Config objects are constructed
programmatically so these tests do NOT depend on the GEECS-Plugins-Configs repo.

Run with:
    pytest -m "integration and data" tests/integration/data/image_analysis/
"""

import math

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.data]

# Device with consistent signal in the canonical Undulator noscan
DEV_NAME = "UC_Amp4_IR_input"

# Scalars always present (and finite) when the image has positive intensity
BEAM_ANALYZER_SCALARS_ALWAYS = [
    "x_CoM",
    "y_CoM",
    "image_total",
    "image_peak_value",
]


def _make_beam_config(name: str):
    """Construct a minimal CameraConfig for beam analysis (no YAML required)."""
    from image_analysis.processing.array2d.config_models import (
        BackgroundConfig,
        CameraConfig,
    )

    return CameraConfig(
        name=name,
        bit_depth=16,
        background=BackgroundConfig(method="constant", constant_level=0),
    )


@pytest.fixture(scope="module")
def beam_analyzer():
    """BeamAnalyzer built from a programmatic config — no configs-repo dependency."""
    from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer

    return BeamAnalyzer(_make_beam_config(DEV_NAME))


@pytest.fixture(scope="module")
def undulator_scan():
    """Canonical Undulator noscan with images for UC_Amp4_IR_input."""
    from geecs_data_utils.scan_data import ScanData

    return ScanData.from_date(
        year=2025, month=2, day=20, number=14, experiment="Undulator", append_paths=True
    )


def test_beam_analyzer_instantiates(beam_analyzer):
    """BeamAnalyzer constructs without error and exposes its config."""
    assert beam_analyzer is not None
    assert beam_analyzer.camera_config is not None
    assert beam_analyzer.camera_config.name == DEV_NAME


def test_beam_analyzer_single_shot(beam_analyzer, undulator_scan):
    """BeamAnalyzer produces a valid result on a real image."""
    file_path = undulator_scan.data_frame[f"{DEV_NAME}_expected_path"].iloc[0]
    assert file_path is not None and str(file_path) != "nan"

    result = beam_analyzer.analyze_image_file(file_path)

    assert result is not None
    assert result.processed_image is not None
    assert result.processed_image.ndim == 2


def test_beam_analyzer_scalars_present(beam_analyzer, undulator_scan):
    """Core scalar metric keys are always returned in result.scalars."""
    file_path = undulator_scan.data_frame[f"{DEV_NAME}_expected_path"].iloc[0]
    result = beam_analyzer.analyze_image_file(file_path)

    for key in BEAM_ANALYZER_SCALARS_ALWAYS:
        prefixed = f"{DEV_NAME}_{key}"
        assert prefixed in result.scalars, f"Missing scalar: {prefixed}"


def test_beam_analyzer_scalars_finite(beam_analyzer, undulator_scan):
    """Core scalar metrics are finite when the image has positive intensity."""
    file_path = undulator_scan.data_frame[f"{DEV_NAME}_expected_path"].iloc[0]
    result = beam_analyzer.analyze_image_file(file_path)

    for key in BEAM_ANALYZER_SCALARS_ALWAYS:
        prefixed = f"{DEV_NAME}_{key}"
        assert prefixed in result.scalars, f"Missing scalar: {prefixed}"
        assert math.isfinite(result.scalars[prefixed]), f"Non-finite scalar: {prefixed}"


def test_beam_analyzer_update_config(beam_analyzer, undulator_scan):
    """update_config changes processing behaviour without raising."""
    from image_analysis.processing.array2d import config_models as cfg

    file_path = undulator_scan.data_frame[f"{DEV_NAME}_expected_path"].iloc[0]

    new_bkg = cfg.BackgroundConfig(method="constant", constant_level=400)
    beam_analyzer.update_config(background=new_bkg)
    result = beam_analyzer.analyze_image_file(file_path)
    assert result.processed_image is not None
