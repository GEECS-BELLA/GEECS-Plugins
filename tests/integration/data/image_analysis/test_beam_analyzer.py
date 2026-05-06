"""Integration tests for ImageAnalysis offline analyzers.

Requires network-mounted GEECS data and the GEECS-Plugins-Configs repo.
Run with:
    pytest -m "integration and data" tests/integration/data/image_analysis/
"""

import math

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.data]

BEAM_ANALYZER_SCALARS = [
    "x_CoM",
    "y_CoM",
    "x_fwhm",
    "y_fwhm",
    "x_rms",
    "y_rms",
    "image_total",
    "image_peak_value",
]


@pytest.fixture(scope="module")
def beam_analyzer():
    from geecs_data_utils.scan_data import ScanPaths
    from geecs_data_utils.config_roots import image_analysis_config
    from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer

    image_analysis_config.set_base_dir(
        ScanPaths.paths_config.image_analysis_configs_path
    )
    return BeamAnalyzer("HTT-C-ASSERTHighR")


@pytest.fixture(scope="module")
def thomson_scan():
    from geecs_data_utils.scan_data import ScanData

    return ScanData.from_date(
        year=2025, month=9, day=24, number=13, experiment="Thomson", append_paths=True
    )


def test_beam_analyzer_instantiates(beam_analyzer):
    """BeamAnalyzer constructs without error and loads its config."""
    assert beam_analyzer is not None
    assert beam_analyzer.camera_config is not None


def test_beam_analyzer_single_shot(beam_analyzer, thomson_scan):
    """BeamAnalyzer produces a valid result on a real image."""
    file_path = thomson_scan.data_frame["HTT-C-ASSERTHighR_expected_path"].iloc[0]
    assert file_path is not None and str(file_path) != "nan"

    result = beam_analyzer.analyze_image_file(file_path)

    assert result is not None
    assert result.processed_image is not None
    assert result.processed_image.ndim == 2


def test_beam_analyzer_scalars_finite(beam_analyzer, thomson_scan):
    """All expected scalar metrics are present and finite."""
    file_path = thomson_scan.data_frame["HTT-C-ASSERTHighR_expected_path"].iloc[0]
    result = beam_analyzer.analyze_image_file(file_path)

    for key in BEAM_ANALYZER_SCALARS:
        prefixed = f"HTT-C-ASSERTHighR_{key}"
        assert prefixed in result.scalars, f"Missing scalar: {prefixed}"
        assert math.isfinite(result.scalars[prefixed]), f"Non-finite scalar: {prefixed}"


def test_beam_analyzer_update_config(beam_analyzer, thomson_scan):
    """update_config changes behavior without raising."""
    from image_analysis.processing.array2d import config_models as cfg

    file_path = thomson_scan.data_frame["HTT-C-ASSERTHighR_expected_path"].iloc[0]

    new_bkg = cfg.BackgroundConfig()
    new_bkg.method = "constant"
    new_bkg.constant_level = 400

    beam_analyzer.update_config(background=new_bkg)
    result = beam_analyzer.analyze_image_file(file_path)
    assert result.processed_image is not None


def test_magspec_analyzer_single_shot(canonical_scan):
    """MagSpecManualCalibAnalyzer produces a result on a real image."""
    from geecs_data_utils.scan_data import ScanPaths
    from geecs_data_utils.config_roots import image_analysis_config
    from image_analysis.offline_analyzers.magspec_manual_calib_analyzer import (
        MagSpecManualCalibAnalyzer,
    )

    image_analysis_config.set_base_dir(
        ScanPaths.paths_config.image_analysis_configs_path
    )

    sd = canonical_scan("undulator_2d")
    dev_name = "UC_HiResMagCam"
    analyzer = MagSpecManualCalibAnalyzer(dev_name)

    tag = sd.paths.get_tag()
    file_path = sd.paths.get_device_shot_path(
        device_name=dev_name, shot_number=1, tag=tag
    )

    result = analyzer.analyze_image_file(image_filepath=file_path)
    assert result is not None
    assert result.processed_image is not None
