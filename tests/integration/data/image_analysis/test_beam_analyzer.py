"""Integration smoke test for BeamAnalyzer.

Verifies the end-to-end path: real image file → BeamAnalyzer → finite scalars.
Algorithm correctness is tested in ImageAnalysis/tests/analyzers/test_beam_analyzer.py
using synthetic data (no network required).

Run with:
    pytest -m "integration and data" tests/integration/data/image_analysis/
"""

import math

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.data]

DEV_NAME = "UC_Amp4_IR_input"
EXPECTED_SCALARS = ["x_CoM", "y_CoM", "image_total", "image_peak_value"]


def test_beam_analyzer_end_to_end():
    """BeamAnalyzer loads a real image and returns finite scalars."""
    from geecs_data_utils.scan_data import ScanData
    from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
    from image_analysis.processing.array2d.config_models import (
        BackgroundConfig,
        CameraConfig,
    )

    scan = ScanData.from_date(
        year=2025, month=2, day=20, number=14, experiment="Undulator", append_paths=True
    )
    file_path = scan.data_frame[f"{DEV_NAME}_expected_path"].iloc[0]
    assert file_path is not None and str(file_path) != "nan"

    analyzer = BeamAnalyzer(
        CameraConfig(
            name=DEV_NAME,
            bit_depth=16,
            background=BackgroundConfig(method="constant", constant_level=0),
        )
    )
    result = analyzer.analyze_image_file(file_path)

    assert result.processed_image is not None
    assert result.processed_image.ndim == 2
    for key in EXPECTED_SCALARS:
        prefixed = f"{DEV_NAME}_{key}"
        assert prefixed in result.scalars, f"Missing scalar: {prefixed}"
        assert math.isfinite(result.scalars[prefixed]), f"Non-finite: {prefixed}"
