"""Integration tests for ScanAnalysis Array2DScanAnalyzer.

Requires network-mounted GEECS data.  Config objects are constructed
programmatically so these tests do NOT depend on the GEECS-Plugins-Configs repo.

Run with:
    pytest -m "integration and data" tests/integration/data/scan_analysis/
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.data]

DEV_NAME = "UC_Amp4_IR_input"


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


def test_array2d_beam_analyzer_noscan():
    """Array2DScanAnalyzer runs BeamAnalyzer on a noscan and produces results.

    Uses flag_save_images=False to avoid writing to the data directory.
    """
    from geecs_data_utils import ScanTag
    from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
    from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer

    image_analyzer = BeamAnalyzer(_make_beam_config(DEV_NAME))
    scan_analyzer = Array2DScanAnalyzer(
        image_analyzer=image_analyzer,
        device_name=DEV_NAME,
        flag_save_images=False,
    )

    tag = ScanTag(year=2025, month=2, day=20, number=14, experiment="Undulator")
    scan_analyzer.run_analysis(scan_tag=tag)

    assert len(scan_analyzer.results) > 0
    result = next(iter(scan_analyzer.results.values()))
    assert result.processed_image is not None
    assert result.processed_image.ndim == 2


def test_array2d_results_have_scalars():
    """Each per-shot result contains the expected beam scalar keys."""
    from geecs_data_utils import ScanTag
    from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
    from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer

    image_analyzer = BeamAnalyzer(_make_beam_config(DEV_NAME))
    scan_analyzer = Array2DScanAnalyzer(
        image_analyzer=image_analyzer,
        device_name=DEV_NAME,
        flag_save_images=False,
    )

    tag = ScanTag(year=2025, month=2, day=20, number=14, experiment="Undulator")
    scan_analyzer.run_analysis(scan_tag=tag)

    for result in scan_analyzer.results.values():
        assert f"{DEV_NAME}_x_CoM" in result.scalars
        assert f"{DEV_NAME}_y_CoM" in result.scalars
        assert f"{DEV_NAME}_image_total" in result.scalars
