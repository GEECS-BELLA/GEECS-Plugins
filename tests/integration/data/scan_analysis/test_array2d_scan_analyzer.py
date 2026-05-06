"""Integration tests for ScanAnalysis Array2DScanAnalyzer.

Requires network-mounted GEECS data and the GEECS-Plugins-Configs repo.

These tests run full scan analysis pipelines and may take O(minutes).
Run with:
    pytest -m "integration and data" tests/integration/data/scan_analysis/
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.data]


@pytest.fixture(scope="module")
def configured_image_analysis():
    """Set up image_analysis_config once for all tests in this module."""
    from geecs_data_utils.scan_data import ScanPaths
    from geecs_data_utils.config_roots import image_analysis_config

    image_analysis_config.set_base_dir(
        ScanPaths.paths_config.image_analysis_configs_path
    )


def test_array2d_beam_analyzer_noscan(configured_image_analysis):
    """Array2DScanAnalyzer runs BeamAnalyzer on a noscan and produces results.

    Uses flag_save_images=False to avoid writing to the data directory.
    """
    from geecs_data_utils import ScanTag
    from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
    from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer

    dev_name = "UC_Amp4_IR_input"
    image_analyzer = BeamAnalyzer(camera_config_name=dev_name)
    scan_analyzer = Array2DScanAnalyzer(
        image_analyzer=image_analyzer,
        device_name=dev_name,
        flag_save_images=False,
    )

    tag = ScanTag(year=2025, month=2, day=20, number=14, experiment="Undulator")
    scan_analyzer.run_analysis(scan_tag=tag)

    assert len(scan_analyzer.results) > 0
    result = scan_analyzer.results[0]
    assert result.processed_image is not None
    assert result.processed_image.ndim == 2


def test_array2d_results_have_scalars(configured_image_analysis):
    """Each per-shot result contains the expected beam scalar keys."""
    from geecs_data_utils import ScanTag
    from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
    from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer

    dev_name = "UC_Amp4_IR_input"
    image_analyzer = BeamAnalyzer(camera_config_name=dev_name)
    scan_analyzer = Array2DScanAnalyzer(
        image_analyzer=image_analyzer,
        device_name=dev_name,
        flag_save_images=False,
    )

    tag = ScanTag(year=2025, month=2, day=20, number=14, experiment="Undulator")
    scan_analyzer.run_analysis(scan_tag=tag)

    for result in scan_analyzer.results:
        assert f"{dev_name}_x_CoM" in result.scalars
        assert f"{dev_name}_y_CoM" in result.scalars
        assert f"{dev_name}_image_total" in result.scalars
