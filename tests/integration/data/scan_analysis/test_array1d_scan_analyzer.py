"""Integration tests for ScanAnalysis Array1DScanAnalyzer.

Requires network-mounted GEECS data AND the GEECS-Plugins-Configs repo.
Config is loaded by name from the configured configs path (string names).

Run with:
    pytest -m "integration and data" tests/integration/data/scan_analysis/
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.data]

ICT_DEV = "U_BCaveICT"
MAGSPEC_DEV = "U_BCaveMagSpec-interpSpec"


# ---------------------------------------------------------------------------
# ICT — Standard1DAnalyzer on a TDMS noscan
# ---------------------------------------------------------------------------


def test_array1d_ict_noscan_runs():
    """Array1DScanAnalyzer processes an ICT noscan without error.

    Uses flag_save_data=False to avoid writing to the data directory.
    """
    from geecs_data_utils import ScanTag
    from image_analysis.offline_analyzers.standard_1d_analyzer import Standard1DAnalyzer
    from scan_analysis.analyzers.common.array1d_scan_analysis import Array1DScanAnalyzer

    image_analyzer = Standard1DAnalyzer(line_config_name=ICT_DEV)
    scan_analyzer = Array1DScanAnalyzer(
        image_analyzer=image_analyzer,
        device_name=ICT_DEV,
        file_tail=".tdms",
        flag_save_data=False,
    )

    tag = ScanTag(year=2025, month=11, day=13, number=1, experiment="Undulator")
    scan_analyzer.run_analysis(scan_tag=tag)

    assert len(scan_analyzer.results) > 0


def test_array1d_ict_results_are_2d():
    """Each per-shot ICT result contains a valid Nx2 line_data array."""
    from geecs_data_utils import ScanTag
    from image_analysis.offline_analyzers.standard_1d_analyzer import Standard1DAnalyzer
    from scan_analysis.analyzers.common.array1d_scan_analysis import Array1DScanAnalyzer

    image_analyzer = Standard1DAnalyzer(line_config_name=ICT_DEV)
    scan_analyzer = Array1DScanAnalyzer(
        image_analyzer=image_analyzer,
        device_name=ICT_DEV,
        file_tail=".tdms",
        flag_save_data=False,
    )

    tag = ScanTag(year=2025, month=11, day=13, number=1, experiment="Undulator")
    scan_analyzer.run_analysis(scan_tag=tag)

    for result in scan_analyzer.results.values():
        assert result.line_data is not None
        assert result.line_data.ndim == 2
        assert result.line_data.shape[1] == 2


# ---------------------------------------------------------------------------
# MagSpec interpSpec — LineAnalyzer on a text-file noscan
# ---------------------------------------------------------------------------


def test_array1d_magspec_noscan_runs():
    """Array1DScanAnalyzer processes a MagSpec interpSpec noscan without error.

    Uses flag_save_data=False to avoid writing to the data directory.
    """
    from geecs_data_utils import ScanTag
    from image_analysis.offline_analyzers.line_analyzer import LineAnalyzer
    from scan_analysis.analyzers.common.array1d_scan_analysis import Array1DScanAnalyzer

    image_analyzer = LineAnalyzer(line_config_name=MAGSPEC_DEV)
    scan_analyzer = Array1DScanAnalyzer(
        image_analyzer=image_analyzer,
        device_name=MAGSPEC_DEV,
        file_tail=".txt",
        flag_save_data=False,
    )

    tag = ScanTag(year=2025, month=11, day=18, number=2, experiment="Undulator")
    scan_analyzer.run_analysis(scan_tag=tag)

    assert len(scan_analyzer.results) > 0


def test_array1d_magspec_results_have_com_scalar():
    """Each per-shot MagSpec result contains the CoM scalar from LineAnalyzer."""
    from geecs_data_utils import ScanTag
    from image_analysis.offline_analyzers.line_analyzer import LineAnalyzer
    from scan_analysis.analyzers.common.array1d_scan_analysis import Array1DScanAnalyzer

    image_analyzer = LineAnalyzer(line_config_name=MAGSPEC_DEV)
    scan_analyzer = Array1DScanAnalyzer(
        image_analyzer=image_analyzer,
        device_name=MAGSPEC_DEV,
        file_tail=".txt",
        flag_save_data=False,
    )

    tag = ScanTag(year=2025, month=11, day=18, number=2, experiment="Undulator")
    scan_analyzer.run_analysis(scan_tag=tag)

    expected_key = f"{MAGSPEC_DEV}_CoM"
    for result in scan_analyzer.results.values():
        assert expected_key in result.scalars, (
            f"Missing scalar '{expected_key}'; got: {list(result.scalars.keys())}"
        )
