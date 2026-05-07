"""Integration tests for ScatterPlotterAnalysis.

Requires real scan data on the data drive (Z:/).  All tests are
@pytest.mark.integration and are skipped when the scan folder is not found.

Test scan: Undulator 2026-05-05 Scan018
  x-axis : U_ModeImagerESP Position.Axis 1 Alias:ModeImager
  y-series: UC_ModeImager_x_rms, UC_ModeImager_y_rms, UC_ModeImager_image_peak_value

Output is written to:
  26_0505/analysis/Scan018/scatter_plots/mode_imager_rms.png
"""

from pathlib import Path

import pytest
from geecs_data_utils import ScanTag

from scan_analysis.analyzers.common.scatter_plotter_analysis import (
    PlotParameter,
    ScatterPlotterAnalysis,
)
from scan_analysis.config.analyzer_config_models import (
    PlotParameterConfig,
    ScatterAnalyzerConfig,
)
from scan_analysis.config.analyzer_factory import create_analyzer

SCAN_FOLDER = Path("Z:/data/Undulator/Y2026/05-May/26_0505/scans/Scan018")

SCAN_TAG = ScanTag(year=2026, month=5, day=5, number=18, experiment="Undulator")

FILENAME = "mode_imager_rms"
# analysis/ is a sibling of scans/ — derived from SCAN_FOLDER without ScanPaths init
EXPECTED_PNG = (
    SCAN_FOLDER.parents[1]
    / "analysis"
    / SCAN_FOLDER.name
    / "scatter_plots"
    / f"{FILENAME}.png"
)

PARAMETERS = [
    PlotParameter(
        key_name="UC_ModeImager_x_rms",
        legend_label="x RMS (px)",
        axis_label="x RMS (px)",
        color="blue",
    ),
    PlotParameter(
        key_name="UC_ModeImager_y_rms",
        legend_label="y RMS (px)",
        axis_label="y RMS (px)",
        color="red",
    ),
    PlotParameter(
        key_name="UC_ModeImager_image_peak_value",
        legend_label="Peak Value",
        axis_label="Peak Value",
        color="green",
    ),
]

X_COLUMN = "U_ModeImagerESP Position.Axis 1 Alias:ModeImager"


@pytest.fixture(scope="module")
def scatter_result():
    if not SCAN_FOLDER.exists():
        pytest.skip(f"Scan folder not found: {SCAN_FOLDER}")

    EXPECTED_PNG.unlink(missing_ok=True)

    analyzer = ScatterPlotterAnalysis(
        use_median=True,
        title="ModeImager Beam Properties",
        parameters=PARAMETERS,
        filename=FILENAME,
        x_column=X_COLUMN,
    )
    return analyzer.run_analysis(SCAN_TAG)


@pytest.mark.integration
class TestScatterPlotterAnalysisIntegration:
    def test_returns_display_contents(self, scatter_result):
        assert scatter_result is not None
        assert len(scatter_result) > 0

    def test_png_created(self, scatter_result):
        assert EXPECTED_PNG.exists(), f"Expected PNG not found: {EXPECTED_PNG}"

    def test_png_path_in_display_contents(self, scatter_result):
        assert str(EXPECTED_PNG) in scatter_result

    def test_png_in_scatter_plots_subdir(self, scatter_result):
        png_path = Path(scatter_result[-1])
        assert png_path.parent.name == "scatter_plots"
        assert png_path.parent.parent.name == SCAN_FOLDER.name  # Scan018
        assert png_path.parent.parent.parent.name == "analysis"


@pytest.mark.integration
class TestScatterAnalyzerConfigFactory:
    """Verify that ScatterAnalyzerConfig → create_analyzer produces a working analyzer."""

    def test_factory_creates_analyzer(self):
        cfg = ScatterAnalyzerConfig(
            title="ModeImager Beam Properties",
            filename=FILENAME,
            x_column=X_COLUMN,
            parameters=[
                PlotParameterConfig(
                    key_name="UC_ModeImager_x_rms", label="x RMS (px)", color="blue"
                ),
                PlotParameterConfig(
                    key_name="UC_ModeImager_y_rms", label="y RMS (px)", color="red"
                ),
                PlotParameterConfig(
                    key_name="UC_ModeImager_image_peak_value",
                    label="Peak Value",
                    color="green",
                ),
            ],
        )
        analyzer = create_analyzer(cfg)
        assert isinstance(analyzer, ScatterPlotterAnalysis)
        assert analyzer.id == FILENAME
        assert analyzer.priority == 200
        assert analyzer.x_column == X_COLUMN
        assert len(analyzer.parameters) == 3
