"""Integration tests for HASOHimgHasProcessor.

Requires:
- WaveKit 4.3 SDK installed and licensed (Windows-only)
- Real .himg file on the data drive (Z:/)
- wavekit_config_path set in ~/.config/geecs_python_api/config.ini:
    [Paths]
    wavekit_config_path = Z:/path/to/WFS_HASO4_LIFT_680_8244_gain_enabled.dat

The entire module is skipped when the WaveKit Python bindings cannot be
imported (non-Windows machines or machines without the SDK).  Individual
tests also skip if the data file or WaveKit config path are not configured.

Note: HASOHimgHasProcessor.analyze_image_file() writes sidecar files
(.has slopes, .tsv phase maps) alongside the source .himg file — this is
intentional and mirrors the notebook workflow.
"""

from pathlib import Path

import numpy as np
import pytest
from geecs_data_utils import GeecsPathsConfig

# Skip the whole module if the WaveKit SDK is unavailable.
pytest.importorskip("image_analysis.third_party_sdks.wavekit_43.wavekit_py")

from image_analysis.offline_analyzers.HASO_himg_has_processor import (  # noqa: E402
    HASOHimgHasProcessor,
)

try:
    WAVEKIT_CONFIG = GeecsPathsConfig().wavekit_config_path
except Exception:
    WAVEKIT_CONFIG = None

DATA_FILE = Path(
    "Z:/data/Undulator/Y2026/03-Mar/26_0310/scans/Scan012"
    "/U_HasoLift/Scan012_U_HasoLift_001.himg"
)

MASK_KWARGS = dict(mask_top=200, mask_bottom=500, mask_left=10, mask_right=680)


@pytest.fixture(scope="module")
def haso_result():
    if WAVEKIT_CONFIG is None:
        pytest.skip("wavekit_config_path not set or not found — add it to config.ini")
    if not DATA_FILE.exists():
        pytest.skip(f"Data file not found: {DATA_FILE}")

    analyzer = HASOHimgHasProcessor(
        wavekit_config_file_path=WAVEKIT_CONFIG,
        **MASK_KWARGS,
    )
    return analyzer.analyze_image_file(image_filepath=DATA_FILE)


@pytest.mark.integration
class TestHASOAnalyzerIntegration:
    def test_data_type_is_2d(self, haso_result):
        assert haso_result.data_type == "2d"

    def test_processed_image_is_2d(self, haso_result):
        assert haso_result.processed_image is not None
        assert haso_result.processed_image.ndim == 2

    def test_processed_image_not_nan_dominated(self, haso_result):
        nan_fraction = np.isnan(haso_result.processed_image).mean()
        assert (
            nan_fraction < 0.5
        ), f"Over 50% NaN in processed_image ({nan_fraction:.1%})"

    def test_processed_image_has_nonzero_extent(self, haso_result):
        """Phase image should have at least a few rows and columns of output."""
        rows, cols = haso_result.processed_image.shape
        assert rows > 10, f"Too few rows in processed_image: {rows}"
        assert cols > 10, f"Too few cols in processed_image: {cols}"

    @pytest.mark.parametrize(
        "suffix",
        [
            "_raw.has",
            "_postprocessed.has",
            "_raw.tsv",
            "_postprocessed.tsv",
            "_intensity.tsv",
        ],
    )
    def test_sidecar_files_created(self, haso_result, suffix):
        expected = DATA_FILE.parent / f"{DATA_FILE.stem}{suffix}"
        assert expected.exists(), f"Expected sidecar file not found: {expected}"
