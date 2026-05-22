"""Integration tests for GrenouilleAnalyzer.

Requires:
- FROG DLL path configured in ~/.config/geecs_python_api/config.ini
  (frog_dll_path, frog_python32_path)
- Real FROG trace image on the data drive (Z:/)

Config is frozen in code (no dependency on the mounted configs repo) so that
changes to the YAML do not silently affect these tests.  To update the frozen
config, read U_FROG_Grenouille-Temporal.yaml and edit _make_config() below.

All tests are @pytest.mark.integration and are skipped when the data file is
absent or the DLL cannot be initialised.
"""

import math
from pathlib import Path

import pytest
from geecs_data_utils import ScanPaths, ScanTag

from image_analysis.offline_analyzers.grenouille_analyzer import GrenouilleAnalyzer
from image_analysis.processing.array2d.config_models import (
    BackgroundConfig,
    CameraConfig,
    FilteringConfig,
    ThresholdingConfig,
)

CAMERA_NAME = "U_FROG_Grenouille-Temporal"

# Resolve the test data path via ScanPaths so it works on any machine where
# the user's config.ini points to an accessible data share (Windows Z:/,
# macOS /Volumes/..., etc.). On a machine without the data, ScanPaths raises
# and DATA_FILE falls back to a sentinel that won't .exists() — the
# fixture's pytest.skip() then fires cleanly.
try:
    DATA_FILE = (
        ScanPaths(
            tag=ScanTag(year=2026, month=2, day=10, number=15, experiment="Undulator")
        ).get_folder()
        / CAMERA_NAME
        / "Scan015_U_FROG_Grenouille_010.png"
    )
except Exception:
    DATA_FILE = Path("__data_not_available__")

EXPECTED_SCALARS = [
    f"{CAMERA_NAME}_temporal_fwhm",
    f"{CAMERA_NAME}_spectral_fwhm",
    f"{CAMERA_NAME}_frog_error",
    f"{CAMERA_NAME}_frog_iterations",
]


def _make_config() -> CameraConfig:
    """Frozen snapshot of U_FROG_Grenouille-Temporal.yaml.

    Edit here (not in the YAML) when config values change and tests need updating.
    """
    return CameraConfig(
        name=CAMERA_NAME,
        bit_depth=16,
        background=BackgroundConfig(method="constant", constant_level=1.0),
        thresholding=ThresholdingConfig(
            enabled=True, method="constant", value=0.0, mode="to_zero", invert=False
        ),
        filtering=FilteringConfig(gaussian_sigma=None, median_kernel_size=5),
        analysis={
            "delt": 0.85,
            "dellam": -0.085,
            "lam0": 400.0,
            "N": 512,
            "target_error": 0.0001,
            "max_time_seconds": 15,
            "max_iterations": 1000000000,
        },
    )


@pytest.fixture(scope="module")
def grenouille_result():
    if not DATA_FILE.exists():
        pytest.skip(f"Data file not found: {DATA_FILE}")

    try:
        analyzer = GrenouilleAnalyzer(camera_config_name=_make_config())
    except Exception as exc:
        pytest.skip(f"Could not initialise GrenouilleAnalyzer: {exc}")

    return analyzer.analyze_image_file(
        image_filepath=DATA_FILE,
        auxiliary_data={"file_path": str(DATA_FILE)},
    )


@pytest.mark.integration
class TestGrenouilleAnalyzerIntegration:
    def test_expected_scalar_keys_present(self, grenouille_result):
        for key in EXPECTED_SCALARS:
            assert key in grenouille_result.scalars, f"Missing scalar: {key}"

    def test_all_scalars_finite(self, grenouille_result):
        for key in EXPECTED_SCALARS:
            assert math.isfinite(grenouille_result.scalars[key]), f"Non-finite: {key}"

    def test_frog_error_plausible(self, grenouille_result):
        """FROG error should be a small positive fraction for a converged retrieval."""
        frog_error = grenouille_result.scalars[f"{CAMERA_NAME}_frog_error"]
        assert 0.0 < frog_error < 0.1, (
            f"FROG error out of expected range: {frog_error:.4f}"
        )

    def test_temporal_fwhm_plausible(self, grenouille_result):
        fwhm = grenouille_result.scalars[f"{CAMERA_NAME}_temporal_fwhm"]
        assert 20.0 < fwhm < 60.0, (
            f"temporal_fwhm out of expected range [20, 60] fs: {fwhm:.1f}"
        )

    def test_spectral_fwhm_plausible(self, grenouille_result):
        fwhm = grenouille_result.scalars[f"{CAMERA_NAME}_spectral_fwhm"]
        assert 10.0 < fwhm < 50.0, (
            f"spectral_fwhm out of expected range [10, 50] nm: {fwhm:.1f}"
        )

    def test_frog_iterations_plausible(self, grenouille_result):
        iterations = grenouille_result.scalars[f"{CAMERA_NAME}_frog_iterations"]
        assert iterations > 50, f"frog_iterations suspiciously low: {iterations}"

    def test_processed_image_is_2d(self, grenouille_result):
        assert grenouille_result.processed_image is not None
        assert grenouille_result.processed_image.ndim == 2

    def test_data_type_is_2d(self, grenouille_result):
        assert grenouille_result.data_type == "2d"

    def test_render_data_has_projections(self, grenouille_result):
        assert "horizontal_projection" in grenouille_result.render_data
        assert "vertical_projection" in grenouille_result.render_data

    def test_sidecar_lineouts_tsv_created(self, grenouille_result):
        # Takes ``grenouille_result`` only to trigger the fixture's skip-when-
        # DATA_FILE-missing behaviour; the analyzer side-effect is what creates
        # the sidecar TSV, so consuming the fixture also guarantees it ran.
        expected = DATA_FILE.parent / f"{DATA_FILE.stem}_retrieved_lineouts.tsv"
        assert expected.exists(), f"Expected sidecar TSV not found: {expected}"
