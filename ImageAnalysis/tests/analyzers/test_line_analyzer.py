"""Unit tests for LineAnalyzer using synthetic 1D Gaussian data."""

import math

import numpy as np
import pytest

from image_analysis.offline_analyzers.line_analyzer import LineAnalyzer
from image_analysis.processing.array1d.config_models import Data1DConfig, Line1DConfig
from image_analysis.tools.synthetic_generators import gaussian_peak_1d


def _make_line_config(name: str = "test_line") -> Line1DConfig:
    """Minimal Line1DConfig — no processing steps."""
    return Line1DConfig(
        name=name,
        description="synthetic test line",
        data_loading=Data1DConfig(data_type="npy"),
    )


def _make_data(center: float = 5.0, sigma: float = 1.0, n: int = 200) -> np.ndarray:
    """Nx2 array with a Gaussian peak at known position."""
    x = np.linspace(0.0, 10.0, n)
    y = gaussian_peak_1d(x, center=center, sigma=sigma, amplitude=1000.0, seed=0)
    return np.column_stack([x, y])


@pytest.fixture
def analyzer():
    return LineAnalyzer(_make_line_config())


class TestLineAnalyzerInstantiation:
    """Construction tests."""

    def test_accepts_line_config_object(self):
        analyzer = LineAnalyzer(_make_line_config("my_line"))
        assert analyzer.line_config.name == "my_line"

    def test_accepts_metric_suffix(self):
        analyzer = LineAnalyzer(_make_line_config(), metric_suffix="v2")
        assert analyzer.metric_suffix == "v2"


class TestLineAnalyzerScalars:
    """Scalar presence and finiteness."""

    EXPECTED = [
        "CoM",
        "rms",
        "fwhm",
        "peak_location",
        "integrated_intensity",
        "peak_value",
    ]

    def test_all_scalars_present(self, analyzer):
        data = _make_data()
        result = analyzer.analyze_image(data)
        for key in self.EXPECTED:
            full_key = f"test_line_{key}"
            assert full_key in result.scalars, f"Missing: {full_key}"

    def test_all_scalars_finite(self, analyzer):
        data = _make_data()
        result = analyzer.analyze_image(data)
        for key in self.EXPECTED:
            full_key = f"test_line_{key}"
            assert math.isfinite(result.scalars[full_key]), f"Non-finite: {full_key}"


class TestLineAnalyzerCentroidAccuracy:
    """CoM accuracy against known ground truth."""

    @pytest.mark.parametrize("center", [2.5, 5.0, 7.5])
    def test_com_matches_ground_truth(self, center):
        """CoM should recover the known peak centre to within 5% of sigma."""
        data = _make_data(center=center, sigma=1.0)
        analyzer = LineAnalyzer(_make_line_config())
        result = analyzer.analyze_image(data)
        com = result.scalars["test_line_CoM"]
        assert abs(com - center) < 0.1, f"CoM={com:.3f}, expected ~{center}"

    def test_peak_value_matches_amplitude(self, analyzer):
        """Peak value should be close to the amplitude passed to the generator."""
        data = _make_data()
        result = analyzer.analyze_image(data)
        assert result.scalars["test_line_peak_value"] == pytest.approx(1000.0, rel=0.01)

    def test_higher_amplitude_gives_higher_integrated_intensity(self):
        x = np.linspace(0.0, 10.0, 200)
        y_low = gaussian_peak_1d(x, center=5.0, sigma=1.0, amplitude=100.0, seed=0)
        y_high = gaussian_peak_1d(x, center=5.0, sigma=1.0, amplitude=5000.0, seed=0)
        analyzer = LineAnalyzer(_make_line_config())
        r_low = analyzer.analyze_image(np.column_stack([x, y_low]))
        r_high = analyzer.analyze_image(np.column_stack([x, y_high]))
        assert (
            r_high.scalars["test_line_integrated_intensity"]
            > r_low.scalars["test_line_integrated_intensity"]
        )


class TestLineAnalyzerResult:
    """Result structure tests."""

    def test_data_type_is_1d(self, analyzer):
        result = analyzer.analyze_image(_make_data())
        assert result.data_type == "1d"

    def test_line_data_is_nx2(self, analyzer):
        result = analyzer.analyze_image(_make_data())
        assert result.line_data is not None
        assert result.line_data.ndim == 2
        assert result.line_data.shape[1] == 2

    def test_metric_suffix_appended(self):
        analyzer = LineAnalyzer(_make_line_config("line"), metric_suffix="cal")
        result = analyzer.analyze_image(_make_data())
        assert "line_CoM_cal" in result.scalars
