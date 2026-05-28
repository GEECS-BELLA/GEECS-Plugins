"""Unit tests for ScanAnalysis scatter-config models.

After the unified-configs migration, image-analyzer-driven scan
analyzers (Array2D / Array1D) are tested via
``test_diagnostic_models.py`` and ``test_diagnostic_factory.py``.
This file covers only the scatter path, which stays on its own
config shape (see ``analyzer_config_models.py``).
"""

import pytest
from pydantic import ValidationError

from scan_analysis.config.analyzer_config_models import (
    PlotParameterConfig,
    ScatterAnalyzerConfig,
)


class TestPlotParameterConfig:
    def test_minimal(self):
        p = PlotParameterConfig(key_name="UC_ModeImager_x_rms")
        assert p.key_name == "UC_ModeImager_x_rms"
        assert p.label is None
        assert p.color == "blue"
        assert p.y_range is None

    def test_full(self):
        p = PlotParameterConfig(
            key_name="UC_ModeImager_x_rms",
            label="x RMS (px)",
            color="red",
            y_range=[0.0, 50.0],
        )
        assert p.label == "x RMS (px)"
        assert p.color == "red"
        assert p.y_range == [0.0, 50.0]

    def test_y_range_wrong_length_raises(self):
        with pytest.raises(ValidationError, match="2 elements"):
            PlotParameterConfig(key_name="x", y_range=[1.0, 2.0, 3.0])


class TestScatterAnalyzerConfig:
    def _make_param(self, key: str = "UC_ModeImager_x_rms") -> PlotParameterConfig:
        return PlotParameterConfig(key_name=key, label="x RMS", color="blue")

    def test_minimal(self):
        cfg = ScatterAnalyzerConfig(
            title="Test Plot",
            filename="test_plot",
            parameters=[self._make_param()],
        )
        assert cfg.type == "scatter"
        assert cfg.use_median is True
        assert cfg.priority == 200
        assert cfg.is_active is True
        assert cfg.gdoc_slot is None
        assert cfg.x_column is None

    def test_id_defaults_to_filename(self):
        cfg = ScatterAnalyzerConfig(
            title="T",
            filename="my_scatter",
            parameters=[self._make_param()],
        )
        assert cfg.id == "my_scatter"

    def test_explicit_id(self):
        cfg = ScatterAnalyzerConfig(
            id="custom_id",
            title="T",
            filename="f",
            parameters=[self._make_param()],
        )
        assert cfg.id == "custom_id"

    def test_x_column(self):
        cfg = ScatterAnalyzerConfig(
            title="T",
            filename="f",
            parameters=[self._make_param()],
            x_column="U_ModeImagerESP Position.Axis 1 Alias:ModeImager",
        )
        assert cfg.x_column == "U_ModeImagerESP Position.Axis 1 Alias:ModeImager"

    def test_parsed_from_dict(self):
        raw = {
            "type": "scatter",
            "title": "ModeImager",
            "filename": "mode_imager",
            "parameters": [
                {"key_name": "UC_ModeImager_x_rms", "label": "x RMS", "color": "blue"},
                {"key_name": "UC_ModeImage_y_rms", "color": "red"},
            ],
        }
        cfg = ScatterAnalyzerConfig.model_validate(raw)
        assert len(cfg.parameters) == 2
        assert cfg.parameters[1].label is None
