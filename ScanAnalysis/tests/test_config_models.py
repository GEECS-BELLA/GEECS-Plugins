"""Unit tests for ScanAnalysis Pydantic config models.

No external data or hardware required — runs anywhere.
"""

import pytest
from pydantic import ValidationError

from scan_analysis.config.analyzer_config_models import (
    Array1DAnalyzerConfig,
    Array2DAnalyzerConfig,
    ExperimentAnalysisConfig,
    ImageAnalyzerConfig,
    IncludeEntry,
    PlotParameterConfig,
    ScatterAnalyzerConfig,
)


# ---------------------------------------------------------------------------
# ImageAnalyzerConfig
# ---------------------------------------------------------------------------


class TestImageAnalyzerConfig:
    def test_valid_config(self):
        cfg = ImageAnalyzerConfig(
            analyzer_class="image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer",
            camera_config_name="UC_GaiaMode",
        )
        assert cfg.analyzer_class.endswith("BeamAnalyzer")
        assert cfg.camera_config_name == "UC_GaiaMode"
        assert cfg.kwargs == {}

    def test_kwargs_passthrough(self):
        cfg = ImageAnalyzerConfig(
            analyzer_class="image_analysis.offline_analyzers.standard_1d_analyzer.Standard1DAnalyzer",
            kwargs={"data_type": "tdms"},
        )
        assert cfg.kwargs["data_type"] == "tdms"

    def test_invalid_class_no_dot(self):
        with pytest.raises(ValidationError, match="fully qualified"):
            ImageAnalyzerConfig(analyzer_class="BeamAnalyzer")

    def test_invalid_class_empty(self):
        with pytest.raises(ValidationError):
            ImageAnalyzerConfig(analyzer_class="")


# ---------------------------------------------------------------------------
# Array2DAnalyzerConfig
# ---------------------------------------------------------------------------


class TestArray2DAnalyzerConfig:
    def _make_image_cfg(
        self, cls="image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer"
    ):
        return ImageAnalyzerConfig(analyzer_class=cls, camera_config_name="UC_GaiaMode")

    def test_defaults(self):
        cfg = Array2DAnalyzerConfig(
            device_name="UC_GaiaMode",
            image_analyzer=self._make_image_cfg(),
        )
        assert cfg.priority == 100
        assert cfg.is_active is True
        assert cfg.flag_save_images is True
        assert cfg.analysis_mode == "per_shot"
        assert cfg.gdoc_slot is None

    def test_id_defaults_to_device_name(self):
        cfg = Array2DAnalyzerConfig(
            device_name="UC_GaiaMode",
            image_analyzer=self._make_image_cfg(),
        )
        assert cfg.id == "UC_GaiaMode"

    def test_explicit_id(self):
        cfg = Array2DAnalyzerConfig(
            id="custom_id",
            device_name="UC_GaiaMode",
            image_analyzer=self._make_image_cfg(),
        )
        assert cfg.id == "custom_id"

    def test_priority_bounds(self):
        with pytest.raises(ValidationError):
            Array2DAnalyzerConfig(
                device_name="UC_GaiaMode",
                priority=-1,
                image_analyzer=self._make_image_cfg(),
            )

    def test_gdoc_slot_bounds(self):
        with pytest.raises(ValidationError):
            Array2DAnalyzerConfig(
                device_name="UC_GaiaMode",
                gdoc_slot=4,
                image_analyzer=self._make_image_cfg(),
            )

    def test_gdoc_slot_valid(self):
        for slot in range(4):
            cfg = Array2DAnalyzerConfig(
                device_name="UC_GaiaMode",
                gdoc_slot=slot,
                image_analyzer=self._make_image_cfg(),
            )
            assert cfg.gdoc_slot == slot

    def test_inactive(self):
        cfg = Array2DAnalyzerConfig(
            device_name="UC_GaiaMode",
            is_active=False,
            image_analyzer=self._make_image_cfg(),
        )
        assert cfg.is_active is False


# ---------------------------------------------------------------------------
# Array1DAnalyzerConfig
# ---------------------------------------------------------------------------


class TestArray1DAnalyzerConfig:
    def _make_image_cfg(self):
        return ImageAnalyzerConfig(
            analyzer_class="image_analysis.offline_analyzers.standard_1d_analyzer.Standard1DAnalyzer",
        )

    def test_defaults(self):
        cfg = Array1DAnalyzerConfig(
            device_name="U_BCaveICT",
            image_analyzer=self._make_image_cfg(),
        )
        assert cfg.priority == 100
        assert cfg.is_active is True
        assert cfg.flag_save_data is True

    def test_id_defaults_to_device_name(self):
        cfg = Array1DAnalyzerConfig(
            device_name="U_BCaveICT",
            image_analyzer=self._make_image_cfg(),
        )
        assert cfg.id == "U_BCaveICT"


# ---------------------------------------------------------------------------
# ExperimentAnalysisConfig
# ---------------------------------------------------------------------------


class TestExperimentAnalysisConfig:
    def _make_2d_cfg(self, device="UC_GaiaMode", priority=0):
        return Array2DAnalyzerConfig(
            device_name=device,
            priority=priority,
            image_analyzer=ImageAnalyzerConfig(
                analyzer_class="image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer",
                camera_config_name=device,
            ),
        )

    def test_minimal_config(self):
        cfg = ExperimentAnalysisConfig(experiment="Undulator")
        assert cfg.experiment == "Undulator"
        assert cfg.analyzers == []

    def test_active_analyzers_filters_inactive(self):
        cfg = ExperimentAnalysisConfig(
            experiment="Undulator",
            analyzers=[
                self._make_2d_cfg("UC_GaiaMode"),
                Array2DAnalyzerConfig(
                    device_name="UC_Other",
                    is_active=False,
                    image_analyzer=ImageAnalyzerConfig(
                        analyzer_class="image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer",
                    ),
                ),
            ],
        )
        assert len(cfg.active_analyzers) == 1
        assert cfg.active_analyzers[0].device_name == "UC_GaiaMode"

    def test_get_analyzers_by_priority_sorted(self):
        cfg = ExperimentAnalysisConfig(
            experiment="Undulator",
            analyzers=[
                self._make_2d_cfg("UC_A", priority=50),
                self._make_2d_cfg("UC_B", priority=10),
                self._make_2d_cfg("UC_C", priority=100),
            ],
        )
        ordered = cfg.get_analyzers_by_priority()
        priorities = [a.priority for a in ordered]
        assert priorities == sorted(priorities)

    def test_get_analyzers_by_priority_max_filter(self):
        cfg = ExperimentAnalysisConfig(
            experiment="Undulator",
            analyzers=[
                self._make_2d_cfg("UC_A", priority=10),
                self._make_2d_cfg("UC_B", priority=50),
                self._make_2d_cfg("UC_C", priority=100),
            ],
        )
        filtered = cfg.get_analyzers_by_priority(max_priority=50)
        assert all(a.priority <= 50 for a in filtered)
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# IncludeEntry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PlotParameterConfig
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ScatterAnalyzerConfig
# ---------------------------------------------------------------------------


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

    def test_in_experiment_config_union(self):
        raw = {
            "experiment": "Undulator",
            "analyzers": [
                {
                    "type": "scatter",
                    "title": "Test",
                    "filename": "test",
                    "parameters": [{"key_name": "col_a", "color": "blue"}],
                }
            ],
        }
        exp = ExperimentAnalysisConfig.model_validate(raw)
        assert len(exp.analyzers) == 1
        assert isinstance(exp.analyzers[0], ScatterAnalyzerConfig)


class TestIncludeEntry:
    def test_ref_only(self):
        entry = IncludeEntry(ref="my_analyzer")
        assert entry.ref == "my_analyzer"
        assert entry.group is None

    def test_group_only(self):
        entry = IncludeEntry(group="my_group")
        assert entry.group == "my_group"
        assert entry.ref is None

    def test_both_raises(self):
        with pytest.raises(ValidationError):
            IncludeEntry(ref="a", group="b")

    def test_neither_raises(self):
        with pytest.raises(ValidationError):
            IncludeEntry()
