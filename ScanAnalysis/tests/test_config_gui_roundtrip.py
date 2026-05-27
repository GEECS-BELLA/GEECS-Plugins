"""Round-trip tests for the ConfigFileGUI config I/O layer + editors.

Two coverage tiers:

* **I/O layer** (``TestCameraConfigRoundtrip``, ``TestLine1DConfigRoundtrip``,
  ``TestFloatPrecisionRoundtrip``, ``TestDiagnosticAnalysisConfigRoundtrip``,
  ``TestAnalysisGroupConfigRoundtrip``) — load/save through
  ``ConfigFileGUI.config_io`` / ``scan_config_io`` only. No Qt or
  display required. Verifies the on-disk YAML round-trips to an
  equivalent typed Pydantic model.
* **Editor layer** (``TestGroupEditorRoundtrip``) — load through the
  full Qt editor and back, exercising the form widgets. Uses the
  ``offscreen`` Qt platform so no display is needed.

The "spirit" of the original test (no silent precision loss) is
preserved by ``TestFloatPrecisionRoundtrip`` and
``TestDiagnosticAnalysisConfigRoundtrip.test_scientific_notation_...``.
"""

from __future__ import annotations

import math

import pytest

from ConfigFileGUI.config_io import load_config, save_config
from image_analysis.config.array2d_processing import CameraConfig
from image_analysis.config.array1d_processing import Line1DConfig, Data1DConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roundtrip_camera(config: CameraConfig, tmp_path) -> CameraConfig:
    path = tmp_path / f"{config.name}.yaml"
    save_config(config, path)
    return load_config(path)


def _roundtrip_line(config: Line1DConfig, tmp_path) -> Line1DConfig:
    path = tmp_path / f"{config.name}.yaml"
    save_config(config, path)
    return load_config(path)


# ---------------------------------------------------------------------------
# CameraConfig round-trips
# ---------------------------------------------------------------------------


class TestCameraConfigRoundtrip:
    def test_minimal_camera_config(self, tmp_path):
        cfg = CameraConfig(name="UC_TestCamera", bit_depth=16)
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert reloaded.name == cfg.name
        assert reloaded.bit_depth == cfg.bit_depth

    def test_name_survives_roundtrip(self, tmp_path):
        cfg = CameraConfig(name="MyCustomName", bit_depth=8)
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert reloaded.name == "MyCustomName"

    def test_description_survives_roundtrip(self, tmp_path):
        cfg = CameraConfig(name="UC_Test", bit_depth=16, description="A test camera")
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert reloaded.description == "A test camera"

    def test_name_independent_of_filename(self, tmp_path):
        cfg = CameraConfig(name="DeviceName", bit_depth=16)
        path = tmp_path / "DifferentFilename.yaml"
        save_config(cfg, path)
        reloaded = load_config(path)
        assert reloaded.name == "DeviceName"

    def test_model_dump_identical(self, tmp_path):
        cfg = CameraConfig(name="UC_Test", bit_depth=16, description="desc")
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert reloaded.model_dump() == cfg.model_dump()


# ---------------------------------------------------------------------------
# Float precision round-trips (scientific notation)
# ---------------------------------------------------------------------------


class TestFloatPrecisionRoundtrip:
    def test_normal_float_survives(self, tmp_path):
        from image_analysis.config.array2d_processing import ThresholdingConfig

        cfg = CameraConfig(
            name="UC_Test",
            bit_depth=16,
            thresholding=ThresholdingConfig(enabled=True, value=0.2008),
        )
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert math.isclose(reloaded.thresholding.value, 0.2008, rel_tol=1e-9)

    def test_scientific_notation_float_survives(self, tmp_path):
        from image_analysis.config.array2d_processing import ThresholdingConfig

        tiny = 1e-13
        cfg = CameraConfig(
            name="UC_Test",
            bit_depth=16,
            thresholding=ThresholdingConfig(enabled=True, value=tiny),
        )
        reloaded = _roundtrip_camera(cfg, tmp_path)
        assert math.isclose(reloaded.thresholding.value, tiny, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Line1DConfig round-trips
# ---------------------------------------------------------------------------


class TestLine1DConfigRoundtrip:
    def test_minimal_line_config(self, tmp_path):
        from image_analysis.config.array1d_processing import Data1DType

        cfg = Line1DConfig(
            name="U_TestSignal",
            data_loading=Data1DConfig(data_type=Data1DType.CSV),
        )
        reloaded = _roundtrip_line(cfg, tmp_path)
        assert reloaded.name == cfg.name

    def test_name_survives_roundtrip(self, tmp_path):
        from image_analysis.config.array1d_processing import Data1DType

        cfg = Line1DConfig(
            name="MySignalName",
            data_loading=Data1DConfig(data_type=Data1DType.CSV),
        )
        reloaded = _roundtrip_line(cfg, tmp_path)
        assert reloaded.name == "MySignalName"

    def test_model_dump_identical(self, tmp_path):
        from image_analysis.config.array1d_processing import Data1DType

        cfg = Line1DConfig(
            name="U_Test",
            description="line desc",
            data_loading=Data1DConfig(data_type=Data1DType.CSV),
        )
        reloaded = _roundtrip_line(cfg, tmp_path)
        assert reloaded.model_dump() == cfg.model_dump()


# ---------------------------------------------------------------------------
# DiagnosticAnalysisConfig round-trips (the post-PR-E unified diagnostic)
# ---------------------------------------------------------------------------


class TestDiagnosticAnalysisConfigRoundtrip:
    """Round-trip a unified diagnostic YAML through the I/O layer.

    Tests the ``scan_config_io`` load/save pair (no editor in the loop)
    against ``DiagnosticAnalysisConfig`` validation. The editor's
    own behaviour is partially covered by
    ``TestScanAnalyzerEditorRoundtrip`` below.
    """

    def _build_data(self, **overrides):
        """Return a minimal diagnostic dict, merge ``overrides`` shallow."""
        base = {
            "name": "UC_Test",
            "image_analyzer": ("image_analysis.analyzers.beam_analyzer.BeamAnalyzer"),
            "image": {"type": "camera", "bit_depth": 16},
            "scan": {"priority": 50, "mode": "per_shot"},
        }
        base.update(overrides)
        return base

    def _roundtrip(self, data: dict, tmp_path):
        from ConfigFileGUI.scan_config_io import (
            load_analyzer_yaml,
            save_analyzer_yaml,
        )

        path = tmp_path / "diag.yaml"
        save_analyzer_yaml(path, data)
        return load_analyzer_yaml(path)

    def test_minimal_camera_diagnostic_roundtrip(self, tmp_path):
        from image_analysis.config import DiagnosticAnalysisConfig

        data = self._build_data()
        reloaded = self._roundtrip(data, tmp_path)
        m_in = DiagnosticAnalysisConfig.model_validate(data)
        m_out = DiagnosticAnalysisConfig.model_validate(reloaded)
        assert m_in == m_out

    def test_line_diagnostic_roundtrip(self, tmp_path):
        from image_analysis.config import DiagnosticAnalysisConfig

        data = self._build_data(
            name="U_Line",
            image_analyzer=(
                "image_analysis.analyzers.standard_1d_analyzer.Standard1DAnalyzer"
            ),
            image={
                "type": "line",
                "data_loading": {"data_type": "tdms_scope"},
            },
        )
        reloaded = self._roundtrip(data, tmp_path)
        m_in = DiagnosticAnalysisConfig.model_validate(data)
        m_out = DiagnosticAnalysisConfig.model_validate(reloaded)
        assert m_in == m_out

    def test_no_image_diagnostic_roundtrip(self, tmp_path):
        """HASO-style diagnostics omit the ``image:`` section entirely."""
        from image_analysis.config import DiagnosticAnalysisConfig

        data = {
            "name": "HasoLift",
            "image_analyzer": {
                "class_path": (
                    "image_analysis.analyzers.HASO_himg_has_processor."
                    "HASOHimgHasProcessor"
                ),
                "kwargs": {"mask_top": 125},
            },
            "scan": {"priority": 10, "save": True},
        }
        reloaded = self._roundtrip(data, tmp_path)
        m_in = DiagnosticAnalysisConfig.model_validate(data)
        m_out = DiagnosticAnalysisConfig.model_validate(reloaded)
        assert m_in == m_out

    def test_scientific_notation_in_diagnostic_roundtrip(self, tmp_path):
        """Tiny float values must survive — same spirit as the bare CameraConfig test."""
        from image_analysis.config import DiagnosticAnalysisConfig

        tiny = 1e-13
        data = self._build_data(
            image={
                "type": "camera",
                "bit_depth": 16,
                "thresholding": {"enabled": True, "value": tiny},
            },
        )
        reloaded = self._roundtrip(data, tmp_path)
        m_in = DiagnosticAnalysisConfig.model_validate(data)
        m_out = DiagnosticAnalysisConfig.model_validate(reloaded)
        assert m_in == m_out
        # Belt-and-suspenders: spot-check the tiny value at the image
        # level — model equality covers it, but if the test ever fails
        # this makes the cause obvious.
        assert math.isclose(m_out.image.thresholding.value, tiny, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# AnalysisGroupConfig round-trips (the post-PR-E per-file groups)
# ---------------------------------------------------------------------------


class TestAnalysisGroupConfigRoundtrip:
    """Round-trip a group YAML through the I/O layer."""

    def _roundtrip(self, data: dict, tmp_path):
        from ConfigFileGUI.scan_config_io import (
            load_group_yaml,
            save_group_yaml,
        )

        path = tmp_path / "group.yaml"
        save_group_yaml(path, data)
        return load_group_yaml(path)

    def test_bare_string_analyzers_roundtrip(self, tmp_path):
        """Plain-string analyzer refs survive verbatim."""
        from scan_analysis.config.diagnostic_models import AnalysisGroupConfig

        data = {
            "name": "HTU_baseline",
            "analyzers": ["UC_GaiaMode", "Amp4Input", "Amp4Output"],
        }
        reloaded = self._roundtrip(data, tmp_path)
        assert reloaded == data
        m_in = AnalysisGroupConfig.model_validate(data)
        m_out = AnalysisGroupConfig.model_validate(reloaded)
        assert m_in == m_out

    def test_mixed_ref_forms_roundtrip(self, tmp_path):
        """Bare strings + dict entries (with overrides) coexist cleanly."""
        from scan_analysis.config.diagnostic_models import AnalysisGroupConfig

        data = {
            "name": "HTU_mixed",
            "analyzers": [
                "Amp4Input",
                {"ref": "Amp4Output", "enabled": False},
                {"ref": "UC_TopView", "priority": 5},
            ],
        }
        reloaded = self._roundtrip(data, tmp_path)
        m_in = AnalysisGroupConfig.model_validate(data)
        m_out = AnalysisGroupConfig.model_validate(reloaded)
        assert m_in == m_out

    def test_description_and_upload_flag_roundtrip(self, tmp_path):
        """Top-level optional fields survive."""
        from scan_analysis.config.diagnostic_models import AnalysisGroupConfig

        data = {
            "name": "HTU_test",
            "description": "Test description",
            "upload_to_scanlog": False,
            "analyzers": ["UC_GaiaMode"],
        }
        reloaded = self._roundtrip(data, tmp_path)
        m_in = AnalysisGroupConfig.model_validate(data)
        m_out = AnalysisGroupConfig.model_validate(reloaded)
        assert m_in == m_out


# ---------------------------------------------------------------------------
# Group editor (Qt) round-trip
# ---------------------------------------------------------------------------


class TestGroupEditorRoundtrip:
    """Round-trip a group YAML through the full editor (load + get).

    Uses the offscreen Qt platform so no display is required. Exercises
    the actual ``GroupEditorPanel`` form fields, not just the I/O.
    """

    @pytest.fixture(autouse=True)
    def _qt_app(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        yield app

    def _editor_roundtrip(self, data: dict) -> dict:
        from ConfigFileGUI.groups_editor import GroupEditorPanel

        editor = GroupEditorPanel()
        editor.load_config(data)
        return editor.get_config_dict()

    def test_bare_strings_survive_editor_roundtrip(self):
        """Bare-string analyzer refs stay bare-string after the editor."""
        from scan_analysis.config.diagnostic_models import AnalysisGroupConfig

        data = {
            "name": "HTU_simple",
            "analyzers": ["UC_GaiaMode", "Amp4Input"],
        }
        out = self._editor_roundtrip(data)
        assert out == data
        m_in = AnalysisGroupConfig.model_validate(data)
        m_out = AnalysisGroupConfig.model_validate(out)
        assert m_in == m_out

    def test_disabled_entry_survives_editor_roundtrip(self):
        """An ``enabled: false`` override stays disabled."""
        from scan_analysis.config.diagnostic_models import AnalysisGroupConfig

        data = {
            "name": "HTU_baseline",
            "analyzers": [
                "Amp4Input",
                {"ref": "Amp4Output", "enabled": False},
            ],
        }
        out = self._editor_roundtrip(data)
        m_in = AnalysisGroupConfig.model_validate(data)
        m_out = AnalysisGroupConfig.model_validate(out)
        assert m_in == m_out

    def test_priority_override_survives_editor_roundtrip(self):
        """A per-group priority override stays attached to the right ref."""
        from scan_analysis.config.diagnostic_models import AnalysisGroupConfig

        data = {
            "name": "HTU_priority",
            "analyzers": [{"ref": "UC_TopView", "priority": 5}],
        }
        out = self._editor_roundtrip(data)
        m_in = AnalysisGroupConfig.model_validate(data)
        m_out = AnalysisGroupConfig.model_validate(out)
        assert m_in == m_out
