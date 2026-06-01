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


@pytest.mark.gui
class TestGroupEditorRoundtrip:
    """Round-trip a group YAML through the full editor (load + get).

    Uses the offscreen Qt platform so no display is required. Exercises
    the actual ``GroupEditorPanel`` form fields, not just the I/O.

    Marked ``gui`` so headless CI (which runs ``pytest -m "not integration
    and not gui"``) deselects them — PyQt5 is not installed in that env.
    """

    @pytest.fixture(autouse=True)
    def _qt_app(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PyQt5")
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


# ---------------------------------------------------------------------------
# Scan analyzer editor (Qt) round-trip
# ---------------------------------------------------------------------------


@pytest.mark.gui
class TestScanAnalyzerEditorRoundtrip:
    """Round-trip a diagnostic YAML through the full scan-analyzer editor.

    Focused on fields the editor adds beyond the I/O layer's coverage —
    primarily the General-section ``metric_prefix`` / ``metric_suffix``
    decoration fields that were added when the prefix/suffix concept
    moved from ImageAnalysis to ScanAnalysis.
    """

    @pytest.fixture(autouse=True)
    def _qt_app(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PyQt5")
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        yield app

    def _editor_roundtrip(self, data: dict) -> dict:
        from ConfigFileGUI.scan_analyzer_editor import ScanAnalyzerEditorPanel

        editor = ScanAnalyzerEditorPanel()
        editor.load_config(data)
        return editor.get_config_dict()

    def test_metric_prefix_survives_editor_roundtrip(self):
        """An explicit ``metric_prefix`` makes it through load + save."""
        data = {
            "name": "UC_Test",
            "metric_prefix": "custom_prefix",
            "image_analyzer": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
            "image": {"type": "camera", "bit_depth": 16},
            "scan": {"priority": 50},
        }
        out = self._editor_roundtrip(data)
        assert out.get("metric_prefix") == "custom_prefix"

    def test_metric_suffix_survives_editor_roundtrip(self):
        """An explicit ``metric_suffix`` makes it through load + save."""
        data = {
            "name": "UC_Test",
            "metric_suffix": "_roi_left",
            "image_analyzer": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
            "image": {"type": "camera", "bit_depth": 16},
            "scan": {"priority": 50},
        }
        out = self._editor_roundtrip(data)
        assert out.get("metric_suffix") == "_roi_left"

    def test_absent_metric_keys_stay_absent(self):
        """A YAML without metric_prefix/suffix doesn't grow them on round-trip.

        Empty edits map to "field absent" — the schema defaults
        (``effective_metric_prefix → name``, suffix → "") then take
        over on consumption. We deliberately do NOT serialise empty
        strings, so the on-disk YAML stays clean.
        """
        data = {
            "name": "UC_Test",
            "image_analyzer": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
            "image": {"type": "camera", "bit_depth": 16},
            "scan": {"priority": 50},
        }
        out = self._editor_roundtrip(data)
        assert "metric_prefix" not in out
        assert "metric_suffix" not in out

    def test_full_diagnostic_roundtrip_with_decoration(self):
        """Both fields set + full diagnostic shape — validates as a
        :class:`DiagnosticAnalysisConfig` and equals the input."""
        from image_analysis.config import DiagnosticAnalysisConfig

        data = {
            "name": "UC_Test",
            "metric_prefix": "custom_pref",
            "metric_suffix": "_v2",
            "image_analyzer": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
            "image": {"type": "camera", "bit_depth": 16},
            "scan": {"priority": 50},
        }
        out = self._editor_roundtrip(data)
        m_in = DiagnosticAnalysisConfig.model_validate(data)
        m_out = DiagnosticAnalysisConfig.model_validate(out)
        assert m_in.metric_prefix == m_out.metric_prefix == "custom_pref"
        assert m_in.metric_suffix == m_out.metric_suffix == "_v2"
        assert m_in.effective_metric_prefix == "custom_pref"

    def test_image_section_does_not_emit_redundant_name(self):
        """Embedded ConfigEditorPanel hides the image-level Name row.

        After PR #420 the top-level ``DiagnosticAnalysisConfig.name`` is
        the source of truth (it's also the default ``metric_prefix``).
        The image-level ``name`` field is ``Optional[str]`` and gets
        injected by the validator on-load. The editor must not emit a
        redundant ``image.name`` on save — that's what created the
        confusing "two name fields" UX before this change.
        """
        data = {
            "name": "UC_Test",
            "image_analyzer": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
            "image": {"type": "camera", "bit_depth": 16},
            "scan": {"priority": 50},
        }
        out = self._editor_roundtrip(data)
        assert "name" not in out.get("image", {}), (
            "image.name leaked into the YAML output — the embedded panel "
            "should run in embedded_mode=True and suppress the Name field."
        )

    def test_legacy_image_name_dropped_on_roundtrip(self):
        """A legacy YAML with redundant ``image.name`` loses it on save.

        Pre-PR #420 the editor injected the top-level name into
        ``image.name`` so it would render in the embedded form. After
        the embedded_mode switch, no field renders it, and the editor
        doesn't emit it. The ``DiagnosticAnalysisConfig`` validator
        still injects ``image.name = name`` at validate-time, so the
        resolved model is unchanged — but the on-disk shape gets
        cleaner.
        """
        from image_analysis.config import DiagnosticAnalysisConfig

        data = {
            "name": "UC_Test",
            "image_analyzer": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
            "image": {"type": "camera", "name": "UC_Test", "bit_depth": 16},
            "scan": {"priority": 50},
        }
        out = self._editor_roundtrip(data)
        assert "name" not in out.get("image", {})
        # Resolved-model equivalence: the validator re-injects
        # ``image.name = name``, so both shapes resolve identically.
        m_in = DiagnosticAnalysisConfig.model_validate(data)
        m_out = DiagnosticAnalysisConfig.model_validate(out)
        assert m_in.image.name == m_out.image.name == "UC_Test"
