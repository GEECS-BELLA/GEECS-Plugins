"""Tests for the diagnostic-factory: ResolvedDiagnosticConfig → ScanAnalyzer."""

from __future__ import annotations

import pytest

from image_analysis.processing.array1d.config_models import Line1DConfig
from image_analysis.processing.array2d.config_models import CameraConfig
from scan_analysis.analyzers.common.array1d_scan_analysis import Array1DScanAnalyzer
from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
from scan_analysis.config.diagnostic_factory import create_diagnostic_analyzer
from scan_analysis.config.diagnostic_models import (
    DiagnosticAnalysisConfig,
    ResolvedDiagnosticConfig,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _resolved(
    *,
    name="UC_Test",
    alias="beam",
    image=None,
    scan=None,
    diagnostic_id="UC_Test",
    priority=42,
) -> ResolvedDiagnosticConfig:
    """Build a ResolvedDiagnosticConfig wrapping a minimal DiagnosticAnalysisConfig."""
    if image is None and alias != "haso":
        image = {"bit_depth": 16}
    return ResolvedDiagnosticConfig(
        id=diagnostic_id,
        enabled=True,
        priority=priority,
        diagnostic=DiagnosticAnalysisConfig(
            name=name,
            image_analyzer=alias,
            image=image,
            scan=scan or {},
        ),
    )


# ---------------------------------------------------------------------------
# Image-section validation (step 1)
# ---------------------------------------------------------------------------


class TestEmbeddedImageSection:
    """Camera/line analyzers consume the image: section; HASO refuses one."""

    def test_camera_alias_produces_validated_camera_config(self):
        analyzer = create_diagnostic_analyzer(_resolved(alias="beam"))
        # BeamAnalyzer stores its CameraConfig on self.camera_config
        assert isinstance(analyzer.image_analyzer.camera_config, CameraConfig)
        assert analyzer.image_analyzer.camera_config.name == "UC_Test"
        assert analyzer.image_analyzer.camera_config.bit_depth == 16

    def test_line_alias_produces_validated_line_config(self):
        resolved = _resolved(
            alias="standard_1d",
            image={"description": "test", "data_loading": {"data_type": "csv"}},
        )
        analyzer = create_diagnostic_analyzer(resolved)
        assert isinstance(analyzer.image_analyzer.line_config, Line1DConfig)
        assert analyzer.image_analyzer.line_config.name == "UC_Test"

    def test_haso_alias_with_image_section_raises(self):
        resolved = _resolved(
            alias="haso",
            image={"bit_depth": 16},  # forbidden for image_kind=none
        )
        with pytest.raises(ValueError, match="image_kind='none'"):
            create_diagnostic_analyzer(resolved)


# ---------------------------------------------------------------------------
# Scan-wrapper selection (step 3)
# ---------------------------------------------------------------------------


class TestScanWrapperSelection:
    """scan_type=array2d wraps in Array2DScanAnalyzer; array1d in Array1DScanAnalyzer."""

    def test_array2d_alias_produces_array2d_wrapper(self):
        analyzer = create_diagnostic_analyzer(_resolved(alias="beam"))
        assert isinstance(analyzer, Array2DScanAnalyzer)

    def test_array1d_alias_produces_array1d_wrapper(self):
        resolved = _resolved(
            alias="standard_1d",
            image={"description": "x", "data_loading": {"data_type": "csv"}},
        )
        analyzer = create_diagnostic_analyzer(resolved)
        assert isinstance(analyzer, Array1DScanAnalyzer)


# ---------------------------------------------------------------------------
# Scan runtime config mapping
# ---------------------------------------------------------------------------


class TestScanRuntimeAttachment:
    """Resolved id/priority + scan.gdoc_slot get attached to the wrapper."""

    def test_id_and_priority_attached(self):
        analyzer = create_diagnostic_analyzer(
            _resolved(diagnostic_id="MyDiag", priority=7)
        )
        assert analyzer.id == "MyDiag"
        assert analyzer.priority == 7

    def test_gdoc_slot_attached(self):
        analyzer = create_diagnostic_analyzer(_resolved(scan={"gdoc_slot": 2}))
        assert analyzer.gdoc_slot == 2

    def test_gdoc_slot_default_is_none(self):
        analyzer = create_diagnostic_analyzer(_resolved())
        assert analyzer.gdoc_slot is None

    def test_save_maps_to_flag_save_images_for_array2d(self):
        on = create_diagnostic_analyzer(_resolved(scan={"save": True}))
        off = create_diagnostic_analyzer(_resolved(scan={"save": False}))
        assert on.flag_save_data is True  # base attr is flag_save_data
        assert off.flag_save_data is False

    def test_save_maps_to_flag_save_data_for_array1d(self):
        line_image = {"description": "x", "data_loading": {"data_type": "csv"}}
        on = create_diagnostic_analyzer(
            _resolved(alias="standard_1d", image=line_image, scan={"save": True})
        )
        off = create_diagnostic_analyzer(
            _resolved(alias="standard_1d", image=line_image, scan={"save": False})
        )
        assert on.flag_save_data is True
        assert off.flag_save_data is False

    def test_analysis_mode_passed_through(self):
        analyzer = create_diagnostic_analyzer(_resolved(scan={"mode": "per_bin"}))
        assert analyzer.analysis_mode == "per_bin"

    def test_device_override_changes_device_name(self):
        analyzer = create_diagnostic_analyzer(
            _resolved(name="UC_Logical", scan={"device": "UC_DataFolder"})
        )
        assert analyzer.device_name == "UC_DataFolder"

    def test_no_device_override_uses_top_level_name(self):
        analyzer = create_diagnostic_analyzer(_resolved(name="UC_Same"))
        assert analyzer.device_name == "UC_Same"

    def test_file_tail_passed_through_when_set(self):
        analyzer = create_diagnostic_analyzer(_resolved(scan={"file_tail": ".himg"}))
        assert analyzer.file_tail == ".himg"


class TestBackgroundSourceAttachment:
    """The scan.background_source directive is attached to the wrapper."""

    def test_default_is_none(self):
        analyzer = create_diagnostic_analyzer(_resolved())
        assert analyzer.background_source is None

    def test_scan_number_directive_attached(self):
        analyzer = create_diagnostic_analyzer(
            _resolved(scan={"background_source": {"scan_number": 5}})
        )
        assert analyzer.background_source is not None
        assert analyzer.background_source.scan_number == 5
        assert analyzer.background_source.from_current_scan is None

    def test_from_current_scan_directive_attached(self):
        analyzer = create_diagnostic_analyzer(
            _resolved(
                scan={
                    "background_source": {
                        "from_current_scan": {
                            "method": "percentile",
                            "percentile": 5,
                        }
                    }
                }
            )
        )
        assert analyzer.background_source is not None
        assert analyzer.background_source.scan_number is None
        assert analyzer.background_source.from_current_scan.method == "percentile"
        assert analyzer.background_source.from_current_scan.percentile == 5
