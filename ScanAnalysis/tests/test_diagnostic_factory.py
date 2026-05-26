"""Tests for the diagnostic-factory: DiagnosticAnalysisConfig → ScanAnalyzer."""

from __future__ import annotations

import pytest

from image_analysis.config import DiagnosticAnalysisConfig
from image_analysis.config.array1d_processing import Line1DConfig
from image_analysis.config.array2d_processing import CameraConfig
from scan_analysis.analyzers.common.array1d_scan_analysis import Array1DScanAnalyzer
from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
from scan_analysis.config.diagnostic_factory import create_scan_analyzer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


# Pre-baked image_analyzer specs for the analyzers exercised in these
# tests. Bare class-path strings default to camera + array2d; verbose
# dicts override for 1D and the HASO no-image case.
_BEAM = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"
_STANDARD_1D = {
    "class_path": "image_analysis.analyzers.standard_1d_analyzer.Standard1DAnalyzer",
    "image_kind": "line",
    "scan_type": "array1d",
}
_HASO = {
    "class_path": "image_analysis.analyzers.HASO_himg_has_processor.HASOHimgHasProcessor",
    "image_kind": "none",
    "scan_type": "array2d",
}

_SPECS_BY_ALIAS = {"beam": _BEAM, "standard_1d": _STANDARD_1D, "haso": _HASO}


def _diag(
    *,
    name="UC_Test",
    alias="beam",
    image=None,
    scan=None,
) -> DiagnosticAnalysisConfig:
    """Build a minimal DiagnosticAnalysisConfig for factory tests.

    ``alias`` here is a test-fixture shorthand for the
    ``image_analyzer`` spec; it has nothing to do with the (now
    deleted) on-disk alias registry.
    """
    if image is None and alias != "haso":
        image = {"bit_depth": 16}
    return DiagnosticAnalysisConfig(
        name=name,
        image_analyzer=_SPECS_BY_ALIAS[alias],
        image=image,
        scan=scan or {},
    )


# ---------------------------------------------------------------------------
# Image-section validation (step 1)
# ---------------------------------------------------------------------------


class TestEmbeddedImageSection:
    """Camera/line analyzers consume the image: section; HASO refuses one."""

    def test_camera_alias_produces_validated_camera_config(self):
        analyzer = create_scan_analyzer(_diag(alias="beam"))
        # BeamAnalyzer stores its CameraConfig on self.camera_config
        assert isinstance(analyzer.image_analyzer.camera_config, CameraConfig)
        assert analyzer.image_analyzer.camera_config.name == "UC_Test"
        assert analyzer.image_analyzer.camera_config.bit_depth == 16

    def test_line_alias_produces_validated_line_config(self):
        diag = _diag(
            alias="standard_1d",
            image={"description": "test", "data_loading": {"data_type": "csv"}},
        )
        analyzer = create_scan_analyzer(diag)
        assert isinstance(analyzer.image_analyzer.line_config, Line1DConfig)
        assert analyzer.image_analyzer.line_config.name == "UC_Test"

    def test_haso_alias_with_image_section_raises(self):
        diag = _diag(
            alias="haso",
            image={"bit_depth": 16},  # forbidden for image_kind=none
        )
        with pytest.raises(ValueError, match="image_kind='none'"):
            create_scan_analyzer(diag)


# ---------------------------------------------------------------------------
# Scan-wrapper selection (step 3)
# ---------------------------------------------------------------------------


class TestScanWrapperSelection:
    """scan_type=array2d wraps in Array2DScanAnalyzer; array1d in Array1DScanAnalyzer."""

    def test_array2d_alias_produces_array2d_wrapper(self):
        analyzer = create_scan_analyzer(_diag(alias="beam"))
        assert isinstance(analyzer, Array2DScanAnalyzer)

    def test_array1d_alias_produces_array1d_wrapper(self):
        diag = _diag(
            alias="standard_1d",
            image={"description": "x", "data_loading": {"data_type": "csv"}},
        )
        analyzer = create_scan_analyzer(diag)
        assert isinstance(analyzer, Array1DScanAnalyzer)


# ---------------------------------------------------------------------------
# Scan runtime config mapping
# ---------------------------------------------------------------------------


class TestScanRuntimeAttachment:
    """id/priority kwargs override defaults; scan.gdoc_slot reaches the wrapper."""

    def test_id_defaults_to_name(self):
        analyzer = create_scan_analyzer(_diag(name="UC_Foo"))
        assert analyzer.id == "UC_Foo"

    def test_id_kwarg_overrides_name(self):
        analyzer = create_scan_analyzer(_diag(name="UC_Foo"), id="MyDiag")
        assert analyzer.id == "MyDiag"

    def test_priority_defaults_to_scan_priority(self):
        analyzer = create_scan_analyzer(_diag(scan={"priority": 7}))
        assert analyzer.priority == 7

    def test_priority_kwarg_overrides_scan_priority(self):
        analyzer = create_scan_analyzer(_diag(scan={"priority": 7}), priority=99)
        assert analyzer.priority == 99

    def test_gdoc_slot_attached(self):
        analyzer = create_scan_analyzer(_diag(scan={"gdoc_slot": 2}))
        assert analyzer.gdoc_slot == 2

    def test_gdoc_slot_default_is_none(self):
        analyzer = create_scan_analyzer(_diag())
        assert analyzer.gdoc_slot is None

    def test_save_maps_to_flag_save_images_for_array2d(self):
        on = create_scan_analyzer(_diag(scan={"save": True}))
        off = create_scan_analyzer(_diag(scan={"save": False}))
        assert on.flag_save_data is True  # base attr is flag_save_data
        assert off.flag_save_data is False

    def test_save_maps_to_flag_save_data_for_array1d(self):
        line_image = {"description": "x", "data_loading": {"data_type": "csv"}}
        on = create_scan_analyzer(
            _diag(alias="standard_1d", image=line_image, scan={"save": True})
        )
        off = create_scan_analyzer(
            _diag(alias="standard_1d", image=line_image, scan={"save": False})
        )
        assert on.flag_save_data is True
        assert off.flag_save_data is False

    def test_analysis_mode_passed_through(self):
        analyzer = create_scan_analyzer(_diag(scan={"mode": "per_bin"}))
        assert analyzer.analysis_mode == "per_bin"

    def test_device_override_changes_device_name(self):
        analyzer = create_scan_analyzer(
            _diag(name="UC_Logical", scan={"device": "UC_DataFolder"})
        )
        assert analyzer.device_name == "UC_DataFolder"

    def test_no_device_override_uses_top_level_name(self):
        analyzer = create_scan_analyzer(_diag(name="UC_Same"))
        assert analyzer.device_name == "UC_Same"

    def test_file_tail_passed_through_when_set(self):
        analyzer = create_scan_analyzer(_diag(scan={"file_tail": ".himg"}))
        assert analyzer.file_tail == ".himg"


class TestBackgroundSourceAttachment:
    """The scan.background_source directive is attached to the wrapper."""

    def test_default_is_none(self):
        analyzer = create_scan_analyzer(_diag())
        assert analyzer.background_source is None

    def test_scan_number_directive_attached(self):
        analyzer = create_scan_analyzer(
            _diag(scan={"background_source": {"scan_number": 5}})
        )
        assert analyzer.background_source is not None
        assert analyzer.background_source.scan_number == 5
        assert analyzer.background_source.from_current_scan is None

    def test_from_current_scan_directive_attached(self):
        analyzer = create_scan_analyzer(
            _diag(
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
