"""Tests for the diagnostic-factory: DiagnosticAnalysisConfig → ScanAnalyzer."""

from __future__ import annotations


from image_analysis.config import DiagnosticAnalysisConfig
from image_analysis.config.array1d_processing import Line1DConfig
from image_analysis.config.array2d_processing import CameraConfig
from scan_analysis.analyzers.common.array1d_scan_analysis import Array1DScanAnalyzer
from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
from scan_analysis.config.diagnostic_factory import create_scan_analyzer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


# Bare class-path strings for the analyzers exercised in these tests.
# The 2D-vs-1D dimension lives on the image: section's ``type`` field,
# not on the analyzer spec, so all of these are bare strings.
_BEAM = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"
_STANDARD_1D = "image_analysis.analyzers.standard_1d_analyzer.Standard1DAnalyzer"
_HASO = "image_analysis.analyzers.HASO_himg_has_processor.HASOHimgHasProcessor"

_SPECS_BY_ALIAS = {"beam": _BEAM, "standard_1d": _STANDARD_1D, "haso": _HASO}


def _diag(
    *,
    name="UC_Test",
    alias="beam",
    image=None,
    scan=None,
) -> DiagnosticAnalysisConfig:
    """Build a minimal DiagnosticAnalysisConfig for factory tests.

    ``alias`` is a test-fixture shorthand for picking which class path
    to use; it's not an on-disk alias registry (those were removed in
    PR-E). The default ``image:`` section matches the alias: camera
    for ``beam``, line for ``standard_1d``, omitted for ``haso``.
    """
    if image is None and alias == "beam":
        image = {"type": "camera", "bit_depth": 16}
    elif image is None and alias == "standard_1d":
        image = {"type": "line", "data_loading": {"data_type": "csv"}}
    # haso: no image section
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
            image={
                "type": "line",
                "description": "test",
                "data_loading": {"data_type": "csv"},
            },
        )
        analyzer = create_scan_analyzer(diag)
        assert isinstance(analyzer.image_analyzer.line_config, Line1DConfig)
        assert analyzer.image_analyzer.line_config.name == "UC_Test"


# ---------------------------------------------------------------------------
# Scan-wrapper selection (step 3)
# ---------------------------------------------------------------------------


class TestScanWrapperSelection:
    """Wrapper class is picked from the type of ``diag.image``."""

    def test_camera_image_produces_array2d_wrapper(self):
        analyzer = create_scan_analyzer(_diag(alias="beam"))
        assert isinstance(analyzer, Array2DScanAnalyzer)

    def test_line_image_produces_array1d_wrapper(self):
        analyzer = create_scan_analyzer(_diag(alias="standard_1d"))
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
        line_image = {
            "type": "line",
            "description": "x",
            "data_loading": {"data_type": "csv"},
        }
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
