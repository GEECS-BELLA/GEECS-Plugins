"""Tests for the diagnostic-factory: AnalysisDiagnostic → ScanAnalyzer."""

from __future__ import annotations

import yaml


from geecs_schemas.analysis import AnalysisDiagnostic
from image_analysis.config import load_diagnostic
from geecs_schemas.analysis.processing_1d import Line1DConfig
from geecs_schemas.analysis.processing_2d import CameraConfig
import pytest

from scan_analysis.analyzers.common.array1d_scan_analysis import Array1DScanAnalyzer
from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
from scan_analysis.config.diagnostic_factory import create_scan_analyzer
from scan_analysis.core_analyzer import CoreScanAnalyzer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


# The analyzer specs exercised in these tests, by a test-local alias. The
# 2D-vs-1D dimension lives on the image: section's ``type`` field.
_SPECS_BY_ALIAS = {
    "beam": {"kind": "beam"},
    "standard_1d": {"kind": "trace"},
    "haso": {"kind": "haso", "wavekit_config_file_path": "/wfs.dat"},
    "ict": {"kind": "ict"},
}


# Schema-valid features the analysis core refuses at compile time, so a
# test can pin the legacy wrappers' behaviour without changing the factory.
_LEGACY_CAMERA = {"pipeline": ["transforms"], "transforms": {"flip_horizontal": True}}
_LEGACY_LINE = {"pipeline": ["roi"], "roi": {"x_min": 0.0, "x_max": 1.0}}


def _diag(
    *,
    name="UC_Test",
    alias="beam",
    image=None,
    scan=None,
    legacy=False,
) -> AnalysisDiagnostic:
    """Build a minimal AnalysisDiagnostic for factory tests.

    ``alias`` is a test-fixture shorthand for picking the analyzer spec.
    The default ``image:`` section matches the alias: camera for ``beam``,
    line for ``standard_1d``, omitted for ``haso``. ``legacy=True`` adds an
    operation the core has not ported, forcing the legacy wrapper route.
    """
    if image is None and alias == "beam":
        image = {"type": "camera", "bit_depth": 16}
    elif image is None and alias == "standard_1d":
        image = {"type": "line", "data_loading": {"data_type": "csv"}}
    if legacy and image is not None:
        image = {
            **image,
            **(_LEGACY_LINE if image["type"] == "line" else _LEGACY_CAMERA),
        }
    # haso: no image section
    return AnalysisDiagnostic(
        name=name,
        analyzer=_SPECS_BY_ALIAS[alias],
        image=image,
        scan=scan or {},
    )


# ---------------------------------------------------------------------------
# Image-section validation (step 1)
# ---------------------------------------------------------------------------


class TestEmbeddedImageSection:
    """Camera/line analyzers consume the image: section; HASO refuses one."""

    def test_camera_alias_produces_validated_camera_config(self):
        analyzer = create_scan_analyzer(_diag(alias="beam", legacy=True))
        # BeamAnalyzer stores its CameraConfig on self.camera_config
        assert isinstance(analyzer.image_analyzer.camera_config, CameraConfig)
        assert analyzer.image_analyzer.camera_config.bit_depth == 16
        # Identity flows through the analyzer's output_name property
        # (#412 — CameraConfig.name is gone; output_name comes from the
        # diagnostic's effective_output_name).
        assert analyzer.image_analyzer.output_name == "UC_Test"

    def test_line_alias_produces_validated_line_config(self):
        diag = _diag(
            alias="standard_1d",
            image={
                "type": "line",
                "description": "test",
                "data_loading": {"data_type": "csv"},
            },
            legacy=True,
        )
        analyzer = create_scan_analyzer(diag)
        assert isinstance(analyzer.image_analyzer.line_config, Line1DConfig)
        assert analyzer.image_analyzer.output_name == "UC_Test"


# ---------------------------------------------------------------------------
# Scan-wrapper selection (step 3)
# ---------------------------------------------------------------------------


class TestScanWrapperSelection:
    """Supported recipes run on the core; the rest pick a wrapper by ``diag.image``."""

    def test_supported_camera_recipe_routes_to_the_core(self):
        analyzer = create_scan_analyzer(_diag(alias="beam"))
        assert isinstance(analyzer, CoreScanAnalyzer)

    def test_supported_line_recipe_routes_to_the_core(self):
        analyzer = create_scan_analyzer(_diag(alias="standard_1d"))
        assert isinstance(analyzer, CoreScanAnalyzer)

    def test_core_route_keeps_its_own_document_copy(self):
        diag = _diag(name="UC_Copy", scan={"mode": "per_bin", "save": False})
        analyzer = create_scan_analyzer(diag)
        assert analyzer.document == diag and analyzer.document is not diag
        assert (analyzer.document.scan.mode, analyzer.document.scan.save) == (
            "per_bin",
            False,
        )

    def test_unported_camera_step_produces_array2d_wrapper(self):
        analyzer = create_scan_analyzer(_diag(alias="beam", legacy=True))
        assert isinstance(analyzer, Array2DScanAnalyzer)

    def test_unported_line_step_produces_array1d_wrapper(self):
        analyzer = create_scan_analyzer(_diag(alias="standard_1d", legacy=True))
        assert isinstance(analyzer, Array1DScanAnalyzer)

    def test_scan_context_background_stays_on_the_wrapper(self):
        analyzer = create_scan_analyzer(
            _diag(scan={"background_source": {"scan_number": 5}})
        )
        assert isinstance(analyzer, Array2DScanAnalyzer)

    def test_unported_kind_stays_on_the_wrapper(self):
        analyzer = create_scan_analyzer(
            _diag(
                alias="ict",
                image={"type": "line", "data_loading": {"data_type": "csv"}},
            )
        )
        assert isinstance(analyzer, Array1DScanAnalyzer)

    def test_injected_data_stays_on_the_wrapper(self):
        analyzer = create_scan_analyzer(_diag(alias="beam"), use_injected_data=True)
        assert isinstance(analyzer, Array2DScanAnalyzer)
        assert analyzer.use_injected_data is True

    def test_route_legacy_forces_the_wrapper_for_a_supported_recipe(self):
        analyzer = create_scan_analyzer(_diag(alias="beam"), route="legacy")
        assert isinstance(analyzer, Array2DScanAnalyzer)

    def test_route_core_refuses_what_the_core_cannot_run(self):
        from geecs_analysis.compat.v2 import UnsupportedRecipe

        with pytest.raises(UnsupportedRecipe):
            create_scan_analyzer(_diag(alias="beam", legacy=True), route="core")
        with pytest.raises(ValueError, match="injected"):
            create_scan_analyzer(
                _diag(alias="beam"), route="core", use_injected_data=True
            )
        with pytest.raises(ValueError, match="route"):
            create_scan_analyzer(_diag(alias="beam"), route="fast")


# ---------------------------------------------------------------------------
# Scan runtime config mapping
# ---------------------------------------------------------------------------


@pytest.fixture(params=[False, True], ids=["core", "legacy"])
def legacy(request):
    """Run a contract test on both factory routes."""
    return request.param


class TestScanRuntimeAttachment:
    """id/priority kwargs override defaults on both routes; wrapper kwargs map."""

    def test_id_defaults_to_name(self, legacy):
        analyzer = create_scan_analyzer(_diag(name="UC_Foo", legacy=legacy))
        assert analyzer.id == "UC_Foo"

    def test_id_kwarg_overrides_name(self, legacy):
        analyzer = create_scan_analyzer(
            _diag(name="UC_Foo", legacy=legacy), id="MyDiag"
        )
        assert analyzer.id == "MyDiag"

    def test_id_defaults_to_loaded_diagnostic_source_id(self, tmp_path):
        path = tmp_path / "PW-MagSpectStitcher.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    "name": "CAM-TEA-MagSpecA-interpSpec",
                    "analyzer": {"kind": "trace"},
                    "image": {
                        "type": "line",
                        "description": "test",
                        "data_loading": {"data_type": "csv"},
                    },
                    "scan": {"priority": 7},
                }
            )
        )

        analyzer = create_scan_analyzer(load_diagnostic(path))

        assert analyzer.id == "PW-MagSpectStitcher"
        assert analyzer.device_name == "CAM-TEA-MagSpecA-interpSpec"

    def test_priority_defaults_to_scan_priority(self, legacy):
        analyzer = create_scan_analyzer(_diag(scan={"priority": 7}, legacy=legacy))
        assert analyzer.priority == 7

    def test_priority_kwarg_overrides_scan_priority(self, legacy):
        analyzer = create_scan_analyzer(
            _diag(scan={"priority": 7}, legacy=legacy), priority=99
        )
        assert analyzer.priority == 99

    def test_retired_gdoc_slot_is_accepted_but_not_attached(self):
        analyzer = create_scan_analyzer(_diag(scan={"gdoc_slot": 2}))
        assert not hasattr(analyzer, "gdoc_slot")

    def test_no_upload_state_on_default_analyzer(self):
        analyzer = create_scan_analyzer(_diag())
        assert not hasattr(analyzer, "gdoc_slot")

    def test_save_maps_to_flag_save_images_for_array2d(self):
        on = create_scan_analyzer(_diag(scan={"save": True}, legacy=True))
        off = create_scan_analyzer(_diag(scan={"save": False}, legacy=True))
        assert on.flag_save_data is True  # base attr is flag_save_data
        assert off.flag_save_data is False

    def test_save_maps_to_flag_save_data_for_array1d(self):
        line_image = {
            "type": "line",
            "description": "x",
            "data_loading": {"data_type": "csv"},
        }
        on = create_scan_analyzer(
            _diag(
                alias="standard_1d", image=line_image, scan={"save": True}, legacy=True
            )
        )
        off = create_scan_analyzer(
            _diag(
                alias="standard_1d", image=line_image, scan={"save": False}, legacy=True
            )
        )
        assert on.flag_save_data is True
        assert off.flag_save_data is False

    def test_analysis_mode_passed_through(self):
        analyzer = create_scan_analyzer(_diag(scan={"mode": "per_bin"}, legacy=True))
        assert analyzer.analysis_mode == "per_bin"

    def test_device_override_routes_to_data_device_name(self):
        """``scan.device`` overrides only the *data folder*, not the GEECS device.

        ``device_name`` keys auxiliary-data lookups and background-image
        paths and must stay the GEECS device identifier;
        ``data_device_name`` is the data subfolder override used for
        post-processed/stitched outputs that live next to the device's
        own folder.
        """
        analyzer = create_scan_analyzer(
            _diag(name="UC_Logical", scan={"device": "UC_DataFolder"}, legacy=True)
        )
        assert analyzer.device_name == "UC_Logical"
        assert analyzer.data_device_name == "UC_DataFolder"

    def test_no_device_override_uses_top_level_name(self):
        analyzer = create_scan_analyzer(_diag(name="UC_Same", legacy=True))
        assert analyzer.device_name == "UC_Same"
        # ``data_device_name`` defaults to ``device_name`` inside the
        # wrapper; the constructor coerces ``None`` → ``device_name``.
        assert analyzer.data_device_name == "UC_Same"

    def test_file_tail_passed_through_when_set(self):
        analyzer = create_scan_analyzer(_diag(scan={"file_tail": ".himg"}, legacy=True))
        assert analyzer.file_tail == ".himg"


class TestBackgroundSourceAttachment:
    """The scan.background_source directive is attached to the wrapper."""

    def test_default_is_none(self):
        # Read the way the legacy runtime reads it; the core route has no
        # such attribute because the core refuses scan-context backgrounds.
        analyzer = create_scan_analyzer(_diag())
        assert getattr(analyzer, "background_source", None) is None

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

    def test_autodetect_directive_attached(self):
        analyzer = create_scan_analyzer(
            _diag(scan={"background_source": {"autodetect": {}}})
        )
        assert analyzer.background_source is not None
        assert analyzer.background_source.scan_number is None
        assert analyzer.background_source.from_current_scan is None
        assert analyzer.background_source.autodetect is not None
