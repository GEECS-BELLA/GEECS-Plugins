"""Tests for the unified diagnostic config and group config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from image_analysis.config import (
    DiagnosticAnalysisConfig,
    ImageAnalyzerSpec,
    resolve_image_analyzer_value,
)
from image_analysis.config.array1d_processing import Line1DConfig
from image_analysis.config.array2d_processing import CameraConfig
from scan_analysis.config.diagnostic_models import (
    AnalysisGroupConfig,
    AnalyzerRef,
    BackgroundSource,
    FromCurrentScanSpec,
    ScanRuntimeConfig,
)


# ``image_analyzer`` now takes either a bare class-path string or a verbose
# ``{class_path, kwargs}`` dict — the 2D-vs-1D dimension lives on the
# ``image:`` section as a ``type: camera`` / ``type: line`` discriminator.


class TestResolveImageAnalyzerValue:
    """Surface-form normalisation: bare class-path string or verbose dict."""

    def test_bare_class_path_string_resolves(self):
        out = resolve_image_analyzer_value(
            "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"
        )
        assert out == {
            "class_path": ("image_analysis.analyzers.beam_analyzer.BeamAnalyzer")
        }
        spec = ImageAnalyzerSpec.model_validate(out)
        assert spec.kwargs == {}

    def test_verbose_dict_with_class_path_passes_through(self):
        out = resolve_image_analyzer_value(
            {
                "class_path": "my.module.MyAnalyzer",
                "kwargs": {"x": 1},
            }
        )
        spec = ImageAnalyzerSpec.model_validate(out)
        assert spec.class_path == "my.module.MyAnalyzer"
        assert spec.kwargs == {"x": 1}

    def test_verbose_dict_accepts_yaml_class_key(self):
        out = resolve_image_analyzer_value(
            {
                "class": "my.module.MyAnalyzer",
                "kwargs": {"x": 1},
            }
        )
        assert out["class_path"] == "my.module.MyAnalyzer"
        assert "class" not in out

    def test_dict_without_class_path_rejected(self):
        with pytest.raises(ValueError, match="must contain 'class_path'"):
            resolve_image_analyzer_value({"kwargs": {"x": 1}})

    def test_non_string_non_mapping_rejected(self):
        with pytest.raises(ValueError, match="must be a class-path string"):
            resolve_image_analyzer_value(42)


class TestImageAnalyzerSpec:
    """The canonical runtime form: just class_path + kwargs, forbids extras."""

    def test_minimal_construction(self):
        spec = ImageAnalyzerSpec(class_path="x.Y")
        assert spec.class_path == "x.Y"
        assert spec.kwargs == {}

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ImageAnalyzerSpec(
                class_path="x.Y",
                unknown_field=1,
            )

    def test_empty_class_path_rejected(self):
        with pytest.raises(ValidationError):
            ImageAnalyzerSpec(class_path="")

    def test_image_kind_field_rejected(self):
        # ``image_kind`` moved to the ``type`` field on CameraConfig /
        # Line1DConfig in PR-E. Should be a hard error if anyone tries
        # to set it on the analyzer spec.
        with pytest.raises(ValidationError):
            ImageAnalyzerSpec(class_path="x.Y", image_kind="line")

    def test_scan_type_field_rejected(self):
        # ``scan_type`` also moved (decided from ``isinstance(diag.image, ...)``).
        with pytest.raises(ValidationError):
            ImageAnalyzerSpec(class_path="x.Y", scan_type="array1d")


class TestScanRuntimeConfig:
    """Defaults match the common case; extras are rejected."""

    def test_defaults(self):
        cfg = ScanRuntimeConfig()
        assert cfg.priority == 100
        assert cfg.mode == "per_shot"
        assert cfg.save is True
        assert cfg.gdoc_slot is None
        assert cfg.device is None
        assert cfg.file_tail is None
        assert cfg.renderer_kwargs == {}

    def test_gdoc_slot_must_be_in_range(self):
        with pytest.raises(ValidationError):
            ScanRuntimeConfig(gdoc_slot=4)
        with pytest.raises(ValidationError):
            ScanRuntimeConfig(gdoc_slot=-1)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ScanRuntimeConfig(unknown="x")


class TestDiagnosticAnalysisConfig:
    """Top-level diagnostic config validates the unified YAML shape."""

    _BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"

    def test_minimal_bare_string_form(self):
        cfg = DiagnosticAnalysisConfig(
            name="UC_GaiaMode", image_analyzer=self._BEAM_PATH
        )
        assert cfg.name == "UC_GaiaMode"
        assert cfg.image_analyzer.class_path == self._BEAM_PATH
        # Omitted ``image:`` stays ``None``; callers that need a typed
        # model to mutate should provide ``image: {}`` explicitly.
        assert cfg.image is None
        # ``scan`` is weakly typed at the ImageAnalysis layer; omitted
        # → ``None``. Scan-side validation happens via
        # ``ScanRuntimeConfig.model_validate(cfg.scan or {})``.
        assert cfg.scan is None
        scan_cfg = ScanRuntimeConfig.model_validate(cfg.scan or {})
        assert scan_cfg.priority == 100

    def test_full_shape_validates_camera(self):
        from image_analysis.config.array2d_processing import BackgroundMethod

        cfg = DiagnosticAnalysisConfig(
            name="UC_GaiaMode",
            image_analyzer=self._BEAM_PATH,
            image={
                "type": "camera",
                "bit_depth": 16,
                "background": {"method": "constant"},
            },
            scan={"priority": 45, "save": False, "mode": "per_shot", "gdoc_slot": 0},
        )
        # ``type: camera`` routes the discriminator to CameraConfig.
        assert isinstance(cfg.image, CameraConfig)
        assert cfg.image.bit_depth == 16
        assert cfg.image.background.method is BackgroundMethod.CONSTANT
        # The top-level name is injected as ``image.name``.
        assert cfg.image.name == "UC_GaiaMode"
        # ``scan`` stays a raw dict at this layer
        assert cfg.scan == {
            "priority": 45,
            "save": False,
            "mode": "per_shot",
            "gdoc_slot": 0,
        }
        # Scan-side validation produces the typed view
        scan_cfg = ScanRuntimeConfig.model_validate(cfg.scan)
        assert scan_cfg.priority == 45
        assert scan_cfg.save is False
        assert scan_cfg.gdoc_slot == 0

    def test_explicit_type_line_routes_to_line1d_config(self):
        cfg = DiagnosticAnalysisConfig(
            name="U_BCaveICT",
            image_analyzer="image_analysis.analyzers.ict_1d_analyzer.ICT1DAnalyzer",
            image={
                "type": "line",
                "data_loading": {"data_type": "tdms_scope"},
            },
        )
        assert isinstance(cfg.image, Line1DConfig)
        assert cfg.image.name == "U_BCaveICT"

    def test_verbose_image_analyzer_form_with_kwargs(self):
        cfg = DiagnosticAnalysisConfig(
            name="UC_Custom",
            image_analyzer={
                "class": "my.module.CustomAnalyzer",
                "kwargs": {"extra": 1},
            },
        )
        assert cfg.image_analyzer.class_path == "my.module.CustomAnalyzer"
        assert cfg.image_analyzer.kwargs == {"extra": 1}
        assert cfg.image is None

    def test_missing_name_rejected(self):
        with pytest.raises(ValidationError):
            DiagnosticAnalysisConfig(image_analyzer=self._BEAM_PATH)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            DiagnosticAnalysisConfig(
                name="x", image_analyzer=self._BEAM_PATH, unknown_field="oops"
            )


class TestFromCurrentScanSpec:
    """method requires the right percentile pairing; defaults to median."""

    def test_default_is_median_no_percentile(self):
        spec = FromCurrentScanSpec()
        assert spec.method == "median"
        assert spec.percentile is None

    def test_percentile_method_requires_value(self):
        with pytest.raises(ValidationError, match="percentile is required"):
            FromCurrentScanSpec(method="percentile")

    def test_percentile_with_method_percentile_validates(self):
        spec = FromCurrentScanSpec(method="percentile", percentile=5)
        assert spec.percentile == 5

    def test_percentile_with_median_method_rejected(self):
        with pytest.raises(
            ValidationError, match="must not be set when method='median'"
        ):
            FromCurrentScanSpec(method="median", percentile=5)

    def test_percentile_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            FromCurrentScanSpec(method="percentile", percentile=101)
        with pytest.raises(ValidationError):
            FromCurrentScanSpec(method="percentile", percentile=-1)


class TestBackgroundSource:
    """Exactly one of scan_number / from_current_scan must be set."""

    def test_scan_number_only(self):
        src = BackgroundSource(scan_number=5)
        assert src.scan_number == 5
        assert src.from_current_scan is None

    def test_from_current_scan_only(self):
        src = BackgroundSource(from_current_scan={"method": "median"})
        assert src.scan_number is None
        assert src.from_current_scan.method == "median"

    def test_neither_set_rejected(self):
        with pytest.raises(ValidationError, match="must specify exactly one source"):
            BackgroundSource()

    def test_both_set_rejected(self):
        with pytest.raises(ValidationError, match="must specify exactly one source"):
            BackgroundSource(scan_number=5, from_current_scan={"method": "median"})

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            BackgroundSource(scan_number=5, file_path="/x/y.npy")

    def test_scan_number_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            BackgroundSource(scan_number=-1)


class TestScanRuntimeConfigBackgroundSource:
    """background_source is optional and validates through to the model."""

    def test_default_is_none(self):
        cfg = ScanRuntimeConfig()
        assert cfg.background_source is None

    def test_accepts_scan_number_directive(self):
        cfg = ScanRuntimeConfig(background_source={"scan_number": 1})
        assert cfg.background_source.scan_number == 1

    def test_accepts_from_current_scan_directive(self):
        cfg = ScanRuntimeConfig(
            background_source={"from_current_scan": {"method": "median"}}
        )
        assert cfg.background_source.from_current_scan.method == "median"

    def test_invalid_directive_rejected_at_scan_level(self):
        with pytest.raises(ValidationError):
            ScanRuntimeConfig(background_source={})


class TestAnalysisGroupConfig:
    """Group config: string refs auto-expand; AnalyzerRef supports overrides."""

    def test_string_refs_normalise(self):
        cfg = AnalysisGroupConfig(name="g", analyzers=["A", "B"])
        assert all(isinstance(a, AnalyzerRef) for a in cfg.analyzers)
        assert [a.ref for a in cfg.analyzers] == ["A", "B"]
        assert all(a.enabled for a in cfg.analyzers)

    def test_dict_form_supports_disable_and_priority_override(self):
        cfg = AnalysisGroupConfig(
            name="g",
            analyzers=[
                "A",
                {"ref": "B", "enabled": False},
                {"ref": "C", "priority": 10},
            ],
        )
        assert cfg.analyzers[1].enabled is False
        assert cfg.analyzers[2].priority == 10

    def test_defaults(self):
        cfg = AnalysisGroupConfig(name="g")
        assert cfg.description is None
        assert cfg.upload_to_scanlog is True
        assert cfg.analyzers == []

    def test_missing_name_rejected(self):
        with pytest.raises(ValidationError):
            AnalysisGroupConfig(analyzers=[])

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            AnalysisGroupConfig(name="g", unknown="x")
