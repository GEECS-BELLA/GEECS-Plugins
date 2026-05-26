"""Tests for the unified diagnostic config and group config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from image_analysis.config import (
    DiagnosticAnalysisConfig,
    ImageAnalyzerSpec,
    ImageKind,
    ScanType,
    resolve_image_analyzer_value,
)
from scan_analysis.config.diagnostic_models import (
    AnalysisGroupConfig,
    AnalyzerRef,
    BackgroundSource,
    FromCurrentScanSpec,
    ScanRuntimeConfig,
)


# The alias registry was removed in PR-E. ``image_analyzer`` now takes
# either a bare class-path string (defaults to camera + array2d) or a
# verbose dict with explicit image_kind / scan_type / kwargs.


class TestResolveImageAnalyzerValue:
    """Surface-form normalisation: bare class-path string or verbose dict."""

    def test_bare_class_path_string_resolves_with_defaults(self):
        out = resolve_image_analyzer_value(
            "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"
        )
        assert out == {
            "class_path": ("image_analysis.analyzers.beam_analyzer.BeamAnalyzer")
        }
        # Model defaults fill in image_kind=camera, scan_type=array2d
        spec = ImageAnalyzerSpec.model_validate(out)
        assert spec.image_kind is ImageKind.CAMERA
        assert spec.scan_type is ScanType.ARRAY2D
        assert spec.kwargs == {}

    def test_verbose_dict_with_class_path_passes_through(self):
        out = resolve_image_analyzer_value(
            {
                "class_path": "my.module.MyAnalyzer",
                "image_kind": "line",
                "scan_type": "array1d",
                "kwargs": {"x": 1},
            }
        )
        spec = ImageAnalyzerSpec.model_validate(out)
        assert spec.class_path == "my.module.MyAnalyzer"
        assert spec.image_kind is ImageKind.LINE
        assert spec.scan_type is ScanType.ARRAY1D
        assert spec.kwargs == {"x": 1}

    def test_verbose_dict_accepts_yaml_class_key(self):
        out = resolve_image_analyzer_value(
            {
                "class": "my.module.MyAnalyzer",
                "image_kind": "camera",
                "scan_type": "array2d",
            }
        )
        assert out["class_path"] == "my.module.MyAnalyzer"
        assert "class" not in out

    def test_dict_without_class_path_rejected(self):
        with pytest.raises(ValueError, match="must contain 'class_path'"):
            resolve_image_analyzer_value({"image_kind": "camera"})

    def test_non_string_non_mapping_rejected(self):
        with pytest.raises(ValueError, match="must be a class-path string"):
            resolve_image_analyzer_value(42)


class TestImageAnalyzerSpec:
    """The canonical runtime form: forbids extras, validates enums."""

    def test_defaults_to_camera_array2d(self):
        spec = ImageAnalyzerSpec(class_path="x.Y")
        assert spec.image_kind is ImageKind.CAMERA
        assert spec.scan_type is ScanType.ARRAY2D
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
        # Bare-string form takes the model defaults
        assert cfg.image_analyzer.image_kind is ImageKind.CAMERA
        assert cfg.image_analyzer.scan_type is ScanType.ARRAY2D
        assert cfg.image is None
        # ``scan`` is weakly typed at the ImageAnalysis layer; omitted
        # → ``None``. Scan-side validation happens via
        # ``ScanRuntimeConfig.model_validate(cfg.scan or {})``.
        assert cfg.scan is None
        scan_cfg = ScanRuntimeConfig.model_validate(cfg.scan or {})
        assert scan_cfg.priority == 100

    def test_full_shape_validates(self):
        cfg = DiagnosticAnalysisConfig(
            name="UC_GaiaMode",
            image_analyzer=self._BEAM_PATH,
            image={"bit_depth": 16, "background": {"method": "constant"}},
            scan={"priority": 45, "save": False, "mode": "per_shot", "gdoc_slot": 0},
        )
        assert cfg.image == {"bit_depth": 16, "background": {"method": "constant"}}
        # ``scan`` is a raw dict at this layer
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

    def test_verbose_image_analyzer_form_with_overrides(self):
        cfg = DiagnosticAnalysisConfig(
            name="UC_Custom",
            image_analyzer={
                "class": "my.module.CustomAnalyzer",
                "image_kind": "line",
                "scan_type": "array1d",
                "kwargs": {"extra": 1},
            },
        )
        assert cfg.image_analyzer.class_path == "my.module.CustomAnalyzer"
        assert cfg.image_analyzer.image_kind is ImageKind.LINE
        assert cfg.image_analyzer.scan_type is ScanType.ARRAY1D
        assert cfg.image_analyzer.kwargs == {"extra": 1}

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
