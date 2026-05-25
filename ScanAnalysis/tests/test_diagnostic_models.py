"""Tests for the unified diagnostic config and group config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from image_analysis.config import (
    ALIAS_REGISTRY,
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


class TestAliasRegistry:
    """The registry pins the alias → ImageAnalyzerSpec mapping for production analyzers."""

    def test_every_alias_is_a_valid_spec(self):
        for alias, spec in ALIAS_REGISTRY.items():
            assert isinstance(spec, ImageAnalyzerSpec), alias

    def test_known_camera_aliases_match_array2d(self):
        for alias in ("beam", "standard_2d", "grenouille", "magspec_manual"):
            spec = ALIAS_REGISTRY[alias]
            assert spec.image_kind is ImageKind.CAMERA
            assert spec.scan_type is ScanType.ARRAY2D

    def test_known_line_aliases_match_array1d(self):
        for alias in ("standard_1d", "line", "ict_1d", "line_stitcher"):
            spec = ALIAS_REGISTRY[alias]
            assert spec.image_kind is ImageKind.LINE
            assert spec.scan_type is ScanType.ARRAY1D

    def test_haso_alias_declares_no_embedded_image_config(self):
        spec = ALIAS_REGISTRY["haso"]
        assert spec.image_kind is ImageKind.NONE
        assert spec.scan_type is ScanType.ARRAY2D


class TestResolveImageAnalyzerValue:
    """Surface-form normalisation: string, alias dict, verbose dict."""

    def test_string_resolves_to_spec_dict(self):
        out = resolve_image_analyzer_value("beam")
        assert out["class_path"] == ALIAS_REGISTRY["beam"].class_path
        assert out["image_kind"] is ImageKind.CAMERA
        assert out["scan_type"] is ScanType.ARRAY2D
        assert out["kwargs"] == {}

    def test_alias_dict_with_kwargs_carries_kwargs(self):
        out = resolve_image_analyzer_value(
            {"alias": "line_stitcher", "kwargs": {"sibling_devices": ["a", "b"]}}
        )
        assert out["class_path"] == ALIAS_REGISTRY["line_stitcher"].class_path
        assert out["kwargs"] == {"sibling_devices": ["a", "b"]}

    def test_verbose_dict_accepts_yaml_class_key(self):
        out = resolve_image_analyzer_value(
            {
                "class": "my.module.MyAnalyzer",
                "image_kind": "camera",
                "scan_type": "array2d",
                "kwargs": {"x": 1},
            }
        )
        assert out["class_path"] == "my.module.MyAnalyzer"
        assert out["kwargs"] == {"x": 1}

    def test_unknown_alias_raises_with_helpful_message(self):
        with pytest.raises(ValueError, match="Unknown image_analyzer alias"):
            resolve_image_analyzer_value("definitely_not_an_alias")

    def test_alias_dict_with_extra_keys_rejected(self):
        with pytest.raises(ValueError, match="extra keys"):
            resolve_image_analyzer_value(
                {"alias": "beam", "image_kind": "camera"}  # image_kind not allowed here
            )

    def test_verbose_dict_without_class_or_alias_rejected(self):
        with pytest.raises(ValueError, match="must contain either 'alias'"):
            resolve_image_analyzer_value({"image_kind": "camera"})

    def test_non_string_non_mapping_rejected(self):
        with pytest.raises(ValueError, match="must be a string alias or a mapping"):
            resolve_image_analyzer_value(42)


class TestImageAnalyzerSpec:
    """The canonical runtime form: forbids extras, validates enums."""

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ImageAnalyzerSpec(
                class_path="x.Y",
                image_kind=ImageKind.CAMERA,
                scan_type=ScanType.ARRAY2D,
                unknown_field=1,
            )

    def test_empty_class_path_rejected(self):
        with pytest.raises(ValidationError):
            ImageAnalyzerSpec(
                class_path="",
                image_kind=ImageKind.CAMERA,
                scan_type=ScanType.ARRAY2D,
            )


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

    def test_minimal_alias_form(self):
        cfg = DiagnosticAnalysisConfig(name="UC_GaiaMode", image_analyzer="beam")
        assert cfg.name == "UC_GaiaMode"
        assert cfg.image_analyzer.class_path.endswith("BeamAnalyzer")
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
            image_analyzer="beam",
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

    def test_verbose_image_analyzer_form_validates(self):
        cfg = DiagnosticAnalysisConfig(
            name="UC_Custom",
            image_analyzer={
                "class": "my.module.CustomAnalyzer",
                "image_kind": "camera",
                "scan_type": "array2d",
                "kwargs": {"extra": 1},
            },
        )
        assert cfg.image_analyzer.class_path == "my.module.CustomAnalyzer"
        assert cfg.image_analyzer.kwargs == {"extra": 1}

    def test_alias_with_kwargs_form_validates(self):
        cfg = DiagnosticAnalysisConfig(
            name="U_Stitched",
            image_analyzer={
                "alias": "line_stitcher",
                "kwargs": {"sibling_devices": ["a", "b"]},
            },
        )
        assert cfg.image_analyzer.kwargs == {"sibling_devices": ["a", "b"]}

    def test_missing_name_rejected(self):
        with pytest.raises(ValidationError):
            DiagnosticAnalysisConfig(image_analyzer="beam")

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            DiagnosticAnalysisConfig(
                name="x", image_analyzer="beam", unknown_field="oops"
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
