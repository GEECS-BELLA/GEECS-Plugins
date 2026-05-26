"""Tests for the ImageAnalysis config loader.

The loader accepts two YAML shapes:

* Flat: legacy ``image_analysis_configs/*.yaml`` files with the camera
  or line config at the top level.
* Diagnostic: unified ``scan_analysis_configs/analyzers/*.yaml`` files
  with ``name``, ``image_analyzer``, ``image:``, and ``scan:`` sections.
  The loader extracts the ``image:`` subdict and injects the top-level
  ``name`` so the embedded section validates as a standalone camera or
  line config.
"""

from __future__ import annotations

import pytest
import yaml

from image_analysis.config.loader import (
    _unwrap_diagnostic_image_section,
    load_camera_config,
    load_line_config,
)


class TestUnwrapDiagnosticImageSection:
    """Pure-function tests of the unwrap helper."""

    def test_flat_config_passes_through(self):
        data = {"name": "UC_X", "bit_depth": 16, "background": {"method": "constant"}}
        assert _unwrap_diagnostic_image_section(data) is data

    def test_unified_yaml_returns_image_subdict(self):
        data = {
            "name": "UC_X",
            "image_analyzer": "beam",
            "image": {"bit_depth": 16},
            "scan": {"priority": 10},
        }
        result = _unwrap_diagnostic_image_section(data)
        assert result["bit_depth"] == 16
        assert "scan" not in result
        assert "image_analyzer" not in result

    def test_top_level_name_is_injected(self):
        data = {"name": "UC_X", "image": {"bit_depth": 16}}
        result = _unwrap_diagnostic_image_section(data)
        assert result["name"] == "UC_X"

    def test_explicit_image_name_is_not_overridden(self):
        data = {"name": "UC_outer", "image": {"name": "UC_inner", "bit_depth": 16}}
        result = _unwrap_diagnostic_image_section(data)
        assert result["name"] == "UC_inner"

    def test_image_section_must_be_a_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _unwrap_diagnostic_image_section(
                {"name": "x", "image": ["not", "a", "dict"]}
            )

    def test_empty_image_section_treated_as_empty_dict(self):
        result = _unwrap_diagnostic_image_section({"name": "UC_X", "image": None})
        assert result == {"name": "UC_X"}


class TestLoadCameraConfigFromDiagnosticPath:
    """End-to-end: ``load_camera_config`` reads a unified diagnostic YAML."""

    def test_load_camera_config_from_unified_yaml(self, tmp_path):
        path = tmp_path / "UC_Test.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "UC_Test",
                    "image_analyzer": "beam",
                    "image": {
                        "bit_depth": 16,
                    },
                    "scan": {"priority": 50},
                }
            )
        )

        cfg = load_camera_config(path)
        assert cfg.name == "UC_Test"
        assert cfg.bit_depth == 16

    def test_load_line_config_from_unified_yaml(self, tmp_path):
        path = tmp_path / "U_Line.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "U_Line",
                    "image_analyzer": "standard_1d",
                    "image": {
                        "description": "test line config",
                        "data_loading": {"data_type": "csv"},
                    },
                    "scan": {"priority": 50},
                }
            )
        )

        cfg = load_line_config(path)
        assert cfg.name == "U_Line"
        assert cfg.data_loading.data_type.value == "csv"

    def test_load_camera_config_from_legacy_flat_yaml_still_works(self, tmp_path):
        path = tmp_path / "UC_Flat.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "UC_Flat",
                    "bit_depth": 12,
                }
            )
        )

        cfg = load_camera_config(path)
        assert cfg.name == "UC_Flat"
        assert cfg.bit_depth == 12
