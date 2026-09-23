"""Tests for the ImageAnalysis config loader.

The loader reads two shapes:

* A diagnostic document (v2) — ``load_diagnostic``
  returns the typed document; ``load_camera_config`` / ``load_line_config``
  return its ``image:`` section.
* A bare camera / line mapping — ``load_camera_config`` /
  ``load_line_config`` validate it directly.
"""

from __future__ import annotations

import pytest
import yaml

from image_analysis.config.loader import (
    load_camera_config,
    load_diagnostic,
    load_line_config,
)


def _v2_camera_doc(**scan):
    return {
        "schema_version": 2,
        "name": "UC_Test",
        "analyzer": {"kind": "beam"},
        "image": {"type": "camera", "bit_depth": 16},
        "scan": {"priority": 50, **scan},
    }


class TestImageSectionLoaders:
    def test_load_diagnostic_preserves_source_filename_stem(self, tmp_path):
        path = tmp_path / "PW-MagSpectStitcher.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    "name": "CAM-TEA-MagSpecA-interpSpec",
                    "analyzer": {"kind": "line"},
                    "image": {
                        "type": "line",
                        "description": "test line config",
                        "data_loading": {"data_type": "csv"},
                    },
                    "scan": {"priority": 50},
                }
            )
        )
        diag = load_diagnostic(path)
        assert diag.name == "CAM-TEA-MagSpecA-interpSpec"
        assert diag.source_id == "PW-MagSpectStitcher"

    def test_load_camera_config_from_diagnostic_yaml(self, tmp_path):
        path = tmp_path / "UC_Test.yaml"
        path.write_text(yaml.safe_dump(_v2_camera_doc()))
        cfg = load_camera_config(path)
        assert cfg.bit_depth == 16

    def test_v1_diagnostic_yaml_is_refused(self, tmp_path):
        path = tmp_path / "UC_Legacy.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "UC_Legacy",
                    "image_analyzer": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
                    "image": {"type": "camera", "bit_depth": 12},
                }
            )
        )
        with pytest.raises(ValueError, match="pre-v2"):
            load_camera_config(path)

    def test_load_line_config_from_diagnostic_yaml(self, tmp_path):
        path = tmp_path / "U_Line.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    "name": "U_Line",
                    "analyzer": {"kind": "line"},
                    "image": {
                        "type": "line",
                        "description": "test line config",
                        "data_loading": {"data_type": "csv"},
                    },
                }
            )
        )
        cfg = load_line_config(path)
        assert cfg.data_loading.data_type.value == "csv"

    def test_wrong_section_kind_is_refused(self, tmp_path):
        path = tmp_path / "UC_Test.yaml"
        path.write_text(yaml.safe_dump(_v2_camera_doc()))
        with pytest.raises(ValueError, match="expected a line image section"):
            load_line_config(path)

    def test_bare_camera_yaml_loads_directly(self, tmp_path):
        path = tmp_path / "UC_Flat.yaml"
        path.write_text(yaml.safe_dump({"bit_depth": 12}))
        assert load_camera_config(path).bit_depth == 12

    def test_bare_camera_dict_loads_directly(self):
        assert load_camera_config({"bit_depth": 8}).bit_depth == 8

    def test_bare_yaml_with_unknown_key_is_refused(self, tmp_path):
        path = tmp_path / "UC_Flat.yaml"
        path.write_text(yaml.safe_dump({"bit_depth": 12, "name": "UC_Flat"}))
        with pytest.raises(ValueError, match="Invalid camera configuration"):
            load_camera_config(path)


class TestLoadDiagnosticOverrides:
    """``load_diagnostic(..., overrides=...)`` patches the YAML before validation."""

    def _write_diagnostic(self, tmp_path, *, scan_mode="per_shot"):
        path = tmp_path / "UC_Test.yaml"
        path.write_text(yaml.safe_dump(_v2_camera_doc(mode=scan_mode)))
        return path

    def test_no_overrides_loads_disk_yaml_as_is(self, tmp_path):
        diag = load_diagnostic(self._write_diagnostic(tmp_path))
        assert diag.scan.mode == "per_shot"

    def test_scan_mode_override_applied(self, tmp_path):
        path = self._write_diagnostic(tmp_path)
        diag = load_diagnostic(path, overrides={"scan": {"mode": "per_bin"}})
        assert diag.scan.mode == "per_bin"
        assert diag.scan.priority == 50  # survived from disk

    def test_invalid_override_value_raises_at_load_time(self, tmp_path):
        path = self._write_diagnostic(tmp_path)
        with pytest.raises(ValueError, match="Invalid diagnostic config"):
            load_diagnostic(path, overrides={"image": {"bit_depth": 99}})
        with pytest.raises(ValueError, match="Invalid diagnostic config"):
            load_diagnostic(path, overrides={"scan": {"mode": "per_frame"}})

    def test_empty_or_none_overrides_are_no_overrides(self, tmp_path):
        path = self._write_diagnostic(tmp_path)
        assert load_diagnostic(path, overrides={}).scan.mode == "per_shot"
        assert load_diagnostic(path, overrides=None).scan.mode == "per_shot"
