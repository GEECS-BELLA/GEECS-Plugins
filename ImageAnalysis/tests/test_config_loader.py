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
    _deep_merge,
    _unwrap_diagnostic_image_section,
    load_camera_config,
    load_diagnostic,
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

    def test_top_level_name_is_not_injected_into_image(self):
        """After #412 the unwrap no longer copies top-level ``name``.

        Camera/Line configs no longer have a ``name`` field; the
        diagnostic factory wires analyzer identity through the
        ``output_name`` constructor kwarg instead.
        """
        data = {"name": "UC_X", "image": {"bit_depth": 16}}
        result = _unwrap_diagnostic_image_section(data)
        assert "name" not in result
        assert result["bit_depth"] == 16

    def test_image_section_must_be_a_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _unwrap_diagnostic_image_section(
                {"name": "x", "image": ["not", "a", "dict"]}
            )

    def test_empty_image_section_treated_as_empty_dict(self):
        result = _unwrap_diagnostic_image_section({"name": "UC_X", "image": None})
        assert result == {}


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
        # ``name`` is no longer a typed CameraConfig field (#412).
        # It survives as an extra (extra="allow") but the asserted
        # contract is that the camera bit_depth round-tripped.
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
        # ``name`` is no longer a typed Line1DConfig field (#412).
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
        assert cfg.bit_depth == 12


class TestDeepMerge:
    """Recursive dict merge: nested mappings merge key-by-key, scalars replace."""

    def test_disjoint_keys_union(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_overlay_scalar_replaces_base_scalar(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_dicts_merge_key_by_key(self):
        base = {"scan": {"mode": "per_shot", "priority": 100}}
        overlay = {"scan": {"mode": "per_bin"}}
        # priority survives because overlay's scan block only sets mode
        assert _deep_merge(base, overlay) == {
            "scan": {"mode": "per_bin", "priority": 100}
        }

    def test_three_level_nesting(self):
        base = {"image": {"background": {"method": "constant", "value": 0.0}}}
        overlay = {"image": {"background": {"value": 12.5}}}
        assert _deep_merge(base, overlay) == {
            "image": {"background": {"method": "constant", "value": 12.5}}
        }

    def test_lists_replace_wholesale(self):
        """Lists do not concatenate — replacing a list is the only sensible semantic.

        An overlay that wants to *replace* a list shouldn't have to know
        how long the base list was; an overlay that wants to *extend*
        a list would have ambiguous semantics (append? prepend? splice?).
        """
        assert _deep_merge({"steps": [1, 2, 3]}, {"steps": [9]}) == {"steps": [9]}

    def test_none_replaces_base(self):
        assert _deep_merge({"a": 1}, {"a": None}) == {"a": None}

    def test_returns_new_dict_does_not_mutate_inputs(self):
        base = {"scan": {"mode": "per_shot"}}
        overlay = {"scan": {"mode": "per_bin"}}
        _deep_merge(base, overlay)
        assert base == {"scan": {"mode": "per_shot"}}
        assert overlay == {"scan": {"mode": "per_bin"}}

    def test_overlay_can_introduce_new_nested_key(self):
        base = {"scan": {"mode": "per_shot"}}
        overlay = {"scan": {"gdoc_slot": 2}}
        assert _deep_merge(base, overlay) == {
            "scan": {"mode": "per_shot", "gdoc_slot": 2}
        }


class TestLoadDiagnosticOverrides:
    """``load_diagnostic(..., overrides=...)`` patches the YAML before validation."""

    def _write_diagnostic(self, tmp_path, *, scan_mode="per_shot"):
        """Write a minimal valid diagnostic YAML; return its path."""
        path = tmp_path / "UC_Test.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "UC_Test",
                    "image_analyzer": (
                        "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"
                    ),
                    "image": {"type": "camera", "bit_depth": 16},
                    "scan": {"mode": scan_mode, "priority": 100},
                }
            )
        )
        return path

    def test_no_overrides_loads_disk_yaml_as_is(self, tmp_path):
        path = self._write_diagnostic(tmp_path, scan_mode="per_shot")
        diag = load_diagnostic(path)
        assert (diag.scan or {}).get("mode") == "per_shot"

    def test_scan_mode_override_applied(self, tmp_path):
        path = self._write_diagnostic(tmp_path, scan_mode="per_shot")
        diag = load_diagnostic(path, overrides={"scan": {"mode": "per_bin"}})
        assert diag.scan["mode"] == "per_bin"

    def test_partial_scan_override_preserves_other_scan_fields(self, tmp_path):
        """Override of ``scan.mode`` leaves ``scan.priority`` from disk in place."""
        path = self._write_diagnostic(tmp_path, scan_mode="per_shot")
        diag = load_diagnostic(path, overrides={"scan": {"mode": "per_bin"}})
        assert diag.scan["mode"] == "per_bin"
        assert diag.scan["priority"] == 100  # survived from disk

    def test_invalid_override_value_raises_at_load_time(self, tmp_path):
        """Override typos surface via Pydantic, not via silent misbehavior.

        Targets ``image.bit_depth`` (strictly typed as Literal in CameraConfig)
        rather than a ``scan`` field — ``scan:`` is intentionally weak-typed
        at this layer (ScanAnalysis re-validates it via ``ScanRuntimeConfig``
        downstream). The override path applies validation to whatever the
        diagnostic *does* strongly type.
        """
        path = self._write_diagnostic(tmp_path)
        with pytest.raises(ValueError, match="Invalid diagnostic config"):
            load_diagnostic(path, overrides={"image": {"bit_depth": 99}})

    def test_empty_overrides_treated_as_no_overrides(self, tmp_path):
        path = self._write_diagnostic(tmp_path, scan_mode="per_shot")
        diag = load_diagnostic(path, overrides={})
        assert diag.scan["mode"] == "per_shot"

    def test_none_overrides_treated_as_no_overrides(self, tmp_path):
        path = self._write_diagnostic(tmp_path, scan_mode="per_shot")
        diag = load_diagnostic(path, overrides=None)
        assert diag.scan["mode"] == "per_shot"
