"""Tests for ``image_analysis.config.factory``.

Covers the Mode-2 entry points: :func:`load_diagnostic` and
:func:`create_image_analyzer`. The lower-level model validation is
covered by the diagnostic-model tests in ScanAnalysis.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from image_analysis.config import (
    DiagnosticAnalysisConfig,
    create_image_analyzer,
    load_diagnostic,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_diagnostic(path: Path, name: str, *, image_analyzer: str = "beam") -> None:
    """Write a minimal diagnostic YAML at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "image_analyzer": image_analyzer,
                "image": {"bit_depth": 16},
                "scan": {"priority": 100},
            }
        )
    )


@pytest.fixture
def configs_tree(tmp_path: Path) -> Path:
    """A small unified-configs tree under tmp_path."""
    _write_diagnostic(
        tmp_path / "analyzers" / "HTU" / "UC_GaiaMode.yaml", "UC_GaiaMode"
    )
    _write_diagnostic(
        tmp_path / "analyzers" / "PW" / "PW_FROG.yaml",
        "FROG-LB-1-Temporal",
        image_analyzer="grenouille",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# load_diagnostic
# ---------------------------------------------------------------------------


class TestLoadDiagnostic:
    """``load_diagnostic`` finds, parses, and validates a unified YAML."""

    def test_load_by_stem_returns_validated_config(self, configs_tree):
        diag = load_diagnostic("UC_GaiaMode", config_dir=configs_tree)
        assert isinstance(diag, DiagnosticAnalysisConfig)
        assert diag.name == "UC_GaiaMode"
        # image: stays a dict at this layer (validated by create_image_analyzer
        # when it's actually consumed)
        assert diag.image == {"bit_depth": 16}
        # scan: also weakly typed at this layer
        assert diag.scan == {"priority": 100}

    def test_load_by_absolute_path(self, configs_tree):
        path = configs_tree / "analyzers" / "PW" / "PW_FROG.yaml"
        diag = load_diagnostic(path)
        assert diag.name == "FROG-LB-1-Temporal"
        assert diag.image_analyzer.class_path.endswith("GrenouilleAnalyzer")

    def test_missing_name_raises_keyerror(self, configs_tree):
        with pytest.raises(KeyError, match="not found"):
            load_diagnostic("DoesNotExist", config_dir=configs_tree)

    def test_missing_path_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_diagnostic(tmp_path / "missing.yaml")

    def test_duplicate_stems_rejected_at_discovery(self, tmp_path):
        _write_diagnostic(tmp_path / "analyzers" / "HTU" / "Shared.yaml", "Shared")
        _write_diagnostic(tmp_path / "analyzers" / "PW" / "Shared.yaml", "Shared")
        with pytest.raises(ValueError, match="Duplicate diagnostic ID"):
            load_diagnostic("Shared", config_dir=tmp_path)

    def test_missing_analyzers_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Analyzer directory"):
            load_diagnostic("anything", config_dir=tmp_path)

    def test_invalid_yaml_surfaces_path(self, tmp_path):
        bad = tmp_path / "analyzers" / "HTU" / "Bad.yaml"
        bad.parent.mkdir(parents=True)
        # Missing required 'name' field
        bad.write_text(yaml.safe_dump({"image_analyzer": "beam"}))
        with pytest.raises(ValueError, match="Invalid diagnostic config"):
            load_diagnostic("Bad", config_dir=tmp_path)


# ---------------------------------------------------------------------------
# create_image_analyzer
# ---------------------------------------------------------------------------


class TestCreateImageAnalyzer:
    """``create_image_analyzer`` builds a live analyzer from a config."""

    def test_camera_analyzer_built_from_alias(self):
        # BeamAnalyzer is registered under the 'beam' alias (camera + array2d).
        diag = DiagnosticAnalysisConfig(
            name="UC_TestBeam",
            image_analyzer="beam",
            image={"bit_depth": 16},
        )
        analyzer = create_image_analyzer(diag)
        # Sanity: the returned object is the right class, configured with
        # a CameraConfig whose name is the diagnostic name.
        assert analyzer.__class__.__name__ == "BeamAnalyzer"
        assert analyzer.camera_config.name == "UC_TestBeam"

    def test_line_analyzer_built_from_alias(self):
        diag = DiagnosticAnalysisConfig(
            name="UC_TestLine",
            image_analyzer="line",
            image={
                "data_loading": {"data_type": "tsv"},
                "background": {"method": "constant", "constant_value": 0.0},
            },
        )
        analyzer = create_image_analyzer(diag)
        assert analyzer.__class__.__name__ == "LineAnalyzer"

    def test_verbose_class_path_form_works(self):
        diag = DiagnosticAnalysisConfig(
            name="UC_TestVerbose",
            image_analyzer={
                "class": "image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer",
                "image_kind": "camera",
                "scan_type": "array2d",
            },
            image={"bit_depth": 16},
        )
        analyzer = create_image_analyzer(diag)
        assert analyzer.__class__.__name__ == "BeamAnalyzer"

    def test_image_kind_none_with_image_section_raises(self):
        # HASO analyzer has image_kind=none; providing image: is an error.
        diag = DiagnosticAnalysisConfig(
            name="HasoLift",
            image_analyzer="haso",
            image={"bit_depth": 16},
        )
        with pytest.raises(ValueError, match="image_kind='none'"):
            create_image_analyzer(diag)

    def test_unknown_class_path_raises(self):
        diag = DiagnosticAnalysisConfig(
            name="UC_Bogus",
            image_analyzer={
                "class": "nonexistent.module.BogusAnalyzer",
                "image_kind": "camera",
                "scan_type": "array2d",
            },
            image={"bit_depth": 16},
        )
        with pytest.raises(ImportError):
            create_image_analyzer(diag)
