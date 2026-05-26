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


_BEAM_PATH = "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"
_GRENOUILLE_PATH = "image_analysis.analyzers.grenouille_analyzer.GrenouilleAnalyzer"
_LINE_PATH = "image_analysis.analyzers.line_analyzer.LineAnalyzer"


def _write_diagnostic(path: Path, name: str, *, image_analyzer=_BEAM_PATH) -> None:
    """Write a minimal diagnostic YAML at ``path``.

    ``image_analyzer`` defaults to the BeamAnalyzer class path (a 2D
    camera analyzer). Pass either a class-path string or a verbose
    dict for 1D / no-image-config analyzers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "image_analyzer": image_analyzer,
                "image": {"type": "camera", "bit_depth": 16},
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
        image_analyzer=_GRENOUILLE_PATH,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# load_diagnostic
# ---------------------------------------------------------------------------


class TestLoadDiagnostic:
    """``load_diagnostic`` finds, parses, and validates a unified YAML."""

    def test_load_by_stem_returns_validated_config(self, configs_tree):
        from image_analysis.config.array2d_processing import CameraConfig

        diag = load_diagnostic("UC_GaiaMode", config_dir=configs_tree)
        assert isinstance(diag, DiagnosticAnalysisConfig)
        assert diag.name == "UC_GaiaMode"
        # image: is a typed CameraConfig at this layer — the top-level
        # ``name`` is injected as the device identity for the embedded
        # image section.
        assert isinstance(diag.image, CameraConfig)
        assert diag.image.bit_depth == 16
        assert diag.image.name == "UC_GaiaMode"
        # scan: stays weakly typed at this layer (ScanAnalysis validates
        # it against its own ScanRuntimeConfig).
        assert diag.scan == {"priority": 100}

    def test_load_by_absolute_path(self, configs_tree):
        path = configs_tree / "analyzers" / "PW" / "PW_FROG.yaml"
        diag = load_diagnostic(path)
        assert diag.name == "FROG-LB-1-Temporal"
        assert diag.image_analyzer.class_path == _GRENOUILLE_PATH

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

    def test_camera_analyzer_built_from_bare_class_path(self):
        diag = DiagnosticAnalysisConfig(
            name="UC_TestBeam",
            image_analyzer=_BEAM_PATH,
            image={"type": "camera", "bit_depth": 16},
        )
        analyzer = create_image_analyzer(diag)
        # Sanity: the returned object is the right class, configured with
        # a CameraConfig whose name is the diagnostic name.
        assert analyzer.__class__.__name__ == "BeamAnalyzer"
        assert analyzer.camera_config.name == "UC_TestBeam"

    def test_line_analyzer_built_from_typed_image(self):
        # 1D analyzers are picked by ``type: line`` on the image section,
        # not by any sibling field on image_analyzer.
        diag = DiagnosticAnalysisConfig(
            name="UC_TestLine",
            image_analyzer=_LINE_PATH,
            image={
                "type": "line",
                "data_loading": {"data_type": "tsv"},
                "background": {"method": "constant", "constant_value": 0.0},
            },
        )
        analyzer = create_image_analyzer(diag)
        assert analyzer.__class__.__name__ == "LineAnalyzer"

    def test_verbose_form_yaml_class_alias_accepted(self):
        diag = DiagnosticAnalysisConfig(
            name="UC_TestVerbose",
            image_analyzer={"class": _BEAM_PATH},
            image={"type": "camera", "bit_depth": 16},
        )
        analyzer = create_image_analyzer(diag)
        assert analyzer.__class__.__name__ == "BeamAnalyzer"

    def test_diagnostic_without_image_section(self):
        # No-image analyzers (HASO-style) just omit the ``image:`` section.
        # ``diag.image`` ends up None; the factory passes only what
        # ``image_analyzer.kwargs`` supplied to the analyzer constructor.
        diag = DiagnosticAnalysisConfig(
            name="UC_NoImage",
            image_analyzer=_BEAM_PATH,
        )
        assert diag.image is None

    def test_unknown_class_path_raises(self):
        diag = DiagnosticAnalysisConfig(
            name="UC_Bogus",
            image_analyzer={"class_path": "nonexistent.module.BogusAnalyzer"},
            image={"type": "camera", "bit_depth": 16},
        )
        with pytest.raises(ImportError):
            create_image_analyzer(diag)
