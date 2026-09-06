"""Tests for ``image_analysis.config.factory`` and ``load_diagnostic`` over v2 documents.

The document model itself is GEECS-Schemas' (``AnalysisDiagnostic``) and is
tested there; here the concern is the Mode-2 path — find the YAML, hand the
typed spec to the right class.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from geecs_schemas.analysis import (
    AnalysisDiagnostic,
    CameraConfig,
    LineStitcherSpec,
)

from image_analysis.config import (
    DiagnosticAnalysisConfig,
    create_image_analyzer,
    load_diagnostic,
)


def _write_diagnostic(
    path: Path, name: str, *, analyzer: dict | None = None, image: dict | None = None
) -> None:
    """Write a minimal v2 diagnostic YAML at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "name": name,
        "analyzer": analyzer or {"kind": "beam"},
        "scan": {"priority": 100},
    }
    payload["image"] = {"type": "camera", "bit_depth": 16} if image is None else image
    path.write_text(yaml.safe_dump(payload))


@pytest.fixture
def configs_tree(tmp_path: Path) -> Path:
    """A small configs tree under tmp_path."""
    _write_diagnostic(
        tmp_path / "analyzers" / "HTU" / "UC_GaiaMode.yaml", "UC_GaiaMode"
    )
    _write_diagnostic(
        tmp_path / "analyzers" / "PW" / "PW_FROG.yaml",
        "FROG-LB-1-Temporal",
        analyzer={"kind": "frog_retrieval", "N": 256},
    )
    return tmp_path


class TestLoadDiagnostic:
    """``load_diagnostic`` finds, parses, and validates a diagnostic YAML."""

    def test_load_by_stem_returns_typed_document(self, configs_tree):
        diag = load_diagnostic("UC_GaiaMode", config_dir=configs_tree)
        assert isinstance(diag, AnalysisDiagnostic)
        assert DiagnosticAnalysisConfig is AnalysisDiagnostic  # transitional alias
        assert diag.name == "UC_GaiaMode"
        assert isinstance(diag.image, CameraConfig)
        assert diag.image.bit_depth == 16
        # scan: is typed in-document now
        assert diag.scan.priority == 100
        assert diag.scan.mode == "per_shot"
        assert diag.source_id == "UC_GaiaMode"

    def test_load_by_absolute_path(self, configs_tree):
        path = configs_tree / "analyzers" / "PW" / "PW_FROG.yaml"
        diag = load_diagnostic(path)
        assert diag.name == "FROG-LB-1-Temporal"
        assert diag.analyzer.kind == "frog_retrieval"
        assert diag.analyzer.N == 256

    def test_v1_document_is_refused_with_converter_hint(self, tmp_path):
        path = tmp_path / "analyzers" / "HTU" / "Legacy.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "UC_Legacy",
                    "image_analyzer": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
                    "image": {"type": "camera"},
                }
            )
        )
        with pytest.raises(ValueError, match="convert.analysis_diagnostics"):
            load_diagnostic("Legacy", config_dir=tmp_path)

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
        bad.write_text(yaml.safe_dump({"analyzer": {"kind": "beam"}}))  # no name
        with pytest.raises(ValueError, match="Invalid diagnostic config"):
            load_diagnostic("Bad", config_dir=tmp_path)

    def test_unknown_parameter_is_a_load_error(self, tmp_path):
        bad = tmp_path / "analyzers" / "HTU" / "Typo.yaml"
        _write_diagnostic(
            bad, "UC_Typo", analyzer={"kind": "beam", "compute_slope": True}
        )
        with pytest.raises(ValueError, match="compute_slope"):
            load_diagnostic("Typo", config_dir=tmp_path)


class TestCreateImageAnalyzer:
    """``create_image_analyzer`` hands the typed spec to the class the kind names."""

    def test_camera_analyzer_gets_spec_and_output_name(self):
        diag = AnalysisDiagnostic(
            name="UC_TestBeam",
            analyzer={"kind": "beam", "compute_slopes": True},
            image={"type": "camera", "bit_depth": 16},
        )
        analyzer = create_image_analyzer(diag)
        assert analyzer.__class__.__name__ == "BeamAnalyzer"
        assert analyzer.output_name == "UC_TestBeam"
        assert analyzer.analysis_config.compute_slopes is True

    def test_line_analyzer_built_from_typed_image(self):
        diag = AnalysisDiagnostic(
            name="UC_TestLine",
            analyzer={"kind": "line"},
            image={
                "type": "line",
                "data_loading": {"data_type": "tsv"},
                "background": {"method": "constant", "constant_level": 0.0},
            },
        )
        analyzer = create_image_analyzer(diag)
        assert analyzer.__class__.__name__ == "LineAnalyzer"
        assert analyzer.line_config.background.constant_level == 0.0

    def test_required_spec_fields_reach_the_constructor(self):
        diag = AnalysisDiagnostic(
            name="Master",
            analyzer={
                "kind": "line_stitcher",
                "sibling_devices": ["A", "B"],
                "output_label": "Stitched",
            },
            image={"type": "line", "data_loading": {"data_type": "tsv"}},
        )
        analyzer = create_image_analyzer(diag)
        assert isinstance(diag.analyzer, LineStitcherSpec)
        assert analyzer.sibling_devices == ["A", "B"]
        assert analyzer.name == "Stitched"  # the stitched-output folder label

    def test_stitcher_label_defaults_to_output_name(self):
        diag = AnalysisDiagnostic(
            name="Master",
            output_name="Master-stitched",
            analyzer={"kind": "line_stitcher", "sibling_devices": ["A"]},
            image={"type": "line", "data_loading": {"data_type": "tsv"}},
        )
        assert create_image_analyzer(diag).name == "Master-stitched"

    def test_trace_kind_is_the_plain_1d_analyzer(self):
        diag = AnalysisDiagnostic(
            name="Spectro",
            analyzer={"kind": "trace"},
            image={"type": "line", "data_loading": {"data_type": "csv"}},
        )
        assert create_image_analyzer(diag).__class__.__name__ == "Standard1DAnalyzer"

    def test_parameterless_kinds_take_no_spec(self):
        diag = AnalysisDiagnostic(
            name="Plain", analyzer={"kind": "standard"}, image={"type": "camera"}
        )
        analyzer = create_image_analyzer(diag)
        assert analyzer.__class__.__name__ == "StandardAnalyzer"

    def test_no_image_kind_without_image_section(self):
        diag = AnalysisDiagnostic(
            name="Phase",
            analyzer={
                "kind": "phase_downramp",
                "pixel_scale": 2.0,
                "wavelength_nm": 800.0,
            },
        )
        assert diag.image is None
        analyzer = create_image_analyzer(diag)
        assert analyzer.config.pixel_scale == 2.0


class TestOutputNameAndMetricSuffix:
    """Per-device output-naming controls (issue #412) on the document."""

    def _diag(self, **fields):
        return AnalysisDiagnostic(
            name="UC_TopView",
            analyzer={"kind": "beam"},
            image={"type": "camera", "bit_depth": 16},
            **fields,
        )

    def test_output_name_defaults_to_name(self):
        diag = self._diag()
        assert diag.output_name is None
        assert diag.effective_output_name == "UC_TopView"

    def test_explicit_output_name_overrides_name(self):
        assert self._diag(output_name="UC_TopView_left").effective_output_name == (
            "UC_TopView_left"
        )

    def test_metric_suffix_passes_through(self):
        assert self._diag().metric_suffix is None
        assert self._diag(metric_suffix="_v1").metric_suffix == "_v1"

    def test_empty_string_output_name_is_distinct_from_none(self):
        """``output_name=""`` is an explicit choice for unprefixed output."""
        assert self._diag(output_name="").effective_output_name == ""
