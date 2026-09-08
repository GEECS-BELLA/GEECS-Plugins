"""AnalysisDiagnostic / AnalysisGroup: the v2 shape and the cross-section checks.

The corpus walk in ``test_analysis_corpus.py`` covers the real files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY
from geecs_schemas.analysis import (
    ANALYZER_SPECS,
    AnalysisDiagnostic,
    AnalysisGroup,
    BeamAnalyzerSpec,
    CameraConfig,
    RendererOptions,
)
from geecs_schemas.analysis.analyzers import AnalyzerSpec

FIXTURES = Path(__file__).parent / "fixtures" / "analysis_diagnostics"


class TestRegistry:
    def test_registered_document_kinds(self):
        assert SCHEMA_REGISTRY["analysis_diagnostic"] is AnalysisDiagnostic
        assert SCHEMA_REGISTRY["analysis_group"] is AnalysisGroup

    def test_every_union_member_has_a_distinct_kind(self):
        assert len(ANALYZER_SPECS) == 15
        for kind, model in ANALYZER_SPECS.items():
            assert model.model_fields["kind"].default == kind
            assert model.image_kind in ("camera", "line", None)


class TestV2Shape:
    def test_minimal_camera_document(self):
        diag = AnalysisDiagnostic.model_validate(
            {"name": "Cam", "analyzer": {"kind": "beam"}, "image": {"type": "camera"}}
        )
        assert diag.schema_version == 2
        assert isinstance(diag.analyzer, BeamAnalyzerSpec)
        assert isinstance(diag.image, CameraConfig)
        assert diag.image.pipeline == []
        assert diag.scan.priority == 100 and diag.scan.mode == "per_shot"
        assert diag.effective_output_name == "Cam"
        assert diag.image_kind == "camera"

    def test_no_image_analyzer_without_image(self):
        diag = AnalysisDiagnostic.model_validate(
            {
                "name": "Haso",
                "analyzer": {"kind": "haso", "wavekit_config_file_path": "/x.dat"},
            }
        )
        assert diag.image is None and diag.image_kind is None
        assert diag.analyzer.mask.top == 1 and diag.analyzer.mask.bottom == -1

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ValidationError, match="kind"):
            AnalysisDiagnostic.model_validate(
                {"name": "x", "analyzer": {"kind": "no_such_analyzer"}}
            )

    def test_unknown_parameter_is_refused(self):
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            AnalysisDiagnostic.model_validate(
                {
                    "name": "x",
                    "analyzer": {"kind": "beam", "compute_slope": True},
                    "image": {"type": "camera"},
                }
            )

    @pytest.mark.parametrize(
        "kind, image, message",
        [
            ("beam", None, "needs image.type 'camera'"),
            (
                "beam",
                {"type": "line", "data_loading": {"data_type": "tsv"}},
                "needs image.type 'camera'",
            ),
            ("ict", {"type": "camera"}, "needs image.type 'line'"),
            ("haso", {"type": "camera"}, "takes no image section"),
        ],
    )
    def test_analyzer_and_image_kind_must_agree(self, kind, image, message):
        analyzer = {"kind": kind}
        if kind == "haso":
            analyzer["wavekit_config_file_path"] = "/x.dat"
        with pytest.raises(ValidationError, match=message):
            AnalysisDiagnostic.model_validate(
                {"name": "x", "analyzer": analyzer, "image": image}
            )

    def test_renderer_line_only_fields_refused_on_camera(self):
        with pytest.raises(ValidationError, match="line diagnostics only"):
            AnalysisDiagnostic.model_validate(
                {
                    "name": "x",
                    "analyzer": {"kind": "beam"},
                    "image": {"type": "camera"},
                    "scan": {"renderer": {"mode": "waterfall"}},
                }
            )

    def test_renderer_camera_only_fields_refused_on_line(self):
        with pytest.raises(ValidationError, match="camera diagnostics only"):
            AnalysisDiagnostic.model_validate(
                {
                    "name": "x",
                    "analyzer": {"kind": "line"},
                    "image": {"type": "line", "data_loading": {"data_type": "tsv"}},
                    "scan": {"renderer": {"figsize": [4, 4]}},
                }
            )

    def test_renderer_as_kwargs_passes_only_set_options(self):
        opts = RendererOptions(cmap="RdBu_r", colormap_mode="diverging")
        assert opts.as_kwargs() == {"cmap": "RdBu_r", "colormap_mode": "diverging"}

    def test_v2_round_trip_is_stable(self):
        diag = AnalysisDiagnostic.model_validate(
            {
                "name": "UC_Spec",
                "analyzer": {"kind": "beam", "compute_slopes": True},
                "image": {
                    "type": "camera",
                    "bit_depth": 16,
                    "roi": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 50},
                    "thresholding": {"method": "constant", "value": 3.0},
                    "pipeline": ["roi", "thresholding"],
                },
                "scan": {"priority": 5, "renderer": {"cmap": "viridis", "vmax": 100.0}},
            }
        )
        dumped = diag.model_dump(mode="json")
        again = AnalysisDiagnostic.model_validate(dumped)
        assert again == diag
        assert again.model_dump(mode="json") == dumped

    def test_v1_layout_is_refused(self):
        v1 = {
            "name": "UC_Cam",
            "image_analyzer": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
            "image": {"type": "camera", "bit_depth": 16},
        }
        with pytest.raises(ValidationError, match="pre-v2"):
            AnalysisDiagnostic.model_validate(v1)
        with pytest.raises(ValidationError, match="pre-v2"):
            AnalysisDiagnostic.model_validate(
                {
                    "schema_version": 1,
                    "name": "x",
                    "analyzer": {"kind": "beam"},
                    "image": {"type": "camera"},
                }
            )

    def test_newer_schema_version_is_kept(self):
        diag = AnalysisDiagnostic.model_validate(
            {
                "schema_version": 3,
                "name": "x",
                "analyzer": {"kind": "beam"},
                "image": {"type": "camera"},
            }
        )
        assert diag.schema_version == 3

    def test_json_schema_exports_every_kind(self):
        schema = AnalysisDiagnostic.model_json_schema()
        defs = schema["$defs"]
        for model in ANALYZER_SPECS.values():
            assert model.__name__ in defs
        assert "AnalysisDiagnostic" in schema.get("title", "AnalysisDiagnostic")


class TestAnalysisGroup:
    def test_bare_ids_expand(self):
        group = AnalysisGroup.model_validate(
            yaml.safe_load((FIXTURES / "groups" / "baseline.yaml").read_text())
        )
        assert group.schema_version == 1
        assert [r.ref for r in group.analyzers] == [
            "Amp2Input",
            "Amp4Output",
            "UC_TopView",
        ]
        assert [r.enabled for r in group.analyzers] == [True, False, True]
        assert [r.priority for r in group.analyzers] == [None, None, 3]
        assert group.upload_to_scanlog is True

    def test_round_trip(self):
        group = AnalysisGroup(name="g", analyzers=[{"ref": "a"}, "b"])
        dumped = group.model_dump(mode="json")
        assert AnalysisGroup.model_validate(dumped) == group

    def test_unknown_key_refused(self):
        with pytest.raises(ValidationError):
            AnalysisGroup.model_validate({"name": "g", "analysers": []})


def test_analyzer_spec_union_is_the_registry_source():
    import typing

    members = set(typing.get_args(typing.get_args(AnalyzerSpec)[0]))
    assert members == set(ANALYZER_SPECS.values())
