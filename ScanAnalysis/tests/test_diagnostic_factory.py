"""The diagnostic factory: an analysis document → its ScanAnalyzer, one route."""

from __future__ import annotations

import pytest
import yaml
from geecs_analysis.compat.v2 import UnsupportedRecipe
from geecs_analysis.recipe import RecipeError
from geecs_schemas.analysis import AnalysisRecipe

from scan_analysis.analyzers.common.himg_kinds import HimgToStackAnalyzer
from scan_analysis.config import load_diagnostic
from scan_analysis.config.diagnostic_factory import (
    DestructiveKindRefused,
    create_scan_analyzer,
)
from scan_analysis.core_analyzer import CoreScanAnalyzer


def recipe(*, device="UC_Test", kind="camera", scan=None, **patch) -> AnalysisRecipe:
    """A minimal v3 recipe: a beam measure over a camera, or a line measure over a trace."""
    document = {
        "schema_version": 3,
        "device": device,
        "input": {"kind": kind}
        if kind == "camera"
        else {"kind": "line", "loading": {"data_type": "csv"}},
        "measure": {"kind": "beam" if kind == "camera" else "line"},
        "scan": scan or {},
        **patch,
    }
    return AnalysisRecipe.model_validate(document)


class TestTheCoreRoute:
    """Every analysis document becomes a CoreScanAnalyzer, compiled at construction."""

    @pytest.mark.parametrize("kind", ["camera", "line"])
    def test_a_recipe_builds_on_the_core(self, kind):
        analyzer = create_scan_analyzer(recipe(kind=kind))
        assert isinstance(analyzer, CoreScanAnalyzer)
        assert analyzer.device_name == "UC_Test"

    def test_the_core_keeps_its_own_document_copy(self):
        doc = recipe(
            device="UC_Copy", scan={"average_frames_first": True, "save": False}
        )
        analyzer = create_scan_analyzer(doc)
        assert analyzer.document == doc and analyzer.document is not doc
        assert analyzer.spec.average_frames_first is True
        assert analyzer.document.scan.save is False

    def test_a_recipe_the_core_cannot_bind_is_refused_before_any_scan(self):
        with pytest.raises(RecipeError, match="does not bind"):
            create_scan_analyzer(recipe(steps=[{"step": "sharpen"}]))

    def test_a_job_document_never_reaches_the_core(self):
        with pytest.raises(UnsupportedRecipe):
            CoreScanAnalyzer(load_job_document("himg_to_stack"), id="Job", priority=1)

    def test_no_route_or_injected_data_keywords(self):
        """#1039: the legacy escape hatches are gone from the signature."""
        with pytest.raises(TypeError):
            create_scan_analyzer(recipe(), route="legacy")
        with pytest.raises(TypeError):
            create_scan_analyzer(recipe(), use_injected_data=True)


def load_job_document(kind: str):
    from geecs_schemas.analysis import load_analysis_document

    return load_analysis_document(
        {"kind": kind, "device": "U_HasoLift", "scan": {"priority": 5}}
    )


class TestRuntimeAttachment:
    """id / priority defaults and overrides, on the core and the job route."""

    def test_id_defaults_to_the_device(self):
        assert create_scan_analyzer(recipe(device="UC_Foo")).id == "UC_Foo"

    def test_id_kwarg_overrides_the_device(self):
        assert create_scan_analyzer(recipe(device="UC_Foo"), id="MyDiag").id == "MyDiag"

    def test_id_defaults_to_a_loaded_documents_source_id(self, tmp_path):
        path = tmp_path / "PW-MagSpectStitcher.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 3,
                    "device": "CAM-TEA-MagSpecA-interpSpec",
                    "input": {"kind": "line", "loading": {"data_type": "csv"}},
                    "measure": {"kind": "line"},
                    "scan": {"priority": 7},
                }
            )
        )
        analyzer = create_scan_analyzer(load_diagnostic(path))
        assert analyzer.id == "PW-MagSpectStitcher"
        assert analyzer.device_name == "CAM-TEA-MagSpecA-interpSpec"
        assert analyzer.priority == 7

    def test_priority_defaults_to_the_documents_and_the_kwarg_overrides_it(self):
        assert create_scan_analyzer(recipe(scan={"priority": 7})).priority == 7
        assert (
            create_scan_analyzer(recipe(scan={"priority": 7}), priority=99).priority
            == 99
        )

    def test_a_job_gets_its_class_id_and_priority(self):
        analyzer = create_scan_analyzer(load_job_document("himg_to_stack"), id="Stack")
        assert isinstance(analyzer, HimgToStackAnalyzer)
        assert (analyzer.id, analyzer.priority) == ("Stack", 5)
        assert analyzer.background_source is None


class TestTheDestructiveGate:
    """A kind that deletes data files is built only for a host that confirmed it."""

    def test_the_compaction_is_refused_without_the_opt_in(self):
        with pytest.raises(DestructiveKindRefused, match="deletes data files"):
            create_scan_analyzer(load_job_document("himg_compact"))

    def test_the_opt_in_builds_it(self):
        from scan_analysis.analyzers.common.himg_kinds import HimgCompactAnalyzer

        analyzer = create_scan_analyzer(
            load_job_document("himg_compact"), allow_destructive=True
        )
        assert isinstance(analyzer, HimgCompactAnalyzer)
