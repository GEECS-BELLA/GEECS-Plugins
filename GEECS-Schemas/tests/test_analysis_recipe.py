"""AnalysisRecipe (format 3): the shape, its refusals, and the version-dispatching loader.

The corpus walk in ``test_analysis_corpus.py`` covers the real files.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY
from geecs_schemas.analysis import (
    SUMMARY_KINDS,
    AnalysisDiagnostic,
    AnalysisRecipe,
    AverageSummary,
    CameraInput,
    ImageGridSummary,
    LineInput,
    WaterfallSummary,
    canonical_document,
    load_analysis_document,
)
from geecs_schemas.schema_export import EXPORTED_SCHEMAS

CAMERA = {
    "schema_version": 3,
    "device": "UC_Test",
    "input": {"kind": "camera"},
    "steps": [
        {"step": "background_constant", "level": 5},
        {"step": "roi", "bounds": [[0, 10], [0, 20]]},
        {"step": "median", "kernel": 3},
    ],
    "measure": {"kind": "beam", "compute_slopes": True},
    "figure": {"imshow": {"vmin": 0}, "overlays": {"com": {"hidden": True}}},
    "summaries": [{"kind": "image_grid", "columns": 2}, {"kind": "average"}],
}
LINE = {
    "schema_version": 3,
    "device": "Spec",
    "output_name": "Spec-interp",
    "scalar_suffix": "_left",
    "input": {
        "kind": "line",
        "folder": "Spec-interp",
        "file_tail": ".txt",
        "loading": {"data_type": "tsv"},
        "x_scale": 1000.0,
        "x_unit": "MeV",
        "label": "Charge density",
    },
    "measure": {"kind": "line"},
    "summaries": [{"kind": "waterfall", "sort_key": "U_Charge"}, {"kind": "average"}],
}
V2 = {"name": "UC_Test", "analyzer": {"kind": "beam"}, "image": {"type": "camera"}}


class TestRegistry:
    def test_registered_and_exported(self):
        assert SCHEMA_REGISTRY["analysis_recipe"] is AnalysisRecipe
        assert EXPORTED_SCHEMAS["analysis_recipe"] is AnalysisRecipe

    def test_summary_kinds_are_the_frozen_three(self):
        assert SUMMARY_KINDS == {
            "image_grid": ImageGridSummary,
            "waterfall": WaterfallSummary,
            "average": AverageSummary,
        }
        assert ImageGridSummary.frame_ndim == {2}
        assert WaterfallSummary.frame_ndim == {1}
        assert AverageSummary.frame_ndim == {1, 2}
        assert CameraInput.ndim == 2 and LineInput.ndim == 1


class TestShape:
    def test_camera_recipe_keeps_step_parameters_as_written(self):
        recipe = AnalysisRecipe.model_validate(CAMERA)
        assert [s.step for s in recipe.steps] == [
            "background_constant",
            "roi",
            "median",
        ]
        assert recipe.steps[1].parameters() == {"bounds": [[0, 10], [0, 20]]}
        assert recipe.measure.kind == "beam"
        assert recipe.measure.parameters() == {"compute_slopes": True}
        assert recipe.effective_output_name == "UC_Test"
        assert recipe.input_kind == "camera"
        assert recipe.figure.overlays == {"com": {"hidden": True}}
        assert [s.kind for s in recipe.summaries] == ["image_grid", "average"]
        assert recipe.summaries[0].columns == 2

    def test_line_recipe_reads_its_loader_and_naming(self):
        recipe = AnalysisRecipe.model_validate(LINE)
        assert isinstance(recipe.input, LineInput)
        assert recipe.input.loading.data_type.value == "tsv"
        assert recipe.input.storage_dtype == "float32"
        assert recipe.effective_output_name == "Spec-interp"
        assert recipe.scalar_suffix == "_left"
        assert recipe.summaries[0].sort_key == "U_Charge"
        assert recipe.summaries[0].sort_sigma == 3.0

    def test_defaults_are_minimal(self):
        recipe = AnalysisRecipe.model_validate(
            {"device": "D", "input": {"kind": "camera"}}
        )
        assert recipe.schema_version == 3
        assert recipe.steps == [] and recipe.measure.kind == "none"
        assert recipe.summaries == [] and recipe.inputs == {}
        assert recipe.scan.priority == 100 and recipe.scan.save
        assert not recipe.scan.average_frames_first

    def test_canonical_form_round_trips_with_step_parameters(self):
        recipe = AnalysisRecipe.model_validate(CAMERA)
        written = canonical_document(recipe)
        assert list(written)[0] == "schema_version"
        assert written["steps"][1] == {"step": "roi", "bounds": [[0, 10], [0, 20]]}
        assert (
            AnalysisRecipe.model_validate(yaml.safe_load(yaml.safe_dump(written)))
            == recipe
        )
        # dumping the dict form again is stable
        assert canonical_document(AnalysisRecipe.model_validate(written)) == written


class TestRefusals:
    @pytest.mark.parametrize(
        "patch,needle",
        [
            ({"summaries": [{"kind": "waterfall"}]}, "does not draw camera"),
            ({"summaries": [{"kind": "gif"}]}, "summaries.0"),
            ({"steps": [{"bounds": [[0, 1]]}]}, "step"),
            ({"bogus": 1}, "bogus"),
            ({"schema_version": 2}, "not an analysis recipe"),
            ({"scan": {"mode": "per_bin"}}, "scan.mode"),
        ],
    )
    def test_shape_errors(self, patch, needle):
        with pytest.raises(ValidationError, match=needle):
            AnalysisRecipe.model_validate({**CAMERA, **patch})

    def test_image_grid_refused_on_a_line_recipe(self):
        with pytest.raises(ValidationError, match="does not draw line"):
            AnalysisRecipe.model_validate(
                {**LINE, "summaries": [{"kind": "image_grid"}]}
            )

    def test_capture_stack_needs_both_switches(self):
        stack = {**LINE["input"], "loading": {"data_type": "pva_stack"}}
        with pytest.raises(ValidationError, match="reads nothing"):
            AnalysisRecipe.model_validate({**LINE, "input": stack})
        AnalysisRecipe.model_validate(
            {**LINE, "input": {**stack, "format": "device_hdf5"}}
        )

    def test_a_v2_document_is_refused_with_a_pointer(self):
        with pytest.raises(ValidationError, match="load_analysis_document"):
            AnalysisRecipe.model_validate(V2)


class TestLoader:
    def test_dispatches_on_schema_version(self):
        assert isinstance(load_analysis_document(CAMERA), AnalysisRecipe)
        assert isinstance(load_analysis_document(V2), AnalysisDiagnostic)
        assert isinstance(
            load_analysis_document({**V2, "schema_version": "2"}), AnalysisDiagnostic
        )
        assert isinstance(
            load_analysis_document({**CAMERA, "schema_version": "3"}), AnalysisRecipe
        )

    def test_a_v3_file_handed_to_the_v2_model_is_refused(self):
        with pytest.raises(ValidationError):
            AnalysisDiagnostic.model_validate(CAMERA)

    def test_the_v1_layout_is_still_refused(self):
        with pytest.raises(ValidationError, match="pre-v2"):
            load_analysis_document({"image_analyzer": "x.y:Z", "name": "D"})


class TestNeutralNames:
    """Both formats answer device / input_kind / data_folder / line_loading the same way."""

    def test_recipe_and_diagnostic_agree(self):
        recipe = AnalysisRecipe.model_validate(LINE)
        diagnostic = AnalysisDiagnostic.model_validate(
            {
                "name": "Spec",
                "output_name": "Spec-interp",
                "analyzer": {"kind": "line"},
                "image": {"type": "line", "data_loading": {"data_type": "tsv"}},
                "scan": {"device": "Spec-interp"},
            }
        )
        for doc in (recipe, diagnostic):
            assert doc.device == "Spec"
            assert doc.input_kind == "line"
            assert doc.data_folder == "Spec-interp"
            assert doc.effective_output_name == "Spec-interp"
            assert doc.line_loading.data_type.value == "tsv"
        camera = AnalysisRecipe.model_validate(CAMERA)
        assert camera.data_folder == "UC_Test" and camera.line_loading is None
        assert AnalysisDiagnostic.model_validate(V2).line_loading is None

    def test_declared_version_is_the_base_parse(self):
        from geecs_schemas._base import declared_schema_version, stale_schema_version

        assert declared_schema_version({"schema_version": "3"}) == 3
        assert declared_schema_version({"schema_version": True}) is None
        assert declared_schema_version({}) is None
        assert stale_schema_version({"schema_version": "1"}, 2)
        assert not stale_schema_version({}, 2)
