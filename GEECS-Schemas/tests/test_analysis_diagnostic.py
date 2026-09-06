"""AnalysisDiagnostic / AnalysisGroup: the v2 shape, the v1 converter, and the cross-section checks.

Fixtures under ``tests/fixtures/analysis_diagnostics/v1/`` are sanitized
copies of real GEECS-Plugins-Configs files in the pre-0.19.0 layout; they
go through the one-shot converter (``geecs_schemas.convert.
analysis_diagnostics``), never through the model directly — the model
refuses v1.  The corpus walk in ``test_analysis_corpus.py`` covers the
real files.
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
    HasoAnalyzerSpec,
    Line1DConfig,
    LineStitcherSpec,
    MagSpecAnalyzerSpec,
    RendererOptions,
)
from geecs_schemas.analysis.analyzers import AnalyzerSpec, DnnAxisCalibrationSpec
from geecs_schemas.convert import SchemaConversionError, convert_v1_diagnostic
from geecs_schemas.convert.analysis_diagnostics import V1_CLASS_PATH_TO_KIND

FIXTURES = Path(__file__).parent / "fixtures" / "analysis_diagnostics"


def load_v1(stem: str) -> dict:
    return yaml.safe_load((FIXTURES / "v1" / f"{stem}.yaml").read_text())


def lift(stem: str) -> AnalysisDiagnostic:
    """Convert a v1 fixture with the one-shot converter and validate the result."""
    return AnalysisDiagnostic.model_validate(convert_v1_diagnostic(load_v1(stem)))


class TestRegistry:
    def test_registered_document_kinds(self):
        assert SCHEMA_REGISTRY["analysis_diagnostic"] is AnalysisDiagnostic
        assert SCHEMA_REGISTRY["analysis_group"] is AnalysisGroup

    def test_every_union_member_has_a_distinct_kind(self):
        assert len(ANALYZER_SPECS) == 15
        for kind, model in ANALYZER_SPECS.items():
            assert model.model_fields["kind"].default == kind
            assert model.image_kind in ("camera", "line", None)

    def test_every_v1_class_path_maps_to_a_registered_kind(self):
        assert set(V1_CLASS_PATH_TO_KIND.values()) <= set(ANALYZER_SPECS)
        # every kind reachable from some v1 class path (nothing orphaned)
        assert set(V1_CLASS_PATH_TO_KIND.values()) == set(ANALYZER_SPECS)


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
        diag = lift("magspec_dnn")
        dumped = diag.model_dump(mode="json")
        again = AnalysisDiagnostic.model_validate(dumped)
        assert again == diag
        assert again.model_dump(mode="json") == dumped

    def test_v1_layout_is_refused_with_a_pointer_to_the_converter(self):
        with pytest.raises(ValidationError, match="convert.analysis_diagnostics"):
            AnalysisDiagnostic.model_validate(load_v1("beam_camera"))
        with pytest.raises(ValidationError, match="convert.analysis_diagnostics"):
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


class TestV1Converter:
    def test_beam_camera(self):
        diag = lift("beam_camera")
        assert diag.schema_version == 2
        assert diag.analyzer == BeamAnalyzerSpec()
        image = diag.image
        assert isinstance(image, CameraConfig)
        assert [s.value for s in image.pipeline][:3] == [
            "background",
            "vignette",
            "crosshair_masking",
        ]
        assert image.background.constant_level == 5.0
        assert image.metadata["spatial_calibration"] == 2.44e-05
        assert diag.scan.priority == 10

    def test_ict_line_params_and_renames(self):
        diag = lift("ict_line")
        assert diag.analyzer.kind == "ict"
        assert diag.analyzer.calibration_factor == 0.2
        assert diag.analyzer.dt == 4e-9
        image = diag.image
        assert isinstance(image, Line1DConfig)
        assert image.label == "volts (V) vs time (s)"
        assert image.background.constant_level == 0.0
        assert image.pipeline == []
        assert diag.scan.renderer.cmap == "RdBu_r"
        assert diag.scan.renderer.colormap_mode == "diverging"
        assert diag.scan.file_tail == ".tdms"

    def test_frog_camera(self):
        diag = lift("frog_camera")
        assert diag.analyzer.kind == "frog_retrieval"
        assert diag.analyzer.max_time_seconds == 15
        assert diag.analyzer.max_iterations == 1_000_000_000
        assert diag.output_name == "U_FROG_Grenouille-Temporal"
        assert diag.scan.device == "U_FROG_Grenouille-Temporal"
        assert diag.scan.save is False

    def test_haso_kwargs_become_spec_fields(self):
        diag = lift("haso_noimage")
        assert isinstance(diag.analyzer, HasoAnalyzerSpec)
        assert diag.image is None
        assert diag.analyzer.mask.model_dump() == {
            "top": 125,
            "bottom": 300,
            "left": 10,
            "right": 670,
        }
        assert str(diag.analyzer.wavekit_config_file_path).endswith(
            "WFS_HASO4_LIFT.dat"
        )
        assert diag.scan.file_tail == ".himg"

    def test_line_stitcher_keeps_its_output_label(self):
        # v1's ``name`` kwarg labelled the stitched-output folder next to the
        # master device; dropping it would make the stitcher write over its
        # own raw input files (the deployed stitchers set it to something
        # other than the diagnostic name).
        diag = lift("line_stitcher")
        assert isinstance(diag.analyzer, LineStitcherSpec)
        assert diag.analyzer.output_label == "HTT-MagSpecStitcher"
        assert len(diag.analyzer.sibling_devices) == 3
        assert diag.image.filtering.kernel_size == 15
        assert diag.image.roi.x_max == 500

    def test_magspec_nested_calibration_and_background_source(self):
        diag = lift("magspec_dnn")
        assert isinstance(diag.analyzer, MagSpecAnalyzerSpec)
        assert isinstance(diag.analyzer.calibration, DnnAxisCalibrationSpec)
        assert diag.analyzer.calibration.camera_number == 1
        assert diag.analyzer.energy_range == (1.2, 1501.3)
        assert diag.image.vignette.full_width == 1384
        assert diag.image.background.method is None
        assert diag.image.background.additional_constant == 1.0
        assert diag.scan.background_source.scan_number == 1

    def test_legacy_nested_magspec_block_is_flattened(self):
        data = load_v1("magspec_dnn")
        data["image"]["analysis"] = {"magspec": data["image"]["analysis"]}
        diag = AnalysisDiagnostic.model_validate(convert_v1_diagnostic(data))
        assert diag.analyzer.num_energy_points == 1000

    def test_ignored_v1_keys_become_errors(self):
        # U_FROG_Beam in the real corpus: FROG keys under a BeamAnalyzer that
        # v1 silently ignored.
        with pytest.raises(SchemaConversionError) as excinfo:
            lift("beam_with_frog_keys")
        message = str(excinfo.value)
        for key in ("delt", "dellam", "lam0", "N"):
            assert key in message

    def test_line_stitcher_label_may_not_be_the_master_folder(self):
        with pytest.raises(ValidationError, match="overwrite the raw input"):
            AnalysisDiagnostic.model_validate(
                {
                    "name": "MagCam1",
                    "analyzer": {
                        "kind": "line_stitcher",
                        "sibling_devices": ["MagCam2"],
                    },
                    "image": {"type": "line", "data_loading": {"data_type": "tsv"}},
                }
            )
        ok = AnalysisDiagnostic.model_validate(
            {
                "name": "MagCam1",
                "output_name": "Stitched",
                "analyzer": {"kind": "line_stitcher", "sibling_devices": ["MagCam2"]},
                "image": {"type": "line", "data_loading": {"data_type": "tsv"}},
            }
        )
        assert ok.analyzer.output_label is None  # output_name carries the label

    def test_explicit_schema_version_1_converts_too(self):
        data = load_v1("beam_camera")
        data["schema_version"] = 1
        diag = AnalysisDiagnostic.model_validate(convert_v1_diagnostic(data))
        assert diag.schema_version == 2
        assert diag.analyzer.kind == "beam"

    def test_standard_1d_is_the_trace_kind(self):
        data = load_v1("ict_line")
        data["image_analyzer"] = (
            "image_analysis.analyzers.standard_1d_analyzer.Standard1DAnalyzer"
        )
        data["image"].pop("analysis")
        assert convert_v1_diagnostic(data)["analyzer"] == {"kind": "trace"}

    def test_data1d_loading_refuses_negative_columns(self):
        from geecs_schemas.analysis import Data1DLoading

        with pytest.raises(ValidationError, match="non-negative"):
            Data1DLoading(data_type="tsv", auxiliary_columns={"w": -3})

    def test_unknown_class_path_is_refused_with_the_kind_list(self):
        data = load_v1("beam_camera")
        data["image_analyzer"] = "some.module.NewAnalyzer"
        with pytest.raises(SchemaConversionError, match="unknown analyzer class"):
            convert_v1_diagnostic(data)

    def test_converter_output_is_canonical(self):
        # set fields only, no default-None noise, schema_version first
        document = convert_v1_diagnostic(load_v1("beam_camera"))
        assert list(document)[0] == "schema_version"
        assert "crosshair_masking" not in document["image"]
        assert "file_path" not in document["image"]["background"]
        assert document["analyzer"] == {"kind": "beam"}

    def test_converted_documents_round_trip(self):
        for stem in (
            "beam_camera",
            "ict_line",
            "frog_camera",
            "haso_noimage",
            "line_stitcher",
            "magspec_dnn",
        ):
            document = convert_v1_diagnostic(load_v1(stem))
            diag = AnalysisDiagnostic.model_validate(document)
            assert "image_analyzer" not in document
            # converting a v2 document is a no-op
            assert convert_v1_diagnostic(document) == document
            assert (
                AnalysisDiagnostic.model_validate(diag.model_dump(mode="json")) == diag
            )

    def test_converter_does_not_mutate_the_input(self):
        data = load_v1("ict_line")
        before = yaml.safe_dump(data)
        convert_v1_diagnostic(data)
        assert yaml.safe_dump(data) == before


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
