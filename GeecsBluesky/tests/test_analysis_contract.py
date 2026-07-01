"""Tests for GeecsBluesky post-run analysis result contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import png
import pytest

from geecs_bluesky.analysis import (
    AnalysisArtifactWriter,
    AnalysisInvocationMetadata,
    AnalysisResult,
    AnalysisScope,
    DerivedAssetRef,
    FeatureRow,
    ImageAnalyzerAdapter,
    InputAssetRef,
    build_analysis_run_documents,
    run_asset_analysis_for_tiled_run,
    run_camera_image_analysis_for_tiled_run,
    resolve_analysis_config_dir,
    resolve_image_analysis_config_dir,
)
from geecs_bluesky.analysis.runner import AnalysisRunner, FilledAsset
from geecs_bluesky.analysis.writer import required_feature_columns
from geecs_bluesky.assets.specs import (
    MAGSPEC_CAMERA_DEVICE_TYPE,
    PICOSCOPE_V2_DEVICE_TYPE,
    POINTGREY_CAMERA_DEVICE_TYPE,
)
from geecs_data_utils.io.array1d import Data1DConfig, Data1DResult, Data1DType


def test_feature_row_flattens_required_columns_and_features() -> None:
    """Feature rows should preserve raw-run identity and analyzer scalars."""
    row = FeatureRow(
        raw_run_uid="run-uid",
        event_uid="event-uid",
        scan_number=6,
        scan_event_index=3,
        shot_number=3,
        device="UC_Amp2_IR_input",
        data_key="uc_amp2_ir_input-image",
        datum_id="resource/0",
        asset_spec="GEECS_CAMERA_IMAGE",
        analyzer_id="beam_centroid_v1",
        features={"centroid_x_px": 12.5},
        derived_assets={"overlay": "assets/shot_000003_overlay.png"},
    )

    flat = row.to_flat_dict()

    for column in required_feature_columns():
        assert column in flat
    assert flat["feature:centroid_x_px"] == 12.5
    assert flat["asset:overlay"] == "assets/shot_000003_overlay.png"


def test_writer_creates_analysis_invocation_without_scan_folder(
    tmp_path: Path,
) -> None:
    """Analysis output should be created under day-level analysis, not scans."""
    day = tmp_path / "Undulator" / "Y2026" / "06-Jun" / "26_0624"
    day.mkdir(parents=True)
    writer = AnalysisArtifactWriter(
        day_analysis_dir=day / "analysis",
        scan_number=6,
        analyzer_id="beam_centroid_v1",
        invocation_id="20260628T190012Z",
    )

    writer.prepare()

    assert writer.output_dir == (
        day / "analysis" / "Scan006" / "beam_centroid_v1" / "20260628T190012Z"
    )
    assert writer.assets_dir.is_dir()
    assert not (day / "scans").exists()


def test_writer_round_trips_metadata_and_jsonl_features(tmp_path: Path) -> None:
    """Writer should persist provenance and dependency-light feature rows."""
    day = tmp_path / "Undulator" / "Y2026" / "06-Jun" / "26_0624"
    day.mkdir(parents=True)
    writer = AnalysisArtifactWriter(
        day_analysis_dir=day / "analysis",
        scan_number=6,
        analyzer_id="beam_centroid_v1",
        invocation_id="20260628T190012Z",
    )
    writer.prepare()

    metadata = AnalysisInvocationMetadata(
        analysis_id="20260628T190012Z_beam_centroid_v1",
        analyzer_id="beam_centroid_v1",
        analyzer_name="Beam centroid",
        raw_run_uid="run-uid",
        scan_number=6,
        experiment="Undulator",
        config={"threshold": 100},
    )
    row = FeatureRow(
        raw_run_uid="run-uid",
        event_uid="event-uid",
        scan_number=6,
        scan_event_index=1,
        shot_number=1,
        device="UC_Amp2_IR_input",
        data_key="uc_amp2_ir_input-image",
        datum_id="resource/0",
        asset_spec="GEECS_CAMERA_IMAGE",
        analyzer_id="beam_centroid_v1",
        features={"total_counts": 42.0},
    )

    metadata_path = writer.write_metadata(metadata)
    features_path = writer.write_features_jsonl([row])

    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_features = [
        json.loads(line)
        for line in features_path.read_text(encoding="utf-8").splitlines()
    ]

    assert saved_metadata["raw_run_uid"] == "run-uid"
    assert saved_metadata["config"] == {"threshold": 100}
    assert saved_features[0]["feature:total_counts"] == 42.0


def test_runner_executes_generic_analyzer_and_writes_outputs(tmp_path: Path) -> None:
    """The runner should orchestrate filled assets without knowing image science."""
    day = tmp_path / "Undulator" / "Y2026" / "06-Jun" / "26_0624"
    day.mkdir(parents=True)
    writer = AnalysisArtifactWriter(
        day_analysis_dir=day / "analysis",
        scan_number=6,
        analyzer_id="toy_analyzer_v1",
        invocation_id="20260628T190012Z",
    )
    runner = AnalysisRunner(writer=writer)
    asset = InputAssetRef(
        raw_run_uid="run-uid",
        event_uid="event-uid",
        scan_number=6,
        scan_event_index=2,
        shot_number=2,
        device="UC_Amp2_IR_input",
        data_key="uc_amp2_ir_input-image",
        datum_id="resource/0",
        asset_spec="GEECS_CAMERA_IMAGE",
    )
    metadata = AnalysisInvocationMetadata(
        analysis_id="20260628T190012Z_toy_analyzer_v1",
        analyzer_id="toy_analyzer_v1",
        analyzer_name="Toy analyzer",
        raw_run_uid="run-uid",
        scan_number=6,
        experiment="Undulator",
    )

    runner.run(
        analyzer=_ToyAnalyzer(),
        assets=[FilledAsset(ref=asset, data=[[1, 2], [3, 4]])],
        metadata=metadata,
    )

    saved_metadata = json.loads(
        (writer.output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    saved_features = [
        json.loads(line)
        for line in (writer.output_dir / "features.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert saved_metadata["inputs"][0]["datum_id"] == "resource/0"
    assert saved_metadata["analysis_output_path"] == (
        "analysis/Scan006/toy_analyzer_v1/20260628T190012Z"
    )
    assert saved_metadata["feature_table"] == "features.jsonl"
    assert saved_features[0]["feature:pixel_sum"] == 10


def test_runner_executes_scan_context_analyzer(tmp_path: Path) -> None:
    """Scan-scope analyzers should prepare context before per-event results."""
    day = tmp_path / "Undulator" / "Y2026" / "06-Jun" / "26_0624"
    day.mkdir(parents=True)
    writer = AnalysisArtifactWriter(
        day_analysis_dir=day / "analysis",
        scan_number=6,
        analyzer_id="dynamic_background_v1",
        invocation_id="20260628T190012Z",
    )
    runner = AnalysisRunner(writer=writer)
    metadata = AnalysisInvocationMetadata(
        analysis_id="20260628T190012Z_dynamic_background_v1",
        analyzer_id="dynamic_background_v1",
        analyzer_name="Dynamic background",
        raw_run_uid="run-uid",
        scan_number=6,
        experiment="Undulator",
    )
    assets = [
        FilledAsset(
            ref=InputAssetRef(
                raw_run_uid="run-uid",
                scan_number=6,
                scan_event_index=index,
                shot_number=index,
                device="UC_Amp2_IR_input",
                data_key="uc_amp2_ir_input-image",
                datum_id=f"resource/{index}",
                asset_spec="GEECS_CAMERA_IMAGE",
            ),
            data=np.asarray(data),
        )
        for index, data in (
            (1, [[1, 2], [3, 4]]),
            (2, [[2, 3], [4, 5]]),
            (3, [[3, 4], [5, 6]]),
        )
    ]

    runner.run(
        analyzer=_DynamicBackgroundAnalyzer(),
        assets=assets,
        metadata=metadata,
    )

    saved_metadata = json.loads(
        (writer.output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    saved_features = [
        json.loads(line)
        for line in (writer.output_dir / "features.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert saved_metadata["analysis_scope"] == "scan"
    assert saved_metadata["outputs"][0]["asset_id"] == "background"
    assert [row["feature:bg_subtracted_sum"] for row in saved_features] == [-4, 0, 4]


def test_build_analysis_run_documents_links_sidecar_to_raw_run(tmp_path: Path) -> None:
    """Derived analysis run docs should point backward and forward."""
    metadata = AnalysisInvocationMetadata(
        analysis_id="20260628T190012Z_beam_centroid_v1",
        analyzer_id="beam_centroid_v1",
        analyzer_name="Beam centroid",
        raw_run_uid="raw-run-uid",
        scan_number=6,
        experiment="Undulator",
        analysis_root="Z:/data",
        analysis_output_path=(
            "Undulator/Y2026/06-Jun/26_0624/analysis/"
            "Scan006/beam_centroid_v1/20260628T190012Z"
        ),
        feature_table="features.jsonl",
        inputs=[
            InputAssetRef(
                raw_run_uid="raw-run-uid",
                scan_number=6,
                scan_event_index=1,
                shot_number=1,
                device="UC_Amp2_IR_input",
                data_key="uc_amp2_ir_input-image",
                datum_id="resource/0",
                asset_spec="GEECS_CAMERA_IMAGE",
            )
        ],
    )

    docs = build_analysis_run_documents(metadata, output_dir=tmp_path / "analysis")

    assert [name for name, _ in docs] == ["start", "descriptor", "event", "stop"]
    start = docs[0][1]
    event = docs[2][1]
    assert metadata.derived_run_uid == start["uid"]
    assert start["purpose"] == "geecs_bluesky_analysis"
    assert start["analysis_of"] == "raw-run-uid"
    assert start["analyzer_id"] == "beam_centroid_v1"
    assert start["analysis_root"] == "Z:/data"
    assert event["data"]["raw_run_uid"] == "raw-run-uid"
    assert event["data"]["analysis_root"] == "Z:/data"
    assert event["data"]["analysis_output_path"].endswith(
        "analysis/Scan006/beam_centroid_v1/20260628T190012Z"
    )
    assert event["data"]["input_count"] == 1


def test_image_analyzer_adapter_exports_feature_scalars() -> None:
    """ImageAnalysis-style analyzers should adapt to AnalysisResult features."""
    adapter = ImageAnalyzerAdapter(
        _FakeImageAnalyzer(),
        analyzer_id="fake_image_analyzer_v1",
        config={"threshold": 4},
    )
    asset = InputAssetRef(
        raw_run_uid="run-uid",
        event_uid="event-uid",
        scan_number=6,
        scan_event_index=2,
        shot_number=2,
        device="UC_Amp2_IR_input",
        data_key="uc_amp2_ir_input-image",
        datum_id="resource/0",
        asset_spec="GEECS_CAMERA_IMAGE",
    )

    result = adapter.analyze([[1, 2], [3, 4]], asset=asset, output_dir=Path("."))

    assert adapter.analyzer_name == "_FakeImageAnalyzer"
    assert adapter.describe_config() == {"threshold": 4}
    assert result.features == {"pixel_sum": 10}


def test_image_analyzer_adapter_unwraps_data_1d_result() -> None:
    """ImageAnalysis 1D analyzers should receive the loaded Nx2 line array."""
    image_analyzer = _FakeData1DImageAnalyzer()
    adapter = ImageAnalyzerAdapter(
        image_analyzer,
        analyzer_id="fake_data_1d_image_analyzer_v1",
    )
    asset = InputAssetRef(
        raw_run_uid="run-uid",
        event_uid="event-uid",
        scan_number=6,
        scan_event_index=2,
        shot_number=2,
        device="U_BCaveICT",
        data_key="u_bcaveict-tdms",
        datum_id="resource/0",
        asset_spec="GEECS_TDMS_SCOPE",
    )
    data = Data1DResult(
        data=np.asarray([[0.0, 1.0], [1.0, 2.0]]),
        x_units="s",
        y_units="V",
        x_label="Time",
        y_label="Voltage",
    )

    result = adapter.analyze(data, asset=asset, output_dir=Path("."))

    assert image_analyzer.data_metadata == {
        "x_units": "s",
        "y_units": "V",
        "x_label": "Time",
        "y_label": "Voltage",
    }
    assert result.features == {"y_sum": 3.0}

    adapter.analyze(
        Data1DResult(data=np.asarray([[0.0, 3.0], [1.0, 4.0]])),
        asset=asset,
        output_dir=Path("."),
    )

    assert image_analyzer.data_metadata == {}


def test_image_analyzer_adapter_selects_configured_text_line_columns() -> None:
    """Line analyzers should receive configured x/y columns from text assets."""
    image_analyzer = _FakeConfiguredLineImageAnalyzer()
    adapter = ImageAnalyzerAdapter(
        image_analyzer,
        analyzer_id="fake_configured_line_image_analyzer_v1",
    )
    asset = InputAssetRef(
        raw_run_uid="run-uid",
        event_uid="event-uid",
        scan_number=6,
        scan_event_index=2,
        shot_number=2,
        device="UC_BCaveMagSpecCam1",
        data_key="uc_bcavemagspeccam1-interpSpec",
        datum_id="resource/0",
        asset_spec="GEECS_TEXT_ARRAY",
    )
    data = np.asarray(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
        ]
    )

    result = adapter.analyze(data, asset=asset, output_dir=Path("."))

    assert result.features == {"x_sum": 30.0, "y_sum": 300.0}


def test_resolve_analysis_config_dir_from_geecs_config(tmp_path: Path) -> None:
    """Config lookup should prefer the unified scan-analysis config root."""
    data_root = tmp_path / "data"
    analysis_root = tmp_path / "configs" / "scan_analysis_configs"
    image_root = analysis_root / "analyzers"
    data_root.mkdir()
    image_root.mkdir(parents=True)
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "\n".join(
            [
                "[Experiment]",
                "expt = Undulator",
                "",
                "[Paths]",
                f"GEECS_DATA_LOCAL_BASE_PATH = {data_root}",
                f"scan_analysis_configs_path = {analysis_root}",
            ]
        ),
        encoding="utf-8",
    )

    resolved = resolve_analysis_config_dir(geecs_config_path=config_path)

    assert resolved == analysis_root
    assert (
        resolve_image_analysis_config_dir(geecs_config_path=config_path)
        == analysis_root
    )


def test_resolve_analysis_config_dir_requires_unified_scan_root(
    tmp_path: Path,
) -> None:
    """The deprecated ImageAnalysis root is not used as a runtime fallback."""
    data_root = tmp_path / "data"
    legacy_root = tmp_path / "configs" / "image_analysis_configs"
    data_root.mkdir()
    legacy_root.mkdir(parents=True)
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "\n".join(
            [
                "[Experiment]",
                "expt = Undulator",
                "",
                "[Paths]",
                f"GEECS_DATA_LOCAL_BASE_PATH = {data_root}",
                f"image_analysis_configs_path = {legacy_root}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scan_analysis_configs_path"):
        resolve_analysis_config_dir(geecs_config_path=config_path)


def test_camera_image_analysis_runs_from_tiled_to_sidecar(tmp_path: Path) -> None:
    """A fake Tiled camera run should produce sidecar analysis artifacts."""
    local_data_root = tmp_path / "Volumes" / "hdna2" / "data"
    day = local_data_root / "Undulator" / "Y2026" / "06-Jun" / "26_0624"
    scan_folder = day / "scans" / "Scan006"
    device_folder = scan_folder / "UC_Amp2_IR_input"
    device_folder.mkdir(parents=True)
    images = {
        1: np.array([[1, 2], [3, 4]], dtype=np.uint8),
        2: np.array([[5, 6], [7, 8]], dtype=np.uint8),
    }
    for shot, image in images.items():
        image_path = device_folder / f"UC_Amp2_IR_input_{1000 + shot:.3f}.png"
        with image_path.open("wb") as stream:
            png.Writer(width=2, height=2, greyscale=True, bitdepth=8).write(
                stream,
                image.tolist(),
            )

    start_doc = {
        "uid": "run-uid",
        "scan_number": 6,
        "scan_folder": str(scan_folder),
        "experiment": "Undulator",
    }
    table = pd.DataFrame(
        [
            {
                "scan_event_index": 1,
                "uc_amp2_ir_input-acq_timestamp": 1001.0,
                "uc_amp2_ir_input-nonscalar_save_path": str(device_folder),
                "uc_amp2_ir_input-image": "resource-one/0",
            },
            {
                "scan_event_index": 2,
                "uc_amp2_ir_input-acq_timestamp": 1002.0,
                "uc_amp2_ir_input-nonscalar_save_path": str(device_folder),
                "uc_amp2_ir_input-image": "resource-two/0",
            },
        ]
    )
    analyzer = ImageAnalyzerAdapter(
        _FakeImageAnalyzer(),
        analyzer_id="fake_image_analyzer_v1",
        config={"mode": "sum"},
    )
    documents = []

    metadata = run_camera_image_analysis_for_tiled_run(
        _FakeRun(start_doc, table),
        device_name="UC_Amp2_IR_input",
        analyzer=analyzer,
        device_type=POINTGREY_CAMERA_DEVICE_TYPE,
        root_map={"Z:/data": str(local_data_root)},
        invocation_id="20260628T190012Z",
        emit_derived_run=True,
        document_callback=lambda name, doc: documents.append((name, doc)),
        retry_intervals=[],
        repo_root=tmp_path,
    )

    output_dir = (
        day / "analysis" / "Scan006" / "fake_image_analyzer_v1" / "20260628T190012Z"
    )
    feature_rows = [
        json.loads(line)
        for line in (output_dir / "features.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    saved_metadata = json.loads(
        (output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )

    assert metadata.feature_table == "features.jsonl"
    assert metadata.analysis_root == "Z:/data"
    assert metadata.analysis_output_path == (
        "Undulator/Y2026/06-Jun/26_0624/analysis/"
        "Scan006/fake_image_analyzer_v1/20260628T190012Z"
    )
    assert [row["scan_event_index"] for row in feature_rows] == [1, 2]
    assert [row["feature:pixel_sum"] for row in feature_rows] == [10, 26]
    assert saved_metadata["raw_run_uid"] == "run-uid"
    assert saved_metadata["derived_run_uid"] == metadata.derived_run_uid
    assert saved_metadata["inputs"][0]["datum_id"] == "resource-one/0"
    assert saved_metadata["inputs"][0]["resource_root"] == "Z:/data"
    assert saved_metadata["inputs"][0]["resource_path"] == (
        "Undulator/Y2026/06-Jun/26_0624/scans/Scan006/"
        "UC_Amp2_IR_input/UC_Amp2_IR_input_1001.000.png"
    )
    assert saved_metadata["config"] == {"mode": "sum"}
    assert [name for name, _ in documents] == ["start", "descriptor", "event", "stop"]
    assert documents[0][1]["analysis_of"] == "run-uid"
    assert documents[0][1]["analysis_root"] == "Z:/data"
    assert documents[2][1]["data"]["feature_table_path"].endswith("features.jsonl")
    serialized_metadata = json.dumps(saved_metadata)
    serialized_docs = json.dumps([doc for _name, doc in documents])
    assert str(local_data_root) not in serialized_metadata
    assert str(local_data_root) not in serialized_docs
    assert not (scan_folder / "analysis").exists()


def test_registered_text_asset_analysis_runs_from_tiled(tmp_path: Path) -> None:
    """Generic asset analysis should handle a selected multi-asset text field."""
    local_data_root = tmp_path / "Volumes" / "hdna2" / "data"
    day = local_data_root / "Undulator" / "Y2026" / "06-Jun" / "26_0624"
    scan_folder = day / "scans" / "Scan042"
    device_folder = scan_folder / "U_BCaveMagSpec"
    text_folder = scan_folder / "U_BCaveMagSpec-interpSpec"
    text_folder.mkdir(parents=True)
    np.savetxt(
        text_folder / "U_BCaveMagSpec-interpSpec_1001.000.txt",
        [[400.0, 1.0], [401.0, 2.0]],
    )

    start_doc = {
        "uid": "run-uid",
        "scan_number": 42,
        "scan_folder": str(scan_folder),
        "experiment": "Undulator",
    }
    table = pd.DataFrame(
        [
            {
                "scan_event_index": 1,
                "u_bcavemagspec-acq_timestamp": 1001.0,
                "u_bcavemagspec-nonscalar_save_path": str(device_folder),
                "u_bcavemagspec-interpspec": "text-resource/0",
            },
        ]
    )

    metadata = run_asset_analysis_for_tiled_run(
        _FakeRun(start_doc, table),
        device_name="U_BCaveMagSpec",
        device_type=MAGSPEC_CAMERA_DEVICE_TYPE,
        event_field="interpSpec",
        analyzer=_ArraySumAnalyzer(),
        root_map={"Z:/data": str(local_data_root)},
        invocation_id="20260628T190012Z",
        retry_intervals=[],
        repo_root=tmp_path,
    )

    output_dir = day / "analysis" / "Scan042" / "array_sum_v1" / "20260628T190012Z"
    saved_metadata = json.loads(
        (output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    feature_rows = [
        json.loads(line)
        for line in (output_dir / "features.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert metadata.feature_table == "features.jsonl"
    assert feature_rows[0]["feature:y_sum"] == 3.0
    assert saved_metadata["inputs"][0]["device_type"] == MAGSPEC_CAMERA_DEVICE_TYPE
    assert saved_metadata["inputs"][0]["event_field"] == "interpSpec"
    assert saved_metadata["inputs"][0]["payload_kind"] == "array_1d"
    assert saved_metadata["inputs"][0]["loader_kind"] == "text_array"
    assert saved_metadata["inputs"][0]["resource_root"] == "Z:/data"
    assert saved_metadata["inputs"][0]["resource_path"] == (
        "Undulator/Y2026/06-Jun/26_0624/scans/Scan042/"
        "U_BCaveMagSpec-interpSpec/U_BCaveMagSpec-interpSpec_1001.000.txt"
    )
    assert str(local_data_root) not in json.dumps(saved_metadata)


def test_data_1d_asset_analysis_uses_explicit_loader_config(tmp_path: Path) -> None:
    """DATA_1D assets should load at analysis time through Data1DConfig."""
    local_data_root = tmp_path / "Volumes" / "hdna2" / "data"
    day = local_data_root / "Undulator" / "Y2026" / "06-Jun" / "26_0624"
    scan_folder = day / "scans" / "Scan007"
    device_folder = scan_folder / "U_Scope"
    device_folder.mkdir(parents=True)
    tdms_path = device_folder / "U_Scope_1001.000.tdms"
    with tdms_path.open("wb") as stream:
        np.save(stream, np.array([[0.0, 1.0, 10.0], [1.0, 2.0, 20.0]]))

    start_doc = {
        "uid": "run-uid",
        "scan_number": 7,
        "scan_folder": str(scan_folder),
        "experiment": "Undulator",
    }
    table = pd.DataFrame(
        [
            {
                "scan_event_index": 1,
                "u_scope-acq_timestamp": 1001.0,
                "u_scope-nonscalar_save_path": str(device_folder),
                "u_scope-tdms": "tdms-resource/0",
            },
        ]
    )
    data_1d_config = Data1DConfig(
        data_type=Data1DType.NPY,
        x_column=0,
        y_column=1,
        auxiliary_columns={"background": 2},
    )

    metadata = run_asset_analysis_for_tiled_run(
        _FakeRun(start_doc, table),
        device_name="U_Scope",
        device_type=PICOSCOPE_V2_DEVICE_TYPE,
        analyzer=_Data1DResultAnalyzer(),
        data_1d_config=data_1d_config,
        root_map={"Z:/data": str(local_data_root)},
        invocation_id="20260628T190012Z",
        retry_intervals=[],
        repo_root=tmp_path,
    )

    output_dir = day / "analysis" / "Scan007" / "data_1d_sum_v1" / "20260628T190012Z"
    saved_metadata = json.loads(
        (output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    feature_rows = [
        json.loads(line)
        for line in (output_dir / "features.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert metadata.feature_table == "features.jsonl"
    assert feature_rows[0]["feature:y_sum"] == 3.0
    assert feature_rows[0]["feature:background_sum"] == 30.0
    assert saved_metadata["inputs"][0]["device_type"] == PICOSCOPE_V2_DEVICE_TYPE
    assert saved_metadata["inputs"][0]["event_field"] == "tdms"
    assert saved_metadata["inputs"][0]["payload_kind"] == "array_1d"
    assert saved_metadata["inputs"][0]["loader_kind"] == "data_1d"
    assert saved_metadata["config"]["data_1d_loader"] == {
        "auxiliary_columns": {"background": 2},
        "data_type": "npy",
        "delimiter": None,
        "trace_index": 0,
        "x_column": 0,
        "x_trace_index": None,
        "y_column": 1,
    }
    assert str(local_data_root) not in json.dumps(saved_metadata)


class _ToyAnalyzer:
    """Small analyzer implementing the runner protocol."""

    analyzer_id = "toy_analyzer_v1"
    analyzer_name = "Toy analyzer"
    analyzer_version = "0.1"

    def describe_config(self) -> dict:
        """Return a serializable config."""
        return {"mode": "sum"}

    def analyze(
        self, data, *, asset: InputAssetRef, output_dir: Path
    ) -> AnalysisResult:
        """Return a scalar and a derived asset reference."""
        _ = asset
        _ = output_dir
        return AnalysisResult(
            features={"pixel_sum": sum(sum(row) for row in data)},
            derived_assets=[
                DerivedAssetRef(
                    asset_id="preview",
                    asset_spec="GEECS_ANALYSIS_PREVIEW",
                    relative_path="assets/shot_000002_preview.png",
                    source_datum_id="resource/0",
                )
            ],
        )


class _FakeImageAnalyzer:
    """Minimal ImageAnalysis-like analyzer."""

    def analyze_image(self, image, auxiliary_data=None):
        """Return a minimal ImageAnalyzerResult-like object."""
        assert auxiliary_data["raw_run_uid"] == "run-uid"
        return _FakeImageAnalyzerResult({"pixel_sum": int(np.asarray(image).sum())})


class _FakeData1DImageAnalyzer:
    """Minimal ImageAnalysis-like 1D analyzer."""

    data_metadata = None

    def analyze_image(self, image, auxiliary_data=None):
        """Return a scalar from a loaded 1D line array."""
        assert auxiliary_data["raw_run_uid"] == "run-uid"
        assert auxiliary_data["device"] == "U_BCaveICT"
        assert isinstance(image, np.ndarray)
        return _FakeImageAnalyzerResult({"y_sum": float(image[:, 1].sum())})


class _FakeConfiguredLineImageAnalyzer:
    """Minimal 1D analyzer with configured text columns."""

    line_config = SimpleNamespace(
        data_loading=Data1DConfig(
            data_type=Data1DType.TSV,
            x_column=1,
            y_column=2,
        )
    )

    def analyze_image(self, image, auxiliary_data=None):
        """Return scalar sums for the selected x/y columns."""
        assert image.shape == (2, 2)
        return _FakeImageAnalyzerResult(
            {
                "x_sum": float(image[:, 0].sum()),
                "y_sum": float(image[:, 1].sum()),
            }
        )


class _FakeImageAnalyzerResult:
    """Minimal object exposing the ImageAnalyzerResult export method."""

    def __init__(self, scalars: dict[str, float]) -> None:
        self.scalars = scalars

    def feature_scalars(self) -> dict[str, float]:
        """Return portable scalar features."""
        return dict(self.scalars)


class _DynamicBackgroundAnalyzer:
    """Fake scan-context analyzer that subtracts the scan mean image."""

    analyzer_id = "dynamic_background_v1"
    analyzer_name = "Dynamic background"
    analyzer_version = "0.1"
    analysis_scope = AnalysisScope.SCAN

    def describe_config(self) -> dict:
        """Return a serializable config."""
        return {"background": "scan_mean"}

    def prepare_scan_context(self, *, assets: list[FilledAsset], output_dir: Path):
        """Compute a scan-level background image."""
        _ = output_dir
        return {"background": np.mean([asset.data for asset in assets], axis=0)}

    def analyze_with_scan_context(
        self,
        data,
        *,
        asset: InputAssetRef,
        output_dir: Path,
        scan_context,
    ) -> AnalysisResult:
        """Subtract scan-level background and return a scalar feature."""
        _ = asset
        _ = output_dir
        corrected = np.asarray(data) - scan_context["background"]
        return AnalysisResult(features={"bg_subtracted_sum": int(corrected.sum())})

    def finalize_scan_context(self, *, scan_context, output_dir: Path):
        """Return a derived asset representing the scan-level background."""
        _ = scan_context
        _ = output_dir
        return [
            DerivedAssetRef(
                asset_id="background",
                asset_spec="GEECS_ANALYSIS_BACKGROUND",
                relative_path="assets/background.npy",
            )
        ]


class _ArraySumAnalyzer:
    """Analyzer for generic loaded array assets."""

    analyzer_id = "array_sum_v1"
    analyzer_name = "Array sum"
    analyzer_version = "0.1"
    analysis_scope = AnalysisScope.EVENT

    def describe_config(self) -> dict:
        """Return a serializable config."""
        return {"mode": "sum_second_column"}

    def analyze(
        self, data, *, asset: InputAssetRef, output_dir: Path
    ) -> AnalysisResult:
        """Sum the second column of a loaded array."""
        _ = asset
        _ = output_dir
        array = np.asarray(data)
        return AnalysisResult(features={"y_sum": float(array[:, 1].sum())})


class _Data1DResultAnalyzer:
    """Analyzer for config-loaded 1D data results."""

    analyzer_id = "data_1d_sum_v1"
    analyzer_name = "Data 1D sum"
    analyzer_version = "0.1"
    analysis_scope = AnalysisScope.EVENT

    def describe_config(self) -> dict:
        """Return a serializable config."""
        return {"mode": "sum_1d_result"}

    def analyze(
        self, data, *, asset: InputAssetRef, output_dir: Path
    ) -> AnalysisResult:
        """Sum selected and auxiliary 1D columns."""
        _ = asset
        _ = output_dir
        assert isinstance(data, Data1DResult)
        return AnalysisResult(
            features={
                "y_sum": float(data.data[:, 1].sum()),
                "background_sum": float(data.auxiliary_column_data["background"].sum()),
            }
        )


class _FakePrimary:
    """Minimal Tiled primary stream shim."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    def read(self) -> pd.DataFrame:
        """Return the fake primary event stream."""
        return self._dataframe


class _FakeRun:
    """Minimal Tiled run shim."""

    def __init__(self, start_doc: dict, dataframe: pd.DataFrame) -> None:
        self.metadata = {"start": start_doc}
        self._primary = _FakePrimary(dataframe)

    def __getitem__(self, key: str) -> _FakePrimary:
        """Return the requested stream."""
        if key != "primary":
            raise KeyError(key)
        return self._primary
