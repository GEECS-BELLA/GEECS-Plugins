"""End-to-end post-run camera analysis from archived Tiled runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from geecs_bluesky.analysis.image_analysis import ImageAnalyzerAdapter
from geecs_bluesky.analysis.models import AnalysisInvocationMetadata, InputAssetRef
from geecs_bluesky.analysis.derived_run import (
    build_analysis_run_documents,
    publish_analysis_run_to_tiled,
    publish_documents,
)
from geecs_bluesky.analysis.provenance import capture_code_version, capture_environment
from geecs_bluesky.analysis.runner import AnalysisRunner, AnalyzerProtocol, FilledAsset
from geecs_bluesky.analysis.writer import AnalysisArtifactWriter
from geecs_bluesky.assets.readback import fill_geecs_documents
from geecs_bluesky.assets.tiled_readback import (
    FilledTiledCameraAsset,
    TiledCameraAsset,
    find_geecs_run,
    load_tiled_client,
    read_geecs_root_map,
    read_primary_dataframe,
    resolve_camera_asset_from_event,
)


def run_tiled_camera_image_analysis(
    *,
    year: int,
    month: int,
    day: int,
    scan_number: int,
    device_name: str,
    image_analyzer: Any,
    analyzer_id: str,
    experiment: str | None = None,
    analyzer_name: str | None = None,
    analyzer_version: str | None = None,
    analyzer_config: dict[str, Any] | None = None,
    device_type: str | None = None,
    tiled_uri: str | None = None,
    tiled_api_key: str | None = None,
    timezone: str = "America/Los_Angeles",
    root_map: Mapping[str, str] | None = None,
    shot_numbers: Iterable[int] | None = None,
    invocation_id: str | None = None,
    feature_table_format: Literal["jsonl", "parquet"] = "jsonl",
    emit_derived_run: bool = False,
    publish_to_tiled: bool = False,
    document_callback: Any | None = None,
    retry_intervals: Iterable[float] | None = None,
    repo_root: Path | None = None,
    notes: str | None = None,
) -> AnalysisInvocationMetadata:
    """Run an ImageAnalysis-style analyzer over camera assets from Tiled."""
    client = load_tiled_client(tiled_uri=tiled_uri, tiled_api_key=tiled_api_key)
    run = find_geecs_run(
        client,
        year=year,
        month=month,
        day=day,
        scan_number=scan_number,
        experiment=experiment,
        timezone=timezone,
    )
    adapter = ImageAnalyzerAdapter(
        image_analyzer,
        analyzer_id=analyzer_id,
        analyzer_name=analyzer_name,
        analyzer_version=analyzer_version,
        config=analyzer_config,
    )
    return run_camera_image_analysis_for_tiled_run(
        run,
        device_name=device_name,
        analyzer=adapter,
        device_type=device_type,
        root_map=root_map,
        shot_numbers=shot_numbers,
        invocation_id=invocation_id,
        feature_table_format=feature_table_format,
        emit_derived_run=emit_derived_run,
        publish_to_tiled=publish_to_tiled,
        publish_tiled_uri=tiled_uri,
        publish_tiled_api_key=tiled_api_key,
        document_callback=document_callback,
        retry_intervals=retry_intervals,
        repo_root=repo_root,
        notes=notes,
    )


def run_camera_image_analysis_for_tiled_run(
    run: Any,
    *,
    device_name: str,
    analyzer: AnalyzerProtocol,
    device_type: str | None = None,
    root_map: Mapping[str, str] | None = None,
    shot_numbers: Iterable[int] | None = None,
    invocation_id: str | None = None,
    feature_table_format: Literal["jsonl", "parquet"] = "jsonl",
    emit_derived_run: bool = False,
    publish_to_tiled: bool = False,
    publish_tiled_uri: str | None = None,
    publish_tiled_api_key: str | None = None,
    document_callback: Any | None = None,
    retry_intervals: Iterable[float] | None = None,
    repo_root: Path | None = None,
    notes: str | None = None,
) -> AnalysisInvocationMetadata:
    """Run one analyzer over camera images from an already-loaded Tiled run."""
    filled_assets = list(
        iter_filled_camera_assets_from_tiled_run(
            run,
            device_name=device_name,
            device_type=device_type,
            root_map=root_map,
            shot_numbers=shot_numbers,
            retry_intervals=retry_intervals,
        )
    )
    if not filled_assets:
        raise LookupError(f"No filled camera assets found for {device_name!r}.")

    first_asset = filled_assets[0].asset
    scan_number = int(first_asset.start_doc["scan_number"])
    writer = AnalysisArtifactWriter(
        day_analysis_dir=_day_analysis_dir_from_asset(first_asset),
        scan_number=scan_number,
        analyzer_id=analyzer.analyzer_id,
        invocation_id=invocation_id,
        storage_root=first_asset.resource_root,
        local_storage_root=first_asset.local_resource_root,
    )
    runner = AnalysisRunner(writer=writer)
    raw_run_uid = str(first_asset.start_doc.get("uid"))
    metadata = AnalysisInvocationMetadata(
        analysis_id=f"{writer.invocation_id}_{analyzer.analyzer_id}",
        analyzer_id=analyzer.analyzer_id,
        analyzer_name=analyzer.analyzer_name,
        analyzer_version=analyzer.analyzer_version,
        raw_run_uid=raw_run_uid,
        scan_number=scan_number,
        experiment=first_asset.start_doc.get("experiment"),
        config=analyzer.describe_config(),
        code_version=capture_code_version(repo_root),
        environment=capture_environment(
            ["geecs-bluesky", "imageanalysis", "geecs-data-utils"]
        ),
        notes=notes,
    )
    completed = runner.run(
        analyzer=analyzer,
        assets=[
            FilledAsset(
                ref=_input_ref_from_filled(asset, device_name=device_name),
                data=asset.image,
            )
            for asset in filled_assets
        ],
        metadata=metadata,
        feature_table_format=feature_table_format,
    )
    if emit_derived_run or publish_to_tiled or document_callback is not None:
        documents = build_analysis_run_documents(
            completed, output_dir=writer.output_dir
        )
        writer.write_metadata(completed)
        if document_callback is not None:
            publish_documents(documents, document_callback)
        if publish_to_tiled:
            publish_analysis_run_to_tiled(
                documents,
                tiled_uri=publish_tiled_uri,
                tiled_api_key=publish_tiled_api_key,
            )
    return completed


def iter_filled_camera_assets_from_tiled_run(
    run: Any,
    *,
    device_name: str,
    device_type: str | None = None,
    root_map: Mapping[str, str] | None = None,
    shot_numbers: Iterable[int] | None = None,
    retry_intervals: Iterable[float] | None = None,
) -> Iterable[FilledTiledCameraAsset]:
    """Yield filled camera assets for selected events in a Tiled run."""
    start_doc = dict(run.metadata.get("start") or {})
    primary_table = read_primary_dataframe(run)
    selected_shots = set(shot_numbers or [])
    effective_root_map = dict(root_map or read_geecs_root_map())

    for event in _iter_primary_events(primary_table):
        scan_event_index = event.get("scan_event_index")
        if selected_shots and int(scan_event_index) not in selected_shots:
            continue
        asset = resolve_camera_asset_from_event(
            start_doc=start_doc,
            event=event,
            device_name=device_name,
            device_type=device_type,
            root_map=effective_root_map,
        )
        filled_docs = fill_geecs_documents(
            asset.documents,
            root_map=effective_root_map,
            include=[asset.data_key],
            retry_intervals=retry_intervals,
        )
        filled_event = next(doc for name, doc in filled_docs if name == "event")
        yield FilledTiledCameraAsset(
            asset=asset,
            data=filled_event["data"][asset.data_key],
        )


def _input_ref_from_filled(
    filled: FilledTiledCameraAsset,
    *,
    device_name: str,
) -> InputAssetRef:
    asset = filled.asset
    event = asset.event
    scan_event_index = _optional_int(event.get("scan_event_index"))
    resource_uid = asset.datum_id.split("/", 1)[0] if "/" in asset.datum_id else None
    spec = _resource_spec(asset)
    return InputAssetRef(
        raw_run_uid=str(asset.start_doc.get("uid")),
        event_uid=_optional_str(event.get("uid")),
        scan_number=_optional_int(asset.start_doc.get("scan_number")),
        scan_event_index=scan_event_index,
        shot_number=scan_event_index,
        device=device_name,
        data_key=asset.data_key,
        datum_id=asset.datum_id,
        resource_uid=resource_uid,
        asset_spec=spec,
        resource_root=asset.resource_root,
        resource_path=asset.resource_path,
    )


def _day_analysis_dir_from_asset(asset: TiledCameraAsset) -> Path:
    scan_folder = asset.file_path.parent.parent
    if scan_folder.parent.name != "scans":
        raise ValueError(f"Cannot infer day folder from asset path: {asset.file_path}")
    return scan_folder.parent.parent / "analysis"


def _iter_primary_events(primary_table: Any) -> Iterable[dict[str, Any]]:
    if hasattr(primary_table, "sort_values") and "scan_event_index" in primary_table:
        primary_table = primary_table.sort_values("scan_event_index")
    if hasattr(primary_table, "iterrows"):
        for _, row in primary_table.iterrows():
            yield dict(row.to_dict())
        return
    for row in primary_table:
        yield dict(row)


def _resource_spec(asset: TiledCameraAsset) -> str | None:
    for name, doc in asset.documents:
        if name == "resource":
            return str(doc.get("spec"))
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    return str(value)
