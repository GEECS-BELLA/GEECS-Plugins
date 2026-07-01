"""Post-run analysis over registered GEECS external assets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from geecs_bluesky.analysis.derived_run import (
    build_analysis_run_documents,
    publish_analysis_run_to_tiled,
    publish_documents,
)
from geecs_bluesky.analysis.models import AnalysisInvocationMetadata, InputAssetRef
from geecs_bluesky.analysis.provenance import capture_code_version, capture_environment
from geecs_bluesky.analysis.runner import AnalysisRunner, AnalyzerProtocol, FilledAsset
from geecs_bluesky.analysis.writer import AnalysisArtifactWriter
from geecs_bluesky.assets.readback import fill_geecs_documents
from geecs_bluesky.assets.registry import AssetDefinition, AssetLoaderKind
from geecs_bluesky.assets.tiled_readback import (
    FilledTiledGeecsAsset,
    TiledGeecsAsset,
    find_geecs_run,
    load_tiled_client,
    read_geecs_root_map,
    read_primary_dataframe,
    resolve_asset_from_event,
)


Data1DConfigInput = Any


def run_tiled_asset_analysis(
    *,
    year: int,
    month: int,
    day: int,
    scan_number: int,
    device_name: str,
    analyzer: AnalyzerProtocol,
    event_field: str | None = None,
    data_1d_config: Data1DConfigInput | None = None,
    experiment: str | None = None,
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
    """Run an analyzer over one registered asset from an archived Tiled run."""
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
    return run_asset_analysis_for_tiled_run(
        run,
        device_name=device_name,
        analyzer=analyzer,
        event_field=event_field,
        data_1d_config=data_1d_config,
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


def run_asset_analysis_for_tiled_run(
    run: Any,
    *,
    device_name: str,
    analyzer: AnalyzerProtocol,
    event_field: str | None = None,
    data_1d_config: Data1DConfigInput | None = None,
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
    """Run one analyzer over a registered asset from an already-loaded Tiled run."""
    filled_assets = list(
        iter_filled_assets_from_tiled_run(
            run,
            device_name=device_name,
            event_field=event_field,
            data_1d_config=data_1d_config,
            device_type=device_type,
            root_map=root_map,
            shot_numbers=shot_numbers,
            retry_intervals=retry_intervals,
        )
    )
    if not filled_assets:
        raise LookupError(f"No filled assets found for {device_name!r}.")

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
        config=_analysis_config(analyzer, data_1d_config=data_1d_config),
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
                ref=input_ref_from_tiled_asset(asset.asset),
                data=asset.data,
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


def iter_filled_assets_from_tiled_run(
    run: Any,
    *,
    device_name: str,
    event_field: str | None = None,
    data_1d_config: Data1DConfigInput | None = None,
    device_type: str | None = None,
    root_map: Mapping[str, str] | None = None,
    shot_numbers: Iterable[int] | None = None,
    retry_intervals: Iterable[float] | None = None,
) -> Iterable[FilledTiledGeecsAsset]:
    """Yield filled assets for selected events in a Tiled run."""
    start_doc = dict(run.metadata.get("start") or {})
    primary_table = read_primary_dataframe(run)
    selected_shots = set(shot_numbers or [])
    effective_root_map = dict(root_map or read_geecs_root_map())

    for event in _iter_primary_events(primary_table):
        scan_event_index = event.get("scan_event_index")
        if selected_shots and int(scan_event_index) not in selected_shots:
            continue
        asset = resolve_asset_from_event(
            start_doc=start_doc,
            event=event,
            device_name=device_name,
            device_type=device_type,
            event_field=event_field,
            root_map=effective_root_map,
        )
        yield FilledTiledGeecsAsset(
            asset=asset,
            data=load_tiled_asset_data(
                asset,
                data_1d_config=data_1d_config,
                root_map=effective_root_map,
                retry_intervals=retry_intervals,
            ),
        )


def load_tiled_asset_data(
    asset: TiledGeecsAsset,
    *,
    data_1d_config: Data1DConfigInput | None = None,
    root_map: Mapping[str, str] | None = None,
    retry_intervals: Iterable[float] | None = None,
) -> Any:
    """Load a resolved Tiled asset through its registry-defined loader path."""
    definition = asset.definition
    if definition.loader_kind is AssetLoaderKind.DATA_1D:
        return _read_data_1d_asset(asset, data_1d_config=data_1d_config)
    if definition.handler_class is None:
        raise ValueError(
            f"Asset field {definition.event_field!r} for device type "
            f"{definition.device_type!r} has no local handler."
        )

    filled_docs = fill_geecs_documents(
        asset.documents,
        root_map=root_map or read_geecs_root_map(),
        include=[asset.data_key],
        retry_intervals=retry_intervals,
    )
    filled_event = next(doc for name, doc in filled_docs if name == "event")
    return filled_event["data"][asset.data_key]


def input_ref_from_tiled_asset(asset: TiledGeecsAsset) -> InputAssetRef:
    """Return portable analysis input metadata for a resolved Tiled asset."""
    event = asset.event
    resource_uid = asset.datum_id.split("/", 1)[0] if "/" in asset.datum_id else None
    return InputAssetRef(
        raw_run_uid=str(asset.start_doc.get("uid")),
        event_uid=_optional_str(event.get("uid")),
        scan_number=_optional_int(asset.start_doc.get("scan_number")),
        scan_event_index=_optional_int(event.get("scan_event_index")),
        shot_number=_optional_int(event.get("scan_event_index")),
        device=asset.device_name,
        device_type=asset.device_type,
        data_key=asset.data_key,
        event_field=asset.definition.event_field,
        datum_id=asset.datum_id,
        resource_uid=resource_uid,
        asset_spec=asset.definition.spec,
        payload_kind=asset.definition.payload_kind.value,
        loader_kind=asset.definition.loader_kind.value,
        resource_root=asset.resource_root,
        resource_path=asset.resource_path,
    )


def _read_data_1d_asset(
    asset: TiledGeecsAsset,
    *,
    data_1d_config: Data1DConfigInput | None,
) -> Any:
    from geecs_data_utils.io.array1d import Data1DConfig, read_1d_data

    if data_1d_config is None and asset.definition.requires_loader_config:
        raise ValueError(
            f"Asset field {asset.definition.event_field!r} for device type "
            f"{asset.definition.device_type!r} requires a Data1DConfig."
        )
    config = _coerce_data_1d_config(data_1d_config, definition=asset.definition)
    if config is None:
        raise ValueError(
            f"Asset field {asset.definition.event_field!r} for device type "
            f"{asset.definition.device_type!r} has no Data1DConfig."
        )
    if not isinstance(config, Data1DConfig):
        raise TypeError(f"Expected Data1DConfig, got {type(config)!r}.")
    return read_1d_data(asset.file_path, config)


def _coerce_data_1d_config(
    data_1d_config: Data1DConfigInput | None,
    *,
    definition: AssetDefinition,
) -> Any | None:
    from geecs_data_utils.io.array1d import Data1DConfig

    if data_1d_config is None:
        if definition.default_data_1d_type is None:
            return None
        return Data1DConfig(data_type=definition.default_data_1d_type)
    if isinstance(data_1d_config, Data1DConfig):
        return data_1d_config
    payload = dict(data_1d_config)
    payload.setdefault("data_type", definition.default_data_1d_type)
    return Data1DConfig(**payload)


def _analysis_config(
    analyzer: AnalyzerProtocol,
    *,
    data_1d_config: Data1DConfigInput | None,
) -> dict[str, Any]:
    analyzer_config = analyzer.describe_config()
    if data_1d_config is None:
        return analyzer_config
    return {
        "analyzer": analyzer_config,
        "data_1d_loader": _serialize_config(data_1d_config),
    }


def _serialize_config(config: Data1DConfigInput) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return dict(config.model_dump(mode="json"))
    return dict(config)


def _day_analysis_dir_from_asset(asset: TiledGeecsAsset) -> Path:
    scan_folder = asset.file_path.parent.parent
    if scan_folder.parent.name != "scans":
        raise ValueError(f"Cannot infer day folder from asset path: {asset.file_path}")
    return scan_folder.parent.parent / "analysis"


def _iter_primary_events(primary_table: Any) -> Iterable[dict[str, Any]]:
    if hasattr(primary_table, "sort_values") and "scan_event_index" in primary_table:
        primary_table = primary_table.sort_values("scan_event_index")
    if hasattr(primary_table, "iterrows"):
        for _index, row in primary_table.iterrows():
            yield dict(row.to_dict())
        return
    for row in primary_table:
        yield dict(row)


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
