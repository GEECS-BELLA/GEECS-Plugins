"""Tiled-backed local readback for GEECS external assets."""

from __future__ import annotations

import configparser
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from geecs_bluesky.assets.readback import (
    Document,
    ExternalAssetDocumentSpec,
    fill_geecs_documents,
)
from geecs_bluesky.assets.registry import (
    AssetDefinition,
    get_asset_definitions,
    get_single_asset_definition,
)
from geecs_bluesky.utils import safe_name


@dataclass(frozen=True)
class TiledGeecsAsset:
    """Resolved native GEECS asset documents from one archived Tiled event."""

    definition: AssetDefinition
    device_name: str
    device_type: str
    documents: list[Document]
    data_key: str
    datum_id: str
    file_path: Path
    resource_root: str
    resource_path: str
    local_resource_root: Path
    acq_timestamp: float
    start_doc: dict[str, Any]
    event: dict[str, Any]


@dataclass(frozen=True)
class FilledTiledGeecsAsset:
    """Filled native GEECS asset loaded through the local GEECS handler."""

    asset: TiledGeecsAsset
    data: Any

    @property
    def image(self) -> Any:
        """Compatibility alias for camera callers."""
        return self.data


TiledCameraAsset = TiledGeecsAsset
FilledTiledCameraAsset = FilledTiledGeecsAsset
Data1DConfigInput = Any


def read_tiled_config() -> tuple[str | None, str | None]:
    """Read Tiled connection details from the shared GEECS config file."""
    config_path = Path.home() / ".config" / "geecs_python_api" / "config.ini"
    if not config_path.exists():
        return None, None

    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    if "tiled" not in cfg:
        return None, None
    return cfg["tiled"].get("uri") or None, cfg["tiled"].get("api_key") or None


def read_geecs_root_map() -> dict[str, str]:
    """Read a device-server to local data-root map from the GEECS config."""
    config_path = Path.home() / ".config" / "geecs_python_api" / "config.ini"
    if not config_path.exists():
        return {}

    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    if "Paths" not in cfg:
        return {}

    remote_root = cfg["Paths"].get("geecs_device_server_data_base_path")
    local_root = cfg["Paths"].get("GEECS_DATA_LOCAL_BASE_PATH")
    if not remote_root or not local_root:
        return {}
    return {remote_root: local_root}


def load_tiled_client(
    *,
    tiled_uri: str | None = None,
    tiled_api_key: str | None = None,
) -> Any:
    """Load the configured Tiled client.

    Parameters
    ----------
    tiled_uri, tiled_api_key:
        Optional connection details. If omitted, values are read from
        ``~/.config/geecs_python_api/config.ini`` under ``[tiled]``.

    Returns
    -------
    object
        A Tiled client container.

    Raises
    ------
    RuntimeError
        If Tiled is unavailable or no URI can be resolved.
    """
    if tiled_uri is None:
        tiled_uri, tiled_api_key = read_tiled_config()
    if not tiled_uri:
        raise RuntimeError(
            "No Tiled URI given and none found in "
            "~/.config/geecs_python_api/config.ini [tiled]"
        )

    try:
        from tiled.client import from_uri
    except ImportError as exc:  # pragma: no cover - exercised without tiled
        raise RuntimeError(
            "tiled is not installed; install the GeecsBluesky tiled extra "
            "before reading archived Tiled runs"
        ) from exc
    return from_uri(tiled_uri, api_key=tiled_api_key)


def find_geecs_run(
    catalog: Any,
    *,
    year: int,
    month: int,
    day: int,
    scan_number: int,
    experiment: str | None = None,
    timezone: str = "America/Los_Angeles",
) -> Any:
    """Find one archived GEECS Bluesky run in a Tiled catalog.

    The lookup is constrained by GEECS scan number and local calendar day. The
    day is required because Bluesky ``scan_id``/GEECS ``scan_number`` is only
    day-scoped.
    """
    queried = _search_catalog(
        catalog,
        year=year,
        month=month,
        day=day,
        scan_number=scan_number,
        experiment=experiment,
        timezone=timezone,
    )
    matches = [
        run
        for run in _iter_runs(queried)
        if _run_matches(
            run,
            year=year,
            month=month,
            day=day,
            scan_number=scan_number,
            experiment=experiment,
            timezone=timezone,
        )
    ]

    if not matches:
        label = (
            f"experiment={experiment!r}, " if experiment is not None else ""
        ) + f"{year:04d}-{month:02d}-{day:02d} Scan{scan_number:03d}"
        raise LookupError(f"No Tiled run found for {label}.")
    if len(matches) > 1:
        uids = [
            str(_start_doc(run).get("uid", "<no uid>"))[:12]
            for run in sorted(matches, key=_run_start_time)
        ]
        raise LookupError(
            "Multiple Tiled runs matched "
            f"{year:04d}-{month:02d}-{day:02d} Scan{scan_number:03d}: {uids}"
        )
    return matches[0]


def read_primary_dataframe(run: Any) -> Any:
    """Read a Tiled run's primary stream as a pandas DataFrame."""
    primary = run["primary"].read()
    if hasattr(primary, "to_dataframe"):
        return primary.to_dataframe().reset_index()
    if hasattr(primary, "reset_index"):
        return primary.reset_index()
    return primary


def event_by_scan_event_index(primary_table: Any, shot_number: int) -> dict[str, Any]:
    """Return the event row whose ``scan_event_index`` matches *shot_number*."""
    if "scan_event_index" not in primary_table:
        raise KeyError("primary stream does not contain scan_event_index")

    matches = primary_table[primary_table["scan_event_index"] == shot_number]
    if len(matches) == 0:
        available = _scan_event_index_summary(primary_table)
        raise LookupError(
            f"No event row found for scan_event_index={shot_number}. "
            f"Available scan_event_index values: {available}."
        )
    if len(matches) > 1:
        raise LookupError(
            f"Multiple event rows found for scan_event_index={shot_number}."
        )
    row = matches.iloc[0] if hasattr(matches, "iloc") else next(iter(matches))
    return _mapping_from_row(row)


def resolve_asset_from_event(
    *,
    start_doc: Mapping[str, Any],
    event: Mapping[str, Any],
    device_name: str,
    device_type: str | None = None,
    event_field: str | None = None,
    root_map: Mapping[str, str] | None = None,
) -> TiledGeecsAsset:
    """Resolve a native GEECS asset from archived Tiled run metadata and one event.

    This is the local-first archive readback path: Tiled provides the event row
    and acquisition timestamp, the GeecsBluesky asset registry provides the
    native file convention, and local handlers fill the file.
    """
    if device_type is None:
        from geecs_ca_gateway.db.geecs_db import GeecsDb

        device_type = GeecsDb.get_device_type(device_name)

    definition = _asset_definition(device_type, event_field=event_field)

    event_dict = dict(event)
    start_dict = dict(start_doc)
    scan_number = _required_int(start_dict, "scan_number")
    effective_root_map = dict(root_map or read_geecs_root_map())
    scan_folder_value = _required_str(start_dict, "scan_folder")
    scan_folder = _mapped_path(scan_folder_value, effective_root_map)
    canonical_scan_folder = _canonical_path(scan_folder_value, effective_root_map)
    resource_root = _resource_root(
        canonical_scan_folder,
        start_doc=start_dict,
        root_map=effective_root_map,
    )
    local_resource_root = _mapped_path(resource_root, effective_root_map)
    data_key = definition.event_key(device_name)
    acq_key = f"{safe_name(device_name)}-acq_timestamp"
    save_path_key = f"{safe_name(device_name)}-nonscalar_save_path"

    acq_timestamp = _required_float(event_dict, acq_key)
    save_path_value = event_dict.get(save_path_key)
    local_save_path = (
        _mapped_path(str(save_path_value), effective_root_map)
        if not _is_missing(save_path_value)
        else scan_folder / device_name
    )
    canonical_save_path = (
        _canonical_path(str(save_path_value), effective_root_map)
        if not _is_missing(save_path_value)
        else str(Path(canonical_scan_folder) / device_name)
    )
    file_path = definition.file_path(
        save_path=local_save_path,
        scan_number=scan_number,
        device_name=device_name,
        acq_timestamp=acq_timestamp,
    )
    canonical_file_path = definition.file_path(
        save_path=canonical_save_path,
        scan_number=scan_number,
        device_name=device_name,
        acq_timestamp=acq_timestamp,
    )
    resource_path = definition.resource_path(
        root=resource_root,
        file_path=canonical_file_path,
    )
    datum_id = _datum_id_from_event(event_dict.get(data_key))
    resource_uid = datum_id.split("/", 1)[0] if "/" in datum_id else str(uuid4())
    timestamp = _event_timestamp(event_dict, start_dict)

    documents = ExternalAssetDocumentSpec(
        start_doc=start_dict,
        event=event_dict,
        definition=definition,
        device_name=device_name,
        data_key=data_key,
        datum_id=datum_id,
        resource_uid=resource_uid,
        resource_root=resource_root,
        resource_path=resource_path,
        timestamp=timestamp,
    ).to_documents()
    return TiledGeecsAsset(
        definition=definition,
        device_name=device_name,
        device_type=device_type,
        documents=documents,
        data_key=data_key,
        datum_id=datum_id,
        file_path=file_path,
        resource_root=resource_root,
        resource_path=resource_path,
        local_resource_root=local_resource_root,
        acq_timestamp=acq_timestamp,
        start_doc=start_dict,
        event=event_dict,
    )


def resolve_camera_asset_from_event(
    *,
    start_doc: Mapping[str, Any],
    event: Mapping[str, Any],
    device_name: str,
    device_type: str | None = None,
    root_map: Mapping[str, str] | None = None,
) -> TiledCameraAsset:
    """Resolve a camera image asset from archived Tiled metadata."""
    return resolve_asset_from_event(
        start_doc=start_doc,
        event=event,
        device_name=device_name,
        device_type=device_type,
        event_field="image",
        root_map=root_map,
    )


def load_asset_from_tiled_run(
    run: Any,
    *,
    device_name: str,
    shot_number: int,
    device_type: str | None = None,
    event_field: str | None = None,
    data_1d_config: Data1DConfigInput | None = None,
    root_map: Mapping[str, str] | None = None,
    retry_intervals: Iterable[float] | None = None,
) -> FilledTiledGeecsAsset:
    """Load one native GEECS asset from a Tiled run."""
    start_doc = dict(run.metadata.get("start") or {})
    primary_table = read_primary_dataframe(run)
    event = event_by_scan_event_index(primary_table, shot_number)
    asset = resolve_asset_from_event(
        start_doc=start_doc,
        event=event,
        device_name=device_name,
        device_type=device_type,
        event_field=event_field,
        root_map=root_map,
    )
    return FilledTiledGeecsAsset(
        asset=asset,
        data=load_tiled_asset_data(
            asset,
            data_1d_config=data_1d_config,
            root_map=root_map,
            retry_intervals=retry_intervals,
        ),
    )


def load_camera_image_from_tiled_run(
    run: Any,
    *,
    device_name: str,
    shot_number: int,
    device_type: str | None = None,
    root_map: Mapping[str, str] | None = None,
    retry_intervals: Iterable[float] | None = None,
) -> FilledTiledCameraAsset:
    """Load one camera image from a Tiled run using local GEECS handlers."""
    return load_asset_from_tiled_run(
        run,
        device_name=device_name,
        shot_number=shot_number,
        device_type=device_type,
        event_field="image",
        root_map=root_map,
        retry_intervals=retry_intervals,
    )


def load_asset_from_tiled(
    *,
    year: int,
    month: int,
    day: int,
    scan_number: int,
    device_name: str,
    shot_number: int,
    data_1d_config: Data1DConfigInput | None = None,
    event_field: str | None = None,
    experiment: str | None = None,
    device_type: str | None = None,
    tiled_uri: str | None = None,
    tiled_api_key: str | None = None,
    timezone: str = "America/Los_Angeles",
    root_map: Mapping[str, str] | None = None,
    retry_intervals: Iterable[float] | None = None,
) -> FilledTiledGeecsAsset:
    """Find a Tiled run by GEECS scan identity and load one native asset."""
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
    return load_asset_from_tiled_run(
        run,
        device_name=device_name,
        shot_number=shot_number,
        device_type=device_type,
        event_field=event_field,
        data_1d_config=data_1d_config,
        root_map=root_map,
        retry_intervals=retry_intervals,
    )


def load_camera_image_from_tiled(
    *,
    year: int,
    month: int,
    day: int,
    scan_number: int,
    device_name: str,
    shot_number: int,
    experiment: str | None = None,
    device_type: str | None = None,
    tiled_uri: str | None = None,
    tiled_api_key: str | None = None,
    timezone: str = "America/Los_Angeles",
    root_map: Mapping[str, str] | None = None,
    retry_intervals: Iterable[float] | None = None,
) -> FilledTiledCameraAsset:
    """Find a Tiled run by GEECS scan identity and load one camera image."""
    return load_asset_from_tiled(
        year=year,
        month=month,
        day=day,
        scan_number=scan_number,
        device_name=device_name,
        shot_number=shot_number,
        event_field="image",
        experiment=experiment,
        device_type=device_type,
        tiled_uri=tiled_uri,
        tiled_api_key=tiled_api_key,
        timezone=timezone,
        root_map=root_map,
        retry_intervals=retry_intervals,
    )


def load_tiled_asset_data(
    asset: TiledGeecsAsset,
    *,
    data_1d_config: Data1DConfigInput | None = None,
    root_map: Mapping[str, str] | None = None,
    retry_intervals: Iterable[float] | None = None,
) -> Any:
    """Load a resolved Tiled asset through its registry-defined loader."""
    definition = asset.definition
    if definition.uses_data_1d_reader:
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


def _read_data_1d_asset(
    asset: TiledGeecsAsset,
    *,
    data_1d_config: Data1DConfigInput | None,
) -> Any:
    from geecs_data_utils.io.array1d import Data1DConfig, read_1d_data

    if data_1d_config is None and asset.definition.requires_loader_config:
        raise ValueError(
            f"Asset field {asset.definition.event_field!r} for device type "
            f"{asset.definition.device_type!r} requires loader config."
        )
    config = _coerce_data_1d_config(data_1d_config, definition=asset.definition)
    if config is None:
        raise ValueError(
            f"Asset field {asset.definition.event_field!r} for device type "
            f"{asset.definition.device_type!r} has no loader config defaults."
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
        if not definition.loader_config_defaults:
            return None
        return Data1DConfig(**definition.loader_config_defaults)
    if isinstance(data_1d_config, Data1DConfig):
        return data_1d_config
    payload = definition.loader_config_defaults
    payload.update(dict(data_1d_config))
    return Data1DConfig(**payload)


def _search_catalog(
    catalog: Any,
    *,
    year: int,
    month: int,
    day: int,
    scan_number: int,
    experiment: str | None,
    timezone: str,
) -> Any:
    try:
        from tiled.queries import Key
    except ImportError:
        return catalog

    start, stop = _day_bounds(year, month, day, timezone)
    result = catalog
    for query in (
        Key("start.scan_number") == scan_number,
        Key("start.time") >= start.timestamp(),
        Key("start.time") < stop.timestamp(),
    ):
        try:
            result = result.search(query)
        except Exception:
            return catalog
    if experiment is not None:
        try:
            result = result.search(Key("start.experiment") == experiment)
        except Exception:
            return catalog
    return result


def _iter_runs(catalog: Any) -> Iterable[Any]:
    if hasattr(catalog, "values"):
        return catalog.values()
    return catalog


def _run_matches(
    run: Any,
    *,
    year: int,
    month: int,
    day: int,
    scan_number: int,
    experiment: str | None,
    timezone: str,
) -> bool:
    start = _start_doc(run)
    if _is_analysis_run(start):
        return False
    if start.get("scan_number") != scan_number:
        return False
    if experiment is not None and start.get("experiment") != experiment:
        return False
    return _date_matches(start, year=year, month=month, day=day, timezone=timezone)


def _start_doc(run: Any) -> dict[str, Any]:
    return dict(run.metadata.get("start") or {})


def _run_start_time(run: Any) -> float:
    return float(_start_doc(run).get("time") or 0.0)


def _is_analysis_run(start_doc: Mapping[str, Any]) -> bool:
    return start_doc.get("purpose") == "geecs_bluesky_analysis" or not _is_missing(
        start_doc.get("analysis_of")
    )


def _date_matches(
    start_doc: Mapping[str, Any],
    *,
    year: int,
    month: int,
    day: int,
    timezone: str,
) -> bool:
    start_time = start_doc.get("time")
    if not _is_missing(start_time):
        dt = datetime.fromtimestamp(float(start_time), ZoneInfo(timezone))
        return (dt.year, dt.month, dt.day) == (year, month, day)

    scan_folder = start_doc.get("scan_folder")
    if _is_missing(scan_folder):
        return False
    return f"Y{year:04d}" in str(
        scan_folder
    ) and f"{year % 100:02d}_{month:02d}{day:02d}" in str(scan_folder)


def _day_bounds(
    year: int,
    month: int,
    day: int,
    timezone: str,
) -> tuple[datetime, datetime]:
    tz = ZoneInfo(timezone)
    start = datetime.combine(
        datetime(year, month, day).date(),
        datetime_time.min,
        tzinfo=tz,
    )
    return start, start + timedelta(days=1)


def _required_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if _is_missing(value):
        raise KeyError(f"start document is missing {key!r}")
    return int(value)


def _required_float(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if _is_missing(value):
        raise KeyError(f"event row is missing {key!r}")
    return float(value)


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if _is_missing(value):
        raise KeyError(f"start document is missing {key!r}")
    return str(value)


def _asset_definition(
    device_type: str,
    *,
    event_field: str | None,
) -> AssetDefinition:
    if event_field is None:
        definition = get_single_asset_definition(device_type)
        if definition is None:
            raise ValueError(
                f"Device type {device_type!r} is not registered as a single "
                "external asset. Pass event_field for multi-asset devices."
            )
        return definition

    matches = [
        definition
        for definition in get_asset_definitions(device_type)
        if definition.event_field == event_field
    ]
    if not matches:
        raise ValueError(
            f"Device type {device_type!r} has no registered asset field "
            f"{event_field!r}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Device type {device_type!r} has multiple registered asset fields "
            f"{event_field!r}."
        )
    return matches[0]


def _mapped_path(value: str, root_map: Mapping[str, str]) -> Path:
    """Return *value* as a local path after applying a tolerant root map."""
    normalized_value = _normalize_path_string(value)
    for remote_root, local_root in root_map.items():
        normalized_remote = _normalize_path_string(str(remote_root))
        if normalized_value == normalized_remote or normalized_value.startswith(
            f"{normalized_remote}/"
        ):
            relative = normalized_value.removeprefix(normalized_remote).lstrip("/")
            return Path(str(local_root)).joinpath(*relative.split("/"))
    return Path(value)


def _canonical_path(value: str, root_map: Mapping[str, str]) -> str:
    """Return *value* translated from local mount to canonical root when possible."""
    normalized_value = _normalize_path_string(value)
    for remote_root, local_root in root_map.items():
        normalized_local = _normalize_path_string(str(local_root))
        if normalized_value == normalized_local or normalized_value.startswith(
            f"{normalized_local}/"
        ):
            relative = normalized_value.removeprefix(normalized_local).lstrip("/")
            return _join_posix(str(remote_root), relative)
    return normalized_value


def _resource_root(
    canonical_scan_folder: str,
    *,
    start_doc: Mapping[str, Any],
    root_map: Mapping[str, str],
) -> str:
    """Return the canonical data root for Resource docs."""
    normalized_scan_folder = _normalize_path_string(canonical_scan_folder)
    for remote_root in root_map:
        normalized_remote = _normalize_path_string(str(remote_root))
        if (
            normalized_scan_folder == normalized_remote
            or normalized_scan_folder.startswith(f"{normalized_remote}/")
        ):
            return normalized_remote

    inferred = _infer_data_root_from_scan_folder(
        normalized_scan_folder,
        experiment=start_doc.get("experiment"),
    )
    return inferred or normalized_scan_folder


def _infer_data_root_from_scan_folder(
    scan_folder: str,
    *,
    experiment: Any,
) -> str | None:
    """Infer the shared data root from a canonical GEECS scan-folder path."""
    if _is_missing(experiment):
        return None
    marker = f"/{experiment}/Y"
    if marker not in scan_folder:
        return None
    return scan_folder.split(marker, 1)[0]


def _join_posix(root: str, relative: str) -> str:
    """Join a root and relative path with POSIX separators."""
    root = _normalize_path_string(root)
    relative = _normalize_path_string(relative).lstrip("/")
    return f"{root}/{relative}" if relative else root


def _normalize_path_string(value: str) -> str:
    """Normalize Windows/POSIX separators for prefix comparisons."""
    normalized = value.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.rstrip("/")


def _datum_id_from_event(value: Any) -> str:
    if _is_missing(value):
        return f"{uuid4()}/0"
    return str(value)


def _event_timestamp(
    event: Mapping[str, Any],
    start_doc: Mapping[str, Any],
) -> float:
    for key in ("time", "timestamp"):
        value = event.get(key)
        if not _is_missing(value):
            return float(value)
    value = start_doc.get("time")
    return float(value) if not _is_missing(value) else datetime.now().timestamp()


def _mapping_from_row(row: Any) -> dict[str, Any]:
    if hasattr(row, "to_dict"):
        return dict(row.to_dict())
    return dict(row)


def _scan_event_index_summary(primary_table: Any) -> str:
    values = primary_table["scan_event_index"]
    if hasattr(values, "dropna"):
        values = values.dropna()
    if hasattr(values, "unique"):
        raw_indices = list(values.unique())
    else:
        raw_indices = list(values)

    indices = sorted({int(value) for value in raw_indices})
    if not indices:
        return "none"

    contiguous = indices == list(range(indices[0], indices[-1] + 1))
    if contiguous:
        if len(indices) == 1:
            return f"{indices[0]} (1 event)"
        return f"{indices[0]}-{indices[-1]} ({len(indices)} events)"

    max_listed = 12
    listed = ", ".join(str(index) for index in indices[:max_listed])
    if len(indices) > max_listed:
        listed = f"{listed}, ... {indices[-1]}"
    return f"{listed} ({len(indices)} events)"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False
