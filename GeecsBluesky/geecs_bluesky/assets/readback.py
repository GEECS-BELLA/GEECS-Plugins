"""Local readback helpers for GEECS external assets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Protocol, TypeAlias
from uuid import uuid4

from event_model import Filler

from geecs_bluesky.assets.handlers import (
    GeecsCameraImageHandler,
    GeecsPathBackedHandler,
    GeecsTextArrayHandler,
)
from geecs_bluesky.assets.registry import AssetDefinition, get_single_asset_definition
from geecs_bluesky.assets.specs import (
    GEECS_CAMERA_IMAGE,
    GEECS_TEXT_ARRAY,
)

HandlerRegistry: TypeAlias = dict[str, type[GeecsPathBackedHandler]]
Document: TypeAlias = tuple[str, dict[str, object]]
EXTERNAL_ASSET_DOCUMENT_SCHEMA = 1


@dataclass(frozen=True)
class ExternalAssetDocumentSpec:
    """Request model for a synthetic fillable external-asset document stream."""

    definition: AssetDefinition
    device_name: str
    resource_root: str
    resource_path: str
    data_key: str | None = None
    datum_id: str | None = None
    resource_uid: str | None = None
    start_doc: Mapping[str, Any] | None = None
    event: Mapping[str, Any] | None = None
    timestamp: float | None = None

    def resource_kwargs(self) -> dict[str, object]:
        """Return Resource metadata used by local asset loaders and provenance."""
        payload: dict[str, object] = {
            "data_key": self.data_key or self.definition.event_key(self.device_name),
            "device_name": self.device_name,
            "device_type": self.definition.device_type,
            "event_field": self.definition.event_field,
            "payload_kind": self.definition.payload_kind.value,
            "loader_kind": self.definition.loader_kind.value,
            "external_asset_document_schema": EXTERNAL_ASSET_DOCUMENT_SCHEMA,
        }
        if self.definition.default_data_1d_type is not None:
            payload["default_data_1d_type"] = self.definition.default_data_1d_type
        if self.definition.requires_loader_config:
            payload["requires_loader_config"] = True
        return payload

    def to_documents(self) -> list[Document]:
        """Build event-model documents for this external-asset request."""
        timestamp = time.time() if self.timestamp is None else self.timestamp
        start_uid = str((self.start_doc or {}).get("uid") or uuid4())
        descriptor_uid = str(uuid4())
        resource_uid = str(self.resource_uid or uuid4())
        datum_id = str(self.datum_id or f"{resource_uid}/0")
        data_key = self.data_key or self.definition.event_key(self.device_name)
        seq_num = int(
            (self.event or {}).get("scan_event_index")
            or (self.event or {}).get("seq_num")
            or 1
        )

        synthetic_start = {
            **dict(self.start_doc or {}),
            "uid": start_uid,
            "time": timestamp,
            "geecs_external_asset_document_schema": EXTERNAL_ASSET_DOCUMENT_SCHEMA,
        }

        return [
            ("start", synthetic_start),
            (
                "descriptor",
                {
                    "uid": descriptor_uid,
                    "run_start": start_uid,
                    "time": timestamp,
                    "name": "primary",
                    "data_keys": {
                        data_key: {
                            "source": (
                                f"geecs://{self.device_name}/"
                                f"{self.definition.event_field}"
                            ),
                            "dtype": "array",
                            "shape": [],
                            "external": "OLD:",
                        }
                    },
                    "configuration": {},
                    "object_keys": {self.device_name: [data_key]},
                    "hints": {},
                },
            ),
            (
                "resource",
                {
                    "uid": resource_uid,
                    "spec": self.definition.spec,
                    "root": self.resource_root,
                    "resource_path": self.resource_path,
                    "resource_kwargs": self.resource_kwargs(),
                    "path_semantics": "posix",
                },
            ),
            (
                "datum",
                {"datum_id": datum_id, "resource": resource_uid, "datum_kwargs": {}},
            ),
            (
                "event",
                {
                    "uid": str(uuid4()),
                    "descriptor": descriptor_uid,
                    "time": timestamp,
                    "seq_num": seq_num,
                    "data": {data_key: datum_id},
                    "timestamps": {data_key: timestamp},
                    "filled": {data_key: False},
                },
            ),
            (
                "stop",
                {
                    "uid": str(uuid4()),
                    "run_start": start_uid,
                    "time": timestamp,
                    "exit_status": "success",
                    "num_events": {"primary": 1},
                },
            ),
        ]


class HandlerRegistrar(Protocol):
    """Object that can register event-model style asset handlers."""

    def register_handler(
        self,
        spec: str,
        handler: type[GeecsPathBackedHandler],
        overwrite: bool = False,
    ) -> None:
        """Register *handler* for *spec*."""


def geecs_asset_handler_registry() -> HandlerRegistry:
    """Return the local handler registry for supported GEECS asset specs."""
    return {
        GEECS_CAMERA_IMAGE: GeecsCameraImageHandler,
        GEECS_TEXT_ARRAY: GeecsTextArrayHandler,
    }


def register_geecs_handlers(
    registrar: HandlerRegistrar,
    *,
    overwrite: bool = False,
) -> None:
    """Register supported GEECS asset handlers on an existing filler/router.

    Parameters
    ----------
    registrar:
        Object exposing ``register_handler(spec, handler, overwrite=False)``.
        This includes :class:`event_model.Filler`.
    overwrite:
        Whether to replace an existing handler for the same asset spec.
    """
    for spec, handler in geecs_asset_handler_registry().items():
        registrar.register_handler(spec, handler, overwrite=overwrite)


def make_geecs_filler(
    *,
    root_map: Mapping[str, str] | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    inplace: bool = False,
    retry_intervals: Sequence[float] | None = None,
) -> Filler:
    """Create an :class:`event_model.Filler` with GEECS handlers registered.

    Parameters
    ----------
    root_map:
        Optional mapping from Resource document roots to locally visible roots.
    include:
        Optional event data keys to fill.
    exclude:
        Optional event data keys to leave unfilled.
    inplace:
        Whether filled documents should mutate the input documents.
    retry_intervals:
        Optional retry delays used while opening not-yet-visible native files.

    Returns
    -------
    event_model.Filler
        Filler configured with the currently supported GEECS asset handlers.
    """
    return Filler(
        geecs_asset_handler_registry(),
        root_map=_filler_root_map(root_map or {}),
        include=include,
        exclude=exclude,
        inplace=inplace,
        retry_intervals=None if retry_intervals is None else list(retry_intervals),
    )


def _filler_root_map(root_map: Mapping[str, str]) -> dict[str, str]:
    """Return exact-match root aliases accepted by event-model Filler."""
    normalized: dict[str, str] = {}
    for remote_root, local_root in root_map.items():
        remote = str(remote_root)
        local = str(local_root)
        normalized[remote] = local
        normalized[_strip_trailing_slashes(remote)] = _strip_trailing_slashes(local)
    return normalized


def _strip_trailing_slashes(path: str) -> str:
    """Strip trailing path separators except for filesystem roots."""
    normalized = path.replace("\\", "/")
    if normalized in ("/", ""):
        return normalized
    return normalized.rstrip("/")


def fill_geecs_documents(
    documents: Iterable[Document],
    *,
    root_map: Mapping[str, str] | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    retry_intervals: Sequence[float] | None = None,
) -> list[Document]:
    """Fill external GEECS assets in a stream of Bluesky documents.

    Parameters
    ----------
    documents:
        Iterable of ``(name, document)`` pairs in Bluesky document order.
    root_map:
        Optional mapping from Resource document roots to locally visible roots.
    include:
        Optional event data keys to fill.
    exclude:
        Optional event data keys to leave unfilled.
    retry_intervals:
        Optional retry delays used while opening not-yet-visible native files.

    Returns
    -------
    list[tuple[str, dict]]
        Filled document stream. Event documents containing supported external
        assets have datum IDs replaced with loaded payloads.
    """
    filler = make_geecs_filler(
        root_map=root_map,
        include=include,
        exclude=exclude,
        retry_intervals=retry_intervals,
    )
    return [filler(name, doc) for name, doc in documents]


def build_external_asset_documents(
    *,
    definition: AssetDefinition,
    device_name: str,
    resource_root: str,
    resource_path: str,
    data_key: str | None = None,
    datum_id: str | None = None,
    resource_uid: str | None = None,
    start_doc: Mapping[str, Any] | None = None,
    event: Mapping[str, Any] | None = None,
    timestamp: float | None = None,
) -> list[Document]:
    """Build a minimal fillable document stream for one native external asset."""
    return ExternalAssetDocumentSpec(
        definition=definition,
        device_name=device_name,
        resource_root=resource_root,
        resource_path=resource_path,
        data_key=data_key,
        datum_id=datum_id,
        resource_uid=resource_uid,
        start_doc=start_doc,
        event=event,
        timestamp=timestamp,
    ).to_documents()


def build_camera_shot_documents(
    *,
    year: int,
    month: int | str,
    day: int,
    scan_number: int,
    device_name: str,
    shot_number: int,
    experiment: str | None = None,
    base_directory: str | Path | None = None,
    device_type: str | None = None,
) -> tuple[list[Document], Path]:
    """Build fillable Resource/Datum/Event docs for an existing camera shot.

    This is intended for notebook readback tests against historical scan
    folders. It resolves the legacy post-filemover camera filename such as
    ``Scan042_UC_TopView_001.png`` and represents it as a
    ``GEECS_CAMERA_IMAGE`` external asset.

    Parameters
    ----------
    year, month, day:
        Date of the scan. ``month`` may be an integer or month name accepted by
        :class:`geecs_data_utils.scan_paths.ScanPaths`.
    scan_number:
        GEECS scan number for that day.
    device_name:
        Camera device name and scan-folder subdirectory.
    shot_number:
        Legacy shot number in the device image filename.
    experiment:
        Optional experiment name. If omitted, ``ScanPaths`` uses the configured
        default experiment.
    base_directory:
        Optional GEECS data root. If omitted, ``ScanPaths`` uses the configured
        base path.
    device_type:
        Optional device type. If omitted, the live GEECS database is queried.

    Returns
    -------
    list[tuple[str, dict]], pathlib.Path
        Ordered Bluesky document stream and the resolved image file path.

    Raises
    ------
    FileNotFoundError
        If the scan folder or requested camera shot file cannot be found.
    ValueError
        If the resolved device type is not registered as a single camera asset.
    """
    from geecs_data_utils.scan_paths import ScanPaths

    if device_type is None:
        from geecs_bluesky.db.geecs_db import GeecsDb

        device_type = GeecsDb.get_device_type(device_name)

    definition = get_single_asset_definition(device_type)
    if definition is None or definition.spec != GEECS_CAMERA_IMAGE:
        raise ValueError(
            f"Device type {device_type!r} is not registered as a single "
            "camera image asset."
        )

    tag = ScanPaths.get_scan_tag(
        year=year,
        month=month,
        day=day,
        number=scan_number,
        experiment=experiment,
    )
    scan_paths = ScanPaths(tag=tag, base_directory=base_directory)
    scan_folder = scan_paths.get_folder()
    if scan_folder is None:
        raise FileNotFoundError(f"Could not resolve scan folder for {tag}")

    file_path = scan_paths.build_asset_path(
        shot=shot_number,
        device=device_name,
        ext="png",
    )
    if not file_path.exists():
        file_map = scan_paths.build_device_file_map(device_name, ".png")
        try:
            file_path = file_map[shot_number]
        except KeyError as exc:
            raise FileNotFoundError(
                "Could not find camera shot file for "
                f"{tag}, device={device_name!r}, shot={shot_number}. "
                f"Expected {file_path}."
            ) from exc

    now = time.time()
    resource_path = file_path.relative_to(scan_folder).as_posix()

    documents = ExternalAssetDocumentSpec(
        definition=definition,
        device_name=device_name,
        resource_root=str(scan_folder),
        resource_path=resource_path,
        start_doc={"scan_id": scan_number},
        timestamp=now,
    ).to_documents()
    return documents, file_path
