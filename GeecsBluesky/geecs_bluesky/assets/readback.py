"""Local readback helpers for GEECS external assets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import time
from typing import Protocol, TypeAlias
from uuid import uuid4

from event_model import Filler

from geecs_bluesky.assets.handlers import GeecsCameraImageHandler
from geecs_bluesky.assets.registry import get_single_asset_definition
from geecs_bluesky.assets.specs import GEECS_CAMERA_IMAGE

HandlerRegistry: TypeAlias = dict[str, type[GeecsCameraImageHandler]]
Document: TypeAlias = tuple[str, dict[str, object]]


class HandlerRegistrar(Protocol):
    """Object that can register event-model style asset handlers."""

    def register_handler(
        self,
        spec: str,
        handler: type[GeecsCameraImageHandler],
        overwrite: bool = False,
    ) -> None:
        """Register *handler* for *spec*."""


def geecs_asset_handler_registry() -> HandlerRegistry:
    """Return the local handler registry for supported GEECS asset specs."""
    return {GEECS_CAMERA_IMAGE: GeecsCameraImageHandler}


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
        root_map=dict(root_map or {}),
        include=include,
        exclude=exclude,
        inplace=inplace,
        retry_intervals=None if retry_intervals is None else list(retry_intervals),
    )


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

    data_key = definition.event_key(device_name)
    resource_uid = str(uuid4())
    datum_id = f"{resource_uid}/0"
    start_uid = str(uuid4())
    descriptor_uid = str(uuid4())
    now = time.time()
    resource_path = file_path.relative_to(scan_folder).as_posix()

    documents: list[Document] = [
        ("start", {"uid": start_uid, "time": now, "scan_id": scan_number}),
        (
            "descriptor",
            {
                "uid": descriptor_uid,
                "run_start": start_uid,
                "time": now,
                "name": "primary",
                "data_keys": {
                    data_key: {
                        "source": f"geecs://{device_name}/{definition.event_field}",
                        "dtype": "array",
                        "shape": [],
                        "external": "OLD:",
                    }
                },
                "configuration": {},
                "object_keys": {device_name: [data_key]},
                "hints": {},
            },
        ),
        (
            "resource",
            {
                "uid": resource_uid,
                "spec": GEECS_CAMERA_IMAGE,
                "root": str(scan_folder),
                "resource_path": resource_path,
                "resource_kwargs": {"data_key": data_key},
                "path_semantics": "posix",
            },
        ),
        ("datum", {"datum_id": datum_id, "resource": resource_uid, "datum_kwargs": {}}),
        (
            "event",
            {
                "uid": str(uuid4()),
                "descriptor": descriptor_uid,
                "time": now,
                "seq_num": 1,
                "data": {data_key: datum_id},
                "timestamps": {data_key: now},
                "filled": {data_key: False},
            },
        ),
        (
            "stop",
            {
                "uid": str(uuid4()),
                "run_start": start_uid,
                "time": now,
                "exit_status": "success",
                "num_events": {"primary": 1},
            },
        ),
    ]
    return documents, file_path
