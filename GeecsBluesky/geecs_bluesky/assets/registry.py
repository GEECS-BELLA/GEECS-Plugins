"""Device-type registry for native GEECS external assets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from geecs_bluesky.assets.specs import (
    FROG_DEVICE_TYPE,
    GEECS_CAMERA_IMAGE,
    GEECS_TDMS_FILE,
    GEECS_TEXT_ARRAY,
    MAGSPEC_CAMERA_DEVICE_TYPE,
    MAGSPEC_STITCHER_DEVICE_TYPE,
    PICOSCOPE_V2_DEVICE_TYPE,
    POINTGREY_CAMERA_DEVICE_TYPE,
    ROHDE_SCHWARZ_RTA4000_DEVICE_TYPE,
    THORLABS_CCS175_SPECTROMETER_DEVICE_TYPE,
    THORLABS_WFS_DEVICE_TYPE,
)
from geecs_bluesky.utils import safe_name

FilePathBuilder = Callable[[str | Path, int, str, float], Path]


class AssetPayloadKind(str, Enum):
    """Coarse payload shape expected from a native external asset."""

    ARRAY_1D = "array_1d"
    ARRAY_2D = "array_2d"
    FILE = "file"


class AssetLoaderKind(str, Enum):
    """Provenance-aware loader for a native external asset."""

    READ_IMAQ_IMAGE = "read_imaq_image"
    TEXT_TABLE = "text_table"
    TDMS_SCOPE = "tdms_scope"
    SDK_FILE = "sdk_file"
    FILE = "file"


AssetLoaderName = AssetLoaderKind


@dataclass(frozen=True)
class AssetLoaderSpec:
    """How to materialize a registered native asset.

    The loader name carries device/event provenance such as ``tdms_scope`` so
    analysis code does not have to rediscover the file interpretation from the
    extension alone. ``default_config`` is the registry-provided baseline; an
    analyzer may still override column/channel selection with its own config.
    """

    name: AssetLoaderName
    payload_kind: AssetPayloadKind
    handler_class: str | None = None
    default_config: dict[str, Any] = field(default_factory=dict)
    requires_config: bool = False
    requires_sdk: tuple[str, ...] = ()
    requires_platform: tuple[str, ...] = ()


def _normalize_path_string(path: str | Path) -> str:
    """Return *path* with POSIX separators and no trailing slash."""
    normalized = str(path).replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.rstrip("/")


def camera_image_filename(
    scan_number: int,
    device_name: str,
    acq_timestamp: float,
) -> str:
    """Return the native GEECS camera image filename.

    GEECS device servers write native files by device name and acquisition
    timestamp. Legacy scan finalization may rename files later, but external
    assets should point at the native direct-save filename.
    """
    return native_file_filename(
        scan_number=scan_number,
        device_name=device_name,
        acq_timestamp=acq_timestamp,
        extension=".png",
    )


def native_file_filename(
    scan_number: int,
    device_name: str,
    acq_timestamp: float,
    extension: str,
) -> str:
    """Return the native GEECS filename for one timestamped device file."""
    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    return f"{device_name}_{acq_timestamp:.3f}{normalized_extension}"


def _camera_image_path(
    save_path: str | Path,
    scan_number: int,
    device_name: str,
    acq_timestamp: float,
) -> Path:
    return Path(save_path) / camera_image_filename(
        scan_number=scan_number,
        device_name=device_name,
        acq_timestamp=acq_timestamp,
    )


def _native_file_path_builder(
    *,
    extension: str,
    directory_suffix: str = "",
) -> FilePathBuilder:
    def build_path(
        save_path: str | Path,
        scan_number: int,
        device_name: str,
        acq_timestamp: float,
    ) -> Path:
        directory = Path(save_path)
        if directory_suffix:
            directory = directory.parent / f"{device_name}{directory_suffix}"
        file_device_name = f"{device_name}{directory_suffix}"
        return directory / native_file_filename(
            scan_number=scan_number,
            device_name=file_device_name,
            acq_timestamp=acq_timestamp,
            extension=extension,
        )

    return build_path


@dataclass(frozen=True)
class AssetDefinition:
    """External-asset contract for one GEECS device type and event field.

    Each definition links a native GEECS file convention to the Bluesky/Tiled
    external-asset metadata needed for portable readback. Handler-backed assets
    can be filled directly from Resource/Datum/Event documents. Provenance-aware
    file loaders such as ``tdms_scope`` keep file interpretation attached to the
    device/event definition, while analyzer configs may still override channel
    or column selection.
    """

    device_type: str
    spec: str
    event_field: str
    extensions: tuple[str, ...]
    path_builder: FilePathBuilder
    loader: AssetLoaderSpec
    directory_suffix: str = ""
    companion_extensions: tuple[str, ...] = ()

    @property
    def payload_kind(self) -> AssetPayloadKind:
        """Return the payload shape produced by this asset's loader."""
        return self.loader.payload_kind

    @property
    def loader_kind(self) -> AssetLoaderName:
        """Return the provenance-aware loader name."""
        return self.loader.name

    @property
    def handler_class(self) -> str | None:
        """Return the local event-model handler class name, when available."""
        return self.loader.handler_class

    @property
    def loader_config_defaults(self) -> dict[str, Any]:
        """Return registry-provided loader config defaults."""
        return dict(self.loader.default_config)

    @property
    def requires_loader_config(self) -> bool:
        """Return whether the loader cannot run from registry defaults alone."""
        return self.loader.requires_config

    @property
    def requires_sdk(self) -> tuple[str, ...]:
        """Return external SDK requirements for this loader."""
        return self.loader.requires_sdk

    @property
    def requires_platform(self) -> tuple[str, ...]:
        """Return platform requirements for this loader."""
        return self.loader.requires_platform

    @property
    def uses_data_1d_reader(self) -> bool:
        """Return whether the shared 1D data reader materializes this asset."""
        return self.loader.name is AssetLoaderName.TDMS_SCOPE

    def event_key(self, device_name: str) -> str:
        """Return the Bluesky event data key for this asset."""
        return f"{safe_name(device_name)}-{safe_name(self.event_field)}"

    def file_path(
        self,
        *,
        save_path: str | Path,
        scan_number: int,
        device_name: str,
        acq_timestamp: float,
    ) -> Path:
        """Return the expected native file path for one event."""
        return self.path_builder(save_path, scan_number, device_name, acq_timestamp)

    def resource_path(
        self,
        *,
        root: str | Path,
        file_path: str | Path,
        local_root: str | Path | None = None,
    ) -> str:
        """Return the POSIX resource path for *file_path* relative to *root*."""
        base = _normalize_path_string(local_root or root)
        path = _normalize_path_string(file_path)
        if path == base:
            return ""
        if not path.startswith(f"{base}/"):
            raise ValueError(f"{file_path!r} is not under resource root {base!r}")
        return path.removeprefix(f"{base}/")

    def companion_file_paths(
        self,
        *,
        save_path: str | Path,
        scan_number: int,
        device_name: str,
        acq_timestamp: float,
    ) -> tuple[Path, ...]:
        """Return expected companion paths for the primary native file."""
        file_path = self.file_path(
            save_path=save_path,
            scan_number=scan_number,
            device_name=device_name,
            acq_timestamp=acq_timestamp,
        )
        return tuple(file_path.with_suffix(ext) for ext in self.companion_extensions)


POINTGREY_CAMERA_ASSET = AssetDefinition(
    device_type=POINTGREY_CAMERA_DEVICE_TYPE,
    spec=GEECS_CAMERA_IMAGE,
    event_field="image",
    extensions=(".png",),
    path_builder=_camera_image_path,
    loader=AssetLoaderSpec(
        name=AssetLoaderName.READ_IMAQ_IMAGE,
        payload_kind=AssetPayloadKind.ARRAY_2D,
        handler_class="GeecsCameraImageHandler",
    ),
)

TDMS_DEVICE_TYPES = (
    PICOSCOPE_V2_DEVICE_TYPE,
    THORLABS_CCS175_SPECTROMETER_DEVICE_TYPE,
    ROHDE_SCHWARZ_RTA4000_DEVICE_TYPE,
    THORLABS_WFS_DEVICE_TYPE,
)


def _camera_asset(
    device_type: str,
    *,
    event_field: str = "image",
    directory_suffix: str = "",
) -> AssetDefinition:
    return AssetDefinition(
        device_type=device_type,
        spec=GEECS_CAMERA_IMAGE,
        event_field=event_field,
        extensions=(".png",),
        path_builder=_native_file_path_builder(
            extension=".png",
            directory_suffix=directory_suffix,
        ),
        loader=AssetLoaderSpec(
            name=AssetLoaderName.READ_IMAQ_IMAGE,
            payload_kind=AssetPayloadKind.ARRAY_2D,
            handler_class="GeecsCameraImageHandler",
        ),
        directory_suffix=directory_suffix,
    )


def _text_array_asset(
    device_type: str,
    *,
    event_field: str,
    directory_suffix: str,
) -> AssetDefinition:
    return AssetDefinition(
        device_type=device_type,
        spec=GEECS_TEXT_ARRAY,
        event_field=event_field,
        extensions=(".txt",),
        path_builder=_native_file_path_builder(
            extension=".txt",
            directory_suffix=directory_suffix,
        ),
        loader=AssetLoaderSpec(
            name=AssetLoaderName.TEXT_TABLE,
            payload_kind=AssetPayloadKind.ARRAY_1D,
            handler_class="GeecsTextArrayHandler",
            default_config={"data_type": "tsv"},
        ),
        directory_suffix=directory_suffix,
    )


def _tdms_asset(device_type: str) -> AssetDefinition:
    return AssetDefinition(
        device_type=device_type,
        spec=GEECS_TDMS_FILE,
        event_field="tdms",
        extensions=(".tdms",),
        path_builder=_native_file_path_builder(extension=".tdms"),
        loader=AssetLoaderSpec(
            name=AssetLoaderName.TDMS_SCOPE,
            payload_kind=AssetPayloadKind.ARRAY_1D,
            default_config={"data_type": "tdms_scope"},
        ),
        companion_extensions=(".tdms_index",),
    )


FROG_ASSETS = (
    _camera_asset(FROG_DEVICE_TYPE, event_field="Spatial", directory_suffix="-Spatial"),
    _camera_asset(
        FROG_DEVICE_TYPE,
        event_field="Temporal",
        directory_suffix="-Temporal",
    ),
)

MAGSPEC_CAMERA_ASSETS = (
    _camera_asset(MAGSPEC_CAMERA_DEVICE_TYPE),
    _camera_asset(
        MAGSPEC_CAMERA_DEVICE_TYPE,
        event_field="interp_image",
        directory_suffix="-interp",
    ),
    _text_array_asset(
        MAGSPEC_CAMERA_DEVICE_TYPE,
        event_field="interpSpec",
        directory_suffix="-interpSpec",
    ),
    _text_array_asset(
        MAGSPEC_CAMERA_DEVICE_TYPE,
        event_field="interpDiv",
        directory_suffix="-interpDiv",
    ),
)

MAGSPEC_STITCHER_ASSETS = (
    _camera_asset(MAGSPEC_STITCHER_DEVICE_TYPE),
    _text_array_asset(
        MAGSPEC_STITCHER_DEVICE_TYPE,
        event_field="interpSpec",
        directory_suffix="-interpSpec",
    ),
    _text_array_asset(
        MAGSPEC_STITCHER_DEVICE_TYPE,
        event_field="interpDiv",
        directory_suffix="-interpDiv",
    ),
)

_REGISTRY: dict[str, tuple[AssetDefinition, ...]] = {
    FROG_DEVICE_TYPE: FROG_ASSETS,
    MAGSPEC_CAMERA_DEVICE_TYPE: MAGSPEC_CAMERA_ASSETS,
    MAGSPEC_STITCHER_DEVICE_TYPE: MAGSPEC_STITCHER_ASSETS,
    POINTGREY_CAMERA_DEVICE_TYPE: (POINTGREY_CAMERA_ASSET,),
    **{device_type: (_tdms_asset(device_type),) for device_type in TDMS_DEVICE_TYPES},
}


def get_asset_definitions(device_type: str) -> tuple[AssetDefinition, ...]:
    """Return registered asset definitions for a GEECS device type."""
    return _REGISTRY.get(device_type, ())


def get_single_asset_definition(device_type: str) -> AssetDefinition | None:
    """Return the only asset definition for *device_type*, if exactly one exists."""
    definitions = get_asset_definitions(device_type)
    if len(definitions) != 1:
        return None
    return definitions[0]


def supports_device_type(device_type: str) -> bool:
    """Return whether the asset registry supports *device_type*."""
    return bool(get_asset_definitions(device_type))
