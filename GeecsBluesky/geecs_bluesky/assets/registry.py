"""Device-type registry for native GEECS external assets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
        return directory / native_file_filename(
            scan_number=scan_number,
            device_name=device_name,
            acq_timestamp=acq_timestamp,
            extension=extension,
        )

    return build_path


@dataclass(frozen=True)
class AssetDefinition:
    """External-asset behavior for one GEECS device type."""

    device_type: str
    spec: str
    event_field: str
    extensions: tuple[str, ...]
    path_builder: FilePathBuilder
    handler_class: str | None = None
    directory_suffix: str = ""
    companion_extensions: tuple[str, ...] = ()

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

    def resource_path(self, *, root: str | Path, file_path: str | Path) -> str:
        """Return the POSIX resource path for *file_path* relative to *root*."""
        return Path(file_path).relative_to(root).as_posix()

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
    handler_class="GeecsCameraImageHandler",
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
        handler_class="GeecsCameraImageHandler",
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
        directory_suffix=directory_suffix,
    )


def _tdms_asset(device_type: str) -> AssetDefinition:
    return AssetDefinition(
        device_type=device_type,
        spec=GEECS_TDMS_FILE,
        event_field="tdms",
        extensions=(".tdms",),
        path_builder=_native_file_path_builder(extension=".tdms"),
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
