"""Device-type registry for native GEECS external assets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from geecs_bluesky.assets.specs import (
    GEECS_CAMERA_IMAGE,
    POINTGREY_CAMERA_DEVICE_TYPE,
)
from geecs_bluesky.utils import safe_name

FilePathBuilder = Callable[[str | Path, int, str, float], Path]


def camera_image_filename(
    scan_number: int,
    device_name: str,
    acq_timestamp: float,
) -> str:
    """Return the native GEECS camera image filename.

    GEECS camera files are named by scan number, device name, and the device
    acquisition timestamp rounded to milliseconds.
    """
    return f"Scan{scan_number:03d}_{device_name}_{acq_timestamp:.3f}.png"


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


@dataclass(frozen=True)
class AssetDefinition:
    """External-asset behavior for one GEECS device type."""

    device_type: str
    spec: str
    event_field: str
    extensions: tuple[str, ...]
    handler_class: str
    path_builder: FilePathBuilder

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


POINTGREY_CAMERA_ASSET = AssetDefinition(
    device_type=POINTGREY_CAMERA_DEVICE_TYPE,
    spec=GEECS_CAMERA_IMAGE,
    event_field="image",
    extensions=(".png",),
    handler_class="GeecsCameraImageHandler",
    path_builder=_camera_image_path,
)

_REGISTRY: dict[str, tuple[AssetDefinition, ...]] = {
    POINTGREY_CAMERA_DEVICE_TYPE: (POINTGREY_CAMERA_ASSET,),
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
