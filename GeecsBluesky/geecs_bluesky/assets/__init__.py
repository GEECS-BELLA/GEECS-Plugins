"""External asset helpers for native GEECS files."""

from geecs_bluesky.assets.handlers import GeecsCameraImageHandler
from geecs_bluesky.assets.registry import (
    AssetDefinition,
    camera_image_filename,
    get_asset_definitions,
    get_single_asset_definition,
    native_file_filename,
    supports_device_type,
)
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

__all__ = [
    "AssetDefinition",
    "FROG_DEVICE_TYPE",
    "GEECS_CAMERA_IMAGE",
    "GEECS_TDMS_FILE",
    "GEECS_TEXT_ARRAY",
    "GeecsCameraImageHandler",
    "MAGSPEC_CAMERA_DEVICE_TYPE",
    "MAGSPEC_STITCHER_DEVICE_TYPE",
    "PICOSCOPE_V2_DEVICE_TYPE",
    "POINTGREY_CAMERA_DEVICE_TYPE",
    "ROHDE_SCHWARZ_RTA4000_DEVICE_TYPE",
    "THORLABS_CCS175_SPECTROMETER_DEVICE_TYPE",
    "THORLABS_WFS_DEVICE_TYPE",
    "camera_image_filename",
    "get_asset_definitions",
    "get_single_asset_definition",
    "native_file_filename",
    "supports_device_type",
]
