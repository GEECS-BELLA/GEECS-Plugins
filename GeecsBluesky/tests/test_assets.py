"""Tests for GeecsBluesky external asset registry and handlers."""

from __future__ import annotations

import numpy as np
import png

from geecs_bluesky.assets import (
    GEECS_CAMERA_IMAGE,
    POINTGREY_CAMERA_DEVICE_TYPE,
    GeecsCameraImageHandler,
    camera_image_filename,
    get_asset_definitions,
    get_single_asset_definition,
    supports_device_type,
)


def test_pointgrey_camera_registry_entry() -> None:
    """Point Grey Camera should register one native camera-image asset."""
    [definition] = get_asset_definitions(POINTGREY_CAMERA_DEVICE_TYPE)
    assert definition.device_type == "Point Grey Camera"
    assert definition.spec == GEECS_CAMERA_IMAGE
    assert definition.extensions == (".png",)
    assert definition.handler_class == "GeecsCameraImageHandler"
    assert definition.event_key("UC_TopView") == "uc_topview-image"
    assert get_single_asset_definition("Point Grey Camera") == definition
    assert supports_device_type("Point Grey Camera")
    assert not supports_device_type("UnknownDeviceType")


def test_camera_image_filename_uses_geecs_convention() -> None:
    """Camera filenames are based on scan number, device name, and timestamp."""
    assert (
        camera_image_filename(
            scan_number=7,
            device_name="UC_TopView",
            acq_timestamp=1234567890.1234,
        )
        == "Scan007_UC_TopView_1234567890.123.png"
    )


def test_camera_definition_builds_file_and_resource_paths(tmp_path) -> None:
    """Registry entries should build native file paths and relative resource paths."""
    definition = get_single_asset_definition("Point Grey Camera")
    assert definition is not None

    root = tmp_path / "Undulator" / "Y2026" / "06-Jun" / "26_0623"
    save_path = root / "scans" / "Scan042" / "UC_TopView"
    file_path = definition.file_path(
        save_path=save_path,
        scan_number=42,
        device_name="UC_TopView",
        acq_timestamp=1000.5,
    )

    assert file_path == save_path / "Scan042_UC_TopView_1000.500.png"
    assert (
        definition.resource_path(root=root, file_path=file_path)
        == "scans/Scan042/UC_TopView/Scan042_UC_TopView_1000.500.png"
    )


def test_camera_image_handler_loads_png(tmp_path) -> None:
    """The camera handler should delegate PNG decoding to geecs_data_utils.io."""
    root = tmp_path / "root"
    image_path = root / "Scan001" / "UC_TopView" / "Scan001_UC_TopView_1.000.png"
    image_path.parent.mkdir(parents=True)

    expected = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    with image_path.open("wb") as stream:
        png.Writer(width=2, height=2, greyscale=True, bitdepth=8).write(
            stream, expected.tolist()
        )

    handler = GeecsCameraImageHandler(
        "Scan001/UC_TopView/Scan001_UC_TopView_1.000.png",
        root=root,
    )
    np.testing.assert_array_equal(handler(), expected)
