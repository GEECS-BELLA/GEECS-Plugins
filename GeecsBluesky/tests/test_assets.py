"""Tests for GeecsBluesky external asset registry and handlers."""

from __future__ import annotations

import numpy as np
import png
from bluesky import RunEngine
import bluesky.plan_stubs as bps

from geecs_bluesky.assets import (
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
    GeecsCameraImageHandler,
    camera_image_filename,
    get_asset_definitions,
    get_single_asset_definition,
    native_file_filename,
    supports_device_type,
)
from geecs_bluesky.devices.nonscalar_save import NonScalarSaveSupport


class _AssetEmitter(NonScalarSaveSupport):
    """Minimal object exposing the NonScalarSaveSupport asset-doc methods."""

    name = "uc_topview"
    parent = None
    _geecs_device_name = "UC_TopView"
    _save_nonscalar_data = True


class _ReadableAssetEmitter(_AssetEmitter):
    """Minimal Bluesky-readable object backed by an external asset."""

    def describe(self):
        """Return the external asset data keys."""
        return self._asset_datakeys()

    def read(self):
        """Return one datum id and queue the matching Resource/Datum docs."""
        reading = {}
        self._emit_asset_readings(
            reading,
            event_timestamp=123.0,
            acq_timestamp=1000.5,
        )
        return reading


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


def test_native_file_filename_normalizes_extension() -> None:
    """Native filenames should use the standard timestamp convention."""
    assert (
        native_file_filename(
            scan_number=7,
            device_name="U_PicoScope",
            acq_timestamp=1234567890.1234,
            extension="tdms",
        )
        == "Scan007_U_PicoScope_1234567890.123.tdms"
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


def test_tdms_device_types_register_primary_file_with_index_companion(tmp_path) -> None:
    """Scope/spectrometer TDMS devices should register a TDMS asset."""
    tdms_device_types = (
        PICOSCOPE_V2_DEVICE_TYPE,
        THORLABS_CCS175_SPECTROMETER_DEVICE_TYPE,
        ROHDE_SCHWARZ_RTA4000_DEVICE_TYPE,
        THORLABS_WFS_DEVICE_TYPE,
    )
    save_path = tmp_path / "scans" / "Scan003" / "U_Scope"

    for device_type in tdms_device_types:
        [definition] = get_asset_definitions(device_type)
        file_path = definition.file_path(
            save_path=save_path,
            scan_number=3,
            device_name="U_Scope",
            acq_timestamp=42.125,
        )

        assert definition.spec == GEECS_TDMS_FILE
        assert definition.event_field == "tdms"
        assert definition.extensions == (".tdms",)
        assert definition.companion_extensions == (".tdms_index",)
        assert definition.handler_class is None
        assert file_path == save_path / "Scan003_U_Scope_42.125.tdms"
        assert definition.companion_file_paths(
            save_path=save_path,
            scan_number=3,
            device_name="U_Scope",
            acq_timestamp=42.125,
        ) == (save_path / "Scan003_U_Scope_42.125.tdms_index",)
        assert supports_device_type(device_type)


def test_frog_registers_spatial_and_temporal_camera_assets(tmp_path) -> None:
    """FROG saves camera images in sibling Spatial and Temporal directories."""
    definitions = get_asset_definitions(FROG_DEVICE_TYPE)
    by_field = {definition.event_field: definition for definition in definitions}
    save_path = tmp_path / "scans" / "Scan015" / "U_FROG_Grenouille"

    assert set(by_field) == {"Spatial", "Temporal"}
    assert get_single_asset_definition(FROG_DEVICE_TYPE) is None
    assert (
        by_field["Spatial"].event_key("U_FROG_Grenouille")
        == "u_frog_grenouille-spatial"
    )
    assert (
        by_field["Temporal"].event_key("U_FROG_Grenouille")
        == "u_frog_grenouille-temporal"
    )
    assert by_field["Spatial"].file_path(
        save_path=save_path,
        scan_number=15,
        device_name="U_FROG_Grenouille",
        acq_timestamp=10.0,
    ) == (
        tmp_path
        / "scans"
        / "Scan015"
        / "U_FROG_Grenouille-Spatial"
        / "Scan015_U_FROG_Grenouille_10.000.png"
    )
    assert by_field["Temporal"].file_path(
        save_path=save_path,
        scan_number=15,
        device_name="U_FROG_Grenouille",
        acq_timestamp=10.0,
    ) == (
        tmp_path
        / "scans"
        / "Scan015"
        / "U_FROG_Grenouille-Temporal"
        / "Scan015_U_FROG_Grenouille_10.000.png"
    )


def test_magspec_camera_registers_image_and_variant_assets(tmp_path) -> None:
    """MagSpecCamera should include base image, interp image, and text variants."""
    definitions = get_asset_definitions(MAGSPEC_CAMERA_DEVICE_TYPE)
    by_field = {definition.event_field: definition for definition in definitions}
    save_path = tmp_path / "scans" / "Scan042" / "U_BCaveMagSpec"

    assert set(by_field) == {"image", "interp_image", "interpSpec", "interpDiv"}
    assert by_field["image"].spec == GEECS_CAMERA_IMAGE
    assert by_field["interp_image"].spec == GEECS_CAMERA_IMAGE
    assert by_field["interpSpec"].spec == GEECS_TEXT_ARRAY
    assert by_field["interpDiv"].spec == GEECS_TEXT_ARRAY
    assert by_field["interp_image"].directory_suffix == "-interp"
    assert by_field["interpSpec"].directory_suffix == "-interpSpec"
    assert by_field["interpDiv"].directory_suffix == "-interpDiv"
    assert (
        by_field["interpSpec"].event_key("U_BCaveMagSpec")
        == "u_bcavemagspec-interpspec"
    )
    assert (
        by_field["interpDiv"].event_key("U_BCaveMagSpec") == "u_bcavemagspec-interpdiv"
    )
    assert by_field["interp_image"].file_path(
        save_path=save_path,
        scan_number=42,
        device_name="U_BCaveMagSpec",
        acq_timestamp=1000.5,
    ) == (
        tmp_path
        / "scans"
        / "Scan042"
        / "U_BCaveMagSpec-interp"
        / "Scan042_U_BCaveMagSpec_1000.500.png"
    )
    assert by_field["interpSpec"].file_path(
        save_path=save_path,
        scan_number=42,
        device_name="U_BCaveMagSpec",
        acq_timestamp=1000.5,
    ) == (
        tmp_path
        / "scans"
        / "Scan042"
        / "U_BCaveMagSpec-interpSpec"
        / "Scan042_U_BCaveMagSpec_1000.500.txt"
    )


def test_magspec_stitcher_omits_interp_image_asset() -> None:
    """MagSpecStitcher should not register the MagSpecCamera interp image."""
    definitions = get_asset_definitions(MAGSPEC_STITCHER_DEVICE_TYPE)
    by_field = {definition.event_field: definition for definition in definitions}

    assert set(by_field) == {"image", "interpSpec", "interpDiv"}
    assert by_field["image"].spec == GEECS_CAMERA_IMAGE
    assert by_field["interpSpec"].spec == GEECS_TEXT_ARRAY
    assert by_field["interpDiv"].spec == GEECS_TEXT_ARRAY
    assert "interp_image" not in by_field


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


def test_nonscalar_save_support_emits_camera_asset_docs(tmp_path) -> None:
    """NonScalarSaveSupport should pair datum readings with Resource/Datum docs."""
    scan_folder = tmp_path / "scans" / "Scan042"
    save_path = scan_folder / "UC_TopView"
    emitter = _AssetEmitter()
    emitter.configure_nonscalar_file_logging(save_path)
    emitter.configure_external_asset_logging(
        scan_number=42,
        asset_definitions=get_asset_definitions(POINTGREY_CAMERA_DEVICE_TYPE),
        root_path=scan_folder,
    )

    data_key = "uc_topview-image"
    data_keys = emitter._asset_datakeys()
    assert data_keys[data_key]["external"] == "OLD:"

    reading = {}
    emitter._emit_asset_readings(
        reading,
        event_timestamp=123.0,
        acq_timestamp=1000.5,
    )
    datum_id = reading[data_key]["value"]
    assert datum_id

    docs = list(emitter.collect_asset_docs())
    assert [name for name, _doc in docs] == ["resource", "datum"]
    resource = docs[0][1]
    datum = docs[1][1]
    assert resource["root"] == str(scan_folder)
    assert resource["resource_path"] == "UC_TopView/Scan042_UC_TopView_1000.500.png"
    assert resource["spec"] == GEECS_CAMERA_IMAGE
    assert resource["resource_kwargs"] == {"data_key": data_key}
    assert datum["datum_id"] == datum_id
    assert datum["resource"] == resource["uid"]
    assert datum["datum_kwargs"] == {}
    assert list(emitter.collect_asset_docs()) == []


def test_nonscalar_save_support_records_tdms_companion_paths(tmp_path) -> None:
    """TDMS assets should include their index file as resource metadata."""
    scan_folder = tmp_path / "scans" / "Scan003"
    save_path = scan_folder / "U_Scope"
    emitter = _AssetEmitter()
    emitter._geecs_device_name = "U_Scope"
    emitter.configure_nonscalar_file_logging(save_path)
    emitter.configure_external_asset_logging(
        scan_number=3,
        asset_definitions=get_asset_definitions(PICOSCOPE_V2_DEVICE_TYPE),
        root_path=scan_folder,
    )

    reading = {}
    emitter._emit_asset_readings(
        reading,
        event_timestamp=123.0,
        acq_timestamp=42.125,
    )
    docs = list(emitter.collect_asset_docs())
    resource = docs[0][1]

    assert reading["u_scope-tdms"]["value"]
    assert resource["resource_path"] == "U_Scope/Scan003_U_Scope_42.125.tdms"
    assert resource["resource_kwargs"]["companion_resource_paths"] == [
        "U_Scope/Scan003_U_Scope_42.125.tdms_index"
    ]


def test_run_engine_emits_external_asset_docs_from_readable(tmp_path) -> None:
    """RunEngine should emit queued Resource/Datum docs with the event."""
    scan_folder = tmp_path / "scans" / "Scan042"
    emitter = _ReadableAssetEmitter()
    emitter.configure_nonscalar_file_logging(scan_folder / "UC_TopView")
    emitter.configure_external_asset_logging(
        scan_number=42,
        asset_definitions=get_asset_definitions(POINTGREY_CAMERA_DEVICE_TYPE),
        root_path=scan_folder,
    )
    docs = []

    def plan():
        yield from bps.open_run()
        yield from bps.create()
        yield from bps.read(emitter)
        yield from bps.save()
        yield from bps.close_run()

    RunEngine({})(plan(), lambda name, doc: docs.append((name, doc)))

    assert [name for name, _doc in docs] == [
        "start",
        "descriptor",
        "resource",
        "datum",
        "event",
        "stop",
    ]
    event = next(doc for name, doc in docs if name == "event")
    resource = next(doc for name, doc in docs if name == "resource")
    datum = next(doc for name, doc in docs if name == "datum")
    assert event["data"]["uc_topview-image"] == datum["datum_id"]
    assert event["filled"]["uc_topview-image"] is False
    assert resource["root"] == str(scan_folder)
    assert resource["resource_path"] == "UC_TopView/Scan042_UC_TopView_1000.500.png"
