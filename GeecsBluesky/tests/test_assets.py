"""Tests for GeecsBluesky external asset registry and handlers."""

from __future__ import annotations

import numpy as np
import png
from bluesky import RunEngine
import bluesky.plan_stubs as bps
from event_model import Filler

from geecs_bluesky.assets import (
    AssetLoaderKind,
    AssetPayloadKind,
    EXTERNAL_ASSET_DOCUMENT_SCHEMA,
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
    GeecsTextArrayHandler,
    build_external_asset_documents,
    build_camera_shot_documents,
    camera_image_filename,
    fill_geecs_documents,
    geecs_asset_handler_registry,
    get_asset_definitions,
    get_single_asset_definition,
    native_file_filename,
    register_geecs_handlers,
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
    assert definition.payload_kind is AssetPayloadKind.ARRAY_2D
    assert definition.loader_kind is AssetLoaderKind.READ_IMAQ_IMAGE
    assert not definition.requires_loader_config
    assert definition.handler_class == "GeecsCameraImageHandler"
    assert definition.event_key("UC_TopView") == "uc_topview-image"
    assert get_single_asset_definition("Point Grey Camera") == definition
    assert supports_device_type("Point Grey Camera")
    assert not supports_device_type("UnknownDeviceType")


def test_camera_image_filename_uses_geecs_convention() -> None:
    """Camera filenames are based on device name and timestamp."""
    assert (
        camera_image_filename(
            scan_number=7,
            device_name="UC_TopView",
            acq_timestamp=1234567890.1234,
        )
        == "UC_TopView_1234567890.123.png"
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
        == "U_PicoScope_1234567890.123.tdms"
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

    assert file_path == save_path / "UC_TopView_1000.500.png"
    assert (
        definition.resource_path(root=root, file_path=file_path)
        == "scans/Scan042/UC_TopView/UC_TopView_1000.500.png"
    )


def test_camera_definition_builds_cross_os_resource_paths() -> None:
    """Registry resource paths should not depend on the local OS path parser."""
    definition = get_single_asset_definition("Point Grey Camera")
    assert definition is not None

    assert (
        definition.resource_path(
            root="Z:/data",
            file_path=(
                r"Z:\data\Undulator\Y2026\06-Jun\26_0625\scans"
                r"\Scan001\UC_Amp4_IR_input\UC_Amp4_IR_input_3865254648.364.png"
            ),
        )
        == "Undulator/Y2026/06-Jun/26_0625/scans/Scan001/"
        "UC_Amp4_IR_input/UC_Amp4_IR_input_3865254648.364.png"
    )
    assert (
        definition.resource_path(
            root="Z:/data",
            local_root="/Volumes/hdna2/data",
            file_path=(
                "/Volumes/hdna2/data/Undulator/Y2026/06-Jun/26_0625/scans/"
                "Scan001/UC_Amp4_IR_input/UC_Amp4_IR_input_3865254648.364.png"
            ),
        )
        == "Undulator/Y2026/06-Jun/26_0625/scans/Scan001/"
        "UC_Amp4_IR_input/UC_Amp4_IR_input_3865254648.364.png"
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
        assert definition.payload_kind is AssetPayloadKind.ARRAY_1D
        assert definition.loader_kind is AssetLoaderKind.TDMS_SCOPE
        assert definition.loader_config_defaults == {"data_type": "tdms_scope"}
        assert not definition.requires_loader_config
        assert definition.companion_extensions == (".tdms_index",)
        assert definition.handler_class is None
        assert file_path == save_path / "U_Scope_42.125.tdms"
        assert definition.companion_file_paths(
            save_path=save_path,
            scan_number=3,
            device_name="U_Scope",
            acq_timestamp=42.125,
        ) == (save_path / "U_Scope_42.125.tdms_index",)
        assert supports_device_type(device_type)


def test_registered_asset_definitions_are_consistent() -> None:
    """All registry entries should satisfy the external-asset contract."""
    device_types = (
        FROG_DEVICE_TYPE,
        MAGSPEC_CAMERA_DEVICE_TYPE,
        MAGSPEC_STITCHER_DEVICE_TYPE,
        PICOSCOPE_V2_DEVICE_TYPE,
        POINTGREY_CAMERA_DEVICE_TYPE,
        ROHDE_SCHWARZ_RTA4000_DEVICE_TYPE,
        THORLABS_CCS175_SPECTROMETER_DEVICE_TYPE,
        THORLABS_WFS_DEVICE_TYPE,
    )

    for device_type in device_types:
        definitions = get_asset_definitions(device_type)
        event_fields = [definition.event_field for definition in definitions]

        assert definitions
        assert len(event_fields) == len(set(event_fields))
        for definition in definitions:
            assert definition.device_type == device_type
            assert definition.extensions
            assert definition.payload_kind in AssetPayloadKind
            assert definition.loader_kind in AssetLoaderKind
            if definition.uses_data_1d_reader:
                assert definition.handler_class is None
                assert definition.loader_config_defaults
            elif definition.loader_kind in {
                AssetLoaderKind.SDK_FILE,
                AssetLoaderKind.FILE,
            }:
                assert definition.handler_class is None
            else:
                assert definition.handler_class is not None


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
        / "U_FROG_Grenouille-Spatial_10.000.png"
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
        / "U_FROG_Grenouille-Temporal_10.000.png"
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
    assert by_field["interpSpec"].payload_kind is AssetPayloadKind.ARRAY_1D
    assert by_field["interpSpec"].loader_kind is AssetLoaderKind.TEXT_TABLE
    assert by_field["interpSpec"].loader_config_defaults == {"data_type": "tsv"}
    assert not by_field["interpSpec"].requires_loader_config
    assert by_field["interp_image"].directory_suffix == "-interp"
    assert by_field["interpSpec"].directory_suffix == "-interpSpec"
    assert by_field["interpDiv"].directory_suffix == "-interpDiv"
    assert by_field["interpSpec"].handler_class == "GeecsTextArrayHandler"
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
        / "U_BCaveMagSpec-interp_1000.500.png"
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
        / "U_BCaveMagSpec-interpSpec_1000.500.txt"
    )


def test_magspec_stitcher_omits_interp_image_asset() -> None:
    """MagSpecStitcher should not register the MagSpecCamera interp image."""
    definitions = get_asset_definitions(MAGSPEC_STITCHER_DEVICE_TYPE)
    by_field = {definition.event_field: definition for definition in definitions}

    assert set(by_field) == {"image", "interpSpec", "interpDiv"}
    assert by_field["image"].spec == GEECS_CAMERA_IMAGE
    assert by_field["interpSpec"].spec == GEECS_TEXT_ARRAY
    assert by_field["interpDiv"].spec == GEECS_TEXT_ARRAY
    assert by_field["interpSpec"].handler_class == "GeecsTextArrayHandler"
    assert "interp_image" not in by_field


def test_camera_image_handler_loads_png(tmp_path) -> None:
    """The camera handler should delegate PNG decoding to geecs_data_utils.io."""
    root = tmp_path / "root"
    image_path = root / "Scan001" / "UC_TopView" / "UC_TopView_1.000.png"
    image_path.parent.mkdir(parents=True)

    expected = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    with image_path.open("wb") as stream:
        png.Writer(width=2, height=2, greyscale=True, bitdepth=8).write(
            stream, expected.tolist()
        )

    handler = GeecsCameraImageHandler(
        "Scan001/UC_TopView/UC_TopView_1.000.png",
        root=root,
    )
    np.testing.assert_array_equal(handler(), expected)


def test_text_array_handler_loads_numeric_text(tmp_path) -> None:
    """The text handler should load native numeric array assets."""
    root = tmp_path / "root"
    array_path = root / "Scan001" / "U_MagSpec" / "U_MagSpec_1.000.txt"
    array_path.parent.mkdir(parents=True)
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.savetxt(array_path, expected)

    handler = GeecsTextArrayHandler(
        "Scan001/U_MagSpec/U_MagSpec_1.000.txt",
        root=root,
    )

    np.testing.assert_array_equal(handler(), expected)


def test_text_array_handler_skips_native_header(tmp_path) -> None:
    """The text handler should tolerate native MagSpec-style headers."""
    root = tmp_path / "root"
    array_path = root / "Scan001" / "U_MagSpec" / "U_MagSpec_1.000.txt"
    array_path.parent.mkdir(parents=True)
    array_path.write_text(
        "\n".join(
            [
                "Momentum_GeV/c\tChargeDen_pC/GeV\tChargeDen_pC/GeV",
                "51.5\t10.0\t10.0",
                "51.8\t20.0\t20.0",
            ]
        ),
        encoding="utf-8",
    )

    handler = GeecsTextArrayHandler(
        "Scan001/U_MagSpec/U_MagSpec_1.000.txt",
        root=root,
    )

    np.testing.assert_array_equal(
        handler(),
        np.array([[51.5, 10.0, 10.0], [51.8, 20.0, 20.0]]),
    )


def test_geecs_handler_registry_maps_supported_specs() -> None:
    """The public readback registry should expose supported native handlers."""
    registry = geecs_asset_handler_registry()
    assert registry == {
        GEECS_CAMERA_IMAGE: GeecsCameraImageHandler,
        GEECS_TEXT_ARRAY: GeecsTextArrayHandler,
    }


def test_register_geecs_handlers_adds_supported_specs_to_filler() -> None:
    """The registration helper should configure an existing filler."""
    filler = Filler({}, inplace=False, retry_intervals=[])
    register_geecs_handlers(filler, overwrite=True)

    assert filler.handler_registry[GEECS_CAMERA_IMAGE] is GeecsCameraImageHandler
    assert filler.handler_registry[GEECS_TEXT_ARRAY] is GeecsTextArrayHandler


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
    assert resource["resource_path"] == "UC_TopView/UC_TopView_1000.500.png"
    assert resource["path_semantics"] == "posix"
    assert resource["spec"] == GEECS_CAMERA_IMAGE
    assert resource["resource_kwargs"]["data_key"] == data_key
    assert resource["resource_kwargs"]["device_name"] == "UC_TopView"
    assert resource["resource_kwargs"]["device_type"] == POINTGREY_CAMERA_DEVICE_TYPE
    assert resource["resource_kwargs"]["event_field"] == "image"
    assert resource["resource_kwargs"]["payload_kind"] == "array_2d"
    assert resource["resource_kwargs"]["loader_name"] == "read_imaq_image"
    assert resource["resource_kwargs"]["loader_kind"] == "read_imaq_image"
    assert datum["datum_id"] == datum_id
    assert datum["resource"] == resource["uid"]
    assert datum["datum_kwargs"] == {}
    assert list(emitter.collect_asset_docs()) == []


def test_nonscalar_save_support_emits_canonical_resource_root(tmp_path) -> None:
    """Resource docs should support canonical roots distinct from local roots."""
    data_root = tmp_path / "data"
    scan_folder = (
        data_root / "Undulator" / "Y2026" / "06-Jun" / "26_0625" / "scans" / "Scan001"
    )
    save_path = scan_folder / "UC_TopView"
    emitter = _AssetEmitter()
    emitter.configure_nonscalar_file_logging(save_path)
    emitter.configure_external_asset_logging(
        scan_number=1,
        asset_definitions=get_asset_definitions(POINTGREY_CAMERA_DEVICE_TYPE),
        root_path="Z:/data",
        local_root_path=data_root,
    )

    reading = {}
    emitter._emit_asset_readings(
        reading,
        event_timestamp=123.0,
        acq_timestamp=3865254648.364,
    )
    resource = list(emitter.collect_asset_docs())[0][1]

    assert reading["uc_topview-image"]["value"]
    assert resource["root"] == "Z:/data"
    assert resource["path_semantics"] == "posix"
    assert resource["resource_path"] == (
        "Undulator/Y2026/06-Jun/26_0625/scans/Scan001/"
        "UC_TopView/UC_TopView_3865254648.364.png"
    )


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
    assert resource["resource_path"] == "U_Scope/U_Scope_42.125.tdms"
    assert resource["resource_kwargs"]["device_type"] == PICOSCOPE_V2_DEVICE_TYPE
    assert resource["resource_kwargs"]["event_field"] == "tdms"
    assert resource["resource_kwargs"]["payload_kind"] == "array_1d"
    assert resource["resource_kwargs"]["loader_name"] == "tdms_scope"
    assert resource["resource_kwargs"]["loader_kind"] == "tdms_scope"
    assert resource["resource_kwargs"]["loader_config_defaults"] == {
        "data_type": "tdms_scope"
    }
    assert "requires_loader_config" not in resource["resource_kwargs"]
    assert resource["resource_kwargs"]["companion_resource_paths"] == [
        "U_Scope/U_Scope_42.125.tdms_index"
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
    assert resource["resource_path"] == "UC_TopView/UC_TopView_1000.500.png"


def test_geecs_documents_fill_camera_image_asset(tmp_path) -> None:
    """GEECS Resource/Datum docs should fill camera datum IDs into arrays."""
    scan_folder = tmp_path / "scans" / "Scan042"
    image_path = scan_folder / "UC_TopView" / "UC_TopView_1000.500.png"
    image_path.parent.mkdir(parents=True)
    expected = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    with image_path.open("wb") as stream:
        png.Writer(width=2, height=2, greyscale=True, bitdepth=8).write(
            stream, expected.tolist()
        )

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

    filled_docs = fill_geecs_documents(docs, retry_intervals=[])
    event = next(doc for name, doc in filled_docs if name == "event")
    original_event = next(doc for name, doc in docs if name == "event")

    assert original_event["filled"]["uc_topview-image"] is False
    assert (
        event["filled"]["uc_topview-image"]
        == original_event["data"]["uc_topview-image"]
    )
    np.testing.assert_array_equal(event["data"]["uc_topview-image"], expected)


def test_geecs_documents_fill_with_trailing_slash_root_map(tmp_path) -> None:
    """Filler root maps should tolerate configured trailing slashes."""
    local_root = tmp_path / "data"
    scan_folder = local_root / "Undulator" / "Y2026" / "07-Jul" / "26_0701"
    image_path = (
        scan_folder / "scans" / "Scan001" / "UC_TopView" / "UC_TopView_1000.500.png"
    )
    image_path.parent.mkdir(parents=True)
    expected = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    with image_path.open("wb") as stream:
        png.Writer(width=2, height=2, greyscale=True, bitdepth=8).write(
            stream, expected.tolist()
        )

    definition = get_single_asset_definition(POINTGREY_CAMERA_DEVICE_TYPE)
    assert definition is not None
    docs = build_external_asset_documents(
        definition=definition,
        device_name="UC_TopView",
        resource_root="Z:/data",
        resource_path=(
            "Undulator/Y2026/07-Jul/26_0701/scans/Scan001/"
            "UC_TopView/UC_TopView_1000.500.png"
        ),
        data_key="uc_topview-image",
    )

    filled_docs = fill_geecs_documents(
        docs,
        root_map={"Z:/data/": f"{local_root}/"},
        retry_intervals=[],
    )
    event = next(doc for name, doc in filled_docs if name == "event")

    np.testing.assert_array_equal(event["data"]["uc_topview-image"], expected)


def test_build_camera_shot_documents_resolves_legacy_scan_file(tmp_path) -> None:
    """Camera shot helper should resolve existing legacy scan-folder images."""
    scan_folder = (
        tmp_path / "Undulator" / "Y2026" / "06-Jun" / "26_0623" / "scans" / "Scan042"
    )
    image_path = scan_folder / "UC_TopView" / "Scan042_UC_TopView_001.png"
    image_path.parent.mkdir(parents=True)
    expected = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    with image_path.open("wb") as stream:
        png.Writer(width=2, height=2, greyscale=True, bitdepth=8).write(
            stream, expected.tolist()
        )

    docs, resolved_path = build_camera_shot_documents(
        year=2026,
        month=6,
        day=23,
        scan_number=42,
        device_name="UC_TopView",
        shot_number=1,
        experiment="Undulator",
        base_directory=tmp_path,
        device_type=POINTGREY_CAMERA_DEVICE_TYPE,
    )

    assert resolved_path == image_path
    start = next(doc for name, doc in docs if name == "start")
    resource = next(doc for name, doc in docs if name == "resource")
    assert (
        start["geecs_external_asset_document_schema"] == EXTERNAL_ASSET_DOCUMENT_SCHEMA
    )
    assert resource["root"] == str(scan_folder)
    assert resource["resource_path"] == "UC_TopView/Scan042_UC_TopView_001.png"
    assert resource["resource_kwargs"]["device_type"] == POINTGREY_CAMERA_DEVICE_TYPE
    assert resource["resource_kwargs"]["event_field"] == "image"
    assert resource["resource_kwargs"]["payload_kind"] == "array_2d"
    assert resource["resource_kwargs"]["loader_name"] == "read_imaq_image"
    assert resource["resource_kwargs"]["loader_kind"] == "read_imaq_image"

    filled_docs = fill_geecs_documents(docs, retry_intervals=[])
    event = next(doc for name, doc in filled_docs if name == "event")
    np.testing.assert_array_equal(event["data"]["uc_topview-image"], expected)
