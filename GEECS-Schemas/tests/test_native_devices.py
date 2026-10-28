"""NativeDevices — the native EPICS device registry (``native_devices.yaml``)."""

import pytest
from pydantic import ValidationError

from geecs_schemas import (
    DETECTOR_KINDS,
    SCHEMA_REGISTRY,
    AravisCamera,
    EpicsMotor,
    NativeDevices,
)


def test_registered_as_a_document_kind() -> None:
    assert SCHEMA_REGISTRY["native_devices"] is NativeDevices


def test_a_registry_validates_and_round_trips() -> None:
    doc = {
        "schema_version": 1,
        "devices": {
            "UC_ExampleCam": {"kind": "aravis_camera", "trigger_source": "software"},
            "UC_Other": {"kind": "aravis_camera", "trigger_source": "line1"},
            "UC_Flir": {"kind": "aravis_camera", "trigger_source": "line0"},
        },
    }
    registry = NativeDevices.model_validate(doc)
    cam = registry.devices["UC_ExampleCam"]
    assert isinstance(cam, AravisCamera) and cam.trigger_source == "software"
    assert registry.devices["UC_Other"].trigger_source == "line1"
    assert registry.devices["UC_Flir"].trigger_source == "line0"
    assert registry.model_dump(mode="json") == doc


def test_a_motor_entry_names_its_axes_and_nothing_else() -> None:
    doc = {
        "schema_version": 1,
        "devices": {"U_ESP_JetXYZ": {"kind": "epics_motor", "axes": ["x", "y", "z"]}},
    }
    registry = NativeDevices.model_validate(doc)
    stage = registry.devices["U_ESP_JetXYZ"]
    assert isinstance(stage, EpicsMotor) and stage.axes == ["x", "y", "z"]
    assert registry.model_dump(mode="json") == doc


def test_a_positioner_is_not_a_detector_kind() -> None:
    assert "aravis_camera" in DETECTOR_KINDS and "epics_motor" not in DETECTOR_KINDS


@pytest.mark.parametrize(
    "entry, why",
    [
        ({"kind": "epics_motor"}, "axes"),  # axes are required
        ({"kind": "epics_motor", "axes": []}, "at least 1"),  # and non-empty
        ({"kind": "epics_motor", "axes": ["x", " "]}, "record name"),
        ({"kind": "epics_motor", "axes": ["Axis 1"]}, "record name"),
        ({"kind": "epics_motor", "axes": ["m1", "M1"]}, "same axis"),  # one attribute
        ({"kind": "epics_motor", "axes": ["x"], "trigger_source": "line1"}, "extra"),
        (
            {"kind": "epics_motor", "axes": ["x"], "tolerance": 0.01},
            "extra",
        ),  # the record's
    ],
)
def test_a_bad_motor_entry_is_refused(entry, why) -> None:
    with pytest.raises(ValidationError, match=why):
        NativeDevices.model_validate({"devices": {"U_Stage": entry}})


def test_an_empty_document_is_no_devices() -> None:
    assert NativeDevices.model_validate({}).devices == {}


@pytest.mark.parametrize(
    "entry",
    [
        {"kind": "basler_camera", "trigger_source": "line1"},  # unknown kind
        {"trigger_source": "line1"},  # no kind
        {"kind": "aravis_camera", "trigger_source": "Line2"},  # bad source
        {"kind": "aravis_camera"},  # source is required
        {"kind": "aravis_camera", "trigger_source": "line1", "prefix": "X:Y:"},
    ],
)
def test_a_bad_entry_is_refused(entry) -> None:
    with pytest.raises(ValidationError):
        NativeDevices.model_validate({"devices": {"UC_Cam": entry}})


@pytest.mark.parametrize("version", [0, 2])
def test_another_schema_version_is_refused(version) -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        NativeDevices.model_validate({"schema_version": version, "devices": {}})


def test_a_blank_device_name_is_refused() -> None:
    entry = {"kind": "aravis_camera", "trigger_source": "line1"}
    with pytest.raises(ValidationError, match="blank"):
        NativeDevices.model_validate({"devices": {" ": entry}})


def test_template_defaults_from_the_kind_and_stays_out_of_the_dump() -> None:
    doc = {
        "schema_version": 1,
        "devices": {
            "U_Stage": {"kind": "epics_motor", "axes": ["x"]},
            "U_Supply": {
                "kind": "epics_motor",
                "axes": ["x"],
                "template": "easydriver",
            },
            "UC_Cam": {"kind": "aravis_camera", "trigger_source": "line1"},
        },
    }
    registry = NativeDevices.model_validate(doc)
    assert {n: d.family_template for n, d in registry.devices.items()} == {
        "U_Stage": "motor_record",
        "U_Supply": "easydriver",
        "UC_Cam": "ad_camera_chain",
    }
    # An entry without a template dumps as it was written: the worker reads this file.
    assert registry.model_dump(mode="json") == doc


@pytest.mark.parametrize(
    "extra, why",
    [
        ({"template": "no_such_family"}, "not an installed family"),
        ({"template": "Motor"}, "pattern"),
        ({"family": "motor_record"}, "extra"),  # unknown keys still bite
    ],
)
def test_a_bad_template_is_refused(extra, why) -> None:
    entry = {"kind": "epics_motor", "axes": ["x"], **extra}
    with pytest.raises(ValidationError, match=why):
        NativeDevices.model_validate({"devices": {"U_Stage": entry}})
