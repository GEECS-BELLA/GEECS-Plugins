"""NativeDevices — the native EPICS device registry (``native_devices.yaml``)."""

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY, AravisCamera, NativeDevices


def test_registered_as_a_document_kind() -> None:
    assert SCHEMA_REGISTRY["native_devices"] is NativeDevices


def test_a_registry_validates_and_round_trips() -> None:
    doc = {
        "schema_version": 1,
        "devices": {
            "UC_ExampleCam": {"kind": "aravis_camera", "trigger_source": "software"},
            "UC_Other": {"kind": "aravis_camera", "trigger_source": "line1"},
        },
    }
    registry = NativeDevices.model_validate(doc)
    cam = registry.devices["UC_ExampleCam"]
    assert isinstance(cam, AravisCamera) and cam.trigger_source == "software"
    assert registry.devices["UC_Other"].trigger_source == "line1"
    assert registry.model_dump(mode="json") == doc


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
