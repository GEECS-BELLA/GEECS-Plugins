"""NativeDevices — the native EPICS device registry (``native_devices.yaml``)."""

import pytest
from pydantic import ValidationError

from geecs_schemas import (
    DETECTOR_KINDS,
    SCHEMA_REGISTRY,
    AravisCamera,
    EpicsMotor,
    REGISTRY_KIND_FAMILIES,
    NativeDevices,
    SetpointDevice,
    load_families,
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


SUPPLY = {
    "kind": "setpoint_device",
    "settables": {"current": {"tolerance": 0.005, "timeout": 10.0}},
    "readbacks": ["voltage"],
    "status": ["fault", "interlock"],
}


def test_a_setpoint_device_names_its_records_and_round_trips() -> None:
    doc = {"schema_version": 1, "devices": {"U_VS1H": SUPPLY}}
    registry = NativeDevices.model_validate(doc)
    supply = registry.devices["U_VS1H"]
    assert isinstance(supply, SetpointDevice)
    assert supply.settables["current"].tolerance == 0.005
    assert supply.readbacks == ["voltage"] and supply.status == ["fault", "interlock"]
    assert supply.family_template == "easydriver"  # the kind's default
    assert registry.model_dump(mode="json") == doc


def test_a_setpoint_device_needs_only_its_settables_tolerances() -> None:
    entry = {"kind": "setpoint_device", "settables": {"current": {"tolerance": 0.01}}}
    supply = NativeDevices.model_validate({"devices": {"U": entry}}).devices["U"]
    assert supply.settables["current"].timeout == 10.0
    assert supply.readbacks == [] and supply.status == []


def test_a_setpoint_device_is_not_a_detector_kind() -> None:
    assert "setpoint_device" not in DETECTOR_KINDS


@pytest.mark.parametrize(
    "change, why",
    [
        ({"settables": {}}, "at least 1"),
        ({"settables": {"current": {}}}, "tolerance"),  # required
        ({"settables": {"current": {"tolerance": 0}}}, "greater than 0"),
        ({"settables": {"current": {"tolerance": 0.1, "timeout": -1}}}, "greater"),
        ({"settables": {"current": {"tolerance": float("inf")}}}, "finite"),
        ({"settables": {"current": {"tolerance": float("nan")}}}, "finite"),
        (
            {"settables": {"current": {"tolerance": 0.1, "timeout": float("inf")}}},
            "finite",
        ),
        ({"settables": {"current": {"tolerance": 0.1, "units": "A"}}}, "extra"),
        ({"settables": {"output current": {"tolerance": 0.1}}}, "record name"),
        ({"readbacks": ["Current"]}, "same record"),  # a settable's readback again
        ({"status": ["fault", "Fault"]}, "same record"),
        ({"readbacks": ["voltage", " "]}, "record name"),
        ({"template": "motor_record"}, "'epics_motor' family"),  # another kind's
        ({"axes": ["x"]}, "extra"),
        (
            {"settables": {"a--b": {"tolerance": 0.1}}, "readbacks": ["a_b"]},
            "same record",
        ),
        ({"settables": {"current:SP": {"tolerance": 0.1}}}, "is a setpoint record"),
        ({"readbacks": ["current:SP"]}, "has it as a setpoint"),  # the family's
        (
            {
                "settables": {"slewrate": {"tolerance": 0.1}},
                "readbacks": ["slewrate:SP"],
            },
            "setpoint records of this device",
        ),
        # Checked against the family: a read-only role is no settable, a
        # bit is no readback, and a status bit is one of the family's faults.
        (
            {"settables": {"voltage": {"tolerance": 0.1}}, "readbacks": []},
            "no setpoint",
        ),
        ({"readbacks": ["enable_output"]}, "a bit, not a readback"),
        ({"status": ["enable_output"]}, "not a fault bit"),  # reads 1 when on
        ({"status": ["interlock"], "template": "fastps"}, "interlock_1, interlock_2"),
    ],
)
def test_a_bad_setpoint_device_is_refused(change, why) -> None:
    with pytest.raises(ValidationError, match=why):
        NativeDevices.model_validate({"devices": {"U_VS1H": {**SUPPLY, **change}}})


def test_records_the_family_does_not_name_are_the_templates_business() -> None:
    """easydriver.json omits the template's slewrate and temperature records."""
    entry = {
        **SUPPLY,
        "settables": {"current": {"tolerance": 0.005}, "slewrate": {"tolerance": 0.01}},
        "readbacks": ["voltage", "temperature"],
    }
    supply = NativeDevices.model_validate({"devices": {"U": entry}}).devices["U"]
    family = load_families()["easydriver"]
    assert supply.setpoint_records(family) == {
        "current": "current:SP",
        "slewrate": "slewrate:SP",
    }
    assert supply.setpoint_records(None) == {
        "current": "current:SP",
        "slewrate": "slewrate:SP",
    }


def test_the_names_normalise_as_geecs_core_does() -> None:
    normalize_component = pytest.importorskip(
        "geecs_core.pv_naming"
    ).normalize_component
    from geecs_schemas.epics import _normalised

    for name in ("a--b", "Current", "_x_", "Position.Axis 1", "a__b", "m1"):
        assert _normalised(name) == normalize_component(name)


def test_the_fastps_family_writes_its_ramp_record() -> None:
    entry = {**SUPPLY, "template": "fastps", "status": ["interlock_1"]}
    supply = NativeDevices.model_validate({"devices": {"U": entry}}).devices["U"]
    families = load_families()
    assert families[supply.family_template].setpoint_for("current") == "current_ramp:SP"
    assert supply.setpoint_records(families["fastps"]) == {"current": "current_ramp:SP"}
    assert families["easydriver"].setpoint_for("current") == "current:SP"
    assert families["easydriver"].setpoint_for("voltage") is None  # read only
    assert families["easydriver"].setpoint_for("slewrate") is None  # not a role


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
            "U_Named": {
                "kind": "epics_motor",
                "axes": ["x"],
                "template": "motor_record",
            },
            "UC_Cam": {"kind": "aravis_camera", "trigger_source": "line1"},
        },
    }
    registry = NativeDevices.model_validate(doc)
    assert {n: d.family_template for n, d in registry.devices.items()} == {
        "U_Stage": "motor_record",
        "U_Named": "motor_record",
        "UC_Cam": "ad_camera_chain",
    }
    # An entry without a template dumps as it was written: the worker reads this file.
    assert registry.model_dump(mode="json") == doc


@pytest.mark.parametrize(
    "extra, why",
    [
        ({"template": "no_such_family"}, "not an installed family"),
        ({"template": "easydriver"}, "'setpoint_device' family"),  # another kind's
        ({"template": "Motor"}, "pattern"),
        ({"family": "motor_record"}, "extra"),  # unknown keys still bite
    ],
)
def test_a_bad_template_is_refused(extra, why) -> None:
    entry = {"kind": "epics_motor", "axes": ["x"], **extra}
    with pytest.raises(ValidationError, match=why):
        NativeDevices.model_validate({"devices": {"U_Stage": entry}})


def test_each_kind_defaults_to_a_family_of_its_own_kind() -> None:
    families = load_families()
    assert {kind: families[t].kind for kind, t in REGISTRY_KIND_FAMILIES.items()} == {
        kind: kind for kind in REGISTRY_KIND_FAMILIES
    }


# ---- the witness ------------------------------------------------------------


def test_a_witness_defaults_its_counter_and_domain_and_round_trips() -> None:
    from geecs_schemas import Witness

    doc = {"schema_version": 1, "devices": {"daqtest1": {"kind": "witness"}}}
    registry = NativeDevices.model_validate(doc)
    witness = registry.devices["daqtest1"]
    assert isinstance(witness, Witness)
    assert (witness.counter, witness.domain) == ("Counter1", "main")
    assert witness.count_record == "Counter1Counts"
    assert witness.family_template == "witness_counter"
    assert "witness" not in DETECTOR_KINDS
    assert NativeDevices.model_validate(registry.model_dump(mode="json")) == registry


@pytest.mark.parametrize(
    "entry, why",
    [
        ({"kind": "witness", "counter": "Counter 1"}, "pattern"),
        ({"kind": "witness", "domain": "Main Gate"}, "pattern"),
        ({"kind": "witness", "counter": "Counter2"}, "counts at 'Counter1Counts'"),
        ({"kind": "witness", "edges": 1}, "Extra inputs"),
    ],
)
def test_a_bad_witness_is_refused(entry, why) -> None:
    with pytest.raises(ValidationError, match=why):
        NativeDevices.model_validate({"devices": {"daqtest1": entry}})


def test_two_witnesses_of_one_domain_are_refused_naming_both() -> None:
    devices = {
        "U_WitnessA": {"kind": "witness"},
        "U_WitnessB": {"kind": "witness", "domain": "main"},
    }
    with pytest.raises(ValidationError, match="U_WitnessA.*U_WitnessB.*one witness"):
        NativeDevices.model_validate({"devices": devices})
    devices["U_WitnessB"]["domain"] = "probe"
    assert len(NativeDevices.model_validate({"devices": devices}).devices) == 2
