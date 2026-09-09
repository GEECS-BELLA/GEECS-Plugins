"""GeecsDevice / GeecsTriggeredDevice — the namespace noun (#807 phase 1).

Mock-backend tests: no gateway, no network.  Pins the design decisions in
``Planning/native_bluesky/01_device_namespace.md``: children by variable,
settables as Movables, non-scalars skipped, selection through ``configure``,
``acq_timestamp`` always read on a triggered device.
"""

from __future__ import annotations

import asyncio

import pytest
from bluesky.protocols import Configurable, Readable, Stageable, Triggerable
from ophyd_async.core import SignalR, set_mock_value

from geecs_bluesky.devices.ca.motor import CaMotor
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.devices.geecs_device import (
    GeecsDevice,
    GeecsTriggeredDevice,
    VariableMeta,
    identifier_name,
)

CAMERA_ROWS = [
    {"name": "acq_timestamp", "settable": False, "variabletype": "numeric"},
    {
        "name": "MeanCounts",
        "settable": False,
        "variabletype": "numeric",
        "units": "cts",
    },
    {"name": "MaxCounts", "settable": False, "variabletype": "numeric"},
    {"name": "exposure", "settable": True, "variabletype": "numeric"},
    {"name": "image", "settable": False, "variabletype": "image"},
    {"name": "localsavingpath", "settable": True, "variabletype": "path"},
    {"name": "CONNECTED", "settable": False, "variabletype": "string"},
]

MAGNET_ROWS = [
    {"name": "current", "settable": True, "variabletype": "numeric", "tolerance": 0.01},
    {"name": "voltage", "settable": False, "variabletype": "numeric"},
    {"name": "Position.Axis 1", "settable": False, "variabletype": "numeric"},
]


def test_identifier_name_keeps_geecs_spelling_when_valid() -> None:
    assert identifier_name("U_S1H") == "U_S1H"
    assert identifier_name("UC_Amp4Input") == "UC_Amp4Input"
    assert identifier_name("Position.Axis 1") == "position_axis_1"


def test_variable_meta_from_db_normalises_type_and_flags() -> None:
    meta = VariableMeta.from_db(
        {"name": "x", "settable": True, "variabletype": " Numeric "}
    )
    assert meta.settable is True and meta.variabletype == "numeric"
    assert meta.is_scalar and meta.datatype is float
    # explicit non-numeric types and untyped rows are inferred at connect
    assert VariableMeta.from_db({"name": "p", "variabletype": "path"}).datatype is None
    assert VariableMeta.from_db({"name": "u"}).datatype is None
    # an untyped row with a tolerance is a numeric (U_S1H:Current in the DB)
    assert VariableMeta.from_db({"name": "c", "tolerance": 0.05}).datatype is float
    assert not VariableMeta.from_db({"name": "im", "variabletype": "image"}).is_scalar


def test_children_follow_variable_kind() -> None:
    cam = GeecsTriggeredDevice("UC_TestCam", CAMERA_ROWS, experiment="Undulator")
    # readbacks are plain signals; settables are Movables; non-scalars absent
    assert isinstance(cam.MeanCounts, SignalR)
    assert isinstance(cam.exposure, CaSettable) and not isinstance(
        cam.exposure, CaMotor
    )
    assert isinstance(cam.localsavingpath, CaSettable)  # string settable
    assert not hasattr(cam, "image")
    assert not hasattr(cam, "CONNECTED") and isinstance(cam.connected_status, SignalR)
    assert cam.variables == ("MeanCounts", "MaxCounts", "exposure", "localsavingpath")
    assert cam.settable_variables == ("exposure", "localsavingpath")
    assert cam.name == "UC_TestCam" and cam.geecs_name == "UC_TestCam"
    assert cam.MeanCounts.name == "UC_TestCam-MeanCounts"
    assert isinstance(cam, Triggerable)


def test_tolerance_or_catalog_kind_makes_a_motor() -> None:
    magnet = GeecsDevice("U_S1H", MAGNET_ROWS, experiment="Undulator")
    assert isinstance(magnet.current, CaMotor)
    assert magnet.current._tolerance == pytest.approx(0.01)
    assert not isinstance(magnet, Triggerable)
    assert magnet.position_axis_1.name == "U_S1H-position_axis_1"
    # no tolerance in the DB, but the catalog says motor → CaMotor with the default tol
    rows = [{"name": "current", "settable": True, "variabletype": "numeric"}]
    by_catalog = GeecsDevice(
        "U_X", rows, experiment="Undulator", motor_variables=["current"]
    )
    assert isinstance(by_catalog.current, CaMotor)
    plain = GeecsDevice("U_Y", rows, experiment="Undulator")
    assert isinstance(plain.current, CaSettable) and not isinstance(
        plain.current, CaMotor
    )


def test_lookups_accept_geecs_or_attribute_spelling() -> None:
    magnet = GeecsDevice("U_S1H", MAGNET_ROWS, experiment="Undulator")
    assert magnet.child("Position.Axis 1") is magnet.position_axis_1
    assert magnet.child("position_axis_1") is magnet.position_axis_1
    assert magnet.child("CURRENT") is magnet.current
    assert magnet.readback("current") is magnet.current.position
    assert magnet.meta("voltage").settable is False
    with pytest.raises(KeyError, match="no scalar variable 'nope'"):
        magnet.child("nope")


def test_attribute_collision_is_loud() -> None:
    rows = [
        {"name": "Position.Axis 1", "variabletype": "numeric"},
        {"name": "position axis 1", "variabletype": "numeric"},
    ]
    with pytest.raises(ValueError, match="collide on attribute"):
        GeecsDevice("U_Z", rows, experiment="Undulator")


async def test_default_selection_is_the_subscribed_list() -> None:
    cam = GeecsTriggeredDevice(
        "UC_TestCam",
        CAMERA_ROWS,
        experiment="Undulator",
        subscribed=["MeanCounts", "bogus"],
    )
    await cam.connect(mock=True)
    assert cam.selected == ("MeanCounts",)
    reading = await cam.read()
    # acq_timestamp always rides along on a triggered device
    assert set(reading) == {"UC_TestCam-acq_timestamp", "UC_TestCam-MeanCounts"}
    described = await cam.describe()
    assert set(described) == set(reading)


async def test_no_subscribed_list_reads_every_scalar() -> None:
    magnet = GeecsDevice("U_S1H", MAGNET_ROWS, experiment="Undulator")
    await magnet.connect(mock=True)
    assert magnet.selected == ("current", "voltage", "Position.Axis 1")
    reading = await magnet.read()
    assert set(reading) == {
        "U_S1H-current-position",  # the CaMotor child's readback signal
        "U_S1H-voltage",
        "U_S1H-position_axis_1",
    }


async def test_configure_selects_and_restores() -> None:
    cam = GeecsTriggeredDevice(
        "UC_TestCam", CAMERA_ROWS, experiment="Undulator", subscribed=["MeanCounts"]
    )
    await cam.connect(mock=True)
    assert isinstance(cam, Configurable) and isinstance(cam, Readable)
    assert isinstance(cam, Stageable)
    old, new = cam.configure(variables=["maxcounts", "exposure"])
    assert old["UC_TestCam-variables"]["value"] == "MeanCounts"
    assert new["UC_TestCam-variables"]["value"] == "MaxCounts,exposure"
    assert set(await cam.read()) == {
        "UC_TestCam-acq_timestamp",
        "UC_TestCam-MaxCounts",
        "UC_TestCam-exposure-readback",
    }
    conf = await cam.read_configuration()
    assert conf["UC_TestCam-variables"]["value"] == "MaxCounts,exposure"
    assert (await cam.describe_configuration())["UC_TestCam-variables"][
        "dtype"
    ] == "string"
    with pytest.raises(KeyError):
        cam.configure(variables=["image"])  # non-scalar: not selectable
    assert cam.selected == ("MaxCounts", "exposure")  # unchanged after the refusal
    cam.configure(variables=None)
    assert cam.selected == ("MeanCounts",)


async def test_stage_caches_only_the_selection() -> None:
    cam = GeecsTriggeredDevice(
        "UC_TestCam", CAMERA_ROWS, experiment="Undulator", subscribed=["MeanCounts"]
    )
    await cam.connect(mock=True)
    await cam.stage()
    assert cam.MeanCounts._get_cache()._staged
    assert cam.acq_timestamp._get_cache()._staged
    assert not cam.MaxCounts._get_cache()._staged
    await cam.unstage()
    assert not cam.MeanCounts._get_cache()._staged


async def test_triggered_device_trigger_completes_on_a_shot() -> None:
    cam = GeecsTriggeredDevice("UC_TestCam", CAMERA_ROWS, experiment="Undulator")
    cam._trigger_timeout = 1.0
    await cam.connect(mock=True)
    set_mock_value(cam.acq_timestamp, 100.0)
    await asyncio.sleep(0)
    status = cam.trigger()
    set_mock_value(cam.acq_timestamp, 101.0)
    await asyncio.wait_for(status, timeout=2.0)
    assert status.done
    await cam.disconnect()
    assert cam._monitoring is False and cam._last_acq is None


async def test_plain_device_has_no_trigger() -> None:
    magnet = GeecsDevice("U_S1H", MAGNET_ROWS, experiment="Undulator")
    assert not hasattr(magnet, "trigger")
    assert not hasattr(magnet, "acq_timestamp")
    await magnet.connect(mock=True)
    await magnet.disconnect()  # uniform teardown, nothing to release


def test_repr_is_informative() -> None:
    magnet = GeecsDevice("U_S1H", MAGNET_ROWS, experiment="Undulator")
    assert "GeecsDevice 'U_S1H' geecs='U_S1H' variables=3 selected=3" in repr(magnet)
