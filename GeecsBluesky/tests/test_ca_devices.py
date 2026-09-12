"""Hermetic tests for the CA-backed devices (ophyd-async mock backend).

No gateway, no lab network, no aioca traffic — the CA signals run on ophyd-async
mock backends.  The live end-to-end behavior (read/set/trigger against the real
gateway) is exercised separately; here we pin the PV-naming contract and the
scalar-device protocols (read / set-forwards-to-setpoint / converge).  The
acquirer (``GeecsDetector``) has its own file, ``test_geecs_detector.py``.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("aioca")  # CA backend needs the `ca` extra

from ophyd_async.core import get_mock_put, set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import (  # noqa: E402
    CaConfirmSettable,
    CaMotor,
    CaSettable,
    CaSnapshotReadable,
)
from geecs_bluesky.devices.ca._pv import ca_pv  # noqa: E402
from geecs_bluesky.devices.detector import (  # noqa: E402
    STRICT_TRIGGER_INFO,
    GeecsDetector,
)
from geecs_bluesky.exceptions import (  # noqa: E402
    GeecsConfirmTimeoutError,
    GeecsMotorTimeoutError,
)
from geecs_core.pv_naming import normalize_component, pv_name  # noqa: E402


# --------------------------------------------------------------------------
# Naming contract (shared with the gateway via geecs_core.pv_naming)
# --------------------------------------------------------------------------


def test_pv_name_policy() -> None:
    """Experiment prefix, dot-escaping, and space collapsing match the gateway."""
    assert pv_name("Undulator", "U_S1H", "Current") == "undulator:u_s1h:current"
    assert pv_name(None, "U_DG645", "Trigger.Source") == "u_dg645:trigger_source"
    assert normalize_component("Beam Current (A)") == "beam_current_a"


def test_ca_pv_pins_the_transport_on_gateway_names() -> None:
    """ca_pv is pv_name with the explicit CA transport scheme prepended."""
    assert ca_pv("Undulator", "U_S1H", "Current") == "ca://undulator:u_s1h:current"
    assert ca_pv(None, "U_DG645", "Trigger.Source") == "ca://u_dg645:trigger_source"


# --------------------------------------------------------------------------
# CA transport pinning (ca:// prefix on every device PV)
# --------------------------------------------------------------------------
#
# ophyd-async picks the default EPICS transport for UN-prefixed PV names by
# import luck (p4p installed + aioca missing flips signals to PVA, and every
# connect against the CA-only gateway times out with a generic error).  Every
# signal a CA device builds must therefore carry an explicit ca:// source.
# The prefix is stripped before the backend stores the PV, so it must appear
# exactly once in the re-derived source string — never doubled, never leaked
# into event keys.


async def test_every_ca_device_signal_pins_the_ca_transport() -> None:
    """All signals on every CA device class carry ca://-prefixed sources."""
    settable = CaSettable("U_S1H", "Current", experiment="Undulator", name="cur")
    motor = CaMotor("U_ESP_JetXYZ", "Position.Axis 1", experiment="Undulator")
    snap = CaSnapshotReadable("U_S1H", "Current", experiment="Undulator", name="s1h")
    det = GeecsDetector(
        "UC_Amp2_IR_input",
        ["centroidx"],
        experiment="Undulator",
        name="amp",
        native_save=True,
    )
    signals = [
        settable.readback,
        settable._setpoint,
        motor.position,
        motor._setpoint,
        snap.current,
        det.centroidx,
        det.acq_timestamp,
        det.connected_status,
        det.localsavingpath,
        det.save,
    ]
    for device in (settable, motor, snap, det):
        await device.connect(mock=True)
    for signal in signals:
        # Mock backends wrap the CA backend: "mock+ca://<pv>".  The prefix
        # appears exactly once and never doubles into the PV portion.
        assert "ca://" in signal.source, signal.source
        assert signal.source.count("ca://") == 1, signal.source
    # Write PVs (":SP") pin the transport too, not just readbacks.
    assert settable._setpoint.source.endswith("ca://undulator:u_s1h:current:SP")


async def test_ca_prefix_does_not_leak_into_event_keys_or_describe() -> None:
    """describe()/read() keys stay `<name>-<var>`; sources are single-ca://."""
    det = GeecsDetector(
        "UC_Amp2_IR_input", ["centroidx"], experiment="Undulator", name="amp"
    )
    await det.connect(mock=True)
    await det.prepare(STRICT_TRIGGER_INFO)  # a StandardDetector describes once prepared
    desc = await det.describe()
    reading = await det.read()
    assert set(desc) == {"amp-centroidx", "amp-acq_timestamp"}
    assert set(reading) == {"amp-centroidx", "amp-acq_timestamp"}
    assert desc["amp-centroidx"]["source"] == (
        "mock+ca://undulator:uc_amp2_ir_input:centroidx"
    )
    # Exporter column headers keep the legacy "Device Variable" form.
    assert det._column_headers == {
        "amp-centroidx": "UC_Amp2_IR_input centroidx",
        "amp-acq_timestamp": "UC_Amp2_IR_input acq_timestamp",
    }


# --------------------------------------------------------------------------
# CaSnapshotReadable (plain readable)
# --------------------------------------------------------------------------


async def test_readable_reads_value() -> None:
    """A readback PV value surfaces under the ``<name>-<safe_var>`` event key."""
    dev = CaSnapshotReadable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    await dev.connect(mock=True)
    set_mock_value(dev.centroidx, 42.0)
    reading = await dev.read()
    assert reading["amp-centroidx"]["value"] == 42.0
    assert dev.centroidx.source.endswith("undulator:uc_amp2_ir_input:centroidx")


async def test_readable_multiple_variables() -> None:
    """Each variable becomes its own readable child signal."""
    dev = CaSnapshotReadable("UC_X", ["centroidx", "centroidy"], name="cam")
    await dev.connect(mock=True)
    set_mock_value(dev.centroidx, 1.0)
    set_mock_value(dev.centroidy, 2.0)
    reading = await dev.read()
    assert reading["cam-centroidx"]["value"] == 1.0
    assert reading["cam-centroidy"]["value"] == 2.0


# --------------------------------------------------------------------------
# CaSettable
# --------------------------------------------------------------------------


async def test_settable_forwards_put_to_setpoint() -> None:
    """set() puts to the ``…:SP`` PV; readback is a separate PV."""
    dev = CaSettable("U_S1H", "Current", experiment="Undulator", name="cur")
    await dev.connect(mock=True)
    await dev.set(0.5)
    put = get_mock_put(dev._setpoint)
    put.assert_called_once()
    assert put.call_args.args[0] == 0.5
    assert dev._setpoint.source.endswith("undulator:u_s1h:current:SP")
    assert dev.readback.source.endswith("undulator:u_s1h:current")


async def test_settable_readback_is_the_reading() -> None:
    """read() reflects the readback PV, not the setpoint echo."""
    dev = CaSettable("U_S1H", "Current", experiment="Undulator", name="cur")
    await dev.connect(mock=True)
    set_mock_value(dev.readback, 0.4997)
    reading = await dev.read()
    assert reading["cur-readback"]["value"] == pytest.approx(0.4997)


# --------------------------------------------------------------------------
# CaMotor
# --------------------------------------------------------------------------


async def test_motor_set_completes_on_arrival() -> None:
    """set() puts the setpoint and resolves once the readback is in tolerance."""
    motor = CaMotor(
        "U_ESP_JetXYZ", "Position.Axis 1", experiment="Undulator", name="jet"
    )
    await motor.connect(mock=True)
    set_mock_value(motor.position, 4.5)  # streamed readback already at target
    await asyncio.wait_for(motor.set(4.5), timeout=2.0)
    put = get_mock_put(motor._setpoint)
    put.assert_called_once()
    assert put.call_args.args[0] == 4.5
    assert motor._setpoint.source.endswith("undulator:u_esp_jetxyz:position_axis_1:SP")
    reading = await motor.read()
    assert reading["jet-position"]["value"] == 4.5


async def test_motor_set_times_out_when_stuck() -> None:
    """Readback never converging raises GeecsMotorTimeoutError."""
    motor = CaMotor(
        "U_ESP_JetXYZ",
        "Position.Axis 1",
        experiment="Undulator",
        name="jet",
        move_timeout=0.3,
    )
    await motor.connect(mock=True)
    set_mock_value(motor.position, 0.0)  # stuck far from target
    with pytest.raises(GeecsMotorTimeoutError):
        await motor.set(4.5)


# --------------------------------------------------------------------------
# CaConfirmSettable — topology-C: set X, confirm on Y
# --------------------------------------------------------------------------


def _emq_confirm_device(**overrides) -> CaConfirmSettable:
    kwargs = dict(
        device="U_EMQTripletBipolar",
        variable="Current_Limit.Ch1",
        confirm_device="U_EMQTripletBipolar",
        confirm_variable="Current.Ch1",
        experiment="Undulator",
        name="emq1",
        timeout=0.3,
    )
    kwargs.update(overrides)
    return CaConfirmSettable(**kwargs)


async def test_confirm_writes_target_variable_reads_confirm_variable() -> None:
    """set() puts the SETPOINT variable but polls the CONFIRM variable's PV."""
    device = _emq_confirm_device()
    await device.connect(mock=True)
    assert device._setpoint.source.endswith(
        "undulator:u_emqtripletbipolar:current_limit_ch1:SP"
    )
    assert device._confirm_readback.source.endswith(
        "undulator:u_emqtripletbipolar:current_ch1"
    )


async def test_confirm_completes_when_confirm_variable_within_tolerance() -> None:
    """set() resolves once the CONFIRM readback (not the target var) matches."""
    device = _emq_confirm_device(tolerance=0.05)
    await device.connect(mock=True)
    set_mock_value(device._confirm_readback, 2.51)  # within 0.05 of target
    await asyncio.wait_for(device.set(2.5), timeout=1.0)
    put = get_mock_put(device._setpoint)
    put.assert_called_once()
    assert put.call_args.args[0] == 2.5


async def test_confirm_times_out_when_confirm_variable_never_matches() -> None:
    """The setpoint variable converging is not enough — confirm must too."""
    device = _emq_confirm_device(tolerance=0.05)
    await device.connect(mock=True)
    set_mock_value(device._confirm_readback, 0.0)  # never converges
    with pytest.raises(GeecsConfirmTimeoutError) as excinfo:
        await device.set(2.5)
    assert excinfo.value.confirm_variable == "U_EMQTripletBipolar:Current.Ch1"


async def test_confirm_discrete_match_is_exact_equality() -> None:
    """A string/enum confirm target (e.g. a future shutter) matches exactly."""
    device = CaConfirmSettable(
        "U_Shutter",
        "Command",
        confirm_device="U_Shutter",
        confirm_variable="LimitSwitch",
        experiment="Undulator",
        name="shutter",
        timeout=0.3,
        datatype=str,
    )
    await device.connect(mock=True)
    set_mock_value(device._confirm_readback, "inserted")
    await asyncio.wait_for(device.set("inserted"), timeout=1.0)


async def test_confirm_discrete_match_rejects_numeric_looking_near_miss() -> None:
    """A str confirm target does not tolerance-match numeric-looking labels.

    Review finding (PR #477): the old ``_matches`` tried ``float()`` on both
    sides before falling back to equality, so a ``datatype=str`` confirm
    target could accept "1.04" as matching "1.0" under the default
    tolerance — silently reintroducing analog matching for a discrete
    variable. Dispatch must be on the declared ``datatype``, not on whether
    the strings happen to be parseable as numbers.
    """
    device = CaConfirmSettable(
        "U_Shutter",
        "Command",
        confirm_device="U_Shutter",
        confirm_variable="LimitSwitch",
        experiment="Undulator",
        name="shutter",
        timeout=0.3,
        datatype=str,
    )
    await device.connect(mock=True)
    set_mock_value(device._confirm_readback, "1.04")
    with pytest.raises(GeecsConfirmTimeoutError):
        await device.set("1.0")


async def test_snapshot_reads_latest_values() -> None:
    """Snapshot: plain per-row sampling, no companion columns."""
    snap = CaSnapshotReadable(
        "U_S1H", ["Current", "Voltage"], experiment="Undulator", name="s1h"
    )
    await snap.connect(mock=True)
    set_mock_value(snap.current, 0.5)
    reading = await snap.read()
    assert reading["s1h-current"]["value"] == 0.5
    assert set(reading) == {"s1h-current", "s1h-voltage"}
    assert snap.current.source.endswith("undulator:u_s1h:current")


def test_ca_motor_locates_by_its_readback_so_relative_plans_work() -> None:
    """``rel_scan`` stashes ``locate()`` (the streamed readback), moves about it, restores it.

    Found on hardware (2b broader set, Scan017): without ``locate`` bluesky
    fell back to ``obj.position`` — the readback signal — and every
    ``rel_*`` plan failed at its first move.
    """
    import bluesky.plans as bp
    from bluesky import RunEngine

    from tests.ca_mock_helpers import connect_mock, follow_setpoint

    RE = RunEngine()
    motor = CaMotor("U_Stage", "Position.Axis1", experiment="TestExp", name="u_stage")
    connect_mock(RE, motor)
    set_mock_value(motor.position, 41342.0)  # where the device is
    set_mock_value(motor._setpoint, 0.0)  # the gateway's :SP: never put through it
    follow_setpoint(motor)

    async def locate():
        return await motor.locate()

    loc = asyncio.run_coroutine_threadsafe(locate(), RE._loop).result(timeout=5)
    assert loc == {"setpoint": 41342.0, "readback": 41342.0}
    RE(bp.rel_scan([], motor, -20, 20, 5))
    puts = [c.args[0] for c in get_mock_put(motor._setpoint).call_args_list]
    assert puts == [41322.0, 41332.0, 41342.0, 41352.0, 41362.0, 41342.0]
