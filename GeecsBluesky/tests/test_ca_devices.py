"""Hermetic tests for the CA-backed devices (ophyd-async mock backend).

No gateway, no lab network, no aioca traffic — the CA signals run on ophyd-async
mock backends.  The live end-to-end behavior (read/set/trigger against the real
gateway) is exercised separately; here we pin the PV-naming contract and the
device protocols (read / set-forwards-to-setpoint / acq_timestamp-gated trigger).
"""

from __future__ import annotations

import asyncio
import math

import pytest

pytest.importorskip("aioca")  # CA backend needs the `ca` extra

from ophyd_async.core import get_mock_put, set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import (  # noqa: E402
    CaConfirmSettable,
    CaGenericDetector,
    CaMotor,
    CaSettable,
    CaSnapshotReadable,
    CaTimestampedReadable,
    CaTriggerable,
)
from geecs_bluesky.devices.ca._pv import ca_pv  # noqa: E402
from geecs_bluesky.exceptions import (  # noqa: E402
    GeecsConfirmTimeoutError,
    GeecsMotorTimeoutError,
    GeecsTriggerTimeoutError,
)
from geecs_ca_gateway.pv_naming import normalize_component, pv_name  # noqa: E402


# --------------------------------------------------------------------------
# Naming contract (shared with the gateway via geecs_ca_gateway.pv_naming)
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
    con = CaTimestampedReadable(
        "UC_TopView",
        ["centroidx"],
        experiment="Undulator",
        name="con",
        save_nonscalar_data=True,
    )
    det = CaGenericDetector(
        "UC_Amp2_IR_input",
        ["centroidx"],
        experiment="Undulator",
        name="amp",
        save_nonscalar_data=True,
    )
    signals = [
        settable.readback,
        settable._setpoint,
        motor.position,
        motor._setpoint,
        snap.current,
        con.centroidx,
        con.acq_timestamp,
        con.localsavingpath,
        con.save,
        det.centroidx,
        det.acq_timestamp,
        det.localsavingpath,
        det.save,
    ]
    for device in (settable, motor, snap, con, det):
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
    det = CaGenericDetector(
        "UC_Amp2_IR_input", ["centroidx"], experiment="Undulator", name="amp"
    )
    await det.connect(mock=True)
    desc = await det.describe()
    reading = await det.read()
    assert set(desc) == {"amp-centroidx", "amp-acq_timestamp"}
    assert set(reading) == {"amp-centroidx", "amp-acq_timestamp"}
    assert desc["amp-centroidx"]["source"] == (
        "mock+ca://undulator:uc_amp2_ir_input:centroidx"
    )
    # Exporter column headers keep the legacy "Device Variable" form.
    assert det._column_headers == {"amp-centroidx": "UC_Amp2_IR_input centroidx"}


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


# --------------------------------------------------------------------------
# CaTriggerable
# --------------------------------------------------------------------------


async def test_trigger_completes_when_acq_timestamp_advances() -> None:
    """trigger() blocks until acq_timestamp changes, then completes."""
    dev = CaTriggerable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    await dev.connect(mock=True)
    set_mock_value(dev.acq_timestamp, 100.0)
    await asyncio.sleep(0)  # deliver the monitor update

    status = dev.trigger()
    await asyncio.sleep(0.05)
    assert not status.done

    set_mock_value(dev.acq_timestamp, 101.0)  # one shot
    await asyncio.wait_for(status, timeout=2.0)
    assert status.done


async def test_trigger_immediate_shot_not_missed() -> None:
    """The strict single-shot race: a shot fired IMMEDIATELY after trigger().

    trigger() must baseline synchronously before returning, so a shot landing
    before the returned coroutine first runs (zero awaits in between here) is
    still detected rather than being folded into the baseline and timing out.
    """
    dev = CaTriggerable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    dev._trigger_timeout = 1.0
    await dev.connect(mock=True)
    set_mock_value(dev.acq_timestamp, 100.0)
    await asyncio.sleep(0)  # deliver the monitor update (cache = 100.0)

    status = dev.trigger()
    set_mock_value(dev.acq_timestamp, 101.0)  # fire NOW — no await since trigger()
    await asyncio.wait_for(status, timeout=2.0)
    assert status.done


async def test_trigger_cold_cache_shot_before_coroutine_runs_not_lost() -> None:
    """Cold-cache race: the first-ever shot fires right after trigger().

    With no monitor update since subscribe (``_last_acq is None``), the old
    cold path took a CA-get baseline inside the coroutine and THEN drained the
    queue — discarding a shot that landed between trigger() returning and the
    coroutine running (and/or baselining on that shot's own timestamp).  A
    cold cache means the monitor has delivered no positive value yet, so any
    queued positive value IS the shot: trigger() must complete, not time out.
    """
    dev = CaTriggerable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    dev._trigger_timeout = 1.0
    await dev.connect(mock=True)
    assert dev._last_acq is None  # cold cache: no update since subscribe

    status = dev.trigger()
    set_mock_value(dev.acq_timestamp, 101.0)  # first shot, before the coroutine runs
    await asyncio.wait_for(status, timeout=2.0)
    assert status.done


async def test_trigger_cold_cache_shot_after_coroutine_starts() -> None:
    """Cold cache, no baseline get: a mid-wait first acquisition is the shot.

    The cold path deliberately takes no CA-get baseline (a get raced the shot
    itself: a first acquisition landing inside the get's round-trip became
    the baseline and the strict shot timed out — flagged in PR #452 review).
    With t0 = None the first positive monitor update completes the trigger,
    no matter when it arrives relative to the coroutine starting.
    """
    dev = CaTriggerable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    dev._trigger_timeout = 1.0
    await dev.connect(mock=True)
    assert dev._last_acq is None  # mock backend initial value is the 0.0 placeholder

    status = dev.trigger()
    await asyncio.sleep(0.05)  # coroutine is already waiting on the queue
    set_mock_value(dev.acq_timestamp, 101.0)  # first real acquisition
    await asyncio.wait_for(status, timeout=2.0)
    assert status.done


async def test_trigger_ignores_stale_updates_before_trigger() -> None:
    """Updates queued before trigger() are drained, not mistaken for a shot."""
    dev = CaTriggerable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    dev._trigger_timeout = 0.3
    await dev.connect(mock=True)
    # Several pushes before the trigger — all stale by trigger() time.
    for ts in (100.0, 101.0, 102.0):
        set_mock_value(dev.acq_timestamp, ts)
        await asyncio.sleep(0)

    status = dev.trigger()  # baseline = 102.0; stale queue drained
    with pytest.raises(GeecsTriggerTimeoutError):
        await status


async def test_trigger_times_out_when_no_shot() -> None:
    """No acq_timestamp advance within the timeout raises GeecsTriggerTimeoutError."""
    dev = CaTriggerable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    dev._trigger_timeout = 0.2
    await dev.connect(mock=True)
    set_mock_value(dev.acq_timestamp, 100.0)
    await asyncio.sleep(0)
    with pytest.raises(GeecsTriggerTimeoutError):
        await dev.trigger()


# --------------------------------------------------------------------------
# CaGenericDetector (schema-v1 companion columns over CA)
# --------------------------------------------------------------------------


async def _shot(det: CaGenericDetector, acq: float) -> dict:
    """Simulate one shot: advance acq_timestamp, await trigger, return read()."""
    status = det.trigger()
    set_mock_value(det.acq_timestamp, acq)
    await asyncio.wait_for(status, timeout=2.0)
    return await det.read()


async def test_generic_detector_companion_columns() -> None:
    """First shot self-seeds the tracker: shot_id=1, offset=0, valid=True."""
    det = CaGenericDetector(
        "UC_Amp2_IR_input", ["centroidx"], experiment="Undulator", name="amp"
    )
    await det.connect(mock=True)
    det.configure_shot_id(rep_rate_hz=1.0)
    set_mock_value(det.acq_timestamp, 100.0)
    await asyncio.sleep(0)

    desc = await det.describe()
    for key in (
        "amp-centroidx",
        "amp-acq_timestamp",
        "amp-t0_acq_timestamp",
        "amp-shot_id",
        "amp-shot_offset",
        "amp-valid",
    ):
        assert key in desc, key

    reading = await det.read()  # first read self-seeds at 100.0
    assert reading["amp-acq_timestamp"]["value"] == 100.0
    assert reading["amp-t0_acq_timestamp"]["value"] == 100.0
    assert reading["amp-shot_id"]["value"] == 1
    assert reading["amp-shot_offset"]["value"] == 0
    assert reading["amp-valid"]["value"] is True


async def test_generic_detector_shot_id_increments() -> None:
    """Shot IDs advance incrementally from acq_timestamp deltas (1 Hz)."""
    det = CaGenericDetector(
        "UC_Amp2_IR_input", ["centroidx"], experiment="Undulator", name="amp"
    )
    await det.connect(mock=True)
    det.configure_shot_id(rep_rate_hz=1.0)
    set_mock_value(det.acq_timestamp, 100.0)
    await asyncio.sleep(0)
    await det.read()  # seeds shot 1 at 100.0

    reading = await _shot(det, 101.0)  # +1 s at 1 Hz → shot 2
    assert reading["amp-shot_id"]["value"] == 2
    reading = await _shot(det, 104.0)  # 3 s dead time → shot 5 (gap preserved)
    assert reading["amp-shot_id"]["value"] == 5
    assert reading["amp-valid"]["value"] is True


async def test_generic_detector_without_shot_ids_is_plain() -> None:
    """Unconfigured tracker → no companion columns (stable minimal schema)."""
    det = CaGenericDetector(
        "UC_Amp2_IR_input", ["centroidx"], experiment="Undulator", name="amp"
    )
    await det.connect(mock=True)
    desc = await det.describe()
    assert "amp-shot_id" not in desc
    reading = await det.read()
    assert "amp-shot_id" not in reading


async def test_generic_detector_save_controls_over_ca() -> None:
    """save_nonscalar_data=True creates CA save controls + the save-path column.

    The controls read the gateway readback PV and write its :SP setpoint, so
    the shared run wrapper can bps.mv(det.localsavingpath, path, det.save, "on")
    exactly as with the direct backend.
    """
    det = CaGenericDetector(
        "UC_Amp2_IR_input",
        ["centroidx"],
        experiment="Undulator",
        name="amp",
        save_nonscalar_data=True,
    )
    await det.connect(mock=True)

    assert det.localsavingpath.source.endswith(
        "undulator:uc_amp2_ir_input:localsavingpath"
    )
    assert det.save.source.endswith("undulator:uc_amp2_ir_input:save")
    await det.save.set("on")
    put = get_mock_put(det.save)
    put.assert_called_once()
    assert put.call_args.args[0] == "on"

    det.configure_nonscalar_file_logging("/data/scans/Scan012/UC_Amp2_IR_input")
    desc = await det.describe()
    assert "amp-nonscalar_save_path" in desc
    reading = await det.read()
    assert (
        reading["amp-nonscalar_save_path"]["value"]
        == "/data/scans/Scan012/UC_Amp2_IR_input"
    )


async def test_generic_detector_no_save_controls_by_default() -> None:
    """Without save_nonscalar_data there are no save controls or save column."""
    det = CaGenericDetector(
        "UC_Amp2_IR_input", ["centroidx"], experiment="Undulator", name="amp"
    )
    await det.connect(mock=True)
    assert not hasattr(det, "localsavingpath")
    desc = await det.describe()
    assert "amp-nonscalar_save_path" not in desc


async def test_generic_detector_dedups_timestamp_variable() -> None:
    """acq_timestamp in variable_list maps onto the dedicated child, not a dup."""
    det = CaGenericDetector(
        "UC_Amp2_IR_input",
        ["centroidx", "acq_timestamp"],
        experiment="Undulator",
        name="amp",
    )
    await det.connect(mock=True)
    reading = await det.read()
    assert set(reading) == {"amp-centroidx", "amp-acq_timestamp"}
    # Exporter headers cover data variables only, not the timestamp column.
    assert det._column_headers == {"amp-centroidx": "UC_Amp2_IR_input centroidx"}


# --------------------------------------------------------------------------
# CaTimestampedReadable (free-run contributor) + CaSnapshotReadable
# --------------------------------------------------------------------------


async def _make_ref_and_contributor() -> tuple[
    CaGenericDetector, CaTimestampedReadable
]:
    """Reference + contributor pair, both seeded at shot 1 (acq=100.0)."""
    ref = CaGenericDetector(
        "UC_Amp2_IR_input", ["centroidx"], experiment="Undulator", name="ref"
    )
    con = CaTimestampedReadable(
        "UC_TopView", ["centroidx"], experiment="Undulator", name="con"
    )
    await ref.connect(mock=True)
    await con.connect(mock=True)
    ref.configure_shot_id(rep_rate_hz=1.0)
    con.configure_shot_id(rep_rate_hz=1.0)
    con.set_reference(ref, grace_wait_s=0)  # no grace wait in tests
    set_mock_value(ref.acq_timestamp, 100.0)
    set_mock_value(con.acq_timestamp, 100.0)
    await asyncio.sleep(0)
    ref.seed_shot_id(100.0)
    con.seed_shot_id(100.0)
    return ref, con


async def test_contributor_on_time_frame_is_valid() -> None:
    """Contributor frame for the row's shot → shot_offset=0, valid=True."""
    ref, con = await _make_ref_and_contributor()
    # Shot 2 on both devices; the reference accepted it (cache advanced).
    set_mock_value(ref.acq_timestamp, 101.0)
    set_mock_value(con.acq_timestamp, 101.0)
    await asyncio.sleep(0)

    reading = await con.read()
    assert reading["con-acq_timestamp"]["value"] == 101.0
    assert reading["con-shot_id"]["value"] == 2
    assert reading["con-shot_offset"]["value"] == 0
    assert reading["con-valid"]["value"] is True


async def test_contributor_late_frame_labeled_offset_minus_one() -> None:
    """A contributor still on the previous shot → shot_offset=-1, valid=False."""
    ref, con = await _make_ref_and_contributor()
    # Reference saw shot 2; the contributor's frame has not arrived.
    set_mock_value(ref.acq_timestamp, 101.0)
    await asyncio.sleep(0)

    reading = await con.read()
    assert reading["con-shot_id"]["value"] == 1  # still shot 1
    assert reading["con-shot_offset"]["value"] == -1
    assert reading["con-valid"]["value"] is False


async def test_contributor_without_reference_never_claims_valid() -> None:
    """No reference → shot_offset NaN, valid False (never false validity)."""
    con = CaTimestampedReadable(
        "UC_TopView", ["centroidx"], experiment="Undulator", name="con"
    )
    await con.connect(mock=True)
    con.configure_shot_id(rep_rate_hz=1.0)
    set_mock_value(con.acq_timestamp, 100.0)
    await asyncio.sleep(0)
    con.seed_shot_id(100.0)

    reading = await con.read()
    assert math.isnan(reading["con-shot_offset"]["value"])
    assert reading["con-valid"]["value"] is False


async def test_contributor_has_no_blocking_trigger() -> None:
    """Contributors must not gate event rows: no trigger() beyond Device's."""
    con = CaTimestampedReadable(
        "UC_TopView", ["centroidx"], experiment="Undulator", name="con"
    )
    assert not hasattr(con, "_wait_for_shot")


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


# --------------------------------------------------------------------------
# Persistent-monitor lifecycle: bounded idle queue + disconnect() teardown
# --------------------------------------------------------------------------


async def test_idle_monitor_updates_do_not_grow_queue_unboundedly() -> None:
    """An undrained device (contributor / idle detector) keeps a bounded queue.

    Only trigger() drains _shot_queue; a free-run contributor never calls it.
    Hundreds of monitor updates must therefore top out at the drop-oldest
    bound, with the cache still tracking the latest value.
    """
    con = CaTimestampedReadable(
        "UC_TopView", ["centroidx"], experiment="Undulator", name="con"
    )
    await con.connect(mock=True)
    for i in range(500):
        set_mock_value(con.acq_timestamp, 100.0 + i)
    await asyncio.sleep(0)  # deliver any pending monitor callbacks

    assert con._shot_queue.qsize() <= con._shot_queue_maxsize
    assert con._last_acq == 599.0  # newest survives the drop-oldest ring


async def test_trigger_completes_after_queue_saturation() -> None:
    """A saturated idle queue must not break the trigger path.

    After far more idle updates than the queue bound, trigger() still drains,
    baselines, and detects the next new timestamp (no regression from the
    drop-oldest ring).
    """
    dev = CaTriggerable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    await dev.connect(mock=True)
    for i in range(200):  # saturate the bounded queue while idle
        set_mock_value(dev.acq_timestamp, 100.0 + i)
    await asyncio.sleep(0)

    status = dev.trigger()
    set_mock_value(dev.acq_timestamp, 1000.0)  # fire immediately (no-blind-window)
    await asyncio.wait_for(status, timeout=2.0)
    assert status.done


async def test_disconnect_unsubscribes_monitor_and_clears_state() -> None:
    """disconnect() stops the monitor and drops cached shot state.

    After disconnect, further acq_timestamp updates must produce no callback
    activity (empty queue, cache back to the never-acquired None), and calling
    it again must be a harmless no-op (idempotent).
    """
    dev = CaTriggerable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    await dev.connect(mock=True)
    set_mock_value(dev.acq_timestamp, 100.0)
    await asyncio.sleep(0)
    assert dev._last_acq == 100.0
    assert dev._shot_queue.qsize() == 1

    await dev.disconnect()
    assert dev._monitoring is False
    assert dev._shot_queue.empty()
    assert dev._last_acq is None

    set_mock_value(dev.acq_timestamp, 101.0)  # post-teardown update
    await asyncio.sleep(0)
    assert dev._shot_queue.empty()  # no further callback activity
    assert dev._last_acq is None

    await dev.disconnect()  # idempotent


async def test_reconnect_after_disconnect_resubscribes() -> None:
    """connect() after disconnect() restores the monitor (per-scan reuse)."""
    dev = CaTriggerable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    await dev.connect(mock=True)
    await dev.disconnect()

    await dev.connect(mock=True)
    set_mock_value(dev.acq_timestamp, 200.0)
    await asyncio.sleep(0)
    assert dev._last_acq == 200.0


async def test_all_ca_devices_define_disconnect() -> None:
    """Every CA device type honours the per-scan teardown contract.

    The runner's cleanup (``session.disconnect``) runs device.disconnect()
    on the RE loop for every device it created; each class must expose it
    as a coroutine that does not raise (previously an AttributeError was
    swallowed by the caller).
    """
    devices = [
        CaSnapshotReadable("U_S1H", "Current", name="snap"),
        CaSettable("U_S1H", "Current", name="cur"),
        CaMotor("U_ESP_JetXYZ", "Position.Axis 1", name="jet"),
        CaTimestampedReadable("UC_TopView", ["centroidx"], name="con"),
        CaGenericDetector("UC_Amp2_IR_input", ["centroidx"], name="det"),
    ]
    for dev in devices:
        await dev.connect(mock=True)
        assert asyncio.iscoroutinefunction(dev.disconnect), type(dev).__name__
        await dev.disconnect()  # must not raise


def test_saving_device_surfaces_acq_timestamp_in_sfile_headers() -> None:
    """A file/image-saving device adds acq_timestamp as an s-file column.

    Saved image filenames are stamped with acq_timestamp, so surfacing it as a
    legacy scalar column lets saved files tie back to scan rows — an
    images-only device otherwise contributes no scalar column at all. A
    pure-scalar device keeps acq_timestamp as an excluded companion column.
    """
    saving = CaGenericDetector(
        "UC_Amp2_IR_input",
        ["centroidx"],
        experiment="Undulator",
        name="amp",
        save_nonscalar_data=True,
    )
    assert saving._column_headers["amp-acq_timestamp"] == (
        "UC_Amp2_IR_input acq_timestamp"
    )
    assert saving._column_headers["amp-centroidx"] == "UC_Amp2_IR_input centroidx"

    scalar_only = CaGenericDetector(
        "UC_Amp2_IR_input",
        ["centroidx"],
        experiment="Undulator",
        name="amp2",
        save_nonscalar_data=False,
    )
    assert "amp2-acq_timestamp" not in scalar_only._column_headers

    # An images-only save (no other scalars) still yields the acq_timestamp
    # column — the one column that ties the saved frames to the rows.
    images_only = CaGenericDetector(
        "UC_Amp2_IR_input",
        [],
        experiment="Undulator",
        name="cam",
        save_nonscalar_data=True,
    )
    assert images_only._column_headers == {
        "cam-acq_timestamp": "UC_Amp2_IR_input acq_timestamp"
    }

    # Same for the free-run contributor.
    contributor = CaTimestampedReadable(
        "UC_TopView",
        ["centroidx"],
        experiment="Undulator",
        name="con",
        save_nonscalar_data=True,
    )
    assert contributor._column_headers["con-acq_timestamp"] == (
        "UC_TopView acq_timestamp"
    )
