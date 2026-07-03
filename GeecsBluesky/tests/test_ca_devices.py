"""Hermetic tests for the CA-backed devices (ophyd-async mock backend).

No gateway, no lab network, no aioca traffic — the CA signals run on ophyd-async
mock backends.  The live end-to-end behavior (read/set/trigger against the real
gateway) is exercised separately; here we pin the PV-naming contract and the
device protocols (read / set-forwards-to-setpoint / acq_timestamp-gated trigger).
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("aioca")  # CA backend needs the `ca` extra

from ophyd_async.core import get_mock_put, set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import (  # noqa: E402
    CaGenericDetector,
    CaMotor,
    CaReadable,
    CaSettable,
    CaTriggerable,
)
from geecs_bluesky.exceptions import (  # noqa: E402
    GeecsMotorTimeoutError,
    GeecsTriggerTimeoutError,
)
from geecs_bluesky.pv_naming import normalize_component, pv_name  # noqa: E402


# --------------------------------------------------------------------------
# Naming contract (shared with the gateway via geecs_bluesky.pv_naming)
# --------------------------------------------------------------------------


def test_pv_name_policy() -> None:
    """Experiment prefix, dot-escaping, and space collapsing match the gateway."""
    assert pv_name("Undulator", "U_S1H", "Current") == "Undulator:U_S1H:Current"
    assert pv_name(None, "U_DG645", "Trigger.Source") == "U_DG645:Trigger_Source"
    assert normalize_component("Beam Current (A)") == "Beam_Current_A"


# --------------------------------------------------------------------------
# CaReadable
# --------------------------------------------------------------------------


async def test_readable_reads_value() -> None:
    """A readback PV value surfaces under the ``<name>-<safe_var>`` event key."""
    dev = CaReadable(
        "UC_Amp2_IR_input", "centroidx", experiment="Undulator", name="amp"
    )
    await dev.connect(mock=True)
    set_mock_value(dev.centroidx, 42.0)
    reading = await dev.read()
    assert reading["amp-centroidx"]["value"] == 42.0
    assert dev.centroidx.source.endswith("Undulator:UC_Amp2_IR_input:centroidx")


async def test_readable_multiple_variables() -> None:
    """Each variable becomes its own readable child signal."""
    dev = CaReadable("UC_X", ["centroidx", "centroidy"], name="cam")
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
    assert dev._setpoint.source.endswith("Undulator:U_S1H:Current:SP")
    assert dev.readback.source.endswith("Undulator:U_S1H:Current")


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
    assert motor._setpoint.source.endswith("Undulator:U_ESP_JetXYZ:Position_Axis_1:SP")
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
