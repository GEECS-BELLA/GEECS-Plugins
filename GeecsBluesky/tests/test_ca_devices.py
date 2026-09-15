"""Hermetic tests for the CA-backed devices (ophyd-async mock backend).

No gateway, no lab network, no aioca traffic — the CA signals run on ophyd-async
mock backends.  The live end-to-end behavior (read/set/trigger against the real
gateway) is exercised separately; here we pin the PV-naming contract and the
scalar-device protocols (read / set-forwards-to-setpoint / converge).  The
acquirer (``GeecsDetector``) has its own file, ``test_geecs_detector.py``.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

pytest.importorskip("aioca")  # CA backend needs the `ca` extra

from ophyd_async.core import (  # noqa: E402
    callback_on_mock_put,
    get_mock_put,
    set_mock_put_proceeds,
    set_mock_value,
)

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


async def test_motor_arrival_exactly_on_tolerance_counts_as_arrived() -> None:
    """A readback exactly one tolerance from target resolves, not times out.

    Regression for U_ModeImagerESP Scan034: the stage reached -10.505 against
    a -10.5 target with tolerance 0.005, but ``abs(-10.505 - -10.5)`` is
    0.005000000000000782 in binary floating point, so the plain ``<=``
    comparison polled the full move_timeout and paused the scan.
    """
    motor = CaMotor(
        "U_ModeImagerESP",
        "Position.Axis 1",
        experiment="Undulator",
        name="mode",
        tolerance=0.005,
        progress_grace=0.1,
        stall_timeout=0.2,
    )
    await motor.connect(mock=True)
    set_mock_value(motor.position, -10.505)
    assert abs(-10.505 - -10.5) > 0.005  # the representation error is real
    await asyncio.wait_for(motor.set(-10.5), timeout=2.0)


async def test_motor_beyond_tolerance_still_times_out() -> None:
    """The epsilon is representation slack, not a widened tolerance."""
    motor = CaMotor(
        "U_ModeImagerESP",
        "Position.Axis 1",
        experiment="Undulator",
        name="mode",
        tolerance=0.005,
        progress_grace=0.1,
        stall_timeout=0.2,
    )
    await motor.connect(mock=True)
    set_mock_value(motor.position, -10.51)  # 0.01 out — twice the tolerance
    with pytest.raises(GeecsMotorTimeoutError):
        await motor.set(-10.5)


async def test_motor_set_times_out_when_stuck() -> None:
    """A ``no error`` reply whose readback never converges: the stall rule fires.

    The mock put completes at once (the reply); the readback then sits out
    of tolerance, so the confirm stalls and names the phase.
    """
    motor = CaMotor(
        "U_ESP_JetXYZ",
        "Position.Axis 1",
        experiment="Undulator",
        name="jet",
        progress_grace=0.1,
        stall_timeout=0.2,
    )
    await motor.connect(mock=True)
    set_mock_value(motor.position, 0.0)  # stuck far from target
    with pytest.raises(GeecsMotorTimeoutError) as info:
        await motor.set(4.5)
    assert info.value.replied is True
    assert "after the device replied" in str(info.value)


# --------------------------------------------------------------------------
# The device's reply is the verdict; the readback stall rule is the only
# client-side timeout (GEECS-Plugins#906)
# --------------------------------------------------------------------------
#
# Timings run at 1/10 scale: PROGRESS_GRACE 5 s -> 0.5 s, STALL_TIMEOUT
# 10 s -> 1.0 s, REPLY_WAIT 90 s -> 9 s, so a reply "at 45 s" arrives at
# 4.5 s and the old 30 s cap would sit at 3.0 s.  The reply ceiling keeps
# its real default unless a test is about it.

_SCALE = 0.1
_GRACE = 5.0 * _SCALE
_STALL = 10.0 * _SCALE
_REPLY_WAIT = 90.0 * _SCALE
_OLD_CAP = 30.0 * _SCALE
_PV = "undulator:u_modeimageresp:position_axis_1:SP"


def _slow_stage(**overrides) -> CaMotor:
    kwargs = dict(
        device="U_ModeImagerESP",
        variable="Position.Axis 1",
        experiment="Undulator",
        name="mode",
        tolerance=0.005,
        progress_grace=_GRACE,
        stall_timeout=_STALL,
        reply_wait=_REPLY_WAIT,
    )
    kwargs.update(overrides)
    return CaMotor(**kwargs)


async def _creep(motor: CaMotor, start: float, target: float, duration: float) -> None:
    """Advance the mock readback from *start* to *target* over *duration* s.

    One step per 0.1 s; every step exceeds the tolerance, so each is progress.
    """
    steps = max(int(duration / 0.1), 1)
    for i in range(1, steps + 1):
        await asyncio.sleep(duration / steps)
        set_mock_value(motor.position, start + (target - start) * i / steps)


def _warning_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_reply_wait_defaults_are_named_bounds_not_a_cap() -> None:
    """The constants: 90 s reply wait, 5 s grace, 10 s stall, 300 s ceiling."""
    from geecs_bluesky.devices.ca import motor as motor_module

    assert motor_module.REPLY_WAIT == 90.0
    assert motor_module.PROGRESS_GRACE == 5.0
    assert motor_module.STALL_TIMEOUT == 10.0
    assert motor_module.REPLY_CEILING == 300.0
    assert not hasattr(motor_module, "_DEFAULT_MOVE_TIMEOUT")


async def test_motor_waits_past_the_old_cap_for_a_late_no_error_reply(caplog) -> None:
    """(a) A ``no error`` reply at "45 s" with the readback advancing completes.

    Scan009 of 26_0914: a 19 mm move the device completed at 32 s, killed by
    a 30 s client cap.  Here the readback creeps -5 → -24 the whole time and
    the reply lands at 4.5 s scaled — 1.5× the old cap, inside the reply
    wait: the reply path, no lost-reply warning.
    """
    motor = _slow_stage()
    await motor.connect(mock=True)
    set_mock_value(motor.position, -5.0)
    set_mock_put_proceeds(motor._setpoint, False)  # the reply is pending

    async def device() -> None:
        await _creep(motor, -5.0, -24.0, 4.5)
        set_mock_put_proceeds(motor._setpoint, True)  # ">>no error" at 45 s

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(device())
    t0 = loop.time()
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.ca"):
        await asyncio.wait_for(motor.set(-24.0), timeout=8.0)
    elapsed = loop.time() - t0
    await task
    assert _OLD_CAP < 4.5 <= elapsed < _REPLY_WAIT
    assert await motor.position.get_value() == pytest.approx(-24.0)
    assert _warning_lines(caplog) == []


async def test_motor_never_answered_fails_at_grace_plus_stall_naming_the_pv(
    caplog,
) -> None:
    """(b) No reply, readback stalled: GeecsMotorTimeoutError at ~grace+stall.

    The one client-side timeout left.  The device's ERROR line names the
    ``:SP`` PV; the exception names the device, the target and the current.
    """
    motor = _slow_stage()
    await motor.connect(mock=True)
    set_mock_value(motor.position, -5.0)
    set_mock_put_proceeds(motor._setpoint, False)  # never answers

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.devices.ca"):
        with pytest.raises(GeecsMotorTimeoutError) as info:
            await asyncio.wait_for(motor.set(-24.0), timeout=5.0)
    elapsed = loop.time() - t0
    assert _GRACE + _STALL <= elapsed < _GRACE + _STALL + 0.6
    exc = info.value
    assert (exc.device_name, exc.variable) == ("U_ModeImagerESP", "Position.Axis 1")
    assert (exc.target, exc.current, exc.replied) == (-24.0, -5.0, False)
    assert "with no reply from the device" in str(exc)
    (line,) = _error_lines(caplog)
    assert f"({_PV})" in line
    assert "GeecsMotorTimeoutError: U_ModeImagerESP/Position.Axis 1" in line
    set_mock_put_proceeds(motor._setpoint, True)  # release the cancelled put


async def test_motor_error_reply_fails_at_once_even_while_advancing() -> None:
    """(c) An error reply is the verdict: the move fails the moment it lands.

    The readback is advancing (progress keeps the stall rule quiet), the
    device answers with its check-values error — a device-side timeout,
    adjusted in LabVIEW, never overridden here — and the put's failure
    propagates untouched, before the grace has even elapsed.
    """
    motor = _slow_stage()
    await motor.connect(mock=True)
    set_mock_value(motor.position, -5.0)
    refusal = _RefusedPut(
        f"{_PV}: Error occurred during check values - Position.Axis 1 was not "
        "change acording to the command. actual value= -6.485000"
    )

    async def error_reply(value, **kwargs):
        await _creep(motor, -5.0, -6.5, 0.3)  # the stage is moving ...
        raise refusal  # ... and then the device's check-values error lands

    callback_on_mock_put(motor._setpoint, error_reply)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with pytest.raises(_RefusedPut) as info:
        await asyncio.wait_for(motor.set(-24.0), timeout=3.0)
    elapsed = loop.time() - t0
    assert info.value is refusal
    assert 0.3 <= elapsed < _GRACE  # at the reply, not the grace nor the stall


async def test_motor_rejected_command_fails_at_once_not_after_grace_plus_stall() -> (
    None
):
    """A put that fails inside the ACK window is a hard failure at once.

    The GEECS set's first reply is the command ACK (GEECS-Core's 1.5 s
    window): no ACK, a rejection (``is not a number``, an unknown
    variable) or a dead device's write failure fails the gateway put inside
    it.  The stall grace must neither swallow nor delay it — only a put
    still pending past the ACK window is "waiting for the device".
    """
    motor = _slow_stage()
    await motor.connect(mock=True)
    set_mock_value(motor.position, -5.0)  # and it never moves
    rejection = _RefusedPut(f"{_PV}: Channel write request failed")

    def reject(value, **kwargs):
        raise rejection

    callback_on_mock_put(motor._setpoint, reject)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with pytest.raises(_RefusedPut) as info:
        await asyncio.wait_for(motor.set(-24.0), timeout=3.0)
    elapsed = loop.time() - t0
    assert info.value is rejection
    assert not isinstance(info.value, GeecsMotorTimeoutError)
    assert elapsed < 0.3  # one poll tick, not grace (0.5) + stall (1.0)


async def test_motor_keeps_waiting_past_reply_wait_while_advancing(caplog) -> None:
    """(d) Slow but continuous progress with no reply is not a failure.

    Every step exceeds the tolerance, so every stall window sees progress;
    the wait crosses ``reply_wait`` (1 s here) with the stage still short
    of the target — keep waiting for the reply — and completes when the
    reply lands at 4 s, past the old cap: the reply path, no warning.
    """
    motor = _slow_stage(reply_wait=1.0)
    await motor.connect(mock=True)
    set_mock_value(motor.position, -5.0)
    set_mock_put_proceeds(motor._setpoint, False)

    async def device() -> None:
        # The natural order: the stream shows the arrival, the reply lands a
        # loop turn later — past reply_wait that must still be the reply
        # path (a reply landing at the threshold wins), not a lost reply.
        await _creep(motor, -5.0, -24.0, 4.0)
        set_mock_put_proceeds(motor._setpoint, True)

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(device())
    t0 = loop.time()
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.ca"):
        await asyncio.wait_for(motor.set(-24.0), timeout=7.0)
    elapsed = loop.time() - t0
    await task
    assert elapsed >= 4.0 > _OLD_CAP
    assert _warning_lines(caplog) == []


async def test_motor_lost_reply_completes_at_reply_wait_when_at_target(caplog) -> None:
    """(e) No reply ever, readback at the target before ``reply_wait``: complete.

    The stage arrives at 0.8 s and sits there — never a stall — and the
    move completes at ``reply_wait`` (2 s here) with the WARNING naming
    the PV.
    """
    motor = _slow_stage(reply_wait=2.0)
    await motor.connect(mock=True)
    set_mock_value(motor.position, -5.0)
    set_mock_put_proceeds(motor._setpoint, False)  # the reply is lost

    loop = asyncio.get_running_loop()
    creep = asyncio.ensure_future(_creep(motor, -5.0, -24.0, 0.8))
    t0 = loop.time()
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.ca"):
        await asyncio.wait_for(motor.set(-24.0), timeout=5.0)
    elapsed = loop.time() - t0
    await creep
    assert 2.0 <= elapsed < 2.0 + 0.4
    (line,) = _warning_lines(caplog)
    assert "no reply from the device within 2 s" in line
    assert "treating the move as complete" in line
    assert f"({_PV})" in line
    set_mock_put_proceeds(motor._setpoint, True)  # release the cancelled put


async def test_motor_ceiling_fails_the_put_while_still_moving() -> None:
    """(f) Past ``reply_wait``, still moving, no reply: the ceiling ends it.

    The put's own timeout names the PV; the readback (short of the target
    at the ceiling) never completed the move.
    """
    motor = _slow_stage(reply_wait=0.5, reply_ceiling=1.5)
    await motor.connect(mock=True)
    set_mock_value(motor.position, -5.0)
    set_mock_put_proceeds(motor._setpoint, False)

    loop = asyncio.get_running_loop()
    creep = asyncio.ensure_future(_creep(motor, -5.0, -24.0, 3.0))
    t0 = loop.time()
    with pytest.raises(TimeoutError, match="position_axis_1:SP.*1.5 s"):
        await asyncio.wait_for(motor.set(-24.0), timeout=4.0)
    elapsed = loop.time() - t0
    creep.cancel()
    assert 1.5 <= elapsed < 1.5 + 0.4
    set_mock_put_proceeds(motor._setpoint, True)


async def test_motor_nan_readback_after_the_reply_is_bounded() -> None:
    """A readback the stall rule cannot see (NaN) fails after the confirm budget.

    The device replied ``no error``; the stream then reads NaN — never
    within tolerance, and ``NaN != anchor`` must not count as progress.
    Without the post-reply bound the loop waited forever (review of #909).
    """
    motor = _slow_stage()
    await motor.connect(mock=True)
    set_mock_value(motor.position, float("nan"))

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with pytest.raises(GeecsMotorTimeoutError) as info:
        await asyncio.wait_for(motor.set(-24.0), timeout=5.0)
    elapsed = loop.time() - t0
    assert _GRACE + _STALL <= elapsed < _GRACE + _STALL + 0.6
    assert info.value.replied is True


async def test_motor_ripple_wider_than_tolerance_after_the_reply_is_bounded() -> None:
    """A readback flapping by more than the tolerance is not endless progress.

    The device replied ``no error``; the readback then flips between two
    values 4× the tolerance apart, never at the target.  The post-reply
    confirm budget (grace + stall from the reply) ends it.
    """
    motor = _slow_stage()
    await motor.connect(mock=True)
    set_mock_value(motor.position, 0.0)

    async def ripple() -> None:
        i = 0
        while True:
            await asyncio.sleep(0.13)
            i += 1
            set_mock_value(motor.position, 0.02 * (i % 2))

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(ripple())
    t0 = loop.time()
    try:
        with pytest.raises(GeecsMotorTimeoutError) as info:
            await asyncio.wait_for(motor.set(-24.0), timeout=5.0)
    finally:
        task.cancel()
    elapsed = loop.time() - t0
    assert _GRACE + _STALL <= elapsed < _GRACE + _STALL + 0.6
    assert info.value.replied is True


async def test_motor_stall_after_progress_still_fails() -> None:
    """Progress resets the stall clock; a stage that then stops is caught.

    Creep for 0.8 s (past the grace), then stop short with no reply: the
    failure lands ~one stall window after the last step, not grace+stall
    from t0.
    """
    motor = _slow_stage()
    await motor.connect(mock=True)
    set_mock_value(motor.position, -5.0)
    set_mock_put_proceeds(motor._setpoint, False)

    loop = asyncio.get_running_loop()
    creep = asyncio.ensure_future(_creep(motor, -5.0, -8.0, 0.8))
    t0 = loop.time()
    with pytest.raises(GeecsMotorTimeoutError) as info:
        await asyncio.wait_for(motor.set(-24.0), timeout=5.0)
    elapsed = loop.time() - t0
    await creep
    assert 0.8 + _STALL <= elapsed < 0.8 + _STALL + 0.6
    assert info.value.current == pytest.approx(-8.0)
    set_mock_put_proceeds(motor._setpoint, True)


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


async def test_confirm_exactly_on_tolerance_counts_as_matched() -> None:
    """The same on-boundary fix applies to the confirming poll (#820).

    |1.20 - 1.15| is 0.050000000000000044, not 0.05: an EMQ set landing
    exactly on the 0.05 default must match, not time out.
    """
    device = _emq_confirm_device(tolerance=0.05)
    await device.connect(mock=True)
    set_mock_value(device._confirm_readback, 1.20)
    assert abs(1.20 - 1.15) > 0.05  # the representation error is real
    await asyncio.wait_for(device.set(1.15), timeout=2.0)


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


# --------------------------------------------------------------------------
# A failed set names its PV in the log (GEECS-Plugins#868)
# --------------------------------------------------------------------------


class _RefusedPut(RuntimeError):
    """The shape of a failed ``aioca.CANothing``: falsy, repr = the bare code.

    Only ``str`` carries the PV and the CA message; a truthy stand-in could
    not catch an ``exc.__cause__ or exc`` selection (#817's bug).
    """

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "_RefusedPut(ECA_DISCONN)"


def _error_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]


async def test_settable_refused_put_is_logged_with_the_pv_and_the_cause(
    caplog,
) -> None:
    """The device's ERROR line: the ``:SP`` PV and the falsy cause by ``str``."""
    dev = CaSettable("U_S1H", "Current", experiment="Undulator", name="cur")
    await dev.connect(mock=True)

    def refuse(value, **kwargs):
        raise _RefusedPut("undulator:u_s1h:current:SP: Virtual circuit disconnect")

    callback_on_mock_put(dev._setpoint, refuse)
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.devices.ca"):
        with pytest.raises(_RefusedPut):
            await dev.set(0.5)
    (line,) = _error_lines(caplog)
    assert "cur: set Current → 0.5 failed (undulator:u_s1h:current:SP)" in line
    assert "_RefusedPut: undulator:u_s1h:current:SP: Virtual circuit disconnect" in line
    assert "ECA_DISCONN" not in line  # the repr, never


async def test_motor_refused_put_and_timeout_are_logged_with_the_pv(caplog) -> None:
    """CaMotor shares the seam: the Layer-1 put and the Layer-2 timeout both name the PV."""
    motor = CaMotor(
        "U_ESP_JetXYZ",
        "Position.Axis 1",
        experiment="Undulator",
        name="jet",
        progress_grace=0.1,
        stall_timeout=0.2,
    )
    await motor.connect(mock=True)
    pv = "undulator:u_esp_jetxyz:position_axis_1:SP"

    def refuse(value, **kwargs):
        raise _RefusedPut(f"{pv}: no ACK within 1.5s")

    callback_on_mock_put(motor._setpoint, refuse)
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.devices.ca"):
        with pytest.raises(_RefusedPut):
            await motor.set(4.5)
    (line,) = _error_lines(caplog)
    assert f"({pv}): _RefusedPut: {pv}: no ACK within 1.5s" in line

    caplog.clear()
    callback_on_mock_put(motor._setpoint, lambda value, **kwargs: None)
    set_mock_value(motor.position, 0.0)  # stuck far from target
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.devices.ca"):
        with pytest.raises(GeecsMotorTimeoutError):
            await motor.set(4.5)
    (line,) = _error_lines(caplog)
    assert f"({pv}): GeecsMotorTimeoutError:" in line


async def test_confirm_settable_refused_put_is_logged_with_the_pv(caplog) -> None:
    """CaConfirmSettable routes through the same seam (its confirm poll is Layer 2)."""
    dev = _emq_confirm_device(timeout=0.3)
    await dev.connect(mock=True)
    pv = "undulator:u_emqtripletbipolar:current_limit_ch1:SP"

    def refuse(value, **kwargs):
        raise _RefusedPut(f"{pv}: Channel write request failed")

    callback_on_mock_put(dev._setpoint, refuse)
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.devices.ca"):
        with pytest.raises(_RefusedPut):
            await dev.set(1.0)
    (line,) = _error_lines(caplog)
    assert f"({pv}): _RefusedPut: {pv}: Channel write request failed" in line


async def test_a_successful_set_logs_no_error(caplog) -> None:
    dev = CaSettable("U_S1H", "Current", experiment="Undulator", name="cur")
    await dev.connect(mock=True)
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.devices.ca"):
        await dev.set(0.5)
    assert _error_lines(caplog) == []


async def test_snapshot_carries_the_liveness_pv_but_never_reads_it() -> None:
    """A scalar-only device's CONNECTED signal is for the liveness gate, not a column."""
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="Undulator", name="g"
    )
    await gauge.connect(mock=True)
    expected = ca_pv("Undulator", "U_Gauge", "CONNECTED").removeprefix("ca://")
    assert gauge.connected_status.source.endswith(expected)
    set_mock_value(gauge.connected_status, "Disconnected")
    assert set(await gauge.read()) == {"g-pressure"}
