"""NativeSetpointDevice — a supply's plain records as confirm-on-readback settables, on mocks.

The record side is simulated on geecs-epics' ``eztest1-sim`` (the PR's
recorded proof).  Pinned here: the registry's ``setpoint_device`` entry
binds a noun whose settables are Movable children found the way a
``CaMotor`` child is (``mv``, a sweep axis, ``variable``, ``get_settable``,
an action step's ``Current``); a value past ``DRVH``/``DRVL``, a tripped
status bit or an INVALID readback refuses the set **before any put**; a
put whose setpoint record goes INVALID fails at once naming the alarm; a
status bit rising mid-set fails naming the bit; the set is done only when
the readback arrives within tolerance, and times out in
``GeecsConfirmTimeoutError``'s words; the device is a per-row scalar
device, a telemetry candidate, and judged down by an INVALID readback.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps
from bluesky import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import callback_on_mock_put, get_mock_put, set_mock_value

from geecs_schemas import NativeDevices
from geecs_schemas.action_plan import ActionPlan

from geecs_bluesky.actions.compiler import compile_action_plan
from geecs_bluesky.devices import native_setpoint
from geecs_bluesky.devices.native_setpoint import NativeSetpoint, NativeSetpointDevice
from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    GeecsConfirmTimeoutError,
    GeecsDeviceDownError,
    GeecsSetpointRefusedError,
)
from geecs_bluesky.namespace import GeecsNamespace
from geecs_bluesky.plans.registry import _mv_named, liveness_gate
from geecs_bluesky.plans.sweep import axis_resolver
from geecs_bluesky.utils import device_reference
from tests.ca_mock_helpers import connect_mock
from tests.test_namespace import ROSTER

SUPPLY = {
    "kind": "setpoint_device",
    "settables": {"current": {"tolerance": 0.005, "timeout": 1.0}},
    "readbacks": ["voltage"],
    "status": ["fault", "interlock"],
}


def _registry(**devices) -> NativeDevices:
    return NativeDevices.model_validate({"devices": devices})


def _run(RE: RunEngine, make_awaitable):
    async def call():
        return await make_awaitable()

    return asyncio.run_coroutine_threadsafe(call(), RE._loop).result(timeout=10.0)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch) -> None:
    monkeypatch.setattr(native_setpoint, "POLL_INTERVAL_S", 0.01)


def _supply(RE: RunEngine) -> NativeSetpointDevice:
    """A connected mock Easy-Driver with the template's drive limits, nothing tripped."""
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_VS1H=SUPPLY)
    )
    supply = ns["U_VS1H"]
    connect_mock(RE, supply)
    set_mock_value(supply.current.drive_high, 5.01)
    set_mock_value(supply.current.drive_low, -5.01)
    for child in (supply.current.setpoint_severity, supply.current.readback_severity):
        set_mock_value(child, "NO_ALARM")
    return supply


def _follows(current: NativeSetpoint, offset: float = 0.0) -> None:
    """The readback arrives at the written value (plus *offset*) on each put."""
    callback_on_mock_put(
        current.setpoint,
        lambda value, **_: set_mock_value(current.readback, value + offset),
    )


# ------------------------------------------------------------- the namespace
def test_each_settable_is_a_child_found_like_a_settable() -> None:
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_VS1H=SUPPLY)
    )
    supply = ns["U_VS1H"]
    assert isinstance(supply, NativeSetpointDevice) and ns.devices["U_VS1H"] is supply
    assert supply.name == "u_vs1h" and supply._geecs_namespace_member
    current = supply.current
    assert isinstance(current, NativeSetpoint) and current.parent is supply
    assert current.readback.source == "ca://testexp:u_vs1h:current"
    assert current.setpoint.source == "ca://testexp:u_vs1h:current:SP"
    assert current.drive_high.source == "ca://testexp:u_vs1h:current:SP.DRVH"
    assert current.setpoint_severity.source == "ca://testexp:u_vs1h:current:SP.SEVR"
    assert current.readback_severity.source == "ca://testexp:u_vs1h:current.SEVR"
    assert current.status_fault.source == "ca://testexp:u_vs1h:fault"
    assert current.status_interlock.source == "ca://testexp:u_vs1h:interlock"
    # Every lookup a CaMotor child answers to — the DB's spelling ``Current``
    # (an action library's ``variable: Current``) included.
    for spelling in ("current", "Current", "CURRENT"):
        assert ns.variable("U_VS1H", spelling) is current
        assert ns.get_settable("U_VS1H", spelling) is current
    assert ns.resolve("U_VS1H:Current") is current
    assert axis_resolver(ns)("U_VS1H.current") is current
    assert device_reference("U_VS1H", "Current") == "U_VS1H.current"
    # A listed readback is a column, readable by a check step, never settable.
    assert ns.get_readable("U_VS1H", "Voltage") is supply.voltage
    with pytest.raises(GeecsConfigurationError, match="not settable"):
        ns.get_settable("U_VS1H", "voltage")
    with pytest.raises(KeyError, match="no served scalar"):
        ns.variable("U_VS1H", "slewrate")
    # The row: the settable's readback and the listed one, with s-file headers.
    assert current.reading_key == "u_vs1h-current-readback"
    assert supply._column_headers == {
        "u_vs1h-current-readback": "U_VS1H current",
        "u_vs1h-voltage": "U_VS1H voltage",
    }


def test_the_family_names_a_setpoint_record_that_is_not_record_sp() -> None:
    """The FAST-PS writes ``current_ramp:SP`` beside its ``current`` readback."""
    fastps = {**SUPPLY, "template": "fastps", "status": []}
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_FPS=fastps)
    )
    current = ns["U_FPS"].current
    assert current.setpoint.source == "ca://testexp:u_fps:current_ramp:SP"
    assert current.readback.source == "ca://testexp:u_fps:current"


def test_a_settable_the_family_does_not_name_is_written_at_record_sp() -> None:
    entry = {**SUPPLY, "settables": {"slewrate": {"tolerance": 0.01}}}
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_VS1H=entry)
    )
    assert ns["U_VS1H"].slewrate.setpoint.source == "ca://testexp:u_vs1h:slewrate:SP"
    assert ns["U_VS1H"].slewrate._timeout == 10.0  # the schema's default


@pytest.mark.parametrize("spelling", ["U_S1H", "u_s1h"])
def test_a_registry_supply_the_db_owns_refuses_to_open(spelling: str) -> None:
    with pytest.raises(GeecsConfigurationError, match="one owner"):
        GeecsNamespace(
            ROSTER,
            file_plugin_hosts=None,
            native_devices=_registry(**{spelling: SUPPLY}),
        )


@pytest.mark.parametrize("record", ["scalars", "settables", "readback_severities"])
def test_a_record_that_would_shadow_the_noun_is_refused(record: str) -> None:
    with pytest.raises(GeecsConfigurationError, match=repr(record)):
        NativeSetpointDevice("S", "undulator:s:", {}, readbacks=[record])


def test_telemetry_reads_a_setpoint_device(RE: RunEngine) -> None:
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_VS1H=SUPPLY)
    )
    assert ns["U_VS1H"] in ns.telemetry()


def test_the_scalars_view_reads_the_readbacks_and_covers_the_settable(RE) -> None:
    supply = _supply(RE)
    set_mock_value(supply.current.readback, 1.25)
    set_mock_value(supply.voltage, 0.5)
    whole = _run(RE, supply.read)
    assert set(whole) == {"u_vs1h-current-readback", "u_vs1h-voltage"}
    assert whole["u_vs1h-current-readback"]["value"] == 1.25
    assert _run(RE, supply.scalars.read) == whole
    assert supply.scalars.covers(supply.current)


def test_a_catalog_motor_target_on_a_native_settable_is_honoured(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.namespace"):
        GeecsNamespace(
            ROSTER,
            file_plugin_hosts=None,
            native_devices=_registry(U_VS1H=SUPPLY),
            motor_targets=["U_VS1H:Current"],
        )
    assert not [r for r in caplog.records if "kind: motor" in r.getMessage()]


def test_the_new_exception_is_exported() -> None:
    from geecs_bluesky import exceptions

    assert "GeecsSetpointRefusedError" in exceptions.__all__


# ------------------------------------------------------------------- the set
def test_a_set_completes_when_the_readback_arrives_within_tolerance(RE) -> None:
    supply = _supply(RE)
    _follows(supply.current, offset=0.004)  # inside 0.005
    _run(RE, lambda: supply.current.set(2.0))
    get_mock_put(supply.current.setpoint).assert_called_once_with(2.0)


def test_a_put_ok_is_not_done_the_readback_is_waited_for(RE) -> None:
    """StreamDevice completes the put on ``#AK``; the readback scans later."""
    supply = _supply(RE)
    current = supply.current

    def arrive_later(value: float, **_) -> None:
        async def later() -> None:
            await asyncio.sleep(0.2)
            set_mock_value(current.readback, value)

        asyncio.get_running_loop().create_task(later())

    callback_on_mock_put(current.setpoint, arrive_later)
    status = _run(RE, lambda: asyncio.sleep(0, result=current.set(3.0)))
    _run(RE, lambda: asyncio.sleep(0.05))
    assert not status.done  # the put completed, the readback has not arrived
    _run(RE, lambda: status)
    assert status.success


@pytest.mark.parametrize("value", [5.02, -5.5, float("nan")])
def test_a_value_past_the_drive_limits_is_refused_before_any_put(RE, value) -> None:
    """An ao clamps to DRVH/DRVL and writes: 6 A would run at 5 A with every row 'ok'."""
    supply = _supply(RE)
    with pytest.raises(GeecsSetpointRefusedError, match="nothing was written") as e:
        _run(RE, lambda: supply.current.set(value))
    assert e.value.written is False
    get_mock_put(supply.current.setpoint).assert_not_called()


def test_drive_limits_both_zero_drive_anything(RE) -> None:
    """An ao with DRVH <= DRVL has no drive limits (the record's own rule)."""
    supply = _supply(RE)
    set_mock_value(supply.current.drive_high, 0.0)
    set_mock_value(supply.current.drive_low, 0.0)
    _follows(supply.current)
    _run(RE, lambda: supply.current.set(7.0))
    get_mock_put(supply.current.setpoint).assert_called_once_with(7.0)


def test_a_tripped_status_bit_refuses_before_any_put(RE) -> None:
    supply = _supply(RE)
    set_mock_value(supply.current.status_interlock, True)
    with pytest.raises(GeecsSetpointRefusedError, match="interlock read 1"):
        _run(RE, lambda: supply.current.set(1.0))
    get_mock_put(supply.current.setpoint).assert_not_called()


def test_an_invalid_readback_refuses_before_any_put(RE) -> None:
    supply = _supply(RE)
    set_mock_value(supply.current.readback_severity, "INVALID")
    set_mock_value(supply.current.readback_alarm, "TIMEOUT")
    with pytest.raises(GeecsSetpointRefusedError, match=r"INVALID \(TIMEOUT\)"):
        _run(RE, lambda: supply.current.set(1.0))
    get_mock_put(supply.current.setpoint).assert_not_called()


@pytest.mark.parametrize("alarm", ["HIHI", "WRITE"])
def test_a_setpoint_record_gone_invalid_fails_at_once_naming_the_alarm(
    RE, alarm: str
) -> None:
    """HIHI: the IOC's IVOA refusal past +-5 A; WRITE: the supply's #NAK (output off)."""
    supply = _supply(RE)
    current = supply.current

    def refused(value: float, **_) -> None:
        set_mock_value(current.setpoint_severity, "INVALID")
        set_mock_value(current.setpoint_alarm, alarm)

    callback_on_mock_put(current.setpoint, refused)
    with pytest.raises(GeecsSetpointRefusedError, match=f"INVALID \\({alarm}\\)") as e:
        _run(RE, lambda: current.set(5.005))
    assert e.value.written is True
    assert "testexp:u_vs1h:current:SP" in str(e.value)


def test_a_status_bit_rising_mid_set_fails_naming_the_bit(RE) -> None:
    supply = _supply(RE)
    current = supply.current
    callback_on_mock_put(
        current.setpoint, lambda value, **_: set_mock_value(current.status_fault, True)
    )
    with pytest.raises(GeecsSetpointRefusedError, match="fault rose") as e:
        _run(RE, lambda: current.set(2.0))
    assert e.value.written is True


def test_a_readback_that_never_arrives_times_out_in_the_confirm_words(RE) -> None:
    supply = _supply(RE)
    set_mock_value(supply.current.readback, 0.5)
    with pytest.raises(GeecsConfirmTimeoutError) as e:
        _run(RE, lambda: supply.current.set(2.0))
    message = str(e.value)
    assert "U_VS1H/current: confirming variable testexp:u_vs1h:current" in message
    assert "reads 0.5, did not match 2.0 within 1.0s" in message


def test_a_readback_arriving_after_the_deadline_times_out(RE) -> None:
    """A slow read is bounded by the set's timeout; its late in-tolerance value is refused."""
    supply = _supply(RE)
    current = supply.current
    current._timeout = 0.05
    real = current.readback.get_value

    async def slow(cached=None):
        await asyncio.sleep(0.2)
        set_mock_value(current.readback, 2.0)  # on target, but too late
        return await real(cached=cached)

    current.readback.get_value = slow  # type: ignore[method-assign]
    loop_time = asyncio.run_coroutine_threadsafe(_now(), RE._loop).result
    started = loop_time()
    with pytest.raises(GeecsConfirmTimeoutError, match="within 0.1s"):
        _run(RE, lambda: current.set(2.0))
    assert loop_time() - started < 0.15


def test_a_read_that_never_returns_times_out_in_the_confirm_words(RE) -> None:
    supply = _supply(RE)
    current = supply.current
    current._timeout = 0.05

    async def stalled(cached=None):
        await asyncio.Event().wait()

    current.readback.get_value = stalled  # type: ignore[method-assign]
    with pytest.raises(GeecsConfirmTimeoutError, match="no read completed"):
        _run(RE, lambda: current.set(2.0))


async def _now() -> float:
    return asyncio.get_running_loop().time()


def test_the_set_reads_the_records_not_the_monitor_cache(RE) -> None:
    supply = _supply(RE)
    current = supply.current
    _follows(current)
    calls: list[object] = []
    real = current.readback.get_value

    async def spy(cached=None):
        calls.append(cached)
        return await real(cached=cached)

    current.readback.get_value = spy  # type: ignore[method-assign]
    _run(RE, lambda: current.set(1.0))
    assert calls and all(c is False for c in calls)


def test_mv_and_an_action_step_set_the_supply_and_a_refusal_is_named(
    RE, caplog
) -> None:
    """``{do: set, device: U_VS1H, variable: Current, value: 0}`` — the configs' own step."""
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_VS1H=SUPPLY)
    )
    supply = ns["U_VS1H"]
    connect_mock(RE, supply)
    set_mock_value(supply.current.drive_high, 5.01)
    set_mock_value(supply.current.drive_low, -5.01)
    set_mock_value(supply.current.readback, 1.0)
    _follows(supply.current)
    RE(_mv_named(supply.current, 3.0))
    plan = ActionPlan.model_validate(
        {
            "steps": [
                {"do": "set", "device": "U_VS1H", "variable": "Current", "value": 0}
            ]
        }
    )
    RE(compile_action_plan(plan, registry={}, settables=ns))
    assert _run(RE, supply.current.readback.get_value) == 0.0
    with caplog.at_level(logging.ERROR, logger=native_setpoint.__name__):
        with pytest.raises(FailedStatus) as caught:
            RE(_mv_named(supply.current, 6.0))
    assert "drive limits" in str(caught.value)
    (record,) = [r for r in caplog.records if r.name == native_setpoint.__name__]
    assert "testexp:u_vs1h:current:SP" in record.getMessage()
    assert [
        c.args[0] for c in get_mock_put(supply.current.setpoint).call_args_list
    ] == [
        3.0,
        0.0,
    ]


def test_mv_through_a_run_engine_connects_the_settable_on_demand(RE) -> None:
    """A plan touching ``U_VS1H.current`` alone connects it (its bits and limits too)."""
    from geecs_bluesky.preprocessors import connect_on_demand

    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_VS1H=SUPPLY)
    )
    current = ns["U_VS1H"].current
    RE.preprocessors.append(lambda plan: connect_on_demand(plan, mock=True))
    RE(bps.mv(current, 0.0))  # the mock readback already reads 0
    get_mock_put(current.setpoint).assert_called_once_with(0.0)


# --------------------------------------------------------------- liveness
def test_the_run_gate_passes_a_live_supply(RE) -> None:
    supply = _supply(RE)
    RE(liveness_gate(object(), [supply]))
    RE(liveness_gate(object(), [supply.scalars]))


@pytest.mark.parametrize("which", ["settable", "readback"])
def test_the_run_gate_refuses_a_supply_with_an_invalid_readback(RE, which) -> None:
    supply = _supply(RE)
    signal = (
        supply.current.readback_severity
        if which == "settable"
        else supply.readback_severities[0]
    )
    set_mock_value(signal, "INVALID")
    with pytest.raises(GeecsDeviceDownError, match="U_VS1H DISCONNECTED"):
        RE(liveness_gate(object(), [supply.scalars]))


def test_the_run_gate_refuses_a_supply_whose_ioc_never_answers(RE, monkeypatch) -> None:
    monkeypatch.setattr(native_setpoint, "LIVENESS_TIMEOUT_S", 0.5)
    supply = NativeSetpointDevice(
        "U_VS1H",
        "geecs_test_no_such_ioc:",
        NativeDevices.model_validate({"devices": {"U_VS1H": SUPPLY}})
        .devices["U_VS1H"]
        .settables,
        name="u_vs1h",
    )
    with pytest.raises(GeecsDeviceDownError, match="U_VS1H DISCONNECTED"):
        RE(liveness_gate(object(), [supply]))
