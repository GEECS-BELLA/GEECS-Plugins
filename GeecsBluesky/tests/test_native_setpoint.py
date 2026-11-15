"""NativeSetpointDevice — a supply's plain records as confirm-on-readback settables, on mocks.

The record side is simulated on geecs-epics' ``eztest1-sim`` (the PR's
recorded proof).  Pinned here: the registry's ``setpoint_device`` entry
binds a noun whose settables are Movable children found the way a
``CaMotor`` child is (``mv``, a sweep axis, ``variable``, ``get_settable``,
an action step's ``Current``); a value past ``DRVH``/``DRVL``, a tripped
status bit or an INVALID readback refuses the set **before any put**; a
put whose setpoint record goes INVALID fails at once naming the alarm; a
status bit rising mid-set fails naming the bit; the set is done only when
the readback arrives within the IOC's set tolerance (its ``current:TOL``
record, read at every set), and times out in ``GeecsConfirmTimeoutError``'s
words; no tolerance (the record never connecting, INVALID, zero, negative,
not finite) refuses the set before any put and a bound scan before its
claim, and a scan records the values it read; the device is a per-row
scalar device, a telemetry candidate, and judged down by an INVALID
readback.
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
from geecs_schemas import Setup

from geecs_bluesky.plans.apply_setup import apply_setup_plan
from geecs_bluesky.devices import native_setpoint
from geecs_bluesky.devices.native_setpoint import NativeSetpoint, NativeSetpointDevice
from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    GeecsConfirmTimeoutError,
    GeecsDeviceDownError,
    GeecsNoSetToleranceError,
    GeecsSetpointRefusedError,
)
from geecs_bluesky.namespace import GeecsNamespace
from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.plans.registry import (
    TriggerProfiles,
    _mv_named,
    bind_plans,
    liveness_gate,
)
from geecs_bluesky.plans.sweep import axis_resolver
from geecs_bluesky.utils import device_reference
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_namespace import ROSTER
from tests.test_plan_registry import payload
from tests.test_strict_plans import WRITES, FakeBox, _camera

SUPPLY = {
    "kind": "setpoint_device",
    "settables": {"current": {"timeout": 1.0}},
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


def _supply(RE: RunEngine, entry: dict = SUPPLY) -> NativeSetpointDevice:
    """A connected mock Easy-Driver: the template's drive limits, nothing tripped, ``current:TOL`` 5 mA."""
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_VS1H=entry)
    )
    supply = ns["U_VS1H"]
    connect_mock(RE, supply)
    set_mock_value(supply.current.drive_high, 5.01)
    set_mock_value(supply.current.drive_low, -5.01)
    for child in (supply.current.setpoint_severity, supply.current.readback_severity):
        set_mock_value(child, "NO_ALARM")
    _serve_tolerance(supply.current, 0.005)
    return supply


def _serve_tolerance(
    current: NativeSetpoint,
    value: float,
    severity: str = "NO_ALARM",
    alarm: str = "NO_ALARM",
) -> None:
    """The IOC's ``current:TOL`` reads *value* at *severity* (a missing value: 0, INVALID, LOLO)."""
    record = current._tolerance
    set_mock_value(record.value, value)
    set_mock_value(record.severity, severity)
    set_mock_value(record.alarm, alarm)


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
    # The set tolerance: the family's tolerance_by role for current:SP, the
    # IOC's record — never a child, so a supply whose IOC does not serve it
    # yet still connects and reads.
    assert current.tolerance_pv == "testexp:u_vs1h:current:TOL"
    assert current._tolerance.value.source == "ca://testexp:u_vs1h:current:TOL"
    assert current._tolerance.severity.source == "ca://testexp:u_vs1h:current:TOL.SEVR"
    sources = {getattr(child, "source", None) for _, child in current.children()}
    assert current._tolerance.value.source not in sources
    assert current.status_fault.source == "ca://testexp:u_vs1h:fault"
    assert current.status_interlock.source == "ca://testexp:u_vs1h:interlock"
    # Every lookup a CaMotor child answers to — the DB's spelling ``Current``
    # (a setup's ``variable: Current``) included.
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


FASTPS = {**SUPPLY, "template": "fastps", "status": []}


def test_the_family_names_a_setpoint_record_that_is_not_record_sp() -> None:
    """The FAST-PS writes ``current_ramp:SP`` and confirms within the one ``current:TOL``."""
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_FPS=FASTPS)
    )
    current = ns["U_FPS"].current
    assert current.setpoint.source == "ca://testexp:u_fps:current_ramp:SP"
    assert current.readback.source == "ca://testexp:u_fps:current"
    assert current.tolerance_pv == "testexp:u_fps:current:TOL"


def test_a_fastps_set_writes_the_ramp_and_confirms_within_current_tol(RE) -> None:
    supply = _supply(RE, FASTPS)
    _serve_tolerance(supply.current, 0.1)  # the FAST-PS's own value
    _follows(supply.current, offset=0.09)
    _run(RE, lambda: supply.current.set(2.0))
    get_mock_put(supply.current.setpoint).assert_called_once_with(2.0)
    _serve_tolerance(supply.current, 0.05)
    with pytest.raises(GeecsConfirmTimeoutError):
        _run(RE, lambda: supply.current.set(1.0))


def test_a_settable_the_family_does_not_name_is_refused() -> None:
    """No setpoint role, no set-tolerance record: the registry does not load."""
    entry = {**SUPPLY, "settables": {"slewrate": {}}}
    with pytest.raises(ValueError, match="'easydriver' names no such record"):
        _registry(U_VS1H=entry)


def test_a_family_with_no_tolerance_record_refuses_to_build(monkeypatch) -> None:
    """No fallback: the worker refuses, naming the family and the setpoint role."""
    from geecs_bluesky import namespace

    families = dict(namespace.load_families())
    family = families["easydriver"]
    families["easydriver"] = family.model_copy(
        update={
            "roles": [
                r.model_copy(update={"tolerance_by": None})
                if r.name == "current_sp"
                else r
                for r in family.roles
            ]
        }
    )
    registry = _registry(U_VS1H=SUPPLY)  # valid against the installed families
    monkeypatch.setattr(namespace, "load_families", lambda: families)
    with pytest.raises(
        GeecsConfigurationError,
        match="U_VS1H: settable 'current': family 'easydriver'.s setpoint role "
        "'current_sp' names no set-tolerance record",
    ):
        GeecsNamespace(ROSTER, file_plugin_hosts=None, native_devices=registry)


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
        NativeSetpointDevice(
            "S", "undulator:s:", {}, setpoints={}, tolerances={}, readbacks=[record]
        )


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


def test_mv_and_a_setup_step_set_the_supply_and_a_refusal_is_named(RE, caplog) -> None:
    """``{do: set, device: U_VS1H, variable: Current, value: 0}`` — the configs' own step."""
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_VS1H=SUPPLY)
    )
    supply = ns["U_VS1H"]
    connect_mock(RE, supply)
    set_mock_value(supply.current.drive_high, 5.01)
    set_mock_value(supply.current.drive_low, -5.01)
    set_mock_value(supply.current.readback, 1.0)
    _serve_tolerance(supply.current, 0.005)
    _follows(supply.current)
    RE(_mv_named(supply.current, 3.0))
    zero = Setup.model_validate(
        {
            "name": "zero",
            "steps": [
                {"do": "set", "device": "U_VS1H", "variable": "Current", "value": 0}
            ],
        }
    )

    class _OneSetup:
        def resolve_setup(self, name):
            return zero, {"zero": zero}

        def setup_repo_path(self, name):
            return "setups/zero.yaml"

    RE(apply_setup_plan(_OneSetup(), ns)("zero"))
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


def test_a_setup_refuses_a_supply_judged_down_before_its_put(RE) -> None:
    """A setup's gate is the run's: an INVALID readback means down, and nothing is put."""
    from geecs_bluesky.exceptions import SetupStepFailedError

    supply = _supply(RE)  # live: every severity NO_ALARM, current:TOL served
    set_mock_value(supply.current.readback_severity, "INVALID")
    zero = Setup.model_validate(
        {
            "name": "zero",
            "steps": [
                {"do": "set", "device": "U_VS1H", "variable": "Current", "value": 0}
            ],
        }
    )

    class _OneSetup:
        def resolve_setup(self, name):
            return zero, {"zero": zero}

        def setup_repo_path(self, name):
            return "setups/zero.yaml"

    class _Factory:  # the namespace's own lookup is pinned by the test above
        def get_settable(self, device, variable):
            return supply.current

        def get_readable(self, device, variable):
            return supply.current

    with pytest.raises(SetupStepFailedError, match="U_VS1H DISCONNECTED") as caught:
        RE(apply_setup_plan(_OneSetup(), _Factory())("zero"))
    assert (caught.value.index, caught.value.written) == (0, ())
    get_mock_put(supply.current.setpoint).assert_not_called()


def test_a_supply_down_by_its_check_fails_it_on_the_retained_readback(RE) -> None:
    """Live at the gate, INVALID by the check: its own judge refuses the check, the write named."""
    from geecs_bluesky.exceptions import SetupStepFailedError

    supply = _supply(RE)

    def drop_off_later(
        value, **_
    ):  # follows, then stops answering after the set confirms
        set_mock_value(supply.current.readback, value)
        asyncio.get_running_loop().call_later(
            0.1, set_mock_value, supply.current.readback_severity, "INVALID"
        )

    callback_on_mock_put(supply.current.setpoint, drop_off_later)
    setup = Setup.model_validate(
        {
            "name": "zero_checked",
            "steps": [
                {"do": "set", "device": "U_VS1H", "variable": "Current", "value": 0},
                {"do": "wait", "seconds": 0.5},
                {
                    "do": "check",
                    "device": "U_VS1H",
                    "variable": "Current",
                    "expected": 0,
                },
            ],
        }
    )

    class _OneSetup:
        def resolve_setup(self, name):
            return setup, {setup.name: setup}

        def setup_repo_path(self, name):
            return "setups/zero_checked.yaml"

    class _Factory:
        def get_settable(self, device, variable):
            return supply.current

        def get_readable(self, device, variable):
            return supply.current

    with pytest.raises(SetupStepFailedError, match="U_VS1H DISCONNECTED") as caught:
        RE(apply_setup_plan(_OneSetup(), _Factory())("zero_checked"))
    assert (caught.value.index, caught.value.written) == (3, (1,))


def test_mv_through_a_run_engine_connects_the_settable_on_demand(RE) -> None:
    """A plan touching ``U_VS1H.current`` alone connects it (its bits, limits and tolerance too)."""
    from geecs_bluesky.preprocessors import connect_on_demand

    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_VS1H=SUPPLY)
    )
    current = ns["U_VS1H"].current
    RE.preprocessors.append(lambda plan: connect_on_demand(plan, mock=True))
    # A fresh mock's current:TOL reads 0: connected with the settable, and no tolerance.
    with pytest.raises(FailedStatus, match="current:TOL reads 0"):
        RE(bps.mv(current, 0.0))
    get_mock_put(current.setpoint).assert_not_called()
    _serve_tolerance(current, 0.005)
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


def _unserved_supply() -> NativeSetpointDevice:
    """The supply at a prefix no IOC serves (real CA, nothing answers)."""
    return NativeSetpointDevice(
        "U_VS1H",
        "geecs_test_no_such_ioc:",
        NativeDevices.model_validate({"devices": {"U_VS1H": SUPPLY}})
        .devices["U_VS1H"]
        .settables,
        setpoints={"current": "current:SP"},
        tolerances={"current": "current:TOL"},
        name="u_vs1h",
    )


def test_the_run_gate_refuses_a_supply_whose_ioc_never_answers(RE, monkeypatch) -> None:
    monkeypatch.setattr(native_setpoint, "LIVENESS_TIMEOUT_S", 0.5)
    supply = _unserved_supply()
    with pytest.raises(GeecsDeviceDownError, match="U_VS1H DISCONNECTED"):
        RE(liveness_gate(object(), [supply]))


# ------------------------------------------------------- the set tolerance
def test_the_retired_registry_tolerance_is_ignored_and_said(RE, caplog) -> None:
    """A file still carrying ``tolerance: 0.5`` confirms within the IOC's 0.005, never 0.5."""
    entry = {**SUPPLY, "settables": {"current": {"tolerance": 0.5, "timeout": 1.0}}}
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.namespace"):
        supply = _supply(RE, entry)
    (warning,) = [r for r in caplog.records if "retired" in r.getMessage()]
    assert "U_VS1H: native_devices.yaml still gives current the retired" in (
        warning.getMessage()
    )
    assert "testexp:u_vs1h:current:TOL" in warning.getMessage()
    _follows(supply.current, offset=0.006)  # inside 0.5, outside the IOC's 0.005
    with pytest.raises(GeecsConfirmTimeoutError):
        _run(RE, lambda: supply.current.set(2.0))


def test_a_readback_just_outside_the_iocs_tolerance_times_out(RE) -> None:
    supply = _supply(RE)
    _follows(supply.current, offset=0.006)  # outside current:TOL's 0.005
    with pytest.raises(GeecsConfirmTimeoutError, match="did not match 2.0"):
        _run(RE, lambda: supply.current.set(2.0))


def test_the_set_logs_the_tolerance_it_confirms_within(RE, caplog) -> None:
    supply = _supply(RE)
    _follows(supply.current)
    with caplog.at_level(logging.INFO, logger=native_setpoint.__name__):
        _run(RE, lambda: supply.current.set(1.0))
    assert any(
        "within 0.005, the IOC's testexp:u_vs1h:current:TOL" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize(
    "value, severity, alarm, read",
    [
        # an IOC loaded without the value serves 0 INVALID (LOLO)
        (0.0, "INVALID", "LOLO", r"reads 0 INVALID \(LOLO\)"),
        (0.005, "INVALID", "UDF", r"reads 0.005 INVALID \(UDF\)"),
        (0.0, "NO_ALARM", "NO_ALARM", r"reads 0 \(not above zero\)"),
        (-0.005, "MAJOR", "LOLO", r"reads -0.005 \(not above zero\)"),
        (float("nan"), "NO_ALARM", "NO_ALARM", r"reads nan \(not a finite number\)"),
        (float("inf"), "NO_ALARM", "NO_ALARM", r"reads inf \(not a finite number\)"),
    ],
)
def test_no_tolerance_refuses_the_set_before_any_put(
    RE, value, severity, alarm, read
) -> None:
    """Never zero, never infinity: the set cannot be confirmed, so nothing is written."""
    supply = _supply(RE)
    _serve_tolerance(supply.current, value, severity, alarm)
    with pytest.raises(GeecsNoSetToleranceError, match=read) as e:
        _run(RE, lambda: supply.current.set(1.0))
    assert isinstance(e.value, GeecsSetpointRefusedError)
    assert e.value.written is False and e.value.target == 1.0
    assert e.value.pv == "testexp:u_vs1h:current:TOL"
    assert str(e.value).startswith(
        "U_VS1H/current: set to 1 refused — no set tolerance: "
        "testexp:u_vs1h:current:TOL reads"
    )
    get_mock_put(supply.current.setpoint).assert_not_called()


def test_a_record_that_never_connects_refuses_the_set(RE, monkeypatch) -> None:
    """The IOC does not serve current:TOL (not restarted onto decision 21): real CA, nothing answers."""
    monkeypatch.setattr(native_setpoint, "TOLERANCE_TIMEOUT_S", 0.3)
    current = _unserved_supply().current
    with pytest.raises(
        GeecsNoSetToleranceError, match="does not connect within 0.3 s"
    ) as e:
        _run(RE, current.read_tolerance)
    assert str(e.value).startswith(
        "U_VS1H/current: cannot be set by a scan — no set tolerance: "
        "geecs_test_no_such_ioc:current:TOL does not connect"
    )
    assert e.value.target is None and e.value.written is False


def test_a_record_that_stops_answering_refuses_the_set(RE) -> None:
    supply = _supply(RE)

    async def silent(cached=None):
        await asyncio.Event().wait()

    supply.current._tolerance.value.get_value = silent  # type: ignore[method-assign]
    native_setpoint.TOLERANCE_TIMEOUT_S, saved = (
        0.05,
        native_setpoint.TOLERANCE_TIMEOUT_S,
    )
    try:
        with pytest.raises(GeecsNoSetToleranceError, match="did not answer within"):
            _run(RE, lambda: supply.current.set(1.0))
    finally:
        native_setpoint.TOLERANCE_TIMEOUT_S = saved
    get_mock_put(supply.current.setpoint).assert_not_called()


def test_a_tolerance_gone_invalid_is_not_remembered(RE) -> None:
    """A set after a good read sees the record as it is now, never the number from before."""
    supply = _supply(RE)
    _follows(supply.current)
    _run(RE, lambda: supply.current.set(1.0))
    _serve_tolerance(supply.current, 0.0, "INVALID", "LOLO")
    with pytest.raises(GeecsNoSetToleranceError):
        _run(RE, lambda: supply.current.set(2.0))
    assert [
        c.args[0] for c in get_mock_put(supply.current.setpoint).call_args_list
    ] == [1.0]


def test_a_changed_tolerance_is_used_and_said(RE, caplog) -> None:
    """The IOC restarted onto a new configuration between scans: the next set uses it."""
    supply = _supply(RE)
    _follows(supply.current, offset=0.004)
    _run(RE, lambda: supply.current.set(1.0))  # inside 0.005
    _serve_tolerance(supply.current, 0.002)
    with caplog.at_level(logging.WARNING, logger=native_setpoint.__name__):
        with pytest.raises(GeecsConfirmTimeoutError):
            _run(RE, lambda: supply.current.set(2.0))  # 0.004 off, outside 0.002
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "current:TOL changed from 0.005 to 0.002" in warning.getMessage()
    caplog.clear()
    _serve_tolerance(supply.current, 0.01)
    _run(RE, lambda: supply.current.set(3.0))  # inside the wider value
    assert "changed from 0.002 to 0.01" in caplog.records[0].getMessage()


# ------------------------------------------- the run: refused before the claim
def _bound_sweep(RE, *, tolerance=(0.005, "NO_ALARM", "NO_ALARM")):
    """A connected mock supply serving *tolerance*, a camera, the bound plans over a fake box."""
    box = FakeBox()
    sc = ShotControl(WRITES, experiment="TestExp", name="htu_test", setter_factory=box)
    connect_mock(RE, sc)
    profiles = TriggerProfiles({"HTU-Test": sc}, default="HTU-Test")
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(U_VS1H=SUPPLY)
    )
    supply = ns["U_VS1H"]
    connect_mock(RE, supply)
    set_mock_value(supply.current.drive_high, 5.01)
    set_mock_value(supply.current.drive_low, -5.01)
    for child in (supply.current.setpoint_severity, supply.current.readback_severity):
        set_mock_value(child, "NO_ALARM")
    _serve_tolerance(supply.current, *tolerance)
    _follows(supply.current)
    cam = _camera(RE, box, "UC_Cam")
    col = DocCollector()
    RE.subscribe(col)
    plans = bind_plans(profiles, settables=ns)
    return supply, cam, col, plans, box


def test_a_sweep_records_the_tolerance_it_confirms_within(RE) -> None:
    supply, cam, col, plans, _ = _bound_sweep(RE)
    RE(plans["sweep"]([cam], sweep=payload("U_VS1H.current", -0.1, 0.1, 3)))
    (start,) = col.docs["start"]
    assert start["set_tolerances"] == {
        "U_VS1H:current": {"pv": "testexp:u_vs1h:current:TOL", "value": 0.005}
    }
    assert [
        c.args[0] for c in get_mock_put(supply.current.setpoint).call_args_list
    ] == pytest.approx([-0.1, 0.0, 0.1])


def test_a_count_moves_nothing_and_records_no_tolerance(RE) -> None:
    """A supply read as a condition needs no tolerance: its readbacks are recorded."""
    supply, cam, col, plans, _ = _bound_sweep(RE, tolerance=(0.0, "INVALID", "LOLO"))
    RE(plans["count"]([cam, supply], 1))
    (start,) = col.docs["start"]
    assert "set_tolerances" not in start
    assert "u_vs1h-current-readback" in col.primary_events()[0]["data"]


@pytest.mark.parametrize("acquisition", ["strict", "fly"])
@pytest.mark.parametrize(
    "tolerance, read",
    [
        ((0.0, "INVALID", "LOLO"), "reads 0 INVALID"),
        ((-1.0, "NO_ALARM", "NO_ALARM"), "not above zero"),
    ],
)
def test_a_sweep_with_no_tolerance_is_refused_before_the_claim(
    RE, tolerance, read, acquisition
) -> None:
    supply, cam, col, plans, box = _bound_sweep(RE, tolerance=tolerance)
    with pytest.raises(GeecsNoSetToleranceError, match=read) as e:
        RE(
            plans["sweep"](
                [cam],
                sweep=payload("U_VS1H.current", -0.1, 0.1, 3),
                acquisition=acquisition,
            )
        )
    assert str(e.value).startswith(
        "U_VS1H/current: cannot be set by a scan — no set tolerance: "
        "testexp:u_vs1h:current:TOL"
    )
    assert any("nothing claimed, no scan folder" in n for n in e.value.__notes__)
    assert box.puts == [] and box.fires == 0  # the box was never driven
    assert col.docs["start"] == []  # no open_run: nothing claimed
    get_mock_put(supply.current.setpoint).assert_not_called()


def test_the_run_gate_names_every_settable_without_a_tolerance(RE) -> None:
    from geecs_bluesky.devices.native_setpoint import native_set_tolerances

    one, two = _supply(RE), _supply(RE)
    _serve_tolerance(one.current, 0.0, "INVALID", "LOLO")
    _serve_tolerance(two.current, -1.0)
    with pytest.raises(GeecsNoSetToleranceError) as e:
        RE(native_set_tolerances([one.current, two.current, one.current]))
    assert "reads 0 INVALID" in str(e.value)
    assert any("reads -1" in n for n in e.value.__notes__)
