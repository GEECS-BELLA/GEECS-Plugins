"""NativeMotor — a positioner's axes as stock EPICS motor records, on mocks.

The record side is simulated on geecs-epics' ``simmotor1`` (the PR's
recorded proof: three moves with readback, an out-of-limit target refused
before the put).  Pinned here: the registry's ``epics_motor`` entry binds a
noun whose axes are Movable children found the way a ``CaMotor`` child is
(``mv``, a sweep axis, ``variable``, ``get_settable``); the soft limits
refuse a target before any put; a move that stops short of the target —
a limit switch, a refused put read as success — fails naming the cause;
the positioner is a per-row scalar device with the ``.scalars`` view, a
telemetry candidate, and never a camera.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps
from bluesky import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import callback_on_mock_put, get_mock_put, set_mock_value
from ophyd_async.epics.motor import MotorLimitsError

from geecs_schemas import NativeDevices

from geecs_bluesky.devices.native import NativeCamera
from geecs_bluesky.devices.native_motor import NativeMotor, NativeMotorAxis
from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    GeecsMotorStoppedShortError,
)
from geecs_bluesky.namespace import GeecsNamespace
from geecs_bluesky.plans.registry import _mv_named
from geecs_bluesky.plans.sweep import axis_resolver
from geecs_bluesky.utils import device_reference
from tests.ca_mock_helpers import connect_mock
from tests.test_native_registry import ENTRY as CAMERA, _Shared
from tests.test_namespace import ROSTER

MOTOR = {"kind": "epics_motor", "axes": ["m1", "m2"]}


def _registry(**devices) -> NativeDevices:
    return NativeDevices.model_validate({"devices": devices})


def _run(RE: RunEngine, make_awaitable):
    async def call():
        return await make_awaitable()

    return asyncio.run_coroutine_threadsafe(call(), RE._loop).result(timeout=10.0)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


def _axis(RE: RunEngine) -> NativeMotorAxis:
    """A connected mock axis with the sim's limits and deadband, moving instantly."""
    motor = NativeMotor("SimMotor1", "undulator:simmotor1:", ["m1"], name="simmotor1")
    connect_mock(RE, motor)
    axis = motor.m1
    for signal, value in (
        (axis.low_limit_travel, -20.0),
        (axis.high_limit_travel, 20.0),
        (axis.dial_low_limit_travel, -20.0),
        (axis.dial_high_limit_travel, 20.0),
        (axis.deadband, 0.01),
        (axis.motor_egu, "mm"),
    ):
        set_mock_value(signal, value)
    return axis


# ------------------------------------------------------------- the namespace
def test_each_axis_is_a_motor_record_child_found_like_a_settable(tmp_path) -> None:
    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(SimMotor1=MOTOR)
    )
    stage = ns["SimMotor1"]
    assert isinstance(stage, NativeMotor) and ns.devices["SimMotor1"] is stage
    assert stage.name == "simmotor1" and stage._geecs_device_name == "SimMotor1"
    assert stage._geecs_namespace_member
    axis = stage.m1
    assert isinstance(axis, NativeMotorAxis) and axis.parent is stage
    assert axis.user_setpoint.source == "ca://testexp:simmotor1:m1.VAL"
    assert axis.user_readback.source == "ca://testexp:simmotor1:m1.RBV"
    assert axis.motor_stop.source == "ca://testexp:simmotor1:m1.STOP"
    # Every lookup a CaMotor child answers to: the namespace's, an action
    # step's, a sweep axis's, and the preset's spelling of the reference.
    assert ns.variable("SimMotor1", "m1") is axis
    assert ns.resolve("SimMotor1:M2") is stage.m2
    assert ns.get_settable("SimMotor1", "m1") is axis
    assert axis_resolver(ns)("SimMotor1.m1") is axis
    assert device_reference("SimMotor1", "m1") == "SimMotor1.m1"
    # The row: each axis's readback under the motor's own key, with its
    # s-file header on the axis and on the noun (the header walk).
    assert axis.reading_key == "simmotor1-m1"
    assert axis._column_headers == {"simmotor1-m1": "SimMotor1 m1"}
    assert stage._column_headers == {
        "simmotor1-m1": "SimMotor1 m1",
        "simmotor1-m2": "SimMotor1 m2",
    }
    with pytest.raises(KeyError, match="no served scalar"):
        ns.variable("SimMotor1", "m3")


def test_a_positioner_needs_no_path_provider_and_a_camera_still_does() -> None:
    GeecsNamespace(ROSTER, file_plugin_hosts=None, native_devices=_registry(S=MOTOR))
    with pytest.raises(GeecsConfigurationError, match="path provider"):
        GeecsNamespace(
            ROSTER, file_plugin_hosts=None, native_devices=_registry(C=CAMERA)
        )


@pytest.mark.parametrize("spelling", ["UC_TestCam", "uc_testcam"])
def test_a_registry_motor_the_db_owns_refuses_to_open(spelling: str) -> None:
    with pytest.raises(GeecsConfigurationError, match="one owner"):
        GeecsNamespace(
            ROSTER,
            file_plugin_hosts=None,
            native_devices=_registry(**{spelling: MOTOR}),
        )


def test_both_kinds_bind_side_by_side(tmp_path: Path) -> None:
    ns = GeecsNamespace(
        ROSTER,
        path_provider=_Shared(tmp_path),
        file_plugin_hosts=None,
        native_devices=_registry(UC_Nat=CAMERA, SimMotor1=MOTOR),
    )
    assert isinstance(ns["UC_Nat"], NativeCamera)
    assert isinstance(ns["SimMotor1"], NativeMotor)


def test_telemetry_reads_a_positioner_never_a_camera(tmp_path: Path) -> None:
    """An axis readback costs no shot (NATIVE_DEVICES.md question 7); a camera's read waits for one."""
    ns = GeecsNamespace(
        ROSTER,
        path_provider=_Shared(tmp_path),
        file_plugin_hosts=None,
        native_devices=_registry(UC_Nat=CAMERA, SimMotor1=MOTOR),
    )
    assert ns["SimMotor1"] in ns.telemetry()
    assert ns["UC_Nat"] not in ns.telemetry()


def test_a_catalog_motor_target_on_a_native_axis_is_honoured(caplog) -> None:
    """The catalog's ``kind: motor`` opt-in is redundant on a native axis, never unhonoured."""
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.namespace"):
        GeecsNamespace(
            ROSTER,
            file_plugin_hosts=None,
            native_devices=_registry(SimMotor1=MOTOR),
            motor_targets=["SimMotor1:m1"],
        )
    assert not [r for r in caplog.records if "kind: motor" in r.getMessage()]


def test_the_scalars_view_reads_the_axes_and_covers_each(RE: RunEngine) -> None:
    stage = NativeMotor("SimMotor1", "undulator:simmotor1:", ["m1", "m2"], name="s")
    connect_mock(RE, stage)
    set_mock_value(stage.m2.user_readback, 2.5)
    whole = _run(RE, stage.read)
    view = _run(RE, stage.scalars.read)
    assert view == whole and view["s-m2"]["value"] == 2.5
    assert stage.scalars.covers(stage.m1) and stage.scalars.covers(stage.m2)
    assert stage.scalars._column_headers == stage._column_headers


@pytest.mark.parametrize("axis", ["axes", "scalars"])
def test_an_axis_that_would_shadow_the_noun_is_refused(axis: str) -> None:
    with pytest.raises(GeecsConfigurationError, match=repr(axis)):
        NativeMotor("S", "undulator:s:", [axis])
    # A name the settable rule already dodges binds with its underscore.
    stage = NativeMotor("S", "undulator:s:", ["name"])
    assert isinstance(stage.name_, NativeMotorAxis)


def test_a_kind_the_worker_does_not_build_refuses_to_open() -> None:
    """A schema kind without a noun here fails loudly at environment open, not with a traceback."""
    from types import SimpleNamespace

    registry = SimpleNamespace(devices={"U_Scope": SimpleNamespace(kind="scope")})
    with pytest.raises(GeecsConfigurationError, match="'scope'.*does not build"):
        GeecsNamespace(ROSTER, file_plugin_hosts=None, native_devices=registry)


def test_the_new_exception_is_exported() -> None:
    from geecs_bluesky import exceptions

    assert "GeecsMotorStoppedShortError" in exceptions.__all__


# ------------------------------------------------------------------ the move
def test_an_out_of_limit_target_is_refused_before_any_put(RE: RunEngine) -> None:
    axis = _axis(RE)
    with pytest.raises(MotorLimitsError, match="motor limits are -20.0"):
        _run(RE, lambda: axis.set(30.0))
    get_mock_put(axis.user_setpoint).assert_not_called()


def test_a_move_within_limits_completes_on_the_readback(RE: RunEngine) -> None:
    axis = _axis(RE)
    _run(RE, lambda: axis.set(5.0))
    get_mock_put(axis.user_setpoint).assert_called_once_with(5.0)
    assert _run(RE, axis.user_readback.get_value) == 5.0


def test_a_move_that_stops_short_fails_naming_the_switch(RE: RunEngine) -> None:
    """The record says done short of the target (a switch): the put "succeeded"."""
    axis = _axis(RE)

    def stop_at_the_switch(value: float, **_) -> None:
        set_mock_value(axis.user_readback, 18.0)
        set_mock_value(axis.high_limit_switch, 1)

    callback_on_mock_put(axis.user_setpoint, stop_at_the_switch)
    with pytest.raises(GeecsMotorStoppedShortError) as caught:
        _run(RE, lambda: axis.set(19.0))
    message = str(caught.value)
    assert "SimMotor1/m1" in message and "stopped at 18" in message
    assert "19 requested" in message and "high limit switch" in message


def test_a_readback_within_the_deadband_is_an_arrival(RE: RunEngine) -> None:
    axis = _axis(RE)
    callback_on_mock_put(
        axis.user_setpoint,
        lambda value, **_: set_mock_value(axis.user_readback, value - 0.009),
    )
    _run(RE, lambda: axis.set(5.0))  # 0.009 < RDBD 0.01


def test_the_confirm_reads_the_record_not_the_monitor_cache(RE: RunEngine) -> None:
    """``set`` observes the readback meanwhile, so a cached read could be the update before the last."""
    axis = _axis(RE)
    calls: list[dict] = []
    real = axis.user_readback.get_value

    async def spy(cached=None):
        calls.append({"cached": cached})
        return await real(cached=cached)

    axis.user_readback.get_value = spy  # type: ignore[method-assign]
    _run(RE, lambda: axis.set(5.0))
    assert {"cached": False} in calls  # the post-move confirm, fresh from the record


def test_an_operator_stop_is_reported_as_a_stop_not_a_fault(RE: RunEngine) -> None:
    """A stop lands mid-move: the record completes short; the status says stopped."""
    axis = _axis(RE)

    def stopped_mid_move(value: float, **_) -> None:
        # What ``StandardMovable.stop()`` does while the put is in flight.
        axis._set_success = False
        axis.movable_logic.stop_requested = True
        set_mock_value(axis.user_readback, 1.0)

    callback_on_mock_put(axis.user_setpoint, stopped_mid_move)
    with pytest.raises(RuntimeError, match="was stopped"):
        _run(RE, lambda: axis.set(5.0))
    # And the device's own stop() is what sets the flag.
    _run(RE, lambda: axis.stop())
    assert axis.movable_logic.stop_requested is True


def test_a_refused_put_read_as_success_is_caught_by_the_readback(RE) -> None:
    """#868: a refused put completes; the readback never moved and says so."""
    axis = _axis(RE)
    callback_on_mock_put(axis.user_setpoint, lambda value, **_: None)  # nothing moves
    with pytest.raises(GeecsMotorStoppedShortError, match="no limit switch set"):
        _run(RE, lambda: axis.set(5.0))


def test_mv_needs_nothing_and_a_refused_move_is_named(RE: RunEngine, caplog) -> None:
    """The stock ``mv`` stub moves an axis; its failure names the cause, logged at ERROR."""
    axis = _axis(RE)
    RE(_mv_named(axis, 3.0))
    assert _run(RE, axis.user_readback.get_value) == 3.0
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.devices.native_motor"):
        with pytest.raises(FailedStatus) as caught:
            RE(_mv_named(axis, 30.0))
    assert "motor limits are -20.0" in str(caught.value)
    (record,) = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and r.name == "geecs_bluesky.devices.native_motor"
    ]
    assert "undulator:simmotor1:m1.VAL" in record.getMessage()
    assert "limits are" in record.getMessage()
    get_mock_put(axis.user_setpoint).assert_called_once_with(3.0)


def test_a_stop_writes_the_records_stop_field(RE: RunEngine) -> None:
    axis = _axis(RE)
    _run(RE, lambda: axis.stop())
    get_mock_put(axis.motor_stop).assert_called_once_with(1)


def test_mv_through_a_run_engine_connects_the_axis_on_demand(RE: RunEngine) -> None:
    """A plan touching ``Stage.m1`` alone connects it (the child of a namespace member)."""
    from geecs_bluesky.preprocessors import connect_on_demand

    ns = GeecsNamespace(
        ROSTER, file_plugin_hosts=None, native_devices=_registry(SimMotor1=MOTOR)
    )
    axis = ns["SimMotor1"].m1
    RE.preprocessors.append(lambda plan: connect_on_demand(plan, mock=True))
    RE(bps.mv(axis, 1.5))
    assert _run(RE, axis.user_readback.get_value) == 1.5
