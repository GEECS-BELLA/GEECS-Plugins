"""Sweep lifecycle and manager JSON contracts, with no hardware."""

import pytest
from bluesky import RunEngine, plan_stubs as bps
from bluesky.utils import FailedStatus, RunEngineInterrupted
from geecs_schemas import Sweep
from ophyd_async.core import AsyncStatus, callback_on_mock_put, set_mock_value

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plans.registry import bind_plans
from geecs_bluesky.plans.sweep import sweep_plan
from tests.ca_mock_helpers import DocCollector, connect_mock, follow_setpoint
from tests import test_plan_registry as registry_tests
from tests.test_plan_registry import Magnet

RE, box, profiles = registry_tests.RE, registry_tests.box, registry_tests.profiles


def payload(relative=True):
    return {
        "trajectory": {
            "kind": "axes",
            "axes": [
                {
                    "kind": "list",
                    "axis": "X.current",
                    "positions": [1, 3],
                    "relative": relative,
                },
                {"kind": "list", "axis": "Y.current", "positions": [2, 4]},
            ],
        }
    }


def setup_axes(engine):
    x, y = Magnet("x"), Magnet("y")
    connect_mock(engine, x, y)
    follow_setpoint(x.current)
    follow_setpoint(y.current)
    engine(bps.mv(x.current, 10, y.current, 20))
    return x, y


@pytest.mark.parametrize("hinted", [False, True])
def test_mixed_frames_restore_before_close_and_unstage(monkeypatch, hinted):
    engine = RunEngine()
    x, y = setup_axes(engine)
    if hinted:
        monkeypatch.setattr(
            type(x.current), "hints", property(lambda m: {"fields": [m.position.name]})
        )
    order = []
    original_stage, original_unstage = x.current.stage, x.current.unstage

    @AsyncStatus.wrap
    async def stage():
        await original_stage()
        # Model the coordinate-frame reset of a relative pseudo at stage.
        set_mock_value(x.current.position, 0)
        order.append("stage")

    @AsyncStatus.wrap
    async def unstage():
        assert await x.current.position.get_value() == 0
        order.append("unstage")
        await original_unstage()

    x.stage, x.unstage = stage, unstage
    col = DocCollector()
    engine.subscribe(col)
    engine.subscribe(lambda name, doc: order.append("stop") if name == "stop" else None)
    engine(sweep_plan({"X": x, "Y": y})([], sweep=payload()))
    events = col.primary_events()
    assert [e["data"]["x-current-position"] for e in events] == [1, 3]
    assert [e["data"]["y-current-position"] for e in events] == [2, 4]
    assert order == ["stage", "stop", "unstage"]
    start = col.docs["start"][0]
    assert start["motors"] == ["x-current", "y-current"]
    if hinted:
        assert start["hints"]["dimensions"] == [
            (["x-current-position", "y-current-position"], "primary")
        ]
    else:
        assert start["hints"] == {}
    assert Sweep.model_validate(start["sweep"]).axis_references()[0].relative


def test_abort_restores_only_relative_axes():
    engine = RunEngine()
    x, y = setup_axes(engine)
    col = DocCollector()
    engine.subscribe(col)

    def pause_after_step(detectors, step, pos_cache):
        yield from bps.one_nd_step(detectors, step, pos_cache)
        yield from bps.pause()

    with pytest.raises(RunEngineInterrupted):
        engine(
            sweep_plan({"X": x, "Y": y})([], sweep=payload(), per_step=pause_after_step)
        )
    engine.abort()
    positions = {}

    def read():
        positions["x"] = yield from bps.rd(x.current)
        positions["y"] = yield from bps.rd(y.current)

    engine(read())
    assert positions == {"x": 10, "y": 2}
    assert col.docs["stop"][0]["exit_status"] == "abort"


def test_restore_failure_fails_run():
    engine = RunEngine()
    x, y = setup_axes(engine)

    def put(value, **kwargs):
        if value == 10:
            raise RuntimeError("restore refused")
        set_mock_value(x.current.position, value)

    callback_on_mock_put(x.current._setpoint, put)
    col = DocCollector()
    engine.subscribe(col)
    with pytest.raises(FailedStatus):
        engine(sweep_plan({"X": x, "Y": y})([], sweep=payload()))
    assert col.docs["stop"][0]["exit_status"] == "fail"
    assert "restore refused" in col.docs["stop"][0]["reason"]


@pytest.mark.parametrize(
    "bad", [None, {"trajectory": {"kind": "axes", "axes": []}}, payload()]
)
def test_invalid_payload_or_unknown_binding_never_moves_box(RE, box, profiles, bad):
    with pytest.raises((ValueError, GeecsConfigurationError)):
        RE(bind_plans(profiles, settables={})["sweep"]([], sweep=bad))
    assert box.puts == []


def test_manager_preserves_nested_axis_strings(RE, box, profiles):
    from bluesky_queueserver.manager.profile_ops import (
        existing_plans_and_devices_from_nspace,
        prepare_plan,
    )

    x, y = setup_axes(RE)
    namespace = {"X": x, "Y": y}
    namespace.update(bind_plans(profiles, settables=namespace))
    plans, devices, pns, dns = existing_plans_and_devices_from_nspace(nspace=namespace)
    result = prepare_plan(
        {
            "name": "sweep",
            "args": [["X.current"]],
            "kwargs": {"sweep": payload()},
            "user_group": "operator",
        },
        plans_in_nspace=pns,
        devices_in_nspace=dns,
        allowed_plans={"operator": plans},
        allowed_devices={"operator": devices},
        nspace=namespace,
    )
    assert result["args"][0] == [x.current]
    assert result["kwargs"]["sweep"] == payload()
