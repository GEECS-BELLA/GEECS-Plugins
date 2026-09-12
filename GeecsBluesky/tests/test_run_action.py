"""``run_action`` — a named action plan as a queue item over the device namespace.

Hermetic: the namespace is built from an explicit roster and connected with
mock backends; the resolver is a stub library.  What is pinned: the
namespace *is* the compiler's ``SettableFactory`` (a ``set`` lands on the
settable child's ``:SP``, a ``check`` reads the served signal), the
refusals are loud and typed, no run is opened, and the plan is registered
under its name beside the scan verbs.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

pytest.importorskip("aioca")

from geecs_schemas.action_plan import ActionPlan, ActionPlanLibrary  # noqa: E402
from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.exceptions import (  # noqa: E402
    ActionCheckFailedError,
    GeecsConfigurationError,
)
from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace  # noqa: E402
from geecs_bluesky.plan_names import GEECS_PLAN_NAMES  # noqa: E402
from geecs_bluesky.plans.action_compiler import (  # noqa: E402
    SettableFactory,
    run_action_plan,
)
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans  # noqa: E402
from geecs_bluesky.run_engine import make_run_engine  # noqa: E402
from tests.ca_mock_helpers import connect_mock  # noqa: E402
from tests.test_namespace import row  # noqa: E402

LIBRARY = ActionPlanLibrary.model_validate(
    {
        "schema_version": 1,
        "plans": {
            "close_shutter": {
                "steps": [
                    {
                        "do": "set",
                        "device": "U_GaiaSVEReader",
                        "variable": "InternalShutterA",
                        "value": 0,
                    }
                ]
            },
            "dump": {
                "steps": [
                    {"do": "run", "plan": "close_shutter"},
                    {
                        "do": "set",
                        "device": "U_148_PLC",
                        "variable": "DO.Ch9",
                        "value": "off",
                    },
                    {"do": "wait", "seconds": 0.01},
                    {
                        "do": "check",
                        "device": "U_148_PLC",
                        "variable": "DI.Ch17",
                        "expected": "off",
                    },
                ]
            },
            "set_readonly": {
                "steps": [
                    {
                        "do": "set",
                        "device": "U_148_PLC",
                        "variable": "DI.Ch17",
                        "value": "on",
                    }
                ]
            },
            "late_typo": {
                "steps": [
                    {"do": "run", "plan": "close_shutter"},
                    {
                        "do": "set",
                        "device": "U_148_PLC",
                        "variable": "DO.Ch9",
                        "value": "off",
                    },
                    {
                        "do": "check",
                        "device": "U_148_PLC",
                        "variable": "DI.Ch99",
                        "expected": "off",
                    },
                ]
            },
        },
    }
)


class StubResolver:
    """The two ``ConfigResolver`` methods ``run_action`` uses, over ``LIBRARY``."""

    def resolve_action_plan(self, name: str) -> ActionPlan:
        try:
            return LIBRARY.plans[name]
        except KeyError:
            raise GeecsConfigurationError(f"action plan {name!r} unknown") from None

    def action_plan_registry(self) -> dict[str, ActionPlan]:
        return dict(LIBRARY.plans)


ROSTER = DeviceRoster(
    experiment="TestExp",
    variables={
        "U_148_PLC": [
            row("DO.Ch9", settable=True, variabletype="choice", choices="on,off"),
            row("DI.Ch17", variabletype="choice", choices="on,off"),
        ],
        "U_GaiaSVEReader": [row("InternalShutterA", settable=True, tolerance=0.0)],
    },
    subscribed={"U_148_PLC": ["DI.Ch17"], "U_GaiaSVEReader": []},
)


@pytest.fixture
def RE():
    return make_run_engine(mock=True)


@pytest.fixture
def namespace(RE) -> GeecsNamespace:
    ns = GeecsNamespace(ROSTER, file_plugin_hosts=None)
    connect_mock(RE, *ns)  # mock backends the checks can drive
    return ns


def _value(RE, signal) -> Any:
    return asyncio.run_coroutine_threadsafe(signal.get_value(), RE._loop).result(10)


def test_the_namespace_is_a_settable_factory(namespace) -> None:
    assert isinstance(namespace, SettableFactory)
    assert (
        namespace.get_settable("U_148_PLC", "DO.Ch9") is namespace["U_148_PLC"].do_ch9
    )
    assert (
        namespace.get_readable("U_148_PLC", "DI.Ch17") is namespace["U_148_PLC"].di_ch17
    )
    with pytest.raises(GeecsConfigurationError, match="not settable"):
        namespace.get_settable("U_148_PLC", "DI.Ch17")
    with pytest.raises(GeecsConfigurationError, match="no served scalar variable"):
        namespace.get_readable("U_148_PLC", "Nope")
    with pytest.raises(GeecsConfigurationError, match="no device"):
        namespace.get_settable("U_Missing", "X")


def test_native_save_controls_are_refused_by_name() -> None:
    roster = DeviceRoster(
        experiment="TestExp",
        variables={
            "UC_Cam": [
                row("MeanCounts"),
                row("trigger", settable=True, choices="on,off"),
                row("save", settable=True, choices="on,off"),
                row("localsavingpath", settable=True, choices="path"),
            ]
        },
        subscribed={"UC_Cam": ["MeanCounts"]},
    )
    ns = GeecsNamespace(roster, file_plugin_hosts=None)
    assert ns["UC_Cam"].native_save
    with pytest.raises(GeecsConfigurationError, match="native saving"):
        ns.get_settable("UC_Cam", "localsavingpath")
    with pytest.raises(GeecsConfigurationError, match="native saving"):
        ns.get_settable("UC_Cam", "save")


def test_run_action_sets_checks_and_opens_no_run(RE, namespace) -> None:
    run_action = run_action_plan(StubResolver(), namespace)
    plc = namespace["U_148_PLC"]
    shutter = namespace["U_GaiaSVEReader"].internalshuttera
    # The check reads the served readback: make the mock say what the PLC would.
    set_mock_value(plc.di_ch17, "off")
    commands: list[str] = []
    RE.msg_hook = lambda msg: commands.append(msg.command)

    RE(run_action("dump"))

    assert _value(RE, plc.do_ch9._setpoint) == "off"
    assert _value(RE, shutter._setpoint) == 0.0
    assert "open_run" not in commands and "set" in commands and "read" in commands


def test_a_failed_check_aborts_the_item(RE, namespace) -> None:
    run_action = run_action_plan(StubResolver(), namespace)
    set_mock_value(namespace["U_148_PLC"].di_ch17, "on")
    with pytest.raises(ActionCheckFailedError):
        RE(run_action("dump"))


def test_refusals_are_loud(RE, namespace) -> None:
    run_action = run_action_plan(StubResolver(), namespace)
    with pytest.raises(GeecsConfigurationError, match="unknown"):
        RE(run_action("no_such_plan"))
    with pytest.raises(GeecsConfigurationError, match="not settable"):
        RE(run_action("set_readonly"))
    hermetic = run_action_plan(None, None)
    with pytest.raises(GeecsConfigurationError, match="no device namespace"):
        RE(hermetic("dump"))


def test_a_late_typo_fails_before_the_first_write(RE, namespace) -> None:
    """Every target is resolved and read before any set: nothing changes on a typo."""
    run_action = run_action_plan(StubResolver(), namespace)
    plc = namespace["U_148_PLC"]
    shutter = namespace["U_GaiaSVEReader"].internalshuttera
    set_mock_value(plc.do_ch9._setpoint, "on")
    set_mock_value(shutter._setpoint, 1.0)
    commands: list[str] = []
    RE.msg_hook = lambda msg: commands.append(msg.command)
    with pytest.raises(GeecsConfigurationError, match="DI.Ch99"):
        RE(run_action("late_typo"))
    assert "set" not in commands  # refused before the plan's first write
    assert _value(RE, plc.do_ch9._setpoint) == "on"
    assert _value(RE, shutter._setpoint) == 1.0


def test_run_action_is_registered_beside_the_scan_verbs() -> None:
    assert "run_action" in GEECS_PLAN_NAMES
    bound = bind_plans(TriggerProfiles({}), resolver=StubResolver(), settables=None)
    assert set(bound) == set(GEECS_PLAN_NAMES)
    plan = bound["run_action"]
    assert plan.__name__ == "run_action" and inspect.isgeneratorfunction(plan)
