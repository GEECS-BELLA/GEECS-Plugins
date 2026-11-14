"""``apply_setup`` — a named setup as one queue item over the device namespace.

Hermetic: the namespace is built from an explicit roster and connected with
mock backends; the resolver is a stub over a few setups. Pinned: the steps
land in order on the namespace's settables, nested setups inline, no run is
opened; every refusal before the first write writes nothing; a failure
mid-setup names the step and the writes already made; a changed setup is
refused against the digest its operator confirmed; a RunEngine pause or
abort passes through untouched.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

pytest.importorskip("aioca")

from bluesky.utils import RequestAbort
from geecs_schemas import Setup, flatten_setup, setup_digest
from ophyd_async.core import callback_on_mock_put, set_mock_value

from geecs_bluesky.exceptions import GeecsConfigurationError, SetupStepFailedError
from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace
from geecs_bluesky.plan_names import GEECS_PLAN_NAMES, NON_SCAN_PLAN_NAMES
from geecs_bluesky.plans.apply_setup import SettableFactory, apply_setup_plan
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans
from geecs_bluesky.run_engine import make_run_engine
from tests.ca_mock_helpers import connect_mock
from tests.test_namespace import row


def _setup(name: str, *steps: dict) -> Setup:
    return Setup.model_validate({"name": name, "steps": list(steps)})


SHUTTER = {"do": "set", "device": "U_GaiaSVEReader", "variable": "InternalShutterA", "value": 0}  # fmt: skip
PLC_OFF = {"do": "set", "device": "U_148_PLC", "variable": "DO.Ch9", "value": "off"}
CHECK_OFF = {"do": "check", "device": "U_148_PLC", "variable": "DI.Ch17", "expected": "off"}  # fmt: skip

SETUPS = {
    s.name: s
    for s in (
        _setup("close_shutter", SHUTTER),
        _setup(
            "dump",
            {"do": "apply", "setup": "close_shutter"},
            PLC_OFF,
            {"do": "wait", "seconds": 0.01},
            CHECK_OFF,
        ),
        _setup("set_readonly", {**PLC_OFF, "variable": "DI.Ch17"}),
        _setup(
            "late_typo",
            {"do": "apply", "setup": "close_shutter"},
            PLC_OFF,
            {**CHECK_OFF, "variable": "DI.Ch99"},
        ),
        _setup(
            "current",
            {
                "do": "set",
                "device": "U_GaiaSVEReader",
                "variable": "InternalShutterA",
                "value": 2.0,
            },
            {
                "do": "check",
                "device": "U_GaiaSVEReader",
                "variable": "InternalShutterA",
                "expected": 2.04,
                "tolerance": 0.05,
            },
        ),
        _setup("loops", {"do": "apply", "setup": "loops_back"}),
        _setup("loops_back", {"do": "apply", "setup": "loops"}),
    )
}


class StubResolver:
    """The resolver calls ``apply_setup`` makes, over :data:`SETUPS`."""

    def __init__(self, setups: dict[str, Setup] | None = None) -> None:
        self.setups = dict(SETUPS if setups is None else setups)

    def resolve_setup(self, name: str) -> tuple[Setup, dict[str, Setup]]:
        if name not in self.setups:
            raise GeecsConfigurationError(f"setup {name!r} unknown")
        return self.setups[name], dict(self.setups)

    def setup_repo_path(self, name: str) -> str:
        return f"scanner_configs/experiments/TestExp/setups/{name}.yaml"


def digest(name: str, setups: dict[str, Setup] | None = None) -> str:
    setups = SETUPS if setups is None else setups
    return setup_digest(flatten_setup(setups[name], setups=setups))


def md(name: str, **record: Any) -> dict:
    return {"geecs": {"operator": "Ada", "setup": {"name": name, **record}}}


ROSTER = DeviceRoster(
    experiment="TestExp",
    variables={
        "U_148_PLC": [
            row("DO.Ch9", settable=True, variabletype="choice", choices="on,off"),
            row("DI.Ch17", variabletype="choice", choices="on,off"),
        ],
        "U_GaiaSVEReader": [row("InternalShutterA", settable=True, tolerance=0.0)],
    },
    subscribed={"U_148_PLC": ["DI.Ch17"], "U_GaiaSVEReader": ["InternalShutterA"]},
)


@pytest.fixture
def RE():
    return make_run_engine(mock=True)


@pytest.fixture
def namespace(RE) -> GeecsNamespace:
    ns = GeecsNamespace(ROSTER, file_plugin_hosts=None)
    connect_mock(RE, *ns)
    return ns


@pytest.fixture
def commands(RE) -> list[str]:
    seen: list[str] = []
    RE.msg_hook = lambda msg: seen.append(msg.command)
    return seen


def _value(RE, signal) -> Any:
    return asyncio.run_coroutine_threadsafe(signal.get_value(), RE._loop).result(10)


def test_the_namespace_is_its_settable_factory(namespace) -> None:
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


def test_a_setup_sets_and_checks_in_order_and_opens_no_run(RE, namespace, commands) -> None:  # fmt: skip
    apply_setup = apply_setup_plan(StubResolver(), namespace)
    plc = namespace["U_148_PLC"]
    shutter = namespace["U_GaiaSVEReader"].internalshuttera
    set_mock_value(plc.di_ch17, "off")
    order: list[str] = []
    callback_on_mock_put(shutter._setpoint, lambda v, *_: order.append("shutter"))
    callback_on_mock_put(plc.do_ch9._setpoint, lambda v, *_: order.append("plc"))

    RE(apply_setup("dump", md=md("dump", digest=digest("dump"))))

    assert order == ["shutter", "plc"]  # the nested setup's step first, inlined
    assert _value(RE, plc.do_ch9._setpoint) == "off"
    assert _value(RE, shutter._setpoint) == 0.0
    assert "open_run" not in commands and "sleep" in commands


def test_a_numeric_check_matches_within_its_tolerance(RE, namespace) -> None:
    apply_setup = apply_setup_plan(StubResolver(), namespace)
    child = namespace.get_readable("U_GaiaSVEReader", "InternalShutterA")
    reading = getattr(child, child._readback_attr_name)
    set_mock_value(reading, 2.0)
    RE(apply_setup("current"))  # reads 2.0, expects 2.04 ± 0.05
    set_mock_value(reading, 1.9)
    with pytest.raises(SetupStepFailedError, match="within 0.05"):
        RE(apply_setup("current"))


def test_a_failed_check_stops_the_setup_naming_the_writes_made(RE, namespace) -> None:
    apply_setup = apply_setup_plan(StubResolver(), namespace)
    set_mock_value(namespace["U_148_PLC"].di_ch17, "on")
    with pytest.raises(SetupStepFailedError) as excinfo:
        RE(apply_setup("dump"))
    err = excinfo.value
    assert (err.index, err.total, err.written) == (4, 4, (1, 2))
    assert "read 'on', expected 'off'" in str(err)
    assert "steps 1, 2 took effect and are not undone" in str(err)


def test_a_device_refusing_mid_setup_stops_it_there(RE, namespace, commands) -> None:
    apply_setup = apply_setup_plan(StubResolver(), namespace)
    plc = namespace["U_148_PLC"]
    set_mock_value(plc.di_ch17, "off")

    def refuse(value, *_):
        raise RuntimeError("the PLC refused the write")

    callback_on_mock_put(plc.do_ch9._setpoint, refuse)
    with pytest.raises(SetupStepFailedError) as excinfo:
        RE(apply_setup("dump"))
    err = excinfo.value
    assert (err.index, err.written) == (2, (1,))
    assert "set U_148_PLC:DO.Ch9 = 'off'" in str(err) and "refused" in str(err)
    assert "sleep" not in commands  # nothing after the refused step ran


def test_refusals_before_the_first_write(RE, namespace, commands) -> None:
    apply_setup = apply_setup_plan(StubResolver(), namespace)
    with pytest.raises(GeecsConfigurationError, match="unknown"):
        RE(apply_setup("no_such_setup"))
    with pytest.raises(GeecsConfigurationError, match="loop"):
        RE(apply_setup("loops"))
    with pytest.raises(SetupStepFailedError, match="not settable") as excinfo:
        RE(apply_setup("set_readonly"))
    assert excinfo.value.written == ()
    assert "nothing was written" in str(excinfo.value)
    with pytest.raises(GeecsConfigurationError, match="no device namespace"):
        RE(apply_setup_plan(None, None)("dump"))
    assert "set" not in commands


def test_a_late_typo_fails_before_the_first_write(RE, namespace, commands) -> None:
    apply_setup = apply_setup_plan(StubResolver(), namespace)
    plc = namespace["U_148_PLC"]
    shutter = namespace["U_GaiaSVEReader"].internalshuttera
    set_mock_value(plc.do_ch9._setpoint, "on")
    set_mock_value(shutter._setpoint, 1.0)
    with pytest.raises(SetupStepFailedError, match="DI.Ch99") as excinfo:
        RE(apply_setup("late_typo"))
    assert excinfo.value.index == 3 and excinfo.value.written == ()
    assert "set" not in commands
    assert _value(RE, plc.do_ch9._setpoint) == "on"
    assert _value(RE, shutter._setpoint) == 1.0


def test_what_was_confirmed_is_what_runs(RE, namespace, commands) -> None:
    """A setup edited (here: a setup it applies) after the operator confirmed it is refused."""
    confirmed = digest("dump")
    edited = dict(SETUPS)
    edited["close_shutter"] = _setup("close_shutter", {**SHUTTER, "value": 1})
    apply_setup = apply_setup_plan(StubResolver(edited), namespace)
    with pytest.raises(GeecsConfigurationError, match="changed since it was confirmed"):
        RE(apply_setup("dump", md=md("dump", digest=confirmed)))
    with pytest.raises(GeecsConfigurationError, match="names the setup 'other'"):
        RE(apply_setup("dump", md=md("other", digest=digest("dump", edited))))
    assert "set" not in commands


def test_a_runengine_abort_passes_through_untouched(namespace) -> None:
    """A pause, stop or abort is the RunEngine's: never re-worded as a step's failure."""
    apply_setup = apply_setup_plan(StubResolver(), namespace)
    gen = apply_setup("dump")
    msg = next(gen)
    while msg.command != "set":
        msg = gen.send(None)
    with pytest.raises(RequestAbort):
        gen.throw(RequestAbort())


def test_apply_setup_is_registered_beside_the_scan_verbs() -> None:
    assert "apply_setup" in GEECS_PLAN_NAMES and "apply_setup" in NON_SCAN_PLAN_NAMES
    bound = bind_plans(TriggerProfiles({}), resolver=StubResolver(), settables=None)
    plan = bound["apply_setup"]
    assert plan.__name__ == "apply_setup" and inspect.isgeneratorfunction(plan)
    params = inspect.signature(plan).parameters
    assert list(params) == ["name", "md"]
    assert params["md"].kind is inspect.Parameter.KEYWORD_ONLY
    # Resolved objects, never postponed strings (the manager evaluates them).
    assert all(not isinstance(p.annotation, str) for p in params.values())
