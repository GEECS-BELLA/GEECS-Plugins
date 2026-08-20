"""Tests for the ActionPlan → Bluesky plan-stub compiler (hermetic).

No gateway, no network: signals come from an in-memory
:class:`MockSignalFactory` built on ophyd-async soft signals, connected into
a real RunEngine's event loop — the compiled plans run through a real RE, so
message semantics (set-and-wait vs fire-and-forget, sleep, read) are the
production ones.

The fidelity pins mirror the legacy ActionManager
(``geecs_scanner.engine.action_manager``) step by step:

- ``set`` — legacy ``device.set(variable, value, sync=wait_for_execution)``.
- ``wait`` — legacy ``time.sleep(seconds)``.
- ``check`` — legacy plain ``==`` after the device layer floated
  numeric-looking strings (``GeecsDevice.interpret_value``).
- ``run`` — legacy nested ``execute_action`` recursion (plus a cycle guard
  legacy never had).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from bluesky import RunEngine
from bluesky.preprocessors import msg_mutator
from bluesky.utils import Msg
from ophyd_async.core import soft_signal_rw

from geecs_schemas.action_plan import ActionPlan
from geecs_schemas.convert import convert_action_library

from geecs_bluesky.exceptions import (
    ActionCheckFailedError,
    ActionPlanCycleError,
    ActionPlanNotFoundError,
)
from geecs_bluesky.plans.action_compiler import (
    SettableFactory,
    compile_action_plan,
    values_match,
)

CORPUS_ACTIONS = (
    Path(__file__).resolve().parents[2]
    / "GEECS-Schemas"
    / "tests"
    / "fixtures"
    / "actions"
    / "actions_undulator.yaml"
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class MockSignalFactory:
    """In-memory SettableFactory: one soft signal per (device, variable).

    ``get_settable`` and ``get_readable`` return the *same* signal for a
    key, so a ``set`` naturally drives the readback a later ``check`` sees —
    the mock analogue of GEECS convergence.
    """

    def __init__(self, run_engine: RunEngine) -> None:
        self._re = run_engine
        self._signals: dict[tuple[str, str], Any] = {}

    def add(
        self,
        device: str,
        variable: str,
        *,
        datatype: type = float,
        initial: Any = 0.0,
    ) -> Any:
        """Create, connect, and register a soft signal for (device, variable)."""
        signal = soft_signal_rw(datatype, initial, name=f"{device}-{variable}")
        asyncio.run_coroutine_threadsafe(signal.connect(), self._re._loop).result(
            timeout=10.0
        )
        self._signals[(device, variable)] = signal
        return signal

    def get_settable(self, device: str, variable: str) -> Any:
        return self._signals[(device, variable)]

    def get_readable(self, device: str, variable: str) -> Any:
        return self._signals[(device, variable)]

    def value_of(self, device: str, variable: str) -> Any:
        """Read a signal's current value directly (test assertion helper)."""
        signal = self._signals[(device, variable)]
        return asyncio.run_coroutine_threadsafe(
            signal.get_value(), self._re._loop
        ).result(timeout=10.0)


@pytest.fixture()
def re_and_factory() -> tuple[RunEngine, MockSignalFactory]:
    run_engine = RunEngine()
    return run_engine, MockSignalFactory(run_engine)


def _collect_messages(plan) -> list[Msg]:
    """Drive a plan without a RunEngine, collecting its messages.

    Only valid for plans that need no responses (set / wait / run); ``check``
    steps must run through the RE.
    """
    messages: list[Msg] = []
    try:
        message = plan.send(None)
        while True:
            messages.append(message)
            message = plan.send(None)
    except StopIteration:
        pass
    return messages


def _instant_sleeps(plan, recorded: list[float]):
    """Record each sleep's requested duration and replace it with zero."""

    def _mutate(msg: Msg) -> Msg:
        if msg.command == "sleep":
            recorded.append(msg.args[0])
            return msg._replace(args=(0.0,))
        return msg

    return msg_mutator(plan, _mutate)


def _plan(steps: list[dict]) -> ActionPlan:
    return ActionPlan.model_validate({"steps": steps})


# ---------------------------------------------------------------------------
# set — wait_for_execution semantics (legacy sync=True/False)
# ---------------------------------------------------------------------------


def test_set_with_wait_emits_set_then_wait(re_and_factory) -> None:
    """wait_for_execution=True (legacy default sync=True) → set-and-wait."""
    run_engine, factory = re_and_factory
    signal = factory.add("U_S1H", "Current")
    plan = _plan(
        [{"do": "set", "device": "U_S1H", "variable": "Current", "value": 5.0}]
    )

    messages = _collect_messages(
        compile_action_plan(plan, registry={}, settables=factory)
    )

    assert [m.command for m in messages] == ["set", "wait"]
    set_msg = messages[0]
    assert set_msg.obj is signal
    assert set_msg.args == (5.0,)
    # The wait targets the set's completion group — set-and-wait, not a sleep.
    assert messages[1].kwargs["group"] == set_msg.kwargs["group"]
    assert set_msg.kwargs["group"] is not None


def test_set_without_wait_is_fire_and_forget(re_and_factory) -> None:
    """wait_for_execution=False (legacy sync=False) → no wait message at all."""
    run_engine, factory = re_and_factory
    signal = factory.add("U_S1H", "Current")
    plan = _plan(
        [
            {
                "do": "set",
                "device": "U_S1H",
                "variable": "Current",
                "value": 2.5,
                "wait_for_execution": False,
            }
        ]
    )

    messages = _collect_messages(
        compile_action_plan(plan, registry={}, settables=factory)
    )

    assert [m.command for m in messages] == ["set"]
    assert messages[0].obj is signal
    assert messages[0].args == (2.5,)


def test_set_through_run_engine_updates_signals(re_and_factory) -> None:
    """Both wait semantics actually land the value on the mock signal."""
    run_engine, factory = re_and_factory
    factory.add("U_S1H", "Current")
    factory.add("U_148_PLC", "DO.Ch9", datatype=str, initial="")
    plan = _plan(
        [
            {"do": "set", "device": "U_S1H", "variable": "Current", "value": 7.25},
            {
                "do": "set",
                "device": "U_148_PLC",
                "variable": "DO.Ch9",
                "value": "on",
                "wait_for_execution": False,
            },
        ]
    )

    run_engine(compile_action_plan(plan, registry={}, settables=factory))

    assert factory.value_of("U_S1H", "Current") == 7.25
    assert factory.value_of("U_148_PLC", "DO.Ch9") == "on"


# ---------------------------------------------------------------------------
# wait — bps.sleep, asserted at message level (no wall clock)
# ---------------------------------------------------------------------------


def test_wait_emits_sleep_message_with_exact_duration(re_and_factory) -> None:
    run_engine, factory = re_and_factory
    plan = _plan([{"do": "wait", "seconds": 3.0}])

    messages = _collect_messages(
        compile_action_plan(plan, registry={}, settables=factory)
    )

    assert [m.command for m in messages] == ["sleep"]
    assert messages[0].args == (3.0,)


# ---------------------------------------------------------------------------
# check — legacy get-and-compare semantics
# ---------------------------------------------------------------------------


def test_check_passes_on_matching_string(re_and_factory) -> None:
    run_engine, factory = re_and_factory
    factory.add("U_148_PLC", "DI.Ch17", datatype=str, initial="off")
    plan = _plan(
        [
            {
                "do": "check",
                "device": "U_148_PLC",
                "variable": "DI.Ch17",
                "expected": "off",
            }
        ]
    )

    run_engine(compile_action_plan(plan, registry={}, settables=factory))


def test_check_passes_with_legacy_float_coercion(re_and_factory) -> None:
    """Numeric expected vs numeric-looking string readback matches.

    Legacy parity: ``GeecsDevice.interpret_value`` floated the wire string
    before the ActionManager's ``==``, so expected 25.0 matched a device
    reporting the string "25".
    """
    run_engine, factory = re_and_factory
    factory.add("U_HP_Daq", "AnalogOutput.Channel 1", datatype=str, initial="25")
    plan = _plan(
        [
            {
                "do": "check",
                "device": "U_HP_Daq",
                "variable": "AnalogOutput.Channel 1",
                "expected": 25.0,
            }
        ]
    )

    run_engine(compile_action_plan(plan, registry={}, settables=factory))


def test_check_failure_names_device_variable_expected_actual(re_and_factory) -> None:
    run_engine, factory = re_and_factory
    factory.add("U_148_PLC", "DI.Ch17", datatype=str, initial="on")
    plan = _plan(
        [
            {
                "do": "check",
                "device": "U_148_PLC",
                "variable": "DI.Ch17",
                "expected": "off",
            }
        ]
    )

    with pytest.raises(ActionCheckFailedError) as excinfo:
        run_engine(compile_action_plan(plan, registry={}, settables=factory))

    err = excinfo.value
    assert err.device == "U_148_PLC"
    assert err.variable == "DI.Ch17"
    assert err.expected == "off"
    assert err.actual == "on"
    message = str(err)
    for fragment in ("U_148_PLC", "DI.Ch17", "'off'", "'on'"):
        assert fragment in message


def test_check_legacy_quirk_string_expected_vs_numeric_readback_fails(
    re_and_factory,
) -> None:
    """A *string* expected value never matches a numeric-looking readback.

    Pins the legacy quirk verbatim: interpret_value floated "25" to 25.0 on
    the actual side only, and ``25.0 == "25"`` is False — so legacy aborted,
    and so do we.
    """
    run_engine, factory = re_and_factory
    factory.add("U_Dev", "Var", datatype=str, initial="25")
    plan = _plan(
        [{"do": "check", "device": "U_Dev", "variable": "Var", "expected": "25"}]
    )

    with pytest.raises(ActionCheckFailedError):
        run_engine(compile_action_plan(plan, registry={}, settables=factory))


def test_values_match_rules() -> None:
    """The comparison helper reproduces legacy interpret_value + '=='."""
    assert values_match("off", "off")
    assert values_match(25.0, "25")  # actual floated, legacy-style
    assert values_match(25, 25.0)  # plain Python numeric equality
    assert values_match(0, 0.0)
    assert not values_match("25", "25")  # legacy quirk: actual floats first
    assert not values_match("on", "off")
    assert not values_match(1.0, "on")  # unparseable string stays a string


# ---------------------------------------------------------------------------
# run — nested plans, missing names, cycles
# ---------------------------------------------------------------------------


def test_nested_plan_executes_child_steps(re_and_factory) -> None:
    run_engine, factory = re_and_factory
    factory.add("U_S1H", "Current")
    child = _plan(
        [{"do": "set", "device": "U_S1H", "variable": "Current", "value": 1.5}]
    )
    parent = _plan([{"do": "run", "plan": "child"}])

    run_engine(
        compile_action_plan(parent, registry={"child": child}, settables=factory)
    )

    assert factory.value_of("U_S1H", "Current") == 1.5


def test_two_level_nesting_runs_in_order(re_and_factory) -> None:
    """A runs B runs C; every level's own steps execute, in plan order."""
    run_engine, factory = re_and_factory
    for variable in ("VarA", "VarB", "VarC"):
        factory.add("U_Dev", variable)
    plan_c = _plan([{"do": "set", "device": "U_Dev", "variable": "VarC", "value": 3}])
    plan_b = _plan(
        [
            {"do": "run", "plan": "C"},
            {"do": "set", "device": "U_Dev", "variable": "VarB", "value": 2},
        ]
    )
    plan_a = _plan(
        [
            {"do": "run", "plan": "B"},
            {"do": "set", "device": "U_Dev", "variable": "VarA", "value": 1},
        ]
    )
    registry = {"A": plan_a, "B": plan_b, "C": plan_c}

    set_order: list[str] = []
    run_engine.msg_hook = lambda msg: (
        set_order.append(msg.obj.name) if msg.command == "set" else None
    )
    run_engine(compile_action_plan(plan_a, registry=registry, settables=factory))

    assert set_order == ["U_Dev-VarC", "U_Dev-VarB", "U_Dev-VarA"]
    assert factory.value_of("U_Dev", "VarA") == 1.0
    assert factory.value_of("U_Dev", "VarB") == 2.0
    assert factory.value_of("U_Dev", "VarC") == 3.0


def test_missing_nested_plan_raises_not_found(re_and_factory) -> None:
    run_engine, factory = re_and_factory
    parent = _plan([{"do": "run", "plan": "ghost"}])

    with pytest.raises(ActionPlanNotFoundError) as excinfo:
        run_engine(
            compile_action_plan(parent, registry={"other": parent}, settables=factory)
        )

    assert excinfo.value.plan_name == "ghost"
    assert "ghost" in str(excinfo.value)
    assert "other" in str(excinfo.value)  # names the known plans


def test_cycle_between_two_plans_is_detected(re_and_factory) -> None:
    """A → B → A raises instead of recursing forever (legacy had no guard)."""
    run_engine, factory = re_and_factory
    plan_a = _plan([{"do": "run", "plan": "B"}])
    plan_b = _plan([{"do": "run", "plan": "A"}])
    registry = {"A": plan_a, "B": plan_b}

    with pytest.raises(ActionPlanCycleError) as excinfo:
        run_engine(compile_action_plan(plan_a, registry=registry, settables=factory))

    assert excinfo.value.chain == ["B", "A", "B"]
    assert "loop" in str(excinfo.value)


def test_direct_self_cycle_is_detected(re_and_factory) -> None:
    run_engine, factory = re_and_factory
    plan_a = _plan([{"do": "run", "plan": "A"}])

    with pytest.raises(ActionPlanCycleError):
        run_engine(
            compile_action_plan(plan_a, registry={"A": plan_a}, settables=factory)
        )


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_empty_plan_is_a_noop(re_and_factory) -> None:
    """Zero steps → zero messages (schema forbids it; compiler is safe anyway)."""
    run_engine, factory = re_and_factory
    empty = ActionPlan.model_construct(steps=[])

    messages = _collect_messages(
        compile_action_plan(empty, registry={}, settables=factory)
    )
    assert messages == []

    run_engine(compile_action_plan(empty, registry={}, settables=factory))


def test_mock_factory_satisfies_protocol(re_and_factory) -> None:
    _run_engine, factory = re_and_factory
    assert isinstance(factory, SettableFactory)


# ---------------------------------------------------------------------------
# Converted-from-legacy corpus — the end-to-end fidelity pin
# ---------------------------------------------------------------------------


def test_corpus_amp4_dump_hp_end_to_end(re_and_factory) -> None:
    """Convert the real legacy actions.yaml, compile, and run against mocks.

    ``Amp4_DUMP_HP`` exercises every step type: nested ``run``
    (close/open_gaia_internal_shutters), ``set`` with the legacy default
    wait, ``wait``, and ``check`` against both a set-driven readback (the
    shutters) and an independent input (the PLC interlock).
    """
    run_engine, factory = re_and_factory
    library = convert_action_library(CORPUS_ACTIONS)
    plan = library.plans["Amp4_DUMP_HP"]

    # Shutters: set and check hit the same variable, so the mock signal
    # naturally plays GEECS convergence. The PLC readback is an independent
    # digital input, pre-set to the value the plan expects.
    factory.add("U_GaiaSVEReader", "InternalShutterA")
    factory.add("U_GaiaSVEReader", "InternalShutterB")
    factory.add("U_148_PLC", "DO.Ch9", datatype=str, initial="")
    factory.add("U_148_PLC", "DI.Ch17", datatype=str, initial="off")

    set_messages: list[tuple[str, Any]] = []
    run_engine.msg_hook = lambda msg: (
        set_messages.append((msg.obj.name, msg.args[0]))
        if msg.command == "set"
        else None
    )

    sleeps: list[float] = []
    run_engine(
        _instant_sleeps(
            compile_action_plan(plan, registry=library.plans, settables=factory),
            sleeps,
        )
    )

    # close shutters (A=0, B=0) → dump beam (DO.Ch9 off) → reopen (A=1, B=1)
    assert set_messages == [
        ("U_GaiaSVEReader-InternalShutterA", 0),
        ("U_GaiaSVEReader-InternalShutterB", 0),
        ("U_148_PLC-DO.Ch9", "off"),
        ("U_GaiaSVEReader-InternalShutterA", 1),
        ("U_GaiaSVEReader-InternalShutterB", 1),
    ]
    # Legacy wait durations preserved verbatim: three 3 s settles per shutter
    # plan, plus the top-level 3 s before the interlock check.
    assert sleeps == [3.0] * 7
    # Final hardware state: shutters reopened, dump commanded off.
    assert factory.value_of("U_GaiaSVEReader", "InternalShutterA") == 1.0
    assert factory.value_of("U_GaiaSVEReader", "InternalShutterB") == 1.0
    assert factory.value_of("U_148_PLC", "DO.Ch9") == "off"


def test_corpus_check_mismatch_aborts_mid_plan(re_and_factory) -> None:
    """The same corpus plan aborts at the interlock check when it disagrees.

    Legacy behaviour: a GetStep mismatch stopped the action (headless
    auto-abort). The reopen steps after the failed check must never run.
    """
    run_engine, factory = re_and_factory
    library = convert_action_library(CORPUS_ACTIONS)
    plan = library.plans["Amp4_DUMP_HP"]

    factory.add("U_GaiaSVEReader", "InternalShutterA")
    factory.add("U_GaiaSVEReader", "InternalShutterB")
    factory.add("U_148_PLC", "DO.Ch9", datatype=str, initial="")
    # Interlock disagrees with the plan's expectation of "off".
    factory.add("U_148_PLC", "DI.Ch17", datatype=str, initial="on")

    with pytest.raises(ActionCheckFailedError) as excinfo:
        run_engine(
            _instant_sleeps(
                compile_action_plan(plan, registry=library.plans, settables=factory),
                [],
            )
        )

    assert excinfo.value.device == "U_148_PLC"
    assert excinfo.value.variable == "DI.Ch17"
    # The reopen (open_gaia_internal_shutters) never ran: shutters still 0.
    assert factory.value_of("U_GaiaSVEReader", "InternalShutterA") == 0.0
    assert factory.value_of("U_GaiaSVEReader", "InternalShutterB") == 0.0
