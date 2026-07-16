"""Direct ActionPlan step executor (G-actions v2, issue #552, PR-2).

Pins that :func:`execute_action_steps_directly` reproduces the compiled
plan's legacy step semantics over a running asyncio loop (the paused RE's
loop, simulated here by a loop in a daemon thread):

* set/wait/check execute strictly in flattened order, nested plans inlined;
* ``wait_for_execution: false`` schedules the put and moves on;
* ``check`` preserves the legacy comparison quirks and raises
  ``ActionCheckFailedError`` on mismatch;
* the abort probe interrupts a long ``wait`` promptly;
* a stuck blocking step trips the step timeout;
* a dead loop is refused loudly (never a silent hang).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from geecs_bluesky.exceptions import ActionCheckFailedError
from geecs_bluesky.plans.action_compiler import flatten_action_steps
from geecs_bluesky.plans.action_direct import (
    ActionExecutionAborted,
    ActionStepTimeoutError,
    execute_action_steps_directly,
)
from geecs_schemas import ActionPlan


@pytest.fixture()
def loop():
    """A running asyncio loop in a daemon thread (the paused RE's loop)."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    # Wait until the loop reports running so is_running() checks are stable.
    while not loop.is_running():
        time.sleep(0.001)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


class _FakeSettable:
    def __init__(self, name: str, journal: list, *, delay_s: float = 0.0) -> None:
        self._name = name
        self._journal = journal
        self._delay_s = delay_s

    def set(self, value: Any):
        async def _run() -> None:
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            self._journal.append(("set", self._name, value))

        return _run()


class _FakeReadable:
    def __init__(self, name: str, journal: list, value: Any) -> None:
        self._name = name
        self._journal = journal
        self._value = value

    async def get_value(self) -> Any:
        self._journal.append(("read", self._name))
        return self._value


class _FakeFactory:
    def __init__(self, journal: list, readbacks: dict[str, Any] | None = None):
        self.journal = journal
        self._readbacks = readbacks or {}
        self.set_delays: dict[str, float] = {}

    def get_settable(self, device: str, variable: str):
        name = f"{device}:{variable}"
        return _FakeSettable(name, self.journal, delay_s=self.set_delays.get(name, 0.0))

    def get_readable(self, device: str, variable: str):
        name = f"{device}:{variable}"
        return _FakeReadable(name, self.journal, self._readbacks.get(name))


def _plan(steps: list[dict]) -> ActionPlan:
    return ActionPlan.model_validate({"steps": steps})


def test_steps_execute_in_flattened_order_nested_inlined(loop) -> None:
    registry = {
        "inner": _plan([{"do": "set", "device": "U_B", "variable": "V", "value": 2}])
    }
    plan = _plan(
        [
            {"do": "set", "device": "U_A", "variable": "V", "value": 1},
            {"do": "run", "plan": "inner"},
            {"do": "wait", "seconds": 0.01},
            {"do": "set", "device": "U_C", "variable": "V", "value": 3},
        ]
    )
    factory = _FakeFactory(journal := [])
    steps = flatten_action_steps(plan, registry=registry)
    execute_action_steps_directly(steps, factory, loop)
    assert journal == [
        ("set", "U_A:V", 1),
        ("set", "U_B:V", 2),
        ("set", "U_C:V", 3),
    ]


def test_no_wait_set_schedules_and_moves_on(loop) -> None:
    plan = _plan(
        [
            {
                "do": "set",
                "device": "U_Slow",
                "variable": "V",
                "value": 1,
                "wait_for_execution": False,
            },
            {"do": "set", "device": "U_Fast", "variable": "V", "value": 2},
        ]
    )
    factory = _FakeFactory(journal := [])
    factory.set_delays["U_Slow:V"] = 0.5
    started = time.monotonic()
    execute_action_steps_directly(
        flatten_action_steps(plan, registry={}), factory, loop
    )
    elapsed = time.monotonic() - started
    # The slow put did not gate the sequence; the fast set finished first.
    assert elapsed < 0.4
    assert journal == [("set", "U_Fast:V", 2)]


def test_check_match_uses_legacy_numeric_string_quirk(loop) -> None:
    plan = _plan([{"do": "check", "device": "U_D", "variable": "V", "expected": 25}])
    factory = _FakeFactory([], readbacks={"U_D:V": "25"})  # string readback
    execute_action_steps_directly(
        flatten_action_steps(plan, registry={}), factory, loop
    )  # no raise — "25" floats to 25


def test_check_mismatch_raises(loop) -> None:
    plan = _plan([{"do": "check", "device": "U_D", "variable": "V", "expected": "on"}])
    factory = _FakeFactory([], readbacks={"U_D:V": "off"})
    with pytest.raises(ActionCheckFailedError):
        execute_action_steps_directly(
            flatten_action_steps(plan, registry={}), factory, loop
        )


def test_abort_interrupts_a_long_wait_promptly(loop) -> None:
    plan = _plan([{"do": "wait", "seconds": 30.0}])
    started = time.monotonic()
    with pytest.raises(ActionExecutionAborted):
        execute_action_steps_directly(
            flatten_action_steps(plan, registry={}),
            _FakeFactory([]),
            loop,
            should_abort=lambda: time.monotonic() - started > 0.1,
        )
    assert time.monotonic() - started < 2.0


def test_blocking_step_timeout_trips(loop) -> None:
    plan = _plan([{"do": "set", "device": "U_Stuck", "variable": "V", "value": 1}])
    factory = _FakeFactory([])
    factory.set_delays["U_Stuck:V"] = 60.0
    with pytest.raises(ActionStepTimeoutError, match="U_Stuck"):
        execute_action_steps_directly(
            flatten_action_steps(plan, registry={}),
            factory,
            loop,
            step_timeout_s=0.2,
        )


def test_dead_loop_refused_loudly() -> None:
    dead = asyncio.new_event_loop()
    dead.close()
    plan = _plan([{"do": "set", "device": "U_A", "variable": "V", "value": 1}])
    with pytest.raises(RuntimeError, match="not running"):
        execute_action_steps_directly(
            flatten_action_steps(plan, registry={}), _FakeFactory([]), dead
        )


class _FailingSettable:
    def __init__(self, *, delay_s: float = 0.0, exc: Exception | None = None) -> None:
        self._delay_s = delay_s
        self._exc = exc or RuntimeError("GEECS refused the set")

    def set(self, value):
        async def _run() -> None:
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            raise self._exc

        return _run()


def test_failed_no_wait_put_aborts_a_still_running_sequence(loop) -> None:
    """RunEngine FailedStatus parity: a mid-sequence put failure aborts."""
    plan = _plan(
        [
            {
                "do": "set",
                "device": "U_Bad",
                "variable": "V",
                "value": 1,
                "wait_for_execution": False,
            },
            {"do": "wait", "seconds": 2.0},
            {"do": "set", "device": "U_B", "variable": "V", "value": 2},
        ]
    )
    factory = _FakeFactory(journal := [])
    factory_get = factory.get_settable

    def get_settable(device, variable):
        if device == "U_Bad":
            return _FailingSettable(delay_s=0.05)
        return factory_get(device, variable)

    factory.get_settable = get_settable
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="GEECS refused the set"):
        execute_action_steps_directly(
            flatten_action_steps(plan, registry={}), factory, loop
        )
    # Aborted during the wait (well before its 2 s), and B was never written.
    assert time.monotonic() - started < 1.5
    assert journal == []


def test_failed_no_wait_put_after_completion_is_pardoned(loop, caplog) -> None:
    """RE end-of-plan parity: a failure landing after the sequence is logged."""
    import logging as _logging

    plan = _plan(
        [
            {
                "do": "set",
                "device": "U_Late",
                "variable": "V",
                "value": 1,
                "wait_for_execution": False,
            },
        ]
    )
    factory = _FakeFactory([])
    factory.get_settable = lambda d, v: _FailingSettable(delay_s=0.3)
    with caplog.at_level(_logging.WARNING, logger="geecs_bluesky.plans.action_direct"):
        execute_action_steps_directly(
            flatten_action_steps(plan, registry={}), factory, loop
        )  # returns normally — the failure has not landed yet
        time.sleep(0.6)
    assert "failed late" in caplog.text


def test_abort_interrupts_a_stuck_blocking_set_promptly(loop) -> None:
    plan = _plan([{"do": "set", "device": "U_Stuck", "variable": "V", "value": 1}])
    factory = _FakeFactory([])
    factory.set_delays["U_Stuck:V"] = 60.0
    started = time.monotonic()
    with pytest.raises(ActionExecutionAborted):
        execute_action_steps_directly(
            flatten_action_steps(plan, registry={}),
            factory,
            loop,
            should_abort=lambda: time.monotonic() - started > 0.1,
        )
    assert time.monotonic() - started < 2.0


def test_steps_own_timeout_error_is_never_relabelled(loop) -> None:
    """A step raising TimeoutError propagates as-is, not as budget expiry."""
    plan = _plan([{"do": "set", "device": "U_T", "variable": "V", "value": 1}])
    factory = _FakeFactory([])
    factory.get_settable = lambda d, v: _FailingSettable(
        exc=TimeoutError("the signal's own richer timeout")
    )
    with pytest.raises(TimeoutError, match="richer"):
        execute_action_steps_directly(
            flatten_action_steps(plan, registry={}), factory, loop
        )


def test_step_timeout_error_is_a_geecs_error() -> None:
    from geecs_bluesky.exceptions import GeecsError

    assert issubclass(ActionStepTimeoutError, GeecsError)
