"""The G-actions v2 pause supervisor (issue #552, PR-3).

Pins the pause-window protocol without hardware: safe-state policy per
acquisition mode, the three-way decision (execute / ignore / abort), the
restore-after-action / no-restore-on-abort rule, failed-step re-prompt,
the shot-control-device refusal (owner decision 11), and the
scan-ended-before-pause "not run" outcome.  A real asyncio loop stands in
for the paused RE's loop; the shot controller is a recording fake.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from geecs_bluesky.events import ActionDecisionRequest
from geecs_bluesky.pause_supervisor import PauseSupervisor, PendingAction
from geecs_schemas.action_plan import SetStep


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    while not loop.is_running():
        time.sleep(0.001)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


class _FakeSession:
    def __init__(self, loop) -> None:
        self.RE = type("RE", (), {"loop": loop})()
        self.disconnected: list = []

    def disconnect(self, factory) -> None:
        self.disconnected.append(factory)


class _RecordingSetter:
    def __init__(self, journal: list, name: str) -> None:
        self._journal = journal
        self._name = name

    def set(self, value):
        async def _run() -> None:
            self._journal.append((self._name, value))

        return _run()


class _FakeController:
    """Records driven states; hands out setters for state_setters()."""

    def __init__(self, journal: list, last_state: str | None = "SCAN") -> None:
        self._journal = journal
        self.last_state = last_state
        self._writes = {
            "OFF": [("U_DG645:Amp", "0.0")],
            "SCAN": [("U_DG645:Amp", "4.0")],
            "STANDBY": [("U_DG645:Amp", "0.5")],
        }

    def state_setters(self, state):
        name = getattr(state, "value", str(state))
        return [
            (_RecordingSetter(self._journal, target), value)
            for target, value in self._writes.get(name, [])
        ]


class _FakeFactory:
    def __init__(self, journal: list) -> None:
        self._journal = journal

    def get_settable(self, device, variable):
        return _RecordingSetter(self._journal, f"{device}:{variable}")

    def get_readable(self, device, variable):  # pragma: no cover
        raise NotImplementedError


def _pending(name="jet_on", journal=None, cleanup=None) -> PendingAction:
    journal = journal if journal is not None else []
    step = SetStep.model_validate(
        {"do": "set", "device": "U_Valve", "variable": "V", "value": 1}
    )
    return PendingAction(
        name=name,
        steps=[(step, None)],
        factory=_FakeFactory(journal),
        cleanup=cleanup or (lambda: None),
    )


def _answering(verdict: str):
    """An ``ask`` seam that answers *verdict* immediately."""

    def ask(request: ActionDecisionRequest) -> None:
        request.verdict[0] = verdict
        request.response_event.set()

    return ask


def test_free_run_drives_off_on_pause_and_restores_after_execute(loop) -> None:
    states: list = []
    steps: list = []
    controller = _FakeController(states, last_state="SCAN")
    session = _FakeSession(loop)
    sup = PauseSupervisor(
        acquisition="free_run",
        shot_controller=lambda: controller,
        ask=_answering("execute"),
    )
    sup.set_pending(_pending(journal=steps))
    verdict = sup.on_pause(session)
    assert verdict == "resume"
    # OFF driven on pause, action step ran, entry state (SCAN) re-asserted.
    assert states == [("U_DG645:Amp", "0.0"), ("U_DG645:Amp", "4.0")]
    assert steps == [("U_Valve:V", 1)]


def test_strict_drives_nothing_on_pause(loop) -> None:
    states: list = []
    controller = _FakeController(states, last_state="ARMED")
    sup = PauseSupervisor(
        acquisition="strict",
        shot_controller=lambda: controller,
        ask=_answering("ignore"),
    )
    sup.set_pending(_pending())
    assert sup.on_pause(_FakeSession(loop)) == "resume"
    assert states == []  # ARMED is already quiescent — nothing driven


def test_abort_skips_restore(loop) -> None:
    states: list = []
    controller = _FakeController(states, last_state="SCAN")
    sup = PauseSupervisor(
        acquisition="free_run",
        shot_controller=lambda: controller,
        ask=_answering("abort"),
    )
    sup.set_pending(_pending())
    assert sup.on_pause(_FakeSession(loop)) == "abort"
    # OFF driven on pause; NO restore write (the finalize chain owns it).
    assert states == [("U_DG645:Amp", "0.0")]


def test_ignore_restores_without_running_the_action(loop) -> None:
    states: list = []
    steps: list = []
    controller = _FakeController(states, last_state="SCAN")
    sup = PauseSupervisor(
        acquisition="free_run",
        shot_controller=lambda: controller,
        ask=_answering("ignore"),
    )
    sup.set_pending(_pending(journal=steps))
    assert sup.on_pause(_FakeSession(loop)) == "resume"
    assert steps == []  # action not run
    assert states == [("U_DG645:Amp", "0.0"), ("U_DG645:Amp", "4.0")]  # restored


def test_failed_step_reprompts_then_ignores(loop) -> None:
    """A check/step failure keeps the scan paused and re-prompts (decision 10)."""
    verdicts = iter(["execute", "ignore"])
    asked: list[str] = []

    def ask(request: ActionDecisionRequest) -> None:
        asked.append(request.message)
        request.verdict[0] = next(verdicts)
        request.response_event.set()

    class _FailingFactory(_FakeFactory):
        def get_settable(self, device, variable):
            class _Bad:
                def set(self, value):
                    async def _run():
                        raise RuntimeError("GEECS refused")

                    return _run()

            return _Bad()

    step = SetStep.model_validate(
        {"do": "set", "device": "U_Valve", "variable": "V", "value": 1}
    )
    pending = PendingAction(
        name="bad", steps=[(step, None)], factory=_FailingFactory([])
    )
    sup = PauseSupervisor(
        acquisition="strict",
        shot_controller=lambda: _FakeController([], last_state="ARMED"),
        ask=ask,
    )
    sup.set_pending(pending)
    assert sup.on_pause(_FakeSession(loop)) == "resume"
    assert len(asked) == 2  # execute (fails) → re-prompt → ignore
    assert "FAILED" in asked[1]


def test_no_consumer_defaults_to_ignore(loop) -> None:
    steps: list = []
    sup = PauseSupervisor(
        acquisition="strict",
        shot_controller=lambda: _FakeController([], last_state="ARMED"),
        ask=None,  # headless / no dialog consumer
    )
    sup.set_pending(_pending(journal=steps))
    assert sup.on_pause(_FakeSession(loop)) == "resume"
    assert steps == []  # defaulted to ignore — never executed


def test_abort_probe_while_parked_yields_abort(loop) -> None:
    aborting = {"v": False}

    def ask(request: ActionDecisionRequest) -> None:
        # Never answer — the abort probe must wake the park loop instead.
        aborting["v"] = True

    sup = PauseSupervisor(
        acquisition="strict",
        shot_controller=lambda: _FakeController([], last_state="ARMED"),
        ask=ask,
        should_abort=lambda: aborting["v"],
    )
    sup.set_pending(_pending())
    assert sup.on_pause(_FakeSession(loop)) == "abort"


def test_set_pending_is_single_slot() -> None:
    sup = PauseSupervisor(acquisition="free_run", shot_controller=lambda: None)
    assert sup.set_pending(_pending("a")) is True
    assert sup.set_pending(_pending("b")) is False
    taken = sup.take_unconsumed_pending()
    assert taken.name == "a"
    assert sup.take_unconsumed_pending() is None


class TestManualPause:
    """The bare operator Pause button: park until Resume or Stop, no action."""

    def test_manual_pause_parks_until_resume_then_restores(self, loop) -> None:
        states: list = []
        controller = _FakeController(states, last_state="SCAN")
        sup = PauseSupervisor(
            acquisition="free_run", shot_controller=lambda: controller
        )
        sup.arm_manual_pause()
        # Resume from another thread shortly after on_pause parks.
        import threading as _t

        _t.Timer(0.15, sup.request_resume).start()
        assert sup.on_pause(_FakeSession(loop)) == "resume"
        # OFF driven on pause, SCAN re-asserted on resume — no action ran.
        assert states == [("U_DG645:Amp", "0.0"), ("U_DG645:Amp", "4.0")]

    def test_manual_pause_strict_drives_nothing(self, loop) -> None:
        states: list = []
        sup = PauseSupervisor(
            acquisition="strict",
            shot_controller=lambda: _FakeController(states, last_state="ARMED"),
        )
        sup.arm_manual_pause()
        import threading as _t

        _t.Timer(0.1, sup.request_resume).start()
        assert sup.on_pause(_FakeSession(loop)) == "resume"
        assert states == []

    def test_manual_pause_abort_via_stop_probe_skips_restore(self, loop) -> None:
        states: list = []
        controller = _FakeController(states, last_state="SCAN")
        stopping = {"v": False}
        sup = PauseSupervisor(
            acquisition="free_run",
            shot_controller=lambda: controller,
            should_abort=lambda: stopping["v"],
        )
        sup.arm_manual_pause()
        import threading as _t

        _t.Timer(0.1, lambda: stopping.update(v=True)).start()
        assert sup.on_pause(_FakeSession(loop)) == "abort"
        assert states == [("U_DG645:Amp", "0.0")]  # OFF only; no restore


def test_manual_pause_with_staged_action_runs_the_action_and_cleans_up(loop):
    """Manual pause + a staged action (raced): the action wins, is decided,
    and its factory is cleaned up exactly once (no leak)."""
    cleaned: list = []
    states: list = []
    steps: list = []
    controller = _FakeController(states, last_state="SCAN")
    sup = PauseSupervisor(
        acquisition="free_run",
        shot_controller=lambda: controller,
        ask=_answering("execute"),
    )
    sup.set_pending(_pending(journal=steps, cleanup=lambda: cleaned.append(True)))
    sup.arm_manual_pause()  # both armed — action must win, not be dropped
    assert sup.on_pause(_FakeSession(loop)) == "resume"
    assert steps == [("U_Valve:V", 1)]  # the action ran
    assert cleaned == [True]  # factory cleaned exactly once (the leak fix)
