"""The status carries the verbs' own verdicts: what the page enables is what the verbs allow.

Each case reads ``GET /api/status``'s blockers and run word over a scripted
manager, then asks the verb itself: the blocker is the verb's refusal
message, byte for byte, or ``None`` exactly when the verb goes through.
The scripted states are ones the manager reports: its run-engine state is
``None`` only while the worker environment is closed or opening.
"""

from __future__ import annotations

from typing import Any

import pytest

from geecs_scanner.service import ScannerService
from geecs_scanner.service.demo import DemoResolver, demo_preflight
from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.models import MoveIn, ReopenIn, VerbIn
from geecs_scanner.service.verdicts import IDLE_ONLY_ITEMS

#: A stopped, ready queue with items waiting: every verb's starting point.
READY = dict(connected=True, re_state="idle", manager_state="idle", worker_exists=True, worker_environment_state="idle", items_in_queue=2)  # fmt: skip
#: The environment as a manager restart leaves it, before the readiness unit opens it.
CLOSED = dict(worker_exists=False, worker_environment_state="closed", re_state=None)
#: The environment while it opens: it exists, its run engine does not answer yet.
OPENING = dict(worker_environment_state="initializing", re_state=None)
MOVE = MoveIn(variable="U_S1H:current", value=1.0)


class _Manager:
    """A scripted manager: one status, a readiness over a fixed plan list, verbs recorded."""

    def __init__(self, plans: tuple[str, ...] = ("count", "mv"), **status: Any) -> None:
        from geecs_bluesky.qs_client import QueueStatus

        self.status_now = QueueStatus(**{**READY, **status})
        # a closed environment knows no plans
        self.plans = list(plans) if self.status_now.worker_exists else []
        self.calls: list[str] = []

    def status(self) -> Any:
        return self.status_now

    def readiness(self, expected_plans: Any = None) -> Any:
        from geecs_bluesky.qs_client import readiness_verdict

        return readiness_verdict(self.status_now, self.plans, expected_plans)

    def queue_start(self) -> tuple[bool, str]:
        self.calls.append("start")
        return True, "started"

    def environment_close(self, **_: Any) -> tuple[bool, str]:
        self.calls.append("close")
        return True, "closed"

    def environment_open(self, **_: Any) -> tuple[bool, str]:
        self.calls.append("open")
        return True, "opened"

    def submit_plan(self, name: str, **_: Any) -> Any:
        from geecs_bluesky.qs_client import SubmitResult

        self.calls.append(f"submit:{name}")
        return SubmitResult(ok=True, item_uid="u")


def _service(manager: _Manager) -> ScannerService:
    return ScannerService(manager, DemoResolver(), experiment="Demo", identity="t", preflight=demo_preflight)  # fmt: skip


def _refusal(call: Any) -> str | None:
    try:
        call()
    except ScannerError as exc:
        return exc.message
    return None


#: The page's former start-queue cases (test_page's node harness), now judged by the server.
START_CASES = {
    "stopped": {},
    "empty": {"items_in_queue": 0},
    "idle and empty": {"items_in_queue": 0, "manager_state": "idle"},
    "running": {"re_state": "running", "manager_state": "executing_queue"},
    "paused": {"re_state": "paused", "manager_state": "paused"},
    "started": {"manager_state": "executing_queue"},
    "closed": CLOSED,
    "closed, empty": {**CLOSED, "items_in_queue": 0},
    "opening": OPENING,
    "down": {"connected": False, "detail": "down"},
}


@pytest.mark.parametrize("case", sorted(START_CASES))
def test_every_blocker_is_the_verbs_own_refusal(case: str) -> None:
    """The status's two blockers equal what start and a move raise, or both go through."""
    status = _service(_Manager(**START_CASES[case])).status()
    assert status.start_queue_blocker == _refusal(lambda: _service(_Manager(**START_CASES[case])).start_queue(VerbIn()))  # fmt: skip
    moved = _refusal(lambda: _service(_Manager(**START_CASES[case])).move(MOVE))
    # the status's one sentence names every idle-only item where the verb names itself
    expected = None if moved is None else moved.replace("a move ", f"{IDLE_ONLY_ITEMS} ", 1)  # fmt: skip
    assert status.idle_blocker == expected


def test_the_start_queue_blocker_cases() -> None:
    """A stopped, ready queue with items waiting starts; every other state says why it does not."""
    got = {k: _service(_Manager(**v)).status().start_queue_blocker for k, v in START_CASES.items()}  # fmt: skip
    assert got.pop("stopped") is None
    assert all(got.values()), got
    assert "nothing waits" in got["empty"]
    assert got["closed"] == "the queue cannot start: the worker environment is not ready (environment_closed)"  # fmt: skip


@pytest.mark.parametrize(("case", "word"), [("closed", "closed"), ("opening", "initializing")])  # fmt: skip
def test_a_closed_environment_blocks_the_idle_only_items(case: str, word: str) -> None:
    """With no run-engine state the manager knows no plans: the move is refused before it is sent."""
    manager = _Manager(**{**START_CASES[case], "items_in_queue": 0})
    status = _service(manager).status()
    assert status.idle_blocker == (
        f"{IDLE_ONLY_ITEMS} needs the worker environment open; it is {word}"
        " — the readiness unit opens a closed one"
    )
    with pytest.raises(ScannerError) as err:
        _service(manager).move(MOVE)
    assert (
        err.value.kind == "policy_refusal"
    )  # not manager_unreachable from the manager's refusal
    assert err.value.message.startswith(f"a move needs the worker environment open; it is {word}")  # fmt: skip
    assert manager.calls == []  # never reached the manager
    assert status.run_kit == "degraded" and status.run_word != "idle"


def test_a_closed_environment_keeps_the_start_and_reopen_wording() -> None:
    """Start queue and a reopen already refused a closed environment, in their own words."""
    manager = _Manager(**CLOSED)
    assert _refusal(lambda: _service(manager).start_queue(VerbIn())) == (
        "the queue cannot start: the worker environment is not ready (environment_closed)"
    )
    assert _refusal(lambda: _service(manager).reopen_environment(ReopenIn())) == (
        "a reopen needs the worker environment open and idle; it is closed"
        " — the readiness unit opens a closed one"
    )
    assert manager.calls == []


def test_a_reopen_in_progress_blocks_on_the_status_as_the_verbs_refuse() -> None:
    """The page ignored a reopen for a move, a setup and Set to best; the verbs refuse it."""
    manager = _Manager(items_in_queue=1)
    service = _service(manager)
    service._reopening.set()  # a reopen holds the lock and runs; the status poll does not wait on it
    status = service.status()
    with pytest.raises(ScannerError) as err:
        service.move(MOVE)
    assert status.idle_blocker == err.value.message and err.value.extra["reopening"]
    with pytest.raises(ScannerError) as err:
        service.start_queue(VerbIn())
    assert status.start_queue_blocker == err.value.message
    assert (status.run_word, status.run_kit) == ("reopening", "running")
    assert manager.calls == []


@pytest.mark.parametrize(
    ("status", "plans", "word", "kit"),
    [
        ({"items_in_queue": 0}, ("count",), "idle", "unknown"),
        (CLOSED, ("count",), "environment_closed", "degraded"),
        (OPENING, ("count",), "environment_opening", "degraded"),
        ({"re_state": "running"}, ("count",), "running", "running"),
        ({"re_state": "paused"}, ("count",), "paused", "paused"),
        ({"re_state": "pausing"}, ("count",), "pausing", "paused"),
        ({"re_state": "suspending"}, ("count",), "suspending", "paused"),
        ({"re_state": "stopping"}, ("count",), "stopping", "degraded"),
        ({"re_state": "aborting"}, ("count",), "aborting", "degraded"),
        ({"re_state": "halting"}, ("count",), "halting", "failed"),
        ({"re_state": "panicked"}, ("count",), "panicked", "failed"),
        ({"re_state": "running"}, (), "running", "running"),  # busy outranks readiness
        ({}, (), "plans_empty", "degraded"),
        ({"connected": False}, ("count",), "unreachable", "failed"),
    ],
)
def test_the_run_word_and_its_kit_state(status: dict, plans: tuple, word: str, kit: str) -> None:  # fmt: skip
    out = _service(_Manager(plans=plans, **status)).status()
    assert (out.run_word, out.run_kit) == (word, kit)
