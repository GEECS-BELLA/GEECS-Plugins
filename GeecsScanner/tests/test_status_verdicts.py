"""The status carries the verbs' own verdicts: what the page enables is what the verbs allow.

Each case reads ``GET /api/status``'s blockers and run word over a scripted
manager, then asks the verb itself: the blocker is the verb's refusal
message, byte for byte, or ``None`` exactly when the verb goes through.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from geecs_scanner.service import ScannerService
from geecs_scanner.service.demo import DemoResolver, demo_preflight
from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.models import MoveIn, ReopenIn, VerbIn
from geecs_scanner.service.verdicts import IDLE_ONLY_ITEMS

#: A stopped, ready queue with items waiting: every verb's starting point.
READY = dict(connected=True, re_state="idle", manager_state="idle", worker_exists=True, worker_environment_state="idle", items_in_queue=2)  # fmt: skip


class _Manager:
    """A scripted manager: one status, a readiness over a fixed plan list, verbs recorded."""

    def __init__(self, plans: tuple[str, ...] = ("count", "mv"), **status: Any) -> None:
        from geecs_bluesky.qs_client import QueueStatus

        self.status_now = QueueStatus(**{**READY, **status})
        self.plans = list(plans)
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
        self.status_now = replace(self.status_now, items_in_queue=0)
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
    "running": {"re_state": "running", "manager_state": "executing_queue"},
    "paused": {"re_state": "paused", "manager_state": "paused"},
    "started": {"manager_state": "executing_queue"},
    "closed": {
        "worker_exists": False,
        "worker_environment_state": "closed",
        "re_state": None,
    },  # fmt: skip
    "down": {"connected": False, "detail": "down"},
    "null re_state": {"re_state": None},
    "null re_state, empty": {"re_state": None, "items_in_queue": 0},
}


@pytest.mark.parametrize("case", sorted(START_CASES))
def test_every_blocker_is_the_verbs_own_refusal(case: str) -> None:
    """The status's three blockers equal what start, a move and a reopen raise, or all go through."""
    status = _service(_Manager(**START_CASES[case])).status()
    assert status.start_queue_blocker == _refusal(lambda: _service(_Manager(**START_CASES[case])).start_queue(VerbIn()))  # fmt: skip
    move = MoveIn(variable="U_S1H:current", value=1.0)
    moved = _refusal(lambda: _service(_Manager(**START_CASES[case])).move(move))
    # the status names every idle-only item where the verb names itself
    expected = None if moved is None else moved.replace("a move is", f"{IDLE_ONLY_ITEMS} is", 1)  # fmt: skip
    assert status.idle_blocker == expected
    assert status.reopen_blocker == _refusal(lambda: _service(_Manager(**START_CASES[case])).reopen_environment(ReopenIn()))  # fmt: skip


def test_the_start_queue_blocker_cases() -> None:
    """A stopped, ready queue with items waiting starts; every other state says why it does not."""
    got = {k: _service(_Manager(**v)).status().start_queue_blocker for k, v in START_CASES.items()}  # fmt: skip
    assert got.pop("stopped") is None
    assert (
        got.pop("null re_state") is None
    )  # a null run-engine state is idle, as the verb reads it
    assert all(got.values()), got
    assert "nothing waits" in got["null re_state, empty"]


def test_a_null_run_engine_state_is_idle_on_the_status_too() -> None:
    """Drift 1: the page read a null ``re_state`` as busy and disabled what the verbs allow."""
    manager = _Manager(re_state=None, items_in_queue=0)
    status = _service(manager).status()
    assert (status.run_word, status.run_kit) == ("idle", "unknown")
    assert status.idle_blocker is None
    _service(manager).move(MoveIn(variable="U_S1H:current", value=1.0))
    assert manager.calls == ["submit:mv"]


def test_a_reopen_in_progress_blocks_on_the_status_as_the_verbs_refuse() -> None:
    """Drift 2: the page ignored a reopen for a move, a setup and Set to best; the verbs refuse it."""
    manager = _Manager(items_in_queue=1)
    service = _service(manager)
    service._reopening.set()  # a reopen holds the lock and runs; the status poll does not wait on it
    status = service.status()
    with pytest.raises(ScannerError) as err:
        service.move(MoveIn(variable="U_S1H:current", value=1.0))
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
        ({"re_state": None}, ("count",), "idle", "unknown"),
        ({"re_state": "running"}, ("count",), "running", "running"),
        ({"re_state": "paused"}, ("count",), "paused", "paused"),
        ({"re_state": "pausing"}, ("count",), "pausing", "paused"),
        ({"re_state": "running"}, (), "running", "running"),  # busy outranks readiness
        ({}, (), "plans_empty", "degraded"),
        ({"connected": False}, ("count",), "unreachable", "failed"),
    ],
)
def test_the_run_word_and_its_kit_state(status: dict, plans: tuple, word: str, kit: str) -> None:  # fmt: skip
    out = _service(_Manager(plans=plans, **status)).status()
    assert (out.run_word, out.run_kit) == (word, kit)
