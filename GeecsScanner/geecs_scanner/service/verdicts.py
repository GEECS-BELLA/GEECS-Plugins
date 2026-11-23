"""Whether a verb may run now, and the run's word: pure functions over one manager poll.

The verbs raise these refusals and ``StatusOut`` carries their messages, so
the page's buttons and the service's answers come from one rule. Each
function takes the snapshot (and the readiness verdict) it judges, so the
status poll judges without the service lock, and a verb judges under it.
"""

from __future__ import annotations

from typing import Any, Optional

from geecs_scanner.service.errors import ScannerError

#: The idle-only items, as the status's blocker names them (a verb names itself).
IDLE_ONLY_ITEMS = "a move, a setup or a calibration"

#: The manager's busy words as kit states (any other busy word is ``running``).
_BUSY_KIT = {"paused": "paused", "pausing": "paused"}


def _unreachable(snap: Any) -> ScannerError:
    return ScannerError("manager_unreachable", snap.detail or "manager not answering")


def reopening_refusal() -> ScannerError:
    """The refusal of every verb that queues while the environment reopens."""
    return ScannerError(
        "policy_refusal",
        "the worker environment is reopening to read a committed configuration "
        "change; try again in a minute",
        reopening=True,
    )


def idle_refusal(snap: Any, what: str) -> Optional[ScannerError]:
    """Why *what* (an item that runs as soon as it is added) is refused, or ``None``.

    Allowed when the manager answers, nothing runs (a ``None`` run-engine
    state is idle) and nothing waits.
    """
    if not snap.connected:
        return _unreachable(snap)
    if snap.re_state not in (None, "idle"):
        return ScannerError(
            "policy_refusal",
            f"{what} is idle-only: a plan is {snap.re_state} — wait for it or stop it first",
            re_state=snap.re_state,
        )
    if snap.items_in_queue:
        return ScannerError(
            "policy_refusal",
            f"{what} is idle-only: {snap.items_in_queue} item(s) wait in the queue "
            "and it would run right after them — clear the queue first",
            items_in_queue=snap.items_in_queue,
        )
    return None


def start_queue_refusal(snap: Any, verdict: Any) -> Optional[ScannerError]:
    """Why the queue cannot be started now, or ``None``.

    Allowed when the environment is ready, nothing runs or is paused, the
    manager's queue is stopped and at least one item waits.
    """
    if not snap.connected:
        return _unreachable(snap)
    if verdict.state != "ready":
        return ScannerError(
            "policy_refusal",
            "the queue cannot start: the worker environment is not ready "
            f"({verdict.state})",
            readiness=verdict.state,
        )
    busy = snap.re_state not in (None, "idle")
    if busy or snap.manager_state != "idle":
        return ScannerError(
            "policy_refusal",
            "the queue is already going: a plan is "
            f"{snap.re_state if busy else snap.manager_state}",
            re_state=snap.re_state,
            manager_state=snap.manager_state,
        )
    if not snap.items_in_queue:
        return ScannerError(
            "policy_refusal",
            "nothing waits in the queue, so there is nothing to start",
            items_in_queue=0,
        )
    return None


def reopen_refusal(snap: Any) -> Optional[ScannerError]:
    """Why the worker environment cannot be reopened now, or ``None``.

    Allowed when the manager answers, the environment exists and is idle,
    nothing runs and nothing waits.
    """
    if not snap.connected:
        return _unreachable(snap)
    if not snap.worker_exists or snap.worker_environment_state != "idle":
        return ScannerError(
            "policy_refusal",
            "a reopen needs the worker environment open and idle; it is "
            f"{snap.worker_environment_state or 'closed'} — the readiness unit opens a closed one",
            worker_environment_state=snap.worker_environment_state,
        )
    if snap.re_state not in (None, "idle") or snap.manager_state != "idle":
        return ScannerError(
            "policy_refusal",
            f"a reopen is idle-only: a plan is {snap.re_state or snap.manager_state}",
            re_state=snap.re_state,
        )
    if snap.items_in_queue:
        return ScannerError(
            "policy_refusal",
            f"a reopen is idle-only: {snap.items_in_queue} item(s) wait in the queue",
            items_in_queue=snap.items_in_queue,
        )
    return None


def run_word(snap: Any, verdict: Any, reopening: bool) -> tuple[str, str]:
    """``(word, kit state)`` for what the manager is doing now.

    A reopen first, then an unreachable manager, then a busy run engine's
    own word (it outranks a readiness complaint), then an environment that
    is not ready; otherwise ``idle``, grey (a ``None`` run-engine state is
    idle, as the verbs read it).
    """
    if reopening:
        return "reopening", "running"
    if not snap.connected:
        return "unreachable", "failed"
    if snap.re_state not in (None, "idle"):
        return snap.re_state, _BUSY_KIT.get(snap.re_state, "running")
    if verdict.state != "ready":
        return verdict.state, "degraded"
    return "idle", "unknown"


def message(refusal: Optional[ScannerError]) -> Optional[str]:
    """The refusal's sentence as the status carries it (``None`` when allowed)."""
    return None if refusal is None else refusal.message
