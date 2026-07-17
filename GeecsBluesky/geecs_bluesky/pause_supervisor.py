"""The G-actions v2 pause supervisor (issue #552, PR-3).

Runs **on the scan thread**, inside :meth:`GeecsSession.scan`'s
``RunEngineInterrupted`` handling, when a deferred pause lands at one of
the plans' checkpoints (PR-1).  Protocol, per the design note + owner
decisions recorded on #552:

1. Capture the shot controller's ``last_state`` (the standing state the
   scan drove before the pause), then drive the mode-specific **safe
   state** directly (plan stubs cannot run on a paused RE — writes are
   dispatched onto the RE's persistent loop, sequentially, preserving
   profile order): free-run → ``OFF`` (jet off, no edges into enabled
   saving — safe indefinitely); strict → nothing (``ARMED`` is already
   quiescent; STANDBY would *start* free-running edges).
2. Emit PAUSED, deliver the three-way
   :class:`~geecs_bluesky.events.ActionDecisionRequest` through the
   injected ``ask`` seam, and park (sliced, abort-aware) on the response.
3. Execute → run the pending action's flattened steps through the direct
   executor (PR-2).  A check/step failure **stays paused and re-prompts**
   (retry / ignore / abort — owner decision 10; auto-resuming past a
   failed check would hide exactly what the check exists to catch).
4. Restore: re-assert the captured entry state (only if step 1 drove a
   safe state), then return ``"resume"`` — the session calls
   ``RE.resume()``.  Abort returns ``"abort"`` with **no restore** — the
   plan's finalize chain (quiesce → save-off → disarm → closeout) owns
   the abort end state.

The strict-mode quiescence re-confirm from the design note §2.3 is
deliberately **not** implemented: owner decision 11 (reversed) refuses any
action that writes to the active scan's shot-control device(s) — enforced
by the bridge before the pause is even requested — so an executed action
can no longer perturb the trigger, which was that re-confirm's only
purpose.  Recorded on #552.

Headless sessions pass no supervisor and keep today's behavior
(``RunEngineInterrupted`` propagates).  A supervisor whose ``ask`` seam is
``None`` answers every decision with its default (``ignore``) and logs —
the same never-block posture as :class:`~geecs_bluesky.operator_channel.NullOperator`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from geecs_bluesky.events import ActionDecisionRequest
from geecs_bluesky.plans.action_direct import (
    ActionExecutionAborted,
    execute_action_steps_directly,
    set_signal_in_loop,
)

logger = logging.getLogger(__name__)

__all__ = ["PauseSupervisor", "PendingAction"]

#: Park-loop slice while waiting for the operator's verdict.
_DECISION_SLICE_S = 0.2

#: Budget for each direct shot-control write (same family as the action
#: signals' put budget — a safe-state write that hasn't completed in 30 s
#: is stuck, not slow).
_STATE_WRITE_TIMEOUT_S = 30.0


@dataclass
class PendingAction:
    """One operator-requested action awaiting the pause window.

    Parameters
    ----------
    name :
        The ActionPlan's library name (operator-facing).
    steps :
        ``flatten_action_steps`` output — validated fail-fast by the
        bridge before the pause was requested.
    factory :
        The connected signal factory the steps execute against; the
        supervisor disconnects it (via *cleanup*) when the pending action
        is consumed.
    cleanup :
        Zero-argument callable releasing *factory*'s signals (the bridge
        binds ``session.disconnect(factory)``).  Always called exactly
        once, whatever the verdict.
    """

    name: str
    steps: list[tuple[Any, str | None]]
    factory: Any
    cleanup: Callable[[], None] = field(default=lambda: None)


class PauseSupervisor:
    """Owns the pause window: safe state → decision → action → restore.

    One instance per scan run, created by the bridge (or any caller that
    wants during-scan actions) and threaded into
    :meth:`GeecsSession.scan`/:meth:`~GeecsSession.optimize` as their
    ``pause_supervisor``.

    Parameters
    ----------
    acquisition :
        ``"free_run"`` or ``"strict"`` — selects the safe-state policy.
    shot_controller :
        Zero-argument callable returning the session's current
        :class:`~geecs_bluesky.shot_controller.ShotController` (or
        ``None`` — no shot control, nothing to drive).
    ask :
        Delivery seam for the three-way decision:
        ``ask(ActionDecisionRequest) -> None`` hands the request to the
        consumer (the bridge emits it as a dialog event) and returns
        immediately; the supervisor parks on the request's
        ``response_event``.  ``None`` (headless) answers the default
        verdict (``ignore``) with a warning.
    should_abort :
        Optional operator-stop probe (the bridge's ``_abort_requested``),
        consulted while parked and while executing — a trip anywhere in
        the window yields the ``"abort"`` outcome.
    on_state :
        Optional lifecycle hook called with ``"paused"`` when the window
        opens (the bridge maps it onto its event stream; PAUSING is
        emitted by the bridge at request time, before the pause lands).
    """

    def __init__(
        self,
        *,
        acquisition: str,
        shot_controller: Callable[[], Any],
        ask: Callable[[ActionDecisionRequest], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        on_state: Callable[[str], None] | None = None,
    ) -> None:
        self._acquisition = acquisition
        self._shot_controller = shot_controller
        self._ask = ask
        self._should_abort = should_abort
        self._on_state = on_state
        self._pending: PendingAction | None = None
        self._lock = threading.Lock()
        self._last_failure: Exception | None = None

    # ------------------------------------------------------------------
    # Bridge side (GUI thread)
    # ------------------------------------------------------------------

    def set_pending(self, pending: PendingAction) -> bool:
        """Stage *pending* for the next pause window.

        Returns ``False`` (and leaves the existing pending action in
        place) when one is already staged — one action per pause window.
        """
        with self._lock:
            if self._pending is not None:
                return False
            self._pending = pending
            return True

    def take_unconsumed_pending(self) -> PendingAction | None:
        """Withdraw a staged action the pause never delivered (scan ended).

        The caller reports it ("scan completed before the pause landed —
        the action was NOT run": the normal outcome recorded on #552, not
        a protocol error) and runs its cleanup.
        """
        with self._lock:
            pending, self._pending = self._pending, None
            return pending

    # ------------------------------------------------------------------
    # Scan-thread side
    # ------------------------------------------------------------------

    def on_pause(self, session: Any) -> str:
        """Run one pause window; return ``"resume"`` or ``"abort"``.

        Called by the session on the scan thread while ``RE.state`` is
        ``"paused"``.  Never raises for operator-shaped outcomes — a
        failed action step re-prompts, an abort anywhere returns
        ``"abort"``.
        """
        with self._lock:
            pending, self._pending = self._pending, None

        controller = self._shot_controller()
        entry_state = getattr(controller, "last_state", None)
        drove_safe_state = False
        if self._acquisition != "strict" and controller is not None:
            # Free-run: a between-rows pause sits inside a SCAN window
            # (jet on, edges into enabled saving) — drive OFF so the
            # window is safe for an unbounded decision time.
            drove_safe_state = self._drive_state(session, controller, "OFF")
        if self._on_state is not None:
            self._on_state("paused")

        outcome = "resume"
        try:
            if pending is None:
                # Pause landed with nothing staged (e.g. the operator
                # already withdrew it) — just resume.
                logger.info("pause window opened with no pending action")
                return outcome
            try:
                outcome = self._decide_and_execute(session, pending)
                return outcome
            finally:
                try:
                    pending.cleanup()
                except Exception:  # noqa: BLE001 — cleanup is best-effort
                    logger.warning("pending-action cleanup failed", exc_info=True)
        finally:
            # The abort outcome deliberately skips the restore: RE.abort()'s
            # finalize chain (quiesce → save-off → disarm → closeout) owns
            # the end state.
            if drove_safe_state and entry_state and outcome == "resume":
                self._drive_state(session, controller, entry_state)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _abort_tripped(self) -> bool:
        return self._should_abort is not None and self._should_abort()

    def _decide_and_execute(self, session: Any, pending: PendingAction) -> str:
        attempt = 0
        while True:
            attempt += 1
            verdict = self._park_for_verdict(pending, attempt)
            if verdict == "abort":
                return "abort"
            if verdict == "ignore":
                logger.info("action %r ignored; resuming the scan", pending.name)
                return "resume"
            try:
                execute_action_steps_directly(
                    pending.steps,
                    pending.factory,
                    session.RE.loop,
                    should_abort=self._should_abort,
                )
            except ActionExecutionAborted:
                return "abort"
            except Exception as exc:  # noqa: BLE001 — re-prompt, decision 10
                logger.error(
                    "action %r failed in the pause window: %s", pending.name, exc
                )
                self._last_failure = exc
                continue
            logger.info("action %r executed in the pause window", pending.name)
            return "resume"

    def _park_for_verdict(self, pending: PendingAction, attempt: int) -> str:
        failure = self._last_failure if attempt > 1 else None
        self._last_failure = None
        if failure is not None:
            body = (
                f"Action {pending.name!r} FAILED: {failure}\n\n"
                "The scan is still paused (machine in its safe state). "
                "Retry the action, ignore it and resume the scan, or "
                "abort the scan?"
            )
        else:
            body = (
                f"Action {pending.name!r} was requested — the scan is "
                "paused (machine in its safe state). Execute the action "
                "and continue, ignore it and continue, or abort the scan?"
            )
        request = ActionDecisionRequest(
            action_name=pending.name,
            message=body,
            step_count=len(pending.steps),
        )
        if self._ask is None:
            logger.warning(
                "no decision consumer attached — action %r defaults to "
                "'ignore' (scan resumes, action NOT run)",
                pending.name,
            )
            return "ignore"
        self._ask(request)
        while not request.response_event.wait(timeout=_DECISION_SLICE_S):
            if self._abort_tripped:
                # Stop clicked while parked: wake the flow as an abort so
                # the scan thread proceeds to RE.abort() promptly.
                return "abort"
        verdict = request.verdict[0]
        return verdict if verdict in ("execute", "ignore", "abort") else "ignore"

    def _drive_state(self, session: Any, controller: Any, state: str) -> bool:
        """Directly drive *state*'s ordered writes onto the RE loop.

        Returns ``True`` when at least one write was issued.  A failed
        write is logged and stops the remaining writes (matching the plan
        stub's sequential each-completes-before-the-next contract).
        """
        writes = controller.state_setters(state)
        for setter, value in writes:
            future = asyncio.run_coroutine_threadsafe(
                set_signal_in_loop(setter, value), session.RE.loop
            )
            try:
                future.result(timeout=_STATE_WRITE_TIMEOUT_S)
            except Exception:  # noqa: BLE001 — surfaced, remaining writes skipped
                future.cancel()
                logger.error(
                    "pause-window shot-control write failed (state %s)",
                    state,
                    exc_info=True,
                )
                return bool(writes)
        if writes:
            logger.info("pause window: shot controller → %s", state)
        return bool(writes)
