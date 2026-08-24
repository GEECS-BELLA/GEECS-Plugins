"""Plan-level pause semantics for the queueserver world (issue #641).

Implements decisions 1 and 4 of the queueserver migration — both defined
in full below (``GeecsBluesky/CLAUDE.md``'s worker section carries the
operational summary) — at the plan layer,
replacing the engine-side pause supervisor (deleted in W5) for the two
behaviors that had to survive it:

**Quiesce-on-pause (decision 1).**  When a scan pauses — either operator verb
(deferred or immediate) or an in-plan :func:`bluesky.plan_stubs.pause` — the
free-running trigger must stop.  The chosen seam is the RunEngine's own
:class:`~bluesky.protocols.Pausable` device notification: while entering the
paused state the RE awaits ``obj.pause()`` on every object it has seen, and
``RunEngine.resume()`` awaits ``obj.resume()`` on the same objects **before**
replaying any cached message.  :class:`ShotControlPauseQuiescer` rides that
seam: on pause it drives the shot controller's ``OFF`` writes (stopping the
trigger) whenever the plan last drove a free-running standing state
(``SCAN``/``STANDBY``), and on resume it re-asserts the captured state —
sequenced strictly before the resume replay, so replayed rows never race the
re-arm.  Why this over the alternatives considered:

- it is one mechanism for **both** pause verbs *and* the in-plan failed-move
  pause (all three funnel through the same RE pausing path);
- the resume-side ordering guarantee (re-assert completes before the first
  replayed message) cannot be had from ``state_hook`` or a document-stream
  callback without re-growing a supervisor thread;
- pure checkpoint placement (candidate (b) on the issue) cannot satisfy
  "the trigger must stop": free-run per-row checkpoints sit inside the SCAN
  window by design (removing them means whole-bin pause latency), and even
  the disarmed step-boundary window is ``STANDBY``, which keeps passing
  external edges — an unbounded pause there saves orphan frames for its
  whole duration (the Gate-2 failure mode, unbounded).

State rules (mode-agnostic, keyed on ``ShotController.last_state``):
``SCAN``/``STANDBY`` → drive ``OFF``, re-assert on resume; ``ARMED`` (strict)
is already quiescent by construction — the single-shot source cannot
free-run — so the quiescer deliberately does nothing (pinned by test);
``OFF``/``None`` mean the trigger is already stopped / never touched.

**The one pause owner.**  The engine-side pause supervisor was deleted in
W5; this quiescer (plus the queue plan's checkpoint discipline) is now the
sole pause-quiesce mechanism.  During the transitional coexistence the two
double-quiesced (~3 redundant gateway writes per pause, end state provably
correct either way); that behavior died with the supervisor (cross-vendor
review on #645).

**Failed-move → pause (decision 4).**
:func:`~geecs_bluesky.plans.step_scan.move_with_failed_move_pause` (living
at the move site itself, in ``step_scan.py``) catches a failed move status,
records the reason as one ERROR log line (format below — scan.log captures
it via the root-logger handler, and ``RE.record_interruptions`` puts the
pause itself in the data record), and issues a **hard**
:func:`bluesky.plan_stubs.pause`.  The hard-pause resume replay is the retry
mechanism (the sandbox-spike scoping note): ``pause`` is uncacheable, so the
message cache at that moment holds exactly the failed ``set``/``wait`` pair
from the last (pre-move) checkpoint — resume replays them, re-issuing the
same absolute target; stop ends the run gracefully through the finalize
chain.  A retry that fails again re-raises into the plan at the pause yield
and pauses again, indefinitely, until the move lands or the operator stops.

The failed-move log-line contract (the reason record; grep for it in
scan.log)::

    FAILED MOVE - pausing for operator: commanded <axes -> targets>, one
    axis failed - see cause for which: <cause>; resume retries the move
    from the last checkpoint, stop ends the scan gracefully

Checkpoint-placement rules for hard-pause replay idempotence live with the
step plans; the audit table is on the PR for issue #641.  Both step plans
also checkpoint immediately after ``per_step()`` (cross-vendor review,
issue #645 addendum, item P1) — a compiled ActionPlan's writes
(``bps.abs_set`` calls) are not guaranteed idempotent the way an absolute
move is, so a hard pause landing between ``per_step()`` finishing and arm
must not replay them.  The irreducible residual — a hard pause landing
DURING ``per_step()`` itself replays the in-flight action sequence from the
post-move checkpoint — is the same class as strict mode's documented
bounded-refire window and is not closable by checkpoint placement alone.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bluesky.utils import Msg

# Re-exported from its historical home here; defined in the import-light
# log_markers so stream-parsing clients (qs_client) never pull bluesky.
from geecs_bluesky.log_markers import FAILED_MOVE_LOG_PREFIX
from geecs_bluesky.models.shot_control import ShotControlState

if TYPE_CHECKING:  # import-light on purpose: step_scan imports this module
    from geecs_bluesky.shot_controller import ShotController

logger = logging.getLogger(__name__)

__all__ = ["FAILED_MOVE_LOG_PREFIX", "QUIESCE_FROM", "ShotControlPauseQuiescer"]

#: Standing states the quiescer must stop the trigger from.  ARMED is
#: deliberately absent: strict mode's single-shot source cannot free-run, so
#: the paused state is quiescent by construction (pinned by test).  OFF and
#: None (never driven) are already stopped / not the scan's to touch.
QUIESCE_FROM = frozenset({ShotControlState.SCAN.value, ShotControlState.STANDBY.value})


class ShotControlPauseQuiescer:
    """Pausable device: shot control → ``OFF`` on pause, re-asserted on resume.

    Registered into a plan via :meth:`register` (one no-op message carrying
    this object, so the RunEngine counts it among the objects seen).  The RE
    then awaits :meth:`pause` while entering the paused state — for a
    deferred pause landing at a checkpoint, an immediate operator pause, and
    an in-plan ``bps.pause()`` alike — and awaits :meth:`resume` inside
    ``RunEngine.resume()`` **before** the rewind replay restarts, so the
    re-asserted trigger state is standing before any replayed message runs.

    Writes are driven through :meth:`ShotController.state_setters` (the
    documented non-plan accessor for exactly this situation), sequentially in
    profile order, on the RE's event loop — the same loop the CA setters are
    bound to.  Failures are logged loudly but never raised: an exception out
    of the pause/resume notification would take down the run for a safety
    side-channel; the loud ERROR plus the downstream trigger-wait timeout is
    the honest failure surface.

    Parameters
    ----------
    controller : ShotController
        The scan's shot controller (its ``last_state`` names what the plan
        last drove; its ``OFF`` write list is the quiesce).
    """

    def __init__(self, controller: ShotController) -> None:
        self._controller = controller
        self._reassert: str | None = None

    def register(self):
        """Plan stub: make the RunEngine see this object (one no-op message)."""
        yield Msg("null", self)

    async def _drive(self, state: str) -> None:
        """Drive *state*'s writes in profile order, each completing first."""
        for setter, value in self._controller.state_setters(state):
            await setter.set(value)
        self._controller.last_state = state

    async def pause(self) -> None:
        """RE pause notification: stop the trigger if the plan left it running."""
        try:
            standing = self._controller.last_state
            if standing not in QUIESCE_FROM:
                logger.debug(
                    "pause quiescer: last standing state %r needs no quiesce",
                    standing,
                )
                return
            off = ShotControlState.OFF.value
            if not self._controller.defines_state(off):
                logger.warning(
                    "pause quiescer: trigger profile %s defines no OFF "
                    "writes - the scan is paused with the trigger still "
                    "free-running (add an OFF state to the profile)",
                    self._controller.describe_target,
                )
                return
            self._reassert = standing
            await self._drive(off)
            logger.info(
                "pause quiescer: trigger stopped (OFF) for the pause; %s "
                "will be re-asserted on resume",
                standing,
            )
        except Exception:
            logger.exception(
                "pause quiescer: quiesce failed - the scan is paused but the "
                "trigger may still be running; check %s",
                self._controller.describe_target,
            )

    async def resume(self) -> None:
        """RE resume notification: re-assert the captured standing state.

        Runs to completion before ``RunEngine.resume()`` replays any cached
        message, so replayed rows see the restored trigger state.
        """
        standing, self._reassert = self._reassert, None
        if standing is None:
            return
        try:
            await self._drive(standing)
            logger.info("pause quiescer: re-asserted %s on resume", standing)
        except Exception:
            logger.exception(
                "pause quiescer: could not re-assert %s on resume - the next "
                "trigger wait will surface this as a timeout; check %s",
                standing,
                self._controller.describe_target,
            )
