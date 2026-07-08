"""Pre-flight checks as a pipeline (vision doc §2).

A pre-flight check is an object with one job: look at the scan about to start
and return one of three outcomes — *pass*, *ask the operator a question*, or
*abort*.  The runner (:func:`run_preflight`) executes a list of checks in
order, routing every question through the injected
:class:`~geecs_bluesky.operator_channel.OperatorChannel`.  New checks are
additions to the list, not new branches in the scanner class.

All checks run **before the scan folder is claimed**, so an abort here burns
no scan number.  The two checks below carry today's exact semantics, moved
verbatim from ``BlueskyScanner._preflight_check_sync_liveness``:

- :class:`GatewayLivenessCheck` — reads each synchronous device's gateway
  ``CONNECTED`` PV (both acquisition modes; fail-open on an unreadable PV).
- :class:`FreeRunStalenessCheck` — free-run mode only: with every device
  CONNECTED, cached ``acq_timestamp`` freshness answers "is the trigger
  free-running?" (all-stale → trigger-off wording; a stale contributor while
  the reference frames → per-device drop offer; stale reference →
  abort-only v1).

Headless / no consumer / no answer → today's behavior is preserved: the check
passes unchanged and the scan fails loudly downstream (t0 sync in free-run,
the liveness-gated refire path in strict).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Union

from geecs_bluesky.exceptions import GeecsDeviceDownError, GeecsStaleDevicesError
from geecs_bluesky.operator_channel import (
    ANSWER_ABORT,
    ANSWER_CONTINUE,
    OperatorChannel,
    OperatorQuestion,
)

logger = logging.getLogger(__name__)

# Seconds between the LabVIEW epoch (1904-01-01) and the Unix epoch
# (1970-01-01).  Device ``acq_timestamp`` values are LabVIEW-epoch.
LABVIEW_EPOCH_OFFSET = 2_082_844_800


# ---------------------------------------------------------------------------
# Context + outcomes
# ---------------------------------------------------------------------------


@dataclass
class PreflightContext:
    """Everything a check needs, injected by the scan owner.

    Parameters
    ----------
    detectors :
        The detector list the scan will run with.  Checks may replace it
        (e.g. after an operator chose drop-and-continue).
    strict :
        ``True`` for ``strict_shot_control`` acquisition, ``False`` for
        free-run.
    read_liveness :
        ``read_liveness(device) -> bool`` — reads the device's gateway
        ``CONNECTED`` PV; must be fail-open (unreadable → ``True``).
    drop_devices :
        ``drop_devices(detectors, drop_ids) -> remaining`` — disconnects and
        removes the given devices (by ``id()``), returning the survivors.
    device_label :
        ``device_label(device) -> str`` — the GEECS device name for
        operator-facing messages.
    dialog_timeout :
        Per-question wait budget, in seconds (``None`` → channel default).
    """

    detectors: list
    strict: bool
    read_liveness: Callable[[Any], bool]
    drop_devices: Callable[[list, set[int]], list]
    device_label: Callable[[Any], str]
    dialog_timeout: Optional[float] = None

    def sync_devices(self) -> list:
        """Return the synchronous devices among the current detectors.

        Returns
        -------
        list
            Devices carrying a persistent-monitor cache (``_last_acq``).
        """
        return [d for d in self.detectors if hasattr(d, "_last_acq")]

    def question(
        self,
        exc: Exception,
        *,
        title: str,
        continue_label: str,
        abort_label: str = "Abort Scan",
    ) -> OperatorQuestion:
        """Build an :class:`OperatorQuestion` with this context's timeout.

        Parameters
        ----------
        exc :
            Operator-facing exception; ``str(exc)`` is the dialog body.
        title, continue_label, abort_label :
            Dialog wording, owned by the check.

        Returns
        -------
        OperatorQuestion
            The assembled question.
        """
        return OperatorQuestion(
            message=str(exc),
            exc=exc,
            title=title,
            continue_label=continue_label,
            abort_label=abort_label,
            timeout=self.dialog_timeout,
        )


@dataclass
class Passed:
    """The check is satisfied.

    Parameters
    ----------
    skip_remaining :
        ``True`` when later checks must not run — e.g. the operator already
        opted in to a degraded scan and stacking further dialogs on top
        would be wrong.
    """

    skip_remaining: bool = False


@dataclass
class Aborted:
    """The scan must not start.

    Parameters
    ----------
    reason :
        Logged at WARNING by the runner.
    """

    reason: str


@dataclass
class Ask:
    """The operator must decide.

    Parameters
    ----------
    question :
        The question, routed through the channel by the runner.
    on_continue :
        Invoked when the operator answered continue; returns the follow-up
        outcome (may mutate the context, e.g. drop devices).
    on_default :
        Invoked when nobody answered (headless / timeout); must preserve
        today's fail-loud default behavior.
    abort_reason :
        Logged at WARNING when the operator answered abort.
    """

    question: OperatorQuestion
    on_continue: Callable[[], Union[Passed, Aborted]]
    on_default: Callable[[], Union[Passed, Aborted]]
    abort_reason: str


CheckResult = Union[Passed, Aborted, Ask]


class PreflightCheck(Protocol):
    """One pre-flight check: inspect the context, return an outcome."""

    def __call__(self, ctx: PreflightContext) -> CheckResult:
        """Run the check against *ctx*."""
        ...


def run_preflight(
    checks: list[PreflightCheck],
    ctx: PreflightContext,
    channel: OperatorChannel,
) -> list | None:
    """Execute *checks* in order, asking the operator through *channel*.

    Parameters
    ----------
    checks :
        The pipeline, run first to last.
    ctx :
        Shared context; checks may replace ``ctx.detectors``.
    channel :
        Where questions go (GUI event stream, headless default, ...).

    Returns
    -------
    list or None
        The (possibly reduced) detector list to proceed with, or ``None``
        when the scan must abort.
    """
    for check in checks:
        result = check(ctx)
        if isinstance(result, Ask):
            answer = channel.ask(result.question)
            if answer == ANSWER_ABORT:
                logger.warning(result.abort_reason)
                return None
            if answer == ANSWER_CONTINUE:
                result = result.on_continue()
            else:
                result = result.on_default()
        if isinstance(result, Aborted):
            logger.warning(result.reason)
            return None
        if result.skip_remaining:
            return ctx.detectors
    return ctx.detectors


# ---------------------------------------------------------------------------
# Check 1 — gateway CONNECTED liveness (both modes)
# ---------------------------------------------------------------------------


@dataclass
class GatewayLivenessCheck:
    """Catch dead sync devices via the gateway ``CONNECTED`` PV (both modes).

    Each synchronous device's ``connected_status`` (the gateway's per-device
    ``CONNECTED`` PV) is read; a device reporting ``Disconnected`` is
    genuinely down — its TCP stream to the gateway is dead.  This is the
    authoritative signal: the gateway serves every DB device's data PVs
    regardless of device state, so CA-connect success never implied liveness
    (an OFF camera's PVs connect fine).  Outcomes:

    - a disconnected *reference* (free-run pacemaker) → abort-only v1
      (the second button is a clearly-labeled "Try Anyway" because the
      dialog channel always offers two options; promotion is deferred) —
      and, on opt-in, later checks are skipped rather than stacking a
      staleness dialog on top;
    - any other disconnected device(s), either mode → drop-and-continue
      (disconnected devices removed from the list) vs abort;
    - an unreadable ``CONNECTED`` PV (old gateway, timeout) → fail-open:
      the injected ``read_liveness`` treats it as live.
    """

    def __call__(self, ctx: PreflightContext) -> CheckResult:
        """Run the liveness stage against the current detector list.

        Parameters
        ----------
        ctx :
            The pre-flight context.

        Returns
        -------
        CheckResult
            ``Passed`` (possibly skipping later checks), or an ``Ask``.
        """
        sync_devices = ctx.sync_devices()
        if not sync_devices:
            return Passed(skip_remaining=True)

        dead = [d for d in sync_devices if not ctx.read_liveness(d)]
        if not dead:
            return Passed()

        dead_ids = {id(d) for d in dead}
        details = ", ".join(
            f"{ctx.device_label(d)} (gateway reports DISCONNECTED)" for d in dead
        )
        reference = sync_devices[0]

        if not ctx.strict and id(reference) in dead_ids:
            # v1: a dead reference (pacemaker) is abort-only — promotion is
            # deliberately out of scope. (Strict mode has no pacemaker; a
            # dead first device there is just another droppable device.)
            exc = GeecsDeviceDownError(
                f"The free-run reference (pacemaker) device is down: "
                f"{details}. The gateway's CONNECTED PV says its TCP "
                "stream is dead — the scan cannot pace without it; "
                "aborting is recommended. Trying anyway will fail at "
                "t0 sync.",
                device_name=ctx.device_label(reference),
            )

            def _proceed_anyway() -> Passed:
                logger.warning(
                    "Pre-flight: proceeding with a DISCONNECTED reference "
                    "(%s) — t0 sync will fail loudly",
                    details,
                )
                # The operator explicitly opted in; skip further pre-flight
                # dialogs rather than stack a staleness dialog on top.
                return Passed(skip_remaining=True)

            return Ask(
                question=ctx.question(
                    exc,
                    title="Reference Device Disconnected",
                    continue_label="Try Anyway",
                ),
                on_continue=_proceed_anyway,
                on_default=_proceed_anyway,
                abort_reason="Pre-flight: operator aborted (reference disconnected)",
            )

        exc = GeecsDeviceDownError(
            f"Synchronous device(s) are down: {details}. The gateway's "
            "CONNECTED PV says their TCP stream is dead (an off/crashed "
            "GEECS device — its data PVs still connect, so this is the "
            "authoritative check). Drop them and continue the scan "
            "without their data, or abort."
        )

        def _drop_dead() -> Passed:
            ctx.detectors = ctx.drop_devices(ctx.detectors, dead_ids)
            if not ctx.sync_devices():
                return Passed(skip_remaining=True)
            return Passed()

        def _proceed_unchanged() -> Passed:
            logger.warning(
                "Pre-flight: device(s) %s are DISCONNECTED per the "
                "gateway but no operator answer — proceeding unchanged; "
                "the scan will fail loudly downstream (t0 sync in "
                "free-run, device-down abort in strict)",
                details,
            )
            return Passed(skip_remaining=True)

        return Ask(
            question=ctx.question(
                exc,
                title="Disconnected Device(s)",
                continue_label="Drop && Continue",
            ),
            on_continue=_drop_dead,
            on_default=_proceed_unchanged,
            abort_reason="Pre-flight: operator aborted (disconnected devices)",
        )


# ---------------------------------------------------------------------------
# Check 2 — free-run trigger-running staleness (free-run mode ONLY)
# ---------------------------------------------------------------------------


def find_stale_sync_devices(
    sync_devices: list, threshold_s: float
) -> list[tuple[Any, float | None]]:
    """Return ``(device, age_seconds_or_None)`` for stale sync devices.

    Parameters
    ----------
    sync_devices :
        Devices carrying a persistent-monitor cache (``_last_acq``,
        LabVIEW-epoch seconds).
    threshold_s :
        Frames older than this (wall-clock seconds; control machines are
        NTP-synced) count as stale.

    Returns
    -------
    list of tuple
        Stale devices with their frame age; ``None`` age means "never
        acquired" (no frame seen since connect).
    """
    now_labview = time.time() + LABVIEW_EPOCH_OFFSET
    stale: list[tuple[Any, float | None]] = []
    for device in sync_devices:
        last = device._last_acq
        if last is None:
            stale.append((device, None))
            continue
        age = now_labview - float(last)
        if age > threshold_s:
            stale.append((device, age))
    return stale


@dataclass
class FreeRunStalenessCheck:
    """Free-run only: is the trigger free-running? (staleness heuristics).

    With every device CONNECTED (the liveness check ran first), cached
    ``acq_timestamp`` freshness answers one remaining question: *is the
    trigger free-running?* (a free-run scan requires it).  All devices
    CONNECTED but all frames stale → the "trigger may be off" dialog (Start
    Anyway / Abort).  The residual case — a CONNECTED-but-stale contributor
    while the reference frames — keeps the drop-or-abort dialog: the fresh
    reference proves the trigger is running, so this is a per-device
    acquisition problem (e.g. camera acquisition stopped while its TCP
    stream stays up), for which drop is the right offer and trigger-off
    wording would be wrong.  A CONNECTED-but-stale *reference* keeps the
    abort-only dialog.

    Strict mode passes immediately: frames are not needed before a strict
    scan (the trigger may legitimately sit OFF; ``ARMED`` starts it), and
    with ``CONNECTED`` authoritative there is no differential-staleness
    inference left to make.

    Parameters
    ----------
    threshold_s :
        Frame age beyond which a device counts as stale.
    recheck_wait_s :
        Grace period before re-checking: a just-connected persistent monitor
        may not have delivered its first frame yet, so one stale verdict
        gets a second look after roughly one trigger period.
    """

    threshold_s: float
    recheck_wait_s: float = 0.0

    def __call__(self, ctx: PreflightContext) -> CheckResult:
        """Run the staleness stage against the current detector list.

        Parameters
        ----------
        ctx :
            The pre-flight context.

        Returns
        -------
        CheckResult
            ``Passed``, or an ``Ask`` for one of the three stale cases.
        """
        if ctx.strict:
            # Frames are not needed before a strict scan (the trigger may
            # legitimately sit OFF; ARMED starts it) and CONNECTED already
            # answered liveness — nothing left to check.
            return Passed()

        sync_devices = ctx.sync_devices()
        if not sync_devices:
            return Passed()

        stale = find_stale_sync_devices(sync_devices, self.threshold_s)
        if stale and self.recheck_wait_s > 0:
            # A just-connected monitor may not have delivered its first frame
            # yet; give the free-running trigger one more period.
            time.sleep(self.recheck_wait_s)
            stale = find_stale_sync_devices(sync_devices, self.threshold_s)
        if not stale:
            return Passed()

        def _describe(device: Any, age: float | None) -> str:
            if age is None:
                return f"{ctx.device_label(device)} (no frames since connect)"
            return f"{ctx.device_label(device)} (last frame {age:.0f} s ago)"

        details = ", ".join(_describe(dev, age) for dev, age in stale)
        stale_ids = {id(dev) for dev, _age in stale}
        reference = sync_devices[0]

        if len(stale) == len(sync_devices):
            # Every sync device is CONNECTED but frameless — unambiguous now:
            # the trigger is off / not free-running.
            exc = GeecsStaleDevicesError(
                f"All synchronous devices are CONNECTED per the gateway but "
                f"none has a fresh frame ({details}). The trigger appears to "
                "be off / not free-running — free-run scans need the trigger "
                "free-running before start. Starting anyway will fail at "
                "t0 sync if nothing is firing."
            )

            def _start_anyway() -> Passed:
                logger.warning(
                    "Pre-flight: proceeding with all sync devices stale (%s) "
                    "— t0 sync will fail loudly if the trigger is really off",
                    details,
                )
                return Passed()

            return Ask(
                question=ctx.question(
                    exc,
                    title="Trigger May Be Off",
                    continue_label="Start Anyway",
                ),
                on_continue=_start_anyway,
                on_default=_start_anyway,
                abort_reason="Pre-flight: operator aborted (all sync stale)",
            )

        if id(reference) in stale_ids:
            # v1: a frameless reference (pacemaker) is abort-only, same as a
            # disconnected one — promotion is deliberately out of scope.
            exc = GeecsStaleDevicesError(
                f"The free-run reference (pacemaker) device is CONNECTED but "
                f"has no fresh frames: {details}. The scan cannot pace "
                "without it; aborting is recommended. Trying anyway will "
                "fail at t0 sync if it is really not acquiring."
            )

            def _try_anyway() -> Passed:
                logger.warning(
                    "Pre-flight: proceeding with a stale reference (%s) — "
                    "t0 sync will fail loudly if it is really not acquiring",
                    details,
                )
                return Passed()

            return Ask(
                question=ctx.question(
                    exc,
                    title="Reference Device Not Acquiring",
                    continue_label="Try Anyway",
                ),
                on_continue=_try_anyway,
                on_default=_try_anyway,
                abort_reason="Pre-flight: operator aborted (stale reference)",
            )

        # Residual case: CONNECTED-but-stale contributor(s) while the
        # reference frames.  The fresh reference proves the trigger is
        # running, so this is a per-device acquisition problem (camera
        # acquisition stopped while its TCP stream stays up) — offer to drop.
        exc = GeecsStaleDevicesError(
            f"Synchronous device(s) are CONNECTED but not producing frames: "
            f"{details}. The reference is framing, so the trigger is "
            "running — these devices are not acquiring. Drop them and "
            "continue the scan without their data, or abort."
        )

        def _drop_stale() -> Passed:
            ctx.detectors = ctx.drop_devices(ctx.detectors, stale_ids)
            return Passed()

        def _proceed_unchanged() -> Passed:
            logger.warning(
                "Pre-flight: stale sync device(s) %s but no operator answer "
                "— proceeding unchanged; t0 sync will fail loudly if they "
                "are really not acquiring",
                details,
            )
            return Passed()

        return Ask(
            question=ctx.question(
                exc,
                title="Device(s) Not Acquiring",
                continue_label="Drop && Continue",
            ),
            on_continue=_drop_stale,
            on_default=_proceed_unchanged,
            abort_reason="Pre-flight: operator aborted (stale sync devices)",
        )
