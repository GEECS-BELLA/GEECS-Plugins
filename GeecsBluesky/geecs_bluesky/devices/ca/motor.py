"""CaMotor — position-feedback motor driven through the CA gateway.

The device's reply is the verdict (GEECS-Plugins#906).  The gateway's
``…:SP`` write rides the GEECS UDP set, which has **two** device replies:

- the **command ACK** on the command port, within GEECS-Core's 1.5 s ACK
  window — no ACK is a device not listening, an ACK other than
  ``accepted`` is a rejection (``is not a number``, an unknown variable);
  the gateway fails the put at once and so does this device: a put that
  fails is raised the moment it fails, inside the grace or not;
- the **executed reply** on the exe socket, once the device has run its
  own convergence check — ``no error`` completes the move; an error reply
  (the device's own check-values timeout included: a device setting,
  adjusted in LabVIEW, never overridden here) fails it at once.

Nothing on our side budgets the executed reply — the connector cannot know
how long a set takes, and a 30 s cap killed a 19 mm move the device
completed at 32 s (Scan009 of 26_0914).  The wait has three named bounds:

1. :data:`REPLY_WAIT` (90 s) — the reply is waited for outright; once it
   is in, the streamed readback is confirmed within ``tolerance`` of the
   target (belt-and-suspenders for devices whose UDP set-completion
   semantics are ambiguous — it adds information only when the readback
   is an independent measurement, a stage encoder, not an echo).
2. The **readback stall rule** — from :data:`PROGRESS_GRACE` onward the
   readback is polled beside the put purely as a stall detector: no
   movement beyond the tolerance for :data:`STALL_TIMEOUT` while the move
   is not at the target fails it now with
   :class:`~geecs_bluesky.exceptions.GeecsMotorTimeoutError` (PV, target,
   current), before ``REPLY_WAIT`` or after it.  A readback sitting at the
   target is not a stall: it is a device that has not answered yet.
3. At ``REPLY_WAIT`` with no reply: a readback within tolerance of the
   target completes the move (the UDP reply was lost — logged at WARNING
   with the PV); a readback still moving keeps the wait alive up to the
   hard :data:`REPLY_CEILING` (300 s), the stall rule still armed;
   anything else is the stall rule's to fail.

The poll reads the readback of the *same* variable it set; the decoupled
set-X-confirm-Y case is
:class:`~geecs_bluesky.devices.ca.confirm.CaConfirmSettable`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys

from ophyd_async.core import AsyncStatus

from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.exceptions import GeecsMotorTimeoutError

logger = logging.getLogger(__name__)

#: Seconds the device's executed reply is waited for outright.  Past it, a
#: readback at the target completes the move as a lost reply; a readback
#: still moving keeps waiting up to :data:`REPLY_CEILING`.  90 s because
#: some devices — the hexapod for one — are very slow to answer even a
#: finished move.  The gateway's own set budget exceeds this, so a reply
#: is never discarded as stale while we wait.
REPLY_WAIT = 90.0

#: Seconds after the put before the readback stall rule applies.  A short
#: move's reply lands well inside it; a readback that has not started
#: moving by then is simply a slow start, not yet a stall.
PROGRESS_GRACE = 5.0

#: Seconds the readback may sit without moving by more than the tolerance
#: (not at the target) before the move is declared failed — the one
#: client-side timeout.  Long enough that the ~5 Hz stream's granularity and
#: a stage's creep-in never trip it; short enough that a dead axis fails a
#: scan in seconds, not minutes.
STALL_TIMEOUT = 10.0

#: Hard ceiling (seconds) on the ``:SP`` put itself — a readback still
#: moving past :data:`REPLY_WAIT` with no reply keeps the wait alive only
#: this long.  A put that reaches it is a gateway that never answered at
#: all; it fails as a put timeout naming the PV.
REPLY_CEILING = 300.0

#: Readback poll period (seconds) — a little faster than the ~5 Hz stream.
_POLL_INTERVAL = 0.1

#: Move-completion tolerance of a bare ``CaMotor(...)`` — tests and ad-hoc
#: devices.  In a scan the motor comes from ``GeecsNamespace``, which passes
#: the variable's positive DB ``tolerance``; a non-positive one binds a plain
#: setpoint (``CaSettable``), so this default is never used for a DB device.
DEFAULT_TOLERANCE = 0.005

# Binary floating point puts an exactly-on-tolerance arrival a few ULPs *over*
# the limit: |-10.505 - -10.5| evaluates to 0.005000000000000782, not 0.005.
# A stage that landed exactly on tolerance therefore polled for the full
# move timeout and paused the scan for an operator (U_ModeImagerESP, Scan034).
#
# The error in |current - value| scales with the *operands*, not the tolerance
# (~ULP(|position|) = |x| * 2.2e-16), so the slack must too: a tolerance-relative
# epsilon under-covers exactly the large-coordinate axes (U_CompAeroTech reads
# ~4e4). Four ULPs of the larger operand covers the subtraction plus the
# comparison with room to spare, and stays far below any real tolerance.
ULP_SLACK = 4 * sys.float_info.epsilon


def within_tolerance(current: float, target: float, tolerance: float) -> bool:
    """Whether *current* is within *tolerance* of *target*, ULP slack included.

    The one tolerance test — the motor's arrival and stall checks and
    :class:`~geecs_bluesky.devices.ca.confirm.CaConfirmSettable`'s analog
    match share it.  A non-finite *current* (a NaN readback) is never
    within tolerance of anything.
    """
    if not math.isfinite(current):
        return False
    slack = ULP_SLACK * max(abs(current), abs(target))
    return abs(current - target) <= tolerance + slack


class CaMotor(CaSettable):
    """GEECS motor over gateway PVs, with position-feedback polling.

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"U_ESP_JetXYZ"``).
    variable : str
        Position variable name (e.g. ``"Position.Axis 1"``).
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    tolerance : float
        Move completion tolerance.  ``set()`` resolves when the device has
        replied ``no error`` and ``|readback − setpoint| ≤ tolerance``, with
        a few ULPs of the larger operand as slack so an arrival landing
        exactly on the tolerance is not lost to binary floating-point
        representation.  The same tolerance is the stall rule's notion of
        "moved".  Defaults to :data:`DEFAULT_TOLERANCE`; ``GeecsNamespace``
        normally passes the GEECS DB's per-variable value instead.
    settle_time : float
        Extra seconds to wait after arrival before completing the status.
    reply_wait : float
        Seconds the device's reply is waited for outright — past it a
        readback at the target completes the move as a lost reply.
        Default :data:`REPLY_WAIT`.
    progress_grace : float
        Seconds after the put before the stall rule applies.  Default
        :data:`PROGRESS_GRACE`.
    stall_timeout : float
        Seconds without readback progress (past the grace, not at the
        target) that fail the move.  Default :data:`STALL_TIMEOUT`.
    reply_ceiling : float
        Hard ceiling in seconds on the ``:SP`` put.  Default
        :data:`REPLY_CEILING`.
    """

    def __init__(
        self,
        device: str,
        variable: str,
        *,
        experiment: str | None = None,
        name: str = "motor",
        tolerance: float = DEFAULT_TOLERANCE,
        settle_time: float = 0.0,
        reply_wait: float = REPLY_WAIT,
        progress_grace: float = PROGRESS_GRACE,
        stall_timeout: float = STALL_TIMEOUT,
        reply_ceiling: float = REPLY_CEILING,
    ) -> None:
        super().__init__(
            device,
            variable,
            experiment=experiment,
            name=name,
            settle_time=settle_time,
            _readback_attr="position",
        )
        self._tolerance = tolerance
        self._reply_wait = reply_wait
        self._progress_grace = progress_grace
        self._stall_timeout = stall_timeout
        self._reply_ceiling = reply_ceiling

    def set(self, value: float) -> AsyncStatus:
        """Move to *value* and return a Status that completes on arrival.

        Implements :class:`bluesky.protocols.Movable`.
        """
        logger.info(
            "%s: moving %s → %s (tol=%.4g)",
            self.name,
            self._variable,
            value,
            self._tolerance,
        )
        return AsyncStatus(self._set_logged(value))

    async def _set_and_wait(self, value: float) -> None:
        """Put the setpoint; wait for the device's reply under the stall rule; confirm.

        The put (the wait for the GEECS UDP reply through the gateway) runs
        as a task beside a readback poll.  The reply is the verdict: a put
        that fails — a rejected or unanswered command ACK (GEECS-Core's
        1.5 s ACK window) or an error executed-reply — propagates
        the moment the put task finishes, never delayed by the grace; a put
        that completes (``no error``) hands over to the readback confirm.
        The poll only decides whether waiting is still reasonable (the
        module docstring's three bounds): a readback that has not moved by
        more than the tolerance for ``stall_timeout`` (after
        ``progress_grace``, not at the target) fails the move with
        :class:`GeecsMotorTimeoutError` naming the PV, the target and the
        current position; at ``reply_wait`` with no reply a readback at
        the target completes the move as a lost reply.  After the reply
        the confirm is bounded outright: ``progress_grace + stall_timeout``
        from the reply, so a readback the stall rule cannot see as stalled
        (a NaN, a ripple wider than the tolerance) fails the move instead
        of holding the scan forever.
        """
        loop = asyncio.get_running_loop()
        position = getattr(self, self._readback_attr_name)
        put = asyncio.ensure_future(self._put.put(value, timeout=self._reply_ceiling))
        try:
            started = loop.time()
            anchor = float(
                await position.get_value()
            )  # last position counted as progress
            # The stall clock starts when the grace ends, or at the last
            # progress — whichever is later.
            stalled_since = started + self._progress_grace
            replied = False
            replied_at = 0.0
            while True:
                if put.done() and not replied:
                    put.result()  # an error reply / a refusal raises here
                    replied = True
                    replied_at = loop.time()
                current = float(await position.get_value())
                at_target = within_tolerance(current, value, self._tolerance)
                now = loop.time()
                if at_target and not replied and now - started >= self._reply_wait:
                    # The stream's arrival and the reply can land in either
                    # order; a reply landing right now still wins.
                    await asyncio.wait({put}, timeout=_POLL_INTERVAL)
                    if put.done():
                        put.result()
                        replied = True
                        replied_at = now
                if at_target and replied:
                    logger.debug(
                        "%s: arrived at %.6g (target=%.6g, tol=%.4g)",
                        self.name,
                        current,
                        value,
                        self._tolerance,
                    )
                    break
                if at_target and now - started >= self._reply_wait:
                    logger.warning(
                        "%s: no reply from the device within %.0f s, but the "
                        "readback %.6g is within tolerance of %.6g — treating "
                        "the move as complete, the reply lost (%s)",
                        self.name,
                        self._reply_wait,
                        current,
                        value,
                        self._setpoint_pv,
                    )
                    break
                confirm_budget = self._progress_grace + self._stall_timeout
                if replied and now - replied_at >= confirm_budget:
                    # The device said converged; the readback never agreed.
                    raise GeecsMotorTimeoutError(
                        self._geecs_device_name,
                        self._variable,
                        target=value,
                        current=current,
                        timeout=confirm_budget,
                        replied=True,
                    )
                moved = math.isfinite(current) and not within_tolerance(
                    current, anchor, self._tolerance
                )
                if at_target or moved:
                    # Progress (or sitting at the target, which is never a
                    # stall): re-anchor and restart the stall clock.
                    anchor = current
                    stalled_since = max(now, started + self._progress_grace)
                elif now - stalled_since >= self._stall_timeout:
                    if put.done() and not replied:
                        put.result()  # a reply that landed this tick is the verdict
                    raise GeecsMotorTimeoutError(
                        self._geecs_device_name,
                        self._variable,
                        target=value,
                        current=current,
                        timeout=self._stall_timeout,
                        replied=replied,
                    )
                await asyncio.sleep(_POLL_INTERVAL)
        finally:
            if not put.done():
                put.cancel()
            elif not put.cancelled():
                put.exception()  # retrieved: never "exception was never retrieved"

        if self._settle_time > 0:
            await asyncio.sleep(self._settle_time)
