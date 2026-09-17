"""The shot-offset calibration and its preflight — the two once-run plans.

Two devices stamp the *same* shot at different times: ``acq_timestamp`` is
the trigger's arrival plus that device's own frame-drain latency, a
per-device constant of tens of milliseconds.  The
s-file join corrects each side by that constant before matching frames to
rows (:mod:`geecs_data_utils.shot_join`), and until this module ran every
constant in the field read ``0.0``.

At 1 Hz the join windows are ±0.5 s and swallow the spread, so nothing is
broken today.  They narrow with the rep rate: at 5 Hz they are ±0.1 s — the
same order as the spread itself, where an uncalibrated offset costs rows.
Measuring these numbers is what makes faster running safe, which is why the
gated batch exists.

Why both plans are once-run, never a scan step
----------------------------------------------
A GEECS device emits its TCP event either on a successful acquisition or,
failing that, when its own timeout expires — and the timeout event carries
an **unchanged** stamp, which the CA gateway's change suppression drops
(measured in M1).  So nothing announces quiescence: the only way to
know the set is quiet is to watch the stamps not move for longer than the
longest device timeout in it.  That is the floor on both plans' cost, and
it is why neither may sit inside a scan.

:func:`measure_shot_offsets_plan`
    Drive the box OFF, wait the set quiet, then fire single shots and read
    every device's stamp.  The spread across devices *is* the calibration.

:func:`check_shot_sync_plan`
    Sam's validation shortcut, and it costs no shot at all: with
    the box OFF and the set quiet, every device still holds the stamp of
    the **last real shot**, so correcting those stalled stamps by the
    stored offsets and comparing the results says whether the stored
    calibration is still true.

Averaging, and why
------------------
A device's offset is not perfectly steady: each host's clock dithers around
its average by up to ~10 ms (higher-end boxes hold ~1 ms) while the domain
keeps the averages on a common target (Sam, 2026-09-13).  One shot
therefore measures the offset only to that precision, against real spreads
of tens to hundreds of milliseconds (0–160 ms measured on HTU, 2026-09-12).  The expensive part of this plan is the quiet wait, paid
once; the shots after it cost a second each.  So the default is several
shots, and the document records each device's **peak-to-peak scatter**
beside its mean — a device whose scatter dwarfs its peers' has a
timekeeping problem an average would hide.  The scatter is relative to the
set (see :class:`OffsetMeasurement`), which is the quantity that matters
because the join is relative too.

Only complete shots contribute — a shot in which every requested device
delivered.  That is not fussiness: the per-shot anchor is the mean across
the devices present, so a shot missing one device would shift the anchor
and bias every other device's offset by a fraction of the missing one's.
Incomplete shots are logged and retaken instead.
"""

from __future__ import annotations

import logging
import math
import statistics
from functools import partial
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import bluesky.plan_stubs as bps
from bluesky.protocols import Readable
from bluesky.utils import separate_devices
from geecs_schemas.shot_offsets import DeviceOffset, ShotOffsets
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.devices.detector import GeecsDetector
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plans.gated import TRIGGER_PERIOD_S, run_bracket
from geecs_bluesky.plans.strict import fire_and_await_shot
from geecs_bluesky.scan_log import plan_report_sink
from geecs_bluesky.utils import resolve_annotations

logger = logging.getLogger(__name__)

#: The logger surfaced for the duration of a calibration plan.  The whole
#: package, not just this module: the resolver's "shot offsets written to
#: <path>" and the detectors' own narrative belong in a calibration's report
#: for the same reason they belong in a scan's ``scan.log``.
REPORT_LOGGER = "geecs_bluesky"

#: The timeout a GEECS device falls back on when no acquisition arrives —
#: 1.5 s for ~95 % of them.  Not discoverable: the experiment DB
#: carries no timeout column, so this is the documented constant, and a set
#: containing a slower device wants an explicit ``quiet_time``.
DEVICE_TIMEOUT_S = 1.5

#: Added to :data:`DEVICE_TIMEOUT_S` for the drain of a frame already in
#: flight when the box went OFF (the TCP message is sent when the
#: exposure completes, so a long exposure lands most of a second late).
QUIET_MARGIN_S = 0.75

#: The confirmation window is this many trigger periods long.  It has to
#: exceed ONE period, or a box that never went OFF is caught only by luck:
#: at 1 Hz a 0.5 s window contains an edge half the time (review of #861,
#: finding 2 — a phase sweep caught a running box in 6 of 12 runs).  1.5
#: periods guarantees at least one edge falls inside it.
QUIET_CONFIRM_PERIODS = 1.5

#: Floor for the confirmation window, for an unusually fast trigger period.
QUIET_CONFIRM_FLOOR_S = 0.5

#: Shots averaged by default — ~3 ms of residual noise against ~10 ms of
#: per-shot dither, for ten seconds of beam time.
DEFAULT_CALIBRATION_SHOTS = 10

#: Total fires allowed, as a multiple of the requested shot count: the
#: requested shots plus that many again to replace incomplete ones.
_RETAKE_BUDGET = 2

#: The trigger period assumed when a caller does not say — HTU's 1 Hz, the
#: same default the gated batch uses.  Sizes the quiet confirmation window
#: and the whole-shot folding in the sync check.
DEFAULT_TRIGGER_PERIOD_S = TRIGGER_PERIOD_S

#: A measured offset larger than this is refused for *writing*.  The
#: measured HTU set spans 0–160 ms (2026-09-12), so a third of a second is
#: already outside anything a frame drain explains and means the measurement
#: is wrong — a device latched a different edge, or the stamp wait did not
#: wait.  Bounded by half the trigger period as well, because a whole-period
#: error is smaller than this cap at any rate above ~3 Hz.  Reported, never silently stored
#: (review of #861, finding 3: a +1.964 s "drain latency" passed every
#: validator).  Overridable for a genuinely slow device.
MAX_PLAUSIBLE_OFFSET_S = 0.3

#: Likewise for the scatter: host dither runs ~1–10 ms (both ends measured
#: on HTU), so an order of magnitude past it means the shots were not all
#: the same shot.
MAX_PLAUSIBLE_SCATTER_S = 0.1

#: Below this many shots the scatter column is not meaningful and the mean
#: carries the full ~10 ms dither; warned about, not refused.
MIN_USEFUL_SHOTS = 3

#: Default tolerance for :func:`check_shot_sync_plan`.  The by-eye version
#: Sam used quotes ~200 ms; 50 ms is well clear of the ~10 ms dither and
#: still catches a device a whole shot out of step.
DEFAULT_SYNC_TOLERANCE_S = 0.05


#: Resolved annotation objects for the two registered plans — see
#: :func:`~geecs_bluesky.utils.resolve_annotations` for why a registered plan
#: cannot carry this module's postponed (string) annotations.  ``detectors``
#: is annotated exactly as the stock verbs annotate theirs, so the manager
#: treats the argument the way it treats ``count``'s.
_PLAN_ANNOTATIONS: dict[str, Any] = {
    "detectors": Sequence[Readable],
    "trigger_profile": Optional[str],
    "shots": int,
    "quiet_time": Optional[float],
    "trigger_period": float,
    "write": bool,
    "max_offset": float,
    "measured_at_rate_hz": Optional[float],
    "description": str,
    "tolerance_s": float,
}


# ---------------------------------------------------------------- pure logic


@dataclass(frozen=True)
class OffsetMeasurement:
    """The aggregate of one :func:`measure_shot_offsets_plan` run.

    Attributes
    ----------
    reference :
        Object name of the device that stamped first — the anchor, whose
        own offset is ``0.0``.
    offsets :
        Object name → seconds after the reference, the mean over shots.
    scatter :
        Object name → peak-to-peak spread of that device's per-shot offset.
        A **relative** quantity, like the offset itself: it is measured
        against the per-shot anchor (the mean of the set), so one device's
        wobble is shared with the anchor and the reported figure understates
        the true dither by ``(N-1)/N`` for a set of *N* devices — with two
        devices each reports half the pair's relative wobble.  That is the
        right quantity for judging the calibration, because the join uses
        relative offsets too; it is not an absolute measure of one host's
        clock quality.
    shots :
        How many complete shots contributed.
    """

    reference: str
    offsets: Mapping[str, float]
    scatter: Mapping[str, float]
    shots: int

    def table(self) -> str:
        """One line per device, widest offset last — the log and the PR body."""
        rows = sorted(self.offsets.items(), key=lambda kv: kv[1])
        width = max((len(n) for n in self.offsets), default=0)
        return "\n".join(
            f"  {name:<{width}}  {value * 1e3:+8.1f} ms"
            f"  (scatter {self.scatter.get(name, 0.0) * 1e3:5.1f} ms)"
            + ("   <- reference" if name == self.reference else "")
            for name, value in rows
        )

    def to_document(
        self,
        *,
        geecs_names: Mapping[str, str] | None = None,
        trigger_profile: str | None = None,
        trigger_rate_hz: float | None = None,
        description: str = "",
        measured_at: str | None = None,
    ) -> ShotOffsets:
        """Render as the configs-repo document the worker reads at startup."""
        names = dict(geecs_names or {})
        return ShotOffsets(
            reference=self.reference,
            devices={
                name: DeviceOffset(
                    offset_s=value,
                    scatter_s=self.scatter.get(name, 0.0),
                    shots=self.shots,
                    geecs_device=names.get(name, ""),
                )
                for name, value in self.offsets.items()
            },
            measured_at=measured_at
            or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            trigger_profile=trigger_profile,
            trigger_rate_hz=trigger_rate_hz,
            description=description,
        )


def offsets_from_shots(shots: Sequence[Mapping[str, float]]) -> OffsetMeasurement:
    """Reduce per-shot stamps to each device's mean offset and its scatter.

    Each element of *shots* is one complete shot: object name → that
    device's ``acq_timestamp`` for it.  Every shot must carry the same
    device set (the caller discards incomplete shots), because the per-shot
    anchor is the mean across the devices present and a changing set would
    move the anchor between shots.

    The reduction, and why each step is the way it is:

    1. **Anchor each shot on the mean of its own stamps.**  Consecutive
       shots are a second apart and the laser's phase drifts, so the raw
       stamps share a large moving component that is not a device property.
       Subtracting a per-shot anchor removes it.  The *mean* is the anchor
       rather than the minimum because the minimum of noisy values is
       biased low and carries the full dither of whichever device happened
       to be earliest, which would leak into every other device's scatter.
    2. **Average each device's anchored offsets** over the shots — this is
       the estimate the join will use.
    3. **Re-zero on the earliest device.**  The anchor above is a fictitious
       mean, so the means are shifted by an arbitrary constant.  That is
       harmless to the join (it subtracts offsets from both sides, so a
       common constant cancels) but the stored document anchors on a real
       device, whose offset is then exactly ``0.0``.

    The reference is resolved **explicitly**: the smallest mean, ties broken
    by object name, and its offset assigned ``0.0`` outright rather than
    left to arithmetic. Two devices genuinely stamping together must not
    leave which one is the reference to float rounding.

    Parameters
    ----------
    shots :
        Per-shot stamps, one mapping per complete shot.

    Returns
    -------
    OffsetMeasurement

    Raises
    ------
    ValueError
        No shots, a shot with no devices, an inconsistent device set, or a
        non-finite stamp.
    """
    if not shots:
        raise ValueError("no complete shots to measure offsets from")
    names = set(shots[0])
    if not names:
        raise ValueError("the first shot recorded no devices")
    for index, shot in enumerate(shots):
        if set(shot) != names:
            raise ValueError(
                f"shot {index} measured devices {sorted(shot)}, expected "
                f"{sorted(names)} — every shot must carry the same set"
            )
        for name, stamp in shot.items():
            if not math.isfinite(stamp):
                raise ValueError(f"shot {index}: {name} stamped {stamp!r}")
    # Sorted, so every mapping below iterates deterministically: `names` is a
    # set, and set order is hash order, which would otherwise decide a tie
    # between two devices that genuinely stamp together.
    anchored: dict[str, list[float]] = {name: [] for name in sorted(names)}
    for shot in shots:
        anchor = statistics.fmean(shot.values())
        for name, stamp in shot.items():
            anchored[name].append(stamp - anchor)
    means = {name: statistics.fmean(values) for name, values in anchored.items()}
    scatter = {
        name: (max(values) - min(values)) if len(values) > 1 else 0.0
        for name, values in anchored.items()
    }
    # Explicit: smallest mean wins; `means` is in name order (above), so a
    # tie goes to the first name rather than to whichever device the hash
    # happened to put first.
    reference = min(means, key=lambda name: means[name])
    zero = means[reference]
    offsets = {name: value - zero for name, value in means.items()}
    offsets[reference] = 0.0
    return OffsetMeasurement(
        reference=reference,
        offsets=offsets,
        scatter=scatter,
        shots=len(shots),
    )


@dataclass(frozen=True)
class SyncVerdict:
    """Whether the stored offsets still describe the hardware.

    Attributes
    ----------
    synced :
        The corrected stamps agreed to within *tolerance_s*, once whole
        trigger periods were folded out.
    deviation_s :
        Object name → its corrected stamp minus the latest device's, with any
        whole number of trigger periods removed.  The spread of these is
        what is judged.
    spread_s :
        Widest disagreement between two devices once whole trigger periods
        are folded out — the judged figure, because it is what the join
        sees.
    tolerance_s :
        The budget it was judged against.
    corrected :
        Object name → its stalled stamp minus its stored offset.
    shots_out :
        Object name → how many whole trigger periods it sits behind the
        latest device (never ``0``).  A device here missed the last edge or two
        before the box went OFF — routine, because STANDBY passes edges
        right up to the moment this plan drives OFF, and a long exposure
        cut by the amplitude drop holds the previous shot.  Reported and
        warned about, **not** a failure: the stamps alone cannot tell that
        apart from a device that stopped being triggered.
    unmeasured :
        Devices holding no usable stamp (never acquired since boot).
    comparable :
        Whether enough devices held a usable stamp to judge at all.  A
        verdict with this ``False`` is "could not check", not "failed".
    detail :
        One line saying what was found — the log line and the error message.
    """

    synced: bool
    deviation_s: Mapping[str, float]
    spread_s: float
    tolerance_s: float
    corrected: Mapping[str, float]
    shots_out: Mapping[str, int]
    unmeasured: tuple[str, ...]
    comparable: bool
    detail: str


def sync_verdict_from_stamps(
    stamps: Mapping[str, float | None],
    offsets: Mapping[str, float],
    *,
    tolerance_s: float = DEFAULT_SYNC_TOLERANCE_S,
    trigger_period_s: float = DEFAULT_TRIGGER_PERIOD_S,
) -> SyncVerdict:
    """Judge stalled stamps against the stored offsets.

    With the box OFF and the set quiet, every device holds the stamp of the
    same last real shot.  Correcting each by its stored offset should
    therefore collapse them onto one instant, to within NTP jitter and the
    hosts' own dither.  A device that lands outside is
    mis-calibrated, and its rows will misjoin once the windows tighten.

    Two refinements over "is the range inside the tolerance", both from the
    review of #861 (finding 4):

    - **The verdict is on the pairwise spread, and the two devices at its
      ends are named.**  What costs rows is two devices disagreeing — the
      join matches a frame to a row by their two corrected stamps — so the
      spread is the quantity judged.  A per-device deviation from a set
      median cannot be: with two devices the median sits midway, so an
      84 ms disagreement reads as ±42 ms and passes a 50 ms budget while
      the join sees all 84 ms.  Nothing in two disagreeing stamps says
      which is wrong, so the message names both ends rather than electing
      a culprit.  Note that a spread is a range statistic
      and grows with the number of devices: with ~10 ms of host dither and
      5-10 ms of NTP, a set of a few dozen devices may need a wider
      *tolerance_s* than the pairwise default.
    - **Whole trigger periods are folded out first.**  STANDBY passes edges
      right up to the moment the caller drives OFF, so a slow camera whose
      exposure was cut by the amplitude drop holds shot *k-1* while the
      fast ones hold *k*.  That is routine and says nothing about the
      calibration, so a deviation within tolerance of a whole number of
      periods is reported as "one shot behind" and warned about rather than
      failing the check.  The stamps alone cannot distinguish it from a
      device that stopped being triggered, so the verdict does not pretend
      to: it says what it saw.

    A stamp of ``None`` or a non-positive value is *unmeasured*, not a
    failure: the gateway publishes ``0.0`` before a device's first
    acquisition, and a device that has not acquired since boot says nothing
    about the calibration.  Fewer than two usable stamps leaves the set
    **not comparable** — reported as "could not check", which is not the
    same as "failed".

    Parameters
    ----------
    stamps :
        Object name → its stalled ``acq_timestamp`` (``None`` if unread).
    offsets :
        Object name → its stored drain offset, seconds.
    tolerance_s :
        Widest accepted disagreement between any two corrected stamps.
    trigger_period_s :
        The trigger period whole multiples of which are folded out.

    Returns
    -------
    SyncVerdict

    Raises
    ------
    ValueError
        *tolerance_s* or *trigger_period_s* is not positive.
    """
    if tolerance_s <= 0:
        raise ValueError(f"tolerance_s must be positive seconds, got {tolerance_s}")
    if trigger_period_s <= 0:
        raise ValueError(
            f"trigger_period_s must be positive seconds, got {trigger_period_s}"
        )
    corrected: dict[str, float] = {}
    unmeasured: list[str] = []
    for name, stamp in stamps.items():
        if stamp is None or not math.isfinite(stamp) or stamp <= 0:
            unmeasured.append(name)
            continue
        corrected[name] = stamp - float(offsets.get(name, 0.0))
    if len(corrected) < 2:
        missing = ", ".join(sorted(unmeasured)) or "none"
        return SyncVerdict(
            synced=False,
            deviation_s={},
            spread_s=0.0,
            tolerance_s=tolerance_s,
            corrected=corrected,
            shots_out={},
            unmeasured=tuple(sorted(unmeasured)),
            comparable=False,
            detail=(
                f"only {len(corrected)} device(s) hold a usable stamp, so the "
                f"set could not be compared (no stamp yet: {missing}) — fire a "
                "few shots by hand and check again"
            ),
        )
    # Fold whole periods against the LATEST device, never against the median.
    # A median is not a real instant: for an even-sized set it sits BETWEEN
    # the groups, so a device exactly one period out lands at half a period
    # from it and `round(±0.5)` is 0 (banker's rounding) — nothing folds, the
    # verdict reads a full period of disagreement and the plan raises on the
    # very case it documents as routine, while a two-period gap folds to a
    # fabricated "one ahead, one behind". Both reproduced in review.
    # The latest corrected stamp is an instant some device actually reported,
    # so every other device is a whole number of periods behind it or is
    # genuinely mis-calibrated.
    anchor = max(corrected.values())
    deviation: dict[str, float] = {}
    shots_out: dict[str, int] = {}
    for name, value in corrected.items():
        raw = value - anchor
        whole = round(raw / trigger_period_s)
        if whole:
            shots_out[name] = int(whole)
        deviation[name] = raw - whole * trigger_period_s
    # The judged quantity is the PAIRWISE disagreement, because that is what
    # the join consumes: a frame is matched to a row by the two corrected
    # stamps, so what costs rows is two devices differing, not either one
    # differing from a notional truth. (A per-device deviation from the
    # median cannot carry this: with two devices the median sits midway and
    # an 84 ms disagreement reads as ±42 ms, inside a 50 ms budget, while
    # the join sees the full 84 ms.) The per-device deviations from the
    # anchor are kept to name the two devices at the ends of the
    # disagreement, not to elect a culprit.
    spread = max(deviation.values()) - min(deviation.values())
    synced = spread <= tolerance_s
    # Name the two devices at the ends of the disagreement, not "the one that
    # is wrong": when two corrected stamps differ, nothing in the data says
    # which of them is off. Anchoring on the latest device would also
    # systematically exonerate it, since its own deviation is 0 by
    # construction.
    earliest = min(sorted(deviation), key=lambda n: deviation[n])
    latest = max(sorted(deviation), key=lambda n: deviation[n])
    detail = (
        f"corrected stamps span {spread * 1e3:.1f} ms against a "
        f"{tolerance_s * 1e3:.0f} ms tolerance"
    )
    if not synced:
        detail += (
            f" — {earliest} and {latest} disagree by that much; the stored "
            "offsets no longer describe this set, so re-run "
            "measure_shot_offsets"
        )
        if shots_out:
            # The fold anchors on the latest stamp, so when the anchor device
            # is the one that is out, the devices that agree with each other
            # are the ones reported behind.
            detail += (
                " (the fold anchors on the latest device: if the devices "
                "reported behind agree with each other, the anchor is the one "
                "that moved)"
            )
    if shots_out:
        behind = ", ".join(
            f"{n} {abs(k)} shot(s) behind"  # anchored on the latest: never ahead
            for n, k in sorted(shots_out.items())
        )
        detail += (
            f" (whole trigger periods folded out: {behind} — routine if they "
            "missed the last edge before OFF, but check they are still being "
            "triggered)"
        )
    if unmeasured:
        detail += f" (not covered, no stamp yet: {', '.join(sorted(unmeasured))})"
    return SyncVerdict(
        synced=synced,
        deviation_s=deviation,
        spread_s=spread,
        tolerance_s=tolerance_s,
        corrected=corrected,
        shots_out=shots_out,
        unmeasured=tuple(sorted(unmeasured)),
        comparable=True,
        detail=detail,
    )


# ------------------------------------------------------------ plan machinery


def _can_write(resolver: Any) -> bool:
    """Whether *resolver* can actually store a measurement.

    Checked **before** the shots are fired, not after: the ``ConfigResolver``
    protocol does not require the write method, so a resolver satisfying the
    protocol without it would otherwise fail with ``AttributeError`` only
    once the whole measurement had been spent (review of #861).
    """
    return resolver is not None and callable(
        getattr(resolver, "write_shot_offsets", None)
    )


def _refuse_profile_without(shot_control: Any, state: TriggerState) -> None:
    """Refuse up front a profile that cannot drive *state*.

    Before the quiet wait, not after: a profile missing a state fails the
    plan either way, but failing after several seconds of waiting with
    ``_drive``'s generic "defines no writes" is a worse way to learn it — and
    a profile missing STANDBY would fail in the bracket's *finalizer*, after
    the shots, leaving the box in the calibration state.
    """
    if not shot_control.defines(state):
        raise GeecsConfigurationError(
            f"trigger profile {shot_control.profile_name!r} defines no writes "
            f"for {state.value} — the calibration plans bracket the box "
            "OFF → … → STANDBY, and measure_shot_offsets fires through ARMED "
            "and SINGLESHOT, so the profile must define every state it drives"
        )


def _refuse_implausible(
    measurement: OffsetMeasurement,
    *,
    max_offset: float,
) -> None:
    """Refuse to *store* a measurement outside the physically possible range.

    A drain offset is a frame-drain latency, and the measured HTU set spans
    0 to 160 ms (2026-09-12; the un-ROI'd camera is the slow one).  A
    measurement far outside that is not an unusual camera — it is a device
    that latched a different edge, or a stamp wait that did not wait — and
    it would be seeded into every future run's join.  The numbers are
    already logged by the caller, so refusing costs only the write.

    The cap is *max_offset* alone, deliberately **not** tightened by the
    trigger period: at 5 Hz a period is 0.2 s and the real ModeImager drain
    is 0.16 s, so a whole-period error and a genuine slow drain are the same
    magnitude and no bound can tell them apart (review of #861, round 3: a
    half-period bound refused the real calibration at 5 Hz, and *max_offset*
    could not lift it).  The defence against a whole-period error is
    upstream — the stamp wait that actually waits (``fly`` cleared at the
    view) and the completeness rule (a shot any device missed is retaken,
    never averaged in).

    Raises
    ------
    GeecsConfigurationError
        An offset exceeds *max_offset*, or a scatter exceeds
        :data:`MAX_PLAUSIBLE_SCATTER_S`.
    """
    bound = max_offset
    wild = {n: v for n, v in measurement.offsets.items() if abs(v) > bound}
    noisy = {
        n: v for n, v in measurement.scatter.items() if v > MAX_PLAUSIBLE_SCATTER_S
    }
    if not wild and not noisy:
        return
    parts = []
    if wild:
        parts.append(
            "offsets past %.0f ms: %s"
            % (
                bound * 1e3,
                ", ".join(f"{n} {v * 1e3:+.0f} ms" for n, v in sorted(wild.items())),
            )
        )
    if noisy:
        parts.append(
            "scatter past %.0f ms: %s"
            % (
                MAX_PLAUSIBLE_SCATTER_S * 1e3,
                ", ".join(f"{n} {v * 1e3:.0f} ms" for n, v in sorted(noisy.items())),
            )
        )
    # Branched advice: max_offset can lift an offset refusal, never a
    # scatter one (review of #861, round 2 finding 10).
    advice = []
    if wild:
        advice.append(
            "An offset that far past the measured 0-160 ms range is a bad "
            "measurement, not unusual hardware: a device latched a different "
            "edge, or its stamp wait returned without waiting. Re-run, and "
            "raise max_offset only if you have reason to believe the hardware."
        )
    if noisy:
        advice.append(
            "Scatter that wide is a host with a timekeeping problem or a box "
            "still passing edges, and max_offset cannot lift it: find the cause "
            "and re-run."
        )
    raise GeecsConfigurationError(
        "refusing to store this measurement — "
        + "; ".join(parts)
        + ". "
        + " ".join(advice)
        + " The table above is the measurement; nothing was written."
    )


def _stamp_views(detectors: Sequence[Any]) -> list[Any]:
    """The scalars view of every named detector — refuse anything without a stamp.

    The view, never the detector itself: it is ``Triggerable`` through the
    same acquire logic (so its wait *is* the shot's stamp wait) but leaves
    the parent's data logics unprepared, so a calibration writes no frame,
    arms no plugin and claims no file (:class:`GeecsDetectorScalars`).
    Passing the detector or its view is therefore the same request, and
    both normalise here.
    """
    views: list[Any] = []
    seen: set[int] = set()
    for device in separate_devices(list(detectors)):
        owner = getattr(device, "_owner", device)
        if not isinstance(owner, GeecsDetector):
            raise GeecsConfigurationError(
                f"{getattr(device, 'name', device)!r} is not a triggered GEECS "
                "device, so it has no acq_timestamp to calibrate — name the "
                "cameras and triggered scalar devices of the set"
            )
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        views.append(owner.scalars)
    if len(views) < 2:
        raise GeecsConfigurationError(
            "a shot-offset calibration needs at least two devices — the "
            "measurement is the spread *between* devices, and one device "
            "is its own reference by definition"
        )
    return views


def _usable(value: float | None) -> bool:
    """A stamp that can be compared: read, and from a real acquisition (not 0.0)."""
    return value is not None and value != 0.0


def _read_stamps(views: Sequence[Any]):
    """Plan: object name → each device's current stamp, by an **uncached** get.

    Deliberately not ``bps.rd``: that issues ``Msg('read', signal)`` and
    ophyd-async's ``SignalR.read`` serves the monitor cache whenever one
    exists — which it always does here, because ``GeecsAcquireLogic.attach``
    subscribes at connect.  Under OFF the stamp PV publishes nothing at all
    (a device's timeout event carries an unchanged stamp and the gateway's
    change suppression drops it), so the cache holds whatever the
    monitor last delivered rather than what the PV holds now.  In practice a
    CA monitor delivers an initial update at subscribe, so the two agree —
    which is why the plan's own connect-touch (``bps.read(view)``) works
    fine.  An explicit uncached get is simply the read that means what this
    function says it means, and it does not depend on that.

    Goes through the stock ``wait_for`` stub so it runs on the RunEngine's
    loop like any other message.
    """
    stamps: dict[str, float | None] = {}
    for view in views:
        owner = view._owner
        try:
            (task,) = yield from bps.wait_for(
                [partial(owner.acq_timestamp.get_value, cached=False)]
            )
            # `wait_for` returns the tasks, not their results; calling
            # `.result()` is also what re-raises a failed get here instead of
            # leaving it as a silently-retrieved exception.
            value = task.result()
        except Exception:
            logger.warning(
                "%s: acq_timestamp could not be read", owner.name, exc_info=True
            )
            value = None
        stamps[owner.name] = None if value is None else float(value)
    return stamps


def _settle_quiet(views: Sequence[Any], quiet_time: float, confirm_time: float):
    """Plan: wait the set quiet with the box already OFF; return the stalled stamps.

    Costs at least the longest device timeout, by construction — there is no
    event to wait for, because the timeout events never reach the gateway.
    So: wait it out, then prove the set is still by re-reading
    over a confirmation window.  A stamp that advanced in that window means
    edges are still arriving, which makes every number this plan would
    produce meaningless.

    *confirm_time* must exceed one trigger period or the check is a coin
    flip: at 1 Hz a 0.5 s window contains an edge only half the time, so a
    running box would be caught in half the runs and silently measured in
    the others (review of #861, finding 2).  The caller sizes it from the
    period; :data:`QUIET_CONFIRM_PERIODS` is the multiple.

    Returns
    -------
    dict
        Object name → stalled stamp (``None`` where unread).
    """
    before = yield from _read_stamps(views)
    logger.info(
        "waiting %.3g s for the device set to go quiet (the longest device "
        "timeout — nothing announces quiescence), then confirming "
        "over %.3g s",
        quiet_time,
        confirm_time,
    )
    yield from bps.sleep(quiet_time)
    settled = yield from _read_stamps(views)
    yield from bps.sleep(confirm_time)
    confirm = yield from _read_stamps(views)
    moving = [
        name
        for name, value in confirm.items()
        if value is not None
        and settled.get(name) is not None
        and value != settled[name]
    ]
    # Two independent signals that the box never went OFF. The second is
    # free and covers what the first can miss: ONE device draining an
    # in-flight frame across the long wait is expected, but the WHOLE set
    # advancing across a wait that already exceeds the device timeout is a
    # box still passing edges.
    # Count only devices that held a usable stamp in BOTH reads. A `None`
    # (unreadable) or a 0.0 (never acquired since boot) cannot advance, so
    # counting one in the denominator would disable this backstop for as
    # long as that device sits there (review of #861, round 3: a camera
    # holding 0.0 let the two others advance across the wait unrefused).
    comparable = [
        name
        for name, value in settled.items()
        if _usable(value) and _usable(before.get(name))
    ]
    drained = [name for name in comparable if settled[name] != before[name]]
    everything_moved = len(comparable) > 1 and len(drained) == len(comparable)
    if moving or everything_moved:
        culprits = sorted(moving) or sorted(drained)
        why = (
            f"still acquiring {confirm_time:.3g} s after {quiet_time:.3g} s in OFF"
            if moving
            else (
                f"every device advanced across the {quiet_time:.3g} s quiet wait, "
                "which already exceeds the device timeout"
            )
        )
        raise GeecsConfigurationError(
            f"{', '.join(culprits)}: {why} — the trigger box is not actually "
            "off for these devices (check the trigger profile's OFF writes, or "
            "whether another source is triggering them). A calibration measured "
            "against a running box is meaningless."
        )
    if drained:
        logger.info(
            "quiet: %s drained a frame that was in flight when the box went OFF",
            ", ".join(sorted(drained)),
        )
    return settled


# ------------------------------------------------------------------- plans


def measure_shot_offsets_plan(
    profiles: Any, resolver: Any | None
) -> Callable[..., Any]:
    """Build the ``measure_shot_offsets`` queue plan.

    Parameters
    ----------
    profiles :
        The experiment's :class:`~geecs_bluesky.plans.registry.TriggerProfiles`.
    resolver :
        The configs-repo resolver the measurement is written through;
        ``None`` registers a plan that can measure but not write (the
        hermetic worker).
    """

    def measure_shot_offsets(
        detectors: Sequence[Any],
        *,
        trigger_profile: str | None = None,
        shots: int = DEFAULT_CALIBRATION_SHOTS,
        quiet_time: float | None = None,
        trigger_period: float = DEFAULT_TRIGGER_PERIOD_S,
        write: bool = False,
        max_offset: float = MAX_PLAUSIBLE_OFFSET_S,
        measured_at_rate_hz: float | None = None,
        description: str = "",
    ):
        """Measure each device's edge-to-stamp latency; optionally store it.

        Drives the trigger box OFF, waits the device set quiet, then fires
        single shots and reads every device's ``acq_timestamp``.  The spread
        across devices for one shot is the calibration; several shots are
        averaged because each host's clock dithers by up to ~10 ms around
        its own average.

        No run is opened: nothing is claimed, no scan number is taken and no
        s-file is written — this is a queue item like ``run_action``, not a
        scan.  The box is returned to STANDBY on every path that unwinds the
        plan, including an exception and ``RE.stop()``; a ``RE.halt()``
        skips finalizers by bluesky contract and would leave it ARMED.

        The measurement is **reported** by default and stored only with
        ``write=True``, so a re-run cannot silently replace a good
        calibration with a worse one.  What is written is the experiment's
        ``shot_offsets.yaml`` in the configs repo — a git working tree,
        usually on the share: committing it is a human act, and the path is
        logged so there is no doubt which file to review.  A stored
        measurement reaches the worker at its **next environment open**, not
        immediately: the offsets are seeded into each detector when the
        namespace is built.

        Parameters
        ----------
        detectors : sequence of devices
            The triggered GEECS devices to calibrate — cameras and triggered
            scalar devices. At least two: the measurement is the spread
            *between* devices. A device's ``.scalars`` view is accepted and
            means the same thing; frames are never written either way.
        trigger_profile : str, optional
            Profile driving the box (the experiment default when omitted).
        shots : int, optional
            Complete shots to average (default 10). More shots cut the
            residual noise as the square root; the quiet wait, which
            dominates the cost, is paid once regardless.
        quiet_time : float, optional
            Seconds to wait in OFF before believing the set is quiet.
            Default is the documented device timeout plus a drain margin;
            raise it for a set containing a device with a longer timeout or
            a long exposure.
        trigger_period : float, optional
            The machine's trigger period, seconds (default 1.0). Sizes the
            confirmation window that proves the box really went OFF — it
            must span more than one period or a running box is caught only
            by luck.
        write : bool, optional
            Store the result in the configs repo (default ``False``).
        measured_at_rate_hz : float, optional
            The machine's rep rate, recorded in the document as provenance
            because a pipelining camera's offset depends on it. **Declared,
            not measured**: this plan fires single shots with a stamp wait
            between them, so its own shot spacing is not the machine's rate
            and cannot be used. Left unset the document records no rate,
            which is honest; setting it wrongly is worse than leaving it out.
        max_offset : float, optional
            Largest offset accepted for *writing*, seconds (default 0.3),
            further bounded by half the trigger period.
            The measured HTU set spans 0-160 ms, so anything near this means
            the measurement is wrong rather than the hardware unusual. Also
            bounded by half the trigger period, since a whole-period error is
            smaller than this cap above ~3 Hz. The measurement is still
            reported; only the write is refused.
        description : str, optional
            Note recorded in the document.

        Returns
        -------
        OffsetMeasurement
            The measured offsets, also logged as a table.
        """
        shots = int(shots)
        if shots < 1:
            raise GeecsConfigurationError(f"shots must be at least 1, got {shots}")
        if quiet_time is None:
            quiet_time = DEVICE_TIMEOUT_S + QUIET_MARGIN_S
        quiet_time = float(quiet_time)
        if quiet_time <= 0:
            raise GeecsConfigurationError(
                f"quiet_time must be positive seconds, got {quiet_time}"
            )
        trigger_period = float(trigger_period)
        if trigger_period <= 0:
            raise GeecsConfigurationError(
                f"trigger_period must be positive seconds, got {trigger_period}"
            )
        if measured_at_rate_hz is not None and not measured_at_rate_hz > 0:
            # Before any shot: the document's own validator (gt=0) would
            # otherwise reject it only after the shots were spent.
            raise GeecsConfigurationError(
                "measured_at_rate_hz must be a positive rate in Hz, or left "
                f"unset — got {measured_at_rate_hz}"
            )
        if write and not _can_write(resolver):
            raise GeecsConfigurationError(
                "measure_shot_offsets: this worker cannot write the "
                "measurement (no configs resolver, or one without "
                "write_shot_offsets) — run with write=False to measure only"
            )
        if shots < MIN_USEFUL_SHOTS:
            logger.warning(
                "measuring from %d shot(s): below %d the scatter column is not "
                "meaningful and the mean still carries the full ~10 ms host "
                "dither — the quiet wait is the expensive part, so more shots "
                "are nearly free",
                shots,
                MIN_USEFUL_SHOTS,
            )
        views = _stamp_views(detectors)
        shot_control = profiles.resolve(trigger_profile)
        # Every state the plan drives, before any wait or shot: run_bracket
        # opens with OFF and its finalizer restores STANDBY, so a profile
        # missing STANDBY would spend the quiet wait and the shots and then
        # fail in the finalizer, leaving the box in the calibration state
        # (Codex review of #861).
        for state in (
            TriggerState.OFF,
            TriggerState.STANDBY,
            TriggerState.ARMED,
            TriggerState.SINGLESHOT,
        ):
            _refuse_profile_without(shot_control, state)
        confirm_time = max(
            QUIET_CONFIRM_FLOOR_S, QUIET_CONFIRM_PERIODS * trigger_period
        )
        profile_key = (
            trigger_profile if trigger_profile is not None else profiles.default
        )

        def inner():
            # Touch every device once: the message that connects it (the
            # worker's connect_on_demand preprocessor), so a device missing
            # from the gateway fails at the top rather than part-way through
            # the measurement. The box is already OFF by here — run_bracket
            # drives it before this generator runs — so this is not "before
            # the box is touched"; the finalizer returns it to STANDBY.
            for view in views:
                yield from bps.read(view)
            yield from _settle_quiet(views, quiet_time, confirm_time)
            # ARMED is the strict source: the box emits one edge per
            # SINGLESHOT and nothing in between, so the set stays quiet
            # between shots and every stamp advance is a shot this plan
            # caused.
            yield from bps.mv(shot_control, TriggerState.ARMED.value)

            def fire():
                yield from bps.mv(shot_control, TriggerState.SINGLESHOT.value)

            complete: list[dict[str, float]] = []
            attempts = 0
            budget = shots * _RETAKE_BUDGET  # the requested shots, plus as many retakes
            while len(complete) < shots and attempts < budget:
                attempts += 1
                missed = yield from fire_and_await_shot(views, fire)
                if missed:
                    logger.warning(
                        "shot %d incomplete (%s did not deliver) — discarded and "
                        "retaken; an incomplete shot would bias every other "
                        "device's offset",
                        attempts,
                        ", ".join(sorted(getattr(m, "name", str(m)) for m in missed)),
                    )
                    continue
                stamps = yield from _read_stamps(views)
                if any(value is None for value in stamps.values()):
                    logger.warning(
                        "shot %d: a stamp could not be read — discarded", attempts
                    )
                    continue
                complete.append({k: float(v) for k, v in stamps.items()})
                logger.info("shot %d of %d recorded", len(complete), shots)
            if len(complete) < shots:
                raise GeecsConfigurationError(
                    f"only {len(complete)} of {shots} shots were complete after "
                    f"{attempts} fires — the set is not delivering reliably "
                    "enough to calibrate (check the devices that missed, above)"
                )
            return complete

        # ONE sink around the whole body, scoped to the package: entering it
        # per logging statement left everything between the blocks discarded
        # — including the resolver's own "shot offsets written to <path>",
        # which is the line telling the operator there is an uncommitted
        # change to review (review of #861, finding 11).
        with plan_report_sink(REPORT_LOGGER):
            complete = yield from run_bracket(inner(), shot_control, TriggerState.OFF)
            measurement = offsets_from_shots(complete)
            logger.info(
                "shot offsets over %d shot(s), reference %s:\n%s",
                measurement.shots,
                measurement.reference,
                measurement.table(),
            )
            if not write:
                logger.info(
                    "measured only — re-run with write=True to store this in "
                    "the experiment's shot_offsets.yaml"
                )
                return measurement
            _refuse_implausible(measurement, max_offset=max_offset)
            document = measurement.to_document(
                geecs_names={
                    view._owner.name: view._owner._geecs_device_name for view in views
                },
                trigger_profile=profile_key,
                trigger_rate_hz=measured_at_rate_hz,
                description=description,
            )
            path = resolver.write_shot_offsets(document)
            logger.info(
                "shot offsets stored in %s — this is an UNCOMMITTED change in "
                "the configs repo: review and commit it. The worker picks the new "
                "offsets up at its next environment open, not now.",
                path,
            )
            return measurement

    return resolve_annotations(measure_shot_offsets, _PLAN_ANNOTATIONS)


def check_shot_sync_plan(profiles: Any) -> Callable[..., Any]:
    """Build the ``check_shot_sync`` queue plan — the calibration's preflight.

    Parameters
    ----------
    profiles :
        The experiment's :class:`~geecs_bluesky.plans.registry.TriggerProfiles`.
    """

    def check_shot_sync(
        detectors: Sequence[Any],
        *,
        trigger_profile: str | None = None,
        tolerance_s: float = DEFAULT_SYNC_TOLERANCE_S,
        quiet_time: float | None = None,
        trigger_period: float = DEFAULT_TRIGGER_PERIOD_S,
    ):
        """Check the stored drain offsets still describe this device set.

        Sam's shortcut, and it costs no shot: with the box OFF and the set
        quiet, every device still holds the stamp of the same last real
        shot, so correcting those stalled stamps by the stored offsets
        should collapse them onto one instant.  The verdict is on the
        **pairwise spread** of the corrected stamps — what the join actually
        consumes — and the message names the two devices at the ends of the
        disagreement rather than pretending to know which one is wrong.  Note
        a spread is a range statistic and grows with the number of devices,
        so a set of a few dozen may want a wider *tolerance_s*.  Whole trigger
        periods are folded out first: STANDBY passes
        edges right up to the moment this plan drives OFF, so a slow camera
        can legitimately hold the previous shot — that is reported and
        warned about, not failed.

        This is a **queue item**, deliberately — never a step inside a scan.
        It costs at least the longest device timeout every time it runs, and
        running it from the queue also means it cannot drive
        the trigger box while a scan is using it.

        Raises when a device is out of tolerance, so a queue that puts this
        ahead of its scans stops before taking data against a stale
        calibration.  A set too sparse to judge — fewer than two devices
        holding a usable stamp — is reported as **could not check** and does
        *not* raise: that is not the same as a failure, and stopping a queue
        for it on a freshly booted set would be wrong.

        No run is opened.  The box is returned to STANDBY on every path that
        unwinds the plan, including an exception and ``RE.stop()``; a
        ``RE.halt()`` skips finalizers by bluesky contract.

        Parameters
        ----------
        detectors : sequence of devices
            The triggered GEECS devices to check (at least two).
        trigger_profile : str, optional
            Profile driving the box (the experiment default when omitted).
        tolerance_s : float, optional
            Widest disagreement accepted between any two corrected stamps,
            seconds (default 0.05 — clear of the ~10 ms host dither and the
            5-10 ms of NTP, tight enough to catch a real mis-calibration).
            A range statistic, so raise it for a large set.
        quiet_time : float, optional
            Seconds to wait in OFF before believing the set is quiet.
        trigger_period : float, optional
            The machine's trigger period, seconds (default 1.0): sizes the
            confirmation window, and whole multiples of it are folded out of
            each device's deviation.

        Returns
        -------
        SyncVerdict
        """
        if quiet_time is None:
            quiet_time = DEVICE_TIMEOUT_S + QUIET_MARGIN_S
        quiet_time = float(quiet_time)
        if quiet_time <= 0:
            raise GeecsConfigurationError(
                f"quiet_time must be positive seconds, got {quiet_time}"
            )
        if tolerance_s <= 0:
            raise GeecsConfigurationError(
                f"tolerance_s must be positive seconds, got {tolerance_s}"
            )
        trigger_period = float(trigger_period)
        if trigger_period <= 0:
            raise GeecsConfigurationError(
                f"trigger_period must be positive seconds, got {trigger_period}"
            )
        views = _stamp_views(detectors)
        shot_control = profiles.resolve(trigger_profile)
        # The bracket drives OFF and restores STANDBY; refuse a profile that
        # cannot, before the quiet wait is spent (Codex review of #861).
        _refuse_profile_without(shot_control, TriggerState.OFF)
        _refuse_profile_without(shot_control, TriggerState.STANDBY)
        confirm_time = max(
            QUIET_CONFIRM_FLOOR_S, QUIET_CONFIRM_PERIODS * trigger_period
        )

        def inner():
            for view in views:
                yield from bps.read(view)
            stalled = yield from _settle_quiet(views, quiet_time, confirm_time)
            offsets: dict[str, float] = {}
            for view in views:
                owner = view._owner
                value = yield from bps.rd(owner.drain_offset)
                offsets[owner.name] = float(value or 0.0)
            return stalled, offsets

        with plan_report_sink(REPORT_LOGGER):
            stalled, offsets = yield from run_bracket(
                inner(), shot_control, TriggerState.OFF
            )
            verdict = sync_verdict_from_stamps(
                stalled,
                offsets,
                tolerance_s=tolerance_s,
                trigger_period_s=trigger_period,
            )
            if not any(offsets.values()):
                logger.warning(
                    "every stored drain offset reads 0.0 — this set has never "
                    "been calibrated, so the check below only says whether the "
                    "devices stamp together, not whether the calibration is "
                    "right. Run measure_shot_offsets."
                )
            if verdict.shots_out:
                logger.warning("shot sync: %s", verdict.detail)
            if not verdict.comparable:
                # Not a failure: "could not check" must not stop a queue.
                logger.warning("shot sync could not be checked: %s", verdict.detail)
            elif verdict.synced:
                logger.info("shot sync OK: %s", verdict.detail)
        if not verdict.comparable or verdict.synced:
            return verdict
        raise GeecsConfigurationError(f"shot sync FAILED: {verdict.detail}")

    return resolve_annotations(check_shot_sync, _PLAN_ANNOTATIONS)


__all__ = [
    "DEFAULT_CALIBRATION_SHOTS",
    "DEFAULT_SYNC_TOLERANCE_S",
    "DEFAULT_TRIGGER_PERIOD_S",
    "DEVICE_TIMEOUT_S",
    "MAX_PLAUSIBLE_OFFSET_S",
    "MAX_PLAUSIBLE_SCATTER_S",
    "MIN_USEFUL_SHOTS",
    "OffsetMeasurement",
    "QUIET_CONFIRM_FLOOR_S",
    "QUIET_CONFIRM_PERIODS",
    "QUIET_MARGIN_S",
    "SyncVerdict",
    "check_shot_sync_plan",
    "measure_shot_offsets_plan",
    "offsets_from_shots",
    "sync_verdict_from_stamps",
]
