"""The shot-offset calibration and its preflight — the two once-run plans.

Two devices stamp the *same* shot at different times: ``acq_timestamp`` is
the trigger's arrival plus that device's own frame-drain latency, a
per-device constant of tens of milliseconds
(``Planning/native_bluesky/03_clean_room_rebuild.md`` §11.3/§11.4).  The
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
(§11.2, measured in M1).  So nothing announces quiescence: the only way to
know the set is quiet is to watch the stamps not move for longer than the
longest device timeout in it.  That is the floor on both plans' cost, and
it is why neither may sit inside a scan.

:func:`measure_shot_offsets_plan`
    Drive the box OFF, wait the set quiet, then fire single shots and read
    every device's stamp.  The spread across devices *is* the calibration.

:func:`check_shot_sync_plan`
    Sam's validation shortcut (§11.7), and it costs no shot at all: with
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
of 36–100 ms.  The expensive part of this plan is the quiet wait, paid
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import bluesky.plan_stubs as bps
from bluesky.utils import separate_devices
from geecs_schemas.shot_offsets import DeviceOffset, ShotOffsets
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.devices.detector import GeecsDetector
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plans.gated import run_bracket
from geecs_bluesky.plans.strict import fire_and_await_shot

logger = logging.getLogger(__name__)

#: The timeout a GEECS device falls back on when no acquisition arrives —
#: 1.5 s for ~95 % of them (§11.2).  Not discoverable: the experiment DB
#: carries no timeout column, so this is the documented constant, and a set
#: containing a slower device wants an explicit ``quiet_time``.
DEVICE_TIMEOUT_S = 1.5

#: Added to :data:`DEVICE_TIMEOUT_S` for the drain of a frame already in
#: flight when the box went OFF (§11.4: the TCP message is sent when the
#: exposure completes, so a long exposure lands most of a second late).
QUIET_MARGIN_S = 0.75

#: After the quiet wait, the window over which the stamps must not move for
#: the set to be called quiet.  Short on purpose: the wait above has already
#: covered the timeout, so this only has to catch a box that never went OFF.
QUIET_CONFIRM_S = 0.5

#: Shots averaged by default — ~3 ms of residual noise against ~10 ms of
#: per-shot dither, for ten seconds of beam time.
DEFAULT_CALIBRATION_SHOTS = 10

#: Extra shots allowed to replace incomplete ones before the plan gives up,
#: as a multiple of the requested count.
_RETAKE_BUDGET = 2

#: Default tolerance for :func:`check_shot_sync_plan`.  §11.7 quotes ~200 ms
#: for the by-eye version; 50 ms is well clear of the ~10 ms dither and
#: still catches a device a whole shot out of step.
DEFAULT_SYNC_TOLERANCE_S = 0.05


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
        Every measurable device's corrected stamp fell inside *tolerance_s*.
    spread_s :
        Widest gap between two corrected stamps.
    tolerance_s :
        The budget it was judged against.
    corrected :
        Object name → its stalled stamp minus its stored offset.
    unmeasured :
        Devices holding no usable stamp (never acquired since boot).
    detail :
        One line saying what was found — the log line and the error message.
    """

    synced: bool
    spread_s: float
    tolerance_s: float
    corrected: Mapping[str, float]
    unmeasured: tuple[str, ...]
    detail: str


def sync_verdict_from_stamps(
    stamps: Mapping[str, float | None],
    offsets: Mapping[str, float],
    *,
    tolerance_s: float = DEFAULT_SYNC_TOLERANCE_S,
) -> SyncVerdict:
    """Judge stalled stamps against the stored offsets.

    With the box OFF and the set quiet, every device holds the stamp of the
    same last real shot.  Correcting each by its stored offset should
    therefore collapse them onto one instant, to within NTP jitter and the
    hosts' own dither (§11.3).  A device that lands outside is either
    mis-calibrated or was not receiving that shot at all — both of which
    cost rows once the join windows tighten, and both worth stopping for.

    A stamp of ``None`` or a non-positive value is *unmeasured*, not a
    failure: the gateway publishes ``0.0`` before a device's first
    acquisition, and a device that has not acquired since boot says nothing
    about the calibration.  It is reported so the operator can see the
    check did not cover it.

    Parameters
    ----------
    stamps :
        Object name → its stalled ``acq_timestamp`` (``None`` if unread).
    offsets :
        Object name → its stored drain offset, seconds.
    tolerance_s :
        Budget for the widest corrected gap.

    Returns
    -------
    SyncVerdict

    Raises
    ------
    ValueError
        *tolerance_s* is not positive.
    """
    if tolerance_s <= 0:
        raise ValueError(f"tolerance_s must be positive seconds, got {tolerance_s}")
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
            spread_s=0.0,
            tolerance_s=tolerance_s,
            corrected=corrected,
            unmeasured=tuple(sorted(unmeasured)),
            detail=(
                f"only {len(corrected)} device(s) hold a usable stamp, so the "
                f"set cannot be compared (no stamp yet: {missing}) — fire a "
                "few shots by hand and check again"
            ),
        )
    low = min(corrected.values())
    high = max(corrected.values())
    spread = high - low
    earliest = min(sorted(corrected), key=lambda n: corrected[n])
    latest = max(sorted(corrected), key=lambda n: corrected[n])
    synced = spread <= tolerance_s
    detail = (
        f"corrected stamps span {spread * 1e3:.1f} ms "
        f"({earliest} earliest, {latest} latest) against a "
        f"{tolerance_s * 1e3:.0f} ms tolerance"
    )
    if not synced:
        detail += (
            " — the stored offsets no longer describe this set; re-run "
            "measure_shot_offsets, or check whether a device stopped "
            "receiving the trigger"
        )
    if unmeasured:
        detail += f" (not covered, no stamp yet: {', '.join(sorted(unmeasured))})"
    return SyncVerdict(
        synced=synced,
        spread_s=spread,
        tolerance_s=tolerance_s,
        corrected=corrected,
        unmeasured=tuple(sorted(unmeasured)),
        detail=detail,
    )


# ------------------------------------------------------------ plan machinery


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


def _read_stamps(views: Sequence[Any]):
    """Plan: object name → each device's current stamp, by explicit CA get.

    A **get**, not the acquire logic's monitored value: in OFF the stamp PV
    publishes nothing at all (the devices' timeout events carry an unchanged
    value and the gateway's change suppression drops them, §11.2), so a
    monitor that attached during the quiet window would still read ``None``
    while the PV itself holds the last shot's stamp perfectly well.
    """
    stamps: dict[str, float | None] = {}
    for view in views:
        owner = view._owner
        try:
            value = yield from bps.rd(owner.acq_timestamp)
        except Exception:
            logger.warning(
                "%s: acq_timestamp could not be read", owner.name, exc_info=True
            )
            value = None
        stamps[owner.name] = None if value is None else float(value)
    return stamps


def _settle_quiet(views: Sequence[Any], quiet_time: float):
    """Plan: wait the set quiet with the box already OFF; return the stalled stamps.

    Costs at least the longest device timeout, by construction — there is no
    event to wait for, because the timeout events never reach the gateway
    (§11.2).  So: wait it out, then prove the set is still by re-reading
    over a short confirmation window.  A stamp that advanced in that window
    means edges are still arriving, which makes every number this plan would
    produce meaningless.

    Returns
    -------
    dict
        Object name → stalled stamp (``None`` where unread).
    """
    before = yield from _read_stamps(views)
    logger.info(
        "waiting %.1f s for the device set to go quiet (the longest device "
        "timeout — nothing announces quiescence, §11.2)",
        quiet_time,
    )
    yield from bps.sleep(quiet_time)
    settled = yield from _read_stamps(views)
    yield from bps.sleep(QUIET_CONFIRM_S)
    confirm = yield from _read_stamps(views)
    moving = [
        name
        for name, value in confirm.items()
        if value is not None
        and settled.get(name) is not None
        and value != settled[name]
    ]
    if moving:
        raise GeecsConfigurationError(
            f"{', '.join(sorted(moving))}: still acquiring {QUIET_CONFIRM_S:.1f} s "
            f"after {quiet_time:.1f} s in OFF — the trigger box is not actually "
            "off for these devices (check the trigger profile's OFF writes, or "
            "whether another source is triggering them). A calibration measured "
            "against a running box is meaningless."
        )
    drained = [
        name
        for name, value in settled.items()
        if value is not None and before.get(name) is not None and value != before[name]
    ]
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
    """Build the ``measure_shot_offsets`` queue plan (§4.F).

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
        write: bool = False,
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
        scan.  The box is left in STANDBY however the plan ends.

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
        write : bool, optional
            Store the result in the configs repo (default ``False``).
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
        if write and resolver is None:
            raise GeecsConfigurationError(
                "measure_shot_offsets: this worker has no configs resolver "
                "(QS_DEVICE_NAMESPACE=off), so the measurement cannot be "
                "written — run with write=False to measure only"
            )
        views = _stamp_views(detectors)
        shot_control = profiles.resolve(trigger_profile)
        profile_key = (
            trigger_profile if trigger_profile is not None else profiles.default
        )

        def inner():
            # Touch every device once: the message that connects it (the
            # worker's connect_on_demand preprocessor), so a device missing
            # from the gateway fails here — before the box is touched —
            # rather than part-way through the measurement.
            for view in views:
                yield from bps.read(view)
            yield from _settle_quiet(views, quiet_time)
            # ARMED is the strict source: the box emits one edge per
            # SINGLESHOT and nothing in between, so the set stays quiet
            # between shots and every stamp advance is a shot this plan
            # caused.
            yield from bps.mv(shot_control, TriggerState.ARMED.value)

            def fire():
                yield from bps.mv(shot_control, TriggerState.SINGLESHOT.value)

            complete: list[dict[str, float]] = []
            attempts = 0
            budget = shots * _RETAKE_BUDGET
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
                "measured only — re-run with write=True to store this in the "
                "experiment's shot_offsets.yaml"
            )
            return measurement
        document = measurement.to_document(
            geecs_names={
                view._owner.name: view._owner._geecs_device_name for view in views
            },
            trigger_profile=profile_key,
            description=description,
        )
        path = resolver.write_shot_offsets(document)
        logger.info(
            "shot offsets stored in %s — this is an UNCOMMITTED change in the "
            "configs repo: review and commit it. The worker picks the new "
            "offsets up at its next environment open, not now.",
            path,
        )
        return measurement

    return measure_shot_offsets


def check_shot_sync_plan(profiles: Any) -> Callable[..., Any]:
    """Build the ``check_shot_sync`` queue plan — the preflight of §4.F/§11.7.

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
    ):
        """Check the stored drain offsets still describe this device set.

        Sam's shortcut, and it costs no shot: with the box OFF and the set
        quiet, every device still holds the stamp of the same last real
        shot, so correcting those stalled stamps by the stored offsets
        should collapse them onto one instant.  A device landing outside the
        tolerance is either mis-calibrated or was not receiving that shot.

        This is a **queue item**, deliberately — never a step inside a scan.
        It costs at least the longest device timeout every time it runs
        (§11.2), and running it from the queue also means it cannot drive
        the trigger box while a scan is using it.

        Raises when the set is out of tolerance, so a queue that puts this
        ahead of its scans stops before taking data against a stale
        calibration.

        No run is opened; the box is left in STANDBY.

        Parameters
        ----------
        detectors : sequence of devices
            The triggered GEECS devices to check (at least two).
        trigger_profile : str, optional
            Profile driving the box (the experiment default when omitted).
        tolerance_s : float, optional
            Widest corrected gap accepted, seconds (default 0.05 — clear of
            the ~10 ms host dither, tight enough to catch a device a whole
            shot out of step).
        quiet_time : float, optional
            Seconds to wait in OFF before believing the set is quiet.

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
        views = _stamp_views(detectors)
        shot_control = profiles.resolve(trigger_profile)

        def inner():
            for view in views:
                yield from bps.read(view)
            stalled = yield from _settle_quiet(views, quiet_time)
            offsets: dict[str, float] = {}
            for view in views:
                owner = view._owner
                value = yield from bps.rd(owner.drain_offset)
                offsets[owner.name] = float(value or 0.0)
            return stalled, offsets

        stalled, offsets = yield from run_bracket(
            inner(), shot_control, TriggerState.OFF
        )
        verdict = sync_verdict_from_stamps(stalled, offsets, tolerance_s=tolerance_s)
        if not any(offsets.values()):
            logger.warning(
                "every stored drain offset reads 0.0 — this set has never been "
                "calibrated, so the check below only says whether the devices "
                "stamp together, not whether the calibration is right. Run "
                "measure_shot_offsets."
            )
        if verdict.synced:
            logger.info("shot sync OK: %s", verdict.detail)
            return verdict
        raise GeecsConfigurationError(f"shot sync FAILED: {verdict.detail}")

    return check_shot_sync


__all__ = [
    "DEFAULT_CALIBRATION_SHOTS",
    "DEFAULT_SYNC_TOLERANCE_S",
    "DEVICE_TIMEOUT_S",
    "OffsetMeasurement",
    "QUIET_CONFIRM_S",
    "QUIET_MARGIN_S",
    "SyncVerdict",
    "check_shot_sync_plan",
    "measure_shot_offsets_plan",
    "offsets_from_shots",
    "sync_verdict_from_stamps",
]
