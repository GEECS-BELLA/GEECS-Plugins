"""geecs_free_run_step_scan — free-run time-sync step scan.

The external trigger free-runs at the machine rep rate; one **reference
device** (pacemaker) gates event creation, and every other shot-synchronized
device contributes its latest timestamped data without ever blocking a row.

How a row is built
------------------
Only the reference is Triggerable, so :func:`bluesky.plan_stubs.trigger_and_read`
waits exclusively on its ``acq_timestamp`` advance ("a physical shot
happened — emit a row").  Contributors
(:class:`~geecs_bluesky.devices.ca.timestamped_readable.CaTimestampedReadable`)
are then read without triggering: each labels its latest cached data with
``shot_id`` / ``shot_offset`` / ``valid`` relative to the reference's accepted
shot.  Snapshot devices are sampled as-is.  Missing or slow devices never
block; their cells are truthfully labeled for downstream realignment.

Run stages
----------
1. **t0 sync** (before the run opens, shot control still disarmed):
   :func:`~geecs_bluesky.plans.t0_sync.geecs_t0_sync` seeds every
   shot-ID-configured device from one common physical trigger; the captured
   t0s land in the start document as ``device_t0s``.
2. **Step loop** — identical bracketing to
   :func:`~geecs_bluesky.plans.step_scan.geecs_step_scan`:
   move → arm → ``shots_per_step`` rows → disarm.
3. **End-of-scan quiesce + tail flush**: after the last step's disarm the
   trigger is stopped again (quiesce → OFF; STANDBY keeps passing external
   edges, which saved orphan frames during the tail window — Gate-2).  Then
   one extra read of all devices goes to a separate ``flush`` event stream,
   so a contributor lagging at ``shot_offset = -1`` still gets its final
   shot recorded (the flush reads cached last values, so flushing while OFF
   is safe).  The caller's outer finalize restores STANDBY afterwards.

Between-step STANDBY frames in multi-step scans are an accepted window —
do not "fix" into per-step save toggling.  Design rationale:
``GeecsBluesky/CLAUDE.md`` (shot-control composition per mode).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, Sequence

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.protocols import Triggerable

from geecs_bluesky.devices.scan_context import ScanContext
from geecs_bluesky.devices.shot_id import ShotIdSupport
from geecs_bluesky.plans.step_scan import (
    motor_md,
    move_changed_axes,
    normalize_motors,
)
from geecs_bluesky.plans.t0_sync import geecs_t0_sync

logger = logging.getLogger(__name__)


def _t0_seed_check(contributors: list) -> None:
    """Warn when the scan's first row shows a contributor off its t0 seed.

    The t0-sync stage seeds every sync device from what should be one common
    physical shot; a seeding error (window wider than the caches' spread —
    possible near the ~50 ms clock-skew floor at fast rep rates) shows up as
    a **constant nonzero ``shot_offset`` from the very first row**.  A
    transiently lagging contributor also lands nonzero here, so this is a
    loud hint, not an abort: offsets stay truthfully labeled and rows remain
    realignable downstream either way (the event-schema contract).
    """
    suspects = [
        getattr(dev, "name", str(dev))
        for dev in contributors
        if getattr(dev, "last_shot_offset", None) not in (None, 0)
    ]
    if suspects:
        logger.warning(
            "t0 seed check: first-row shot_offset != 0 for %s — either the "
            "t0 seeding is off by a trigger period (check t0_sync_window_s "
            "vs the rep rate) or the device lagged the first shot; "
            "shot_offset labels remain truthful and realignable",
            suspects,
        )


def geecs_free_run_step_scan(
    motor: Any | Sequence[Any] | None,
    positions: Iterable[Any],
    reference: Any,
    detectors: list[Any],
    shots_per_step: int = 5,
    arm_trigger: Callable | None = None,
    disarm_trigger: Callable | None = None,
    quiesce_trigger: Callable | None = None,
    per_step: Callable | None = None,
    enable_saving: Callable | None = None,
    t0_sync_window_s: float = 0.2,
    tail_flush: bool = True,
    md: dict[str, Any] | None = None,
):
    """Free-run step scan: rows paced by *reference*, contributors never block.

    Parameters
    ----------
    motor:
        Any Movable/settable device — a stage axis, power supply, pressure
        controller, etc. (anything with ``set() → status``, e.g. built on
        :class:`~geecs_bluesky.devices.ca.settable.CaSettable`).  A
        **sequence** of Movables is a multi-axis grid scan (one motor per
        axis, outermost first; each position is then a tuple aligned with
        the motors, and only the axes whose target changed are re-moved).
        ``None`` means no scan variable is moved — statistics collection;
        pass ``positions=[None]`` for a single no-move bin.  The name
        follows the bluesky ``scan(detectors, motor, ...)`` convention.
    positions:
        Iterable of motor positions to visit — floats for a single motor,
        tuples for a grid.
    reference:
        The pacemaker — a Triggerable sync device
        (:class:`~geecs_bluesky.devices.ca.generic_detector.CaGenericDetector`)
        with shot IDs configured.  Its awaited ``acq_timestamp`` advance
        defines each event row.
    detectors:
        Non-triggerable contributors and snapshots
        (:class:`~geecs_bluesky.devices.ca.timestamped_readable.CaTimestampedReadable`,
        :class:`~geecs_bluesky.devices.ca.snapshot.CaSnapshotReadable`).
        Contributors are auto-anchored to *reference* (existing grace-wait
        settings are preserved).
    shots_per_step:
        Rows to emit at each motor position.
    arm_trigger / disarm_trigger:
        Optional plan-stub callables bracketing each step's acquisition
        window (e.g. DG645 SCAN / STANDBY), as in the strict plan.
    quiesce_trigger:
        Optional plan-stub callable that *stops* the free-running trigger
        (DG645 ``OFF``) before t0 sync, so every device's cache settles to
        the same last physical shot.  ``SCAN``/``STANDBY`` keep the trigger
        free-running, so they cannot serve this role.  Falls back to
        ``disarm_trigger`` when not given.  Also run at **end of scan** and
        — via an internal finalize — on abort, so native saving is never
        left enabled while the trigger passes external edges.
    per_step:
        Optional plan-stub callable run at **every** step boundary — after
        the move completes, before that step's ``arm_trigger``/shots.  A
        ScanRequest's ``actions.per_step`` plans land here; the arm/disarm
        bracketing means they always run *disarmed*.
    enable_saving:
        Optional plan-stub callable that turns native file saving on
        (typically :func:`~geecs_bluesky.plans.run_wrapper.save_enable_plan`).
        Run once, immediately after the quiesce and before t0 sync — the
        earliest orphan-free moment (Gate-2 save windowing:
        ``GeecsBluesky/CLAUDE.md``).  Without a quiesce hook it runs at the
        same point, unwindowed.  Save-*off* stays the run wrapper's
        innermost finalize, before the caller's disarm.
    t0_sync_window_s:
        Acceptance window for the t0-sync stage.  Capped at
        ``0.4 / rep_rate_hz`` (from the reference's shot-ID tracker) so the
        window can never span more than one trigger period — a window wider
        than a period silently accepts caches seeded one shot apart.  The
        effective value is recorded in the start document.  Note the design
        floor: the window must stay above inter-machine clock skew
        (~50 ms), which at 5 Hz leaves 50–80 ms of real margin; rates much
        beyond 5 Hz need a redesigned seeding stage, not a smaller window.
    tail_flush:
        Emit one final ``flush``-stream event after the last disarm so
        lagging contributors' final shot is captured.
    md:
        Extra metadata merged into the start document.

    Raises
    ------
    TypeError
        If *reference* is not a bluesky ``Triggerable``.
        ``bps.trigger_and_read`` only waits on devices that implement
        ``trigger()`` — a non-Triggerable reference would never pace the
        rows, so the plan would record unpaced duplicates of whatever frame
        each device last cached (silent data corruption).
    ValueError
        If *reference* has no shot-ID tracker configured — contributors
        cannot compute row validity without it.
    GeecsT0SyncError
        If the t0-sync stage cannot establish a common physical shot.
    """
    if not isinstance(reference, Triggerable):
        raise TypeError(
            "free-run scan requires a Triggerable reference (pacemaker) — "
            "without a real trigger(), trigger_and_read would not wait for "
            "shots and every row would duplicate the last cached frame; got "
            f"{type(reference).__name__} "
            f"{getattr(reference, 'name', reference)!r}"
        )
    if not isinstance(reference, ShotIdSupport) or reference.shot_id_tracker is None:
        raise ValueError(
            "free-run scan requires a reference with configure_shot_id() done; "
            f"got {getattr(reference, 'name', reference)!r}"
        )

    _positions = list(positions)
    _motors = normalize_motors(motor)
    rep_rate_hz = reference.shot_id_tracker.rep_rate_hz
    effective_window_s = min(t0_sync_window_s, 0.4 / rep_rate_hz)
    if effective_window_s < t0_sync_window_s:
        logger.info(
            "t0-sync window capped to %.3fs (%.3fs requested) for rep rate "
            "%.1f Hz — the window must stay under one trigger period",
            effective_window_s,
            t0_sync_window_s,
            rep_rate_hz,
        )
    scan_context = ScanContext()
    _read_devices = [reference, *detectors]
    _read_devices.extend(_motors)
    _read_devices.append(scan_context)
    sync_devices = [
        d
        for d in (reference, *detectors)
        if isinstance(d, ShotIdSupport) and d.shot_id_tracker is not None
    ]
    for det in detectors:
        if hasattr(det, "set_reference"):
            det.set_reference(reference)
    # Soft shot-context candidates: telemetry members (inside a group) with
    # a configured tracker.  They are anchored to the reference like
    # contributors, and SEEDED best-effort below — only devices whose
    # acq_timestamp is positive at the quiesced t0 snapshot (i.e. observed
    # to have actually fired) get companion columns; the rest stay
    # value-columns-only.  Soft: never gates, never fails the scan.
    soft_shot_members = [
        m
        for d in detectors
        for m in getattr(d, "members", [])
        if isinstance(m, ShotIdSupport) and m.shot_id_tracker is not None
    ]
    for member in soft_shot_members:
        if hasattr(member, "set_row_reference"):
            member.set_row_reference(reference)

    _md: dict[str, Any] = {
        "plan_name": "geecs_free_run_step_scan",
        "acquisition_mode": "free_run_time_sync",
        "geecs_event_schema": 1,
        "reference_device": getattr(
            reference, "_geecs_device_name", getattr(reference, "name", "")
        ),
        "t0_sync_window_s": effective_window_s,
        "motor": motor_md(_motors),
        "detectors": [getattr(d, "name", str(d)) for d in (reference, *detectors)],
        "positions": _positions,
        "shots_per_step": shots_per_step,
        "num_points": len(_positions),
        **(md or {}),
    }

    # t0 sync runs before the run opens so the captured t0s can land in the
    # start document.  Quiesce first (stop the trigger) so every device's
    # cache settles to the same last physical shot; no-op without shot control.
    _quiesce = quiesce_trigger or disarm_trigger
    if _quiesce is not None:
        yield from _quiesce()
    if enable_saving is not None:
        # Trigger stopped through t0 sync until the first arm[SCAN] — the
        # earliest orphan-free moment to start native saving (Gate-2).
        yield from enable_saving()
    t0s = yield from geecs_t0_sync(sync_devices, window_s=effective_window_s)
    _md["device_t0s"] = t0s
    if soft_shot_members:
        # Soft seeding from the same quiesced snapshot: a positive cached
        # acq_timestamp means the device has genuinely fired; 0.0 is the
        # gateway's never-acquired placeholder (PV_CONTRACT.md §3) — those
        # devices stay unseeded and emit no companion columns this scan.
        seeded: list[str] = []
        for member in soft_shot_members:
            ts = yield from bps.rd(member.acq_timestamp)
            if ts is not None and float(ts) > 0:
                member.seed_shot_id(float(ts))
                seeded.append(getattr(member, "name", str(member)))
        logger.info(
            "telemetry shot context: %d of %d devices seeded (fired before "
            "t0); the rest record value columns only",
            len(seeded),
            len(soft_shot_members),
        )
        _md["telemetry_shot_seeded"] = seeded

    @bpp.run_decorator(md=_md)
    def _inner():
        scan_event_index = 0
        previous: tuple | None = None
        for bin_number, pos in enumerate(_positions, start=1):
            if _motors and pos is not None:
                previous = yield from move_changed_axes(_motors, pos, previous)
            if per_step is not None:
                # After the move, before arming: per-step actions run with
                # the shot controller disarmed (outside the SCAN window).
                yield from per_step()
            if arm_trigger is not None:
                yield from arm_trigger()
            for shot_index_in_bin in range(1, shots_per_step + 1):
                scan_event_index += 1
                scan_context.set_context(
                    bin_number=bin_number,
                    shot_index_in_bin=shot_index_in_bin,
                    scan_event_index=scan_event_index,
                )
                row_t0 = time.monotonic()
                yield from bps.trigger_and_read(_read_devices)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "row %d: %.1f ms (trigger wait + %d device reads)",
                        scan_event_index,
                        (time.monotonic() - row_t0) * 1e3,
                        len(_read_devices),
                    )
                if scan_event_index == 1:
                    _t0_seed_check(detectors)
            if disarm_trigger is not None:
                yield from disarm_trigger()
        # End of scan: stop the trigger BEFORE the tail machinery — STANDBY
        # passes external edges, so orphan frames get saved otherwise (Gate-2).
        if _quiesce is not None:
            yield from _quiesce()
            _end_quiesced["done"] = True
        if tail_flush:
            # No trigger: the reference would wait for a shot that may never
            # come once the trigger window is closed.
            yield from bps.create(name="flush")
            for dev in _read_devices:
                yield from bps.read(dev)
            yield from bps.save()

    _end_quiesced = {"done": False}

    def _quiesce_before_cleanup():
        # Abort parity: guarantee the trigger is stopped before the caller's
        # finalize save-off runs, making completion and abort uniform
        # (quiesce[OFF] → save-off → disarm[STANDBY] → closeout).  Skipped
        # when the end-of-scan quiesce above already ran, so a completed
        # scan does not write OFF twice.
        if _quiesce is not None and not _end_quiesced["done"]:
            yield from _quiesce()

    yield from bpp.finalize_wrapper(_inner(), _quiesce_before_cleanup)
