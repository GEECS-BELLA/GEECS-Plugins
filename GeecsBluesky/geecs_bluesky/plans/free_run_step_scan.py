"""geecs_free_run_step_scan — free-run time-sync step scan.

The external trigger free-runs at the machine rep rate; one **reference
device** (pacemaker) gates event creation, and every other shot-synchronized
device contributes its latest timestamped data without ever blocking a row.

How a row is built
------------------
Only the reference is Triggerable, so :func:`bluesky.plan_stubs.trigger_and_read`
waits exclusively on its ``acq_timestamp`` advance ("a physical shot
happened — emit a row").  Contributors
(:class:`~geecs_bluesky.devices.timestamped_readable.GeecsTimestampedReadable`)
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
3. **Tail flush** (after the final disarm): one extra read of all devices
   emitted to a separate ``flush`` event stream, so a contributor lagging at
   ``shot_offset = -1`` still gets its final shot recorded.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

from geecs_bluesky.devices.scan_context import ScanContext
from geecs_bluesky.devices.shot_id import ShotIdSupport
from geecs_bluesky.plans.t0_sync import geecs_t0_sync


def geecs_free_run_step_scan(
    motor: Any,
    positions: Iterable[float],
    reference: Any,
    detectors: list[Any],
    shots_per_step: int = 5,
    arm_trigger: Callable | None = None,
    disarm_trigger: Callable | None = None,
    t0_sync_window_s: float = 0.2,
    tail_flush: bool = True,
    md: dict[str, Any] | None = None,
):
    """Free-run step scan: rows paced by *reference*, contributors never block.

    Parameters
    ----------
    motor:
        A Movable device, as in :func:`~geecs_bluesky.plans.step_scan.geecs_step_scan`.
    positions:
        Iterable of motor positions to visit.
    reference:
        The pacemaker — a Triggerable sync device
        (:class:`~geecs_bluesky.devices.generic_detector.GeecsGenericDetector`)
        with shot IDs configured.  Its awaited ``acq_timestamp`` advance
        defines each event row.
    detectors:
        Non-triggerable contributors and snapshots
        (:class:`~geecs_bluesky.devices.timestamped_readable.GeecsTimestampedReadable`,
        :class:`~geecs_bluesky.devices.snapshot.GeecsSnapshotReadable`).
        Contributors are auto-anchored to *reference* (existing grace-wait
        settings are preserved).
    shots_per_step:
        Rows to emit at each motor position.
    arm_trigger / disarm_trigger:
        Optional plan-stub callables bracketing each step's acquisition
        window (e.g. DG645 SCAN / STANDBY), as in the strict plan.
    t0_sync_window_s:
        Acceptance window for the t0-sync stage.
    tail_flush:
        Emit one final ``flush``-stream event after the last disarm so
        lagging contributors' final shot is captured.
    md:
        Extra metadata merged into the start document.

    Raises
    ------
    ValueError
        If *reference* has no shot-ID tracker configured — contributors
        cannot compute row validity without it.
    GeecsT0SyncError
        If the t0-sync stage cannot establish a common physical shot.
    """
    if not isinstance(reference, ShotIdSupport) or reference.shot_id_tracker is None:
        raise ValueError(
            "free-run scan requires a reference with configure_shot_id() done; "
            f"got {getattr(reference, 'name', reference)!r}"
        )

    _positions = list(positions)
    scan_context = ScanContext()
    _read_devices = [reference, *detectors, motor, scan_context]
    sync_devices = [
        d
        for d in (reference, *detectors)
        if isinstance(d, ShotIdSupport) and d.shot_id_tracker is not None
    ]
    for det in detectors:
        if hasattr(det, "set_reference"):
            det.set_reference(reference)

    _md: dict[str, Any] = {
        "plan_name": "geecs_free_run_step_scan",
        "acquisition_mode": "free_run_time_sync",
        "geecs_event_schema": 1,
        "reference_device": getattr(
            reference, "_geecs_device_name", getattr(reference, "name", "")
        ),
        "t0_sync_window_s": t0_sync_window_s,
        "motor": getattr(motor, "name", str(motor)),
        "detectors": [getattr(d, "name", str(d)) for d in (reference, *detectors)],
        "positions": _positions,
        "shots_per_step": shots_per_step,
        "num_points": len(_positions),
        **(md or {}),
    }

    # t0 sync runs before the run opens so the captured t0s can land in the
    # start document.  The shot control must still be disarmed here.
    t0s = yield from geecs_t0_sync(sync_devices, window_s=t0_sync_window_s)
    _md["device_t0s"] = t0s

    @bpp.run_decorator(md=_md)
    def _inner():
        scan_event_index = 0
        for bin_number, pos in enumerate(_positions, start=1):
            yield from bps.mv(motor, pos)
            if arm_trigger is not None:
                yield from arm_trigger()
            for shot_index_in_bin in range(1, shots_per_step + 1):
                scan_event_index += 1
                scan_context.set_context(
                    bin_number=bin_number,
                    shot_index_in_bin=shot_index_in_bin,
                    scan_event_index=scan_event_index,
                )
                yield from bps.trigger_and_read(_read_devices)
            if disarm_trigger is not None:
                yield from disarm_trigger()
        if tail_flush:
            # No trigger: the reference would wait for a shot that may never
            # come once the trigger window is closed.
            yield from bps.create(name="flush")
            for dev in _read_devices:
                yield from bps.read(dev)
            yield from bps.save()

    yield from _inner()
